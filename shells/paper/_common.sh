#!/usr/bin/env bash
# Shared paper experiment configuration: canonical model IDs, repo root, machine config,
# virtualenv, and GPU pinning. Per-experiment model selection, workload, thresholds, and
# output paths live in each experiment script.

set -euo pipefail

readonly OURO_1_4B_MODEL_ID="KristianS7/Ouro-1.4B"
readonly OURO_2_6B_MODEL_ID="KristianS7/Ouro-2.6B"
readonly HUGINN_MODEL_ID="tomg-group-umd/huginn-0125"

cd "$REPO_ROOT"
echo "repo: $REPO_ROOT (submitted from ${SLURM_SUBMIT_DIR:-unset})"

CONFIG="${MACHINE_CONFIG:?exported by shells/_submit.sh}"
# shellcheck source=/dev/null
source "$CONFIG"
validate_config || exit 1

FLASH_ATTENTION="${FLASH_ATTENTION:-flash_attention_3}" # set by the machine config
case "$FLASH_ATTENTION" in
    flash_attention_3)
        uv sync --frozen
        BLOCK_SIZE="${BLOCK_SIZE:-16}"
        ;;
    flash_attention_2)
        uv sync --frozen --extra fa2
        # FA2's paged decode kernel requires a page size that is a multiple of 256.
        BLOCK_SIZE="${BLOCK_SIZE:-256}"
        ;;
    *)
        echo "ERROR: FLASH_ATTENTION must be flash_attention_3 or flash_attention_2, got '$FLASH_ATTENTION'" >&2
        exit 1
        ;;
esac
export ATTN_IMPL="paged|$FLASH_ATTENTION"
export BLOCK_SIZE
# KV cache reservation, in tokens: the engines allocate (NUM_BLOCKS + 2) * BLOCK_SIZE
CACHE_TOKENS="${CACHE_TOKENS:-327680}"
export NUM_BLOCKS=$((CACHE_TOKENS / BLOCK_SIZE))
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
# Requests each closed-loop throughput run replays (the first N of the bundle): enough to leave the
# steady-state window, which opens after 2W completions and closes at the last admission, several
# thousand completions at width 512 (about 4400 on ShareGPT, 8500 on Alpaca, whose wider length
# spread needs more). One value per dataset for every width and backend of a sweep, so every point
# replays the same requests.
export NUM_REQUESTS_SHAREGPT="${NUM_REQUESTS_SHAREGPT:-6000}"
export NUM_REQUESTS_ALPACA="${NUM_REQUESTS_ALPACA:-10000}"
export NUM_REQUESTS_ARXIV="${NUM_REQUESTS_ARXIV:-1000}"
num_requests_for() {
    case "$1" in
        sharegpt) echo "$NUM_REQUESTS_SHAREGPT" ;;
        alpaca)   echo "$NUM_REQUESTS_ALPACA" ;;
        arxiv)    echo "$NUM_REQUESTS_ARXIV" ;;
        *) echo "unknown dataset '$1' (expected: sharegpt, alpaca, arxiv)" >&2; return 1 ;;
    esac
}
# Bundle path of a (model, dataset) pair: <model>_<dataset>_recur<depth>[_<count>].json, where the
# depth is the model's full recurrent budget and the count marks a bundle cut to a fixed size.
workload_for() {
    local depth suffix
    case "$1" in
        ouro)   depth=4 ;;
        huginn) depth=16 ;;
        *) echo "unknown model '$1' (expected: ouro, huginn)" >&2; return 1 ;;
    esac
    case "$2" in
        sharegpt) suffix=_10k ;;
        alpaca)   suffix="" ;;
        arxiv)    suffix=_6k ;;
        *) echo "unknown dataset '$2' (expected: sharegpt, alpaca, arxiv)" >&2; return 1 ;;
    esac
    echo "outputs/workloads/${1}_${2}_recur${depth}${suffix}.json"
}
# shellcheck source=/dev/null
source .venv/bin/activate
echo "Attention: $ATTN_IMPL (block size $BLOCK_SIZE)"

# Thermal pinning for reproducible timing: pin compute to the fixed device named by
# PIN_H100_UUID (set with its rationale in the machine config), leaving the job's other GPU
# idle, so timing always lands on the same GPU regardless of SLURM's gres ordering. The pin
# only works if the reservation includes that GPU, so a machine config that sets it should
# also reserve both GPUs (NUM_GPUS=2).
if [[ -n "${PIN_H100_UUID:-}" ]]; then
    # Captured rather than piped into grep -q: grep's early exit would SIGPIPE nvidia-smi
    # under pipefail and fail the check even when the UUID matched.
    gpu_uuids="$(nvidia-smi --query-gpu=uuid --format=csv,noheader)"
    if grep -q "$PIN_H100_UUID" <<<"$gpu_uuids"; then
        export CUDA_VISIBLE_DEVICES="$PIN_H100_UUID"
        echo "Pinned CUDA_VISIBLE_DEVICES=$PIN_H100_UUID"
    else
        echo "ERROR: pinned GPU $PIN_H100_UUID is not in this job's reservation." >&2
        echo "Reserve both GPUs (NUM_GPUS=2) or disable the pin (PIN_H100_UUID=)." >&2
        exit 1
    fi
fi
nvidia-smi --query-gpu=index,name,uuid,temperature.gpu --format=csv,noheader || true

# Translate backend name into engine flags.
engine_flags() {
    case "$1" in
        cb) printf '%s\n' --backend cb ;;
        cdb-norefill) printf '%s\n' --backend cdb --no-refill ;;
        cdb-refill) printf '%s\n' --backend cdb ;;
        *) echo "unknown backend '$1' (expected: cb, cdb-norefill, cdb-refill)" >&2; exit 1 ;;
    esac
}
