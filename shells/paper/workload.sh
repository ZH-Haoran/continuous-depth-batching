#!/usr/bin/env bash
# Record request lengths and threshold-free per-token exit trajectories.
# Throughput, serving, and profiling scripts replay these bundles.
# Existing bundles at the output path are overwritten.
#
# All models' dataset bundles hold the same requests in the same replay order: a request is
# kept only if the other model's tokenizer also passes the length and context filters, and the
# replay shuffle is keyed on the request ids.
#
# Submit with: MODEL=ouro DATASET=sharegpt DATA_PATH=path/to/dataset.json \
#   ./shells/_submit.sh shells/paper/workload.sh

set -euo pipefail
source "${REPO_ROOT:?submit with ./shells/_submit.sh}/shells/paper/_common.sh"

MODEL="${MODEL:?set MODEL to ouro or huginn}"
DATASET="${DATASET:?set DATASET to sharegpt, alpaca, or arxiv}"
DATA_PATH="${DATA_PATH:?set DATA_PATH to the raw $DATASET file (JSON, JSONL, or Parquet)}"

case "$MODEL" in
    ouro)   MODEL_ID="$OURO_1_4B_MODEL_ID"; OTHER_MODEL_ID="$HUGINN_MODEL_ID"; RECUR_STEPS=4 ;;
    huginn) MODEL_ID="$HUGINN_MODEL_ID"; OTHER_MODEL_ID="$OURO_1_4B_MODEL_ID"; RECUR_STEPS=16 ;;
    *) echo "unknown MODEL '$MODEL' (expected: ouro, huginn)" >&2; exit 1 ;;
esac

case "$DATASET" in
    sharegpt) LIMIT=10000 ;;
    alpaca)   LIMIT="" ;;
    arxiv)    LIMIT=6000 ;;
    *) echo "unknown DATASET '$DATASET' (expected: sharegpt, alpaca, arxiv)" >&2; exit 1 ;;
esac

OUTPUT_PATH="$(workload_for "$MODEL" "$DATASET")"

echo "Recording:  $DATASET bundle for $MODEL ($RECUR_STEPS recurrent steps)"
echo "Output:     $OUTPUT_PATH"
echo ""

args=(
    --model "$MODEL_ID"
    --model-family "$MODEL"
    --dataset "$DATASET"
    --data-path "$DATA_PATH"
    --recur-steps "$RECUR_STEPS"
    --max-model-len "$MAX_MODEL_LEN"
    --attn-impl "$FLASH_ATTENTION"
    --filter-models "$OTHER_MODEL_ID"
    --shuffle-seed 0
    --output-path "$OUTPUT_PATH"
)
[[ -n "$LIMIT" ]] && args+=(--limit "$LIMIT")

uv run python scripts/create_workload_json.py "${args[@]}" "$@"
