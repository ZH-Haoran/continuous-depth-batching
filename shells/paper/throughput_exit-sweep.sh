#!/usr/bin/env bash
# Offline throughput against the exit threshold, at a fixed maximum decode batch size.
#
# Exit thresholds are chosen to retain good GSM8k accuracy.
# Note that lower thresholds lead to fewer exits for Huginn and more for Ouro, so they are inverted.
#
# Submit with: MODEL=ouro DATASET=sharegpt ./shells/_submit.sh shells/paper/throughput_exit-sweep.sh

set -euo pipefail
source "${REPO_ROOT:?submit with ./shells/_submit.sh}/shells/paper/_common.sh"

MODEL="${MODEL:?set MODEL to ouro or huginn}"
DATASET="${DATASET:?set DATASET to sharegpt, alpaca, or arxiv}"
if [ "$DATASET" = arxiv ]; then
    DECODE_WIDTH=8
else
    DECODE_WIDTH=64
fi

case "$MODEL" in
    ouro)
        MODEL_ID="$OURO_1_4B_MODEL_ID"
        RECUR_STEPS=4
        MIN_CODA_BATCH=""
        THRESHOLDS="${THRESHOLDS:-0.2 0.4 0.5 0.7 1.0}"
        CONTROL=1.0
        ;;
    huginn)
        MODEL_ID="$HUGINN_MODEL_ID"
        RECUR_STEPS=16
        MIN_CODA_BATCH=$((DECODE_WIDTH / 4))
        THRESHOLDS="${THRESHOLDS:-0.28 0.16 0.1 0.0}"
        CONTROL=0.0
        ;;
    *) echo "unknown MODEL '$MODEL' (expected: ouro, huginn)" >&2; exit 1 ;;
esac

WORKLOAD="$(workload_for "$MODEL" "$DATASET")"
OUTPUT_PATH="outputs/exit-sweep/exit-sweep_${MODEL}_${DATASET}.jsonl"
NUM_REQUESTS="$(num_requests_for "$DATASET")"

BACKENDS="${BACKENDS:-cb cdb-norefill cdb-refill}"

mkdir -p "$(dirname "$OUTPUT_PATH")"
echo "Experiment: drain throughput vs exit threshold, $MODEL_ID, $DATASET"
echo "Thresholds: $THRESHOLDS (no-exit control at $CONTROL)"
echo "Max decode batch size: $DECODE_WIDTH"
echo "Output:     $OUTPUT_PATH"
echo ""

for backend in $BACKENDS; do
    # cb decodes at full depth and ignores the threshold, so measure it once.
    if [ "$backend" = cb ]; then thresholds="${THRESHOLDS%% *}"; else thresholds="$THRESHOLDS"; fi
    for threshold in $thresholds; do
        echo "=== backend: $backend  threshold: $threshold ==="
        mapfile -t args < <(engine_flags "$backend")
        args+=(
            --model "$MODEL_ID"
            --workload "$WORKLOAD"
            --exit-threshold "$threshold"
            --max-recurrent-depth "$RECUR_STEPS"
            --max-model-len "$MAX_MODEL_LEN"
            --max-num-seqs "$DECODE_WIDTH"
            --attn-implementation "$ATTN_IMPL"
            --num-blocks "$NUM_BLOCKS"
            --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
            --block-size "$BLOCK_SIZE"
            --num-requests "$NUM_REQUESTS"
            --summary-output "$OUTPUT_PATH"
        )
        [[ "$backend" == cdb-refill && -n "$MIN_CODA_BATCH" ]] && args+=(--min-coda-batch-size "$MIN_CODA_BATCH")
        uv run python scripts/benchmark_throughput.py "${args[@]}"
        echo ""
    done
done
