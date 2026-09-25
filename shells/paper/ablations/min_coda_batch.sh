#!/usr/bin/env bash
# Ablates the minimum coda batch size for offline throughput (ShareGPT); only relevant for CDB with refill.
#
# Sweeps the minimum coda batch size per maximum decode batch size.
# MIN_CODA_BATCHES accepts absolute sizes and the fractions 'eighth', 'quarter', and 'half'.
# Fractional sizes are clamped to one.
#
# Submit with:
#   MODEL=ouro   ./shells/_submit.sh shells/paper/ablations/min_coda_batch.sh
#   MODEL=huginn ./shells/_submit.sh shells/paper/ablations/min_coda_batch.sh

set -euo pipefail
source "${REPO_ROOT:?submit with ./shells/_submit.sh}/shells/paper/_common.sh"

MODEL="${MODEL:?set MODEL to ouro or huginn}"

case "$MODEL" in
    ouro)
        MODEL_ID="$OURO_1_4B_MODEL_ID"
        RECUR_STEPS=4
        THRESHOLD=0.2 # earliest threshold from exit sweep
        MIN_CODA_BATCHES="${MIN_CODA_BATCHES:-1 eighth quarter half}"
        DECODE_WIDTHS="${DECODE_WIDTHS:-64}"
        ;;
    huginn)
        MODEL_ID="$HUGINN_MODEL_ID"
        RECUR_STEPS=16
        THRESHOLD=0.28 # earliest threshold from exit sweep
        MIN_CODA_BATCHES="${MIN_CODA_BATCHES:-1 eighth quarter half}"
        DECODE_WIDTHS="${DECODE_WIDTHS:-16 32 64 512}"
        ;;
    *) echo "unknown MODEL '$MODEL' (expected: ouro, huginn)" >&2; exit 1 ;;
esac
WORKLOAD="$(workload_for "$MODEL" sharegpt)"
OUTPUT_PATH="outputs/ablations/min-coda-batch/min-coda-batch_${MODEL}_sharegpt.jsonl"

NUM_REQUESTS="$(num_requests_for sharegpt)"

mkdir -p "$(dirname "$OUTPUT_PATH")"
echo "Experiment: minimum coda batch (offline throughput), $MODEL_ID, sharegpt, q=$THRESHOLD"
echo "Minimum coda batch sizes: $MIN_CODA_BATCHES"
echo "Max decode batch sizes:     $DECODE_WIDTHS"
echo "Output:     $OUTPUT_PATH"
echo ""

for width in $DECODE_WIDTHS; do
    for min_coda in $MIN_CODA_BATCHES; do
        case "$min_coda" in
            eighth)  min_coda=$((width / 8)) ;;
            quarter) min_coda=$((width / 4)) ;;
            half)    min_coda=$((width / 2)) ;;
        esac
        if ((min_coda < 1)); then
            min_coda=1
        fi
        echo "=== width: $width  min coda batch: $min_coda ==="
        mapfile -t args < <(engine_flags cdb-refill)
        args+=(
            --model "$MODEL_ID"
            --workload "$WORKLOAD"
            --exit-threshold "$THRESHOLD"
            --max-recurrent-depth "$RECUR_STEPS"
            --max-model-len "$MAX_MODEL_LEN"
            --max-num-seqs "$width"
            --attn-implementation "$ATTN_IMPL"
            --num-blocks "$NUM_BLOCKS"
            --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
            --block-size "$BLOCK_SIZE"
            --num-requests "$NUM_REQUESTS"
            --min-coda-batch-size "$min_coda"
            --summary-output "$OUTPUT_PATH"
        )
        uv run python scripts/benchmark_throughput.py "${args[@]}"
        echo ""
    done
done
