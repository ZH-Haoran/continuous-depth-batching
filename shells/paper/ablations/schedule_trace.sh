#!/usr/bin/env bash
# Record CDB stage launches for the appendix tables.
# Traced throughput includes recording overhead.
# Submit with: MODEL=ouro DATASET=sharegpt ./shells/_submit.sh shells/paper/ablations/schedule_trace.sh

set -euo pipefail
source "${REPO_ROOT:?submit with ./shells/_submit.sh}/shells/paper/_common.sh"

MODEL="${MODEL:?set MODEL to ouro or huginn}"
DATASET="${DATASET:?set DATASET to sharegpt or alpaca}"

case "$MODEL" in
    ouro)
        MODEL_ID="$OURO_1_4B_MODEL_ID"
        RECUR_STEPS=4
        CODA_BATCHES="1 16"
        THRESHOLD=0.2
        ;;
    huginn)
        MODEL_ID="$HUGINN_MODEL_ID"
        RECUR_STEPS=16
        CODA_BATCHES="1 16"
        THRESHOLD=0.28
        ;;
    *) echo "unknown MODEL '$MODEL' (expected: ouro, huginn)" >&2; exit 1 ;;
esac

case "$DATASET" in
    sharegpt|alpaca) WORKLOAD="$(workload_for "$MODEL" "$DATASET")" ;;
    *) echo "unknown DATASET '$DATASET' (expected: sharegpt, alpaca)" >&2; exit 1 ;;
esac

WIDTH="${WIDTH:-64}"
MIN_CODA_BATCHES="${MIN_CODA_BATCHES:-$CODA_BATCHES}"
NUM_REQUESTS="${NUM_REQUESTS:-256}"
BACKENDS="${BACKENDS:-cdb-norefill cdb-refill}"
MAX_NUM_BATCHED_TOKENS=8192
OUTPUT_DIR="${OUTPUT_DIR:-outputs/ablations/schedule-trace}"
SUMMARY_PATH="$OUTPUT_DIR/schedule-trace_${MODEL}_${DATASET}_w${WIDTH}.jsonl"

mkdir -p "$OUTPUT_DIR"
echo "Experiment: schedule trace, $MODEL_ID, $DATASET, q=$THRESHOLD, width $WIDTH, $NUM_REQUESTS requests"
echo "Output:     $OUTPUT_DIR"
echo ""

run_trace() {
    local variant="$1"
    shift
    local trace_path="$OUTPUT_DIR/schedule-trace_${MODEL}_${DATASET}_w${WIDTH}_${variant}.jsonl"
    if [[ -e "$trace_path" ]]; then
        echo "ERROR: refusing to overwrite $trace_path" >&2
        return 1
    fi
    local args=("${base_args[@]}" "$@")
    args+=(--trace-output "$trace_path")
    uv run python scripts/benchmark_throughput.py "${args[@]}"
    echo ""
}

for backend in $BACKENDS; do
    mapfile -t base_args < <(engine_flags "$backend")
    base_args+=(
        --model "$MODEL_ID"
        --workload "$WORKLOAD"
        --num-requests "$NUM_REQUESTS"
        --exit-threshold "$THRESHOLD"
        --max-recurrent-depth "$RECUR_STEPS"
        --max-model-len "$MAX_MODEL_LEN"
        --max-num-seqs "$WIDTH"
        --attn-implementation "$ATTN_IMPL"
        --num-blocks "$NUM_BLOCKS"
        --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
        --block-size "$BLOCK_SIZE"
        --summary-output "$SUMMARY_PATH"
    )
    if [[ "$backend" == cdb-refill ]]; then
        for min_coda in $MIN_CODA_BATCHES; do
            [[ "$min_coda" == width ]] && min_coda=$WIDTH
            echo "=== backend: $backend, minimum coda batch $min_coda ==="
            run_trace "${backend}_k${min_coda}" --min-coda-batch-size "$min_coda"
        done
    else
        echo "=== backend: $backend ==="
        run_trace "$backend"
    fi
done
