#!/usr/bin/env bash
# Offline throughput against the maximum decode batch size, at a fixed exit threshold.
#
# Maximum decode batch sizes are chosen based on saturation batch size B* for each dataset.
# All runs are on average close to the configured maximum decode batch size.
#
# Submit with: MODEL=ouro DATASET=sharegpt ./shells/_submit.sh shells/paper/throughput_width-sweep.sh

set -euo pipefail
source "${REPO_ROOT:?submit with ./shells/_submit.sh}/shells/paper/_common.sh"

MODEL="${MODEL:?set MODEL to ouro or huginn}"
DATASET="${DATASET:?set DATASET to sharegpt, alpaca, or arxiv}"

case "$MODEL" in
    ouro)
        MODEL_ID="$OURO_1_4B_MODEL_ID"
        RECUR_STEPS=4
        MIN_CODA_BATCH=""
        THRESHOLD=0.2
        ;;
    huginn)
        MODEL_ID="$HUGINN_MODEL_ID"
        RECUR_STEPS=16
        MIN_CODA_BATCH=quarter
        THRESHOLD=0.28
        ;;
    *) echo "unknown MODEL '$MODEL' (expected: ouro, huginn)" >&2; exit 1 ;;
esac

WORKLOAD="$(workload_for "$MODEL" "$DATASET")"
OUTPUT_PATH="outputs/width-sweep/width-sweep_${MODEL}_${DATASET}.jsonl"
NUM_REQUESTS="$(num_requests_for "$DATASET")"
WIDTHS="${WIDTHS:-16 32 64 128 256 512}"
if [[ "$DATASET" == arxiv ]]; then
    WIDTHS="${WIDTHS_ARXIV:-2 4 8 16 32 64}"
fi

BACKENDS="${BACKENDS:-cb cdb-norefill cdb-refill}"
REPEATS=1

mkdir -p "$(dirname "$OUTPUT_PATH")"
echo "Experiment: drain throughput vs maximum decode batch size, $MODEL_ID, $DATASET, q=$THRESHOLD"
echo "Max decode batch sizes: $WIDTHS"
echo "Requests:   $NUM_REQUESTS"
echo "Output:     $OUTPUT_PATH"
echo ""

for repeat in $(seq 0 $((REPEATS - 1))); do
    for width in $WIDTHS; do
        for backend in $BACKENDS; do
            echo "=== repeat: $repeat  width: $width  backend: $backend ==="
            mapfile -t args < <(engine_flags "$backend")
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
                --repeat-index "$repeat"
                --summary-output "$OUTPUT_PATH"
            )
            if [[ "$backend" == cdb-refill && -n "$MIN_CODA_BATCH" ]]; then
                min_coda=$MIN_CODA_BATCH
                if [[ "$min_coda" == quarter ]]; then
                    min_coda=$((width / 4))
                    if ((min_coda < 1)); then
                        min_coda=1
                    fi
                fi
                args+=(--min-coda-batch-size "$min_coda")
            fi
            uv run python scripts/benchmark_throughput.py "${args[@]}"
            echo ""
        done
    done
done
