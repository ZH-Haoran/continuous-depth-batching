#!/usr/bin/env bash
# Ablates Huginn's prelude-core-coda layer split for offline throughput (ShareGPT).
#
# The replay still forces the exit depths recorded from the released
# 2-4-2 model, so every split serves an identical work schedule and only the fixed-cost
# structure changes.
#
# Submit with: ./shells/_submit.sh shells/paper/ablations/layer_split_huginn.sh

set -euo pipefail
source "${REPO_ROOT:?submit with ./shells/_submit.sh}/shells/paper/_common.sh"

MODEL_ID="$HUGINN_MODEL_ID"
RECUR_STEPS=16
WORKLOAD="$(workload_for huginn sharegpt)"
THRESHOLD=0.28 # earliest threshold from exit sweep
OUTPUT_PATH="outputs/ablations/layer-split/layer-split_huginn_sharegpt.jsonl"

SPLITS="${SPLITS:-0-4-0 1-4-1}"
BACKENDS="${BACKENDS:-cb cdb-norefill cdb-refill}"
MIN_CODA_BATCHES="${MIN_CODA_BATCHES:-1 8 16 32 64}"

DECODE_WIDTH=64 # Maximum decode batch size used in the throughput sweeps
NUM_REQUESTS="$(num_requests_for sharegpt)"

mkdir -p "$(dirname "$OUTPUT_PATH")"
echo "Experiment: prelude-core-coda layer split (offline throughput), $MODEL_ID, sharegpt, q=$THRESHOLD"
echo "Splits:     $SPLITS (backends: $BACKENDS)"
echo "Max decode batch size: $DECODE_WIDTH"
echo "Output:     $OUTPUT_PATH"
echo ""

for split in $SPLITS; do
    for backend in $BACKENDS; do
        # Only cdb-refill batches codas.
        batches="-"
        [[ "$backend" == cdb-refill ]] && batches="$MIN_CODA_BATCHES"
        for min_coda in $batches; do
            echo "=== split: $split  backend: $backend  min coda batch: $min_coda ==="
            mapfile -t args < <(engine_flags "$backend")
            args+=(
                --model "$MODEL_ID"
                --layer-split "$split"
                --workload "$WORKLOAD"
                --exit-threshold "$THRESHOLD"
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
            [[ "$backend" == cdb-refill ]] && args+=(--min-coda-batch-size "$min_coda")
            uv run python scripts/benchmark_throughput.py "${args[@]}"
            echo ""
        done
    done
done
