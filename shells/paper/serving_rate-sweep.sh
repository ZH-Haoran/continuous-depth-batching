#!/usr/bin/env bash
# Open-loop serving latency against the offered request rate.
#
# Each point uses a seeded Poisson arrival trace shared across backends at that rate.
#
# The engine uses the same maximum decode batch size as the offline throughput sweep
# for each workload.
# Each backend's capacity equals its offline drain throughput divided by the workload's mean output
# length, so the throughput figure predicts the latency knee.
# The rate grids below are placed around those predicted knees.
# Submit with: MODEL=ouro DATASET=sharegpt ./shells/_submit.sh shells/paper/serving_rate-sweep.sh

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
        THRESHOLDS="${THRESHOLDS:-0.2 0.4 0.5 0.7}"
        ;;
    huginn)
        MODEL_ID="$HUGINN_MODEL_ID"
        RECUR_STEPS=16
        MIN_CODA_BATCH=$((DECODE_WIDTH / 4))
        THRESHOLDS="${THRESHOLDS:-0.28 0.16 0.1}"
        ;;
    *) echo "unknown MODEL '$MODEL' (expected: ouro, huginn)" >&2; exit 1 ;;
esac

# The exit-depth policy (minimum depth and the delayed-scheduler step) comes from the bundle's
# recorded depth_defaults, so a replay matches how the workload was recorded.
#
# One rate grid per workload, shared by every threshold: ~6 points spanning cb's knee (the
# floor, threshold-independent) up past the widest-gap threshold's cdb-refill knee (the ceiling;
# comments list cb / cdb-norefill / cdb-refill in req/s at that threshold). Points above a backend's
# capacity diverge and read as "saturated", so each backend gets at most one or two of them.
case "$MODEL/$DATASET" in
    ouro/sharegpt)
        # Knees at q=0.2: cb 4.7, cdb-norefill 5.7, cdb-refill 7.2 req/s.
        DEFAULT_RATES="3 4 5 6 7 8"
        ;;
    ouro/alpaca)
        # Knees at q=0.2: cb 34.9, cdb-norefill 39.5, cdb-refill 53.9 req/s.
        DEFAULT_RATES="25 30 35 40 50 60"
        ;;
    ouro/arxiv)
        # Knees at q=0.2 and W=8: cb 0.393, cdb-norefill 0.474, cdb-refill 0.518 req/s.
        DEFAULT_RATES="0.36 0.40 0.44 0.48 0.52 0.56"
        ;;
    huginn/sharegpt)
        # Knees at q=0.28: cb 1.84, cdb-norefill 2.44, cdb-refill 2.43 req/s.
        DEFAULT_RATES="1 1.5 2 2.5 3 3.5"
        ;;
    huginn/alpaca)
        # Knees at q=0.28: cb 13.8, cdb-norefill 18.0, cdb-refill 17.6 req/s.
        DEFAULT_RATES="5 10 15 20 25 30"
        ;;
    huginn/arxiv)
        # Knees at q=0.28 and W=8: cb 0.178, cdb-norefill 0.244, cdb-refill 0.233 req/s.
        DEFAULT_RATES="0.15 0.18 0.20 0.23 0.25 0.28"
        ;;
    *) echo "unknown DATASET '$DATASET' (expected: sharegpt, alpaca, arxiv)" >&2; exit 1 ;;
esac
RATES="${RATES:-$DEFAULT_RATES}"
WORKLOAD="$(workload_for "$MODEL" "$DATASET")"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/serving-rate/serving-rate_${MODEL}_${DATASET}.jsonl}"
RUN_TAG="$(date -u +%Y%m%dT%H%M%S)-${SLURM_JOB_ID:-$$}"
LATENCY_EVENTS_DIR="${LATENCY_EVENTS_DIR:-results/serving-rate/events_${MODEL}_${DATASET}_${RUN_TAG}}"

# Overridable for partial runs: cb ignores the exit threshold, so a second-threshold sweep
# reuses the cb rows already measured at the default threshold (BACKENDS="cdb-norefill cdb-refill").
BACKENDS="${BACKENDS:-cb cdb-norefill cdb-refill}"
REPEATS=1
# Base seed of the Poisson arrival trace; repeat r uses ARRIVAL_SEED + r.
ARRIVAL_SEED=0
# Used to set the number of arrivals as ceil(rate * TRACE_DURATION_S).
# Every run drains its full trace.
TRACE_DURATION_S=600
# Default admission keeps residency near the maximum decode batch size.

mkdir -p "$(dirname "$OUTPUT_PATH")"
echo "Experiment: open-loop serving vs request rate, $MODEL_ID, $DATASET"
echo "Thresholds: $THRESHOLDS"
echo "Rates:      $RATES req/s (${TRACE_DURATION_S}s target trace duration, arrival seed $ARRIVAL_SEED)"
echo "Max decode batch size: $DECODE_WIDTH"
echo "Output:     $OUTPUT_PATH"
echo ""

for repeat in $(seq 0 $((REPEATS - 1))); do
    for backend in $BACKENDS; do
        # cb decodes at full depth and ignores the threshold, so measure it once.
        if [ "$backend" = cb ]; then thresholds="${THRESHOLDS%% *}"; else thresholds="$THRESHOLDS"; fi
        for threshold in $thresholds; do
            for rate in $RATES; do
                echo "=== repeat: $repeat  threshold: $threshold  rate: $rate req/s  backend: $backend ==="
                mapfile -t args < <(engine_flags "$backend")
                args+=(
                    --model "$MODEL_ID"
                    --workload "$WORKLOAD"
                    --exit-threshold "$threshold"
                    --max-recurrent-depth "$RECUR_STEPS"
                    --max-model-len "$MAX_MODEL_LEN"
                    --request-rate "$rate"
                    --arrival-seed "$((ARRIVAL_SEED + repeat))"
                    --trace-duration-s "$TRACE_DURATION_S"
                    --max-num-seqs "$DECODE_WIDTH"
                    --attn-implementation "$ATTN_IMPL"
                    --num-blocks "$NUM_BLOCKS"
                    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
                    --block-size "$BLOCK_SIZE"
                    --repeat-index "$repeat"
                    --summary-output "$OUTPUT_PATH"
                    --latency-events-dir "$LATENCY_EVENTS_DIR"
                )
                [[ -n "${WANDB_PROJECT:-}" ]] && args+=(--wandb-project "$WANDB_PROJECT")
                # No-refill needs no minimum-batch flag: the engine's wave loop is the strict wave,
                # running one wave-sized coda at the boundary.
                [[ "$backend" == cdb-refill && -n "$MIN_CODA_BATCH" ]] && args+=(--min-coda-batch-size "$MIN_CODA_BATCH")
                uv run --no-sync python scripts/benchmark_throughput.py "${args[@]}"
                echo ""
            done
        done
    done
done
uv run --no-sync python scripts/exporters/export_latency_results.py "$LATENCY_EVENTS_DIR" \
    --output "$LATENCY_EVENTS_DIR/comparison.csv"
