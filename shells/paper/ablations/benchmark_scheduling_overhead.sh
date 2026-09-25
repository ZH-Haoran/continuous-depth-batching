#!/usr/bin/env bash
# Measures GPU idle time under CDB with synchronous scheduling, asynchronous scheduling, and lookahead.
# The analysis reports GPU busy/idle and per-NVTX-stage time.
#
# Submit with:
#   ./shells/_submit.sh shells/paper/ablations/benchmark_scheduling_overhead.sh

set -euo pipefail
source "${REPO_ROOT:?submit with ./shells/_submit.sh}/shells/paper/_common.sh"

NSYS_BIN="${NSYS_BIN:-nsys}"
command -v "$NSYS_BIN" >/dev/null || { echo "ERROR: nsys is required" >&2; exit 1; }
MODEL_ID="$OURO_1_4B_MODEL_ID"
WORKLOAD="$(workload_for ouro sharegpt)"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/ablations/scheduling-overhead}"
mkdir -p "$OUTPUT_DIR"

for variant in sync async lookahead; do
    flags=()
    case "$variant" in
        sync) flags+=(--sync --no-delay-gate-consumption) ;;
        async) flags+=(--no-delay-gate-consumption) ;;
        lookahead) ;;
    esac
    rep="$OUTPUT_DIR/$variant.nsys-rep"
    sqlite="$OUTPUT_DIR/$variant.sqlite"
    for artifact in "$rep" "$sqlite" "$OUTPUT_DIR/$variant.jsonl"; do
        if [[ -e "$artifact" ]]; then
            echo "ERROR: refusing to overwrite $artifact" >&2
            exit 1
        fi
    done
    echo "Profiling $variant: graphs=all, predicted retirement, W=64, K=1, q=0.2"
    "$NSYS_BIN" profile \
        --trace=cuda,nvtx --sample=none --cpuctxsw=none \
        --cuda-graph-trace=graph --cuda-event-trace=false \
        --output="$rep" \
        python scripts/benchmark_throughput.py \
        --backend cdb --model "$MODEL_ID" --workload "$WORKLOAD" \
        --exit-threshold 0.2 --max-recurrent-depth 4 \
        --max-model-len "$MAX_MODEL_LEN" --max-num-seqs 64 \
        --min-coda-batch-size 1 --min-free-slots 8 \
        --attn-implementation "$ATTN_IMPL" --num-blocks "$NUM_BLOCKS" \
        --block-size "$BLOCK_SIZE" --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
        --cuda-graph-mode all --no-replay-eos-finishes \
        --limit 2000 --steady-warmup-requests 128 \
        --summary-output "$OUTPUT_DIR/$variant.jsonl" --nvtx "${flags[@]}"
    "$NSYS_BIN" export --type=sqlite --output="$sqlite" "$rep"
    python scripts/analyze_paper_nsys.py "$sqlite" --label "$variant" \
        --output-json "$OUTPUT_DIR/${variant}_analysis.json" \
        --output-csv "$OUTPUT_DIR/${variant}_analysis.csv"
done

echo "Scheduling controls: $OUTPUT_DIR"
