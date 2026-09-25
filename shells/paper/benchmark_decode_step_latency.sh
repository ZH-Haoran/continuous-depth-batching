#!/usr/bin/env bash
# Decode step latency vs (batch size, context length), measured through the
# continuous-batching engine with CUDA graphs on.
#
# Submit with: MODEL=ouro ./shells/_submit.sh shells/paper/benchmark_decode_step_latency.sh

set -euo pipefail
source "${REPO_ROOT:?submit with ./shells/_submit.sh}/shells/paper/_common.sh"

MODEL="${MODEL:?set MODEL to ouro, ouro26, or huginn}"

case "$MODEL" in
    ouro)
        MODEL_ID="$OURO_1_4B_MODEL_ID"
        DEPTHS="${DEPTHS:-1,2,3,4}"
        ;;
    ouro26)
        MODEL_ID="$OURO_2_6B_MODEL_ID"
        DEPTHS="${DEPTHS:-1,2,3,4}"
        ;;
    huginn)
        MODEL_ID="$HUGINN_MODEL_ID"
        DEPTHS="${DEPTHS:-2,4,8,12,16}"
        ;;
    *) echo "unknown MODEL '$MODEL' (expected: ouro, ouro26, huginn)" >&2; exit 1 ;;
esac
OUTPUT_PATH="${OUTPUT_PATH:-outputs/decode_step_latency/decode_step_latency_${MODEL}.jsonl}"
CONTEXT_LENGTHS="${CONTEXT_LENGTHS:-128 512 2048 8192}"

mkdir -p "$(dirname "$OUTPUT_PATH")"
echo "Experiment: decode-step latency, $MODEL_ID"
echo "Contexts:   $CONTEXT_LENGTHS"
echo "Output:     $OUTPUT_PATH"
echo ""

for ctx in $CONTEXT_LENGTHS; do
    echo "=== context length: $ctx ==="
    uv run python scripts/benchmark_decode_step_latency.py \
        --model "$MODEL_ID" \
        --attn-implementation "$ATTN_IMPL" \
        --block-size "$BLOCK_SIZE" \
        --depths "$DEPTHS" \
        --context-length "$ctx" \
        --output-path "$OUTPUT_PATH"
    echo ""
done
