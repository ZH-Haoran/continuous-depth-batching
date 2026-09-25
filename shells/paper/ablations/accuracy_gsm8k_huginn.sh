#!/usr/bin/env bash
# GSM8K (gsm8k_cot) accuracy sweep for Huginn.
#
# Fixed-depth rows run on CB, otherwise on CDB.
#
# Every row runs in sequence within one job:
#   ./shells/_submit.sh shells/paper/ablations/accuracy_gsm8k_huginn.sh

source "${REPO_ROOT:?submit with ./shells/_submit.sh}/shells/paper/_common.sh"

MODEL_ID="$HUGINN_MODEL_ID"
# 8-shot matches the Huginn paper.
NUM_FEWSHOT="${NUM_FEWSHOT:-8}"
# Below the checkpoint's trained 32: accuracy is flat within noise above 16.
MAX_RECUR="${MAX_RECUR:-16}"
LIMIT="${LIMIT:-}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/ablations/accuracy/paper_gsm8k_huginn.jsonl}"

FIXED_STEPS="${FIXED_STEPS-6 7 8 9 10 12 14 16 20 24 32}"
THRESHOLDS="${THRESHOLDS-0.02 0.06 0.1 0.14 0.16 0.2 0.24 0.28 0.3 0.34}"
# Every fixed depth runs under every layout.
FIXED_LAYOUTS="${FIXED_LAYOUTS-depth_indexed single first_then_shared}"
# Every gated row runs under every layout.
GATED_LAYOUTS="${GATED_LAYOUTS-single last_exited}"

run_eval() {
    local label="$1" kv_policy="$2"
    shift 2
    local -a args=(
        --task gsm8k_cot --model "$MODEL_ID"
        --num-fewshot "$NUM_FEWSHOT"
        --attn-implementation "$ATTN_IMPL" --block-size "$BLOCK_SIZE"
        --kv-policy "$kv_policy" "$@"
        --summary-output "$OUTPUT_PATH"
    )
    [[ -n "$LIMIT" ]] && args+=(--limit "$LIMIT")

    echo "=== $label ==="
    uv run python scripts/evaluate_accuracy.py "${args[@]}"
    echo ""
}

mkdir -p "$(dirname "$OUTPUT_PATH")"
echo "Experiment: gsm8k_cot accuracy sweep, $MODEL_ID (${NUM_FEWSHOT}-shot)"
echo "Fixed:      ${FIXED_STEPS:-none} over layouts: ${FIXED_LAYOUTS:-none}"
echo "Thresholds: ${THRESHOLDS:-none} over layouts: ${GATED_LAYOUTS:-none} (at depth $MAX_RECUR)"
echo "Limit:      ${LIMIT:-full test set}"
echo "Output:     $OUTPUT_PATH"
echo ""

for layout in $FIXED_LAYOUTS; do
    for steps in $FIXED_STEPS; do
        # first_then_shared keeps a private slot for the first recurrent step and shares a
        # second, so it does not exist below two steps.
        if [[ "$layout" == "first_then_shared" && "$steps" -lt 2 ]]; then
            continue
        fi
        run_eval "fixed depth $steps, kv $layout" "$layout" --backend cb --recur-steps "$steps"
    done
done

for layout in $GATED_LAYOUTS; do
    for threshold in $THRESHOLDS; do
        run_eval "threshold $threshold, immediate, kv $layout" "$layout" \
            --backend cdb --recur-steps "$MAX_RECUR" --exit-threshold "$threshold" \
            --no-delay-gate-consumption
        run_eval "threshold $threshold, delayed, kv $layout" "$layout" \
            --backend cdb --recur-steps "$MAX_RECUR" --exit-threshold "$threshold"
    done
done
