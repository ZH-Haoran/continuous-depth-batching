#!/usr/bin/env bash
# GSM8K (gsm8k_cot) accuracy sweep for Ouro.
#
# Fixed-depth rows run on CB, otherwise on CDB.
#
# The CDB scheduler consumes an exit decision one recurrent step late, which needs to be disabled:
#   gated      the published gate, disabled delayed gate (--no-delay-gate-consumption)
#   lookahead  a trained gate that predicts the exit one step ahead.
#   preloop    a trained gate that fixes each token's depth before the loop starts
# Trained gates load <gate>_exit_gate.safetensors from TRAINED_GATE_DIR
# (see shells/paper/ablations/train_ouro_gate.sh).
#
# Every row runs in sequence within one job:
#   ./shells/_submit.sh shells/paper/ablations/accuracy_gsm8k_ouro.sh

source "${REPO_ROOT:?submit with ./shells/_submit.sh}/shells/paper/_common.sh"

MODEL_ID="$OURO_1_4B_MODEL_ID"
# 3-shot matches the Ouro paper.
NUM_FEWSHOT="${NUM_FEWSHOT:-3}"
MAX_RECUR="${MAX_RECUR:-4}"
LIMIT="${LIMIT:-}"
TRAINED_GATE_DIR="${TRAINED_GATE_DIR:-outputs/ablations/ouro_gate}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/ablations/accuracy/paper_gsm8k_ouro.jsonl}"

FIXED_STEPS="${FIXED_STEPS-1 2 3 4}"
THRESHOLDS="${THRESHOLDS-0.01 0.1 0.2 0.25 0.3 0.4 0.5 0.7 0.9}"
GATES="${GATES-gated lookahead preloop}"
# Every valid fixed-depth and layout pair is evaluated.
FIXED_LAYOUTS="${FIXED_LAYOUTS-depth_indexed single first_then_shared}"
# Every gated row runs under every layout.
GATED_LAYOUTS="${GATED_LAYOUTS-last_exited}"

gate_path() { echo "$TRAINED_GATE_DIR/${1}_exit_gate.safetensors"; }

# Check that the trained gates exist.
for gate in ${THRESHOLDS:+$GATES}; do
    case "$gate" in
        gated) ;;
        lookahead | preloop)
            if [[ -z "$TRAINED_GATE_DIR" ]]; then
                echo "ERROR: GATES '$GATES' includes a trained gate; set TRAINED_GATE_DIR (or GATES=gated)" >&2
                exit 1
            fi
            if [[ ! -f "$(gate_path "$gate")" ]]; then
                echo "ERROR: trained gate not found: $(gate_path "$gate")" >&2
                exit 1
            fi
            ;;
        *) echo "ERROR: unknown gate '$gate' (expected: gated, lookahead, preloop)" >&2; exit 1 ;;
    esac
done

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
echo "Thresholds: ${THRESHOLDS:-none} over gates: ${GATES:-none}, layouts: ${GATED_LAYOUTS:-none}"
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
    for gate in $GATES; do
        for threshold in $THRESHOLDS; do
            case "$gate" in
                gated)
                    run_eval "gate q=$threshold, kv $layout" "$layout" \
                        --backend cdb --recur-steps "$MAX_RECUR" --exit-threshold "$threshold" \
                        --min-recurrent-steps 2 --no-delay-gate-consumption
                    ;;
                lookahead)
                    run_eval "lookahead q=$threshold, kv $layout" "$layout" \
                        --backend cdb --recur-steps "$MAX_RECUR" --exit-threshold "$threshold" \
                        --exit-gate-type lookahead --exit-gate-path "$(gate_path lookahead)"
                    ;;
                preloop)
                    run_eval "preloop q=$threshold, kv $layout" "$layout" \
                        --backend cdb --recur-steps "$MAX_RECUR" --exit-threshold "$threshold" \
                        --exit-gate-type preloop --exit-gate-path "$(gate_path preloop)" \
                        --min-recurrent-steps 2
                    ;;
            esac
        done
    done
done
