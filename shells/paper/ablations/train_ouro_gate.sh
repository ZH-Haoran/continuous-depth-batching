#!/usr/bin/env bash
# Train the Ouro exit gates by distillation against the model's built-in gate:
# the lookahead and preloop gates plus a same-step control.
#
# Submit with: ./shells/_submit.sh shells/paper/ablations/train_ouro_gate.sh

source "${REPO_ROOT:?submit with ./shells/_submit.sh}/shells/paper/_common.sh"

MODEL_ID="$OURO_1_4B_MODEL_ID"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/ablations/ouro_gate}"

mkdir -p "$OUTPUT_DIR"
echo "Experiment: Ouro exit-gate training (lookahead + preloop, same-step control)"
echo "Output:     $OUTPUT_DIR"
echo ""

uv run --extra train python scripts/train_ouro_gate.py \
    --model-name-or-path "$MODEL_ID" \
    --output-dir "$OUTPUT_DIR"
