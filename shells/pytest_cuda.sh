#!/usr/bin/env bash
# Run the CUDA-marked pytest suite on a GPU node.
#
# Usage:
#   ./shells/_submit.sh shells/pytest_cuda.sh

set -euo pipefail

export REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$HOME/looped-lm-continuous-batching}}"
cd "$REPO_ROOT"

CONFIG="${MACHINE_CONFIG:-shells/_machine_config.sh}"
# shellcheck source=/dev/null
source "$CONFIG"
validate_config || exit 1
# FA2 (the non-Hopper fallback) is an extra because it compiles from source and needs
# nvcc; force the local compile so the binary never depends on a GitHub download. The
# build runs without isolation against the venv, so sync the base environment first.
export FLASH_ATTENTION_FORCE_BUILD=TRUE
uv sync --frozen
uv sync --frozen --extra fa2
# shellcheck source=/dev/null
source .venv/bin/activate
# shellcheck source=/dev/null
source "$CONFIG"

nvidia-smi --query-gpu=name --format=csv,noheader || true
LOOPED_CDB_RUN_CUDA_TESTS=1 uv run pytest -m cuda -q "${PYTEST_ARGS:-}"
