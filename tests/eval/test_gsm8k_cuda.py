"""End-to-end GSM8k harness tests on a real Ouro model (CUDA-gated)."""

import pytest
from evaluate_accuracy import build_arg_parser, build_backend

from looped_cdb.eval.runner import run_task
from looped_cdb.eval.tasks import get_task

pytestmark = pytest.mark.cuda

LIMIT = 8
NUM_FEWSHOT = 3
RECUR_STEPS = 4


def _run(backend_name: str):
    argv = [
        "--backend",
        backend_name,
        "--num-fewshot",
        str(NUM_FEWSHOT),
        "--limit",
        str(LIMIT),
        "--recur-steps",
        str(RECUR_STEPS),
        "--num-blocks",
        "1024",
    ]
    args = build_arg_parser().parse_args(argv)
    backend = build_backend(args)
    result = run_task(get_task("gsm8k_cot"), backend, num_fewshot=NUM_FEWSHOT, limit=LIMIT, log_samples=True)
    assert result.num_docs == LIMIT
    for name, value in result.metrics.items():
        assert 0.0 <= value <= 1.0, f"{name} out of range: {value}"
    return result


def test_gsm8k_cot_cb_end_to_end():
    result = _run("cb")
    # Ouro-1.4B on 3-shot gsm8k_cot should get at least one of eight correct.
    assert result.metrics["flexible-extract"] > 0.0


def test_gsm8k_cot_cdb_smoke():
    # Full-depth CDB (no trained gate) should run and score in range.
    result = _run("cdb")
    assert result.metrics["flexible-extract"] >= 0.0


def test_gsm8k_cot_cdb_early_exit_routes_depth():
    # CDB early exit (built-in gate) must run, produce an exit-depth histogram
    # respecting min_exit_step, and actually exit some tokens early.
    argv = [
        "--backend",
        "cdb",
        "--num-fewshot",
        str(NUM_FEWSHOT),
        "--limit",
        "8",
        "--recur-steps",
        str(RECUR_STEPS),
        "--exit-threshold",
        "0.5",
        "--min-recurrent-steps",
        "2",
        "--num-blocks",
        "1024",
        # The depth-indexed default rejects gated exits shallower than its slot count;
        # gated rows serve copy-on-exit routing, as in the accuracy sweep.
        "--kv-policy",
        "last_exited",
    ]
    args = build_arg_parser().parse_args(argv)
    backend = build_backend(args)
    result = run_task(get_task("gsm8k_cot"), backend, num_fewshot=NUM_FEWSHOT, limit=8)
    assert result.exit_depth_counts, "expected a non-empty exit-depth histogram"
    assert min(result.exit_depth_counts) >= 2  # min_exit_step=2 (1-indexed)
    assert result.mean_exit_depth is not None
    assert 2.0 <= result.mean_exit_depth < float(RECUR_STEPS)  # some tokens exit before full depth
