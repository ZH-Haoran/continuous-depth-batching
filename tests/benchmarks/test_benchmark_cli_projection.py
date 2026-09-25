"""Projection of the parsed CLI onto the engine and run configs.

The two projections hand the same namespace to different dataclasses, so a flag routed to the
wrong one raises only once the script runs, which nothing else here does. Building both from a
real ``parse_args`` namespace fails on a misrouted flag instead of on a submitted sweep.
"""

from __future__ import annotations

import sys

from benchmark_throughput import engine_args_from_cli, parse_args, run_config_from_cli

from looped_cdb.benchmarks.metrics import BenchmarkConfig


def _args(argv: list[str]):
    original = sys.argv
    sys.argv = ["benchmark_throughput.py", *argv]
    try:
        return parse_args()
    finally:
        sys.argv = original


def test_cli_flags_reach_the_config_that_accepts_them() -> None:
    args = _args(
        [
            "--backend",
            "cdb",
            "--min-recurrent-steps",
            "3",
            "--min-exit-step",
            "2",
            "--exit-delay-steps",
            "1",
            "--min-coda-batch-size",
            "16",
            "--layer-split",
            "1-4-1",
        ]
    )

    engine_args = engine_args_from_cli(args)
    run_config = run_config_from_cli(args)

    # min_recurrent_steps is the engine-side depth floor, so it belongs to the engine args; the
    # run config carries the offline depth-derivation settings and the model-shape overrides.
    assert engine_args.min_recurrent_steps == 3
    assert engine_args.min_coda_batch_size == 16
    assert run_config.min_exit_step == 2
    assert run_config.exit_delay_steps == 1
    assert run_config.layer_split == "1-4-1"


def test_schedule_flags_have_somewhere_to_be_recorded() -> None:
    # A flag that changes the served schedule but has no BenchmarkConfig field cannot be recorded
    # at all, so its rows claim whatever the default was and aggregate against runs that differ.
    args = _args(["--backend", "cdb"])
    recorded = set(BenchmarkConfig.__dataclass_fields__)

    schedule_flags = {
        "min_recurrent_steps",
        "min_exit_step",
        "exit_delay_steps",
        "min_coda_batch_size",
        "exit_threshold",
        "layer_split",
        "max_num_seqs",
        "min_free_slots",
        "safety_margin",
        "max_model_len",
        "kv_pressure_mode",
    }
    assert schedule_flags <= recorded
    assert schedule_flags <= set(vars(args))
