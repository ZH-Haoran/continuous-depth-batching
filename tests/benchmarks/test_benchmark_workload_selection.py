"""Workload selection for the throughput benchmark: cap to the context, then slice.

``--max-model-len`` drops unservable prompts, so slicing to ``--num-requests`` first
would replay fewer requests than the summary claims were run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
from benchmark_throughput import build_workload_from_cli, resolve_exit_policy

from looped_cdb.benchmarks.workload import Workload


def _bundle(path: Path, input_lens: list[int], output_lens: list[int]) -> Path:
    """Write a recorded bundle whose requests have the given lengths."""

    offsets = np.zeros(len(input_lens) + 1, dtype=np.int64)
    np.cumsum(np.asarray(output_lens, dtype=np.int64), out=offsets[1:])
    total = int(offsets[-1])
    pdf = np.tile(np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float16), (total, 1))
    workload = Workload(
        ids=[f"r{i}" for i in range(len(input_lens))],
        input_lens=np.asarray(input_lens, dtype=np.int32),
        output_lens=np.asarray(output_lens, dtype=np.int32),
        offsets=offsets,
        exit_pdf=pdf,
        meta={"dataset": "test"},
    )
    json_path, _ = workload.save(path)
    return json_path


def _args(**overrides) -> argparse.Namespace:
    base = {
        "dataset": "bundle",
        "workload": None,
        "num_requests": None,
        "max_model_len": 64,
        "exit_dist": None,
        "input_len": 8,
        "output_len": 4,
        "input_len_high": None,
        "output_len_high": None,
        "max_recurrent_depth": 4,
        "seed": 0,
        "request_rate": None,
        "trace_duration_s": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_num_requests_is_exact_when_the_context_filter_drops_prompts(tmp_path: Path) -> None:
    # Six requests, three with a prompt too long to serve at max_model_len=64. Slicing to 3
    # before the filter would leave fewer than 3 to replay.
    path = _bundle(tmp_path / "w.json", input_lens=[10, 100, 10, 100, 10, 100], output_lens=[4] * 6)

    workload = build_workload_from_cli(_args(workload=path, num_requests=3, max_model_len=64))

    assert workload.num_requests == 3
    assert [int(x) for x in workload.input_lens] == [10, 10, 10]


def test_asking_for_more_requests_than_fit_the_context_is_an_error(tmp_path: Path) -> None:
    # Reporting throughput over "4 requests" while replaying 3 would misstate the workload.
    path = _bundle(tmp_path / "w.json", input_lens=[10, 100, 10, 10], output_lens=[4] * 4)

    with pytest.raises(ValueError, match="only 3 in"):
        build_workload_from_cli(_args(workload=path, num_requests=4, max_model_len=64))


def test_trace_duration_sizes_the_request_count(tmp_path: Path) -> None:
    # A target duration sets the count to ceil(rate * duration); the realized Poisson
    # duration remains stochastic.
    path = _bundle(tmp_path / "w.json", input_lens=[10] * 8, output_lens=[4] * 8)

    workload = build_workload_from_cli(_args(workload=path, request_rate=2.0, trace_duration_s=2.6))

    assert workload.num_requests == 6

    # A trace the bundle cannot fill raises rather than quietly ending early.
    with pytest.raises(ValueError, match="only 8 in"):
        build_workload_from_cli(_args(workload=path, request_rate=5.0, trace_duration_s=10.0))


def test_outputs_are_truncated_to_the_context_before_slicing(tmp_path: Path) -> None:
    # A servable prompt with an over-long output is kept, with its output truncated.
    path = _bundle(tmp_path / "w.json", input_lens=[60, 10], output_lens=[20, 4])

    workload = build_workload_from_cli(_args(workload=path, num_requests=2, max_model_len=64))

    assert workload.num_requests == 2
    assert [int(x) for x in workload.output_lens] == [4, 4]  # 64 - 60 = 4


def test_a_bundle_that_fits_is_unchanged(tmp_path: Path) -> None:
    path = _bundle(tmp_path / "w.json", input_lens=[10, 12, 14], output_lens=[4, 4, 4])

    workload = build_workload_from_cli(_args(workload=path, num_requests=None, max_model_len=64))

    assert workload.num_requests == 3
    assert [int(x) for x in workload.output_lens] == [4, 4, 4]


def test_synthetic_requests_that_cannot_be_served_are_an_error() -> None:
    # Every synthetic prompt is longer than the context, so none could be replayed.
    with pytest.raises(ValueError, match="no request fits"):
        build_workload_from_cli(
            _args(dataset="random", num_requests=4, input_len=100, exit_dist="bimodal_50", max_model_len=64)
        )


def test_exit_policy_resolves_from_bundle_defaults_as_a_pair() -> None:
    # A bundle's recorded (min_exit_step, exit_delay_steps) is one policy: unset flags take
    # both values, explicit flags take both, and a partial override would silently produce a
    # pairing the bundle was never recorded for, so it raises instead.
    meta = {"depth_defaults": {"min_exit_step": 1, "exit_delay_steps": 1}}

    assert resolve_exit_policy(None, None, meta) == (1, 1)
    assert resolve_exit_policy(2, 0, meta) == (2, 0)
    with pytest.raises(ValueError, match="as a pair"):
        resolve_exit_policy(None, 0, meta)
    with pytest.raises(ValueError, match="as a pair"):
        resolve_exit_policy(1, None, meta)

    # Without recorded defaults the legacy per-flag globals apply, partially or fully.
    assert resolve_exit_policy(None, None, {}) == (2, 0)
    assert resolve_exit_policy(None, 1, {}) == (2, 1)
    assert resolve_exit_policy(3, None, {}) == (3, 0)


def test_synthetic_workload_keeps_every_requested_request() -> None:
    workload = build_workload_from_cli(
        _args(dataset="random", num_requests=5, input_len=8, output_len=4, exit_dist="bimodal_50", max_model_len=64)
    )

    assert workload.num_requests == 5
