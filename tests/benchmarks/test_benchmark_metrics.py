import json
from types import SimpleNamespace

import pytest

from looped_cdb.benchmarks.flop_bound import StageFlops
from looped_cdb.benchmarks.metrics import (
    BenchmarkConfig,
    BenchmarkSummary,
    ExitDistributionSummary,
    request_latency_summary,
)


def _length_stats(prefix: str) -> dict[str, int | float | None]:
    return {
        f"{prefix}_total_tokens": 0,
        f"{prefix}_min_tokens": None,
        f"{prefix}_max_tokens": None,
        f"{prefix}_mean_tokens": 0.0,
    }


def _config(**overrides: object) -> BenchmarkConfig:
    kwargs: dict[str, object] = {
        "backend": "cb",
        "model": "dummy",
        "max_recurrent_depth": 4,
        "workload_path": "outputs/workloads/toy.json",
        "workload_name": "toy",
        "num_requests": 2,
        "measured_repeat": 0,
        "seed": 3,
        "attn_implementation": "paged|flash_attention_3",
        "use_async_batching": True,
        "use_cuda_graph": True,
        "cuda_graph_mode": "decode",
        "block_size": 256,
        "num_blocks": 16,
        "max_num_batched_tokens": 32,
        **_length_stats("prompt"),
        **_length_stats("output"),
    }
    kwargs.update(overrides)
    return BenchmarkConfig(**kwargs)


def test_exit_distribution_summary_from_depths() -> None:
    summary = ExitDistributionSummary.from_depths("workload", max_depth=4, depths=[1, 1, 4, 2])
    assert summary.token_count == 4
    assert summary.total_depth_work == 8
    assert summary.mean_depth == 2.0
    assert summary.depth_histogram == {"1": 2, "2": 1, "4": 1}


def test_exit_distribution_summary_from_histogram_matches_depths() -> None:
    from_hist = ExitDistributionSummary.from_histogram("workload", max_depth=4, depth_histogram={1: 2, 4: 1})
    from_depths = ExitDistributionSummary.from_depths("workload", max_depth=4, depths=[1, 1, 4])
    assert from_hist == from_depths


def test_exit_distribution_summary_rejects_out_of_range_depth() -> None:
    with pytest.raises(ValueError, match=r"\[1, 4\]"):
        ExitDistributionSummary.from_depths("workload", max_depth=4, depths=[5])


def test_benchmark_config_to_json_carries_workload_fields() -> None:
    data = _config(exit_threshold=0.5, min_exit_step=2).to_json_dict()
    assert data["workload_name"] == "toy"
    assert data["num_requests"] == 2
    assert data["exit_threshold"] == 0.5
    assert data["min_exit_step"] == 2
    assert "exit_plan" not in data


def test_benchmark_summary_serializes_stable_throughput_fields() -> None:
    requested = ExitDistributionSummary.from_depths("workload", max_depth=4, depths=[1] * 10)
    effective = ExitDistributionSummary.from_depths("all_full", max_depth=4, depths=[4] * 10)
    summary = BenchmarkSummary(
        config=_config(),
        run_id="cb-D4-toy-qnone-repeat0",
        wall_time_s=2.0,
        generated_tokens=10,
        completed_requests=2,
        requested_exit_distribution=requested,
        effective_exit_distribution=effective,
        peak_cuda_memory_allocated_bytes=100,
        peak_cuda_memory_reserved_bytes=200,
        kv_cache_num_blocks=16,
        kv_cache_block_size=256,
        kv_cache_max_num_batched_tokens=32,
        kv_cache_num_pages=4096,
        kv_cache_peak_blocks_used=8,
        kv_cache_final_blocks_used=0,
        stage_flops=StageFlops(f0=0, fr=100),
        request_latency={"end_to_end_latency_s": {"p50": 1.2}},
    )

    data = json.loads(summary.to_json())

    assert data["generated_tokens_per_second"] == 5.0
    assert data["completed_requests_per_second"] == 1.0
    assert data["recurrent_steps_per_second"] == 20.0
    assert data["full_depth_work"] == 40
    assert data["skipped_depth_work_available"] == 30
    assert data["effective_skipped_depth_work"] == 0
    assert data["depth_work_over_requested"] == 30
    # Weightless boundary stages (f0=0) reduce the bound to the depth ratio 4 / 1.
    assert data["flop_bound_speedup"] == 4.0
    assert data["stage_flops"] == {"f0_params": 0, "fr_params": 100}
    assert data["config"]["cdb_kv_policy"] == "single"
    assert data["config"]["workload_name"] == "toy"
    assert data["kv_cache"]["peak_blocks_used"] == 8
    assert data["kv_cache"]["peak_blocks_used_fraction"] == 0.5
    assert data["kv_cache"]["final_blocks_used"] == 0
    assert data["request_latency"]["end_to_end_latency_s"]["p50"] == 1.2


def test_benchmark_summary_without_requested_distribution_serializes_nulls() -> None:
    # The full-depth cb baseline ignores the exit schedule; its rows record no requested
    # distribution, and everything derived from one must read as absent rather than as a value
    # echoing whatever threshold the sweep launched the row under.
    effective = ExitDistributionSummary.from_depths("all_full", max_depth=4, depths=[4] * 10)
    summary = BenchmarkSummary(
        config=_config(),
        run_id="cb-D4-toy-qnone-repeat0",
        wall_time_s=2.0,
        generated_tokens=10,
        completed_requests=2,
        requested_exit_distribution=None,
        effective_exit_distribution=effective,
        stage_flops=StageFlops(f0=0, fr=100),
        peak_cuda_memory_allocated_bytes=100,
        peak_cuda_memory_reserved_bytes=200,
        kv_cache_num_blocks=16,
        kv_cache_block_size=256,
        kv_cache_max_num_batched_tokens=32,
        kv_cache_num_pages=4096,
    )

    data = json.loads(summary.to_json())

    assert data["requested_exit_distribution"] is None
    assert data["skipped_depth_work_available"] is None
    assert data["depth_work_over_requested"] is None
    assert data["flop_bound_speedup"] is None
    assert data["effective_skipped_depth_work"] == 0


def _finished_output(
    *,
    request_id: str = "request-0",
    created: float,
    scheduled: float,
    first_token: float,
    finished: float,
    generated: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        created_time=created,
        lifespan=(scheduled, finished),
        first_token_time=first_token,
        generated_tokens=list(range(generated)),
    )


def test_request_latency_summary_reports_per_request_latencies() -> None:
    outputs = [
        _finished_output(created=10.0, scheduled=10.5, first_token=11.0, finished=13.0, generated=5),
        _finished_output(
            request_id="request-1", created=20.0, scheduled=20.0, first_token=20.2, finished=20.2, generated=1
        ),
    ]

    summary = request_latency_summary(outputs)

    assert summary["num_requests"] == 2
    assert summary["queue_s"]["mean"] == pytest.approx(0.25)
    assert summary["queue_s"]["n"] == 2
    assert summary["ttft_s"]["mean"] == pytest.approx(0.6)
    assert summary["ttft_s"]["max"] == pytest.approx(1.0)
    assert summary["e2e_s"]["mean"] == pytest.approx(1.6)
    assert summary["e2e_s"]["p50"] == pytest.approx(1.6)
    # TPOT excludes the single-token request: (13.0 - 11.0) / 4 decode gaps, over a
    # population of one that the stat block must record.
    assert summary["tpot_s"]["mean"] == pytest.approx(0.5)
    assert summary["tpot_s"]["n"] == 1
    # Normalized latency is per-request e2e over output length, averaged afterwards:
    # (3.0 / 5 + 0.2 / 1) / 2, not mean(e2e) / mean(len).
    assert summary["norm_e2e_s_per_token"]["mean"] == pytest.approx(0.4)


def test_request_latency_summary_rejects_missing_stamps() -> None:
    with pytest.raises(ValueError, match="at least one output"):
        request_latency_summary([])
    unfinished = _finished_output(created=1.0, scheduled=1.5, first_token=-1.0, finished=2.0, generated=0)
    with pytest.raises(ValueError, match="latency stamp"):
        request_latency_summary([unfinished])
    single_token_only = [_finished_output(created=0.0, scheduled=0.0, first_token=0.1, finished=0.1, generated=1)]
    assert request_latency_summary(single_token_only)["tpot_s"] is None
