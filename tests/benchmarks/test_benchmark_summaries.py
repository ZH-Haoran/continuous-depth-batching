"""Aggregation of raw benchmark summary rows into one row per measured configuration."""

import json
import statistics
from pathlib import Path
from typing import Any

import pytest

from looped_cdb.benchmarks.summaries import (
    aggregate_summaries,
    baseline_row,
    load_summaries,
    threshold_of,
)


def _summary(
    *,
    backend: str = "cb",
    refill: bool = True,
    exit_threshold: float | None = None,
    gen_tps: float = 100.0,
    ideal: float | None = 1.0,
    depth: int = 4,
    workload_name: str = "alpaca",
    workload_id: str | None = None,
    request_rate_rps: float | None = None,
    arrival_seed: int | None = None,
    max_num_seqs: int = 8,
    layer_split: str | None = None,
    request_latency: dict[str, Any] | None = None,
    steady_state: dict[str, Any] | None = None,
    kv_frac: float | None = 0.8,
    depth_queue: int | None = None,
    preemptions: int | None = None,
    batch_stats: dict[str, float] | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "config": {
            "backend": backend,
            "refill": refill,
            "exit_threshold": exit_threshold,
            "workload_name": workload_name,
            "workload_id": workload_id,
            "num_requests": 2,
            "max_num_batched_tokens": 32,
            "max_recurrent_depth": depth,
            "num_blocks": 16,
            "max_num_seqs": max_num_seqs,
            "layer_split": layer_split,
            "request_rate_rps": request_rate_rps,
            "arrival_seed": arrival_seed,
        },
        "generated_tokens": 10,
        "generated_tokens_per_second": gen_tps,
        "flop_bound_speedup": ideal,
        "peak_cuda_memory_allocated_bytes": 1024**3,
        "recurrent_steps_per_second": gen_tps * depth,
        "wall_time_s": 10 / gen_tps,
    }
    if kv_frac is not None:
        row["kv_cache"] = {"peak_blocks_used_fraction": kv_frac}
    backend_stats: dict[str, Any] = {}
    if depth_queue is not None:
        backend_stats["max_depth_queue_size"] = depth_queue
    if preemptions is not None:
        backend_stats["preemptions"] = preemptions
    if batch_stats is not None:
        backend_stats.update(batch_stats)
    if backend_stats:
        row["backend_stats"] = backend_stats
    if request_latency is not None:
        row["request_latency"] = request_latency
    if steady_state is not None:
        row["steady_state"] = steady_state
    if config_overrides is not None:
        row["config"].update(config_overrides)
    return row


def test_load_summaries_reads_a_jsonl_directory(tmp_path: Path) -> None:
    directory = tmp_path / "summaries"
    directory.mkdir()
    rows = [_summary(gen_tps=100.0), _summary(gen_tps=120.0)]
    (directory / "task_0.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    loaded = load_summaries(directory)

    assert len(loaded) == 2


def test_load_summaries_rejects_an_empty_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no summary rows"):
        load_summaries(tmp_path)


def test_repeats_of_one_configuration_average_together() -> None:
    aggregates = aggregate_summaries([_summary(gen_tps=100.0), _summary(gen_tps=120.0)])

    assert len(aggregates) == 1
    assert aggregates[0].repeats == 2
    assert aggregates[0].gen_tps_mean == 110.0
    assert aggregates[0].gen_tps_std == pytest.approx(14.142, abs=1e-3)


def test_bundles_sharing_a_dataset_name_stay_separate() -> None:
    # Two recordings of the same dataset (different seeds or sizes) share a workload name; only
    # the bundle id keeps their rows from averaging together as repeats of one configuration.
    rows = [
        _summary(workload_name="sharegpt", workload_id="sharegpt-d4-n56351-s0-sh0", gen_tps=100.0),
        _summary(workload_name="sharegpt", workload_id="sharegpt-d4-n10000-s1-sh1", gen_tps=200.0),
    ]

    aggregates = aggregate_summaries(rows)

    assert len(aggregates) == 2
    assert all(row.repeats == 1 for row in aggregates)
    # Rows recorded before the field existed group by name alone, as before.
    legacy = [_summary(workload_name="sharegpt", gen_tps=100.0), _summary(workload_name="sharegpt", gen_tps=200.0)]
    assert aggregate_summaries(legacy)[0].repeats == 2


def test_cb_baseline_is_labeled_full_even_when_a_threshold_is_recorded() -> None:
    # The sweep runs the full-depth cb baseline once and stamps it with the first swept threshold,
    # but cb ignores it and decodes at full depth. Labelling it by that threshold would leave the
    # comparison with no baseline to normalize against.
    aggregates = aggregate_summaries([_summary(backend="cb", exit_threshold=0.2, ideal=None)])

    assert aggregates[0].exit_label == "full"
    assert aggregates[0].is_baseline
    assert threshold_of(aggregates[0]) is None
    # A single run has no variance estimate, and cb rows carry no ideal-speedup ceiling; both
    # must read as absent rather than as measured zeros.
    assert aggregates[0].gen_tps_std is None
    assert aggregates[0].flop_bound_speedup is None


def test_refill_and_no_refill_stay_separate() -> None:
    # The early-exit baseline (cdb, refill off) and CDB (cdb, refill on) share a backend,
    # threshold, and workload; without refill in the key they would average into one row.
    rows = [
        _summary(backend="cdb", refill=False, exit_threshold=0.5, gen_tps=120.0),
        _summary(backend="cdb", refill=True, exit_threshold=0.5, gen_tps=180.0),
    ]

    aggregates = aggregate_summaries(rows)

    assert {row.refill: row.gen_tps_mean for row in aggregates} == {False: 120.0, True: 180.0}


def test_layer_split_overrides_stay_separate() -> None:
    # A layer-split override loads a different model, so its rows must not average with the
    # checkpoint's own split. Rows recorded before the field existed fold with the unset default.
    rows = [
        _summary(backend="cdb", exit_threshold=0.5, layer_split="1-4-1", gen_tps=200.0),
        _summary(backend="cdb", exit_threshold=0.5, gen_tps=120.0),
    ]
    legacy = _summary(backend="cdb", exit_threshold=0.5, gen_tps=140.0)
    del legacy["config"]["layer_split"]

    aggregates = aggregate_summaries([*rows, legacy])

    assert {row.layer_split: row.gen_tps_mean for row in aggregates} == {"1-4-1": 200.0, None: 130.0}


def test_distinct_thresholds_stay_separate() -> None:
    rows = [
        _summary(backend="cdb", exit_threshold=0.2),
        _summary(backend="cdb", exit_threshold=0.5),
    ]

    assert sorted(row.exit_label for row in aggregate_summaries(rows)) == ["q0.2", "q0.5"]
    assert sorted(threshold_of(row) for row in aggregate_summaries(rows)) == [0.2, 0.5]


def test_offered_rates_and_launch_widths_stay_separate() -> None:
    # Rows at different offered loads (or decode launch widths) are different operating points,
    # not repeats of one configuration: folding them would average a saturated run's latency into
    # an unloaded one's and report a throughput that describes neither.
    rows = [
        _summary(gen_tps=100.0),
        _summary(request_rate_rps=6.0, gen_tps=50.0),
        _summary(request_rate_rps=8.0, gen_tps=70.0),
        _summary(max_num_seqs=128, gen_tps=90.0),
    ]

    aggregates = aggregate_summaries(rows)

    assert len(aggregates) == 4
    assert all(row.repeats == 1 for row in aggregates)
    assert {row.request_rate_rps for row in aggregates} == {None, 6.0, 8.0}
    assert {row.max_num_seqs for row in aggregates} == {8, 128}
    assert {(row.request_rate_rps, row.max_num_seqs) for row in aggregates} == {
        (6.0, 8),
        (8.0, 8),
        (None, 128),
        (None, 8),
    }


def test_rows_differing_only_in_serving_policy_stay_separate() -> None:
    # Admission policy, KV layout, context limit, served model and attention kernel all change
    # what a row measures without changing its backend, threshold or workload. Folding them would
    # average, say, a "reserve" run into a "recompute" one and report a throughput for a policy
    # neither used.
    policies: list[dict[str, Any]] = [
        {},
        {"model": "other/model"},
        {"attn_implementation": "paged|flash_attention_2"},
        {"max_model_len": 4096},
        {"max_num_seqs": 64},
        {"kv_pressure_mode": "reserve"},
        {"safety_margin": 0.0},
        {"cdb_kv_policy": "last_exited"},
        {"min_exit_step": 3},
        {"exit_delay_steps": 1},
        {"min_recurrent_steps": 2},
        {"block_size": 32},
        {"cpu_offload_space": 8.0},
        {"delay_gate_consumption": False},
    ]
    rows = [_summary(config_overrides=overrides) for overrides in policies]

    aggregates = aggregate_summaries(rows)

    assert len(aggregates) == len(policies)
    assert all(row.repeats == 1 for row in aggregates)


def test_arrival_seeds_at_one_rate_fold_as_repeats() -> None:
    # Traces at the same offered rate are repeats: varying the seed across repeats folds
    # arrival-trace variance into the aggregate instead of splitting rows apart.
    rows = [
        _summary(request_rate_rps=6.0, arrival_seed=0, gen_tps=100.0),
        _summary(request_rate_rps=6.0, arrival_seed=1, gen_tps=120.0),
    ]

    aggregates = aggregate_summaries(rows)

    assert len(aggregates) == 1
    assert aggregates[0].repeats == 2
    assert aggregates[0].gen_tps_mean == 110.0


def test_latency_statistics_average_across_repeats_and_read_absent_when_missing() -> None:
    def _latency(norm_mean: float, norm_p99: float, ttft_mean: float) -> dict[str, Any]:
        return {
            "norm_e2e_s_per_token": {"mean": norm_mean, "p99": norm_p99},
            "ttft_s": {"mean": ttft_mean},
            # A run where every request emitted one token records no TPOT.
            "tpot_s": None,
        }

    rows = [
        _summary(request_rate_rps=6.0, request_latency=_latency(0.1, 0.3, 1.0)),
        _summary(request_rate_rps=6.0, request_latency=_latency(0.2, 0.5, 3.0)),
    ]

    aggregate = aggregate_summaries(rows)[0]

    assert aggregate.norm_lat_mean_s_per_tok == pytest.approx(0.15)
    assert aggregate.norm_lat_p99_s_per_tok == pytest.approx(0.4)
    assert aggregate.ttft_mean_s == pytest.approx(2.0)
    # Absent statistics read as unmeasured, never as zero: TPOT was null on every repeat and the
    # e2e series was never recorded.
    assert aggregate.tpot_mean_s is None
    assert aggregate.e2e_p99_s is None
    # Rows recorded before the latency block existed aggregate without one.
    assert aggregate_summaries([_summary()])[0].norm_lat_mean_s_per_tok is None


def test_missing_kv_measurement_is_skipped_rather_than_counted_as_zero() -> None:
    # An absent measurement must not drag peak occupancy toward under-utilization, which would
    # disguise a scheduling asymmetry between backends as a depth-mechanism difference.
    rows = [_summary(kv_frac=0.8), _summary(kv_frac=None)]

    assert aggregate_summaries(rows)[0].kv_peak_blocks_frac_mean == 0.8
    assert aggregate_summaries([_summary(kv_frac=None)])[0].kv_peak_blocks_frac_mean == 0.0


def test_preemptions_average_across_repeats_and_default_to_zero() -> None:
    # Repeated runs of a preempting mode average their preemption counts into the aggregate.
    rows = [_summary(backend="cb", preemptions=4), _summary(backend="cb", preemptions=6)]
    assert aggregate_summaries(rows)[0].preemptions_mean == 5.0
    # A run that never preempts (none/reserve, or a mode that just never filled the cache) reports zero.
    assert aggregate_summaries([_summary(backend="cb")])[0].preemptions_mean == 0.0


def test_depth_queue_is_zero_for_engines_without_backend_stats() -> None:
    # A no-refill engine reports no depth queue; a refilling one reports the tail it must drain.
    no_refill = aggregate_summaries([_summary(backend="cdb", refill=False, exit_threshold=0.2)])[0]
    refill = aggregate_summaries([_summary(backend="cdb", refill=True, exit_threshold=0.2, depth_queue=3060)])[0]

    assert no_refill.max_depth_queue_mean == 0.0
    assert refill.max_depth_queue_mean == 3060.0


def test_prefill_and_decode_batch_means_average_across_repeats() -> None:
    # The shared prefill/decode batch counters fold across repeats like every other backend stat.
    rows = [
        _summary(
            backend="cb",
            batch_stats={
                "prefill_batches": 2,
                "mean_prefill_batch_size": 4.0,
                "decode_batches": 8,
                "mean_decode_batch_size": 3.0,
            },
        ),
        _summary(
            backend="cb",
            batch_stats={
                "prefill_batches": 4,
                "mean_prefill_batch_size": 2.0,
                "decode_batches": 12,
                "mean_decode_batch_size": 5.0,
            },
        ),
    ]

    aggregate = aggregate_summaries(rows)[0]

    assert aggregate.prefill_batches_mean == 3.0
    assert aggregate.prefill_batch_size_mean == 3.0
    assert aggregate.decode_batches_mean == 10.0
    assert aggregate.decode_batch_size_mean == 4.0


def test_residency_reaches_the_aggregate_and_reads_absent_when_unmeasured() -> None:
    # Residency says whether admission held the batch near the cap the point was swept at, so it has
    # to survive aggregation; rows recorded before it was measured read absent rather than as zero.
    rows = [
        _summary(backend="cb", batch_stats={"mean_resident_requests": 6.0}),
        _summary(backend="cb", batch_stats={"mean_resident_requests": 8.0}),
    ]

    assert aggregate_summaries(rows)[0].resident_requests_mean == 7.0
    assert aggregate_summaries([_summary(backend="cb")])[0].resident_requests_mean is None


def test_steady_state_throughput_reaches_the_aggregate_and_reads_absent_when_unmeasured() -> None:
    def _steady(tps: float, rps: float, window_s: float, requests: int, resident: float) -> dict[str, Any]:
        return {
            "generated_tokens_per_second": tps,
            "requests_per_second": rps,
            "window_s": window_s,
            "completed_requests": requests,
            "mean_resident_requests": resident,
        }

    rows = [
        _summary(steady_state=_steady(90.0, 1.0, 5.0, 5, 7.5)),
        _summary(steady_state=_steady(110.0, 2.0, 7.0, 14, 8.0)),
    ]

    aggregate = aggregate_summaries(rows)[0]

    assert aggregate.steady_gen_tps_mean == pytest.approx(100.0)
    assert aggregate.steady_gen_tps_std == pytest.approx(statistics.stdev([90.0, 110.0]))
    assert aggregate.steady_rps_mean == pytest.approx(1.5)
    assert aggregate.steady_window_s_mean == pytest.approx(6.0)
    assert aggregate.steady_requests_mean == pytest.approx(9.5)
    assert aggregate.steady_resident_requests_mean == pytest.approx(7.75)
    # A single run has no spread, and a row without a window reads as unmeasured, never as zero.
    assert aggregate_summaries([_summary(steady_state=_steady(90.0, 1.0, 5.0, 5, 7.5))])[0].steady_gen_tps_std is None
    assert aggregate_summaries([_summary()])[0].steady_gen_tps_mean is None


def test_batch_means_default_to_zero_when_a_backend_omits_them() -> None:
    # A backend_stats payload that never carried the shared batch keys must aggregate without error,
    # reporting zero rather than raising on the missing key.
    aggregate = aggregate_summaries([_summary(backend="cdb", refill=True, exit_threshold=0.2, depth_queue=5)])[0]

    assert aggregate.prefill_batches_mean == 0.0
    assert aggregate.prefill_batch_size_mean == 0.0
    assert aggregate.decode_batches_mean == 0.0
    assert aggregate.decode_batch_size_mean == 0.0


def test_baseline_row_requires_exactly_one_baseline() -> None:
    with pytest.raises(ValueError, match="baseline"):
        baseline_row(aggregate_summaries([_summary(backend="cdb", exit_threshold=0.2)]))

    two_baselines = aggregate_summaries([_summary(backend="cb", workload_name="a"), _summary(backend="cb", depth=8)])
    with pytest.raises(ValueError, match="baseline"):
        baseline_row(two_baselines)
