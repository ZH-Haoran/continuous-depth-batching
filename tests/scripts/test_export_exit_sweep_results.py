"""Promotion of exit-threshold sweeps into the paper's committed results JSON."""

import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
from repo_paths import EXPORTERS_DIR
from summary_rows import bundle, recorded_bundle, summary_row, write_summaries

from looped_cdb.benchmarks.exports import CbStageProfile

EXPORT_SCRIPT = EXPORTERS_DIR / "export_exit_sweep_results.py"
SPEC = spec_from_file_location("export_exit_sweep_results", EXPORT_SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
export_exit_sweep_results = module_from_spec(SPEC)
sys.modules[SPEC.name] = export_exit_sweep_results
SPEC.loader.exec_module(export_exit_sweep_results)

build_results = export_exit_sweep_results.build_results
build_workload_entry = export_exit_sweep_results.build_workload_entry


def test_prefill_adjustment_is_exported_for_the_dense_curve_and_measured_points(tmp_path: Path) -> None:
    recorded_bundle(tmp_path / "outputs" / "workloads")
    rows = [
        summary_row(max_num_seqs=64),
        summary_row(backend="cdb", refill=True, exit_threshold=0.2, bound=4 / 3, max_num_seqs=64),
    ]
    for row in rows:
        row["generated_tokens"] = 8
        row["first_token_full_depth_count"] = 4
    summary = write_summaries(tmp_path / "alpaca.jsonl", rows)
    profile = CbStageProfile(
        source="outputs/nsys/test/cb_full_analysis.json",
        summary_source="outputs/nsys/test/cb_full_summary.jsonl",
        model="test-model",
        workload="alpaca",
        width=64,
        num_requests=4,
        prefill_gpu_s=20.0,
        decode_gpu_s=80.0,
    )

    entry = build_workload_entry(
        summary,
        tmp_path,
        cb_profiles={(profile.model, profile.workload, profile.width): profile},
    )

    assert entry["bound_timing"]["prefill_fraction"] == 0.2
    assert entry["bound_curve"]["e2e_bound"][0] == pytest.approx(5 / 3)
    adaptive = next(point for point in entry["points"] if point["backend"] == "cdb")
    assert adaptive["e2e_bound"] == pytest.approx(5 / 3)


def test_bound_curve_densely_spans_the_measured_thresholds(tmp_path: Path) -> None:
    recorded_bundle(tmp_path / "outputs" / "workloads")
    summary = write_summaries(
        tmp_path / "alpaca.jsonl",
        [
            summary_row(),
            summary_row(backend="cdb", refill=True, exit_threshold=0.2, bound=4 / 3),
            summary_row(backend="cdb", refill=True, exit_threshold=0.5, bound=8 / 7),
        ],
    )

    curve = build_workload_entry(summary, tmp_path)["bound_curve"]

    assert "e2e_bound" not in curve
    assert curve["thresholds"][0] == 0.2
    assert curve["thresholds"][-1] == 0.5
    assert len(curve["thresholds"]) == 31  # 0.01-spaced grid, measured endpoints included
    by_threshold = dict(zip(curve["thresholds"], curve["flop_bound"], strict=True))
    assert by_threshold[0.2] == pytest.approx(4 / 3)
    assert by_threshold[0.5] == pytest.approx(8 / 7)
    # A higher threshold can only delay exits, so the curve never rises.
    assert curve["flop_bound"] == sorted(curve["flop_bound"], reverse=True)


def test_bound_curve_contradicting_the_recorded_bound_is_rejected(tmp_path: Path) -> None:
    # The curve must describe the same schedule the runs replayed. A recorded bound the bundle
    # cannot reproduce means the bundle, the exit flags or the layer split changed since the
    # measurement.
    recorded_bundle(tmp_path / "outputs" / "workloads")
    summary = write_summaries(
        tmp_path / "alpaca.jsonl",
        [summary_row(), summary_row(backend="cdb", refill=True, exit_threshold=0.2, bound=1.5)],
    )

    with pytest.raises(ValueError, match="bound curve disagrees"):
        build_workload_entry(summary, tmp_path)


def test_bound_curve_is_omitted_when_the_bundle_is_absent(tmp_path: Path) -> None:
    summary = write_summaries(
        tmp_path / "alpaca.jsonl",
        [summary_row(), summary_row(backend="cdb", refill=True, exit_threshold=0.2, bound=4 / 3)],
    )

    assert "bound_curve" not in build_workload_entry(summary, tmp_path)


def test_bound_curve_prices_in_the_boundary_stages(tmp_path: Path) -> None:
    # Boundary stages that cost one core application each pull the bound well below the depth
    # ratio: at mean depth 3 of 4 the ratio is 4/3, but the bound is only (2 + 4) / (2 + 3).
    recorded_bundle(tmp_path / "outputs" / "workloads")
    heavy = {"f0_params": 200, "fr_params": 100}
    summary = write_summaries(
        tmp_path / "alpaca.jsonl",
        [
            summary_row(stage_flops=heavy),
            summary_row(backend="cdb", refill=True, exit_threshold=0.2, bound=6 / 5, stage_flops=heavy),
        ],
    )

    entry = build_workload_entry(summary, tmp_path)

    assert entry["stage_flops"] == heavy
    by_threshold = dict(zip(entry["bound_curve"]["thresholds"], entry["bound_curve"]["flop_bound"], strict=True))
    assert by_threshold[0.2] == pytest.approx(6 / 5)


def test_rows_served_with_different_layer_splits_are_rejected(tmp_path: Path) -> None:
    # The layer split sets F_0, so two splits are two different bounds. Averaging their rows
    # into one panel would compare each against a ceiling only one of them was measured under.
    summary = write_summaries(
        tmp_path / "alpaca.jsonl",
        [
            summary_row(stage_flops={"f0_params": 200, "fr_params": 100}),
            summary_row(backend="cdb", refill=True, exit_threshold=0.2, stage_flops={"f0_params": 0, "fr_params": 100}),
        ],
    )

    with pytest.raises(ValueError, match="per-stage FLOP weights"):
        build_workload_entry(summary, tmp_path)


def test_speedups_are_normalized_to_the_full_depth_cb_baseline(tmp_path: Path) -> None:
    summary = write_summaries(
        tmp_path / "alpaca.jsonl",
        [
            summary_row(gen_tps=100.0),
            summary_row(backend="cdb", refill=False, exit_threshold=0.2, gen_tps=150.0),
            summary_row(backend="cdb", refill=True, exit_threshold=0.2, gen_tps=90.0),
        ],
    )

    entry = build_workload_entry(summary, tmp_path)

    speedups = {(p["backend"], p["refill"]): p["speedup"] for p in entry["points"]}
    assert speedups[("cb", True)] == 1.0
    assert speedups[("cdb", False)] == 1.5
    assert speedups[("cdb", True)] == 0.9
    assert entry["baseline_gen_tps"] == 100.0
    # Rows without a steady-state window leave the steady speed-up unmeasured rather than zero.
    assert {p["steady_speedup"] for p in entry["points"]} == {None}


def test_steady_state_speedups_are_normalized_to_the_baseline_steady_window(tmp_path: Path) -> None:
    summary = write_summaries(
        tmp_path / "alpaca.jsonl",
        [
            summary_row(gen_tps=100.0, steady_gen_tps=120.0),
            summary_row(backend="cdb", refill=False, exit_threshold=0.2, gen_tps=150.0, steady_gen_tps=240.0),
            summary_row(backend="cdb", refill=True, exit_threshold=0.2, gen_tps=90.0, steady_gen_tps=60.0),
        ],
    )

    entry = build_workload_entry(summary, tmp_path)

    steady = {(p["backend"], p["refill"]): p["steady_speedup"] for p in entry["points"]}
    assert steady[("cb", True)] == 1.0
    assert steady[("cdb", False)] == 2.0
    assert steady[("cdb", True)] == 0.5


def test_points_are_ordered_baseline_then_no_refill_then_refill(tmp_path: Path) -> None:
    summary = write_summaries(
        tmp_path / "alpaca.jsonl",
        [
            summary_row(backend="cdb", refill=True, exit_threshold=0.5, gen_tps=80.0),
            summary_row(backend="cdb", refill=False, exit_threshold=0.5, gen_tps=140.0),
            summary_row(gen_tps=100.0),
            summary_row(backend="cdb", refill=False, exit_threshold=0.2, gen_tps=150.0),
        ],
    )

    points = build_workload_entry(summary, tmp_path)["points"]

    assert [(p["backend"], p["refill"], p["threshold"]) for p in points] == [
        ("cb", True, None),
        ("cdb", False, 0.2),
        ("cdb", False, 0.5),
        ("cdb", True, 0.5),
    ]


def test_mixing_request_counts_in_one_panel_is_rejected(tmp_path: Path) -> None:
    # A subsampled run and a full run are not comparable: the fixed end-of-run drain is a much
    # larger fraction of a short run, so averaging them into one panel would fabricate a number.
    summary = write_summaries(
        tmp_path / "alpaca.jsonl",
        [
            summary_row(num_requests=4),
            summary_row(backend="cdb", refill=False, exit_threshold=0.2, num_requests=400),
        ],
    )

    with pytest.raises(ValueError, match="mixes request counts"):
        build_workload_entry(summary, tmp_path)


def test_missing_baseline_is_rejected(tmp_path: Path) -> None:
    summary = write_summaries(tmp_path / "alpaca.jsonl", [summary_row(backend="cdb", refill=False, exit_threshold=0.2)])

    with pytest.raises(ValueError, match="baseline"):
        build_workload_entry(summary, tmp_path)


def test_subsample_is_flagged_against_the_bundle_size(tmp_path: Path) -> None:
    bundle(tmp_path / "outputs" / "workloads", num_requests=10)
    summary = write_summaries(tmp_path / "alpaca.jsonl", [summary_row(num_requests=4)])

    entry = build_workload_entry(summary, tmp_path)

    assert entry["full_num_requests"] == 10
    assert entry["is_subsample"] is True
    assert entry["num_requests"] == 4


def test_full_run_is_not_flagged_as_subsample(tmp_path: Path) -> None:
    bundle(tmp_path / "outputs" / "workloads", num_requests=4)
    summary = write_summaries(tmp_path / "alpaca.jsonl", [summary_row(num_requests=4)])

    assert build_workload_entry(summary, tmp_path)["is_subsample"] is False


def test_each_summary_becomes_its_own_workload_panel(tmp_path: Path) -> None:
    alpaca = write_summaries(tmp_path / "alpaca.jsonl", [summary_row()])
    sharegpt = write_summaries(tmp_path / "sharegpt.jsonl", [summary_row(workload_name="sharegpt")])

    results = build_results([alpaca, sharegpt], tmp_path)

    assert [workload["name"] for workload in results["workloads"]] == ["alpaca", "sharegpt"]
    assert json.dumps(results)  # the export must be serializable as written
