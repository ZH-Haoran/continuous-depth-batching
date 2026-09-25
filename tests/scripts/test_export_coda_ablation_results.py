"""Coda sweeps must keep workload controls fixed and normalize each curve independently."""

from pathlib import Path

import pytest
from exporters.export_coda_ablation_results import build_results
from summary_rows import summary_row, write_summaries


def make_rows() -> list[dict]:
    rows = []
    for k, throughput in [(1, 100.0), (16, 120.0)]:
        row = summary_row(backend="cdb", refill=True, exit_threshold=0.28, gen_tps=throughput, min_coda_batch_size=k)
        row["backend_stats"] = {"recurrent_steps": 600, "recurrent_batches": 10, "coda_tokens": 60, "coda_batches": 6}
        rows.append(row)
    return rows


def export(tmp_path: Path, rows: list[dict]) -> dict:
    path = tmp_path / "summary.jsonl"
    write_summaries(path, rows)
    return build_results([path], tmp_path)


def test_normalizes_throughput_and_pools_launch_counts(tmp_path: Path) -> None:
    rows = make_rows()
    repeat = make_rows()[0]
    repeat["config"]["measured_repeat"] = 1
    repeat["backend_stats"].update(recurrent_steps=200, recurrent_batches=10, coda_tokens=60, coda_batches=3)
    result = export(tmp_path, [*rows, repeat])["panels"][0]
    assert result["speedup_over_immediate"] == [1.0, 1.2]
    assert result["recurrent_batch_size"] == [40.0, 60.0]
    assert result["coda_batch_size"] == pytest.approx([120 / 9, 10])
    assert result["gen_tps_std"][1] is None


@pytest.mark.parametrize("field,value", [("exit_threshold", 0.5), ("num_requests", 100), ("replay_eos_finishes", True)])
def test_rejects_mixed_controls(tmp_path: Path, field: str, value: object) -> None:
    rows = make_rows()
    rows[1]["config"][field] = value
    with pytest.raises(ValueError, match="Mixed configuration"):
        export(tmp_path, rows)


def test_requires_immediate_baseline(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Missing K=1"):
        export(tmp_path, make_rows()[1:])


def test_rejects_duplicate_run(tmp_path: Path) -> None:
    rows = make_rows()
    with pytest.raises(ValueError, match="Duplicate"):
        export(tmp_path, [*rows, rows[0]])


def test_rejects_changed_exit_schedule(tmp_path: Path) -> None:
    rows = make_rows()
    rows[1]["requested_exit_distribution"] = {"mean_depth": 3}
    with pytest.raises(ValueError, match="Mixed exit schedules"):
        export(tmp_path, rows)


def test_cb_normalization_matches_split_configuration(tmp_path: Path) -> None:
    rows = make_rows()
    baseline = make_rows()[0]
    baseline["config"]["backend"] = "cb"
    baseline["generated_tokens_per_second"] = 80
    baseline["config"]["exit_threshold"] = None
    wrong = make_rows()[0]
    wrong["config"].update(backend="cb", replay_eos_finishes=True)
    wrong["generated_tokens_per_second"] = 10
    result = export(tmp_path, [*rows, baseline, wrong])["panels"][0]
    assert result["cb_gen_tps"] == 80
    assert result["speedup_over_cb"] == [1.25, 1.5]


def test_explicit_calibration_input_selects_baseline_campaign(tmp_path: Path) -> None:
    rows = make_rows()
    summary_path = tmp_path / "summary.jsonl"
    write_summaries(summary_path, rows)
    baseline = make_rows()[0]
    baseline["config"].update(backend="cb", replay_eos_finishes=True)
    baseline["generated_tokens_per_second"] = 80
    baseline_path = tmp_path / "baseline.jsonl"
    write_summaries(baseline_path, [baseline])

    result = build_results(
        [summary_path],
        tmp_path,
        calibration_paths=[baseline_path],
    )["panels"][0]

    assert result["cb_gen_tps"] == 80
    assert result["cb_match"] == "calibration_input"
    assert result["speedup_over_cb"] == [1.25, 1.5]


def test_calibration_input_takes_precedence_over_exact_baseline(tmp_path: Path) -> None:
    rows = make_rows()
    summary_path = tmp_path / "summary.jsonl"
    write_summaries(summary_path, rows)
    exact = make_rows()[0]
    exact["config"]["backend"] = "cb"
    exact["generated_tokens_per_second"] = 50
    calibration = make_rows()[0]
    calibration["config"].update(backend="cb", replay_eos_finishes=True)
    calibration["generated_tokens_per_second"] = 80
    baseline_path = tmp_path / "baseline.jsonl"
    write_summaries(summary_path, [*rows, exact])
    write_summaries(baseline_path, [calibration])

    result = build_results(
        [summary_path],
        tmp_path,
        calibration_paths=[baseline_path],
    )["panels"][0]

    assert result["cb_gen_tps"] == 80
    assert result["cb_match"] == "calibration_input"


def test_explicit_calibration_rejects_non_cb_rows(tmp_path: Path) -> None:
    source = write_summaries(tmp_path / "sweep.jsonl", make_rows())
    with pytest.raises(ValueError, match="must contain only CB"):
        build_results([source], tmp_path, calibration_paths=[source])


def test_explicit_calibration_requires_matching_configuration(tmp_path: Path) -> None:
    source = write_summaries(tmp_path / "sweep.jsonl", make_rows())
    baseline = make_rows()[0]
    baseline["config"].update(backend="cb", max_num_seqs=999)
    calibration = write_summaries(tmp_path / "calibration.jsonl", [baseline])
    with pytest.raises(ValueError, match="Missing explicit calibration"):
        build_results([source], tmp_path, calibration_paths=[calibration])


def test_explicit_calibration_rejects_duplicate_repeats(tmp_path: Path) -> None:
    source = write_summaries(tmp_path / "sweep.jsonl", make_rows())
    baseline = make_rows()[0]
    baseline["config"]["backend"] = "cb"
    calibration = write_summaries(tmp_path / "calibration.jsonl", [baseline, baseline])
    with pytest.raises(ValueError, match="Duplicate CB baseline repeat"):
        build_results([source], tmp_path, calibration_paths=[calibration])
