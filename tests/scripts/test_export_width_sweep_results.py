"""Promotion of launch-width sweeps into the paper's committed results JSON."""

import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import pytest
from repo_paths import EXPORTERS_DIR
from summary_rows import recorded_bundle, summary_row, write_summaries

from looped_cdb.benchmarks.exports import CbStageProfile

EXPORT_SCRIPT = EXPORTERS_DIR / "export_width_sweep_results.py"
SPEC = spec_from_file_location("export_width_sweep_results", EXPORT_SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
export_width_sweep_results = module_from_spec(SPEC)
sys.modules[SPEC.name] = export_width_sweep_results
SPEC.loader.exec_module(export_width_sweep_results)

build_results = export_width_sweep_results.build_results
b_star_at = export_width_sweep_results.b_star_at


def _width_rows(
    widths: tuple[int, ...] = (16, 32, 64),
    *,
    cb: float = 100.0,
    norefill: float = 120.0,
    refill: float = 150.0,
    min_coda_batch_size: int = 1,
    min_coda_batch_sizes: dict[int, int] | None = None,
    workload_name: str = "alpaca",
    steady_scale: float | None = None,
) -> list[dict[str, Any]]:
    """A launch-width sweep: baseline and both depth policies at every width.

    ``steady_scale`` adds a steady-state window to every row, at that multiple of the whole-run rate.
    """

    rows = []
    for index, width in enumerate(widths):
        scale = 1.0 + index  # throughput grows with the launch width, as a real sweep does
        common = {"workload_name": workload_name, "max_num_seqs": width}

        def steady(tps: float) -> float | None:
            return None if steady_scale is None else tps * steady_scale

        rows.append(summary_row(gen_tps=cb * scale, steady_gen_tps=steady(cb * scale), **common))
        rows.append(
            summary_row(
                backend="cdb",
                refill=False,
                exit_threshold=0.2,
                bound=4 / 3,
                gen_tps=norefill * scale,
                steady_gen_tps=steady(norefill * scale * 1.1),
                **common,
            )
        )
        rows.append(
            summary_row(
                backend="cdb",
                refill=True,
                exit_threshold=0.2,
                bound=4 / 3,
                gen_tps=refill * scale,
                min_coda_batch_size=(min_coda_batch_sizes or {}).get(width, min_coda_batch_size),
                steady_gen_tps=steady(refill * scale),
                **common,
            )
        )
    return rows


def _panel(summary_paths: list[Path], repo_root: Path, **kwargs: Any) -> dict[str, Any]:
    return build_results(summary_paths, repo_root, **kwargs)["workloads"][0]


def test_prefill_adjusted_bound_tracks_the_cb_stage_mix_at_each_width(tmp_path: Path) -> None:
    widths = (16, 32, 64)
    rows = _width_rows(widths)
    for row in rows:
        row["generated_tokens"] = 8
        row["first_token_full_depth_count"] = 4
    summary = write_summaries(tmp_path / "widths.jsonl", rows)
    profiles = {}
    for width, prefill in zip(widths, (20.0, 50.0, 80.0), strict=True):
        profile = CbStageProfile(
            source=f"outputs/nsys/alpaca_w{width}/cb_full_analysis.json",
            summary_source=f"outputs/nsys/alpaca_w{width}/cb_full_summary.jsonl",
            model="test-model",
            workload="alpaca",
            width=width,
            num_requests=4,
            prefill_gpu_s=prefill,
            decode_gpu_s=100.0 - prefill,
        )
        profiles[(profile.model, profile.workload, profile.width)] = profile

    panel = _panel([summary], tmp_path, cb_profiles=profiles)

    assert [timing["prefill_fraction"] for timing in panel["bound_timings"]] == [0.2, 0.5, 0.8]
    assert panel["e2e_bound"] == pytest.approx([5 / 3, 4 / 3, 10 / 9], abs=1e-6)


def test_each_width_is_normalized_against_its_own_baseline(tmp_path: Path) -> None:
    # The baseline itself moves with the launch width, so a panel-wide baseline would report the
    # baseline's own scaling as a CDB speed-up.
    summary = write_summaries(tmp_path / "widths.jsonl", _width_rows())

    panel = _panel([summary], tmp_path)

    assert panel["widths"] == [16, 32, 64]
    assert panel["cb"] == [100.0, 200.0, 300.0]
    assert panel["refill"] == [150.0, 300.0, 450.0]
    assert panel["refill_over_cb"] == [1.5, 1.5, 1.5]
    assert panel["norefill_over_cb"] == [1.2, 1.2, 1.2]
    assert panel["threshold"] == 0.2
    assert panel["flop_bound"] == pytest.approx(4 / 3, abs=1e-6)
    assert "e2e_bound" not in panel
    assert panel["min_coda_batch_sizes"] == [1, 1, 1]


def test_width_dependent_minimum_coda_batch_is_recorded(tmp_path: Path) -> None:
    summary = write_summaries(
        tmp_path / "widths.jsonl",
        _width_rows(min_coda_batch_sizes={16: 4, 32: 8, 64: 16}),
    )

    panel = _panel([summary], tmp_path)

    assert panel["min_coda_batch_sizes"] == [4, 8, 16]


def test_steady_state_throughput_is_exported_next_to_the_whole_run(tmp_path: Path) -> None:
    # The steady series carries its own ratios, so a figure can switch to it without re-deriving
    # anything; rows without a window leave the series unmeasured rather than zero.
    panel = _panel([write_summaries(tmp_path / "steady.jsonl", _width_rows(steady_scale=1.2))], tmp_path)

    assert panel["cb_steady"] == [120.0, 240.0, 360.0]
    assert panel["refill_steady_over_cb"] == [1.5, 1.5, 1.5]
    assert panel["norefill_steady_over_cb"] == pytest.approx([1.32, 1.32, 1.32])

    plain = _panel([write_summaries(tmp_path / "plain.jsonl", _width_rows())], tmp_path)

    assert plain["cb_steady"] == [None, None, None]
    assert plain["refill_steady_over_cb"] == [None, None, None]


def test_a_series_with_a_gap_is_rejected(tmp_path: Path) -> None:
    # A curve that stops short would be read as a crossover that the sweep never measured.
    dropped = {"backend": "cdb", "refill": True, "max_num_seqs": 64}
    rows = [row for row in _width_rows() if not all(row["config"][key] == value for key, value in dropped.items())]
    summary = write_summaries(tmp_path / "widths.jsonl", rows)

    with pytest.raises(ValueError, match=r"refill series was not measured at width\(s\) \[64\]"):
        build_results([summary], tmp_path)


def test_a_sweep_over_the_minimum_coda_batch_is_rejected(tmp_path: Path) -> None:
    # Two coda policies at one width is the minimum-coda-batch ablation, which has its own export.
    summary = write_summaries(tmp_path / "widths.jsonl", [*_width_rows(), *_width_rows(min_coda_batch_size=32)])

    with pytest.raises(ValueError, match="minimum coda batches"):
        build_results([summary], tmp_path)


def test_the_threshold_is_held_fixed(tmp_path: Path) -> None:
    rows = [
        *_width_rows(),
        summary_row(backend="cdb", refill=True, exit_threshold=0.5, max_num_seqs=16),
    ]
    summary = write_summaries(tmp_path / "widths.jsonl", rows)

    with pytest.raises(ValueError, match="holds the exit threshold fixed"):
        build_results([summary], tmp_path)


def test_a_panel_is_assembled_from_every_summary_that_holds_its_rows(tmp_path: Path) -> None:
    # The baseline and the depth policies are separate runs, and a width range re-measured later is
    # a separate file; all of them make up one panel.
    rows = _width_rows()
    baseline = write_summaries(tmp_path / "cb.jsonl", [row for row in rows if row["config"]["backend"] == "cb"])
    depth = write_summaries(tmp_path / "cdb.jsonl", [row for row in rows if row["config"]["backend"] == "cdb"])

    panel = _panel([baseline, depth], tmp_path)

    assert panel["widths"] == [16, 32, 64]
    assert panel["source_summaries"] == [str(baseline), str(depth)]


def test_workloads_are_kept_apart_in_one_export(tmp_path: Path) -> None:
    # Two workloads of one model are exported together, as two panels; folding their rows into one
    # would average sweeps that replayed different requests.
    rows = [*_width_rows(), *_width_rows(workload_name="sharegpt", cb=200.0, refill=260.0)]
    summary = write_summaries(tmp_path / "widths.jsonl", rows)

    results = build_results([summary], tmp_path)

    assert [workload["name"] for workload in results["workloads"]] == ["alpaca", "sharegpt"]
    assert results["workloads"][1]["cb"] == [200.0, 400.0, 600.0]
    assert results["workloads"][1]["refill_over_cb"] == [1.3, 1.3, 1.3]
    assert json.dumps(results)  # the export must be serializable as written


def _decode_latency_export(path: Path, *, model_id: str = "test-model", max_depth: int = 4) -> Path:
    """A decode-step latency export with B* fitted at a geometric sweep of context lengths."""

    path.write_text(
        json.dumps(
            {
                "model_id": model_id,
                "max_depth": max_depth,
                "floor_ms": 5.0,
                "contexts": [
                    {"context_length": 128, "b_star": 136.9, "batch_range": [1, 256]},
                    {"context_length": 512, "b_star": 70.4, "batch_range": [1, 256]},
                    {"context_length": 2048, "b_star": 22.8, "batch_range": [1, 256]},
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _b_star_at(source: Path, mean_sequence_length: float, **kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("models", {"test-model"})
    kwargs.setdefault("max_depth", 4)
    return b_star_at(source, mean_sequence_length, **kwargs)


def test_b_star_is_taken_from_the_closest_measured_context(tmp_path: Path) -> None:
    # B* belongs to the decode-latency experiment, and its context lengths are geometric, so the
    # closest one is chosen on a log scale: 580 tokens sits nearer 512 than 2048.
    source = _decode_latency_export(tmp_path / "decode_step_latency_test.json")

    assert _b_star_at(source, 580.0)["b_star_approx"] == 70
    assert _b_star_at(source, 78.0)["b_star_approx"] == 137
    assert _b_star_at(source, 4000.0)["b_star_approx"] == 23


def test_b_star_records_the_fit_it_came_from(tmp_path: Path) -> None:
    # A marker whose origin is not recorded cannot be checked against the experiment that fitted it.
    source = _decode_latency_export(tmp_path / "decode_step_latency_test.json")

    basis = _b_star_at(source, 580.0)["b_star_basis"]

    assert "L = 512" in basis
    assert source.name in basis


def test_b_star_from_another_model_is_rejected(tmp_path: Path) -> None:
    # Nothing about a width panel contradicts a B* marker, so a fit from the wrong checkpoint or
    # the wrong depth would be published as if it belonged there.
    source = _decode_latency_export(tmp_path / "decode_step_latency_other.json", model_id="other-model")

    with pytest.raises(ValueError, match="different model"):
        _b_star_at(source, 580.0)

    same_name = _decode_latency_export(tmp_path / "decode_step_latency_deep.json", max_depth=16)
    with pytest.raises(ValueError, match="different recurrent-depth budget"):
        _b_star_at(same_name, 580.0)


def test_the_same_checkpoint_under_two_names_is_accepted(tmp_path: Path) -> None:
    # The two sweeps name the model independently: one run records the Hub id, another the local
    # snapshot directory it was loaded from, which mangles the id rather than suffixing it. Those
    # are the same checkpoint and must not block the export.
    source = _decode_latency_export(tmp_path / "decode.json", model_id="org/test-model")

    assert _b_star_at(source, 580.0, models={"local_org_test-model"})["b_star_approx"] == 70


def test_an_extrapolated_b_star_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # A knee past the largest batch measured comes from the fit, not from a measurement, and the
    # reader of the width panel never sees the sweep that says so.
    source = tmp_path / "decode_step_latency_narrow.json"
    source.write_text(
        json.dumps(
            {
                "model_id": "test-model",
                "max_depth": 4,
                "floor_ms": 5.0,
                "contexts": [{"context_length": 512, "b_star": 70.4, "batch_range": [1, 32]}],
            }
        ),
        encoding="utf-8",
    )

    _b_star_at(source, 580.0)

    assert "extrapolated" in capsys.readouterr().err


def test_a_panel_marks_b_star_for_its_mean_sequence_length(tmp_path: Path) -> None:
    recorded_bundle(tmp_path / "outputs" / "workloads")  # 3-token prompts, 2-token outputs
    source = _decode_latency_export(tmp_path / "decode_step_latency_test.json")
    summary = write_summaries(tmp_path / "widths.jsonl", _width_rows())

    panel = _panel([summary], tmp_path, b_star_source=source)

    assert panel["mean_sequence_length"] == 5
    assert panel["b_star_approx"] == 137  # the shortest fitted context is the closest
    assert panel["name"] == "alpaca"


def test_b_star_cannot_be_placed_without_the_bundle(tmp_path: Path) -> None:
    # The context length that selects the fit comes from the bundle; guessing one would publish a
    # marker at a length nobody measured.
    source = _decode_latency_export(tmp_path / "decode_step_latency_test.json")
    summary = write_summaries(tmp_path / "widths.jsonl", _width_rows())

    with pytest.raises(ValueError, match="mean sequence length"):
        build_results([summary], tmp_path, b_star_source=source)
