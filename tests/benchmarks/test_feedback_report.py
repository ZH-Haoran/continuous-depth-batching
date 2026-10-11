"""End-to-end exports, certified ratios, and finite feedback schedule replay."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from looped_cdb.benchmarks.feedback import CostTable, evaluate_actions, generate_trace
from looped_cdb.benchmarks.feedback_report import load_trace, run_experiment

FIXTURES = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).resolve().parents[2]


def fixture():
    return CostTable.from_dict(json.loads((FIXTURES / "feedback_costs.json").read_text())), load_trace(
        FIXTURES / "feedback_trace.json"
    )


def test_exported_oracle_schedule_replays_to_exact_runtime(tmp_path):
    costs, trace = fixture()
    report = run_experiment(costs, trace, tmp_path, thresholds=(1, 2, 3, 4), plots=False)
    assert report["oracle"]["exact"]
    assert report["oracle"]["solver"] == "dp"
    assert report["oracle"]["runtime_ms"] == 30
    assert report["oracle"]["states_evaluated"] > 0
    assert "max_states" not in report["oracle"]
    assert report["best_static"]["runtime_ms"] == 32
    assert report["gap"]["lower"] == report["gap"]["upper"] == pytest.approx(32 / 30)
    assert report["gap"]["numerator"] == "best_static"
    with (tmp_path / "oracle_schedule.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    replay = evaluate_actions(trace, costs, tuple(row["action"] for row in rows[1:]))
    assert replay.runtime_ms == float(rows[-1]["end_ms"]) == 30
    assert load_trace(tmp_path / "trace.json") == trace
    assert json.loads((tmp_path / "report.json").read_text())["cost_metadata"] == costs.metadata
    assert len(list(tmp_path.glob("static_k_*_schedule.csv"))) == 4


def test_budget_stop_reports_bounds_without_claiming_exact_opt(tmp_path):
    costs, trace = fixture()
    report = run_experiment(costs, trace, tmp_path, thresholds=(1, 4), oracle="astar", max_states=0, plots=False)
    assert not report["oracle"]["exact"]
    assert report["gap"]["numerator"] == "best_scanned_static"
    assert report["gap"]["lower"] <= 32 / 30 <= report["gap"]["upper"]
    assert "exact OPT" not in (tmp_path / "summary.md").read_text()


@pytest.mark.parametrize("solver", [None, "dp", "astar"])
def test_cli_runs_without_importing_torch_or_plotting(tmp_path, solver):
    script = ROOT / "scripts/simulate_feedback_batching.py"
    # A fresh stdlib-only interpreter with explicit src path cannot accidentally
    # depend on the pytest process's already-imported torch or matplotlib.
    code = (
        f"import sys,runpy; sys.path.insert(0,{str(ROOT / 'src')!r}); "
        "sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')"
    )
    run = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            code,
            str(script),
            "--costs",
            str(FIXTURES / "feedback_costs.json"),
            "--trace",
            str(FIXTURES / "feedback_trace.json"),
            "--output-dir",
            str(tmp_path),
            "--all-k",
            "--no-plots",
            *([] if solver is None else ["--oracle", solver]),
        ],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["oracle"]["upper_bound_ms"] == 30
    assert report["oracle"]["solver"] == (solver or "dp")


def test_zero_cost_gap_is_undefined_not_nonfinite_json(tmp_path):
    _, trace = fixture()
    costs = CostTable(4, *((0,) * 5 for _ in range(3)))
    report = run_experiment(costs, trace, tmp_path, thresholds=(1, 2, 3, 4), plots=False)
    assert report["gap"]["lower"] is report["gap"]["upper"] is None
    assert "undefined" in (tmp_path / "summary.md").read_text()


def test_plot_artifacts(tmp_path):
    pytest.importorskip("matplotlib")
    costs, trace = fixture()
    run_experiment(costs, trace, tmp_path, thresholds=(1, 2, 3, 4))
    for name in ("stage_costs", "static_vs_oracle"):
        assert (tmp_path / f"{name}.png").stat().st_size > 1000
        assert "<svg" in (tmp_path / f"{name}.svg").read_text()


def test_serving_benchmark_public_exports_survive_lazy_loading():
    from looped_cdb.benchmarks import BenchmarkConfig, SampledRequest, Workload
    from looped_cdb.benchmarks.datasets import SampledRequest as DatasetRequest
    from looped_cdb.benchmarks.metrics import BenchmarkConfig as MetricsConfig
    from looped_cdb.benchmarks.workload import Workload as ServingWorkload

    assert (BenchmarkConfig, SampledRequest, Workload) == (MetricsConfig, DatasetRequest, ServingWorkload)


def test_trace_metadata_roundtrip_and_legacy_trace_is_not_clipped(tmp_path):
    costs, _ = fixture()
    trace = generate_trace(sequences=4, tokens_per_sequence=2, max_depth=2)
    report = run_experiment(costs, trace, tmp_path, thresholds=(1, 2, 3, 4), plots=False)
    assert report["trace_metadata"] == trace.metadata
    assert report["trace_metadata"]["max_depth"] == 2
    assert report["model"]["observed_max_depth"] <= 2
    assert load_trace(tmp_path / "trace.json") == trace
    # Previously saved chains retain their demands even if above the new default.
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"format": "looped-cdb-recurrence-trace-v1", "chains": [[40]]}))
    assert load_trace(legacy).chains == ((40,),)
    assert load_trace(legacy).metadata == {}


def test_unknown_oracle_is_rejected_before_writing_report(tmp_path):
    costs, trace = fixture()
    with pytest.raises(ValueError, match="oracle must be"):
        run_experiment(costs, trace, tmp_path, thresholds=(1, 4), oracle="invalid", plots=False)
    assert not (tmp_path / "report.json").exists()
