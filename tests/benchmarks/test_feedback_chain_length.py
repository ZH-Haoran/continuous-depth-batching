"""The length diagnostic preserves the short prefix and never solves an oracle."""

import json

import diagnose_feedback_chain_length as diagnostic
import pytest

from looped_cdb.benchmarks.feedback import CostTable, RecurrenceTrace
from looped_cdb.benchmarks.feedback_report import load_trace, write_json


def test_extension_preserves_every_original_demand_and_is_reproducible():
    trace = RecurrenceTrace(((1, 3, 4, 2), (3, 2, 7, 1)), metadata={"seed": 0})
    extended = diagnostic.extend_chains(trace)
    assert extended == diagnostic.extend_chains(trace)
    assert all(long[:4] == short and len(long) == 64 for short, long in zip(trace.chains, extended.chains))
    assert extended.metadata["extension_metadata"]["seed"] == 1
    assert extended.metadata["extension_metadata"]["max_depth"] == 32
    assert extended.metadata["base_metadata"] == trace.metadata


def test_diagnostic_exports_both_static_scans_without_oracle(tmp_path, monkeypatch, capsys):
    from looped_cdb.benchmarks import feedback_report

    def forbidden(*args, **kwargs):
        pytest.fail("the length diagnostic called an oracle")

    monkeypatch.setattr(feedback_report, "solve_dp_oracle", forbidden)
    monkeypatch.setattr(feedback_report, "solve_oracle", forbidden)
    costs = CostTable(16, *((0, *[1] * 16) for _ in range(3)))
    costs_file = tmp_path / "costs.json"
    trace_file = tmp_path / "trace.json"
    output_dir = tmp_path / "diagnostic"
    write_json(costs_file, costs.to_dict())
    write_json(trace_file, {"format": "looped-cdb-recurrence-trace-v1", "chains": [[1, 2, 3, 4], [2, 3, 4, 1]]})
    monkeypatch.setattr(
        "sys.argv",
        ["diagnostic", "--costs", str(costs_file), "--trace", str(trace_file), "--output-dir", str(output_dir)],
    )
    diagnostic.main()
    comparisons = json.loads((output_dir / "comparison.json").read_text())
    assert [row["tokens_per_sequence"] for row in comparisons] == [4, 64]
    assert all(row["best_static"]["all_thresholds_scanned"] for row in comparisons)
    assert all(row["k16_comparison"]["runtime_ratio_to_best"] >= 1 for row in comparisons)
    base = load_trace(trace_file)
    assert load_trace(output_dir / "short" / "trace.json") == base
    long = load_trace(output_dir / "long" / "trace.json")
    assert all(chain[:4] == original for chain, original in zip(long.chains, base.chains))
    for case in ("short", "long"):
        report = json.loads((output_dir / case / "report.json").read_text())
        assert report["oracle"] is report["gap"] is None
        assert (output_dir / case / "static.csv").exists()
        assert not (output_dir / case / "oracle_schedule.csv").exists()
    captured = capsys.readouterr()
    assert "runtime_ratio_to_best" in captured.out
    assert "[static] scanned 8/16" in captured.err
    assert "[dp]" not in captured.err


@pytest.mark.parametrize("chains", [((1,), (1, 2)), ((1,) * 64,)])
def test_unsuitable_base_chain_lengths_are_rejected(chains):
    with pytest.raises(ValueError):
        diagnostic.extend_chains(RecurrenceTrace(chains))
