"""Export of validated scheduling-control idle fractions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from exporters.export_scheduling_overhead_results import load_idle_percent, render_macros


def _analysis(path: Path, *, label: str = "sync", fraction: float = 0.1009936) -> Path:
    payload = {
        "label": label,
        "measured_label": "benchmark.steady",
        "measured_range_count": 1,
        "diagnostics": {"warnings": []},
        "measured": {"100us": {"gpu_idle_fraction": fraction}},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_idle_percent_validates_and_converts_the_fraction(tmp_path: Path) -> None:
    assert load_idle_percent(_analysis(tmp_path / "sync.json"), expected_label="sync") == pytest.approx(10.09936)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("measured_label", "benchmark.generate", "benchmark.steady"),
        ("measured_range_count", 2, "exactly one"),
        ("diagnostics", {"warnings": ["incomplete"]}, "warnings"),
    ],
)
def test_invalid_analysis_is_rejected(tmp_path: Path, field: str, value: object, message: str) -> None:
    path = _analysis(tmp_path / "sync.json")
    payload = json.loads(path.read_text())
    payload[field] = value
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=message):
        load_idle_percent(path, expected_label="sync")


def test_render_macros_uses_three_decimal_values_and_two_decimal_labels() -> None:
    rendered = render_macros({"sync": 10.09936, "async_immediate": 9.98446, "lookahead": 0.55802})
    assert r"\def\idleSync{10.099}" in rendered
    assert r"\def\idleAsyncImmediateLabel{9.98\%}" in rendered
    assert r"\def\idleLookaheadLabel{0.56\%}" in rendered
