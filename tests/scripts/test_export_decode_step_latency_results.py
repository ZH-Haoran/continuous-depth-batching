"""Deriving the decode-step latency figure data from a benchmark sweep."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from exporters.export_decode_step_latency_results import (
    build_results,
    context_series,
    latest_points,
    load_rows,
    run_meta,
)


def _meta(model: str = "org/model", device: str = "NVIDIA H100 PCIe") -> dict[str, Any]:
    return {
        "run_config": {"model": model},
        "device_name": device,
        "max_depth": 4,
    }


def _point(batch: int, context: int, core_ms: float | None, status: str = "ok") -> dict[str, Any]:
    return {"batch_size": batch, "context_length": context, "core_step_ms": core_ms, "status": status}


def _row(batch: int, context: int, core_ms: float | None, status: str = "ok", **meta: Any) -> dict[str, Any]:
    return {"meta": _meta(**meta), **_point(batch, context, core_ms, status)}


def _linear_points(context: int, floor: float, per_seq: float, batches: list[int]) -> list[dict[str, Any]]:
    return [_point(b, context, floor + per_seq * b) for b in batches]


def _linear_rows(context: int, floor: float, per_seq: float, batches: list[int], **meta: Any) -> list[dict[str, Any]]:
    return [_row(b, context, floor + per_seq * b, **meta) for b in batches]


def test_fit_recovers_floor_slope_and_b_star_from_linear_data() -> None:
    # t(B) = 5.0 + 0.04 * B, so the roofline knee sits at B* = 125.
    series = context_series(_linear_points(128, floor=5.0, per_seq=0.04, batches=[1, 4, 16, 64, 256]))

    assert len(series) == 1
    fit = series[0]
    assert fit["context_length"] == 128
    assert fit["floor_ms"] == pytest.approx(5.0)
    assert fit["per_seq_ms"] == pytest.approx(0.04)
    assert fit["b_star"] == pytest.approx(125.0)
    assert [p["batch_size"] for p in fit["points"]] == [1, 4, 16, 64, 256]
    # The fit's anchoring is recorded so truncated rows (an early OOM) stay visible.
    assert fit["num_points"] == 5
    assert fit["batch_range"] == [1, 256]
    assert fit["fit_r2"] == pytest.approx(1.0)


def test_contexts_are_fit_separately_and_sorted() -> None:
    primitives = _linear_points(2048, floor=5.0, per_seq=0.5, batches=[1, 8, 64]) + _linear_points(
        128, floor=5.0, per_seq=0.05, batches=[1, 8, 64]
    )

    series = context_series(primitives)

    assert [fit["context_length"] for fit in series] == [128, 2048]
    # The longer context streams more KV per sequence, so its knee comes earlier.
    assert series[1]["b_star"] < series[0]["b_star"]


def test_failed_points_are_excluded_from_the_fit() -> None:
    points = _linear_points(128, floor=5.0, per_seq=0.04, batches=[1, 16, 64])
    points.append(_point(512, 128, None, status="oom"))
    points.append(_point(256, 128, None, status="error:capture failed"))

    series = context_series(points)

    assert [p["batch_size"] for p in series[0]["points"]] == [1, 16, 64]
    assert series[0]["floor_ms"] == pytest.approx(5.0)


def test_a_single_point_cannot_anchor_a_fit() -> None:
    primitives = [*_linear_points(128, floor=5.0, per_seq=0.04, batches=[1, 16]), _point(1, 8192, 6.0)]

    with pytest.raises(ValueError, match="needs at least two"):
        context_series(primitives)


def test_latency_that_shrinks_with_batch_is_rejected() -> None:
    # A negative slope has no compute-bound asymptote, so no B* exists.
    with pytest.raises(ValueError, match="positive"):
        context_series([_point(1, 128, 6.0), _point(64, 128, 5.0)])


def test_results_average_the_per_context_floors_and_preserve_model_identity() -> None:
    rows = _linear_rows(128, floor=4.0, per_seq=0.04, batches=[1, 16, 64]) + _linear_rows(
        512, floor=6.0, per_seq=0.08, batches=[1, 16, 64]
    )

    results = build_results(rows)

    assert results["floor_ms"] == pytest.approx(5.0)
    assert results["model_id"] == "org/model"
    assert results["max_depth"] == 4
    assert [fit["context_length"] for fit in results["contexts"]] == [128, 512]


def test_a_rerun_supersedes_the_earlier_measurement_of_a_point() -> None:
    # Each context length is swept in its own process and appended to the shared file, so a
    # rerun leaves both the old and the new row behind.
    rows = [*_linear_rows(128, floor=4.0, per_seq=0.04, batches=[1, 16, 64]), _row(16, 128, 99.0)]

    points = latest_points(rows)

    assert len(points) == 3
    assert [p["core_step_ms"] for p in points if p["batch_size"] == 16] == [99.0]


def test_a_rerun_can_turn_a_measured_point_into_an_oom() -> None:
    rows = [*_linear_rows(128, floor=4.0, per_seq=0.04, batches=[1, 16, 64]), _row(64, 128, None, status="oom")]

    series = context_series(latest_points(rows))

    assert [p["batch_size"] for p in series[0]["points"]] == [1, 16]


def test_rows_from_a_different_model_or_device_are_rejected() -> None:
    rows = _linear_rows(128, floor=4.0, per_seq=0.04, batches=[1, 16, 64]) + _linear_rows(
        512, floor=6.0, per_seq=0.08, batches=[1, 16, 64], device="NVIDIA A100"
    )

    with pytest.raises(ValueError, match="not one sweep"):
        build_results(rows)


def test_rows_are_read_back_from_jsonl(tmp_path: Path) -> None:
    rows = _linear_rows(128, floor=4.0, per_seq=0.04, batches=[1, 16])
    sweep = tmp_path / "sweep.jsonl"
    sweep.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    assert load_rows(sweep) == rows


def test_an_empty_sweep_file_is_rejected(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep.jsonl"
    sweep.write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no sweep rows"):
        load_rows(sweep)


def test_legacy_and_explicit_default_block_sizes_are_compatible() -> None:
    legacy = _row(1, 128, 4.0)
    explicit = _row(16, 128, 5.0)
    explicit["meta"]["block_size"] = 16

    assert run_meta([legacy, explicit])["block_size"] == 16


def test_rows_with_different_measurement_settings_are_rejected() -> None:
    first = _row(1, 128, 4.0)
    second = _row(16, 128, 5.0)
    first["meta"]["block_size"] = 16
    second["meta"]["block_size"] = 256

    with pytest.raises(ValueError, match="block_size"):
        run_meta([first, second])
