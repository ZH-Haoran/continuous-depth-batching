"""Export a decode-step latency sweep for the paper plots.

For each context length, fit the additive linear model
``t(B) = intercept + per_sequence * B`` over measured decode-step latencies.
The coefficients approximate batch-independent and per-sequence costs.
Their ratio is the fitted balance batch ``B*`` where both terms are equal.

Example:
    uv run python scripts/exporters/export_decode_step_latency_results.py \
        outputs/decode_step_latency/decode_step_latency_ouro.jsonl \
        --output docs/paper/figs/decode_latency/decode_step_latency_ouro.json
    uv run python scripts/exporters/export_decode_step_latency_results.py \
        outputs/decode_step_latency/decode_step_latency_ouro26.jsonl \
        --output docs/paper/figs/decode_latency/decode_step_latency_ouro26.json
    uv run python scripts/exporters/export_decode_step_latency_results.py \
        outputs/decode_step_latency/decode_step_latency_huginn.jsonl \
        --output docs/paper/figs/decode_latency/decode_step_latency_huginn.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from looped_cdb.benchmarks.exports import warn

LEGACY_BLOCK_SIZE = 16
VARIABLE_META_FIELDS = {"batch_sizes", "context_length"}


def _campaign_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Normalize metadata that must agree across one sweep."""

    normalized = {**meta, "run_config": dict(meta.get("run_config", {}))}
    normalized.setdefault("block_size", LEGACY_BLOCK_SIZE)
    for field in VARIABLE_META_FIELDS:
        normalized.pop(field, None)
    return normalized


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{path} holds no sweep rows")
    return rows


def latest_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per (context length, batch size), keeping the last one measured."""

    latest = {(row["context_length"], row["batch_size"]): row for row in rows}
    return list(latest.values())


def run_meta(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return metadata after validating that every row belongs to one sweep."""

    meta = rows[0]["meta"]
    expected = _campaign_meta(meta)
    for other in (row["meta"] for row in rows[1:]):
        actual = _campaign_meta(other)
        if actual != expected:
            differing = sorted(key for key in expected.keys() | actual.keys() if expected.get(key) != actual.get(key))
            raise ValueError(f"rows disagree on sweep metadata {differing}; they are not one sweep")
    return {**meta, "block_size": meta.get("block_size", LEGACY_BLOCK_SIZE)}


def context_series(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Measured points and additive linear fit for each context length."""

    ok = [p for p in points if p["status"] == "ok" and p["core_step_ms"] is not None]
    if not ok:
        raise ValueError("the sweep holds no successfully measured points")

    series: list[dict[str, Any]] = []
    for context in sorted({p["context_length"] for p in ok}):
        context_points = sorted((p["batch_size"], p["core_step_ms"]) for p in ok if p["context_length"] == context)
        if len(context_points) < 2:
            raise ValueError(
                f"context {context} has {len(context_points)} measured point(s); the fit needs at least two"
            )
        batches = np.asarray([b for b, _ in context_points], dtype=np.float64)
        latencies = np.asarray([ms for _, ms in context_points], dtype=np.float64)
        per_seq, floor = np.polyfit(batches, latencies, 1)
        if per_seq <= 0 or floor <= 0:
            raise ValueError(
                f"context {context} fits to floor={floor:.4f} ms, per-seq={per_seq:.6f} ms: "
                "the linear fit requires positive coefficients."
            )
        residuals = latencies - (per_seq * batches + floor)
        ss_tot = float(((latencies - latencies.mean()) ** 2).sum())
        series.append(
            {
                "context_length": context,
                "points": [{"batch_size": b, "core_step_ms": ms} for b, ms in context_points],
                "floor_ms": float(floor),
                "per_seq_ms": float(per_seq),
                "b_star": float(floor / per_seq),
                "num_points": len(context_points),
                "batch_range": [int(context_points[0][0]), int(context_points[-1][0])],
                "fit_r2": 1.0 - float((residuals**2).sum()) / ss_tot if ss_tot > 0 else 1.0,
            }
        )
    return series


def build_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    meta = run_meta(rows)
    contexts = context_series(latest_points(rows))
    return {
        "model_id": meta["run_config"]["model"],
        "device_name": meta["device_name"],
        "max_depth": meta["max_depth"],
        # The figure draws one batch-independent intercept: the mean of the per-context
        # fits, which can spread by up to about 10%.
        "floor_ms": sum(c["floor_ms"] for c in contexts) / len(contexts),
        "contexts": contexts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep", type=Path, help="Raw sweep JSONL from benchmark_decode_step_latency.py.")
    parser.add_argument("--output", type=Path, required=True, help="Figure JSON committed beside the plot script.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.sweep)
    results = build_results(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {args.output}")
    print(f"  {results['model_id']} on {results['device_name']}, floor {results['floor_ms']:.2f} ms")
    for context in results["contexts"]:
        low, high = context["batch_range"]
        print(
            f"  L={context['context_length']:>5}: {context['num_points']} points (B {low}-{high}), "
            f"per-seq {context['per_seq_ms']:.4f} ms, B* {context['b_star']:.0f}, fit R^2 {context['fit_r2']:.4f}"
        )
        if context["b_star"] > high:
            warn(
                f"B* at L={context['context_length']} lies beyond the largest measured batch "
                f"({high}); the knee is extrapolated, not observed."
            )
    # oom points are the expected capacity boundary; error points mean a measurement
    # silently went missing from the fit.
    for point in latest_points(rows):
        if point["status"].startswith("error:"):
            warn(
                f"L={point['context_length']} B={point['batch_size']} failed and is "
                f"missing from the fit: {point['status']}"
            )


if __name__ == "__main__":
    main()
