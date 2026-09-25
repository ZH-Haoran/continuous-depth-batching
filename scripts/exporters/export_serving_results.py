"""Export online serving points and optional drain anchors for the paper plots.

Reads summary JSONL written by ``scripts/benchmark_throughput.py``, aggregates the
operating points, records their configuration, and writes one self-describing JSON
next to the paper's plotting script.

Example:
    uv run python scripts/exporters/export_serving_results.py \
        outputs/serving-rate/serving-rate_ouro_sharegpt.jsonl \
        outputs/serving-rate/serving-rate_ouro_alpaca.jsonl \
        outputs/serving-rate/serving-rate_ouro_arxiv.jsonl \
        --output docs/paper/figs/serving/serving_ouro.json
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from looped_cdb.benchmarks.exports import measured_device, shared_meta
from looped_cdb.benchmarks.summaries import (
    AggregateRow,
    aggregate_summaries,
    load_summaries,
    threshold_of,
)


def _point(row: AggregateRow, max_num_seqs: set[Any]) -> dict[str, Any]:
    return {
        "backend": row.backend,
        "refill": row.refill,
        "min_coda_batch_size": row.min_coda_batch_size,
        "threshold": threshold_of(row),
        "rate_rps": row.request_rate_rps,
        "max_num_seqs": sorted(max_num_seqs)[0] if len(max_num_seqs) == 1 else None,
        "norm_lat_mean_s_per_tok": row.norm_lat_mean_s_per_tok,
        "norm_lat_p99_s_per_tok": row.norm_lat_p99_s_per_tok,
        "queue_mean_s": row.queue_mean_s,
        "ttft_mean_s": row.ttft_mean_s,
        "ttft_p99_s": row.ttft_p99_s,
        "e2e_mean_s": row.e2e_mean_s,
        "e2e_p99_s": row.e2e_p99_s,
        "tpot_mean_s": row.tpot_mean_s,
        "gen_tps": row.gen_tps_mean,
        "recurrent_batch_size": row.recurrent_batch_size_mean,
        "coda_batch_size": row.coda_batch_size_mean,
        "resident_requests": row.resident_requests_mean,
        "num_requests": row.num_requests,
        "repeats": row.repeats,
    }


def _sort_key(point: dict[str, Any]) -> tuple[int, bool, int, float, float]:
    is_cdb = point["backend"] == "cdb"
    rate = point["rate_rps"] if point["rate_rps"] is not None else float("inf")
    return (int(is_cdb), bool(point["refill"]), point["min_coda_batch_size"], point["threshold"] or 0.0, rate)


def build_workload_entry(name: str, raw_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """One workload bundle's operating points at a consistent maximum decode batch size."""

    workload_ids = {row["config"].get("workload_id") for row in raw_rows}
    if None in workload_ids or len(workload_ids) != 1:
        raise ValueError(f"the {name} panel mixes workload bundle IDs: {sorted(str(value) for value in workload_ids)}")
    rows = aggregate_summaries(raw_rows)
    caps = {row["config"].get("max_num_seqs") for row in raw_rows}
    if len(caps) != 1:
        values = ", ".join(str(cap) for cap in sorted(caps, key=lambda cap: (cap is None, int(cap or 0))))
        raise ValueError(f"the {name} panel mixes max_num_seqs values: {values}")
    populations: dict[int, set[float]] = {}
    for row in raw_rows:
        config = row["config"]
        populations.setdefault(int(config["num_requests"]), set()).add(float(config["output_mean_tokens"]))
    ambiguous = {count: sorted(means) for count, means in populations.items() if len(means) != 1}
    if ambiguous:
        raise ValueError(f"the {name} panel records different output populations at one request count: {ambiguous}")
    output_populations = [
        {"num_requests": count, "output_mean_tokens": means.pop()} for count, means in sorted(populations.items())
    ]
    points = sorted((_point(row, caps) for row in rows), key=_sort_key)
    return {
        "name": name,
        "workload_id": workload_ids.pop(),
        "output_populations": output_populations,
        "points": points,
    }


MODEL_KEYS_BY_NAME = {
    "Ouro-1.4B": "ouro",
    "huginn-0125": "huginn",
}


def infer_model_key(raw_rows: list[dict[str, Any]]) -> str:
    """Identify the model family independently of the Hub repository owner.

    Repository ownership can change for a checkpoint.
    This classification does not establish weight equivalence or alter recorded provenance.
    """

    model_ids = {str(row["config"]["model"]) for row in raw_rows}
    model_names = {model_id.rsplit("/", 1)[-1] for model_id in model_ids}
    unknown = sorted(model_names - MODEL_KEYS_BY_NAME.keys())
    if unknown:
        raise ValueError(f"unknown serving model ID(s): {unknown}")
    model_keys = {MODEL_KEYS_BY_NAME[name] for name in model_names}
    if len(model_keys) != 1:
        raise ValueError(f"serving export mixes model families: {sorted(model_keys)}")
    return model_keys.pop()


def default_output_path(model_key: str) -> Path:
    """Canonical paper JSON path for one model family."""

    return Path(f"docs/paper/figs/serving/serving_{model_key}.json")


def build_results(summary_paths: list[Path]) -> dict[str, Any]:
    raw_rows: list[dict[str, Any]] = []
    by_workload: dict[str, list[dict[str, Any]]] = {}
    for path in summary_paths:
        rows = load_summaries(path)
        raw_rows.extend(rows)
        for row in rows:
            by_workload.setdefault(row["config"]["workload_name"], []).append(row)
    if not raw_rows:
        raise ValueError(f"no summary rows in {[str(path) for path in summary_paths]}")

    model_key = infer_model_key(raw_rows)
    meta = shared_meta(summary_paths)
    meta["exported_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    meta["device_name"] = measured_device(summary_paths)
    meta["model_key"] = model_key

    return {"meta": meta, "workloads": [build_workload_entry(name, rows) for name, rows in sorted(by_workload.items())]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summaries", type=Path, nargs="+", help="Serving summary JSONL files and drain anchors.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination JSON (default: canonical paper path inferred from the input model).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = build_results(args.summaries)
    output = args.output or default_output_path(results["meta"]["model_key"])

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {output}")
    for workload in results["workloads"]:
        serving = [point for point in workload["points"] if point["rate_rps"] is not None]
        print(
            f"  {workload['name']}: {len(serving)} serving points, "
            f"{len(workload['points']) - len(serving)} drain anchors"
        )
    print(f"  device: {results['meta'].get('device_name') or 'unrecorded'}")


if __name__ == "__main__":
    main()
