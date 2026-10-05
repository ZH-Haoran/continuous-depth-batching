"""Upload a completed serving run after timing has stopped."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from looped_cdb.benchmarks.latency_events import comparison_row


def _git_commit() -> str | None:
    repository = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _queue_table_rows(path: Path, limit: int = 10000) -> list[list[float | int]]:
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["type"] == "queue":
            samples.append([row["time_s"], row["waiting_requests"]])
    if len(samples) <= limit:
        return samples
    stride = max(1, (len(samples) - 1) // (limit - 1))
    reduced = samples[::stride]
    if reduced[-1] != samples[-1]:
        reduced.append(samples[-1])
    return reduced[:limit - 1] + [samples[-1]] if len(reduced) > limit else reduced


def _kv_table_rows(path: Path, *, num_blocks: int, limit: int = 10000) -> list[list[float]]:
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["type"] == "kv":
            rows.append([row["time_s"], 100.0 * row["used_blocks"] / num_blocks])
    if len(rows) <= limit:
        return rows
    stride = max(1, (len(rows) - 1) // (limit - 1))
    reduced = rows[::stride]
    if reduced[-1] != rows[-1]:
        reduced.append(rows[-1])
    return reduced[: limit - 1] + [rows[-1]] if len(reduced) > limit else reduced


def _resident_table_rows(path: Path, limit: int = 10000) -> list[list[float | int]]:
    rows: list[list[float | int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["type"] == "resident":
            rows.append([row["time_s"], row["requests"]])
    if len(rows) <= limit:
        return rows
    stride = max(1, (len(rows) - 1) // (limit - 1))
    reduced = rows[::stride]
    if reduced[-1] != rows[-1]:
        reduced.append(rows[-1])
    return reduced[: limit - 1] + [rows[-1]] if len(reduced) > limit else reduced


def log_serving_run(path: Path, *, project: str) -> str | None:
    """Log comparable scalar metrics, queue samples, and the complete event file to W&B."""

    import wandb

    row = comparison_row(path)
    with path.open(encoding="utf-8") as handle:
        summary = json.loads(handle.readline())["summary"]
    config: dict[str, Any] = dict(summary["config"])
    config.update(run_id=summary["run_id"], device_name=summary.get("device_name"), git_commit=_git_commit())
    metrics = {
        "throughput/requests_per_s": row["requests_per_s"],
        "throughput/output_tokens_per_s": row["output_tokens_per_s"],
        "latency/normalized_mean_ms_per_output_token": row["normalized_mean_ms_per_token"],
        "queue/peak_waiting_requests": row["queue_peak_waiting"],
    }
    for name in ("ttft", "itl", "e2e", "request_max_itl"):
        for percentile in (50, 95, 99):
            value = row[f"{name}_p{percentile}_ms"]
            if value is not None:
                metrics[f"latency/{name}_p{percentile}_ms"] = value
                if name == "itl":
                    metrics[f"latency/tbt_p{percentile}_ms"] = value
    with wandb.init(project=project, name=summary["run_id"], config=config) as run:
        run.log(metrics)
        table = wandb.Table(columns=["elapsed_s", "waiting_requests"], data=_queue_table_rows(path))
        run.log({"queue/waiting_requests": table})
        num_blocks = (summary.get("kv_cache") or {}).get("num_blocks")
        if num_blocks:
            kv_rows = _kv_table_rows(path, num_blocks=num_blocks)
            if kv_rows:
                kv_table = wandb.Table(columns=["elapsed_s", "used_pct"], data=kv_rows)
                run.log(
                    {
                        "kv/occupancy_pct": wandb.plot.line(
                            kv_table, "elapsed_s", "used_pct", title="KV cache occupancy (%)"
                        )
                    }
                )
        resident_rows = _resident_table_rows(path)
        if resident_rows:
            resident_table = wandb.Table(columns=["elapsed_s", "requests"], data=resident_rows)
            run.log(
                {
                    "batch/resident_requests": wandb.plot.line(
                        resident_table, "elapsed_s", "requests", title="Active requests"
                    )
                }
            )
        artifact = wandb.Artifact(name=f"serving-events-{run.id}", type="serving-events")
        artifact.add_file(str(path))
        run.log_artifact(artifact)
        return run.url
