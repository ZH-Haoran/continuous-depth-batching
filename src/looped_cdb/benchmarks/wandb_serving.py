"""Upload a completed serving run after timing has stopped."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

from looped_cdb.benchmarks.latency_events import comparison_row


def _preemption_stalls(path: Path) -> dict[str, list[list[float | str]]]:
    """Delay from preemption start to that request's next delivered token."""

    requests: dict[str, list[float]] = {}
    preemptions: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event["type"] == "request":
            requests[event["request_id"]] = event["token_ready_s"]
        elif event["type"] == "preemption":
            preemptions.append(event)
    result: dict[str, list[list[float | str]]] = {"offload": [], "recompute": []}
    for event in preemptions:
        next_token = next((stamp for stamp in requests.get(event["request_id"], [])
                           if stamp >= event["time_s"]), None)
        if next_token is not None:
            result[event["policy"]].append(
                [event["time_s"], 1000 * (next_token - event["time_s"]), event["request_id"]]
            )
    return result


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


def _kv_admission_pause_rows(path: Path) -> list[list[float | int]]:
    rows: list[list[float | int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["type"] == "kv_admission_pause":
            rows.append([row["time_s"], row["paused"]])
    return rows


def _kv_transfer_rows(path: Path) -> dict[str, list[list[float]]]:
    """Cumulative bytes of KV copies enqueued on the compute stream."""

    events: dict[str, list[list[float]]] = {"gpu_to_cpu": [], "cpu_to_gpu": []}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["type"] == "kv_transfer":
            events[row["direction"]].append([row["time_s"], row["bytes"]])
    for direction, rows in events.items():
        total = 0.0
        for row in sorted(rows, key=lambda item: item[0]):
            total += row[1] / 1024**3
            row[1] = total
        events[direction] = sorted(rows, key=lambda item: item[0])
    return events


def _arrival_window_token_rate(path: Path) -> tuple[float, list[list[float]]] | None:
    """Token delivery through the last scheduled arrival, excluding the drain tail."""

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if rows[0]["summary"]["config"].get("request_rate_rps") is None:
        return None
    requests = [row for row in rows[1:] if row["type"] == "request"]
    if not requests:
        return None
    end_s = max(row["arrival_s"] for row in requests)
    if end_s <= 0:
        return None
    token_times = [stamp for row in requests for stamp in row["token_ready_s"] if 0 <= stamp <= end_s]
    bins = [0] * math.ceil(end_s)
    for stamp in token_times:
        bins[min(int(stamp), len(bins) - 1)] += 1
    timeline = [[float(index), count / min(1.0, end_s - index)] for index, count in enumerate(bins)]
    return len(token_times) / end_s, timeline


def _arrival_window_backlog(path: Path) -> tuple[int, list[list[float | int]]] | None:
    """Arrived requests still unfinished when the scheduled arrival trace ends."""

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if rows[0]["summary"]["config"].get("request_rate_rps") is None:
        return None
    requests = [row for row in rows[1:] if row["type"] == "request"]
    if not requests:
        return None
    end_s = max(row["arrival_s"] for row in requests)
    if end_s <= 0:
        return None
    changes = [(row["arrival_s"], 1) for row in requests]
    changes.extend((row["finish_s"], -1) for row in requests if row["finish_s"] <= end_s)
    changes.sort()
    count = 0
    timeline: list[list[float | int]] = [[0.0, 0]]
    for stamp, change in changes:
        count += change
        timeline.append([stamp, count])
    return count, timeline


def _recompute_rows(path: Path) -> list[list[float | int]]:
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["type"] == "recompute":
            events.append((row["time_s"], row["tokens_to_reprefill"]))
    total = 0
    rows: list[list[float | int]] = []
    for stamp, tokens in sorted(events):
        total += tokens
        rows.append([stamp, total])
    return rows


def _stage_launch_rows(path: Path) -> dict[str, list[list[float | int]]]:
    stages: dict[str, list[list[float | int]]] = {name: [] for name in ("prefill", "prelude", "recurrent", "coda")}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["type"] == "stage_launch":
            stages[row["stage"]].append([row["time_s"], row["batch_size"]])
    return stages


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
        pause_rows = _kv_admission_pause_rows(path)
        if pause_rows:
            paused_s = sum(
                (later[0] - earlier[0]) * earlier[1] for earlier, later in zip(pause_rows, pause_rows[1:])
            )
            run.log({"kv/admission_paused_s": paused_s})
            chart_rows = pause_rows
            if len(chart_rows) > 10000:
                stride = max(1, (len(chart_rows) - 1) // 9999)
                chart_rows = chart_rows[::stride]
                if chart_rows[-1] != pause_rows[-1]:
                    chart_rows = chart_rows[:9999] + [pause_rows[-1]]
            pause_table = wandb.Table(columns=["elapsed_s", "paused"], data=chart_rows)
            run.log(
                {
                    "kv/admission_paused": wandb.plot.line(
                        pause_table, "elapsed_s", "paused", title="New-request admission paused by KV headroom"
                    )
                }
            )
        for policy, stalls in _preemption_stalls(path).items():
            if stalls:
                durations = sorted(row[1] for row in stalls)
                index = int(0.95 * (len(durations) - 1))
                run.log({f"preemption/{policy}_to_next_token_p95_ms": durations[index]})
                table = wandb.Table(columns=["elapsed_s", "stall_ms", "request_id"], data=stalls[:10000])
                run.log({f"preemption/{policy}_to_next_token_ms": wandb.plot.scatter(
                    table, "elapsed_s", "stall_ms", title=f"{policy.title()} to next output token (ms)"
                )})
        for direction, transfers in _kv_transfer_rows(path).items():
            if transfers:
                run.log({f"kv_transfer/{direction}_gib": transfers[-1][1]})
                chart_rows = transfers
                if len(chart_rows) > 10000:
                    chart_rows = transfers[::max(1, len(transfers) // 9999)][:9999] + [transfers[-1]]
                table = wandb.Table(columns=["elapsed_s", "cumulative_gib"], data=chart_rows)
                run.log({f"kv_transfer/{direction}_cumulative_gib": wandb.plot.line(
                    table, "elapsed_s", "cumulative_gib", title=f"KV {direction} copied (GiB enqueued)"
                )})
        arrival_rate = _arrival_window_token_rate(path)
        if arrival_rate is not None:
            rate, timeline = arrival_rate
            run.log({"throughput/arrival_window_output_tokens_per_s": rate})
            if len(timeline) > 10000:
                stride = math.ceil(len(timeline) / 10000)
                timeline = timeline[::stride]
            table = wandb.Table(columns=["elapsed_s", "output_tokens_per_s"], data=timeline)
            run.log({"throughput/arrival_window_token_rate": wandb.plot.line(
                table, "elapsed_s", "output_tokens_per_s", title="Output tokens/s while requests arrive"
            )})
        backlog = _arrival_window_backlog(path)
        if backlog is not None:
            count, timeline = backlog
            run.log({"backlog/requests_at_last_arrival": count})
            if len(timeline) > 10000:
                stride = math.ceil(len(timeline) / 10000)
                timeline = timeline[::stride][:9999] + [timeline[-1]]
            table = wandb.Table(columns=["elapsed_s", "unfinished_requests"], data=timeline)
            run.log({"backlog/unfinished_during_arrivals": wandb.plot.line(
                table, "elapsed_s", "unfinished_requests", title="Arrived but unfinished requests"
            )})
        recompute_rows = _recompute_rows(path)
        if recompute_rows:
            run.log({"preemption/tokens_to_reprefill": recompute_rows[-1][1]})
            chart_rows = recompute_rows
            if len(chart_rows) > 10000:
                stride = math.ceil(len(chart_rows) / 10000)
                chart_rows = chart_rows[::stride][:9999] + [recompute_rows[-1]]
            table = wandb.Table(columns=["elapsed_s", "cumulative_tokens"], data=chart_rows)
            run.log({"preemption/cumulative_tokens_to_reprefill": wandb.plot.line(
                table, "elapsed_s", "cumulative_tokens", title="Prompt tokens scheduled for recomputation"
            )})
        for stage, stage_rows in _stage_launch_rows(path).items():
            if stage_rows:
                chart_rows = stage_rows
                if len(chart_rows) > 10000:
                    stride = math.ceil(len(chart_rows) / 10000)
                    chart_rows = chart_rows[::stride][:9999] + [stage_rows[-1]]
                table = wandb.Table(columns=["elapsed_s", "batch_size"], data=chart_rows)
                run.log({f"stage/{stage}_batch_size": wandb.plot.scatter(
                    table, "elapsed_s", "batch_size", title=f"CDB {stage} launches: batch size"
                )})
        artifact = wandb.Artifact(name=f"serving-events-{run.id}", type="serving-events")
        artifact.add_file(str(path))
        run.log_artifact(artifact)
        return run.url
