"""Optional per-request serving events and a small, offline comparison export."""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def write_latency_events(
    directory: Path,
    *,
    summary: dict[str, Any],
    outputs: Iterable[Any],
    queue_samples: Iterable[tuple[float, int]],
    start_time: float,
    kv_usage_samples: Iterable[tuple[float, int]] = (),
    resident_usage_samples: Iterable[tuple[float, int]] = (),
    kv_admission_pause_samples: Iterable[tuple[float, int]] = (),
    preemption_events: Iterable[tuple[float, str, str]] = (),
    kv_transfer_events: Iterable[tuple[float, str, int]] = (),
) -> Path:
    """Write one measured run; all timestamps are seconds since the run began."""

    directory.mkdir(parents=True, exist_ok=True)
    run_id = re.sub(r"[^A-Za-z0-9._-]", "_", str(summary["run_id"]))
    path = directory / f"{run_id}-{uuid.uuid4().hex[:8]}.events.jsonl"
    with path.open("x", encoding="utf-8") as handle:
        _write_line(handle, {"type": "run", "schema_version": 1, "summary": summary})
        for output in outputs:
            token_times = output.token_ready_times
            if len(token_times) != len(output.generated_tokens):
                raise ValueError(f"request {output.request_id} lacks a timestamp for an output token")
            _write_line(
                handle,
                {
                    "type": "request",
                    "request_id": output.request_id,
                    "arrival_s": output.created_time - start_time,
                    "scheduled_s": output.lifespan[0] - start_time,
                    "first_token_s": output.first_token_time - start_time,
                    "finish_s": output.lifespan[1] - start_time,
                    "prompt_tokens": len(output.prompt_ids),
                    "output_tokens": len(output.generated_tokens),
                    "token_ready_s": [stamp - start_time for stamp in token_times],
                },
            )
        for stamp, waiting in queue_samples:
            _write_line(handle, {"type": "queue", "time_s": stamp - start_time, "waiting_requests": waiting})
        for stamp, used_blocks in kv_usage_samples:
            _write_line(handle, {"type": "kv", "time_s": stamp - start_time, "used_blocks": used_blocks})
        for stamp, resident in resident_usage_samples:
            _write_line(handle, {"type": "resident", "time_s": stamp - start_time, "requests": resident})
        for stamp, paused in kv_admission_pause_samples:
            _write_line(handle, {"type": "kv_admission_pause", "time_s": stamp - start_time, "paused": paused})
        for stamp, request_id, policy in preemption_events:
            _write_line(handle, {"type": "preemption", "time_s": stamp - start_time,
                                 "request_id": request_id, "policy": policy})
        for stamp, direction, bytes_copied in kv_transfer_events:
            _write_line(handle, {"type": "kv_transfer", "time_s": stamp - start_time,
                                 "direction": direction, "bytes": bytes_copied})
    return path


def _write_line(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def _percentile(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    left = int(position)
    fraction = position - left
    return ordered[left] + (ordered[min(left + 1, len(ordered) - 1)] - ordered[left]) * fraction


def comparison_row(path: Path) -> dict[str, Any]:
    """Recompute comparison values from raw request events, not rounded run summaries."""

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or rows[0].get("type") != "run":
        raise ValueError(f"{path} is missing its run header")
    summary = rows[0]["summary"]
    config = summary["config"]
    requests = [row for row in rows[1:] if row["type"] == "request"]
    queues = [row for row in rows[1:] if row["type"] == "queue"]
    if len(requests) != summary["completed_requests"]:
        raise ValueError(
            f"{path} has {len(requests)} request records but summary reports {summary['completed_requests']}"
        )
    ttft_ms: list[float] = []
    e2e_ms: list[float] = []
    itl_ms: list[float] = []
    request_max_itl_ms: list[float] = []
    normalized_ms: list[float] = []
    for request in requests:
        stamps = request["token_ready_s"]
        if len(stamps) != request["output_tokens"] or not stamps:
            raise ValueError(f"{path}: incomplete token times for {request['request_id']}")
        ttft_ms.append(1000 * (stamps[0] - request["arrival_s"]))
        duration_ms = 1000 * (request["finish_s"] - request["arrival_s"])
        e2e_ms.append(duration_ms)
        normalized_ms.append(duration_ms / len(stamps))
        gaps = [1000 * (later - earlier) for earlier, later in zip(stamps, stamps[1:])]
        itl_ms.extend(gaps)
        if gaps:
            request_max_itl_ms.append(max(gaps))
    result: dict[str, Any] = {
        "run_id": summary["run_id"],
        "events_file": str(path),
        "mode": (
            "cb" if config["backend"] == "cb" else ("cdb-refill" if config["refill"] else "cdb-norefill")
        ),
        "workload": config["workload_name"],
        "rate_rps": config["request_rate_rps"],
        "threshold": config["exit_threshold"],
        "repeat": config["measured_repeat"],
        "requests": len(requests),
        "output_tokens": summary["generated_tokens"],
        "requests_per_s": summary["completed_requests_per_second"],
        "output_tokens_per_s": summary["generated_tokens_per_second"],
        "normalized_mean_ms_per_token": sum(normalized_ms) / len(normalized_ms),
        "queue_peak_waiting": max((row["waiting_requests"] for row in queues), default=0),
    }
    series = (("ttft", ttft_ms), ("itl", itl_ms), ("e2e", e2e_ms), ("request_max_itl", request_max_itl_ms))
    for name, values in series:
        for percentile in (50, 95, 99):
            result[f"{name}_p{percentile}_ms"] = _percentile(values, percentile)
    return result
