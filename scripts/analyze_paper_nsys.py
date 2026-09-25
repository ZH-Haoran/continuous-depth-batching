"""Analyze paper Nsight Systems SQLite exports."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GPU_ACTIVITY_TABLES = (
    "CUPTI_ACTIVITY_KIND_KERNEL",
    "CUPTI_ACTIVITY_KIND_MEMCPY",
    "CUPTI_ACTIVITY_KIND_MEMSET",
    # Whole-graph GPU execution intervals, present when the trace was captured with
    # --cuda-graph-trace=graph; graph child kernels do not appear as KERNEL rows there.
    "CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
)
KERNEL_TABLES = ("CUPTI_ACTIVITY_KIND_KERNEL",)
MEMCPY_TABLES = ("CUPTI_ACTIVITY_KIND_MEMCPY",)
CUDA_API_TABLES = (
    "CUPTI_ACTIVITY_KIND_RUNTIME",
    "CUPTI_ACTIVITY_KIND_DRIVER",
)
# Preferred measured windows, in selection order.
MEASURED_LABELS = ("benchmark.steady", "benchmark.generate")
# Display order for the per-stage summary (engine pipeline order); stages are discovered
# from the trace, and labels missing here are appended alphabetically.
STAGE_ORDER = (
    "cb.schedule",
    "cb.prepare",
    "cb.prepare.stage",
    "cb.prepare.h2d",
    "cb.compute.prefill",
    "cb.compute.decode",
    "cb.retrieve",
    "cb.update",
    "cb.output_wait",
    "cdb.schedule",
    "cdb.prefill",
    "cdb.prefill.stage",
    "cdb.prefill.h2d",
    "cdb.prefill.launch",
    "cdb.prefill_wait",
    "cdb.prefill_consume",
    "cdb.prelude.after_prefill",
    "cdb.prelude.after_prefill_eager",
    "cdb.prelude.after_coda_eager",
    "cdb.prelude.after_coda_staged",
    "cdb.prelude.resumed",
    "cdb.prelude.wave_cohort",
    "cdb.prelude.buffer_wait",
    "cdb.recurrent",
    "cdb.recurrent.buffer_wait",
    "cdb.recurrent.stage",
    "cdb.stage_attention.buffer_wait",
    "cdb.recurrent.h2d",
    "cdb.recurrent.launch",
    "cdb.recurrent.gate_sync",
    "cdb.gate_wait",
    "cdb.exit_route",
    "cdb.exit_kv_copy",
    "cdb.coda_stage",
    "cdb.coda.buffer_wait",
    "cdb.coda_consume",
    "cdb.coda_wait",
    "runner.graph_replay",
    "runner.graph_capture",
    "runner.prefill_graph_replay",
    "runner.prefill_graph_capture",
)
GAP_THRESHOLDS_NS = {
    "100us": 100_000,
    "1ms": 1_000_000,
}


@dataclass(frozen=True)
class Interval:
    start: int
    end: int

    @property
    def duration(self) -> int:
        return max(0, self.end - self.start)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--label", default=None)
    parser.add_argument(
        "--stage",
        action="append",
        default=None,
        help=(
            "Restrict the per-stage summary to these NVTX labels. By default, include every "
            "non-window label overlapping the selected measured window."
        ),
    )
    parser.add_argument(
        "--measured-label",
        choices=MEASURED_LABELS,
        default=None,
        help="Select an NVTX measurement window. By default, prefer benchmark.steady when present.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_sqlite(
        args.sqlite,
        label=args.label,
        stage_labels=tuple(args.stage) if args.stage is not None else None,
        measured_label=args.measured_label,
    )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, sort_keys=True, indent=2), encoding="utf-8")
    if args.output_csv is not None:
        write_summary_csv(summary, args.output_csv)
    print(json.dumps(summary, sort_keys=True))


def summarize_sqlite(
    path: Path,
    *,
    label: str | None,
    stage_labels: tuple[str, ...] | None,
    measured_label: str | None = None,
) -> dict[str, Any]:
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        strings = string_ids(conn)
        busy_intervals = load_gpu_intervals(conn)
        nvtx_by_label = load_nvtx_intervals(conn, strings)
        candidates = MEASURED_LABELS if measured_label is None else (measured_label,)
        measured_label, measured_windows = measured_ranges(nvtx_by_label, candidates=candidates)
        if not measured_windows:
            raise ValueError(f"no {' or '.join(candidates)} NVTX range found in {path}")

        measured_windows = merge_intervals(measured_windows)
        gpu_ns_by_corr = load_gpu_ns_by_correlation(conn)
        launches = load_runtime_launches(conn, measured_windows)
        launched_ns_by_stage = launched_gpu_ns_by_stage(nvtx_by_label, launches, gpu_ns_by_corr)
        measured_launched_ns = sum(gpu_ns_by_corr.get(corr, 0) for _, _, _, corr in launches)

        known = [label for label in STAGE_ORDER if label in nvtx_by_label]
        extra = sorted(set(nvtx_by_label) - set(STAGE_ORDER))
        stages = {}
        for stage_label in known + extra:
            if stage_labels is not None and stage_label not in stage_labels:
                continue
            clipped = clip_intervals_to_windows(flatten_ranges(nvtx_by_label[stage_label]), measured_windows)
            if clipped:
                stages[stage_label] = threshold_summaries(
                    clipped, busy_intervals, launched_ns=launched_ns_by_stage.get(stage_label, 0)
                )

        return {
            "path": str(path),
            "label": label or path.stem,
            "measured_label": measured_label,
            "measured_range_count": len(measured_windows),
            "gpu_activity_interval_count": len(busy_intervals),
            "diagnostics": capture_diagnostics(conn, measured_windows, strings),
            "measured": threshold_summaries(measured_windows, busy_intervals, launched_ns=measured_launched_ns),
            "stages": stages,
            "kernels": kernel_summary(conn, measured_windows, strings),
            "memcpy": memcpy_summary(conn, measured_windows, strings),
            "cuda_api": cuda_api_summary(conn, measured_windows, strings),
        }


def table_names(conn: sqlite3.Connection) -> set[str]:
    return {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def string_ids(conn: sqlite3.Connection) -> dict[int, str]:
    if "StringIds" not in table_names(conn):
        return {}
    columns = table_columns(conn, "StringIds")
    if not {"id", "value"}.issubset(columns):
        return {}
    return {int(row["id"]): str(row["value"]) for row in conn.execute("SELECT id, value FROM StringIds")}


def load_gpu_intervals(conn: sqlite3.Connection) -> list[Interval]:
    intervals = []
    tables = table_names(conn)
    for table in GPU_ACTIVITY_TABLES:
        if table not in tables:
            continue
        intervals.extend(intervals_from_table(conn, table))
    return merge_intervals(intervals)


def load_nvtx_intervals(conn: sqlite3.Connection, strings: dict[int, str]) -> dict[str, dict[int, list[Interval]]]:
    """Load NVTX ranges grouped by label, then by emitting thread (globalTid, 0 when absent)."""

    if "NVTX_EVENTS" not in table_names(conn):
        return {}
    has_tid = "globalTid" in table_columns(conn, "NVTX_EVENTS")
    intervals: dict[str, dict[int, list[Interval]]] = {}
    for row in conn.execute("SELECT * FROM NVTX_EVENTS"):
        row_label = row_name(row, strings)
        if row_label is None:
            continue
        interval = interval_from_row(row)
        if interval is None:
            continue
        tid = int(row["globalTid"]) if has_tid and row["globalTid"] is not None else 0
        intervals.setdefault(row_label, {}).setdefault(tid, []).append(interval)
    return intervals


def flatten_ranges(by_tid: dict[int, list[Interval]]) -> list[Interval]:
    return [interval for intervals in by_tid.values() for interval in intervals]


def measured_ranges(
    nvtx_by_label: dict[str, dict[int, list[Interval]]], *, candidates: tuple[str, ...] = MEASURED_LABELS
) -> tuple[str, list[Interval]]:
    """Pop the window the trace is clipped to: the first candidate label the trace holds.

    Every benchmark window is removed from ``nvtx_by_label`` so none is reported as a stage.
    """

    chosen, windows = candidates[-1], []
    for candidate in MEASURED_LABELS:
        ranges = flatten_ranges(nvtx_by_label.pop(candidate, {}))
        if candidate not in candidates:
            continue
        if ranges and not windows:
            chosen, windows = candidate, ranges
    return chosen, windows


def load_gpu_ns_by_correlation(conn: sqlite3.Connection) -> dict[int, int]:
    """Total GPU-activity nanoseconds per CUDA correlation id."""

    totals: dict[int, int] = {}
    tables = table_names(conn)
    for table in GPU_ACTIVITY_TABLES:
        if table not in tables or "correlationId" not in table_columns(conn, table):
            continue
        for row in conn.execute(f"SELECT start, end, correlationId FROM {table}"):
            if row["correlationId"] is None or row["start"] is None or row["end"] is None:
                continue
            duration = int(row["end"]) - int(row["start"])
            if duration > 0:
                corr = int(row["correlationId"])
                totals[corr] = totals.get(corr, 0) + duration
    return totals


def load_runtime_launches(conn: sqlite3.Connection, windows: list[Interval]) -> list[tuple[int, int, int, int]]:
    """CUDA runtime and driver calls overlapping the measured windows.

    Returns ``(tid, start, end, correlationId)`` tuples.
    """

    launches: list[tuple[int, int, int, int]] = []
    tables = table_names(conn)
    for table in CUDA_API_TABLES:
        if table not in tables:
            continue
        columns = table_columns(conn, table)
        if not {"correlationId", "globalTid"}.issubset(columns):
            continue
        for row in conn.execute(f"SELECT start, end, globalTid, correlationId FROM {table}"):
            if row["correlationId"] is None or row["start"] is None or row["end"] is None:
                continue
            interval = Interval(int(row["start"]), int(row["end"]))
            if not interval_overlaps_windows(interval, windows):
                continue
            tid = int(row["globalTid"]) if row["globalTid"] is not None else 0
            launches.append((tid, interval.start, interval.end, int(row["correlationId"])))
    launches.sort(key=lambda item: item[1])
    return launches


def launched_gpu_ns_by_stage(
    nvtx_by_label: dict[str, dict[int, list[Interval]]],
    launches: list[tuple[int, int, int, int]],
    gpu_ns_by_corr: dict[int, int],
) -> dict[str, int]:
    """GPU nanoseconds attributed to the stage whose thread issued the launching API call.

    Unlike wall-clock overlap, this follows the CUDA correlation id from each
    runtime/driver call to the GPU work it produced, so asynchronous execution is
    credited to the stage that launched it even when the GPU runs long after the
    host range closed.

    A launch is credited to every NVTX range containing it, so nested ranges (e.g.
    ``cdb.prefill`` and ``cdb.prefill.launch``) each carry it: the per-stage column
    is a tree, not a partition, and summing it exceeds the measured total.
    """

    launches_by_tid: dict[int, list[tuple[int, int, int]]] = {}
    for tid, start, end, corr in launches:
        launches_by_tid.setdefault(tid, []).append((start, end, corr))

    totals: dict[str, int] = {}
    for label, ranges_by_tid in nvtx_by_label.items():
        total = 0
        for tid, ranges in ranges_by_tid.items():
            tid_launches = launches_by_tid.get(tid)
            if not tid_launches:
                continue
            starts = [launch[0] for launch in tid_launches]
            for stage_range in merge_intervals(ranges):
                index = bisect_left(starts, stage_range.start)
                while index < len(tid_launches):
                    launch_start, launch_end, corr = tid_launches[index]
                    if launch_start > stage_range.end:
                        break
                    # Containment, not overlap: a launch crossing the range end is credited
                    # to no stage (the measured total admits it via window overlap).
                    if launch_end <= stage_range.end:
                        total += gpu_ns_by_corr.get(corr, 0)
                    index += 1
        if total:
            totals[label] = total
    return totals


def intervals_from_table(conn: sqlite3.Connection, table: str) -> list[Interval]:
    if not {"start", "end"}.issubset(table_columns(conn, table)):
        return []
    return [
        interval for row in conn.execute(f"SELECT * FROM {table}") if (interval := interval_from_row(row)) is not None
    ]


def interval_from_row(row: sqlite3.Row) -> Interval | None:
    try:
        start = row["start"]
        end = row["end"]
    except (IndexError, KeyError):
        return None
    if start is None or end is None:
        return None
    interval = Interval(int(start), int(end))
    return interval if interval.end > interval.start else None


def row_name(row: sqlite3.Row, strings: dict[int, str]) -> str | None:
    keys = set(row.keys())
    for key in ("text", "name", "demangledName", "shortName", "mangledName"):
        if key in keys and row[key] is not None:
            value = row[key]
            if isinstance(value, int):
                return strings.get(int(value), str(value))
            return str(value)
    for key in ("textId", "nameId", "demangledNameId", "shortNameId", "mangledNameId"):
        if key in keys and row[key] is not None:
            value = strings.get(int(row[key]))
            if value is not None:
                return value
    return None


def threshold_summaries(
    windows: list[Interval], busy_intervals: list[Interval], *, launched_ns: int = 0
) -> dict[str, dict[str, float | int]]:
    summaries = {}
    for label, threshold_ns in GAP_THRESHOLDS_NS.items():
        summary = summarize_disjoint_windows(windows, busy_intervals, gap_threshold_ns=threshold_ns)
        # Launch-site attribution is gap-threshold independent; repeated per row for flat CSV output.
        summary["gpu_launched_s"] = launched_ns / 1e9
        summaries[label] = summary
    return summaries


def summarize_disjoint_windows(
    windows: list[Interval],
    busy_intervals: list[Interval],
    *,
    gap_threshold_ns: int,
) -> dict[str, float | int]:
    windows = merge_intervals(windows)
    duration_ns = total_duration(windows)
    busy_ns, gaps = busy_duration_and_gaps(windows, busy_intervals)
    large_gaps = [gap for gap in gaps if gap.duration >= gap_threshold_ns]
    idle_ns = duration_ns - busy_ns
    return {
        "duration_s": duration_ns / 1e9,
        "gpu_busy_s": busy_ns / 1e9,
        "gpu_idle_s": idle_ns / 1e9,
        "gpu_busy_fraction": busy_ns / duration_ns if duration_ns else 0.0,
        "gpu_idle_fraction": idle_ns / duration_ns if duration_ns else 0.0,
        "idle_gap_count": len(gaps),
        "large_idle_gap_count": len(large_gaps),
        "large_idle_gap_total_s": total_duration(large_gaps) / 1e9,
        "largest_idle_gap_ms": max((gap.duration for gap in gaps), default=0) / 1e6,
    }


def busy_duration_and_gaps(windows: list[Interval], busy_intervals: list[Interval]) -> tuple[int, list[Interval]]:
    windows = merge_intervals(windows)
    busy_intervals = merge_intervals(busy_intervals)
    busy_ns = 0
    gaps: list[Interval] = []
    busy_index = 0
    for window in windows:
        cursor = window.start
        while busy_index < len(busy_intervals) and busy_intervals[busy_index].end <= window.start:
            busy_index += 1
        scan_index = busy_index
        while scan_index < len(busy_intervals):
            interval = busy_intervals[scan_index]
            if interval.start >= window.end:
                break
            start = max(interval.start, window.start)
            end = min(interval.end, window.end)
            if end > start:
                if start > cursor:
                    gaps.append(Interval(cursor, start))
                busy_ns += end - start
                cursor = max(cursor, end)
            scan_index += 1
        if cursor < window.end:
            gaps.append(Interval(cursor, window.end))
    return busy_ns, gaps


def merge_intervals(intervals: list[Interval]) -> list[Interval]:
    intervals = sorted(
        (interval for interval in intervals if interval.end > interval.start), key=lambda item: item.start
    )
    if not intervals:
        return []
    merged = [intervals[0]]
    for interval in intervals[1:]:
        previous = merged[-1]
        if interval.start <= previous.end:
            merged[-1] = Interval(previous.start, max(previous.end, interval.end))
        else:
            merged.append(interval)
    return merged


def intersect_intervals(intervals: list[Interval], window: Interval) -> list[Interval]:
    clipped = []
    for interval in intervals:
        start = max(interval.start, window.start)
        end = min(interval.end, window.end)
        if end > start:
            clipped.append(Interval(start, end))
    return merge_intervals(clipped)


def clip_intervals_to_windows(intervals: list[Interval], windows: list[Interval]) -> list[Interval]:
    return merge_intervals([clipped for window in windows for clipped in intersect_intervals(intervals, window)])


def total_duration(intervals: list[Interval]) -> int:
    return sum(interval.duration for interval in intervals)


def interval_overlaps_windows(interval: Interval, windows: list[Interval]) -> bool:
    return any(min(interval.end, window.end) > max(interval.start, window.start) for window in windows)


def capture_diagnostics(conn: sqlite3.Connection, windows: list[Interval], strings: dict[int, str]) -> dict[str, Any]:
    cuda_graph_launch_count = cuda_api_call_count(conn, windows, strings, name_prefix="cudaGraphLaunch")
    has_graph_node_kernel_activity = kernel_graph_node_activity_count(conn) > 0
    has_graph_trace_activity = graph_trace_activity_count(conn) > 0
    warnings = []
    if cuda_graph_launch_count > 0 and not (has_graph_node_kernel_activity or has_graph_trace_activity):
        warnings.append(
            "CUDA graph launches were observed, but neither graph-node kernel activity nor "
            "whole-graph GRAPH_TRACE intervals are present. This usually means the profiler "
            "capture started after CUDA graph construction. GPU busy/idle is likely incomplete. "
            "Capture with --cuda-graph-trace=graph (or node) over a range that includes CUDA "
            "graph warmup/construction."
        )
    return {
        "cuda_graph_launch_count": cuda_graph_launch_count,
        "has_graph_node_kernel_activity": has_graph_node_kernel_activity,
        "has_graph_trace_activity": has_graph_trace_activity,
        "warnings": warnings,
    }


def graph_trace_activity_count(conn: sqlite3.Connection) -> int:
    if "CUPTI_ACTIVITY_KIND_GRAPH_TRACE" not in table_names(conn):
        return 0
    return int(conn.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE").fetchone()[0])


def kernel_graph_node_activity_count(conn: sqlite3.Connection) -> int:
    count = 0
    tables = table_names(conn)
    for table in KERNEL_TABLES:
        if table not in tables or "graphNodeId" not in table_columns(conn, table):
            continue
        count += int(conn.execute(f"SELECT COUNT(*) FROM {table} WHERE graphNodeId IS NOT NULL").fetchone()[0])
    return count


def cuda_api_call_count(
    conn: sqlite3.Connection,
    windows: list[Interval],
    strings: dict[int, str],
    *,
    name_prefix: str,
) -> int:
    count = 0
    tables = table_names(conn)
    for table in CUDA_API_TABLES:
        if table not in tables:
            continue
        for row in conn.execute(f"SELECT * FROM {table}"):
            interval = interval_from_row(row)
            if interval is None or not interval_overlaps_windows(interval, windows):
                continue
            name = row_name(row, strings)
            if name is not None and name.startswith(name_prefix):
                count += 1
    return count


def kernel_summary(conn: sqlite3.Connection, windows: list[Interval], strings: dict[int, str]) -> dict[str, Any]:
    groups: dict[str, dict[str, float | int | str]] = {}
    tables = table_names(conn)
    for table in KERNEL_TABLES:
        if table not in tables:
            continue
        for row in conn.execute(f"SELECT * FROM {table}"):
            interval = interval_from_row(row)
            if interval is None or not interval_overlaps_windows(interval, windows):
                continue
            name = shorten_kernel_name(row_name(row, strings) or "unknown")
            group = groups.setdefault(name, {"name": name, "calls": 0, "total_s": 0.0})
            group["calls"] = int(group["calls"]) + 1
            group["total_s"] = float(group["total_s"]) + interval.duration / 1e9
    top = sorted(groups.values(), key=lambda item: float(item["total_s"]), reverse=True)[:20]
    return {
        "kernel_launch_count": sum(int(item["calls"]) for item in groups.values()),
        "kernel_total_s": sum(float(item["total_s"]) for item in groups.values()),
        "top_kernel_groups": top,
    }


def memcpy_summary(conn: sqlite3.Connection, windows: list[Interval], strings: dict[int, str]) -> dict[str, Any]:
    by_kind: dict[str, dict[str, float | int | str]] = {}
    tables = table_names(conn)
    for table in MEMCPY_TABLES:
        if table not in tables:
            continue
        for row in conn.execute(f"SELECT * FROM {table}"):
            interval = interval_from_row(row)
            if interval is None or not interval_overlaps_windows(interval, windows):
                continue
            kind = copy_kind(row, strings)
            group = by_kind.setdefault(kind, {"kind": kind, "count": 0, "total_s": 0.0, "bytes": 0})
            group["count"] = int(group["count"]) + 1
            group["total_s"] = float(group["total_s"]) + interval.duration / 1e9
            group["bytes"] = int(group["bytes"]) + row_bytes(row)
    return {
        "memcpy_count": sum(int(item["count"]) for item in by_kind.values()),
        "memcpy_total_s": sum(float(item["total_s"]) for item in by_kind.values()),
        "by_kind": sorted(by_kind.values(), key=lambda item: str(item["kind"])),
    }


def cuda_api_summary(conn: sqlite3.Connection, windows: list[Interval], strings: dict[int, str]) -> dict[str, Any]:
    groups: dict[str, dict[str, float | int | str]] = {}
    tables = table_names(conn)
    for table in CUDA_API_TABLES:
        if table not in tables:
            continue
        for row in conn.execute(f"SELECT * FROM {table}"):
            interval = interval_from_row(row)
            if interval is None or not interval_overlaps_windows(interval, windows):
                continue
            name = row_name(row, strings) or table
            group = groups.setdefault(name, {"name": name, "calls": 0, "total_s": 0.0})
            group["calls"] = int(group["calls"]) + 1
            group["total_s"] = float(group["total_s"]) + interval.duration / 1e9
    top = sorted(groups.values(), key=lambda item: float(item["total_s"]), reverse=True)[:20]
    return {
        "cuda_api_call_count": sum(int(item["calls"]) for item in groups.values()),
        "cuda_api_total_s": sum(float(item["total_s"]) for item in groups.values()),
        "top_cuda_api": top,
    }


def copy_kind(row: sqlite3.Row, strings: dict[int, str]) -> str:
    keys = set(row.keys())
    for key in ("copyKind", "copyKindName", "srcKind", "name"):
        if key in keys and row[key] is not None:
            value = row[key]
            if isinstance(value, int):
                return strings.get(int(value), str(value))
            return str(value)
    if "copyKindId" in keys and row["copyKindId"] is not None:
        return strings.get(int(row["copyKindId"]), str(row["copyKindId"]))
    return "unknown"


def row_bytes(row: sqlite3.Row) -> int:
    keys = set(row.keys())
    for key in ("bytes", "size", "copySize"):
        if key in keys and row[key] is not None:
            return int(row[key])
    return 0


def shorten_kernel_name(name: str) -> str:
    name = name.replace("void ", "")
    if len(name) <= 160:
        return name
    return name[:157] + "..."


def write_summary_csv(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for threshold_label, values in summary["measured"].items():
        rows.append(csv_row("measured", summary["label"], threshold_label, values))
    for stage_label, stage_summary in summary["stages"].items():
        for threshold_label, values in stage_summary.items():
            rows.append(csv_row("stage", stage_label, threshold_label, values))

    fieldnames = [
        "section",
        "label",
        "gap_threshold",
        "duration_s",
        "gpu_busy_s",
        "gpu_idle_s",
        "gpu_busy_fraction",
        "gpu_idle_fraction",
        "gpu_launched_s",
        "idle_gap_count",
        "large_idle_gap_count",
        "large_idle_gap_total_s",
        "largest_idle_gap_ms",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def csv_row(section: str, label: str, threshold: str, values: dict[str, float | int]) -> dict[str, Any]:
    row = {"section": section, "label": label, "gap_threshold": threshold}
    row.update(values)
    return row


if __name__ == "__main__":
    main()
