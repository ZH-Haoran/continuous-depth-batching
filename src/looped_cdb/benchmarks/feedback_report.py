"""Reproducible CPU experiment reports and plots for feedback batching."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .feedback import CostTable, RecurrenceTrace, ScheduleResult, simulate_static
from .feedback_dp import solve_dp_oracle
from .feedback_oracle import solve_oracle


def load_trace(path: Path) -> RecurrenceTrace:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format") != "looped-cdb-recurrence-trace-v1":
        raise ValueError("expected looped-cdb-recurrence-trace-v1")
    return RecurrenceTrace(tuple(tuple(chain) for chain in data["chains"]), metadata=data.get("metadata", {}))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_schedule(path: Path, result: ScheduleResult) -> None:
    rows = [asdict(step) for step in result.steps]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(
    costs: CostTable,
    trace: RecurrenceTrace,
    output_dir: Path,
    *,
    thresholds: tuple[int, ...] = (1, 4, 8, 16, 32, 64),
    oracle: str = "dp",
    max_states: int = 100_000,
    time_limit_s: float = 30,
    plots: bool = True,
) -> dict[str, Any]:
    """Export the same finite token chains under each policy, including the tail."""
    thresholds = tuple(sorted(set(thresholds)))
    if not thresholds or any(type(k) is not int or not 1 <= k <= costs.resident_cap for k in thresholds):
        raise ValueError("thresholds must be integers in 1..resident_cap")
    if oracle not in ("dp", "astar"):
        raise ValueError("oracle must be dp or astar")
    schedules = {k: simulate_static(trace, costs, k) for k in thresholds}
    if oracle == "dp":
        result = solve_dp_oracle(trace, costs)
        oracle_schedule = result.schedule
        oracle_info = {
            "solver": "dp",
            "exact": True,
            "runtime_ms": result.runtime_ms,
            "lower_bound_ms": result.runtime_ms,
            "upper_bound_ms": result.runtime_ms,
            "states_evaluated": result.states_evaluated,
            "stop_reason": "solved",
            "search_seconds": result.search_seconds,
            "core_segments": result.core_segments,
        }
    else:
        result = solve_oracle(trace, costs, thresholds=thresholds, max_states=max_states, time_limit_s=time_limit_s)
        oracle_schedule = result.schedule
        oracle_info = {
            "solver": "astar",
            "exact": result.exact,
            "lower_bound_ms": result.lower_bound_ms,
            "upper_bound_ms": result.upper_bound_ms,
            "expanded_states": result.expanded_states,
            "stop_reason": result.stop_reason,
            "search_seconds": result.search_seconds,
            "max_states": max_states,
            "time_limit_s": time_limit_s,
        }
    best_k = min(schedules, key=lambda k: schedules[k].runtime_ms)
    best = schedules[best_k].runtime_ms
    all_k = thresholds == tuple(range(1, costs.resident_cap + 1))
    gap_lower = best / oracle_info["upper_bound_ms"] if oracle_info["upper_bound_ms"] else None
    gap_upper = best / oracle_info["lower_bound_ms"] if oracle_info["lower_bound_ms"] else None
    rows = []
    for k, result in schedules.items():
        summary = result.to_dict()
        summary.pop("steps")
        rows.append({"k": k, **summary})
    report = {
        "format": "looped-cdb-feedback-report-v1",
        "model": {
            "resident_cap": costs.resident_cap,
            "initial_sequences": len(trace.chains),
            "token_count": trace.token_count,
            "core_visits": trace.core_visits,
            "observed_max_depth": max(max(chain) for chain in trace.chains),
            "observed_mean_depth": trace.core_visits / trace.token_count,
            "successor_release": "after_coda_and_prelude",
            "action_space": "core_all_runnable_or_boundary_flush_all_pending",
            "timing_scope": "serial_fixed_context_decode_stages_including_initial_prelude_and_final_coda",
        },
        "cost_metadata": costs.metadata,
        "trace_metadata": trace.metadata,
        "static": rows,
        "best_static": {"k": best_k, "runtime_ms": best, "all_thresholds_scanned": all_k},
        "oracle": oracle_info,
        "gap": {
            "numerator": "best_static" if all_k else "best_scanned_static",
            "lower": gap_lower,
            "upper": gap_upper,
            "exact": oracle_info["exact"],
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "costs.json", costs.to_dict())
    write_json(
        output_dir / "trace.json",
        {"format": "looped-cdb-recurrence-trace-v1", "chains": trace.chains, "metadata": trace.metadata},
    )
    write_json(output_dir / "report.json", report)
    with (output_dir / "static.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for k, result in schedules.items():
        write_schedule(output_dir / f"static_k_{k}_schedule.csv", result)
    write_schedule(output_dir / "oracle_schedule.csv", oracle_schedule)
    source = costs.metadata.get("source", "unspecified_cost_table")
    interval = (
        "undefined (zero cost)"
        if gap_lower is None
        else (
            f"{gap_lower:.4f}"
            if oracle_info["exact"]
            else f"[{gap_lower:.4f}, {gap_upper:.4f}]"
            if gap_upper
            else f">={gap_lower:.4f}"
        )
    )
    state_count = oracle_info.get("states_evaluated", oracle_info.get("expanded_states"))
    summary = (
        f"# Feedback batching result\n\nCost source: `{source}`.\n\n"
        f"Resident cap: {costs.resident_cap}; sequences: {len(trace.chains)}; tokens: {trace.token_count}.\n\n"
        f"Best {'static' if all_k else 'scanned static'}: K={best_k}, {best:.4f} ms.\n\n"
        f"Oracle ({oracle}): {'exact OPT' if oracle_info['exact'] else 'bounded search'}, "
        f"L={oracle_info['lower_bound_ms']:.4f} ms, U={oracle_info['upper_bound_ms']:.4f} ms "
        f"({oracle_info['stop_reason']}, {state_count} states, {oracle_info['search_seconds']:.3f} s).\n\n"
        f"{'Best static' if all_k else 'Best scanned static'} / OPT: {interval}.\n\n"
        "Core advances all runnable tokens. Boundary flushes all pending tokens and releases successors. "
        "The reported OPT is over these two actions with serial, fixed-context stage costs.\n"
    )
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    if plots:
        plot_report(costs, report, output_dir)
    return report


def plot_report(costs: CostTable, report: dict[str, Any], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source = costs.metadata.get("source", "unspecified_cost_table")
    batches = range(1, costs.resident_cap + 1)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(batches, costs.core_ms[1:], label="Core (one loop)")
    ax.plot(batches, [costs.coda_ms[k] + costs.prelude_ms[k] for k in batches], label="Coda + successor prelude")
    ax.set(xlabel="Effective stage batch", ylabel="Stage cost (ms)", title=f"Fixed resident cap B={costs.resident_cap}")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.text(0.01, 0.01, f"Cost source: {source}", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"stage_costs.{suffix}", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(
        [row["k"] for row in report["static"]], [row["runtime_ms"] for row in report["static"]], "o-", label="Static K"
    )
    oracle = report["oracle"]
    if oracle["exact"]:
        ax.axhline(oracle["upper_bound_ms"], linestyle="--", color="tab:orange", label="Offline OPT (flush-all)")
    else:
        ax.axhspan(
            oracle["lower_bound_ms"], oracle["upper_bound_ms"], alpha=0.15, color="tab:orange", label="OPT bound [L, U]"
        )
        ax.axhline(oracle["upper_bound_ms"], linestyle="--", color="tab:orange", label="Oracle feasible U")
    ax.set(xlabel="Coda threshold K", ylabel="Total serial runtime (ms)", title="Static K versus feedback oracle")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.text(0.01, 0.01, f"Cost source: {source}", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"static_vs_oracle.{suffix}", dpi=160)
    plt.close(fig)
