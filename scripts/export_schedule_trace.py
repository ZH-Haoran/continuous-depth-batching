r"""Render a JSONL schedule trace as a LaTeX table.

Run these commands after the ShareGPT trace jobs to regenerate the appendix tables:

    trace_dir=outputs/ablations/schedule-trace
    table_dir=docs/paper/figs/schedule_trace
    uv run python scripts/export_schedule_trace.py "$trace_dir/schedule-trace_ouro_sharegpt_w64_cdb-norefill.jsonl" \
        --launches 1:20 --title '\textsc{Ouro: no-refill}' \
        --output "$table_dir/schedule_trace_ouro_sharegpt_w64_norefill.tex"
    uv run python scripts/export_schedule_trace.py "$trace_dir/schedule-trace_ouro_sharegpt_w64_cdb-refill_k1.jsonl" \
        --launches 1:20 --title '\textsc{Ouro: refill} ($K{=}1$)' \
        --output "$table_dir/schedule_trace_ouro_sharegpt_w64_refill_k1.tex"
    uv run python scripts/export_schedule_trace.py "$trace_dir/schedule-trace_ouro_sharegpt_w64_cdb-refill_k16.jsonl" \
        --launches 1:20 --title '\textsc{Ouro: refill} ($K{=}16$)' --coda-threshold 16 \
        --output "$table_dir/schedule_trace_ouro_sharegpt_w64_refill_k16.tex"
    uv run python scripts/export_schedule_trace.py "$trace_dir/schedule-trace_huginn_sharegpt_w64_cdb-refill_k1.jsonl" \
        --launches 1:20 --title '\textsc{Huginn: refill} ($K{=}1$)' \
        --output "$table_dir/schedule_trace_huginn_sharegpt_w64_refill_k1.tex"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from looped_cdb.continuous_depth_batching.schedule_trace import QueueSnapshot, ScheduleTrace, TraceEvent

STAGE_LABELS = {"prefill": "prefill", "prelude": "prelude", "recurrent": "loop", "coda": "coda"}
INITIAL_STAGE_LABEL = "init"
SITE_LABELS = {
    "after_prefill": "after prefill",
    "after_prefill_eager": "after prefill",
    "after_coda": "after coda",
    "after_coda_eager": "after coda",
    "after_coda_staged": "after coda",
    "resumed": "resumed",
    "wave_cohort": "",
}
HEAD_COLUMNS = ("", "Stage", "$B$", "Detail")
TAIL_COLUMNS = ("$Q_P$",)
DEFAULT_DETAIL_WIDTH = "84pt"
REFILL_QUEUE_COLUMNS = ("$Q_R$", "$Q_C$")
WAVE_QUEUE_COLUMNS = ("$Q_D$",)


def parse_launch_window(spec: str) -> tuple[int, int]:
    """Parse ``A:B`` (1-based, inclusive) into a launch window."""

    first_text, _, last_text = spec.partition(":")
    first, last = int(first_text), int(last_text)
    if first < 1 or last < first:
        raise ValueError(f"--launches must be A:B with 1 <= A <= B, got {spec!r}")
    return first, last


def steady_state_window(events: list[TraceEvent], *, length: int) -> tuple[int, int]:
    """Start at the first launch marked steady."""

    if length < 1:
        raise ValueError(f"--length must be positive, got {length}")
    anchor = next((event for event in events if event.steady), None)
    if anchor is None:
        raise ValueError(
            "trace holds no steady-state window: the run never filled the resident set with prompts "
            "still waiting. Trace a longer run, or pass --launches to pick the window by hand."
        )
    first = next(event.launch for event in events if event.tick == anchor.tick and event.steady)
    return first, first + length - 1


def detail_of(event: TraceEvent, *, coda_threshold: int | None = None) -> str:
    """Format the detail column for one stage."""

    if event.stage == "recurrent":
        assert event.depths is not None
        # Internal steps start at zero; the paper labels the first step r_1.
        mix = " ".join(f"$\\mathrm{{r}}_{{{step + 1}}}{{:}}{count}$" for step, count in sorted(event.depths.items()))
        if not event.exited:
            return mix
        return f"{mix}; {event.exited} exit" + ("s" if event.exited > 1 else "")
    if event.stage == "prelude":
        return SITE_LABELS.get(event.site or "", event.site or "")
    if event.stage == "prefill":
        return f"{event.tokens} prompt tokens"
    if event.stage == "coda" and coda_threshold is not None and event.size >= coda_threshold:
        return f"$\\lvert Q_C\\rvert\\ge {coda_threshold}$"
    return ""


def size_of(event: TraceEvent) -> str:
    """Highlight recurrent batch sizes in the table."""

    if event.stage == "recurrent":
        return f"\\tracebatch{{{event.size}}}"
    return str(event.size)


def initial_queues(events: list[TraceEvent]) -> QueueSnapshot:
    """Recover the initial prompt count from the first prefill."""

    first = events[0]
    if first.launch != 1 or first.stage != "prefill":
        raise ValueError(f"trace does not start at a prefill launch (got {first.stage} at launch {first.launch})")
    return QueueSnapshot(
        waiting=first.queues.waiting + first.size, recurrent=0, coda=0, decode=0, coda_in_flight=0, active=0
    )


def is_refill_trace(events: list[TraceEvent]) -> bool:
    """Identify the refill loop by its scheduler queues."""

    return any(event.queues.recurrent or event.queues.coda for event in events)


def render_row(launch: str, stage: str, size: str, detail: str, queues: QueueSnapshot, *, refill: bool) -> str:
    cells = [launch, stage, size, detail]
    cells += [str(queues.recurrent), str(queues.coda)] if refill else [str(queues.decode)]
    cells.append(str(queues.waiting))
    return " & ".join(cells) + r" \\"


def render_rows(events: list[TraceEvent], *, refill: bool, coda_threshold: int | None) -> list[str]:
    rows = []
    for event in events:
        row = render_row(
            str(event.launch),
            STAGE_LABELS[event.stage],
            size_of(event),
            detail_of(event, coda_threshold=coda_threshold),
            event.queues,
            refill=refill,
        )
        rows.append(row)
    return rows


def render_table(
    events: list[TraceEvent],
    *,
    source: Path,
    first: int,
    last: int,
    detail_width: str = DEFAULT_DETAIL_WIDTH,
    coda_threshold: int | None = None,
    title: str | None = None,
) -> str:
    """Render the launches in ``[first, last]`` as a complete ``tabular``."""

    if coda_threshold is not None and coda_threshold < 1:
        raise ValueError(f"coda_threshold must be positive, got {coda_threshold}")
    if first < 1 or last < first:
        raise ValueError(f"invalid launch window {first}:{last}")
    window = [event for event in events if first <= event.launch <= last]
    if len(window) != last - first + 1 or any(event.launch != launch for launch, event in enumerate(window, first)):
        raise ValueError(f"requested launch window {first}:{last} is incomplete")
    refill = is_refill_trace(events)
    columns = (*HEAD_COLUMNS, *(REFILL_QUEUE_COLUMNS if refill else WAVE_QUEUE_COLUMNS), *TAIL_COLUMNS)
    detail_column = ">{\\raggedright\\arraybackslash}p{" + detail_width + "}"
    header = [
        f"% Generated by scripts/export_schedule_trace.py from {source.name} (launches {first}-{last} of {len(events)}).",
        "\\begin{tabular}{" + "rlr" + detail_column + "r" * (len(columns) - len(HEAD_COLUMNS)) + "}",
        r"  \toprule",
    ]
    if title is not None:
        header.append(f"  \\multicolumn{{{len(columns)}}}{{c}}{{{title}}} \\\\")
    header += ["  " + " & ".join(columns) + r" \\", r"  \midrule"]
    rows = render_rows(window, refill=refill, coda_threshold=coda_threshold)
    if first == 1:
        rows.insert(0, render_row("0", INITIAL_STAGE_LABEL, "", "", initial_queues(events), refill=refill))
    body = ["  " + row for row in rows]
    footer = [r"  \bottomrule", r"\end{tabular}"]
    return "\n".join(header + body + footer) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace", type=Path, help="Schedule trace JSONL from benchmark_throughput.py --trace-output.")
    window = parser.add_mutually_exclusive_group()
    window.add_argument(
        "--launches",
        type=parse_launch_window,
        help="Explicit launch window A:B (1-based, inclusive). Defaults to the run's first --length launches.",
    )
    window.add_argument(
        "--from-steady-state",
        action="store_true",
        help="Open the window with the run's steady-state window instead of at the start of the run.",
    )
    parser.add_argument("--length", type=int, default=20, help="Launches in the window when it is not given as A:B.")
    parser.add_argument(
        "--detail-width",
        default=DEFAULT_DETAIL_WIDTH,
        help=f"Width of the wrapped detail column, as a LaTeX length (default {DEFAULT_DETAIL_WIDTH}).",
    )
    parser.add_argument(
        "--coda-threshold",
        type=int,
        help="Annotate coda launches whose incoming queue meets this minimum size.",
    )
    parser.add_argument(
        "--title",
        help="Optional LaTeX title spanning all table columns.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Destination .tex file (a full tabular).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = ScheduleTrace.load_jsonl(args.trace)
    if args.launches is not None:
        first, last = args.launches
    elif args.from_steady_state:
        first, last = steady_state_window(events, length=args.length)
    else:
        first, last = 1, args.length
    table = render_table(
        events,
        source=args.trace,
        first=first,
        last=last,
        detail_width=args.detail_width,
        coda_threshold=args.coda_threshold,
        title=args.title,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table, encoding="utf-8")
    rows = sum(1 for line in table.splitlines() if line.strip().endswith(r"\\") and not line.strip().startswith("&"))
    print(f"Wrote {args.output} ({rows} rows)")


if __name__ == "__main__":
    main()
