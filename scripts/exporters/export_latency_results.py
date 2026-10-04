"""Create one CSV comparison table from per-run serving event files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from looped_cdb.benchmarks.latency_events import comparison_row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events_dir", type=Path, help="Directory written by --latency-events-dir")
    parser.add_argument("--output", type=Path, required=True, help="Comparison CSV destination")
    args = parser.parse_args()
    paths = sorted(args.events_dir.glob("*.events.jsonl"))
    if not paths:
        raise ValueError(f"no event files in {args.events_dir}")
    rows = [comparison_row(path) for path in paths]
    rows.sort(
        key=lambda row: (
            row["workload"], row["rate_rps"] or 0, row["threshold"] or 0, row["mode"], row["repeat"]
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} runs to {args.output}")


if __name__ == "__main__":
    main()
