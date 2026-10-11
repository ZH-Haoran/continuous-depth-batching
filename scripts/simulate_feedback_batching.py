"""Compare fixed K with a feedback-aware offline oracle using stage-cost JSON."""

import argparse
import json
from pathlib import Path

from looped_cdb.benchmarks.feedback import CostTable, generate_trace
from looped_cdb.benchmarks.feedback_report import load_trace, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--costs", type=Path, required=True)
    parser.add_argument("--trace", type=Path, help="saved finite token chains; replaces trace generation")
    parser.add_argument("--output-dir", type=Path, default=Path("results/feedback/geometric"))
    parser.add_argument("--sequences", type=int, help="defaults to the cost table's resident cap")
    parser.add_argument("--tokens-per-sequence", type=int, default=4)
    parser.add_argument("--mean-depth", type=float, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--distribution", choices=("geometric", "two_point"), default="geometric")
    parser.add_argument("--high-depth", type=int, default=16)
    scan = parser.add_mutually_exclusive_group()
    scan.add_argument("--thresholds", help="comma-separated K values; default 1,4,8,16,32,64 within cap")
    scan.add_argument("--all-k", action="store_true", help="scan every threshold 1..B to obtain best static K")
    parser.add_argument("--oracle-max-states", type=int, default=100_000)
    parser.add_argument("--oracle-time-limit", type=float, default=30)
    parser.add_argument("--no-plots", action="store_true", help="stdlib-only report; skip matplotlib figures")
    args = parser.parse_args()
    costs = CostTable.from_dict(json.loads(args.costs.read_text(encoding="utf-8")))
    trace = (
        load_trace(args.trace)
        if args.trace
        else generate_trace(
            sequences=costs.resident_cap if args.sequences is None else args.sequences,
            tokens_per_sequence=args.tokens_per_sequence,
            mean_depth=args.mean_depth,
            seed=args.seed,
            distribution=args.distribution,
            high_depth=args.high_depth,
        )
    )
    if args.all_k:
        thresholds = tuple(range(1, costs.resident_cap + 1))
    elif args.thresholds:
        thresholds = tuple(int(value) for value in args.thresholds.split(","))
    else:
        thresholds = tuple(k for k in (1, 4, 8, 16, 32, 64) if k <= costs.resident_cap)
    report = run_experiment(
        costs,
        trace,
        args.output_dir,
        thresholds=thresholds,
        max_states=args.oracle_max_states,
        time_limit_s=args.oracle_time_limit,
        plots=not args.no_plots,
    )
    print(
        json.dumps({"best_static": report["best_static"], "oracle": report["oracle"], "gap": report["gap"]}, indent=2)
    )
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
