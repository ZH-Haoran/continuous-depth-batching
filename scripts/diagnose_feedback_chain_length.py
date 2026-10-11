"""Compare static K on the saved short chains and the same chains extended to 64 tokens."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

from looped_cdb.benchmarks.feedback import CostTable, RecurrenceTrace, generate_trace
from looped_cdb.benchmarks.feedback_report import load_trace, run_experiment, write_json


def extend_chains(trace: RecurrenceTrace, target_length: int = 64) -> RecurrenceTrace:
    """Keep every original demand, appending independent geometric(mean=4) draws."""
    lengths = {len(chain) for chain in trace.chains}
    if len(lengths) != 1:
        raise ValueError("chain-length diagnostic requires equally sized base chains")
    base_length = lengths.pop()
    if target_length <= base_length:
        raise ValueError("target length must exceed the saved chain length")
    extension = generate_trace(
        sequences=len(trace.chains),
        tokens_per_sequence=target_length - base_length,
        distribution="geometric",
        mean_depth=4,
        max_depth=32,
        seed=1,
    )
    return RecurrenceTrace(
        tuple(base + extra for base, extra in zip(trace.chains, extension.chains)),
        metadata={
            "source": "saved_trace_with_geometric_extension",
            "base_tokens_per_sequence": base_length,
            "target_tokens_per_sequence": target_length,
            "base_metadata": trace.metadata,
            "extension_metadata": extension.metadata,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--costs", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True, help="the saved 64-sequence, 4-token trace")
    parser.add_argument("--output-dir", type=Path, default=Path("results/feedback/chain_length"))
    args = parser.parse_args()
    costs = CostTable.from_dict(json.loads(args.costs.read_text(encoding="utf-8")))
    short = load_trace(args.trace)
    long = extend_chains(short)
    thresholds = tuple(range(1, costs.resident_cap + 1))
    comparison = []
    for name, trace in (("short", short), ("long", long)):
        started = perf_counter()
        report = run_experiment(
            costs,
            trace,
            args.output_dir / name,
            thresholds=thresholds,
            oracle="none",
            progress_callback=lambda message: print(message, file=sys.stderr, flush=True),
        )
        row = {
            "case": name,
            "tokens_per_sequence": len(trace.chains[0]),
            "best_static": report["best_static"],
            "k16_comparison": report["k16_comparison"],
            "best_ms_per_token": report["best_static"]["runtime_ms"] / trace.token_count,
            "scan_seconds": perf_counter() - started,
        }
        comparison.append(row)
        print(json.dumps(row, indent=2), flush=True)
    write_json(args.output_dir / "comparison.json", comparison)
    print(f"wrote {args.output_dir / 'comparison.json'}", flush=True)


if __name__ == "__main__":
    main()
