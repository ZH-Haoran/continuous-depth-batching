"""Export an exit-threshold sweep into a JSON file for the paper plots.

Reads the raw summary JSONL written by ``scripts/benchmark_throughput.py`` and normalizes CDB
throughput against the full-depth CB row, one panel per workload. It also computes the
theoretical maximum decode speedup (FLOP bound) on a dense threshold grid. When full-window CB
Nsight analyses are supplied, it computes the e2e bound by adjusting the speedup for the time spent
on prefill.

The sweep is produced by ``shells/paper/throughput_exit-sweep.sh``.

Example:
    uv run python scripts/exporters/export_exit_sweep_results.py \
        outputs/exit-sweep/exit-sweep_ouro_alpaca.jsonl \
        outputs/exit-sweep/exit-sweep_ouro_sharegpt.jsonl \
        outputs/exit-sweep/exit-sweep_ouro_arxiv.jsonl \
        --cb-profile-root outputs/nsys \
        --output docs/paper/figs/throughput/exit_sweep_ouro.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from looped_cdb.benchmarks.exports import (
    CbProfileKey,
    CbStageProfile,
    agreed_config_value,
    bundle_fingerprint,
    export_meta,
    first_output_token_fraction,
    load_cb_stage_profiles,
    prefill_adjusted_speedup,
    recorded_stage_flops,
    replayed_workload,
    require_cb_stage_profile,
    warn,
    write_export,
)
from looped_cdb.benchmarks.flop_bound import StageFlops
from looped_cdb.benchmarks.summaries import (
    AggregateRow,
    aggregate_summaries,
    baseline_row,
    load_summaries,
    threshold_of,
)

# Spacing of the FLOP-bound threshold grid. Fine enough that the plotted bound reads as a
# smooth curve; the measured thresholds are always included exactly on top of the grid.
BOUND_CURVE_STEP = 0.01


def bound_curve(
    raw_rows: list[dict[str, Any]],
    rows: list[AggregateRow],
    workload_path: Path,
    flops: StageFlops,
) -> dict[str, Any] | None:
    """Computes the decode-step FLOP bound of the recorded exit schedule on a dense threshold grid.

    The curve rebuilds the exact schedule the runs replayed (bundle, model-length truncation,
    request subset, exit policy) and must reproduce every measured row's recorded bound. Returns
    ``None`` when there is nothing to draw: no thresholded rows, a bundle that is not on this
    machine, or a synthetic depth schedule, which is threshold-independent.
    """

    measured: dict[float, list[float]] = {}
    for row in rows:
        threshold = threshold_of(row)
        if threshold is not None and row.flop_bound_speedup:
            measured.setdefault(threshold, []).append(row.flop_bound_speedup)
    if not measured or not workload_path.is_file():
        return None

    workload = replayed_workload(raw_rows, workload_path)
    if workload.exit_depths is not None:
        return None
    min_exit_step = agreed_config_value(raw_rows, "min_exit_step", None) or 1
    exit_delay_steps = agreed_config_value(raw_rows, "exit_delay_steps", None) or 0
    max_depth = agreed_config_value(raw_rows, "max_recurrent_depth", None)

    lo, hi = min(measured), max(measured)
    ticks = {
        round(step * BOUND_CURVE_STEP, 4)
        for step in range(round(lo / BOUND_CURVE_STEP), round(hi / BOUND_CURVE_STEP) + 1)
    }
    thresholds = sorted(ticks | set(measured))
    bound = {
        q: flops.ideal_speedup(
            max_depth=max_depth,
            mean_depth=workload.mean_requested_depth_at(
                threshold=q, min_exit_step=min_exit_step, exit_delay_steps=exit_delay_steps
            ),
        )
        for q in thresholds
    }
    for q, recorded_values in measured.items():
        for recorded in recorded_values:
            if abs(bound[q] - recorded) > 1e-6 * recorded:
                raise ValueError(
                    f"bound curve disagrees with the recorded FLOP bound at threshold {q} "
                    f"({bound[q]:.6f} vs {recorded:.6f}). The bundle at {workload_path}, the exit flags "
                    "or the layer split no longer match the published runs; re-measure or restore the "
                    "recorded bundle."
                )
    return {"thresholds": thresholds, "flop_bound": [round(bound[q], 6) for q in thresholds]}


def _point(row: AggregateRow, baseline_tps: float, baseline_steady_tps: float | None) -> dict[str, Any]:
    threshold = threshold_of(row)
    steady = row.steady_gen_tps_mean
    return {
        "backend": row.backend,
        "refill": row.refill,
        "threshold": threshold,
        "depth": row.depth,
        "gen_tps": row.gen_tps_mean,
        "gen_tps_std": row.gen_tps_std,
        "speedup": row.gen_tps_mean / baseline_tps,
        # Steady-state throughput, with the fill and drain cut away (``None`` where a row holds none).
        "steady_gen_tps": steady,
        "steady_gen_tps_std": row.steady_gen_tps_std,
        "steady_speedup": None if steady is None or baseline_steady_tps is None else steady / baseline_steady_tps,
        "flop_bound": row.flop_bound_speedup,
        "wall_s": row.wall_s_mean,
        "preemptions": row.preemptions_mean,
        "prefill_batches": row.prefill_batches_mean,
        "prefill_batch_size": row.prefill_batch_size_mean,
        "decode_batches": row.decode_batches_mean,
        "decode_batch_size": row.decode_batch_size_mean,
        "recurrent_batch_size": row.recurrent_batch_size_mean,
        "coda_batch_size": row.coda_batch_size_mean,
        "resident_requests": row.resident_requests_mean,
        "repeats": row.repeats,
    }


def _sort_key(point: dict[str, Any]) -> tuple[int, bool, float]:
    # cb baseline first, then CDB without refill, then CDB, each by ascending threshold.
    is_cdb = point["backend"] == "cdb"
    return (int(is_cdb), bool(point["refill"]), point["threshold"] or 0.0)


def build_workload_entry(
    summary_path: Path,
    repo_root: Path,
    *,
    cb_profiles: dict[CbProfileKey, CbStageProfile] | None = None,
) -> dict[str, Any]:
    """Aggregate one summary file into a single publishable workload panel."""

    raw = load_summaries(summary_path)
    rows = aggregate_summaries(raw)
    if not rows:
        raise ValueError(f"no summaries found in {summary_path}")

    names = {row.workload_name for row in rows}
    if len(names) != 1:
        raise ValueError(f"{summary_path} mixes workloads {sorted(names)}; export one workload per file")
    sizes = {row.num_requests for row in rows}
    if len(sizes) != 1:
        raise ValueError(
            f"{summary_path} mixes request counts {sorted(sizes)}. A subsampled run and a full run are not "
            "comparable (the fixed end-of-run drain cost is a larger fraction of a short run), so they must "
            "not share a panel."
        )

    baseline = baseline_row(rows)
    points = sorted((_point(row, baseline.gen_tps_mean, baseline.steady_gen_tps_mean) for row in rows), key=_sort_key)

    workload_path = repo_root / str(raw[0]["config"]["workload_path"])
    fingerprint = bundle_fingerprint(workload_path, summary_path)
    num_requests = baseline.num_requests
    full_size = fingerprint.pop("full_num_requests")
    flops = recorded_stage_flops(raw)

    entry = {
        "name": baseline.workload_name,
        "num_requests": num_requests,
        "full_num_requests": full_size,
        "is_subsample": None if full_size is None else num_requests < full_size,
        "source_summary": str(summary_path),
        "baseline_gen_tps": baseline.gen_tps_mean,
        **fingerprint,
        "stage_flops": flops.to_json_dict(),
        "points": points,
    }
    curve = bound_curve(raw, rows, workload_path, flops)
    if curve is not None:
        entry["bound_curve"] = curve
    if cb_profiles is not None:
        first_fraction = first_output_token_fraction(raw)
        model = str(raw[0]["config"]["model"])
        width = int(agreed_config_value(raw, "max_num_seqs", None))
        profile = require_cb_stage_profile(
            cb_profiles,
            model=model,
            workload=baseline.workload_name,
            width=width,
            num_requests=num_requests,
        )
        entry["bound_timing"] = profile.to_json_dict()
        if curve is not None:
            curve["e2e_bound"] = [
                round(
                    prefill_adjusted_speedup(value, profile.prefill_fraction, first_output_fraction=first_fraction), 6
                )
                for value in curve["flop_bound"]
            ]
        for point in points:
            point["e2e_bound"] = (
                None
                if point["flop_bound"] is None
                else prefill_adjusted_speedup(
                    point["flop_bound"], profile.prefill_fraction, first_output_fraction=first_fraction
                )
            )
    return entry


def build_results(
    summary_paths: list[Path],
    repo_root: Path,
    *,
    cb_profiles: dict[CbProfileKey, CbStageProfile] | None = None,
) -> dict[str, Any]:
    """One panel per summary file, each a workload."""

    return {
        "meta": export_meta(summary_paths),
        "workloads": [build_workload_entry(path, repo_root, cb_profiles=cb_profiles) for path in summary_paths],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summaries", type=Path, nargs="+", help="Benchmark summary JSONL files, one per workload.")
    parser.add_argument(
        "--cb-profile-root",
        type=Path,
        help=(
            "Directory containing full-window CB Nsight analyses used for prefill adjustment. "
            "When omitted, the export contains only the decode FLOP bound."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/paper/figs/throughput/exit_sweep_ouro.json"),
        help="Destination JSON, committed alongside the paper's plotting script.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = None if args.cb_profile_root is None else load_cb_stage_profiles(args.cb_profile_root)
    results = build_results(
        args.summaries,
        Path(__file__).resolve().parents[2],
        cb_profiles=profiles,
    )
    write_export(results, args.output)

    for workload in results["workloads"]:
        flag = " (subsample)" if workload["is_subsample"] else ""
        bundle = workload["workload_sha256"] or "unverifiable"
        print(
            f"  {workload['name']}: {workload['num_requests']} requests{flag}, "
            f"{len(workload['points'])} points, bundle {bundle}"
        )
        if not workload["fingerprint_trusted"]:
            warn(
                f"{workload['name']}: the workload bundle is newer than its summary, so the bundle that was "
                "replayed is gone. Re-run this workload for a verifiable fingerprint."
            )


if __name__ == "__main__":
    main()
