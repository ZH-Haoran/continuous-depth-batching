"""Export a maximum decode batch size sweep for the paper plots.

Reads the raw summary JSONL written by ``scripts/benchmark_throughput.py`` and normalizes CDB
throughput against the full-depth CB row, one panel per workload. The exit threshold is fixed
while the maximum decode batch size varies. It also computes the theoretical maximum decode
speedup (FLOP bound). When full-window CB Nsight analyses are supplied, it computes the e2e bound
by adjusting the speedup for the fraction of time spent on prefill.
When a decode-step latency export is supplied, its fitted balance batch ``B*`` is recorded.

The sweep is produced by ``shells/paper/throughput_width-sweep.sh``.
The CB Nsight analyses are produced by ``shells/paper/ablations/profile_nsys.sh``.
The decode-step latency results are produced by ``shells/paper/benchmark_decode_step_latency.sh``.

Example:
    uv run python scripts/exporters/export_width_sweep_results.py \
        outputs/width-sweep/width-sweep_ouro_alpaca.jsonl \
        outputs/width-sweep/width-sweep_ouro_sharegpt.jsonl \
        outputs/width-sweep/width-sweep_ouro_arxiv.jsonl \
        --cb-profile-root outputs/nsys \
        --b-star-from docs/paper/figs/decode_latency/decode_step_latency_ouro.json \
        --output docs/paper/figs/throughput/width_sweep_ouro.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
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
from looped_cdb.benchmarks.summaries import AggregateRow, aggregate_summaries, load_summaries, threshold_of

# Series in a maximum decode batch size panel: baseline and two depth policies.
WIDTH_SERIES = ("cb", "norefill", "refill")


def _model_key(name: str) -> str:
    """A model identifier reduced to what survives being written down two different ways."""

    return re.sub(r"[^a-z0-9]", "", name.lower())


def b_star_at(
    decode_latency_path: Path,
    mean_sequence_length: float,
    *,
    models: set[str],
    max_depth: int | None,
) -> dict[str, Any]:
    """Read the fitted balance batch nearest this workload's mean sequence length.

    The sweep's context lengths are geometric, so the closest context is chosen on a log scale.
    Model names are compared heuristically after removing separators and case.
    """

    export = json.loads(decode_latency_path.read_text(encoding="utf-8"))
    if max_depth is not None and export["max_depth"] != max_depth:
        raise ValueError(
            f"{decode_latency_path.name} was measured at max depth {export['max_depth']}, this panel at "
            f"{max_depth}; its B* belongs to a different recurrent-depth budget."
        )
    fitted = _model_key(str(export["model_id"]))
    if models and not any(fitted in _model_key(name) or _model_key(name) in fitted for name in models):
        raise ValueError(
            f"{decode_latency_path.name} was measured on {export['model_id']}, this panel on "
            f"{sorted(models)}; its B* belongs to a different model."
        )

    fit = min(export["contexts"], key=lambda fit: abs(math.log(fit["context_length"] / mean_sequence_length)))
    # The source export warns about this too, but a reader of the width panel never sees that run.
    if fit["b_star"] > fit["batch_range"][1]:
        warn(
            f"B* {fit['b_star']:.0f} at L={fit['context_length']} lies beyond the largest batch measured "
            f"there ({fit['batch_range'][1]}), so the marker is extrapolated from the fit, not observed"
        )
    return {
        "b_star": fit["b_star"],
        "b_star_approx": round(fit["b_star"]),
        "b_star_basis": (
            f"fitted B* = floor / per_seq at L = {fit['context_length']} in {decode_latency_path.name}, "
            f"the closest measured context to this workload (mean sequence length "
            f"{round(mean_sequence_length)})"
        ),
    }


def width_series(rows: list[AggregateRow]) -> tuple[dict[str, dict[int, AggregateRow]], dict[int, int]]:
    """Sort rows by series and maximum decode batch size.

    Also returns the minimum coda batch at each width. It may follow a width-dependent policy such
    as ``W / 4``; multiple values at one width are a coda-batch ablation and remain invalid here.
    """

    coda_batches: dict[int, set[int]] = {}
    for row in rows:
        if row.backend == "cdb" and row.refill:
            coda_batches.setdefault(row.max_num_seqs, set()).add(row.min_coda_batch_size)
    if not coda_batches:
        raise ValueError("a maximum decode batch size panel needs refill rows; none of the summaries hold any")
    ambiguous = {width: sorted(values) for width, values in coda_batches.items() if len(values) > 1}
    if ambiguous:
        raise ValueError(
            f"refill rows were measured with multiple minimum coda batches at the same width: {ambiguous}. "
            "A sweep over the minimum coda batch is an ablation with its own export."
        )

    series: dict[str, dict[int, AggregateRow]] = {name: {} for name in WIDTH_SERIES}
    for row in rows:
        name = "cb" if row.backend == "cb" else ("refill" if row.refill else "norefill")
        series[name][row.max_num_seqs] = row
    return series, {width: values.pop() for width, values in coda_batches.items()}


def build_workload_entry(
    workload_name: str,
    raw: list[dict[str, Any]],
    summary_paths: list[Path],
    repo_root: Path,
    *,
    b_star_source: Path | None = None,
    cb_profiles: dict[CbProfileKey, CbStageProfile] | None = None,
) -> dict[str, Any]:
    """Aggregate one workload's maximum decode batch size rows."""

    rows = aggregate_summaries(raw)
    sizes = {row.num_requests for row in rows}
    if len(sizes) != 1:
        raise ValueError(
            f"{workload_name} mixes request counts {sorted(sizes)}. A subsampled run and a full run are not "
            "comparable (the fixed end-of-run drain cost is a larger fraction of a short run), so they must "
            "not share a panel."
        )
    thresholds = {threshold_of(row) for row in rows if not row.is_baseline}
    if len(thresholds) != 1:
        raise ValueError(
            f"a maximum decode batch size panel holds the exit threshold fixed, but {workload_name} "
            f"sweeps {sorted(thresholds)}"
        )
    # The bound describes the exit schedule, so the full-depth baseline has none to contribute.
    bounds = {row.flop_bound_speedup for row in rows if not row.is_baseline and row.flop_bound_speedup is not None}
    if not bounds:
        raise ValueError(f"no {workload_name} row recorded a FLOP bound, so the panel has no ceiling to draw")
    if max(bounds) - min(bounds) > 1e-6 * max(bounds):
        raise ValueError(f"{workload_name} rows at one threshold recorded different FLOP bounds {sorted(bounds)}")

    series, served_coda_batches = width_series(rows)
    widths = sorted({width for by_width in series.values() for width in by_width})
    baseline = series["cb"]
    measured: dict[str, Any] = {"widths": widths}
    for name in WIDTH_SERIES:
        by_width = series[name]
        # A gap would silently shorten one curve against the others, and the crossover the panel
        # exists to show is exactly where two curves are compared width by width.
        missing = [width for width in widths if width not in by_width]
        if missing:
            raise ValueError(f"the {name} series was not measured at width(s) {missing}; the panel has gaps")
        measured[name] = [round(by_width[width].gen_tps_mean, 1) for width in widths]
        for stage, attribute in (("recurrent", "recurrent_batch_size_mean"), ("coda", "coda_batch_size_mean")):
            measured[f"{name}_{stage}_batch_size"] = [
                None if (value := getattr(by_width[width], attribute)) is None else round(value, 1) for width in widths
            ]
        # Residency against the width each point was swept at, which is what says whether admission
        # kept the batch at the width the panel labels it with.
        measured[f"{name}_resident_requests"] = [
            None if (value := by_width[width].resident_requests_mean) is None else round(value, 1) for width in widths
        ]
        # Steady-state throughput, with the fill and drain cut away (``None`` where a row holds none).
        measured[f"{name}_steady"] = [
            None if (value := by_width[width].steady_gen_tps_mean) is None else round(value, 1) for width in widths
        ]
        if name != "cb":
            measured[f"{name}_over_cb"] = [
                round(by_width[width].gen_tps_mean / baseline[width].gen_tps_mean, 4) for width in widths
            ]
            measured[f"{name}_steady_over_cb"] = [
                None
                if (value := by_width[width].steady_gen_tps_mean) is None
                or (base := baseline[width].steady_gen_tps_mean) is None
                else round(value / base, 4)
                for width in widths
            ]

    workload_path = repo_root / str(raw[0]["config"]["workload_path"])
    # The oldest summary is the conservative check: the bundle has to predate every run it is
    # claimed to be the input of, not just the last one.
    fingerprint = bundle_fingerprint(workload_path, min(summary_paths, key=lambda path: path.stat().st_mtime))
    full_size = fingerprint.pop("full_num_requests")
    num_requests = sizes.pop()
    decode_bound = max(bounds)
    entry: dict[str, Any] = {
        "name": workload_name,
        "num_requests": num_requests,
        "full_num_requests": full_size,
        "is_subsample": None if full_size is None else num_requests < full_size,
        "source_summaries": [str(path) for path in summary_paths],
        "threshold": thresholds.pop(),
        "min_coda_batch_sizes": [served_coda_batches[width] for width in widths],
        **fingerprint,
        "flop_bound": round(decode_bound, 6),
        "stage_flops": recorded_stage_flops(raw).to_json_dict(),
    }
    if cb_profiles is not None:
        first_fraction = first_output_token_fraction(raw)
        model = str(raw[0]["config"]["model"])
        profiles = [
            require_cb_stage_profile(
                cb_profiles,
                model=model,
                workload=workload_name,
                width=width,
                num_requests=num_requests,
            )
            for width in widths
        ]
        entry["bound_timings"] = [profile.to_json_dict() for profile in profiles]
        entry["e2e_bound"] = [
            round(
                prefill_adjusted_speedup(decode_bound, profile.prefill_fraction, first_output_fraction=first_fraction),
                6,
            )
            for profile in profiles
        ]
    if workload_path.is_file():
        workload = replayed_workload(raw, workload_path)
        mean_sequence_length = float((workload.input_lens + workload.output_lens).mean())
        entry["mean_sequence_length"] = round(mean_sequence_length)
        if b_star_source is not None:
            entry.update(
                b_star_at(
                    b_star_source,
                    mean_sequence_length,
                    models={str(row["config"]["model"]) for row in raw},
                    max_depth=agreed_config_value(raw, "max_recurrent_depth", None),
                )
            )
    elif b_star_source is not None:
        raise ValueError(
            f"B* is chosen by the workload's mean sequence length, which is read from the bundle at "
            f"{workload_path}; it is not on this machine, so the panel cannot record one."
        )
    return {**entry, **measured}


def build_results(
    summary_paths: list[Path],
    repo_root: Path,
    *,
    b_star_source: Path | None = None,
    cb_profiles: dict[CbProfileKey, CbStageProfile] | None = None,
) -> dict[str, Any]:
    """One panel per workload, each assembled from every summary that holds rows for it."""

    by_workload: dict[str, tuple[list[dict[str, Any]], list[Path]]] = {}
    for path in summary_paths:
        for row in load_summaries(path):
            name = str(row["config"].get("workload_name", "workload"))
            raw, sources = by_workload.setdefault(name, ([], []))
            raw.append(row)
            if path not in sources:
                sources.append(path)
    if not by_workload:
        raise ValueError(f"no summary rows in {[str(path) for path in summary_paths]}")

    return {
        "meta": export_meta(summary_paths),
        "workloads": [
            build_workload_entry(
                name,
                raw,
                sources,
                repo_root,
                b_star_source=b_star_source,
                cb_profiles=cb_profiles,
            )
            for name, (raw, sources) in sorted(by_workload.items())
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "summaries",
        type=Path,
        nargs="+",
        help="Benchmark summary JSONL files; rows are grouped by the workload they replayed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/paper/figs/throughput/width_sweep_ouro.json"),
        help="Destination JSON, committed alongside the paper's plotting script.",
    )
    parser.add_argument(
        "--cb-profile-root",
        type=Path,
        help=(
            "Directory containing full-window CB Nsight analyses used for prefill adjustment. "
            "When omitted, the export contains only the decode FLOP bound."
        ),
    )
    parser.add_argument(
        "--b-star-from",
        type=Path,
        default=None,
        help=(
            "Decode-step latency export containing a fitted balance batch B*. The fit nearest each "
            "workload's mean sequence length is recorded with its panel."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = None if args.cb_profile_root is None else load_cb_stage_profiles(args.cb_profile_root)
    results = build_results(
        args.summaries,
        Path(__file__).resolve().parents[2],
        b_star_source=args.b_star_from,
        cb_profiles=profiles,
    )
    write_export(results, args.output)

    for workload in results["workloads"]:
        widths = workload["widths"]
        b_star = f", B* {workload['b_star_approx']}" if "b_star_approx" in workload else ""
        print(
            f"  {workload['name']} at q={workload['threshold']:g}: widths {widths[0]}-{widths[-1]}, "
            f"bound {workload['flop_bound']:.3f}{b_star}"
        )
        if not workload["fingerprint_trusted"]:
            warn(
                f"{workload['name']}: the workload bundle is newer than its summary, so the bundle that was "
                "replayed is gone. Re-run this workload for a verifiable fingerprint."
            )


if __name__ == "__main__":
    main()
