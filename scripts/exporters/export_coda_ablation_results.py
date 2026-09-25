"""Export the coda-batching ablations into a JSON file for the paper plots.

Reads the minimum-coda and layer-split summary JSONL files written by the paper experiment
scripts. It reports throughput, speedup, and stage batch sizes for each curve.
Explicit CB calibration inputs provide the full-depth throughput baselines used for normalization.

The sweeps are produced by ``shells/paper/ablations/min_coda_batch.sh`` and
``shells/paper/ablations/layer_split_huginn.sh``.

Example:
    uv run python scripts/exporters/export_coda_ablation_results.py \
        outputs/ablations/min-coda-batch/min-coda-batch_ouro_sharegpt.jsonl \
        outputs/ablations/min-coda-batch/min-coda-batch_huginn_sharegpt.jsonl \
        outputs/ablations/layer-split/layer-split_huginn_sharegpt.jsonl \
        --calibration-baselines \
        outputs/ablations/cb-calibration/cb_ouro_sharegpt.jsonl \
        outputs/ablations/cb-calibration/cb_huginn_sharegpt.jsonl \
        --output docs/paper/figs/ablations/coda_ablations.json
"""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

from looped_cdb.benchmarks.exports import export_meta, write_export
from looped_cdb.benchmarks.summaries import aggregate_summaries, load_summaries


def build_results(
    summary_paths: list[Path],
    repo_root: Path,
    baseline_paths: list[Path] | None = None,
    calibration_paths: list[Path] | None = None,
) -> dict[str, Any]:
    """Export throughput normalizations and stage batch sizes for each coda sweep."""
    groups: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    sources: dict[tuple[str, str, int, str], set[str]] = defaultdict(set)
    hashes = {}
    baselines = []
    calibration_baselines = []
    all_paths = [*summary_paths, *(baseline_paths or []), *(calibration_paths or [])]
    for path in all_paths:
        name = path.resolve().relative_to(repo_root.resolve()).as_posix()
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = load_summaries(path)
        if path in (calibration_paths or []):
            if any(row["config"]["backend"] != "cb" for row in rows):
                raise ValueError(f"Calibration input {path} must contain only CB rows")
            calibration_baselines.extend((row, name) for row in rows)
        else:
            baselines.extend((row, name) for row in rows if row["config"]["backend"] == "cb")
    for path in summary_paths:
        name = path.resolve().relative_to(repo_root.resolve()).as_posix()
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for row in load_summaries(path):
            config = row["config"]
            if config["backend"] != "cdb" or not config["refill"]:
                continue
            key = (
                config["model"],
                config.get("layer_split") or "default",
                config["max_num_seqs"],
                config["workload_name"],
            )
            groups[key].append(row)
            sources[key].add(name)
    if not groups:
        raise ValueError("No refill rows found")
    panels = []
    for key, raw in sorted(groups.items()):
        model, split, width, workload = key
        configs = [
            {k: v for k, v in row["config"].items() if k not in {"min_coda_batch_size", "measured_repeat"}}
            for row in raw
        ]
        if any(config != configs[0] for config in configs[1:]):
            raise ValueError(f"Mixed configuration within {key}; only K and repeat may vary")
        schedules = [row.get("requested_exit_distribution") for row in raw]
        if any(schedule != schedules[0] for schedule in schedules[1:]):
            raise ValueError(f"Mixed exit schedules within {key}")
        identities = [(row["config"]["min_coda_batch_size"], row["config"].get("measured_repeat", 0)) for row in raw]
        if len(set(identities)) != len(identities):
            raise ValueError(f"Duplicate K/repeat within {key}")
        rows = sorted(aggregate_summaries(raw), key=lambda row: row.min_coda_batch_size)
        if rows[0].min_coda_batch_size != 1:
            raise ValueError(f"Missing K=1 baseline for {key}")
        if any(row.gen_tps_mean <= 0 for row in rows):
            raise ValueError(f"Nonpositive throughput for {key}")
        stats_by_k = {
            row.min_coda_batch_size: [
                r["backend_stats"] for r in raw if r["config"]["min_coda_batch_size"] == row.min_coda_batch_size
            ]
            for row in rows
        }
        batch_sizes = {}
        for stage, numerator, denominator in [
            ("recurrent", "recurrent_steps", "recurrent_batches"),
            ("coda", "coda_tokens", "coda_batches"),
        ]:
            values = []
            for row in rows:
                stats = stats_by_k[row.min_coda_batch_size]
                count = sum(s[denominator] for s in stats)
                if count <= 0:
                    raise ValueError(f"No {stage} launches for {key}")
                values.append(sum(s[numerator] for s in stats) / count)
            batch_sizes[f"{stage}_batch_size"] = values
        config = configs[0].copy()
        workload_path = Path(config["workload_path"])
        if workload_path.is_absolute():
            config["workload_path"] = workload_path.resolve().relative_to(repo_root.resolve()).as_posix()
        # CB runs full depth, so its exit threshold and refill/coda controls do not affect execution.
        ignored = {"backend", "refill", "min_coda_batch_size", "measured_repeat", "exit_threshold"}
        reference_config = {k: v for k, v in configs[0].items() if k not in ignored}
        matching = [
            (row, name)
            for row, name in baselines
            if {k: v for k, v in row["config"].items() if k not in ignored} == reference_config
        ]
        baseline_match = "exact"
        if calibration_paths:
            calibration_ignored = ignored | {"replay_eos_finishes"}
            calibration_reference = {k: v for k, v in configs[0].items() if k not in calibration_ignored}
            calibration_matching = [
                (row, name)
                for row, name in calibration_baselines
                if {k: v for k, v in row["config"].items() if k not in calibration_ignored} == calibration_reference
            ]
            if not calibration_matching:
                raise ValueError(f"Missing explicit calibration baseline for {key}")
            matching = calibration_matching
            baseline_match = "calibration_input"
        repeat_ids = [row["config"].get("measured_repeat", 0) for row, _ in matching]
        if len(set(repeat_ids)) != len(repeat_ids):
            raise ValueError(f"Duplicate CB baseline repeat for {key}; select one baseline campaign")
        cb = aggregate_summaries([row for row, _ in matching]) if matching else []
        if len(cb) > 1:
            raise ValueError(f"Ambiguous CB baseline for {key}")
        cb_tps = cb[0].gen_tps_mean if cb else None
        if cb_tps is not None and cb_tps <= 0:
            raise ValueError(f"Nonpositive CB throughput for {key}")
        panels.append(
            {
                "model": model,
                "layer_split": split,
                "width": width,
                "name": workload,
                "threshold": config["exit_threshold"],
                "num_requests": config["num_requests"],
                "config": config,
                "source_summaries": sorted(sources[key]),
                "requested_exit_distribution": schedules[0],
                "k_values": [r.min_coda_batch_size for r in rows],
                "gen_tps": [r.gen_tps_mean for r in rows],
                "gen_tps_std": [r.gen_tps_std for r in rows],
                "cb_gen_tps": cb_tps,
                "cb_source_summaries": sorted({name for _, name in matching}),
                "cb_num_requests": sorted({row["config"]["num_requests"] for row, _ in matching}),
                "cb_match": baseline_match if matching else None,
                "speedup_over_cb": [r.gen_tps_mean / cb_tps for r in rows] if cb_tps else None,
                "speedup_over_immediate": [r.gen_tps_mean / rows[0].gen_tps_mean for r in rows],
                "repeats": [r.repeats for r in rows],
                **batch_sizes,
            }
        )
    return {
        "meta": {
            **export_meta(all_paths),
            "source_sha256": hashes,
            "throughput_basis": "Generated tokens divided by whole-run wall time; arithmetic mean over repeats.",
            "batch_size_basis": "Active token-steps or coda tokens divided by stage launches, pooled over repeats.",
            "configuration_scope": "Settings are held fixed within each curve; inspect panel configs before comparing curves.",
        },
        "panels": panels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--baselines", nargs="+", type=Path, default=[])
    parser.add_argument(
        "--calibration-baselines",
        nargs="+",
        type=Path,
        default=[],
        help="Explicit CB-only calibration files; permit replay_eos_finishes to differ from the coda sweep.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_export(
        build_results(
            args.summaries,
            Path(__file__).resolve().parents[2],
            args.baselines,
            args.calibration_baselines,
        ),
        args.output,
    )


if __name__ == "__main__":
    main()
