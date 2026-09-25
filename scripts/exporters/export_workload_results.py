"""Export workload lengths and two-model exit distributions for the paper.

ShareGPT, Alpaca, and ArXiv bundles are required for Ouro and Huginn.
Lengths use Ouro tokenization.
Exit distributions use cumulative gate probabilities for Ouro and delayed convergence exits for Huginn.

Example:
    uv run python scripts/exporters/export_workload_results.py \
        --workload-dir outputs/workloads --out-dir docs/paper/figs/workloads
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from looped_cdb.benchmarks.workload import Workload, depths_from_pdf, depths_from_values

NUM_QUANTILES = 256
MIN_EXIT_STEP = 2
THRESHOLDS = (0.2, 0.4, 0.5, 0.7, 1.0)
WORKLOADS = (("ShareGPT", "sharegpt", "_10k"), ("Alpaca", "alpaca", ""), ("ArXiv", "arxiv", "_6k"))
MODEL_SETTINGS = {
    "ouro": (4, THRESHOLDS, 0),
    "huginn": (16, (0.28, 0.16, 0.1, 0.0), 1),
}


def length_quantiles(lengths: np.ndarray, quantiles: np.ndarray) -> list[float]:
    """Sample the empirical CDF of lengths at the requested quantiles."""
    return [float(v) for v in np.quantile(np.asarray(lengths, dtype=float), quantiles)]


def exit_counts(workload: Workload, threshold: float, *, exit_delay_steps: int = 0) -> list[int]:
    """Count every output token at its derived exit depth."""
    if workload.exit_values is not None:
        derive, values = depths_from_values, workload.exit_values
    elif workload.exit_pdf is not None:
        derive, values = depths_from_pdf, workload.exit_pdf
    else:
        raise ValueError("Workload distributions require threshold-free recorded trajectories")
    depths = derive(values, threshold=threshold, min_exit_step=MIN_EXIT_STEP, exit_delay_steps=exit_delay_steps)
    return np.bincount(depths, minlength=workload.max_depth + 1)[1:].tolist()


def bundle_provenance(workload: Workload) -> dict[str, Any]:
    """Distinguish a recorded sample from its source request pool."""
    return {
        "num_requests": workload.num_requests,
        "sampled_from": workload.meta.get("sampled_from", workload.num_requests),
    }


def build_length_distributions(bundles: list[tuple[str, Workload]], max_model_len: int) -> dict[str, Any]:
    quantiles = np.linspace(0.0, 1.0, NUM_QUANTILES)
    return {
        "quantiles": [float(q) for q in quantiles],
        "max_model_len": max_model_len,
        "workloads": [
            {
                "label": label,
                **bundle_provenance(workload),
                "input": {"values": length_quantiles(workload.input_lens, quantiles)},
                "output": {"values": length_quantiles(workload.output_lens, quantiles)},
            }
            for label, workload in bundles
        ],
    }


def _validate_bundle(model: str, dataset: str, label: str, workload: Workload, depth: int, delay: int) -> None:
    """Validate the recorded policy and provenance used by one paper panel."""

    if workload.meta.get("model_family") != model:
        raise ValueError(f"{model} {label}: expected model_family {model!r}")
    if workload.meta.get("dataset") != dataset:
        raise ValueError(f"{model} {label}: expected dataset {dataset!r}")
    expected_defaults = {"min_exit_step": MIN_EXIT_STEP, "exit_delay_steps": delay}
    if workload.meta.get("depth_defaults") != expected_defaults:
        raise ValueError(f"{model} {label}: expected depth_defaults {expected_defaults}")
    if workload.max_depth != depth:
        raise ValueError(f"{model} {label}: expected depth {depth}, got {workload.max_depth}")


def build_exit_distributions(bundles: dict[str, list[tuple[str, Workload]]]) -> dict[str, Any]:
    """Export the counts and fractions of the paper's six workload heatmaps."""
    models = {}
    expected_labels = [label for label, _, _ in WORKLOADS]
    for model_bundles in bundles.values():
        if [label for label, _ in model_bundles] != expected_labels:
            raise ValueError("each model requires ShareGPT, Alpaca, and ArXiv in paper order")
    for index, label in enumerate(expected_labels):
        ouro_ids = bundles["ouro"][index][1].ids
        huginn_ids = bundles["huginn"][index][1].ids
        if ouro_ids != huginn_ids:
            raise ValueError(f"{label}: Ouro and Huginn bundles have different request IDs or replay order")

    for model, (depth, thresholds, delay) in MODEL_SETTINGS.items():
        model_bundles = bundles[model]
        panels = []
        for (label, dataset, _), (_, workload) in zip(WORKLOADS, model_bundles, strict=True):
            _validate_bundle(model, dataset, label, workload, depth, delay)
            values = workload.exit_pdf if model == "ouro" else workload.exit_values
            if values is None or len(values) == 0:
                raise ValueError(f"{model} {label}: missing or empty recorded exit trajectories")
            points = []
            for threshold in thresholds:
                counts = exit_counts(workload, threshold, exit_delay_steps=delay)
                points.append(
                    {
                        "threshold": threshold,
                        "exit_counts": counts,
                        "exit_fractions": [count / sum(counts) for count in counts],
                    }
                )
            panels.append(
                {"label": label, **bundle_provenance(workload), "num_output_tokens": len(values), "points": points}
            )
        models[model] = {
            "min_exit_step": MIN_EXIT_STEP,
            "max_depth": depth,
            "exit_delay_steps": delay,
            "thresholds": list(thresholds),
            "workloads": panels,
        }
    return {"models": models}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workload-dir", type=Path, default=Path("outputs/workloads"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/paper/figs/workloads"))
    args = parser.parse_args(argv)
    bundles = {
        model: [
            (label, Workload.load(args.workload_dir / f"{model}_{dataset}_recur{depth}{suffix}.json"))
            for label, dataset, suffix in WORKLOADS
        ]
        for model, (depth, _, _) in MODEL_SETTINGS.items()
    }
    max_model_lens = {int(w.meta["max_model_len"]) for items in bundles.values() for _, w in items}
    if len(max_model_lens) != 1:
        raise ValueError(f"Bundles have different context limits: {sorted(max_model_lens)}")
    # Validate all six panels before publishing either output.
    exits = build_exit_distributions(bundles)
    lengths = build_length_distributions(bundles["ouro"], max_model_lens.pop())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for kind, payload in [("length", lengths), ("exit", exits)]:
        path = args.out_dir / f"workload_{kind}_distributions.json"
        path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
