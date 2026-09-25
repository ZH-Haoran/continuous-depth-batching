"""Export GSM8K accuracy sweeps for the paper plots.

Reads JSONL written by ``scripts/evaluate_accuracy.py``.
Fixed-depth curves are split by KV layout.
Adaptive curves are split by gate, consumption timing, refill policy, and KV layout.
Cache sizes are derived from each checkpoint configuration.

The sweeps are produced by ``shells/paper/ablations/accuracy_gsm8k_{ouro,huginn}.sh``.

Example:
    uv run python scripts/exporters/export_accuracy_results.py \
        --ouro outputs/ablations/accuracy/paper_gsm8k_ouro.jsonl \
        --huginn outputs/ablations/accuracy/paper_gsm8k_huginn.jsonl \
        --output docs/paper/figs/accuracy/accuracy.json
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from looped_cdb.benchmarks.exports import warn
from looped_cdb.kv_cache_policy import DEPTH_INDEXED, LAST_EXITED, resolve_kv_slots_per_layer

METRIC = "flexible-extract"

GATE_NAMES = {"early_exit": "gate", "lookahead": "lookahead", "preloop": "preloop"}
TIMING_NAMES = {True: "delayed", False: "immediate"}

# Threshold at which each exit test stops firing. Ouro's gate accumulates probability upward
# and Huginn's criterion decays, so the no-exit end sits at opposite ends of the two scales.
NO_EXIT_THRESHOLD = {"ouro": 1.0, "huginn": 0.0}

SHARING_TABLE_LAYOUTS = (DEPTH_INDEXED, "single", "first_then_shared")
BYTES_PER_ELEMENT = 2  # bfloat16


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Read one sweep's JSONL."""

    if not path.is_file():
        raise SystemExit(f"no such sweep: {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"no rows in {path}")
    return rows


def agreed(rows: list[dict[str, Any]], key: str) -> Any:
    """The one value every row recorded for ``key``."""

    values = {json.dumps(row.get(key), sort_keys=True) for row in rows}
    if len(values) != 1:
        raise SystemExit(f"rows disagree on {key!r}: {sorted(values)}")
    return json.loads(values.pop())


def reported(rows: list[dict[str, Any]], key: str, what: str) -> Any:
    """Like :func:`agreed`, but a disagreement is reported rather than fatal."""

    values = sorted({json.dumps(row.get(key), sort_keys=True) for row in rows})
    if len(values) > 1:
        warn(f"{what} mixes {key} ({', '.join(values)})")
        return [json.loads(value) for value in values]
    return json.loads(values[0])


def latest(rows: list[dict[str, Any]], key: Any) -> list[dict[str, Any]]:
    """Keep the last row for each explicitly supplied configuration key."""

    return list({key(row): row for row in rows}.values())


def dense_exit_counts(counts: dict[str, int], max_depth: int) -> list[int]:
    """Expand a sparse ``{depth: count}`` map into a dense 1..max_depth list."""

    dense = [0] * max_depth
    for depth, count in counts.items():
        index = int(depth) - 1
        if not 0 <= index < max_depth:
            raise SystemExit(f"exit depth {depth} outside budget {max_depth}")
        dense[index] = count
    return dense


def curve_name(row: dict[str, Any], model: str) -> str:
    """Canonical paper name for one adaptive-depth policy and KV layout."""

    if model == "huginn":
        policy = TIMING_NAMES[bool(row["delay_gate_consumption"])]
    else:
        gate = row["exit_gate_type"]
        if gate not in GATE_NAMES:
            raise SystemExit(f"unknown exit_gate_type {gate!r}; expected one of {sorted(GATE_NAMES)}")
        policy = GATE_NAMES[gate]
    return f"{policy}_{row['kv_policy']}"


def curve_key(row: dict[str, Any], model: str) -> tuple[str, bool, bool]:
    """Identity of an adaptive curve, including execution timing."""

    return curve_name(row, model), bool(row["delay_gate_consumption"]), bool(row.get("refill", True))


def cache_geometry(model: str, checkpoint: str) -> tuple[int, int, int]:
    """Boundary layers, core layers and KV width per layer, off the checkpoint config."""

    if model == "huginn":
        from looped_cdb.models.huginn import HuginnConfig

        config = HuginnConfig.from_pretrained(checkpoint)
        boundary = config.n_layers_in_prelude + config.n_layers_in_coda
        core = config.n_layers_in_recurrent_block
    elif model == "ouro":
        from looped_cdb.models.ouro import OuroConfig

        config = OuroConfig.from_pretrained(checkpoint)
        boundary, core = 0, config.num_hidden_layers
    else:
        raise SystemExit(f"no cache geometry for {model!r}")
    return boundary, core, config.num_key_value_heads * config.head_dim


def kv_layouts(rows: list[dict[str, Any]], model: str, checkpoint: str, max_depth: int) -> list[dict[str, Any]]:
    """Accuracy and cache size of each KV layout at the serving budget.

    A core layer holds one KV slot per recurrent slot and a boundary layer holds one, so a
    token costs ``(boundary + core * slots) * kv_width`` for K and V.
    """

    accuracy = {
        row["kv_policy"]: row["metrics"][METRIC]
        for row in rows
        if row["backend"] == "cb" and row["recur_steps"] == max_depth
    }
    missing = [policy for policy in SHARING_TABLE_LAYOUTS if policy not in accuracy]
    if missing:
        warn(f"{model} has no fixed-depth run at depth {max_depth} under {', '.join(missing)}")

    boundary, core, kv_width = cache_geometry(model, checkpoint)
    layouts = []
    for policy in SHARING_TABLE_LAYOUTS:
        if policy not in accuracy:
            continue
        slots = resolve_kv_slots_per_layer(policy, total_recurrent_steps=max_depth)
        kib = (boundary + core * slots) * kv_width * 2 * BYTES_PER_ELEMENT / 1024
        layouts.append(
            {
                "kv_policy": policy,
                "slots": slots,
                "kib_per_token": kib,
                "acc": accuracy[policy],
            }
        )
    return layouts


def build_model(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    """Aggregate one model's sweep."""

    checkpoint = agreed(rows, "model")
    agreed(rows, "task")
    agreed(rows, "num_fewshot")
    agreed(rows, "num_docs")
    kv_pressure_mode = agreed(rows, "kv_pressure_mode")
    num_blocks = reported(rows, "num_blocks", model)
    block_size = agreed(rows, "block_size")
    fixed_rows = latest(
        [row for row in rows if row["backend"] == "cb"], lambda row: (row["recur_steps"], row["kv_policy"])
    )
    gated_rows = latest(
        [row for row in rows if row["backend"] == "cdb"],
        lambda row: (
            curve_key(row, model),
            row["exit_threshold"],
            row["recur_steps"],
            row["min_recurrent_steps"],
            row["delay_gate_consumption"],
            row.get("refill", True),
            row.get("exit_gate_path"),
        ),
    )
    if not fixed_rows or not gated_rows:
        raise SystemExit(f"{model} needs both fixed-depth and gated rows; got {len(fixed_rows)} and {len(gated_rows)}")

    max_depth = agreed(gated_rows, "recur_steps")

    fixed_curves: dict[str, list[dict[str, Any]]] = {}
    for row in fixed_rows:
        fixed_curves.setdefault(row["kv_policy"], []).append(
            {"depth": row["recur_steps"], "acc": row["metrics"][METRIC]}
        )
    for points in fixed_curves.values():
        points.sort(key=lambda point: point["depth"])

    def anchor(kv_policy: str) -> float:
        """Fixed-depth accuracy at the budget."""

        reference = DEPTH_INDEXED if kv_policy == LAST_EXITED else kv_policy
        for point in fixed_curves.get(reference, []):
            if point["depth"] == max_depth:
                return point["acc"]
        raise SystemExit(f"{model} has no fixed-depth run at {max_depth} under {reference} to anchor its curves")

    curves: dict[tuple[str, bool, bool], list[dict[str, Any]]] = {}
    floors: dict[tuple[str, bool, bool], list[dict[str, Any]]] = {}
    for row in gated_rows:
        key = curve_key(row, model)
        if "exit_depth_counts" not in row:
            raise SystemExit(f"{model} gated row at threshold {row['exit_threshold']} recorded no exit depths")
        floors.setdefault(key, []).append(row)
        curves.setdefault(key, []).append(
            {
                "threshold": row["exit_threshold"],
                "acc": row["metrics"][METRIC],
                "mean_depth": row["mean_exit_depth"],
                "exit_counts": dense_exit_counts(row["exit_depth_counts"], max_depth),
            }
        )

    variants = [
        {"name": f"fixed_depth_{policy}", "kind": "fixed", "kv_policy": policy, "points": points}
        for policy, points in sorted(fixed_curves.items())
    ]
    base_counts: dict[str, int] = {}
    for base, _, _ in curves:
        base_counts[base] = base_counts.get(base, 0) + 1
    for key in sorted(curves):
        base, delayed, refill = key
        name = base
        if base_counts[base] > 1:
            refill_name = "refill" if refill else "norefill"
            name = f"{base}_{TIMING_NAMES[delayed]}_{refill_name}"
        points = curves[key]
        # The token total carries over from the point that already runs deepest; its depth
        # distribution does not.
        deepest = max(points, key=lambda point: point["mean_depth"])
        points.append(
            {
                "threshold": NO_EXIT_THRESHOLD[model],
                "acc": anchor(agreed(floors[key], "kv_policy")),
                "mean_depth": float(max_depth),
                "exit_counts": [0] * (max_depth - 1) + [sum(deepest["exit_counts"])],
            }
        )
        points.sort(key=lambda point: point["threshold"])
        variants.append(
            {
                "name": name,
                "kind": "gated",
                "kv_policy": agreed(floors[key], "kv_policy"),
                "delay_gate_consumption": agreed(floors[key], "delay_gate_consumption"),
                "refill": floors[key][0].get("refill", True),
                "exit_gate_type": agreed(floors[key], "exit_gate_type"),
                "exit_gate_path": agreed(floors[key], "exit_gate_path"),
                "max_depth": max_depth,
                "min_exit_step": floor if (floor := agreed(floors[key], "min_recurrent_steps")) is not None else 1,
                "points": points,
            }
        )

    return {
        "checkpoint": checkpoint,
        "num_fewshot": agreed(rows, "num_fewshot"),
        "num_docs": agreed(rows, "num_docs"),
        "max_depth": max_depth,
        "kv_pressure_mode": kv_pressure_mode,
        "num_blocks": num_blocks,
        "block_size": block_size,
        "attn_implementation": reported(rows, "attn_implementation", model),
        "variants": variants,
        "kv_layouts": kv_layouts(fixed_rows, model, checkpoint, max_depth),
    }


def build_results(sweeps: dict[str, Path]) -> dict[str, Any]:
    rows_by_model = {model: load_rows(path) for model, path in sweeps.items()}
    tasks = {agreed(rows, "task") for rows in rows_by_model.values()}
    if len(tasks) != 1:
        raise SystemExit(f"accuracy sweeps disagree on task: {sorted(tasks)}")
    return {
        "exported_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "task": tasks.pop(),
        "metric": METRIC,
        "models": {model: build_model(rows, model) for model, rows in rows_by_model.items()},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ouro", type=Path, required=True, help="Ouro accuracy sweep JSONL.")
    parser.add_argument("--huginn", type=Path, required=True, help="Huginn accuracy sweep JSONL.")
    parser.add_argument("--output", type=Path, required=True, help="Committed accuracy JSON to write.")
    args = parser.parse_args(argv)

    results = build_results({"ouro": args.ouro, "huginn": args.huginn})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")

    print(f"Wrote {args.output}")
    for model, entry in results["models"].items():
        print(f"  {model}: R={entry['max_depth']}, {entry['num_fewshot']}-shot")
        for variant in entry["variants"]:
            print(f"    {variant['name']:12s} {len(variant['points']):3d} points")


if __name__ == "__main__":
    main()
