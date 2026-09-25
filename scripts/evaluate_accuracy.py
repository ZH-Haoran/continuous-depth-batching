"""Measure task accuracy of one model.

Builds a task from :mod:`looped_cdb.eval`, loads the model in memory, constructs the
requested backend, runs the eval, and prints/writes the aggregated metrics.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

import torch

from looped_cdb.eval.runner import EvalResult, run_task
from looped_cdb.eval.tasks import get_task
from looped_cdb.kv_cache_policy import DEPTH_INDEXED, KV_CACHE_POLICIES

logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the eval CLI argument parser."""

    parser = argparse.ArgumentParser(prog="evaluate_accuracy.py", description="GSM8k eval harness.")
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging level for the accuracy run.",
    )
    parser.add_argument("--task", default="gsm8k_cot", help="Registered task name.")
    parser.add_argument("--backend", default="cb", choices=["cb", "cdb"])
    parser.add_argument("--model", default="KristianS7/Ouro-1.4B")
    parser.add_argument("--num-fewshot", type=int, default=8)
    parser.add_argument("--recur-steps", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N sequences.")
    parser.add_argument("--max-gen-toks", type=int, default=None, help="Override the task's max generated tokens.")
    parser.add_argument(
        "--attn-implementation",
        default="paged|flash_attention_3",
        help="Attention backend. CB and CDB both page the cache, so this carries the 'paged|' prefix. FA3 needs Hopper.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=None,
        help="Usable KV blocks; omit for automatic sizing.",
    )
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=None,
        help="GPU memory fraction for weights and KV cache; default 0.8 when sizing automatically.",
    )
    parser.add_argument(
        "--no-delay-gate-consumption",
        action="store_true",
        help=(
            "Disable CDB's delayed gate consumption. The engine otherwise consumes gate output one recurrent step late. "
            "This has no effect on preloop gates, which decide before recurrence."
        ),
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help=(
            "Paged KV block size (CB/CDB). Defaults to the engine's setting; FA2's paged decode "
            "kernel requires a multiple of 256."
        ),
    )
    parser.add_argument(
        "--kv-pressure-mode",
        default="reserve",
        choices=["none", "reserve", "recompute", "offload"],
        help=("How the engine handles KV-cache exhaustion. `reserve` guarantees no preemption."),
    )
    parser.add_argument(
        "--kv-policy",
        default=DEPTH_INDEXED,
        choices=list(KV_CACHE_POLICIES),
        help="Recurrent KV layout. Defaults to the depth-indexed layout the checkpoints are trained under.",
    )
    parser.add_argument(
        "--exit-gate-type",
        default=None,
        help="early_exit | same_step | lookahead | preloop (omit to use the model's built-in gate).",
    )
    parser.add_argument("--exit-gate-path", default=None, help="Path to a trained gate .safetensors (cdb).")
    parser.add_argument(
        "--exit-threshold",
        type=float,
        default=None,
        help=(
            "Adaptive-depth threshold. Ouro exits when cumulative gate probability reaches it; "
            "Huginn exits when relative state change falls below it. The scales are not comparable."
        ),
    )
    parser.add_argument(
        "--no-refill",
        action="store_true",
        help="Disable depth refill (cdb): freed depth slots stay empty until the wave boundary.",
    )
    parser.add_argument(
        "--min-recurrent-steps",
        type=int,
        default=None,
        help="Minimum recurrent steps before early exit (cdb).",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="JSONL file to append this run's summary to. One line per run, as the benchmarks write.",
    )
    return parser


def reject_trained_gate_flags(args: argparse.Namespace) -> None:
    """Reject trained-gate flags on a model that has no trained gate.

    Huginn exits on a state-convergence test rather than a gate head, so a gate type or
    path would be a silent no-op instead of the exit policy the caller asked for.
    """

    for flag, value in (("--exit-gate-type", args.exit_gate_type), ("--exit-gate-path", args.exit_gate_path)):
        if value is not None:
            raise SystemExit(f"{flag} applies to trained gates; Huginn exits on a state-convergence threshold.")


def build_backend(args: argparse.Namespace) -> Any:
    """Load the model and construct the requested generation backend."""

    from looped_cdb.eval.backends import CBBackend, CDBBackend
    from looped_cdb.eval.model_loading import (
        load_huginn_model,
        load_huginn_tokenizer,
        load_ouro_model,
        load_ouro_tokenizer,
        resolve_model_family,
    )

    if args.backend == "cb" and args.kv_policy == "last_exited":
        # The recorded kv_policy would otherwise label a full-depth run with routing that cannot fire.
        raise SystemExit(
            "--kv-policy last_exited never routes on cb (fixed depth, no early exit) and its layout "
            "is exactly depth_indexed; use --kv-policy depth_indexed or --backend cdb."
        )
    family = resolve_model_family(args.model)
    if family == "huginn":
        reject_trained_gate_flags(args)
        model = load_huginn_model(
            args.model,
            recur_steps=args.recur_steps,
            kv_policy=args.kv_policy,
            attn_impl=args.attn_implementation,
            device=args.device,
        )
        tokenizer = load_huginn_tokenizer(args.model)
    else:
        model = load_ouro_model(
            args.model,
            recur_steps=args.recur_steps,
            exit_gate_type=args.exit_gate_type,
            exit_gate_path=args.exit_gate_path,
            kv_policy=args.kv_policy,
            attn_impl=args.attn_implementation,
            device=args.device,
        )
        tokenizer = load_ouro_tokenizer(args.model)

    if args.backend == "cb":
        early_exit_flags = {
            "--exit-threshold": args.exit_threshold,
            "--exit-gate-type": args.exit_gate_type,
            "--exit-gate-path": args.exit_gate_path,
            "--min-recurrent-steps": args.min_recurrent_steps,
        }
        set_flags = [name for name, value in early_exit_flags.items() if value is not None]
        if set_flags:
            raise SystemExit(
                f"cb is fixed-depth only and ignores early-exit flags ({', '.join(set_flags)}); "
                "use --backend cdb for adaptive early exit."
            )
        from looped_cdb.continuous_batching.config import ContinuousBatchingConfig

        cb_config = ContinuousBatchingConfig(
            num_blocks=args.num_blocks,
            mem_fraction_static=args.mem_fraction_static,
            kv_pressure_mode=args.kv_pressure_mode,
        )
        if args.block_size is not None:
            cb_config = dataclasses.replace(cb_config, block_size=args.block_size)
        return CBBackend.from_model(model, tokenizer, cb_config=cb_config)

    # cdb
    if args.recur_steps is None:
        raise SystemExit("cdb requires --recur-steps (the maximum recurrent step count).")
    from looped_cdb.continuous_depth_batching import ContinuousDepthBatchingConfig

    model_adapter = None
    exit_threshold = args.exit_threshold

    cdb_config = ContinuousDepthBatchingConfig(
        num_blocks=args.num_blocks,
        mem_fraction_static=args.mem_fraction_static,
        max_recurrent_steps=args.recur_steps,
        exit_threshold=exit_threshold,
        min_recurrent_steps=args.min_recurrent_steps or 1,
        refill=not args.no_refill,
        kv_pressure_mode=args.kv_pressure_mode,
    )
    if args.block_size is not None:
        cdb_config = dataclasses.replace(cdb_config, block_size=args.block_size)
    if args.no_delay_gate_consumption:
        cdb_config = dataclasses.replace(cdb_config, delay_gate_consumption=False)
    # CDB stages recurrent work itself, so its cache reads the layout from the config rather
    # than from the model attributes the CB path uses.
    cdb_config = dataclasses.replace(cdb_config, kv_policy=args.kv_policy)
    return CDBBackend.from_model(model, tokenizer, cdb_config=cdb_config, model_adapter=model_adapter)


def _result_payload(result: EvalResult, args: argparse.Namespace, backend: Any) -> dict[str, Any]:
    """Build the JSON-serializable summary shared by stdout and the results file."""

    payload: dict[str, Any] = {
        "task": result.task,
        "backend": args.backend,
        "model": args.model,
        "recur_steps": args.recur_steps,
        "kv_policy": args.kv_policy,
        "kv_pressure_mode": args.kv_pressure_mode,
        "num_blocks": backend.engine.cache.num_blocks,
        "mem_fraction_static": args.mem_fraction_static,
        "block_size": args.block_size,
        "attn_implementation": args.attn_implementation,
        "num_fewshot": result.num_fewshot,
        "num_docs": result.num_docs,
        "metrics": result.metrics,
    }
    if args.backend == "cdb":
        # Both fields describe what the engine served, not what was requested. The gate
        # type falls back to the checkpoint's own when the flag is omitted, and a pre-loop
        # gate decides before the loop runs, so there is no in-flight decision to consume
        # a step later and the flag has no effect on it.
        preloop_gate = backend.decides_exit_before_loop
        payload["delay_gate_consumption"] = not args.no_delay_gate_consumption and not preloop_gate
        payload["refill"] = not args.no_refill
        payload["min_recurrent_steps"] = args.min_recurrent_steps
        payload["exit_gate_type"] = backend.exit_gate_type
        payload["exit_gate_path"] = args.exit_gate_path
        payload["exit_threshold"] = args.exit_threshold
    if result.exit_depth_counts is not None:
        payload["exit_depth_counts"] = {str(k): v for k, v in sorted(result.exit_depth_counts.items())}
        payload["mean_exit_depth"] = result.mean_exit_depth
    return payload


def _append_summary(summary_output: str, payload: dict[str, Any]) -> None:
    """Append one run's summary to a JSONL file, creating it if needed.

    A sweep appends every row to one file, so the whole matrix is a single artifact
    rather than a directory tree that has to be walked and name-parsed to read back.
    """

    path = Path(summary_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")


def main(argv: list[str] | None = None) -> None:
    """Parse args, run the eval, and report metrics."""

    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    task = get_task(args.task)
    if args.max_gen_toks is not None:
        task = dataclasses.replace(task, max_gen_toks=args.max_gen_toks)

    backend = build_backend(args)
    result = run_task(task, backend, num_fewshot=args.num_fewshot, limit=args.limit)

    cache = backend.engine.cache
    if cache.device.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info(cache.device)
        logger.info("KV cache after accuracy run: free=%.2f GiB", free_bytes / 1024**3)

    payload = _result_payload(result, args, backend)
    print(json.dumps(payload, indent=2))
    if args.summary_output is not None:
        _append_summary(args.summary_output, payload)


if __name__ == "__main__":
    main()
