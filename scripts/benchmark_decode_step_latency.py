"""Decode-step latency vs (batch size, context length), through the CB serving engine.

Sweeps batches ``B`` at context ``L`` at each recurrent depth ``S`` and measures
average wall-clock of a window of decode steps. Regressing

    decode_step(S) = intercept + slope * S

over the recurrent depths ``S`` splits the per-recurrent-step latency (slope) from
the fixed per-token cost outside the loop (intercept: any prelude/coda layers), per
``(B, L)`` point.

The setup relies on two properties of the CB engine:
1. a batch is always all-prefill or all-decode (never mixed)
2. prefill is scheduled before decode
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import socket
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from looped_cdb.benchmarks.runner import (
    DEFAULT_MODEL_ID,
    EngineArgs,
    RunConfig,
    ServingSession,
)
from looped_cdb.benchmarks.workload import Workload
from looped_cdb.continuous_batching.continuous_api import CacheFullError

# The decode step is graphed; prefill stays eager (it is outside the timed window)
CUDA_GRAPH_MODE = "decode"
DEFAULT_BLOCK_SIZE = 16
# The KV pool is allocated 25% larger than the exact per-request need, accounting for block
# rounding and fragmentation.
KV_POOL_HEADROOM = 1.25
# Additive spare blocks: at small batches the fractional headroom rounds to almost
# nothing, so a few extra blocks provide the slack instead.
SPARE_BLOCKS = 8
# Per-request max_new_tokens margin past the timed window, so no request drops out of
# the batch mid-measurement.
DECODE_MARGIN = 8
# Abort sweep after MAX_ERRORS (oom doesn't count)
MAX_ERRORS = 3


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #
@dataclass
class LatencyPoint:
    """One measured (batch_size, context_length) point of the latency curve."""

    batch_size: int
    context_length: int
    status: str  # "ok" | "oom" | "error:<msg>"
    full_decode_ms: float | None = None  # decode-step latency at the deepest regressed depth
    core_step_ms: float | None = None  # per-recurrent-step latency = slope of decode_step(S) = intercept + slope * S
    overhead_ms: float | None = None  # fixed per-token cost outside the loop (prelude/coda) = intercept
    regression_r2: float | None = None  # R^2 of decode_step(S); ~1.0 confirms latency is linear in depth
    gpu_core_step_ms: float | None = None  # slope of the GPU-only (CUDA-event) timing; agrees when GPU-bound
    depth_step_ms: list[float] | None = None  # regression inputs: best window per depth, aligned with regression_depths
    gpu_depth_step_ms: list[float] | None = None  # CUDA-event counterpart of depth_step_ms
    context_max: int | None = None  # largest KV length across the batch when timing opened (drift check)
    peak_mem_gb: float | None = None


def _fit_core_overhead(depths: list[int], decode_ms: list[float]) -> tuple[float, float, float]:
    """Fit ``decode_step(S) = intercept + slope * S``; returns ``(slope, intercept, r2)``.

    ``r2`` R^2 near 1.0 confirms the decode latency is linear in depth.
    """

    x = np.asarray(depths, dtype=np.float64)
    y = np.asarray(decode_ms, dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(slope), float(intercept), r2


def _cache_blocks(batch: int, context: int, max_new: int, block_size: int) -> tuple[int, int]:
    """Size the paged KV cache to hold ``batch`` resident sequences of ``context + max_new`` tokens.

    ``blocks_per_request`` also becomes the engine's per-request block-table width (via ``max_model_len``)
    """

    blocks_per_request = math.ceil((context + max_new) / block_size) + 1
    num_blocks = math.ceil(batch * blocks_per_request * KV_POOL_HEADROOM) + SPARE_BLOCKS
    return num_blocks, blocks_per_request


def measure_point(
    session: ServingSession,
    run_config: RunConfig,
    *,
    batch_size: int,
    context_length: int,
    depths: list[int],
    decode_steps: int,
    warmup_decode_steps: int,
    max_num_batched_tokens: int,
    kv_policy: str,
    repeats: int,
    seed: int,
    block_size: int,
) -> LatencyPoint:
    """Measure the per-recurrent-step decode latency at ``(batch_size, context_length)``.

    Per depth the session's shared model is set to that depth and an engine is rebuilt.
    The best of ``repeats`` timed windows is kept, and the per-step latencies are
    regressed against depth.
    """

    max_new = warmup_decode_steps + decode_steps + DECODE_MARGIN
    num_blocks, blocks_per_request = _cache_blocks(batch_size, context_length, max_new, block_size)
    # Workload for warmup with prepare_run: B prompts of length L plus the
    # per-request output budget. Identical at every depth.
    workload = Workload.synthetic(
        num_requests=batch_size,
        input_len=context_length,
        output_len=max_new,
        exit_dist="all_full",
        max_depth=max(depths),
        seed=seed,
    )
    per_step_ms: list[float] = []
    gpu_step_ms: list[float] = []
    peak_gb = 0.0
    context_max = context_length
    try:
        torch.cuda.reset_peak_memory_stats()
        for depth in depths:
            session.set_depth(depth)
            engine_args = EngineArgs(
                backend="cb",
                block_size=block_size,
                num_blocks=num_blocks,
                max_num_batched_tokens=max_num_batched_tokens,
                # The fast decode path derives its per-request block table as
                # ceil(max_model_len / block_size), so this pins it to blocks_per_request.
                max_model_len=blocks_per_request * block_size,
                max_recurrent_depth=depth,
                # One setting for the batch: B requests resident, and a decode launch that
                # advances all of them. The cap is always a decode graph bucket, so the launch
                # replays an unpadded graph whatever B is.
                max_num_seqs=batch_size,
                # Admission is irrelevant here: the batch is submitted at once and drained before any
                # decode runs, so the threshold never holds a launch.
                min_free_slots=1,
                min_recurrent_steps=1,
                no_delay_gate_consumption=False,
                kv_policy=kv_policy,
                # None resolves to the policy's own slot count (1 for single, R for depth_indexed).
                kv_slots_per_layer=None,
                exit_threshold=None,
                sync=False,
                refill=True,
                min_coda_batch_size=1,
                # Drain all prefill before any decode so the full batch of B decodes at
                # a uniform context length; see measure_decode_step_latency.
                safety_margin=0.0,
                # The cache is sized for the whole batch, so no reservation or
                # preemption is needed; "none" raises on full instead.
                kv_pressure_mode="none",
                cpu_offload_space=None,
            )
            with session.prepared(workload, engine_args, run_config) as prepared:
                timings = [
                    prepared.engine.measure_decode_step_latency(
                        prepared.input_ids,
                        max_new_tokens=max_new,
                        warmup_decode_steps=warmup_decode_steps,
                        timed_decode_steps=decode_steps,
                        model_kwargs=prepared.model_kwargs,
                    )
                    for _ in range(repeats)
                ]
                best = min(timings, key=lambda t: t.per_step_ms)
                per_step_ms.append(best.per_step_ms)
                gpu_step_ms.append(min(t.gpu_per_step_ms for t in timings))
                context_max = max(context_max, best.context_max)
                peak_gb = max(peak_gb, torch.cuda.max_memory_allocated() / 1e9)
    except (torch.cuda.OutOfMemoryError, CacheFullError):
        # A capacity boundary, not an engine fault: the point simply does not fit here.
        torch.cuda.empty_cache()
        return LatencyPoint(batch_size, context_length, "oom")
    except RuntimeError as exc:
        torch.cuda.empty_cache()
        if "out of memory" in str(exc).lower():
            return LatencyPoint(batch_size, context_length, "oom")
        # An unexpected engine error: surface it so a failed point is not silent.
        msg = str(exc).splitlines()[0][:200]
        print(f"  error at B={batch_size} L={context_length}: {msg}", file=sys.stderr)
        return LatencyPoint(batch_size, context_length, f"error:{msg}")

    slope, intercept, r2 = _fit_core_overhead(depths, per_step_ms)
    gpu_slope, _gpu_intercept, _gpu_r2 = _fit_core_overhead(depths, gpu_step_ms)
    return LatencyPoint(
        batch_size,
        context_length,
        "ok",
        full_decode_ms=slope * max(depths) + intercept,
        core_step_ms=slope,
        overhead_ms=intercept,
        regression_r2=r2,
        gpu_core_step_ms=gpu_slope,
        depth_step_ms=per_step_ms,
        gpu_depth_step_ms=gpu_step_ms,
        context_max=context_max,
        peak_mem_gb=peak_gb,
    )


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
@dataclass
class RunMeta:
    run_config: RunConfig
    device_name: str
    torch_version: str
    python_version: str
    dtype: str
    max_depth: int
    hostname: str
    batch_sizes: list[int]
    context_length: int
    regression_depths: list[int]
    decode_steps: int
    warmup_decode_steps: int
    repeats: int
    max_num_batched_tokens: int
    block_size: int
    backend: str
    kv_policy: str


def parse_int_csv(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging level for the benchmark run.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--attn-implementation", default="paged|flash_attention_3")
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument(
        "--depths",
        type=parse_int_csv,
        required=True,
        help="Depths to regress over; at least two, the largest being the model's full depth R_max.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=parse_int_csv,
        default="1,2,4,8,16,32,64,128,256,512",
        help="Decode batch widths to time; each one builds an engine capped at that width.",
    )
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--decode-steps", type=int, default=16, help="Decode steps timed inside the window.")
    parser.add_argument(
        "--warmup-decode-steps",
        type=int,
        default=3,
        help="Number of decode ticks before starting time measurement. Slightly increases context length.",
    )
    parser.add_argument(
        "--repeats", type=int, default=3, help="Number of times to repeat the measurement; the min is kept."
    )
    parser.add_argument(
        "--kv-policy",
        choices=["single", "depth_indexed"],
        default="single",
        help="single vs. depth_indexed just affects memory usage and not the latency.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/decode_step_latency/decode_step_latency.jsonl"),
        help="Append one JSONL row per measured point here.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    if args.context_length < 1:
        raise SystemExit(f"--context-length must be >= 1, got {args.context_length}.")
    if args.block_size < 1:
        raise SystemExit(f"--block-size must be >= 1, got {args.block_size}.")

    depths = sorted(set(args.depths))
    if len(depths) < 2:
        raise SystemExit("Need at least two depths to regress decode latency against depth.")
    if depths[0] < 1:
        raise SystemExit(f"--depths values must be >= 1, got {depths}.")
    max_depth = depths[-1]

    # Ascending, so the sweep can stop a context's row at the first batch that no longer fits.
    batch_sizes = sorted(set(args.batch_sizes))
    if batch_sizes[0] < 1:
        raise SystemExit(f"--batch-sizes values must be >= 1, got {batch_sizes}.")
    # Limits the number of tokens in each batch.
    # Kept large for efficient prefill. Should be at least as large as the largest swept decode batch size.
    max_num_batched_tokens = max(2048, batch_sizes[-1])

    run_config = RunConfig(
        model=args.model,
        attn_implementation=args.attn_implementation,
        seed=args.seed,
        min_exit_step=1,
        exit_delay_steps=0,
        cuda_graph_mode=CUDA_GRAPH_MODE,
    )
    # Load the model once and reuse it across the whole sweep
    session = ServingSession(
        run_config,
        max_recurrent_depth=max_depth,
        kv_policy=args.kv_policy,
    )

    meta = RunMeta(
        run_config=run_config,
        device_name=torch.cuda.get_device_name(0),
        torch_version=torch.__version__,
        python_version=platform.python_version(),
        dtype=str(session.dtype),
        max_depth=max_depth,
        hostname=socket.gethostname(),
        batch_sizes=batch_sizes,
        context_length=args.context_length,
        regression_depths=depths,
        decode_steps=args.decode_steps,
        warmup_decode_steps=args.warmup_decode_steps,
        repeats=args.repeats,
        max_num_batched_tokens=max_num_batched_tokens,
        block_size=args.block_size,
        backend="cb",
        kv_policy=args.kv_policy,
    )

    for key, value in asdict(meta).items():
        print(f"{key}: {value}")
    print()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    def save_results(point: LatencyPoint, elapsed_s: float) -> None:
        row = {"meta": asdict(meta), "elapsed_s": round(elapsed_s, 1), **asdict(point)}
        with args.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    errors = 0
    for bsz in batch_sizes:
        point_start = time.perf_counter()
        point = measure_point(
            session,
            run_config,
            batch_size=bsz,
            context_length=args.context_length,
            depths=depths,
            decode_steps=args.decode_steps,
            warmup_decode_steps=args.warmup_decode_steps,
            max_num_batched_tokens=max_num_batched_tokens,
            kv_policy=args.kv_policy,
            repeats=args.repeats,
            seed=args.seed,
            block_size=args.block_size,
        )
        save_results(point, time.perf_counter() - point_start)
        core = f"{point.core_step_ms:.3f}" if point.core_step_ms is not None else "-"
        r2 = f"{point.regression_r2:.5f}" if point.regression_r2 is not None else "-"
        print(f"L={args.context_length} B={bsz}: {point.status} core_ms={core} r2={r2}")
        if point.status == "oom":
            # A larger batch can only need more cache
            break
        if point.status.startswith("error:"):
            errors += 1
            if errors >= MAX_ERRORS:
                raise SystemExit(f"{MAX_ERRORS} measurement errors; aborting (results so far are saved).")

    print(f"\nWrote to {args.output_path}. Benchmark took {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
