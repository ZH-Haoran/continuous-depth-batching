"""Measure decode throughput of one serving backend on one workload.

A workload supplies prompt/output lengths and per-output-token exit trajectories.
Two sources share the same replay path:

- ``--dataset bundle`` loads a recorded ShareGPT, Alpaca, or ArXiv bundle.
  Ouro bundles contain exit PDFs; Huginn bundles contain convergence values.
- ``--dataset random`` synthesizes a workload with fixed or uniformly sampled lengths
  and explicit per-token exit depths.

One ``--backend`` is run per invocation (loop the script to compare):

- ``cb``            full-depth baseline (lengths only; runs all recurrent steps).
- ``cdb``           replays the schedule; the depth-batched scheduler shrinks the batch
                    as tokens exit and skips recurrent steps.

By default the workload is submitted upfront and drained (offline batch throughput).
``--request-rate`` switches to an open-loop serving run: requests are released on a
seeded Poisson arrival trace and each summary row additionally records per-request
queue/TTFT/E2E/TPOT and normalized-latency percentiles under ``request_latency``.
``--trace-duration-s`` sets the request count to
``ceil(request_rate * target_duration)``.
The realized Poisson trace duration remains stochastic.

The measured path (model load, engine build, warmup, timed run) lives in
``looped_cdb.benchmarks.runner``; this script is the thin CLI around it. GPU runs
should be submitted through SLURM.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any, get_args

from looped_cdb.arrivals import poisson_arrival_offsets
from looped_cdb.benchmarks import nvtx
from looped_cdb.benchmarks.exit_distributions import DISTRIBUTION_KINDS
from looped_cdb.benchmarks.flop_bound import stage_flops
from looped_cdb.benchmarks.metrics import (
    BenchmarkConfig,
    BenchmarkSummary,
    request_latency_summary,
)
from looped_cdb.benchmarks.runner import (
    BACKENDS,
    DEFAULT_MODEL_ID,
    EARLY_EXIT_BACKENDS,
    EngineArgs,
    RunConfig,
    _length_summary,
    backend_stats,
    effective_summary_for_backend,
    first_token_full_depth_count,
    measure_once,
    prepare_run,
    requested_exit_distribution,
)
from looped_cdb.benchmarks.workload import Workload
from looped_cdb.continuous_depth_batching.schedule_trace import ScheduleTrace
from looped_cdb.kv_cache_policy import KV_CACHE_POLICIES
from looped_cdb.utils import DEFAULT_MAX_NUM_SEQS, KVPressureMode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging level for the benchmark run.",
    )
    parser.add_argument("--backend", choices=BACKENDS, default="cb")
    parser.add_argument(
        "--refill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="cdb only: refill freed depth slots (continuous depth batching). "
        "Pass --no-refill for the sequence-level no-refill baseline.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--attn-implementation", default="paged|flash_attention_3")
    parser.add_argument("--max-recurrent-depth", type=int, default=4)
    parser.add_argument(
        "--layer-split",
        default=None,
        help=(
            "Huginn only: override the prelude-core-coda layer split, e.g. '1-4-1' or '0-4-0'. "
            "Checkpoint layers outside the split are dropped at load, so generated text stops being "
            "meaningful."
        ),
    )

    # Workload source: recorded bundle or synthetic random-length.
    parser.add_argument("--dataset", choices=["bundle", "random"], default="bundle")
    parser.add_argument("--workload", type=Path, default=None, help="Recorded workload bundle JSON (--dataset bundle).")
    parser.add_argument(
        "--num-requests",
        "--limit",
        dest="num_requests",
        type=int,
        default=None,
        help="Requests to run: first N of a bundle, or the number to synthesize for --dataset random.",
    )
    parser.add_argument("--input-len", type=int, default=512, help="Prompt length for --dataset random.")
    parser.add_argument("--output-len", type=int, default=128, help="Generation length for --dataset random.")
    parser.add_argument(
        "--input-len-high", type=int, default=None, help="If set, sample prompt len in [input-len, high]."
    )
    parser.add_argument(
        "--output-len-high", type=int, default=None, help="If set, sample output len in [output-len, high]."
    )
    parser.add_argument("--exit-dist", choices=DISTRIBUTION_KINDS, default=None, help="Exit distribution for random.")

    parser.add_argument(
        "--exit-threshold",
        type=float,
        default=None,
        help=(
            "Threshold used to derive per-token depths from a recorded trajectory. "
            "Ouro exits when cumulative gate probability reaches the threshold; Huginn exits when "
            "relative state change falls below it. Required for early-exit bundle replays; ignored "
            "for synthetic workloads."
        ),
    )
    parser.add_argument(
        "--min-exit-step",
        type=int,
        default=None,
        help=(
            "1-based minimum recurrent depth. Unset falls back to the workload bundle's recorded "
            "depth_defaults, then to 2, so the async delayed cdb path (which cannot represent a "
            "depth-1 exit) works out of the box; pass 1 together with --sync or "
            "--no-delay-gate-consumption to allow depth-1 exits. On a bundle with recorded "
            "defaults this flag and --exit-delay-steps must be set together or not at all."
        ),
    )
    parser.add_argument(
        "--exit-delay-steps",
        type=int,
        default=None,
        help=(
            "Extra recurrent steps after the exit depth. Unset falls back to the workload bundle's "
            "recorded depth_defaults, then to 0. On a bundle with recorded defaults this flag and "
            "--min-exit-step must be set together or not at all."
        ),
    )
    parser.add_argument(
        "--no-delay-gate-consumption",
        action="store_true",
        help="Consume CDB exit decisions immediately instead of the async one-step delayed policy.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=None,
        help=(
            "Offered request rate in requests/s for an open-loop serving run: requests are released "
            "on a seeded Poisson arrival trace instead of all upfront, and per-request latency "
            "(queue/TTFT/E2E/TPOT) is recorded. Default (unset) drains the whole workload."
        ),
    )
    parser.add_argument(
        "--arrival-seed",
        type=int,
        default=0,
        help="Seed of the Poisson arrival trace; keep it fixed across backends so they replay one trace.",
    )
    parser.add_argument(
        "--trace-duration-s",
        type=float,
        default=None,
        help=(
            "Set the request count to ceil(request_rate * target_duration). The realized Poisson "
            "duration remains stochastic. Requires --request-rate; mutually exclusive with "
            "--num-requests; raises if the workload bundle has too few requests."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat-index", type=int, default=0)
    parser.add_argument("--num-blocks", type=int, default=None, help="Usable KV blocks; omit for automatic sizing.")
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=None,
        help="GPU memory fraction for weights and KV cache; default 0.8 when sizing automatically.",
    )
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=DEFAULT_MAX_NUM_SEQS,
        help="Maximum decode batch size. At most this many requests are resident or advanced by one "
        "decode launch. Must not exceed --max-num-batched-tokens.",
    )
    parser.add_argument(
        "--min-free-slots",
        type=int,
        default=None,
        help="Resident slots that must stand open before a prefill launch runs, for both backends; the "
        "launch then takes every waiting prompt that fits. Bounds how far the decode batch falls below "
        "--max-num-seqs between refills, and is resolved against it. 1 admits whenever a slot is free; "
        "unset waits for an eighth of --max-num-seqs.",
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=16384,
        help=(
            "Max sequence length (prompt + generated) per request, vLLM-style (default 16384). Over-length "
            "prompts are rejected at admission and generation stops on reaching it; bundle requests are "
            "filtered/truncated to fit before replay. Also sizes the fast decode path block table."
        ),
    )
    parser.add_argument("--kv-policy", choices=KV_CACHE_POLICIES, default="single")
    parser.add_argument("--kv-slots-per-layer", type=int, default=None)
    parser.add_argument("--min-recurrent-steps", type=int, default=1)
    parser.add_argument(
        "--min-coda-batch-size",
        type=int,
        default=1,
        help="cdb only: hold the coda bucket until this many exited tokens wait (1 launches every coda immediately).",
    )
    parser.add_argument(
        "--kv-pressure-mode",
        choices=get_args(KVPressureMode),
        default="recompute",
        help="KV-cache-pressure policy: none (no preemption), reserve (worst-case admission), "
        "recompute (soft-reset preemption), or offload (CPU-swap preemption).",
    )
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=0.2,
        help="Fraction of free KV blocks below which prompt prefill stops being admitted. 0.0 removes "
        "the throttle and leaves the resident cap and the pressure policy as the only bounds; "
        "kv-pressure-mode reserve zeroes it regardless, its admission being exact already.",
    )
    parser.add_argument(
        "--cpu-offload-space",
        type=float,
        default=None,
        help="Pinned CPU swap-pool size in GiB; required (>0) for --kv-pressure-mode offload.",
    )
    parser.add_argument(
        "--sync", action="store_true", help="Serialize the engine loop (no host run-ahead); CUDA graphs still replay."
    )
    parser.add_argument(
        "--cuda-graph-mode",
        choices=["all", "decode", "none"],
        default="all",
        help="CUDA graphs for decode launches and the varlen prefill forward ('all'), decode only ('decode'), "
        "or fully eager ('none').",
    )
    parser.add_argument(
        "--steady-warmup-requests",
        type=int,
        default=None,
        help="Completed requests before the steady-state throughput window opens; it closes at the last "
        "admission, so the row's steady_state block excludes the initial fill and the final drain. "
        "Defaults to twice --max-num-seqs. Closed-loop runs only.",
    )
    parser.add_argument(
        "--replay-eos-finishes",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="A replayed request ends after its recorded number of output tokens, which the scheduler "
        "predicts; free generation ends on a sampled EOS instead, which the async scheduler only sees "
        "one token late. On by default (measured throughput-neutral): the finish is discovered at "
        "consume time and one lagged token per request is computed and discarded "
        "(cancelled_depth_items), as an EOS finish behaves. --sync implies off; the explicit "
        "combination is rejected.",
    )
    parser.add_argument("--summary-output", type=Path, default=None, help="Append benchmark summary JSONL here.")
    parser.add_argument(
        "--trace-output",
        type=Path,
        default=None,
        help="Write the measured CDB run's stage launches as JSONL. Recording adds host bookkeeping.",
    )
    parser.add_argument(
        "--nvtx",
        action="store_true",
        help="Emit NVTX ranges for the engine phases and wrap the measured run in a 'benchmark.generate' "
        "range and the steady-state window in 'benchmark.steady' (for Nsight Systems; "
        "scripts/analyze_paper_nsys.py clips to the steady window when present, else to the run). Adds "
        "per-batch host overhead, so do not mix profiled rows into throughput aggregates.",
    )
    return parser.parse_args()


def engine_args_from_cli(args: argparse.Namespace) -> EngineArgs:
    """Project the CLI namespace onto the engine-construction fields."""

    return EngineArgs(
        backend=args.backend,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        mem_fraction_static=args.mem_fraction_static,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        max_recurrent_depth=args.max_recurrent_depth,
        max_num_seqs=args.max_num_seqs,
        min_free_slots=args.min_free_slots,
        steady_warmup_requests=args.steady_warmup_requests,
        min_recurrent_steps=args.min_recurrent_steps,
        no_delay_gate_consumption=args.no_delay_gate_consumption,
        kv_policy=args.kv_policy,
        kv_slots_per_layer=args.kv_slots_per_layer,
        exit_threshold=args.exit_threshold,
        sync=args.sync,
        refill=args.refill,
        min_coda_batch_size=args.min_coda_batch_size,
        safety_margin=args.safety_margin,
        kv_pressure_mode=args.kv_pressure_mode,
        cpu_offload_space=args.cpu_offload_space,
        replay_eos_finishes=args.replay_eos_finishes if args.replay_eos_finishes is not None else not args.sync,
    )


def run_config_from_cli(args: argparse.Namespace) -> RunConfig:
    """Project the CLI namespace onto the run-level (non-engine) fields."""

    return RunConfig(
        model=args.model,
        attn_implementation=args.attn_implementation,
        seed=args.seed,
        min_exit_step=args.min_exit_step,
        exit_delay_steps=args.exit_delay_steps,
        cuda_graph_mode=args.cuda_graph_mode,
        layer_split=args.layer_split,
    )


def build_workload_from_cli(args: argparse.Namespace) -> Workload:
    """Load a recorded bundle or synthesize a random workload, already capped to the servable context.

    ``--max-model-len`` drops prompts that leave no room to generate, so it is applied *before*
    ``--num-requests`` slices the bundle. Slicing first replays fewer requests than were asked
    for, by an amount that varies with the context length, and reports the number requested.
    Both paths therefore raise rather than quietly hand back a smaller workload: the request
    count is what throughput is measured against.

    ``--trace-duration-s`` sets the request count to ``ceil(rate * target_duration)``.
    The realized Poisson trace duration remains stochastic.
    """

    num_requests = args.num_requests
    if args.trace_duration_s is not None:
        num_requests = math.ceil(args.request_rate * args.trace_duration_s)

    if args.dataset == "random":
        if num_requests is None:
            raise ValueError("--num-requests is required for --dataset random (how many requests to synthesize)")
        if args.exit_dist is None:
            raise ValueError("--exit-dist is required for --dataset random")
        workload = Workload.synthetic(
            num_requests=num_requests,
            input_len=args.input_len,
            output_len=args.output_len,
            exit_dist=args.exit_dist,
            max_depth=args.max_recurrent_depth,
            seed=args.seed,
            input_len_high=args.input_len_high,
            output_len_high=args.output_len_high,
        ).apply_max_model_len(args.max_model_len)
        if workload.num_requests != num_requests:
            raise ValueError(
                f"synthesized {num_requests} requests but only {workload.num_requests} have a prompt "
                f"shorter than --max-model-len {args.max_model_len}"
            )
        return workload

    if args.workload is None:
        raise ValueError("--workload is required for --dataset bundle")
    if num_requests is not None and num_requests <= 0:
        raise ValueError(f"--num-requests must be positive when set, got {num_requests}")
    workload = Workload.load(args.workload).apply_max_model_len(args.max_model_len)
    if num_requests is not None and num_requests > workload.num_requests:
        raise ValueError(
            f"requested {num_requests} requests but only {workload.num_requests} in {args.workload} "
            f"fit --max-model-len {args.max_model_len}; a shorter trace or a larger bundle is needed"
        )
    return workload.take(num_requests)


def resolve_exit_policy(
    min_exit_step: int | None, exit_delay_steps: int | None, meta: dict[str, Any]
) -> tuple[int, int]:
    """Resolve the exit-schedule flags against the bundle's recorded ``depth_defaults``.

    The two values describe one recorded policy, so a bundle's defaults apply only as a
    pair: overriding one flag while silently inheriting the other from the bundle can
    produce a pairing the bundle was never recorded for (a recorded depth-1 exit served
    at depth 2 by the async delayed path, for example). Workloads without recorded
    defaults keep the legacy per-flag globals.
    """

    defaults = meta.get("depth_defaults") or {}
    if defaults and (min_exit_step is None) != (exit_delay_steps is None):
        raise ValueError(
            "--min-exit-step and --exit-delay-steps override the bundle's recorded exit policy "
            f"(min_exit_step={defaults.get('min_exit_step')}, exit_delay_steps={defaults.get('exit_delay_steps')}) "
            "as a pair; set both flags or neither"
        )
    if min_exit_step is None:
        min_exit_step = int(defaults.get("min_exit_step", 2))
    if exit_delay_steps is None:
        exit_delay_steps = int(defaults.get("exit_delay_steps", 0))
    return min_exit_step, exit_delay_steps


def _workload_id(workload_name: str, meta: dict[str, Any]) -> str:
    """Group bundles by recorded depth, size, and sampling seeds.

    Synthetic workloads carry their identity elsewhere in the summary configuration.
    """

    parts = [workload_name]
    for key, prefix in (("recur_steps", "d"), ("num_requests", "n"), ("sampling_seed", "s"), ("shuffle_seed", "sh")):
        value = meta.get(key)
        if value is not None:
            parts.append(f"{prefix}{value}")
    return "-".join(parts)


def write_summary(path: Path, summary: BenchmarkSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(summary.to_json() + "\n")


def main() -> None:
    import torch

    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    if args.nvtx:
        nvtx.set_enabled(True)
        print("warning: NVTX instrumentation enabled; timing includes profiling overhead")
    if args.max_recurrent_depth <= 0:
        raise ValueError(f"max_recurrent_depth must be positive, but got {args.max_recurrent_depth}")
    if args.max_num_seqs <= 0:
        raise ValueError(f"max_num_seqs must be positive, but got {args.max_num_seqs}")
    if args.min_free_slots is not None and args.min_free_slots <= 0:
        raise ValueError(f"--min-free-slots must be positive, but got {args.min_free_slots}")
    if args.steady_warmup_requests is not None and args.steady_warmup_requests < 0:
        raise ValueError(f"--steady-warmup-requests must be non-negative, but got {args.steady_warmup_requests}")
    if not torch.cuda.is_available() and not args.sync:
        raise RuntimeError("Default benchmark uses async CUDA. Pass --sync for CPU/debug mode.")
    # 0.0 is a meaningful boundary for a convergence-value trace: the criterion never falls
    # below it, so every token runs full depth (the no-exit control of a threshold sweep).
    if args.exit_threshold is not None and not (0 <= args.exit_threshold <= 1):
        raise ValueError(f"--exit-threshold must be in [0, 1], got {args.exit_threshold}")
    if args.request_rate is not None and args.request_rate <= 0:
        raise ValueError(f"--request-rate must be positive when set, got {args.request_rate}")
    if args.trace_duration_s is not None:
        if args.trace_duration_s <= 0:
            raise ValueError(f"--trace-duration-s must be positive when set, got {args.trace_duration_s}")
        if args.request_rate is None:
            raise ValueError("--trace-duration-s requires --request-rate (it sizes the arrival trace)")
        if args.num_requests is not None:
            raise ValueError("pass either --trace-duration-s or --num-requests, not both")
    # A recorded bundle needs a threshold to derive depths from its model-specific trajectory.
    # Synthetic workloads already carry explicit depths.
    if args.dataset == "bundle" and args.backend in EARLY_EXIT_BACKENDS and args.exit_threshold is None:
        raise ValueError(f"--exit-threshold is required for --backend {args.backend} on --dataset bundle")
    if args.kv_policy == "last_exited" and args.backend not in EARLY_EXIT_BACKENDS:
        # Copy-on-exit routing never fires without early exits, so a full-depth row labelled
        # last_exited would claim routing that did not happen.
        raise ValueError(
            f"--kv-policy last_exited requires an early-exit backend; --backend {args.backend} runs full "
            "depth, where the layout is exactly depth_indexed. Use --kv-policy depth_indexed."
        )

    workload = build_workload_from_cli(args)
    # Depth-derivation settings travel with the recorded bundle so a bare invocation replays
    # the exit schedule the bundle was recorded for; explicit flags override as a pair.
    args.min_exit_step, args.exit_delay_steps = resolve_exit_policy(
        args.min_exit_step, args.exit_delay_steps, workload.meta
    )
    workload_name = str(workload.meta.get("dataset") or (args.workload.stem if args.workload else "random"))
    workload_path = str(args.workload) if args.workload else f"random:{args.exit_dist}"
    workload_id = _workload_id(workload_name, workload.meta)

    engine_args = engine_args_from_cli(args)
    run_config = run_config_from_cli(args)

    print(f"model={args.model}")
    print(f"backend={args.backend}")
    print(f"dataset={args.dataset} workload={workload_name} ({workload.num_requests} requests)")
    print(f"max_recurrent_depth={args.max_recurrent_depth}")
    print(f"exit_threshold={args.exit_threshold} min_exit_step={args.min_exit_step} delay={args.exit_delay_steps}")
    print(f"delay_gate_consumption={not args.no_delay_gate_consumption} use_async_batching={not args.sync}")

    prepared = prepare_run(workload, engine_args, run_config)
    print(f"cuda_graph_mode={run_config.cuda_graph_mode}")
    if args.trace_output is not None:
        if args.backend != "cdb":
            raise ValueError("--trace-output requires --backend cdb")
        prepared.engine.schedule_trace = ScheduleTrace()

    arrival_offsets_s = None
    if args.request_rate is not None:
        arrival_offsets_s = poisson_arrival_offsets(workload.num_requests, args.request_rate, seed=args.arrival_seed)
        print(f"request_rate={args.request_rate} req/s arrival_seed={args.arrival_seed}")

    measured = measure_once(
        prepared,
        nvtx_label="benchmark.generate" if args.nvtx else None,
        arrival_offsets_s=arrival_offsets_s,
    )

    engine = prepared.engine
    if args.trace_output is not None:
        engine.schedule_trace.write_jsonl(args.trace_output)
        print(f"schedule_trace={args.trace_output} ({len(engine.schedule_trace.events)} launches)")
    outputs = measured.outputs
    generated_tokens = sum(len(output.generated_tokens) for output in outputs)
    completed_requests = sum(output.is_finished() for output in outputs)
    full_depth_prefix_count = first_token_full_depth_count(args.backend, engine, outputs)
    effective_exit_distribution = effective_summary_for_backend(
        args.backend,
        engine,
        max_depth=args.max_recurrent_depth,
        generated_tokens=generated_tokens,
        full_depth_prefix_count=full_depth_prefix_count,
    )
    requested = (
        requested_exit_distribution(
            prepared.materialized,
            max_depth=args.max_recurrent_depth,
            generated_tokens=generated_tokens,
            full_depth_prefix_count=full_depth_prefix_count,
        )
        if args.backend != "cb"
        else None
    )
    config = BenchmarkConfig(
        backend=args.backend,
        model=args.model,
        max_recurrent_depth=args.max_recurrent_depth,
        layer_split=args.layer_split,
        workload_path=workload_path,
        workload_name=workload_name,
        workload_id=workload_id,
        num_requests=workload.num_requests,
        measured_repeat=args.repeat_index,
        seed=args.seed,
        attn_implementation=args.attn_implementation,
        use_async_batching=not args.sync,
        use_cuda_graph=run_config.cuda_graph_mode != "none",
        cuda_graph_mode=run_config.cuda_graph_mode,
        block_size=args.block_size,
        num_blocks=engine.cache.num_blocks,
        mem_fraction_static=args.mem_fraction_static,
        max_num_batched_tokens=args.max_num_batched_tokens,
        # Read back off the scheduler rather than the CLI, so a row cannot claim a batch it never ran.
        max_num_seqs=engine.scheduler.max_num_seqs,
        # Read back off the scheduler too: the threshold is resolved against the cap.
        min_free_slots=engine.scheduler.min_free_slots,
        max_model_len=args.max_model_len,
        cdb_kv_policy=args.kv_policy,
        kv_pressure_mode=engine_args.kv_pressure_mode,
        # The scheduler zeroes the margin under kv_pressure_mode="reserve", whose admission is
        # already exact. Record what it holds, not what was requested, so a reserve row does not
        # claim a margin it never applied.
        safety_margin=engine.scheduler.safety_margin,
        cpu_offload_space=engine_args.cpu_offload_space,
        replay_eos_finishes=engine_args.replay_eos_finishes,
        exit_threshold=args.exit_threshold,
        min_exit_step=args.min_exit_step,
        exit_delay_steps=args.exit_delay_steps,
        min_recurrent_steps=args.min_recurrent_steps,
        delay_gate_consumption=not args.no_delay_gate_consumption,
        refill=args.refill,
        min_coda_batch_size=args.min_coda_batch_size,
        request_rate_rps=args.request_rate,
        arrival_seed=args.arrival_seed if args.request_rate is not None else None,
        **_length_summary(workload.input_lens, "prompt"),
        **_length_summary(workload.output_lens, "output"),
    )
    threshold_slug = "none" if args.exit_threshold is None else f"{args.exit_threshold}".replace(".", "p")
    rate_slug = "" if args.request_rate is None else f"-r{args.request_rate}".replace(".", "p")
    summary = BenchmarkSummary(
        config=config,
        run_id=f"{args.backend}-D{args.max_recurrent_depth}-{workload_name}-q{threshold_slug}{rate_slug}-repeat{args.repeat_index}",
        wall_time_s=measured.wall_time_s,
        generated_tokens=generated_tokens,
        completed_requests=completed_requests,
        requested_exit_distribution=requested,
        effective_exit_distribution=effective_exit_distribution,
        peak_cuda_memory_allocated_bytes=measured.peak_allocated,
        peak_cuda_memory_reserved_bytes=measured.peak_reserved,
        kv_cache_num_blocks=engine.cache.num_blocks,
        kv_cache_block_size=engine.cache.block_size,
        kv_cache_max_num_batched_tokens=engine.cache.max_num_batched_tokens,
        kv_cache_num_pages=engine.cache.num_pages,
        kv_cache_peak_blocks_used=engine.cache.peak_num_allocated_blocks,
        kv_cache_final_blocks_used=engine.cache.get_num_allocated_blocks(),
        first_token_full_depth_count=full_depth_prefix_count,
        stage_flops=stage_flops(engine.model),
        backend_stats=backend_stats(engine),
        request_latency=request_latency_summary(outputs),
        # Open-loop serving has no fill-steady-drain shape: its queue empties whenever arrivals pause.
        steady_state=engine.scheduler.steady_state_summary() if args.request_rate is None else None,
        device_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    )

    if args.summary_output is not None:
        write_summary(args.summary_output, summary)
    print(f"benchmark_summary={summary.to_json()}")


if __name__ == "__main__":
    main()
