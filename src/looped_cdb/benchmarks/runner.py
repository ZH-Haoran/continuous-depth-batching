"""Shared CB/CDB benchmark harness: build a run from a workload, then measure it.

This is the single measured path used by every benchmark script. A caller supplies
a :class:`Workload` (recorded or synthetic), typed :class:`EngineArgs`, and a
:class:`RunConfig`; :func:`prepare_run` loads the model, derives the replay schedule,
builds prompts and the engine, and warms up; :func:`measure_once` runs one timed
generation and returns a :class:`MeasuredRun`. Summary construction stays with the
caller (scripts differ in what they emit), using the shared distribution/stat helpers
here so requested-vs-effective accounting is identical everywhere.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from looped_cdb.benchmarks import nvtx
from looped_cdb.benchmarks.metrics import ExitDistributionSummary
from looped_cdb.benchmarks.workload import Workload
from looped_cdb.models.ouro.kv_cache_policy import configure_kv_cache_policy
from looped_cdb.utils import KVPressureMode

if TYPE_CHECKING:
    import torch
    from transformers.tokenization_utils import PreTrainedTokenizer
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

    from looped_cdb.continuous_batching import ContinuousBatchingEngine
    from looped_cdb.continuous_depth_batching import ContinuousDepthBatchingEngine

    Engine = ContinuousBatchingEngine | ContinuousDepthBatchingEngine

DEFAULT_MODEL_ID = "KristianS7/Ouro-1.4B"
BACKENDS = ("cb", "cdb")
EARLY_EXIT_BACKENDS = ("cdb",)

# Output cap per warmup request (see :func:`warm_up_engine`): long enough to span several
# recurrent-depth waves at the deepest model served (r_max=16), short enough that the pass
# costs seconds.
WARMUP_OUTPUT_TOKENS = 32


# --------------------------------------------------------------------------- typed config


@dataclass(frozen=True)
class EngineArgs:
    """Typed engine-construction arguments shared by every benchmark caller.

    ``build_engine`` and ``_delayed_synthetic_active`` read exactly these fields.
    Requiring every field (no defaults) makes a non-CLI caller fail at construction
    if it omits one, rather than raising ``AttributeError`` deep in a later check.
    """

    backend: str
    block_size: int
    num_blocks: int | None
    max_num_batched_tokens: int
    # Max sequence length (prompt + generated) per request, vLLM-style; also sizes the fast decode block table.
    max_model_len: int
    max_recurrent_depth: int
    # The decode batch, for both backends: the resident cap, which is also the widest launch.
    max_num_seqs: int
    # Prefill admission, for both backends: the resident slots that must stand open before a prefill
    # launch runs.
    min_free_slots: int | None
    min_recurrent_steps: int
    no_delay_gate_consumption: bool
    kv_policy: str
    kv_slots_per_layer: int | None
    exit_threshold: float | None
    sync: bool
    # ``cdb`` only: refill freed depth slots (continuous depth batching) vs. the
    # sequence-level no-refill baseline. Ignored by ``cb`` (always full depth).
    refill: bool
    # ``cdb`` only: hold the coda bucket until this many exited tokens wait, amortizing the coda
    # stage's weight reload per launch. 1 launches every coda immediately.
    min_coda_batch_size: int
    # Fraction of free KV blocks below which prefill admission stops, for both backends. 0.2
    # mirrors production serving; 0.0 drains all prefill before any decode (the decode-latency
    # measurement needs this).
    safety_margin: float
    # KV-cache-pressure policy: "none" (no reservation/preemption; raise on full - for workloads sized
    # to fit, e.g. the decode-latency microbenchmark), "reserve", "recompute", or "offload".
    kv_pressure_mode: KVPressureMode
    # Pinned CPU swap-pool budget (GiB); required (> 0) for kv_pressure_mode="offload", else None.
    cpu_offload_space: float | None
    # Completions before the steady-state throughput window opens (see
    # ``BaseServingScheduler.steady_state_summary``); ``None`` resolves to twice the resident cap.
    steady_warmup_requests: int | None = None
    # Simulate sampled-EOS finishes under replay: the length finish is discovered at consume time
    # and one lagged token per request is computed and discarded (see the engine configs).
    replay_eos_finishes: bool = False
    mem_fraction_static: float | None = None


@dataclass(frozen=True)
class RunConfig:
    """Everything a run needs beyond the workload and engine construction."""

    model: str
    attn_implementation: str
    seed: int
    min_exit_step: int
    exit_delay_steps: int
    cuda_graph_mode: str  # "all" | "decode" | "none"
    # Huginn only: prelude-core-coda layer-split override, e.g. "1-4-1"
    layer_split: str | None = None


@dataclass
class PreparedRun:
    """A loaded, warmed-up run ready to measure one or more times."""

    engine: Any
    input_ids: list[list[int]]
    max_new_tokens: list[int]
    exit_depths: list[list[int]] | None
    materialized: list[list[int]] | None
    model_kwargs: dict[str, Any]
    backend: str
    max_recurrent_depth: int


@dataclass
class MeasuredRun:
    """The result of one timed generation."""

    outputs: list[Any]
    wall_time_s: float
    peak_allocated: int | None
    peak_reserved: int | None


def _delayed_synthetic_active(engine_args: EngineArgs) -> bool:
    """Return whether CDB replay uses the async one-step-delayed decision policy.

    cdb always replays a schedule (from a recorded threshold or explicit
    synthetic depths), so this tracks the engine's ``delay_gate_consumption`` rather
    than whether a threshold was given. Both refill and no-refill run the async
    delayed-gate path by default, so the delayed replay offset applies to either.
    """

    return engine_args.backend == "cdb" and not engine_args.sync and not engine_args.no_delay_gate_consumption


CUDA_GRAPH_MODES = ("all", "decode", "none")


def resolve_cuda_graph(cuda_graph_mode: str) -> tuple[bool, bool]:
    """Map the CUDA-graph mode onto the engine's two graph switches, ``(decode, prefill)``.

    "all" graphs the decode launches and the varlen prefill forward, "decode" only the former,
    "none" runs eagerly. Graph replay is independent of scheduler serialization; combine "none"
    with ``sync`` to bisect on an eager, fully synchronous engine.
    """

    if cuda_graph_mode not in CUDA_GRAPH_MODES:
        raise ValueError(f"cuda_graph_mode must be one of {CUDA_GRAPH_MODES}, got {cuda_graph_mode!r}")
    return cuda_graph_mode != "none", cuda_graph_mode == "all"


# --------------------------------------------------------------------------- prompts / replay


def resolve_filler_token_id(tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast) -> int:
    """Return a single fixed, in-vocab, non-special id for constant filler prompts."""

    bos = getattr(tokenizer, "bos_token_id", None)
    return bos if isinstance(bos, int) and bos >= 0 else 0


def _prompt_lengths(input_lens: np.ndarray) -> list[int]:
    """Validate and return per-request prompt lengths (all >= 1)."""

    lengths = [int(n) for n in input_lens]
    invalid = [idx for idx, n in enumerate(lengths) if n < 1]
    if invalid:
        raise ValueError(f"input_lens must be >= 1 for every request, but got <1 at indices {invalid[:5]}")
    return lengths


def build_prompts(input_lens: np.ndarray, *, tokenizer: Any, seed: int = 0) -> list[list[int]]:
    """Build seeded random in-vocab prompt token ids of the requested per-request lengths.

    Exit depths are replayed from the schedule (not the live gate) and only prompt
    length affects KV/compute, so the token *content* is free. We use random ids rather
    than a constant filler so the pad id (whose embedding is zeroed) is never a whole
    prompt, and so the harness stays correct if a live gate is ever enabled. Deterministic
    given ``seed``.
    """

    lengths = _prompt_lengths(input_lens)
    vocab_size = int(getattr(tokenizer, "vocab_size", None) or len(tokenizer))
    avoid_id = getattr(tokenizer, "pad_token_id", None)
    rng = np.random.default_rng(seed)
    prompts: list[list[int]] = []
    for length in lengths:
        ids = rng.integers(0, vocab_size, size=length)
        if avoid_id is not None and 0 <= avoid_id < vocab_size:
            ids[ids == avoid_id] = (avoid_id + 1) % vocab_size
        prompts.append(ids.astype(int).tolist())
    return prompts


def build_filler_prompts(input_lens: np.ndarray, filler_token_id: int) -> list[list[int]]:
    """Build constant-filler prompt token ids of the requested lengths (deterministic helper)."""

    return [[filler_token_id] * n for n in _prompt_lengths(input_lens)]


def materialized_to_replay(materialized: list[list[int]], *, delayed: bool) -> list[list[int]]:
    """Convert 1-indexed per-request depths into a 0-based replay schedule.

    The first output token per request is produced at full depth (CDB prefill coda /
    CB prefill), so it is dropped. The remaining depths become 0-based path-offset exit
    steps: ``depth - 1`` for the immediate policy, or ``depth - 2`` for the CDB async
    delayed policy (which applies a decision one recurrent step later). A depth of 1 is
    unrepresentable on the delayed path and raises.
    """

    offset = 2 if delayed else 1
    schedule: list[list[int]] = []
    for request_idx, depths in enumerate(materialized):
        row = []
        for depth in depths[1:]:
            value = depth - offset
            if value < 0:
                raise ValueError(
                    f"request {request_idx}: recorded depth {depth} cannot be replayed on the delayed CDB path "
                    "(needs depth >= 2). Pass --min-exit-step 2, or --no-delay-gate-consumption/--sync."
                )
            row.append(value)
        schedule.append(row)
    return schedule


def _length_summary(lengths: np.ndarray, prefix: str) -> dict[str, int | float | None]:
    """Return compact length-summary fields under ``<prefix>_*`` keys."""

    values = [int(n) for n in lengths]
    total = sum(values)
    return {
        f"{prefix}_total_tokens": total,
        f"{prefix}_min_tokens": min(values) if values else None,
        f"{prefix}_max_tokens": max(values) if values else None,
        f"{prefix}_mean_tokens": total / len(values) if values else 0.0,
    }


# --------------------------------------------------------------------------- CUDA helpers


def maybe_sync_cuda() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak_memory_stats() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory_stats() -> tuple[int | None, int | None]:
    import torch

    if not torch.cuda.is_available():
        return None, None
    return torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()


# --------------------------------------------------------------------------- model / engine


def load_serving_model(
    model_id: str,
    *,
    max_recurrent_depth: int,
    attn_implementation: str,
    dtype: Any,
    device: str,
    kv_policy: str = "single",
    kv_slots_per_layer: int | None = None,
    layer_split: str | None = None,
) -> Any:
    """Load a looped causal LM for serving benchmarks, dispatching on the model family.

    Ouro and Huginn are loaded through the in-repo modeling code with the recurrent
    depth and KV layout applied at load time. ``kv_policy`` selects the recurrent KV
    layout; ``depth_indexed`` keeps one slot per recursion, which the decode-latency
    benchmark uses to compare layouts at equal per-step latency. ``layer_split``
    overrides Huginn's prelude-core-coda layer counts for boundary-stage throughput
    ablations; Ouro checkpoints fix their own layout, so it raises there.
    """

    from looped_cdb.eval.model_loading import load_huginn_model, resolve_model_family

    if resolve_model_family(model_id) == "huginn":
        return load_huginn_model(
            model_id,
            recur_steps=max_recurrent_depth,
            kv_policy=kv_policy,
            kv_slots_per_layer=kv_slots_per_layer,
            state_init="zero",
            layer_split=layer_split,
            dtype=dtype,
            attn_impl=attn_implementation,
            device=device,
        )

    if layer_split is not None:
        raise ValueError(f"layer_split is only supported for Huginn models, got model {model_id!r}")

    from looped_cdb.models.ouro import OuroConfig, OuroForCausalLM

    config = OuroConfig.from_pretrained(model_id)
    config.total_ut_steps = int(max_recurrent_depth)
    model = OuroForCausalLM.from_pretrained(
        model_id,
        config=config,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    model = model.to(device).eval()
    model.set_attn_implementation(attn_implementation)
    configure_kv_cache_policy(model, kv_policy=kv_policy, kv_slots_per_layer=kv_slots_per_layer)
    return model


def set_recurrent_depth(model: Any, depth: int) -> None:
    """Make the recurrent-depth setting explicit on a loaded Ouro or Huginn model."""

    model_type = str(getattr(getattr(model, "config", None), "model_type", ""))
    if model_type.startswith("huginn"):
        model.config.total_recurrent_steps = depth
        return
    if hasattr(model, "config"):
        model.config.total_ut_steps = depth
    with suppress(AttributeError):
        model.total_ut_steps = depth
    inner_model = getattr(model, "model", None)
    if inner_model is not None and hasattr(inner_model, "config"):
        inner_model.config.total_ut_steps = depth
        with suppress(AttributeError):
            inner_model.total_ut_steps = depth


def build_engine(
    engine_args: EngineArgs,
    model: Any,
    *,
    dtype: torch.dtype,
    use_cuda_graph: bool,
    use_cuda_graph_prefill: bool,
) -> Engine:
    """Create the selected benchmark backend."""

    from looped_cdb.continuous_batching import ContinuousBatchingEngine
    from looped_cdb.continuous_batching.config import ContinuousBatchingConfig
    from looped_cdb.continuous_depth_batching import ContinuousDepthBatchingConfig, ContinuousDepthBatchingEngine

    common_kwargs = {
        "num_blocks": engine_args.num_blocks,
        "mem_fraction_static": engine_args.mem_fraction_static,
        "max_num_batched_tokens": engine_args.max_num_batched_tokens,
        "max_num_seqs": engine_args.max_num_seqs,
        "min_free_slots": engine_args.min_free_slots,
        "block_size": engine_args.block_size,
        "max_model_len": engine_args.max_model_len,
        "use_async_batching": not engine_args.sync,
        "use_cuda_graph": use_cuda_graph,
        "use_cuda_graph_prefill": use_cuda_graph_prefill,
        "safety_margin": engine_args.safety_margin,
        "kv_pressure_mode": engine_args.kv_pressure_mode,
        "cpu_offload_space": engine_args.cpu_offload_space,
        "replay_eos_finishes": engine_args.replay_eos_finishes,
    }
    engine: Engine
    if engine_args.backend == "cb":
        engine = ContinuousBatchingEngine.from_model(
            model=model,
            cb_config=ContinuousBatchingConfig(**common_kwargs),
            dtype=dtype,
        )
    else:
        engine = ContinuousDepthBatchingEngine.from_model(
            model=model,
            cdb_config=ContinuousDepthBatchingConfig(
                **common_kwargs,
                max_recurrent_steps=engine_args.max_recurrent_depth,
                min_recurrent_steps=engine_args.min_recurrent_steps,
                delay_gate_consumption=not engine_args.no_delay_gate_consumption,
                kv_policy=engine_args.kv_policy,
                kv_slots_per_layer=engine_args.kv_slots_per_layer,
                synthetic_exit_replay=True,
                refill=engine_args.refill,
                min_coda_batch_size=engine_args.min_coda_batch_size,
            ),
            dtype=dtype,
        )
    return engine


# --------------------------------------------------------------------------- generation


def generate_once(
    engine: Any,
    input_ids: list[list[int]],
    *,
    backend: str,
    max_new_tokens: list[int],
    eos_token_id: int | list[int] | None,
    warmup: bool,
    model_kwargs: dict[str, Any] | None,
    exit_depths: list[list[int]] | None = None,
    nvtx_label: str | None = None,
    nvtx_registered: bool = False,
    arrival_offsets_s: list[float] | None = None,
    record_token_times: bool = False,
) -> list[Any]:
    """Run one generation call against the selected backend."""

    range_context = nvtx.range(nvtx_label, registered=nvtx_registered) if nvtx_label is not None else nullcontext()
    with range_context:
        if backend == "cb":
            return engine.generate_batch(
                input_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                warmup=warmup,
                model_kwargs=model_kwargs,
                arrival_offsets_s=arrival_offsets_s,
                record_token_times=record_token_times,
            )
        return engine.generate_batch(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            warmup=warmup,
            model_kwargs=model_kwargs,
            exit_depths=exit_depths,
            arrival_offsets_s=arrival_offsets_s,
            record_token_times=record_token_times,
        )


def warm_up_engine(
    engine: Any,
    *,
    backend: str,
    input_ids: list[list[int]],
    max_new_tokens: list[int],
    exit_depths: list[list[int]] | None,
    model_kwargs: dict[str, Any] | None,
) -> None:
    """Warm the engine on a short slice of the workload before timing.

    The pass runs with ``warmup=True``, so the engine first pre-captures its CUDA graphs on
    synthetic state: the decode fast path for the full-depth engine, the prelude, recurrent and
    coda stages for the depth engine, each at every batch bucket a launch can land in, and the
    breakable prefill graph at every token bucket. The replay then drives the real path on real
    prompts, so first-touch allocations and host staging buffers land here instead of in the
    measured run.

    Twice the resident cap of requests at :data:`WARMUP_OUTPUT_TOKENS` output tokens each fills
    the decode batch and turns it over, so the cost is seconds and scales with the engine's
    width, not the workload.
    """

    resident_cap = engine.scheduler.max_num_seqs
    num_requests = max(1, min(len(input_ids), 2 * int(resident_cap)))
    capped_new_tokens = [min(tokens, WARMUP_OUTPUT_TOKENS) for tokens in max_new_tokens[:num_requests]]
    # Each replay schedule is cut with its request, keeping the engine's
    # ``len(exit_depths[i]) == max_new_tokens[i] - 1`` invariant.
    sliced_exit_depths = (
        [depths[: tokens - 1] for depths, tokens in zip(exit_depths[:num_requests], capped_new_tokens, strict=True)]
        if exit_depths is not None
        else None
    )
    maybe_sync_cuda()
    generate_once(
        engine,
        input_ids[:num_requests],
        backend=backend,
        max_new_tokens=capped_new_tokens,
        eos_token_id=None,
        warmup=True,
        model_kwargs=model_kwargs,
        exit_depths=sliced_exit_depths,
    )
    maybe_sync_cuda()


# --------------------------------------------------------------------------- prepare / measure


def prepare_run(
    workload: Workload,
    engine_args: EngineArgs,
    run_config: RunConfig,
    *,
    model: Any = None,
    tokenizer: Any = None,
) -> PreparedRun:
    """Load the model, derive the replay schedule, build prompts + engine, and warm up.

    ``model`` and ``tokenizer`` may be supplied to reuse a preloaded model across many
    runs (e.g. a latency sweep that rebuilds only the engine per point) instead of
    reloading from disk each call; when ``None`` they are loaded from ``run_config.model``.
    A supplied model must already match ``engine_args`` (recurrent depth, KV policy); a
    caller that sweeps the recurrent depth should set it via :func:`set_recurrent_depth`
    before each call so the warmup captures graphs at that depth.
    """

    import torch
    from transformers import AutoTokenizer

    backend = engine_args.backend
    use_cuda_graph, use_cuda_graph_prefill = resolve_cuda_graph(run_config.cuda_graph_mode)

    max_new_tokens = [int(n) for n in workload.output_lens]
    if any(n < 1 for n in max_new_tokens):
        raise ValueError("workload output_lens must all be >= 1")

    materialized: list[list[int]] | None = None
    if backend in EARLY_EXIT_BACKENDS or engine_args.exit_threshold is not None:
        materialized = workload.materialize_depths(
            threshold=engine_args.exit_threshold,
            min_exit_step=run_config.min_exit_step,
            exit_delay_steps=run_config.exit_delay_steps,
        )
        for idx, depths in enumerate(materialized):
            if len(depths) != max_new_tokens[idx]:
                raise ValueError(
                    f"materialized depths for request {idx} have {len(depths)} entries but output_len is "
                    f"{max_new_tokens[idx]}"
                )

    exit_depths: list[list[int]] | None = None
    if backend == "cdb":
        exit_depths = materialized_to_replay(materialized, delayed=_delayed_synthetic_active(engine_args))

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model is None:
        model = load_serving_model(
            run_config.model,
            max_recurrent_depth=engine_args.max_recurrent_depth,
            attn_implementation=run_config.attn_implementation,
            dtype=dtype,
            device=device,
            kv_policy=engine_args.kv_policy,
            kv_slots_per_layer=engine_args.kv_slots_per_layer,
            layer_split=run_config.layer_split,
        )
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(run_config.model, trust_remote_code=True)
    input_ids = build_prompts(workload.input_lens, tokenizer=tokenizer, seed=run_config.seed)
    engine = build_engine(
        engine_args, model, dtype=dtype, use_cuda_graph=use_cuda_graph, use_cuda_graph_prefill=use_cuda_graph_prefill
    )
    # Replay drives the exit decision, but for cdb the live gate still runs every recurrent step so
    # its compute and GPU->CPU readout are timed (a blocking stall on --sync, overlapped by the
    # lookahead on the async path) instead of being hidden. cb runs full depth and never consults
    # the gate, so leave it off there.
    model_kwargs = {"use_early_exit_gate": backend == "cdb"}

    warm_up_engine(
        engine,
        backend=backend,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        exit_depths=exit_depths,
        model_kwargs=model_kwargs,
    )
    # Armed after warm-up so the warm-up replay cannot open a window, and resolved against the cap the
    # scheduler applied, which the token budget may have clamped.
    engine.scheduler.steady_warmup_requests = (
        2 * engine.scheduler.max_num_seqs
        if engine_args.steady_warmup_requests is None
        else engine_args.steady_warmup_requests
    )
    return PreparedRun(
        engine=engine,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        exit_depths=exit_depths,
        materialized=materialized,
        model_kwargs=model_kwargs,
        backend=backend,
        max_recurrent_depth=engine_args.max_recurrent_depth,
    )


def measure_once(
    prepared: PreparedRun,
    *,
    nvtx_label: str | None = None,
    arrival_offsets_s: list[float] | None = None,
    record_token_times: bool = False,
) -> MeasuredRun:
    """Run one timed generation against a prepared run.

    ``eos_token_id`` is None so every request emits exactly its recorded output length
    (fixed work). CDB resets its exit stats per generate_batch, so the measured
    histogram reflects only this run. ``arrival_offsets_s`` switches the run open-loop
    (timed request release, see :mod:`looped_cdb.arrivals`); warmup replays its slice
    upfront regardless, so graph capture and first touch precede timing either way.
    """

    maybe_sync_cuda()
    reset_peak_memory_stats()
    start = time.perf_counter()
    range_context = nvtx.range(nvtx_label, registered=True) if nvtx_label is not None else nullcontext()
    with range_context:
        outputs = generate_once(
            prepared.engine,
            prepared.input_ids,
            backend=prepared.backend,
            max_new_tokens=prepared.max_new_tokens,
            eos_token_id=None,
            warmup=False,
            model_kwargs=prepared.model_kwargs,
            exit_depths=prepared.exit_depths,
            arrival_offsets_s=arrival_offsets_s,
            record_token_times=record_token_times,
        )
        maybe_sync_cuda()
    wall_time_s = time.perf_counter() - start
    peak_allocated, peak_reserved = peak_memory_stats()
    return MeasuredRun(
        outputs=outputs,
        wall_time_s=wall_time_s,
        peak_allocated=peak_allocated,
        peak_reserved=peak_reserved,
    )


def release_engine(prepared: PreparedRun) -> None:
    """Free a prepared run's engine and its KV cache, keeping any shared model alive.

    Drops the engine reference (and with it the paged KV cache) and returns freed
    blocks to the allocator, so a sweep that rebuilds an engine per point does not
    accumulate GPU memory. The model is referenced elsewhere (e.g. a
    :class:`ServingSession`) and is deliberately not touched.
    """

    import gc

    import torch

    prepared.engine = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class ServingSession:
    """A loaded serving model (Ouro or Huginn) and tokenizer, reused across many prepared runs.

    Loading the model is the expensive step, so a sweep over batch size, context
    length, or recurrent depth loads it once and rebuilds only the engine per point.
    Construct once, optionally :meth:`set_depth`, then use :meth:`prepared` per point;
    its context manager frees that point's engine (and KV cache) on exit while the
    model stays resident.
    """

    def __init__(
        self,
        run_config: RunConfig,
        *,
        max_recurrent_depth: int,
        kv_policy: str = "single",
        kv_slots_per_layer: int | None = None,
    ) -> None:
        import torch
        from transformers import AutoTokenizer

        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = load_serving_model(
            run_config.model,
            max_recurrent_depth=max_recurrent_depth,
            attn_implementation=run_config.attn_implementation,
            dtype=self.dtype,
            device=device,
            kv_policy=kv_policy,
            kv_slots_per_layer=kv_slots_per_layer,
            layer_split=run_config.layer_split,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(run_config.model, trust_remote_code=True)

    def set_depth(self, depth: int) -> None:
        """Set the recurrent depth on the reused model; the next engine graphs at this depth."""

        set_recurrent_depth(self.model, depth)

    @contextmanager
    def prepared(self, workload: Workload, engine_args: EngineArgs, run_config: RunConfig) -> Iterator[PreparedRun]:
        """Prepare a run against the reused model, freeing its engine + KV cache on exit."""

        prepared = prepare_run(workload, engine_args, run_config, model=self.model, tokenizer=self.tokenizer)
        try:
            yield prepared
        finally:
            release_engine(prepared)


# --------------------------------------------------------------------------- distribution / stat helpers


def first_token_full_depth_count(backend: str, engine: Any, outputs: list[Any]) -> int:
    """Return how many generated tokens were forced through full-depth prefill."""

    if backend == "cdb":
        from looped_cdb.continuous_depth_batching import ContinuousDepthBatchingEngine

        assert isinstance(engine, ContinuousDepthBatchingEngine)
        return sum(len(output.generated_tokens) for output in outputs) - engine.last_stats.coda_tokens
    return sum(1 for output in outputs if output.generated_tokens)


def requested_exit_distribution(
    materialized: list[list[int]] | None,
    *,
    max_depth: int,
    generated_tokens: int,
    full_depth_prefix_count: int,
) -> ExitDistributionSummary:
    """Summarize the requested exit opportunity (or full depth when no schedule)."""

    if materialized is None:
        return ExitDistributionSummary.from_depths("all_full", max_depth, [max_depth] * generated_tokens)
    histogram: dict[int, int] = {max_depth: full_depth_prefix_count}
    for row in materialized:
        for depth in row[1:]:
            histogram[depth] = histogram.get(depth, 0) + 1
    return ExitDistributionSummary.from_histogram(kind="workload", max_depth=max_depth, depth_histogram=histogram)


def effective_summary_for_backend(
    backend: str,
    engine: Any,
    *,
    max_depth: int,
    generated_tokens: int,
    full_depth_prefix_count: int,
) -> ExitDistributionSummary:
    """Build the effective recurrent-work distribution actually paid by a backend."""

    if backend == "cb":
        return ExitDistributionSummary.from_depths("all_full", max_depth, [max_depth] * generated_tokens)

    histogram: dict[int, int] = {max_depth: full_depth_prefix_count}
    from looped_cdb.continuous_depth_batching import ContinuousDepthBatchingEngine

    assert isinstance(engine, ContinuousDepthBatchingEngine)
    for exit_step, count in engine.last_stats.exit_depth_histogram.items():
        depth = exit_step + 1
        histogram[depth] = histogram.get(depth, 0) + count
    return ExitDistributionSummary.from_histogram(kind="workload", max_depth=max_depth, depth_histogram=histogram)


def backend_stats(engine: Any) -> dict[str, Any] | None:
    """Return backend-specific counters that are useful for benchmark debugging.

    Both engines report KV-cache-pressure preemption counters (from the shared offloading manager)
    and a common set of prefill/decode batch counters, so a plotting script can read a CB row and a
    CDB row through the same keys. The shared keys are:

    - ``prefill_batches`` / ``mean_prefill_batch_size``: prefill batch count and mean requests per
      prefill batch. Prefill is identical across the backends (a full-depth forward over prompt
      tokens), so these compare directly.
    - ``decode_batches`` / ``mean_decode_batch_size``: the backend's decode-side launch count and mean
      launch width. The counts compare directly as launches per generated token, which is what the
      depth engine pays extra for: CDB's ``decode_batches`` sums its recurrent and coda batches, since
      both are launches, while CB issues one launch that runs the prelude, the whole recurrent stack,
      and the coda together. The widths are the same concept in different units: a CB decode launch
      advances whole sequences one token at full depth, while a CDB recurrent launch carries token-steps
      at a single depth.

    The CDB engine additionally reports its native scheduler, depth-histogram, and recurrent-graph
    counters, including the ``recurrent_batches`` and ``coda_batches`` totals that ``decode_batches``
    sums, and their separate mean widths.
    """

    from looped_cdb.continuous_depth_batching import ContinuousDepthBatchingEngine
    from looped_cdb.continuous_depth_batching.continuous_api import PRELUDE_SITES

    collected: dict[str, Any] = {}
    manager = getattr(engine, "offloading_manager", None)
    if manager is not None:
        collected.update(manager.preemption_stats())
    scheduler = getattr(engine, "scheduler", None)
    if scheduler is not None:
        # Peak resident set against the cap that was applied: together they say whether the cap bound
        # this run. Reported for both engines, since the cap is shared.
        collected["max_resident_requests"] = scheduler.max_resident_requests
        # Mean residency over scheduler ticks, distinct from the stage widths reported below.
        collected["mean_resident_requests"] = scheduler.mean_resident_requests
        collected["max_num_seqs"] = scheduler.max_num_seqs
    runner = getattr(engine, "runner", None)
    if runner is not None and runner.warmup_report is not None:
        # Graph counts, capture seconds and the device / pinned-host memory warm-up pinned
        # down, so a row carries the cost of its graphs next to the throughput they buy.
        collected["warmup"] = runner.warmup_report.as_dict()

    if not isinstance(engine, ContinuousDepthBatchingEngine):
        cb_stats = getattr(engine, "last_stats", None)
        if cb_stats is not None:
            collected.update(
                {
                    "prefill_batches": cb_stats.prefill_batches,
                    "mean_prefill_batch_size": cb_stats.mean_prefill_batch_size,
                    "mean_prefill_query_tokens": cb_stats.mean_prefill_query_tokens,
                    "decode_batches": cb_stats.decode_batches,
                    "mean_decode_batch_size": cb_stats.mean_decode_batch_size,
                }
            )
        return collected or None
    stats = engine.last_stats
    collected.update(
        {
            "prefill_batches": stats.prefill_batches,
            "mean_prefill_batch_size": stats.mean_prefill_batch_size,
            "decode_batches": stats.decode_batches,
            "mean_decode_batch_size": stats.mean_decode_batch_size,
            "recurrent_batches": stats.recurrent_batches,
            "mean_recurrent_batch_size": stats.mean_recurrent_batch_size,
            "mean_coda_batch_size": stats.mean_coda_batch_size,
            "prelude_batches": {
                site: int(stats.prelude_batches[site]) for site in PRELUDE_SITES if stats.prelude_batches[site]
            },
            "prelude_tokens": {
                site: int(stats.prelude_tokens[site]) for site in PRELUDE_SITES if stats.prelude_batches[site]
            },
            "prelude_gathered_tokens": {
                site: int(stats.prelude_gathered_tokens[site])
                for site in PRELUDE_SITES
                if stats.prelude_gathered_tokens[site]
            },
            "prelude_fused_batches": {
                site: int(stats.prelude_fused_batches[site])
                for site in PRELUDE_SITES
                if stats.prelude_fused_batches[site]
            },
            "mean_prelude_batch_size": {
                site: stats.prelude_tokens[site] / stats.prelude_batches[site]
                for site in PRELUDE_SITES
                if stats.prelude_batches[site]
            },
            "cancelled_depth_items": stats.cancelled_depth_items,
            "mixed_recurrent_batches": stats.mixed_recurrent_batches,
            "coda_batches": stats.coda_batches,
            "recurrent_steps": stats.recurrent_steps,
            "prefill_recurrent_steps": stats.prefill_recurrent_steps,
            "coda_tokens": stats.coda_tokens,
            "prefill_coda_tokens": stats.prefill_coda_tokens,
            "fixed_depth_recurrent_steps": stats.fixed_depth_recurrent_steps,
            "skipped_recurrent_steps": stats.skipped_recurrent_steps,
            "scalar_split_count": stats.scalar_split_count,
            "scheduler_refills": stats.scheduler_refills,
            "max_depth_queue_size": stats.max_depth_queue_size,
            "max_in_flight_requests": stats.max_in_flight_requests,
            "max_live_requests": stats.max_live_requests,
            "max_growing_requests": stats.max_growing_requests,
            "scheduler_ticks": stats.scheduler_ticks,
            "coda_suppressed_ticks": stats.coda_suppressed_ticks,
            "max_pending_coda_results": stats.max_pending_coda_results,
            "coda_pipeline_blocked_ticks": stats.coda_pipeline_blocked_ticks,
            "mean_coda_latency_ticks": stats.mean_coda_latency_ticks,
            "max_mixed_depths_per_batch": stats.max_mixed_depths_per_batch,
            "recurrent_graph_hits": stats.recurrent_graph_hits,
            "recurrent_graph_captures": stats.recurrent_graph_captures,
            "stage_graph_hits": stats.stage_graph_hits,
            "stage_graph_captures": stats.stage_graph_captures,
            "exit_depth_histogram": {
                str(key + 1): int(value) for key, value in sorted(stats.exit_depth_histogram.items())
            },
            "recurrent_step_histogram": {
                str(key + 1): int(value) for key, value in sorted(stats.recurrent_step_histogram.items())
            },
            "mixed_recurrent_depth_histogram": {
                str(key + 1): int(value) for key, value in sorted(stats.mixed_recurrent_depth_histogram.items())
            },
        }
    )
    return collected
