"""High-level token API for continuous depth batching."""

from __future__ import annotations

from collections import Counter, deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Literal, get_args

import torch
from torch import nn

from looped_cdb.arrivals import ArrivalQueue
from looped_cdb.benchmarks import nvtx
from looped_cdb.kv_cache_policy import LoopedKvLayout, copies_exit_kv, resolve_kv_slots_per_layer
from looped_cdb.kv_cache_sizing import resolve_num_blocks
from looped_cdb.offloading_manager import OffloadingManager
from looped_cdb.paged_cache_geometry import PagedKVCacheGeometry
from looped_cdb.scheduler_base import resolve_max_num_seqs
from looped_cdb.utils import cap_max_new_tokens_to_model_len, normalize_max_new_tokens

from .cache import PagedAttentionCache
from .config import ContinuousDepthBatchingConfig
from .exit_policy import ExitPolicy
from .input_outputs import ContinuousDepthBatchingIOs
from .model_adapter import CDBModelAdapter, resolve_cdb_model_adapter
from .model_runner import CODA_PIPELINE_DEPTH, ModelRunner, PendingCodaResult, PendingPrefillResult
from .requests import (
    TMP_TOKEN_ID,
    DepthWorkItem,
    FutureRequestState,
    GenerationOutput,
    PendingGateResult,
    RequestState,
    RequestStatus,
)
from .schedule_trace import QueueSnapshot, ScheduleTrace, TraceStage
from .scheduler import CDBScheduler, ScheduledCDBBatch

# One O(active) scan per this many scheduler ticks. Frequent enough to catch the peak of a curve that
# moves on the timescale of a request's lifetime, cheap enough not to show up in the tick.
_GROWING_SAMPLE_EVERY = 64


class CacheFullError(RuntimeError):
    """Raised when the KV cache cannot place any work and no preemption can relieve the pressure."""


#: Where a prelude launch runs. ``after_prefill`` takes each request's first decode token as its
#: prompt retires; ``after_prefill_eager`` takes it straight from the prefill batch's device
#: tokens, before the host saw them; ``after_coda_eager`` takes an exiting token's successor
#: straight from the coda batch's device tokens; ``after_coda_staged`` catches the successors the
#: eager path skipped; ``resumed`` covers offload restores and restaged stalled decoders;
#: ``wave_cohort`` is the no-refill loop staging a whole decode wave, and is that loop's only
#: site: carried boundary successors are gathered from device tokens inside the same single
#: launch, never in a launch of their own. Each site matches the ``cdb.prelude.<site>`` NVTX range
#: wrapping its launch.
PreludeSite = Literal[
    "after_prefill", "after_prefill_eager", "after_coda_eager", "after_coda_staged", "resumed", "wave_cohort"
]
PRELUDE_SITES: tuple[PreludeSite, ...] = get_args(PreludeSite)


@dataclass
class CarriedPreludeEntries:
    """Successor items staged for a no-refill wave whose tokens the host has not seen yet.

    ``items`` hold placeholder token ids; ``rows`` locate each item's sampled token in
    ``result``'s device tokens. The wave's single prelude launch gathers them from there,
    so the boundary readback never blocks the next wave's entry.
    """

    result: PendingCodaResult | PendingPrefillResult
    items: list[DepthWorkItem]
    rows: list[int]


@dataclass
class NoRefillBoundary:
    """The work one no-refill iteration leaves in flight for the next.

    ``carried`` are successors staged before their prelude and ``coda`` is the boundary coda
    holding their sampled tokens; the next wave's single prelude launch gathers the two together.
    ``prefills`` are the prompt readbacks this iteration's prefill batches left outstanding, which
    the loop applies at the top of the next iteration, before the coda that produces each of those
    requests' second token.
    """

    carried: list[CarriedPreludeEntries] = field(default_factory=list)
    coda: PendingCodaResult | None = None
    prefills: list[PendingPrefillResult] = field(default_factory=list)

    def num_carried_items(self) -> int:
        """How many launch slots the carried successors already claim in the next wave."""

        return sum(len(entries.items) for entries in self.carried)

    def take_prefills(self) -> list[PendingPrefillResult]:
        """Remove and return the outstanding prompt readbacks."""

        prefills, self.prefills = self.prefills, []
        return prefills

    def take_decode_carry(self) -> tuple[list[CarriedPreludeEntries], PendingCodaResult | None]:
        """Remove and return the successors and boundary coda the next wave enters through."""

        carried, coda = self.carried, self.coda
        self.carried, self.coda = [], None
        return carried, coda


@dataclass
class ContinuousDepthBatchingStats:
    """Runtime counters for CDB verification and benchmarking."""

    prefill_batches: int = 0
    prefill_requests: int = 0
    recurrent_batches: int = 0
    coda_batches: int = 0
    recurrent_steps: int = 0
    coda_tokens: int = 0
    generated_tokens: int = 0
    prefill_recurrent_steps: int = 0
    prefill_coda_tokens: int = 0
    fixed_depth_recurrent_steps: int = 0
    exit_depth_histogram: Counter[int] = field(default_factory=Counter)
    recurrent_step_histogram: Counter[int] = field(default_factory=Counter)
    mixed_recurrent_batches: int = 0
    mixed_recurrent_depth_histogram: Counter[int] = field(default_factory=Counter)
    max_mixed_depths_per_batch: int = 0
    scalar_split_count: int = 0
    scheduler_refills: int = 0
    max_depth_queue_size: int = 0
    # hits + captures = graphed launches; a miss always captures immediately, so captures
    # doubles as the miss counter, and captures growing with runtime instead of plateauing
    # at the number of live keys means the graph cache is eviction-thrashing.
    recurrent_graph_hits: int = 0
    recurrent_graph_captures: int = 0
    stage_graph_hits: int = 0
    stage_graph_captures: int = 0
    max_in_flight_requests: int = 0
    max_live_requests: int = 0
    # Prelude launches and the tokens they carried, keyed by ``PRELUDE_SITES`` site;
    # summing over sites gives the totals. ``prelude_gathered_tokens`` counts the subset whose id
    # was gathered on device from a staged batch's sampled tokens, so they entered the prelude
    # without a host round trip; ``prelude_fused_batches`` counts launches that carried both kinds,
    # which is the no-refill wave admitting new decoders alongside its carried successors.
    prelude_batches: Counter[PreludeSite] = field(default_factory=Counter)
    prelude_tokens: Counter[PreludeSite] = field(default_factory=Counter)
    prelude_gathered_tokens: Counter[PreludeSite] = field(default_factory=Counter)
    prelude_fused_batches: Counter[PreludeSite] = field(default_factory=Counter)
    # Depth items dropped because their request finished while the item was queued or in flight:
    # an EOS, or any finish under ``replay_eos_finishes``.
    cancelled_depth_items: int = 0
    # Scheduler ticks that ran with a coda batch still in flight, so the coda bucket was suppressed and
    # a recurrent batch was built without the tokens that batch will return. Divided by ``coda_batches``
    # this is the coda round trip in ticks, which is how far the head-of-line working set can drift
    # above one launch width.
    coda_suppressed_ticks: int = 0
    # Deepest the coda readback pipeline ever got, and the ticks where it was full while a coda
    # the scheduler would otherwise have run was waiting, so the bound rather than the absence of
    # work is what held the bucket back. Both describe the refill loop only; the no-refill loop
    # stages its coda at the wave boundary and never queues readbacks, so it reports zero.
    max_pending_coda_results: int = 0
    coda_pipeline_blocked_ticks: int = 0
    scheduler_ticks: int = 0
    # Peak number of active requests that have generated past their prefill token, i.e. whose KV is
    # still growing. Sampled, not exact: an O(active) scan every tick would dominate the tick.
    max_growing_requests: int = 0

    @property
    def skipped_recurrent_steps(self) -> int:
        return max(0, self.fixed_depth_recurrent_steps - self.recurrent_steps)

    @property
    def mean_coda_latency_ticks(self) -> float:
        """Mean scheduler ticks a staged coda batch stays in flight, or 0.0 when none was staged.

        Zero means every coda batch is consumed before the next recurrent batch is built, so the
        working set never has to be backfilled while its tokens are in the coda pipeline.
        """

        return self.coda_suppressed_ticks / self.coda_batches if self.coda_batches else 0.0

    @property
    def mean_prefill_batch_size(self) -> float:
        """Mean number of requests per prefill batch, or 0.0 when no prefill batch ran."""

        return self.prefill_requests / self.prefill_batches if self.prefill_batches else 0.0

    @property
    def mean_recurrent_batch_size(self) -> float:
        """Mean number of token-steps per recurrent batch, or 0.0 when no recurrent batch ran.

        ``recurrent_steps`` already sums the item count over every recurrent batch, so the mean
        follows from it directly without a separate accumulator.
        """

        return self.recurrent_steps / self.recurrent_batches if self.recurrent_batches else 0.0

    @property
    def mean_coda_batch_size(self) -> float:
        """Mean number of tokens per coda batch, or 0.0 when no coda batch ran.

        ``coda_tokens`` already sums the item count over every coda batch, so the mean follows
        from it directly without a separate accumulator.
        """

        return self.coda_tokens / self.coda_batches if self.coda_batches else 0.0

    @property
    def decode_batches(self) -> int:
        """Decode-side launches: every recurrent step and every coda batch.

        Both are launches the depth engine pays per generated token, so both belong in a launch count
        compared against the full-depth engine, whose single decode launch runs the prelude, the whole
        recurrent stack, and the coda together.
        """

        return self.recurrent_batches + self.coda_batches

    @property
    def mean_decode_batch_size(self) -> float:
        """Mean number of token-steps per decode-side launch, or 0.0 when none ran."""

        decode_batches = self.decode_batches
        return (self.recurrent_steps + self.coda_tokens) / decode_batches if decode_batches else 0.0


@dataclass
class ContinuousDepthBatchingEngine:
    """Native CDB engine for adapter-backed looped language models.

    Prompt prefill is scheduled by the same serving scheduler as recurrent and
    coda work. Prefill chunks run full depth; decode tokens then enter staged
    recurrent execution and can exit before ``max_recurrent_steps``.
    """

    model: nn.Module
    cdb_config: ContinuousDepthBatchingConfig
    cache: PagedAttentionCache
    scheduler: CDBScheduler
    inputs_and_outputs: ContinuousDepthBatchingIOs
    runner: ModelRunner
    model_adapter: CDBModelAdapter
    exit_policy: ExitPolicy
    offloading_manager: OffloadingManager
    last_stats: ContinuousDepthBatchingStats = field(default_factory=ContinuousDepthBatchingStats)
    pending_coda_results: deque[PendingCodaResult] = field(default_factory=deque)
    # Implicitly bounded: every pending result holds in-flight requests counted against the
    # resident cap, so no explicit pipeline depth is needed (unlike the coda pipeline, whose
    # pending results pin hidden slots and rotating buffers).
    pending_prefill_results: deque[PendingPrefillResult] = field(default_factory=deque)
    # Active requests whose next decode token could not allocate a KV block. They keep their KV and their
    # sampled token, and are retried whenever other work frees blocks. Only when no other work can run at
    # all is one of them preempted, chosen by computed length rather than by position here: iteration
    # order is stall order, which says nothing about how much work a request would lose.
    stalled_decoders: dict[str, RequestState] = field(default_factory=dict)
    # Each request's output, keyed by id, captured when it finishes. Soft-reset preemption replaces a
    # request's state object (same id) with a fresh one, so the original ``states`` list can hold a stale
    # object; the finishing object is the source of truth.
    finished_outputs: dict[str, GenerationOutput] = field(default_factory=dict)
    # The model's cache layout, resolved from the adapter only for KV policies that copy a token's
    # exit-step KV into its deeper slots (``last_exited``); ``None`` for every other policy.
    exit_kv_layout: LoopedKvLayout | None = None
    # Tokens that exited early since the last flush, with their exit steps. The copy that propagates
    # their exit-step KV into deeper slots is batched per recurrent launch, not issued per token.
    _exit_kv_backlog: list[tuple[DepthWorkItem, int]] = field(default_factory=list)
    schedule_trace: ScheduleTrace | None = None

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        cdb_config: ContinuousDepthBatchingConfig,
        *,
        dtype: torch.dtype | None = None,
        model_adapter: CDBModelAdapter | None = None,
    ) -> ContinuousDepthBatchingEngine:
        """Create a continuous depth batching engine for an already-loaded model."""

        model_config = model.config
        cdb_config = cdb_config.get_resolved(model_config)
        model_adapter = resolve_cdb_model_adapter(model, model_adapter)
        assert cdb_config.max_recurrent_steps is not None
        model_adapter.configure_recurrent_steps(cdb_config.max_recurrent_steps)
        model_adapter.configure_exit_signal(cdb_config.exit_threshold is not None or cdb_config.synthetic_exit_replay)
        exit_policy = ExitPolicy(
            spec=model_adapter.exit_policy_spec(),
            threshold=cdb_config.exit_threshold,
            min_recurrent_steps=cdb_config.min_recurrent_steps,
            max_recurrent_steps=cdb_config.max_recurrent_steps,
            delay_gate_consumption=cdb_config.delay_gate_consumption,
            use_async_batching=cdb_config.use_async_batching,
            synthetic_exit_replay=cdb_config.synthetic_exit_replay,
        )
        configure_kv_policy = getattr(model_adapter, "configure_kv_policy", None)
        if configure_kv_policy is not None:
            configure_kv_policy(cdb_config.kv_policy, cdb_config.kv_slots_per_layer)
        first_param = next(model.parameters())
        device = first_param.device
        dtype = first_param.dtype if dtype is None else dtype
        geometry = PagedKVCacheGeometry.from_model(
            model_config,
            block_size=cdb_config.block_size,
            dtype=dtype,
            kv_policy=cdb_config.kv_policy,
            kv_slots_per_layer=cdb_config.kv_slots_per_layer,
            total_recurrent_steps=cdb_config.max_recurrent_steps,
            assume_all_layers_recurrent=True,
        )
        cdb_config.num_blocks = resolve_num_blocks(
            cdb_config.num_blocks, cdb_config.mem_fraction_static, geometry, device
        )
        cache = PagedAttentionCache(
            config=model_config,
            continuous_batching_config=cdb_config,
            device=device,
            dtype=dtype,
            geometry=geometry,
        )
        # Resolved onto the config, not just inside the scheduler: the model runner pads decode
        # batches and sizes stage buffers up to ``max_num_seqs``, and those buffers are sized by the
        # cache's token budget, so the two must be the same number.
        cdb_config.max_num_seqs = resolve_max_num_seqs(cdb_config.max_num_seqs, cache.max_num_batched_tokens)
        io_kwargs = {
            "cache": cache,
            "config": model_config,
            "device": device,
            "model_dtype": dtype,
        }
        inputs_and_outputs = ContinuousDepthBatchingIOs(**io_kwargs)
        runner = ModelRunner(
            engine_config=cdb_config,
            inputs_and_outputs=inputs_and_outputs,
            cache=cache,
            model_adapter=model_adapter,
            exit_policy=exit_policy,
        )
        runner.configure_sampled_row_gather(model)
        scheduler = CDBScheduler(
            cache=cache,
            max_recurrent_steps=cdb_config.max_recurrent_steps,
            max_num_seqs=cdb_config.max_num_seqs,
            min_free_slots=cdb_config.min_free_slots,
            safety_margin=cdb_config.safety_margin,
            kv_pressure_mode=cdb_config.kv_pressure_mode,
            max_model_len=cdb_config.max_model_len,
            min_coda_batch_size=cdb_config.min_coda_batch_size,
        )
        offloading_manager = OffloadingManager(
            cache,
            scheduler,
            cpu_offload_space_gib=cdb_config.cpu_offload_space,
            compute_stream=inputs_and_outputs.compute_stream,
            pin_memory=cache.device.type == "cuda",
            allow_recompute_fallback=not copies_exit_kv(cdb_config.kv_policy),
        )
        # Resolved here so a policy that needs copy-on-exit routing fails at engine build,
        # not on the first early exit, when the adapter does not expose its layout.
        exit_kv_layout = model_adapter.kv_cache_layout() if copies_exit_kv(cdb_config.kv_policy) else None
        return cls(
            model=model,
            cdb_config=cdb_config,
            cache=cache,
            scheduler=scheduler,
            inputs_and_outputs=inputs_and_outputs,
            runner=runner,
            model_adapter=model_adapter,
            exit_policy=exit_policy,
            offloading_manager=offloading_manager,
            exit_kv_layout=exit_kv_layout,
        )

    @torch.no_grad()
    def generate_batch(
        self,
        input_ids: list[list[int]],
        *,
        max_new_tokens: int | list[int],
        eos_token_id: int | list[int] | None,
        stop_sequences: list[list[int]] | None = None,
        warmup: bool = True,
        model_kwargs: dict | None = None,
        exit_depths: list[list[int]] | None = None,
        arrival_offsets_s: list[float] | None = None,
        record_token_times: bool = False,
        record_queue_samples: bool = False,
    ) -> list[GenerationOutput]:
        """Generate from tokenized prompts with decode-time CDB.

        ``max_new_tokens`` may be a scalar applied to every request or a per-request
        list. ``exit_depths`` optionally supplies a per-request replay schedule (one
        0-based recurrent step per staged decode token, i.e. ``max_new_tokens - 1``
        entries) that drives exits instead of the live gate; it requires the config's
        ``synthetic_exit_replay`` policy. ``arrival_offsets_s`` optionally releases the
        requests open-loop at wall-clock offsets from the start of generation (one
        non-decreasing offset per request, see :mod:`looped_cdb.arrivals`) instead of
        submitting them all upfront; the refill loop admits new arrivals at tick
        boundaries, the no-refill loop at wave boundaries, and both sleep when idle.
        """

        self._validate_generate_inputs(input_ids, max_new_tokens)
        max_new_tokens = normalize_max_new_tokens(max_new_tokens, len(input_ids))
        max_new_tokens = cap_max_new_tokens_to_model_len(input_ids, max_new_tokens, self.cdb_config.max_model_len)
        self._validate_model_adapter()
        self._validate_exit_gate_serving()
        self._validate_model_kwargs(model_kwargs)
        self._validate_exit_depths(input_ids, max_new_tokens, exit_depths)
        self.reset()
        self.scheduler.record_queue_samples = record_queue_samples
        if warmup:
            self.runner.warmup(self.model, model_kwargs)

        states = [
            RequestState(
                request_id=f"request-{idx}",
                initial_tokens=prompt_ids,
                max_new_tokens=max_new_tokens[idx],
                eos_token_id=eos_token_id,
                stop_sequences=[list(sequence) for sequence in stop_sequences] if stop_sequences else [],
                synthetic_exit_depths=list(exit_depths[idx]) if exit_depths is not None else [],
                replay_eos_finishes=self.cdb_config.replay_eos_finishes,
                record_token_times=record_token_times,
            )
            for idx, prompt_ids in enumerate(input_ids)
        ]
        arrivals: ArrivalQueue | None = None
        if arrival_offsets_s is None:
            for state in states:
                self.scheduler.add_waiting_request(state)
        else:
            arrivals = ArrivalQueue(states, arrival_offsets_s)
            arrivals.start()

        if self.cdb_config.refill:
            self._run_scheduler_loop(model_kwargs=model_kwargs, arrivals=arrivals)
        else:
            self._run_no_refill_loop(model_kwargs=model_kwargs, arrivals=arrivals)
        self.scheduler.close_steady_window()
        self.scheduler.sample_waiting_queue()
        self._free_finished_or_active()
        # Read outputs from the finish-time capture, not the original ``states`` list: soft-reset
        # preemption replaces a request's object (same id), leaving the original entry stale.
        outputs = [self.finished_outputs[state.request_id] for state in states]
        self.last_stats.generated_tokens = sum(len(output.generated_tokens) for output in outputs)
        return outputs

    def reset(self) -> None:
        """Reset engine-owned scheduler, I/O, cache, and counters."""

        self.offloading_manager.free_all_waiting_cpu_caches()
        self.scheduler.reset()
        self.inputs_and_outputs.reset()
        self.cache.free_all_requests()
        self.cache.reset_peak_usage()
        self.runner.reset_cdb_runtime_counters()
        self.offloading_manager.reset()
        self.last_stats = ContinuousDepthBatchingStats()
        self.pending_coda_results.clear()
        self.pending_prefill_results.clear()
        self.stalled_decoders.clear()
        self.finished_outputs.clear()
        self._exit_kv_backlog.clear()
        if self.schedule_trace is not None:
            self.schedule_trace.reset()

    def _trace(self, stage: TraceStage, size: int, **detail: object) -> None:
        """Record a stage launch and the resulting queue occupancy when tracing is enabled."""

        trace = self.schedule_trace
        if trace is None:
            return
        decoding = sum(1 for state in self.scheduler.active_requests.values() if state.status == RequestStatus.DECODING)
        queues = QueueSnapshot(
            waiting=len(self.scheduler.waiting_requests),
            recurrent=len(self.scheduler.ready_queue),
            coda=len(self.scheduler.coda_queue),
            decode=decoding - trace.running_cohort,
            coda_in_flight=sum(len(pending.items) for pending in self.pending_coda_results),
            active=len(self.scheduler.active_requests),
        )
        trace.record(stage, size, queues, steady=self.scheduler.steady_window_open, **detail)

    def _trace_tick(self, tick: int) -> None:
        if self.schedule_trace is not None:
            self.schedule_trace.tick = tick

    def _trace_cohort(self, size: int) -> None:
        if self.schedule_trace is not None:
            self.schedule_trace.running_cohort = size

    def _run_scheduler_loop(self, *, model_kwargs: dict | None, arrivals: ArrivalQueue | None = None) -> None:
        """Run unified CDB scheduler ticks until all requests finish (and all arrivals are served)."""

        recurrent_kwargs = {} if model_kwargs is None else dict(model_kwargs)
        while (
            (arrivals is not None and arrivals.pending)
            or self.scheduler.has_pending_work()
            or self.pending_coda_results
            or self.pending_prefill_results
        ):
            if arrivals is not None:
                for state in arrivals.release_due():
                    self.scheduler.add_waiting_request(state)
            # Prefill results are consumed before coda results at every site: a request's first
            # token (prefill) must be recorded before its second (first coda), and the shared D2H
            # stream guarantees the prefill copy lands first.
            self._consume_ready_prefill_results(block=not self.scheduler.has_pending_work())
            self._consume_ready_coda_results(block=not self.scheduler.has_pending_work())
            if not self.scheduler.has_pending_work():
                if arrivals is not None and arrivals.pending:
                    # Nothing resident and no coda in flight (the blocking consume above drained
                    # them): idle until the next arrival.
                    arrivals.wait_for_next_arrival()
                continue
            # Restore before scheduling, so the prefill gate sees the restored decoders and holds back
            # new prompts in their favor rather than admitting on top of the blocks they just reclaimed.
            # Restored requests already hold a block for their pending token, so staging them cannot
            # stall; they go through the one staging path all decode tokens use. Their items and the
            # restaged ones run together, in a single launch per tick.
            restored = self._restore_offloaded_requests()
            resumed = [item for state in restored if (item := self._stage_decode_token(state)) is not None]
            # Retry the decode tokens that could not allocate last tick before placing any new work.
            resumed += self._restage_stalled_decoders()
            self._prelude_and_enqueue_depth(resumed, site="resumed", front=True)
            self.last_stats.scheduler_ticks += 1
            self._trace_tick(self.last_stats.scheduler_ticks)
            pending_codas = len(self.pending_coda_results)
            if pending_codas:
                self.last_stats.coda_suppressed_ticks += 1
            if self.last_stats.scheduler_ticks % _GROWING_SAMPLE_EVERY == 0:
                self._sample_growing_requests()
            with nvtx.range("cdb.schedule"):
                scheduled = self.scheduler.schedule_next(
                    token_budget=self.cache.max_num_batched_tokens,
                    cache_budget=self.cache.num_pages,
                    # With on-device re-entry a pending coda no longer blocks its tokens' successors,
                    # so the coda pipeline may run CODA_PIPELINE_DEPTH deep; the bound keeps
                    # hidden-slot and pinned buffer usage finite when the host falls behind, and the
                    # runner's rotating coda buffers hold exactly that many turns.
                    allow_coda=len(self.pending_coda_results) < CODA_PIPELINE_DEPTH,
                )
            # The bound withheld a coda the scheduler would otherwise have run: through the priority
            # bucket once the queue reaches ``min_coda_batch_size``, or through the drain-tail flush,
            # which is reachable exactly when nothing else could be placed.
            if (
                pending_codas >= CODA_PIPELINE_DEPTH
                and self.scheduler.coda_queue
                and (scheduled is None or len(self.scheduler.coda_queue) >= self.scheduler.min_coda_batch_size)
            ):
                self.last_stats.coda_pipeline_blocked_ticks += 1
            if scheduled is None:
                if self.pending_prefill_results:
                    self._consume_ready_prefill_results(block=True)
                    continue
                if self.pending_coda_results:
                    self._consume_ready_coda_results(block=True)
                    continue
                if self.stalled_decoders:
                    # Nothing else can run and no stalled decoder can allocate: yield one request's KV
                    # so the rest of the batch drains. The last resort, as in the full-depth engine.
                    self._preempt_stalled_decoder()
                    continue
                # No coda, no recurrent work, and prefill placed nothing, with nothing in flight to
                # wait on. Nothing will change on the next tick, so this is a dead end, not a stall to
                # spin on.
                self._raise_scheduler_stall()

            if scheduled.kind == "prefill":
                with nvtx.range("cdb.prefill"):
                    self._run_prefill_batch(scheduled)
            elif scheduled.kind == "coda":
                if scheduled.depth_items is None:
                    raise RuntimeError("Coda scheduler tick is missing depth items")
                self._stage_coda_batch(scheduled.depth_items)
            elif scheduled.kind == "recurrent":
                if scheduled.depth_items is None:
                    raise RuntimeError("Recurrent scheduler tick is missing depth items")
                self._run_mixed_recurrent_batch(scheduled.depth_items, recurrent_kwargs=recurrent_kwargs)
            else:
                raise AssertionError(f"Unhandled scheduler tick kind: {scheduled.kind}")

        self._consume_ready_prefill_results(block=True)
        self._consume_ready_coda_results(block=True)
        self.last_stats.scheduler_refills = self.scheduler.refill_events
        self.last_stats.max_depth_queue_size = self.scheduler.max_depth_queue_size
        self.last_stats.recurrent_graph_hits = self.runner.recurrent_graph_hits
        self.last_stats.recurrent_graph_captures = self.runner.recurrent_graph_captures
        self.last_stats.stage_graph_hits = self.runner.stage_graph_hits
        self.last_stats.stage_graph_captures = self.runner.stage_graph_captures
        self.last_stats.max_in_flight_requests = self.scheduler.max_in_flight_requests
        self.last_stats.max_live_requests = self.scheduler.max_live_requests
        max_steps = self.cdb_config.max_recurrent_steps
        assert max_steps is not None
        self.last_stats.fixed_depth_recurrent_steps = self.last_stats.coda_tokens * max_steps

    def _run_no_refill_loop(self, *, model_kwargs: dict | None, arrivals: ArrivalQueue | None = None) -> None:
        """Drive sequence-level, no-refill decoding for the no-refill baseline.

        Each outer iteration prefills to exhaustion and then runs one lockstep decode wave
        over every active decode token. Prefill keeps launching monolithic variable-length
        forwards while
        :meth:`~looped_cdb.scheduler_base.BaseServingScheduler.admit_prefill` and the
        batch gate still place prompts, so a wave opens on the fullest resident set the
        cache and ``max_num_seqs`` allow. A single batch carries only
        ``max_num_batched_tokens`` of prompt, and the full-depth engine gives prefill
        priority on every tick, so both fill the resident set with consecutive batches:
        prompt admission is not what separates this baseline from continuous depth
        batching. The decode wave runs the recurrent core once per step on the tokens
        that have not yet exited, so the active batch shrinks across recurrent steps
        and stops as soon as every token has exited; the freed slots are never
        refilled with other work. Exited tokens wait for
        the wave: one wave-sized coda batch is staged once the cohort has drained,
        and every wave enters through a single prelude launch over its whole cohort
        (see ``_run_no_refill_decode_wave``). The exit decision uses the same
        synchronous or delayed-gate execution as continuous depth batching; the
        loop only differs in never touching the mixed-depth ready queue, so
        ``scheduler_refills`` stays zero by construction -- the honest
        sequence-level comparison against CDB.
        """

        recurrent_kwargs = {} if model_kwargs is None else dict(model_kwargs)
        # Wave-boundary carry: a wave hands its staged boundary coda and its exiters' successors,
        # staged but before their prelude, to the next wave, whose single prelude launch
        # gathers their tokens from the coda's device output; carried requests are excluded
        # from that wave's cohort scan so no token is staged twice. Prefill carries the same
        # way; its readback is consumed before the carried coda so each request's first token
        # is recorded before its second.
        boundary = NoRefillBoundary()
        wave = 0
        while (arrivals is not None and arrivals.pending) or self.scheduler.has_pending_work():
            wave += 1
            self._trace_tick(wave)
            for pending_prefill in boundary.take_prefills():
                self._consume_no_refill_prefill(pending_prefill)
            if arrivals is not None:
                for state in arrivals.release_due():
                    self.scheduler.add_waiting_request(state)
                if not self.scheduler.has_pending_work():
                    if arrivals.pending:
                        # Only future arrivals remain; carried wave-boundary work implies active
                        # requests, so nothing is in flight while the loop sleeps.
                        arrivals.wait_for_next_arrival()
                    continue
            # Restored requests re-enter as decode work, so they must be collected before the prefill
            # gate reads the decode set and before the decode wave forms its cohort.
            self._restore_offloaded_requests()
            # The gate re-reads the decode set every pass, as the full-depth engine does every
            # tick, so a batch that manufactures decoders tightens the KV rule for the batch
            # behind it.
            # Sampled here rather than inside a scheduler entry point, since this loop never calls
            # ``schedule_next``: once per tick, after restores and before this tick admits anything,
            # which is where both other loops sample.
            self.scheduler.record_resident_sample()
            did_prefill = False
            while self.scheduler.admit_prefill(self.scheduler.has_decode_work()):
                if not self._run_no_refill_prefill_batch(recurrent_kwargs, boundary):
                    break
                did_prefill = True
            did_decode = self._run_no_refill_decode_wave(recurrent_kwargs, boundary)
            if not did_prefill and not did_decode:
                if self.stalled_decoders:
                    # No wave could run and no stalled decoder can allocate: yield one request's KV so
                    # the rest of the batch drains. The last resort, as in the full-depth engine.
                    self._preempt_stalled_decoder()
                    continue
                if not self.scheduler.has_pending_work():
                    # The prefill consume at the iteration top retired the remaining requests, so
                    # there is nothing left to run rather than a stall.
                    continue
                raise CacheFullError(
                    "No-refill loop stalled with pending work: the KV cache cannot admit any waiting request "
                    "and no decode work is ready (over-subscribed batch)."
                )

        assert not boundary.prefills, (
            "No-refill loop exited with an unconsumed prefill readback; its requests are DECODING, "
            "so has_pending_work cannot have gone false before it was consumed."
        )
        assert not boundary.carried and boundary.coda is None, (
            "No-refill loop exited with carried wave-boundary work; requests with an in-flight coda "
            "are active, so has_pending_work cannot have gone false before it was consumed."
        )
        max_steps = self.cdb_config.max_recurrent_steps
        assert max_steps is not None
        self.last_stats.scheduler_refills = self.scheduler.refill_events
        self.last_stats.recurrent_graph_hits = self.runner.recurrent_graph_hits
        self.last_stats.recurrent_graph_captures = self.runner.recurrent_graph_captures
        self.last_stats.stage_graph_hits = self.runner.stage_graph_hits
        self.last_stats.stage_graph_captures = self.runner.stage_graph_captures
        self.last_stats.max_in_flight_requests = self.scheduler.max_in_flight_requests
        self.last_stats.max_live_requests = self.scheduler.max_live_requests
        self.last_stats.fixed_depth_recurrent_steps = self.last_stats.coda_tokens * max_steps

    def _no_refill_decode_cohort(self) -> list[RequestState]:
        """The decoders that run together in one lockstep wave.

        The sampled-token requirement is a precondition of :meth:`_stage_decode_token`, which raises on
        a request whose next token has not been sampled yet, rather than a convenience filter. Prefill
        marks a request ``DECODING`` before its forward runs (so async batching stays coherent), so a
        request is briefly decoding with only a placeholder token; the wave must not pick it up.
        """

        return [
            state
            for state in self.scheduler.active_requests.values()
            if state.status == RequestStatus.DECODING and state.tokens_to_process and state.tokens_to_process[0] >= 0
        ]

    def _run_no_refill_prefill_batch(self, recurrent_kwargs: dict[str, object], boundary: NoRefillBoundary) -> bool:
        """Run one full-depth monolithic prefill batch for the no-refill baseline.

        Prefill is identical to the refill engine's: a single variable-length forward
        at the full recurrent depth (prefill never exits early), sharing the same
        ``stage_prefill_batch`` primitive rather than the staged recurrent path. On the
        async delayed path (the same condition the decode wave uses for its boundary
        coda) each fully-consumed prompt's first decode token is staged as a carried
        entry for this same iteration's decode wave, whose single prelude launch
        gathers it from the batch's device tokens, so the host never blocks on the
        prefill forward; the sampled tokens are applied at the top of the next
        iteration as bookkeeping. Prefill admission is token-budget-gated, so the batches
        of one iteration can consume more prompts than the decode launch width; requests
        that cannot enter eagerly (length limit met by the pending token, no KV block, or
        no width left beside the successors already carried) are picked up by a later
        wave's cohort scan once their token has been applied. The carried entries and the
        pending readback are handed to ``boundary``; returns whether a batch ran.
        """

        scheduled = self.scheduler.schedule_prefill_batch(
            token_budget=self.cache.max_num_batched_tokens,
            cache_budget=self.cache.num_pages,
        )
        if scheduled is None or not scheduled.requests:
            return False
        eager_budget = max(0, self.cdb_config.max_num_seqs - boundary.num_carried_items())
        delayed = self._uses_delayed_synthetic_exit() or self._uses_delayed_gate_consumption(recurrent_kwargs)

        self.last_stats.prefill_batches += 1
        self.last_stats.prefill_requests += len(scheduled.requests)
        max_steps = self.cdb_config.max_recurrent_steps
        assert max_steps is not None
        self.last_stats.prefill_recurrent_steps += scheduled.num_q_tokens * max_steps
        with nvtx.range("cdb.prefill"):
            pending = self.runner.stage_prefill_batch(
                self.model,
                scheduled.requests,
                num_q_tokens=scheduled.num_q_tokens,
                max_kv_read=scheduled.max_kv_read,
                allow_async=self.cdb_config.use_async_batching,
            )
            self._trace("prefill", len(scheduled.requests), tokens=scheduled.num_q_tokens)
            eager_flags: list[bool] = []
            eager_items: list[DepthWorkItem] = []
            eager_rows: list[int] = []
            for future_state in pending.futures:
                state = future_state.state
                if future_state.has_new_token:
                    self.last_stats.prefill_coda_tokens += 1
                    eager = False
                    if (
                        delayed
                        and len(eager_items) < eager_budget
                        and not state.will_finish_on_pending_token()
                        and self._try_allocate_one_decode_token(state)
                    ):
                        eager = True
                        eager_rows.append(len(eager_flags))
                        eager_items.append(self._build_decode_item(state, token_id=TMP_TOKEN_ID))
                    eager_flags.append(eager)
                self.scheduler.release_in_flight_request(state.request_id)
            pending.eager_entries = eager_flags
        if not delayed:
            self._consume_no_refill_prefill(pending)
            return True
        if eager_items:
            boundary.carried.append(CarriedPreludeEntries(result=pending, items=eager_items, rows=eager_rows))
        boundary.prefills.append(pending)
        return True

    def _consume_no_refill_prefill(self, pending: PendingPrefillResult) -> None:
        """Apply a prefill batch's sampled tokens; eagerly entered requests take them as bookkeeping."""

        with nvtx.range("cdb.prefill_consume"):
            new_tokens = self.runner.consume_prefill_result(pending)
        eager_entries = pending.eager_entries or [False] * len(new_tokens)
        new_token_idx = 0
        for future_state in pending.futures:
            if not future_state.has_new_token:
                continue
            state = future_state.state
            token_id = new_tokens[new_token_idx]
            was_eager = eager_entries[new_token_idx]
            new_token_idx += 1
            if state.status == RequestStatus.FINISHED:
                continue
            if was_eager:
                # The first decode token already ran in the previous wave; a finish here can only
                # be EOS, and retiring cancels any of its queued work at the next pop.
                if state.update_and_check_completion(token_id):
                    self._retire_request(state)
                continue
            self._advance_after_prefill(state, token_id)

    def _run_no_refill_decode_wave(self, recurrent_kwargs: dict[str, object], boundary: NoRefillBoundary) -> bool:
        """Run one lockstep decode wave over every active decode token.

        Each step removes exited tokens without refilling their slots.
        Exited tokens run coda together after the cohort drains.
        Async mode carries their staged successors into the next wave.
        A late EOS cancels a successor after its first recurrent step.
        """

        # A prefill consume at the iteration top can retire a request (EOS on its first token)
        # whose carried successor has not launched yet; dropping it here keeps a dead request's
        # item from writing KV that a prefill batch may already have reallocated.
        carried, carried_coda = boundary.take_decode_carry()
        carried = self._drop_cancelled_carried(carried)
        carried_items = [item for entries in carried for item in entries.items]
        # Requests with a carried successor or an unconsumed coda hold host token state that is one
        # wave behind; the scan must not re-stage their previous token. Coda-carried successors are
        # covered by the carried coda's id set; prefill-carried entries are excluded by the cohort's
        # placeholder-token guard (their sampled token is applied one iteration later).
        in_flight_ids = {item.state.request_id for item in carried_coda.items} if carried_coda is not None else set()
        decode_cohort = [state for state in self._no_refill_decode_cohort() if state.request_id not in in_flight_ids]
        # Bound the cohort to the recurrent launch shape; any overflow decodes in the
        # next wave, exactly as sequence-level CB caps its decode batch by memory.
        # Carried successors already hold KV slots, so they take priority.
        decode_cohort = decode_cohort[: max(0, self.cdb_config.max_num_seqs - len(carried_items))]

        # A request whose next token cannot allocate a KV block stalls rather than joining this cohort;
        # it is retried on a later wave, and preempted only if no wave can run at all.
        new_items = [item for state in decode_cohort if (item := self._stage_decode_token(state)) is not None]
        items = carried_items + new_items
        if not items:
            if carried_coda is not None:
                # Nothing can launch, but the carried coda may retire requests (or free their
                # tokens for the next scan); consuming it is this wave's work.
                self._consume_no_refill_coda(carried_coda)
                return True
            return False
        # The wave's one prelude launch: carried entries gather their tokens from their staged
        # results' device outputs, scanned decoders bring host-sampled ids, all in one batch.
        self._record_prelude("wave_cohort", len(items), gathered=len(carried_items))
        with nvtx.range("cdb.prelude.wave_cohort"):
            self.runner.compute_prelude_batch(
                items, device_groups=[(entries.result, entries.rows) for entries in carried]
            )
        self._trace_cohort(len(items))
        self._trace("prelude", len(items), site="wave_cohort")
        max_steps = self.cdb_config.max_recurrent_steps
        assert max_steps is not None

        delayed_synthetic = self._uses_delayed_synthetic_exit()
        delayed_gate = self._uses_delayed_gate_consumption(recurrent_kwargs)
        delayed = delayed_synthetic or delayed_gate
        stage_signal = self._uses_delayed_gate_readout(recurrent_kwargs)

        survivors = items
        exited_wave: list[DepthWorkItem] = []
        for step in range(max_steps):
            for item in survivors:
                item.recurrent_step = step
            exit_signals = self._launch_no_refill_recurrent(survivors, recurrent_kwargs)
            self.last_stats.recurrent_step_histogram[step] += len(survivors)
            # Stage the gate GPU->CPU copy whenever a live gate runs: the real gate drives the exit,
            # and synthetic replay discards the value but must still pay the readout cost. When the
            # gate is off (no cost to time) nothing is staged even on the delayed execution path.
            pending_signal_results = (
                self.runner.stage_delayed_exit_signals(survivors, exit_signals) if stage_signal else None
            )
            if carried_coda is not None:
                # The wave's first launch and gate copies are enqueued; the previous wave's coda
                # wait now overlaps them. A finish revealed here (EOS) cancels that request's
                # successor below, before it is routed to the boundary coda.
                self._consume_no_refill_coda(carried_coda)
                carried_coda = None

            forced = step == max_steps - 1
            exited: list[DepthWorkItem] = []
            continuing: list[DepthWorkItem] = []
            cancelled: list[DepthWorkItem] = []
            for batch_index, item in enumerate(survivors):
                if item.state.status == RequestStatus.FINISHED:
                    # Finished (EOS) by the carried coda consumed after this step's launch; the
                    # successor's remaining depth work is cancelled rather than routed.
                    cancelled.append(item)
                    continue
                if delayed_synthetic:
                    should_exit = item.apply_exit_gate and self._consume_delayed_synthetic_exit(item)
                elif delayed_gate:
                    should_exit = item.apply_exit_gate and self._consume_delayed_gate(item)
                else:
                    should_exit = self._should_exit(item, step, exit_signals, batch_index)
                if forced or should_exit:
                    self._record_exit(item, step)
                    exited.append(item)
                else:
                    if delayed_synthetic:
                        item.pending_synthetic_exit_step = step
                        # Awaited (and timed) next step when the gate ran; None when it is off.
                        item.pending_gate = (
                            pending_signal_results[batch_index] if pending_signal_results is not None else None
                        )
                    elif delayed_gate:
                        assert pending_signal_results is not None
                        item.pending_gate = pending_signal_results[batch_index]
                    continuing.append(item)
            if cancelled:
                self.runner.release_hidden_slots(cancelled)
                self.last_stats.cancelled_depth_items += len(cancelled)
            # Propagate this step's exit KV before the next launch reads the survivors' deeper slots.
            self._flush_exit_kv_copies()
            if self.schedule_trace is not None:
                self._trace("recurrent", len(survivors), depths={step: len(survivors)}, exited=len(exited))
            exited_wave.extend(exited)
            survivors = continuing
            if not survivors:
                break
        self._trace_cohort(0)

        if delayed and exited_wave:
            # Consume this coda after the next wave's first launch.
            # A successor without KV capacity remains carried for one extra wave.
            with nvtx.range("cdb.coda_stage"):
                pending = self.runner.stage_coda_batch(exited_wave, allow_async=True)
            self._trace("coda", len(exited_wave))
            eager_items, eager_rows = self._select_eager_reentries(pending)
            if eager_items:
                boundary.carried.append(CarriedPreludeEntries(result=pending, items=eager_items, rows=eager_rows))
            boundary.coda = pending
            return True
        self._coda_and_sample_no_refill(exited_wave)
        return True

    def _launch_no_refill_recurrent(
        self,
        items: list[DepthWorkItem],
        recurrent_kwargs: dict[str, object],
    ) -> torch.Tensor:
        """Run one homogeneous single-step recurrent launch and return its gate logits."""

        with nvtx.range("cdb.recurrent"):
            _, exit_signals = self.runner.compute_recurrent_batch(items, recurrent_kwargs=recurrent_kwargs)
        self.last_stats.recurrent_batches += 1
        self.last_stats.recurrent_steps += len(items)
        return exit_signals

    def _coda_and_sample_no_refill(self, items: list[DepthWorkItem]) -> None:
        """Run one synchronous coda/LM-head batch and advance each request by a token."""

        if not items:
            return
        with nvtx.range("cdb.coda_stage"):
            pending = self.runner.stage_coda_batch(items, allow_async=False)
        self._trace("coda", len(items))
        self._consume_no_refill_coda(pending)

    def _consume_no_refill_coda(self, pending: PendingCodaResult) -> None:
        """Consume a staged coda result, advancing each request by its sampled token.

        The no-refill loop re-scans active requests each wave, so unlike the refill path this only
        advances each request; the next decode token is staged when a wave forms. A token whose
        successor was eagerly re-entered at the wave boundary is applied as bookkeeping only, and a
        finish it reveals (EOS) cancels the successor already running in the current wave.
        """

        with nvtx.range("cdb.coda_consume"):
            new_tokens = self.runner.consume_coda_result(pending)
        items = pending.items
        self.last_stats.coda_batches += 1
        self.last_stats.coda_tokens += len(items)
        eager_reentries = pending.eager_reentries or [False] * len(items)
        for item, token_id, was_eager in zip(items, new_tokens, eager_reentries, strict=True):
            state = item.state
            if state.status == RequestStatus.FINISHED:
                # Finished (EOS) while this token was in flight; the sampled token is discarded.
                continue
            if was_eager:
                if state.update_and_check_completion(token_id):
                    self._retire_request(state)
                continue
            self._advance_after_coda(item, token_id)

    def _run_prefill_batch(self, scheduled: ScheduledCDBBatch) -> None:
        """Run one full-depth monolithic prefill and stage the first decode tokens.

        Prefill does not enter the mixed-depth queue or exit early.
        Async mode can feed sampled device tokens directly into their decode preludes.
        """

        requests_in_batch = scheduled.requests
        if not requests_in_batch:
            raise RuntimeError("Prefill scheduler tick has no requests")

        self.last_stats.prefill_batches += 1
        self.last_stats.prefill_requests += len(requests_in_batch)
        max_steps = self.cdb_config.max_recurrent_steps
        assert max_steps is not None
        # Prefill runs the full recurrent depth for every prompt token in one forward; count the
        # equivalent recurrent-step work so the FLOP accounting matches the staged decode path.
        self.last_stats.prefill_recurrent_steps += scheduled.num_q_tokens * max_steps
        pending = self.runner.stage_prefill_batch(
            self.model,
            requests_in_batch,
            num_q_tokens=scheduled.num_q_tokens,
            max_kv_read=scheduled.max_kv_read,
            allow_async=self.cdb_config.use_async_batching,
        )
        self._trace("prefill", len(requests_in_batch), tokens=scheduled.num_q_tokens)

        eager_flags: list[bool] = []
        eager_items: list[DepthWorkItem] = []
        eager_rows: list[int] = []
        for future_state in pending.futures:
            state = future_state.state
            if not future_state.has_new_token:
                # Prompt was chunked across waves; release it so its next chunk can be scheduled.
                self.scheduler.release_in_flight_request(state.request_id)
                continue
            self.last_stats.prefill_coda_tokens += 1
            eager = False
            if (
                self.cdb_config.use_async_batching
                and not state.will_finish_on_pending_token()
                and self._try_allocate_one_decode_token(state)
            ):
                eager = True
                eager_rows.append(len(eager_flags))
                eager_items.append(self._build_decode_item(state, token_id=TMP_TOKEN_ID))
            eager_flags.append(eager)
        pending.eager_entries = eager_flags
        if eager_items:
            self._record_prelude("after_prefill_eager", len(eager_items), gathered=len(eager_items))
            with nvtx.range("cdb.prelude.after_prefill_eager"):
                self.runner.compute_prelude_batch(eager_items, device_groups=[(pending, eager_rows)])
            self._enqueue_prelude_items(eager_items, front=False, site="after_prefill_eager")
        self.pending_prefill_results.append(pending)

    def _consume_ready_prefill_results(self, *, block: bool) -> None:
        while self.pending_prefill_results:
            pending = self.pending_prefill_results[0]
            if not block and not pending.is_ready():
                return
            self.pending_prefill_results.popleft()
            self._consume_prefill_result(pending)

    def _consume_prefill_result(self, pending: PendingPrefillResult) -> None:
        with nvtx.range("cdb.prefill_consume"):
            new_tokens = self.runner.consume_prefill_result(pending)
        eager_entries = pending.eager_entries or [False] * len(new_tokens)

        new_items: list[DepthWorkItem] = []
        new_token_idx = 0
        for future_state in pending.futures:
            if not future_state.has_new_token:
                continue
            state = future_state.state
            token_id = new_tokens[new_token_idx]
            was_eager = eager_entries[new_token_idx]
            new_token_idx += 1
            if state.status == RequestStatus.FINISHED:
                # Finished while its first token was in flight; the sampled token is discarded.
                continue
            if was_eager:
                # The first decode token already ran its prelude and is queued; the token is bookkeeping
                # only. A finish here can only be a content finish (EOS or stop match; a guaranteed
                # length finish is never entered eagerly), and retiring cancels the queued item at
                # its next pop.
                if state.update_and_check_completion(token_id):
                    self._retire_request(state)
                continue
            if self._advance_after_prefill(state, token_id):
                continue
            item = self._stage_decode_token(state)
            if item is not None:
                new_items.append(item)
        self._prelude_and_enqueue_depth(new_items, site="after_prefill")

    def _run_mixed_recurrent_batch(
        self,
        items: list[DepthWorkItem],
        *,
        recurrent_kwargs: dict[str, object],
    ) -> None:
        """Run one shared-KV mixed-depth recurrent batch."""

        items = self._drop_cancelled_items(items)
        if not items:
            return
        self.last_stats.mixed_recurrent_batches += 1
        depth_counts = Counter(item.recurrent_step for item in items)
        self.last_stats.max_mixed_depths_per_batch = max(
            self.last_stats.max_mixed_depths_per_batch,
            len(depth_counts),
        )
        for recurrent_step, count in depth_counts.items():
            self.last_stats.mixed_recurrent_depth_histogram[recurrent_step] += count

        with nvtx.range("cdb.recurrent"):
            _, exit_signals = self.runner.compute_recurrent_batch(
                items,
                recurrent_kwargs=recurrent_kwargs,
            )
        self.last_stats.recurrent_batches += 1
        self.last_stats.recurrent_steps += len(items)

        pending_signal_results = (
            self.runner.stage_delayed_exit_signals(items, exit_signals)
            if self._uses_delayed_gate_readout(recurrent_kwargs)
            else None
        )

        if self._uses_delayed_gate_consumption(recurrent_kwargs):
            assert pending_signal_results is not None
            with nvtx.range("cdb.exit_route"):
                refills = []
                for idx, item in enumerate(items):
                    recurrent_step = item.recurrent_step
                    self.last_stats.recurrent_step_histogram[recurrent_step] += 1
                    should_exit = item.apply_exit_gate and self._consume_delayed_gate(item)
                    refill = self._schedule_after_delayed_recurrent_step(
                        item,
                        recurrent_step,
                        should_exit=should_exit,
                        next_pending_gate=pending_signal_results[idx] if item.apply_exit_gate else None,
                    )
                    if refill is not None:
                        refills.append(refill)
                self._enqueue_refills(refills)
                self._flush_exit_kv_copies()
            if self.schedule_trace is not None:
                self._trace("recurrent", len(items), depths=dict(depth_counts), exited=len(items) - len(refills))
            return

        if self._uses_delayed_synthetic_exit():
            with nvtx.range("cdb.exit_route"):
                refills = []
                for idx, item in enumerate(items):
                    recurrent_step = item.recurrent_step
                    self.last_stats.recurrent_step_histogram[recurrent_step] += 1
                    should_exit = item.apply_exit_gate and self._consume_delayed_synthetic_exit(item)
                    refill = self._schedule_after_delayed_synthetic_step(
                        item,
                        recurrent_step,
                        should_exit=should_exit,
                        next_pending_gate=(
                            pending_signal_results[idx]
                            if pending_signal_results is not None and item.apply_exit_gate
                            else None
                        ),
                    )
                    if refill is not None:
                        refills.append(refill)
                self._enqueue_refills(refills)
                self._flush_exit_kv_copies()
            if self.schedule_trace is not None:
                self._trace("recurrent", len(items), depths=dict(depth_counts), exited=len(items) - len(refills))
            return

        with nvtx.range("cdb.exit_route"):
            refills = []
            for idx, item in enumerate(items):
                recurrent_step = item.recurrent_step
                self.last_stats.recurrent_step_histogram[recurrent_step] += 1
                should_exit = self._should_exit(item, recurrent_step, exit_signals, idx)
                refill = self._schedule_after_recurrent_step(item, recurrent_step, should_exit)
                if refill is not None:
                    refills.append(refill)
            self._enqueue_refills(refills)
        self._flush_exit_kv_copies()
        if self.schedule_trace is not None:
            self._trace("recurrent", len(items), depths=dict(depth_counts), exited=len(items) - len(refills))

    def _schedule_after_recurrent_step(
        self,
        item: DepthWorkItem,
        recurrent_step: int,
        should_exit: bool,
    ) -> DepthWorkItem | None:
        """Route one stepped item: to coda if it exited, else back as the refill item for its caller.

        Returns the item to re-queue rather than queueing it, so the caller can return a whole stepped
        batch to the head of the depth queue in one ordered call.
        """

        if should_exit:
            self._record_exit(item, recurrent_step)
            self.scheduler.enqueue_coda(item)
            return None

        item.recurrent_step = recurrent_step + 1
        return item

    def _record_exit(self, item: DepthWorkItem, recurrent_step: int) -> None:
        """Record one token's exit depth and clear its pending gate/replay state.

        Shared by the refill scheduler and the no-refill decode wave so the exit histogram the
        paper reports has a single source of truth. Under a copy-on-exit KV policy the token is
        also queued for exit-KV routing; the caller flushes the queue before the next recurrent
        launch could read the token's deeper slots.
        """

        if item.apply_exit_gate:
            item.state.current_token_exit_depth = recurrent_step
            item.state.exit_depths.append(recurrent_step)
            self.last_stats.exit_depth_histogram[recurrent_step] += 1
        item.pending_gate = None
        item.pending_synthetic_exit_step = None
        max_steps = self.cdb_config.max_recurrent_steps
        assert max_steps is not None
        if self.exit_kv_layout is not None and recurrent_step < max_steps - 1:
            self._exit_kv_backlog.append((item, recurrent_step))

    def _flush_exit_kv_copies(self) -> None:
        """Copy each backlogged token's exit-step KV rows into its deeper KV slots.

        Copies run on the compute stream before deeper slots are read.
        This relies on at most one recurrent token per request being in flight.
        """

        if not self._exit_kv_backlog:
            return
        assert self.exit_kv_layout is not None
        backlog, self._exit_kv_backlog = self._exit_kv_backlog, []
        by_step: dict[int, list[DepthWorkItem]] = {}
        for item, exit_step in backlog:
            by_step.setdefault(exit_step, []).append(item)
        compute_stream = self.inputs_and_outputs.compute_stream
        stream_scope = torch.cuda.stream(compute_stream) if compute_stream is not None else nullcontext()
        with nvtx.range("cdb.exit_kv_copy"), stream_scope:
            for exit_step, items in sorted(by_step.items()):
                rows = self.cache.gather_token_kv_rows([(item.state.request_id, item.token_position) for item in items])
                for source_layer_idx, target_layer_idxs in self.exit_kv_layout.exit_copy_layer_groups(exit_step):
                    self.cache.copy_kv_rows(source_layer_idx, target_layer_idxs, rows)

    def _schedule_after_delayed_recurrent_step(
        self,
        item: DepthWorkItem,
        recurrent_step: int,
        *,
        should_exit: bool,
        next_pending_gate: PendingGateResult | None,
    ) -> DepthWorkItem | None:
        max_steps = self.cdb_config.max_recurrent_steps
        assert max_steps is not None
        if recurrent_step == max_steps - 1 or should_exit:
            return self._schedule_after_recurrent_step(item, recurrent_step, True)

        item.pending_gate = next_pending_gate
        return self._schedule_after_recurrent_step(item, recurrent_step, False)

    def _schedule_after_delayed_synthetic_step(
        self,
        item: DepthWorkItem,
        recurrent_step: int,
        *,
        should_exit: bool,
        next_pending_gate: PendingGateResult | None = None,
    ) -> DepthWorkItem | None:
        max_steps = self.cdb_config.max_recurrent_steps
        assert max_steps is not None
        if recurrent_step == max_steps - 1 or should_exit:
            return self._schedule_after_recurrent_step(item, recurrent_step, True)

        item.pending_synthetic_exit_step = recurrent_step
        # Carry the staged gate copy so it is awaited (and timed) at the next recurrent step; its
        # value is discarded because the exit decision comes from the recorded schedule.
        item.pending_gate = next_pending_gate
        return self._schedule_after_recurrent_step(item, recurrent_step, False)

    def _stage_coda_batch(self, items: list[DepthWorkItem]) -> None:
        items = self._drop_cancelled_items(items)
        if not items:
            return
        with nvtx.range("cdb.coda_stage"):
            pending = self.runner.stage_coda_batch(
                items,
                allow_async=self.cdb_config.use_async_batching,
            )
        self._trace("coda", len(items))
        if self.cdb_config.use_async_batching:
            self._stage_eager_reentries(pending)
        self.pending_coda_results.append(pending)
        self.last_stats.max_pending_coda_results = max(
            self.last_stats.max_pending_coda_results, len(self.pending_coda_results)
        )

    def _stage_eager_reentries(self, pending: PendingCodaResult) -> None:
        """Re-enter each coda token's successor at the head of the mixed-depth queue."""

        self._enqueue_prelude_items(self._build_eager_reentries(pending), front=True, site="after_coda_eager")

    def _select_eager_reentries(self, pending: PendingCodaResult) -> tuple[list[DepthWorkItem], list[int]]:
        """Stage each coda token's successor before the host has seen the sampled token.

        The prelude gathers the real token from the coda output on device.
        Length-limited or allocation-blocked requests fall back to consume-time staging.
        Late finishes cancel their staged successors.
        """

        eager_flags = [False] * len(pending.items)
        eager_items: list[DepthWorkItem] = []
        eager_rows: list[int] = []
        for idx, item in enumerate(pending.items):
            state = item.state
            if state.will_finish_on_pending_token():
                continue
            state.position_offset += 1
            if not self._try_allocate_one_decode_token(state):
                state.position_offset -= 1
                continue
            eager_flags[idx] = True
            eager_items.append(self._build_decode_item(state, token_id=TMP_TOKEN_ID))
            eager_rows.append(idx)
        pending.eager_reentries = eager_flags
        return eager_items, eager_rows

    def _build_eager_reentries(self, pending: PendingCodaResult) -> list[DepthWorkItem]:
        """Stage each coda token's successor and run its prelude, for the refill scheduler's mixed-depth queue."""

        eager_items, eager_rows = self._select_eager_reentries(pending)
        if not eager_items:
            return []
        self._record_prelude("after_coda_eager", len(eager_items), gathered=len(eager_items))
        with nvtx.range("cdb.prelude.after_coda_eager"):
            self.runner.compute_prelude_batch(eager_items, device_groups=[(pending, eager_rows)])
        return eager_items

    def _drop_cancelled_items(self, items: list[DepthWorkItem]) -> list[DepthWorkItem]:
        """Drop items whose request finished while the item was queued (a finish discovered one consume late)."""

        live = [item for item in items if item.state.status != RequestStatus.FINISHED]
        if len(live) != len(items):
            cancelled = [item for item in items if item.state.status == RequestStatus.FINISHED]
            self.runner.release_hidden_slots(cancelled)
            self.last_stats.cancelled_depth_items += len(cancelled)
        return live

    def _drop_cancelled_carried(self, carried: list[CarriedPreludeEntries]) -> list[CarriedPreludeEntries]:
        """Drop carried wave entries whose request finished before their wave launched.

        Carried entries have not run their prelude, so a cancelled one holds no hidden slot; its
        KV is freed by the retirement that cancelled it.
        """

        live: list[CarriedPreludeEntries] = []
        for entries in carried:
            keep = [
                (item, row)
                for item, row in zip(entries.items, entries.rows, strict=True)
                if item.state.status != RequestStatus.FINISHED
            ]
            self.last_stats.cancelled_depth_items += len(entries.items) - len(keep)
            if keep:
                items, rows = (list(part) for part in zip(*keep, strict=True))
                live.append(CarriedPreludeEntries(result=entries.result, items=items, rows=rows))
        return live

    def _consume_ready_coda_results(self, *, block: bool) -> None:
        while self.pending_coda_results:
            pending = self.pending_coda_results[0]
            if not block and not pending.is_ready():
                return
            self.pending_coda_results.popleft()
            self._consume_coda_result(pending)

    def _consume_coda_result(self, pending: PendingCodaResult) -> None:
        with nvtx.range("cdb.coda_consume"):
            new_tokens = self.runner.consume_coda_result(pending)
        items = pending.items
        self.last_stats.coda_batches += 1
        self.last_stats.coda_tokens += len(items)

        eager_reentries = pending.eager_reentries or [False] * len(items)
        new_items = []
        for item, token_id, was_eager in zip(items, new_tokens, eager_reentries, strict=True):
            state = item.state
            if state.status == RequestStatus.FINISHED:
                # Finished (EOS) while this token was in flight; the sampled token is discarded.
                continue
            if was_eager:
                # The successor already ran its prelude and is queued; the token is bookkeeping only. A
                # finish here is a content finish (EOS or stop match) or, under replay_eos_finishes,
                # a length finish whose prediction was disabled; either way a predicted length finish
                # is never re-entered, and retiring cancels the queued successor at its next pop.
                if state.update_and_check_completion(token_id):
                    self._retire_request(state)
                continue
            if self._advance_after_coda(item, token_id):
                continue
            depth_item = self._stage_decode_token(state)
            if depth_item is not None:
                new_items.append(depth_item)

        self._prelude_and_enqueue_depth(new_items, site="after_coda_staged", front=True)

    def _advance_after_coda(self, item: DepthWorkItem, token_id: int) -> bool:
        """Advance a request by its sampled coda token; return whether it finished.

        Shared by the refill and no-refill coda consumers so the position/completion contract
        has one implementation.
        """

        state = item.state
        state.position_offset += 1
        is_finished = state.update_and_check_completion(token_id)
        if is_finished:
            self._retire_request(state)
        return is_finished

    def _advance_after_prefill(self, state: RequestState, token_id: int) -> bool:
        """Register a prompt's first decode token; return whether the request finished.

        Prefill already advanced ``position_offset`` past the prompt while writing KV,
        so this only records the sampled token as the first generated token (which sets
        ``tokens_to_process`` for the decode phase) and finishes the request when the
        prompt's ``max_new_tokens`` is 1.
        """

        is_finished = state.update_and_check_completion(token_id)
        if is_finished:
            self._retire_request(state)
        return is_finished

    def _retire_request(self, state: RequestState) -> None:
        """Finish a completed request: free its blocks, capture its output, and lift the drain block.

        The output is captured from the finishing object (the post-reset object under recompute
        preemption, whose ``to_generation_output`` recovers the true prompt/generation split), keyed by
        id. A finish frees blocks, so any post-preemption admission block is lifted here.
        """

        self.scheduler.record_completion(state)
        self.scheduler.finish_request(state.request_id)
        self.stalled_decoders.pop(state.request_id, None)
        self.finished_outputs[state.request_id] = state.to_generation_output()
        self.scheduler.block_new_requests = False

    def _raise_scheduler_stall(self) -> None:
        """Report a scheduler tick that could place no work and has nothing in flight to wait on.

        Reached only when there is no coda work, no recurrent work, no coda result in flight, no stalled
        decoder, and no prompt that could be placed - the caller has ruled out each in turn. So the cache
        is genuinely stuck, and the cause is the leading prefill candidate. Mirrors the two failures the
        full-depth engine reports.

        Reads the bucket before the launch gate. A gate that withholds a bucket it judges too small to be
        worth a forward pass says nothing about why the cache is stuck, and an empty list from it would
        send the reader looking for a bookkeeping bug that is not there.
        """

        candidates = self.scheduler.get_prefill_bucket()
        if not candidates:
            raise CacheFullError(
                "The CDB scheduler could place no work and has no coda result in flight, with no prefill "
                "candidate to explain it. This is a scheduler bookkeeping bug."
            )
        state = candidates[0]
        request_len = min(len(state.remaining_prefill_tokens), self.cache.max_num_batched_tokens)
        blocks_needed = -(-(state.current_len() + request_len) // self.cache.block_size) - state.allocated_blocks
        if not self.cache.will_allocation_be_successful(blocks_needed, state.allocated_blocks):
            if self.cdb_config.kv_pressure_mode == "reserve":
                # Impossible under reserve admission; a failure here signals a reservation-accounting bug.
                raise CacheFullError(
                    f"Reserve admission over-committed the KV cache: request {state.request_id} could not "
                    "allocate its prefill blocks. This is a bug in reservation accounting."
                )
            raise CacheFullError(
                f"KV cache full: request {state.request_id} needs {blocks_needed} blocks for its next "
                f"prefill chunk, {self.cache.get_num_free_blocks()} are free, and no request is decoding "
                "whose drain would free more (increase num_blocks or reduce max_model_len)."
            )
        raise RuntimeError(
            f"No request could be scheduled: request {state.request_id} reads {state.current_len()} KV "
            f"tokens, exceeding the cache budget of {self.cache.num_pages} (the prompt is too long for "
            "the KV cache)."
        )

    def _restore_offloaded_requests(self) -> list[RequestState]:
        """Bring offloaded requests back into the active set and copy their KV in from the CPU pool.

        KV is restored before compute can read the request.
        The refill loop enqueues the returned requests as depth work.
        """

        if not self.offloading_manager.offloading_enabled:
            return []
        restored = self.scheduler.admit_offloaded_restores()
        if not restored:
            return []
        self.offloading_manager.restore_scheduled_requests(
            [FutureRequestState(state, has_new_token=True, query_length=1) for state in restored]
        )
        return restored

    def _sample_growing_requests(self) -> None:
        """Record how many active requests have generated past their prefill token.

        A request parked behind the working set holds only its prompt's KV; one inside the set grows a
        block every ``block_size`` tokens. The peak of this count, not the resident count, is what the
        cache has to absorb.

        Counted against the request's true prompt rather than its generated-token count, which a soft
        reset zeroes while folding the generation onto the prompt.
        """

        growing = sum(1 for state in self.scheduler.active_requests.values() if state.has_grown_past_prompt())
        self.last_stats.max_growing_requests = max(self.last_stats.max_growing_requests, growing)

    def _enqueue_refills(self, items: list[DepthWorkItem]) -> None:
        """Re-queue the items a recurrent batch stepped but did not retire, at the head of the queue.

        See :meth:`~looped_cdb.continuous_depth_batching.scheduler.CDBScheduler.enqueue_depth_front`.
        """

        if items:
            self.scheduler.enqueue_depth_front(items)

    def _record_prelude(self, site: PreludeSite, count: int, *, gathered: int = 0) -> None:
        """Count one prelude launch carrying ``count`` tokens. Callers skip empty launches.

        ``gathered`` is how many of them take their id from a staged batch's device tokens
        instead of the host.
        """

        self.last_stats.prelude_batches[site] += 1
        self.last_stats.prelude_tokens[site] += count
        if gathered:
            self.last_stats.prelude_gathered_tokens[site] += gathered
        if 0 < gathered < count:
            self.last_stats.prelude_fused_batches[site] += 1

    def _prelude_and_enqueue_depth(self, items: list[DepthWorkItem], *, site: PreludeSite, front: bool = False) -> None:
        """Embed freshly sampled tokens and queue their recurrent-step-0 work.

        ``front`` is set for a request resuming a launch slot it already held (a coda token, a restored
        offload, an unstalled decoder) and left unset for a prompt's first token, which queues behind
        the requests already being served.
        """

        if items:
            self._record_prelude(site, len(items))
            with nvtx.range(f"cdb.prelude.{site}"):
                self.runner.compute_prelude_batch(items)
        self._enqueue_prelude_items(items, front=front, site=site)

    def _enqueue_prelude_items(self, items: list[DepthWorkItem], *, front: bool, site: PreludeSite) -> None:
        if front:
            self.scheduler.enqueue_depth_front(items)
        else:
            for item in items:
                self.scheduler.enqueue_depth(item)
        if items:
            self._trace("prelude", len(items), site=site)

    def _build_decode_item(self, state: RequestState, *, token_id: int | None = None) -> DepthWorkItem:
        """Reset per-token depth state and build the recurrent-step-0 work item for the pending token.

        The decode block for the token must already be allocated. ``token_id`` overrides the host-known
        pending token for eager re-entry, whose input is the placeholder the device-side gather fills
        from the coda batch's sampled tokens.
        """

        state.reset_depth_state_for_next_token()
        return DepthWorkItem(
            state=state,
            token_id=state.tokens_to_process[0] if token_id is None else token_id,
            token_position=state.position_offset,
            recurrent_step=0,
            synthetic_exit_depth=state.next_synthetic_exit_depth(),
        )

    def _stage_decode_token(self, state: RequestState) -> DepthWorkItem | None:
        """Build the next decode work item, or stall the request when the KV cache cannot hold its token.

        Failed allocations stall at a clean token boundary and retry after other work frees blocks.
        Preemption occurs only when no other work can run.
        """

        if not state.tokens_to_process or state.tokens_to_process[0] < 0:
            raise RuntimeError(f"Request {state.request_id} has no sampled token ready for CDB decode")
        if self._try_allocate_one_decode_token(state):
            self.stalled_decoders.pop(state.request_id, None)
            return self._build_decode_item(state)
        if self.cdb_config.kv_pressure_mode == "none":
            # No pressure policy: the workload over-subscribed a cache it was assumed to fit.
            raise CacheFullError(
                f"KV cache full staging a decode block for request {state.request_id} and "
                "kv_pressure_mode='none' does no preemption. Increase num_blocks, reduce load, or select "
                "kv_pressure_mode='reserve'/'recompute'/'offload'."
            )
        if self.cdb_config.kv_pressure_mode == "reserve":
            # Impossible under reserve admission; a failure here signals a reservation-accounting bug.
            raise CacheFullError(
                f"Reserve admission over-committed the KV cache: request {state.request_id} could not "
                "allocate a decode block. This is a bug in reservation accounting."
            )
        self.stalled_decoders[state.request_id] = state
        return None

    def _restage_stalled_decoders(self) -> list[DepthWorkItem]:
        """Retry the decode tokens that could not allocate, now that other work may have freed blocks."""

        items: list[DepthWorkItem] = []
        for state in list(self.stalled_decoders.values()):
            item = self._stage_decode_token(state)
            if item is None:
                # Every stalled request needs one new block, so if the leading one cannot get it,
                # neither can any behind it. Stopping here skips no stageable work.
                break
            items.append(item)
        return items

    def _preempt_stalled_decoder(self) -> None:
        """Yield one stalled request's KV so the rest of the batch can drain.

        Reached only when nothing else can run: no coda, no recurrent work, and no stalled decoder can
        allocate. The victim is drawn from the stalled set rather than from all active requests, because
        only a stalled request is at a clean token boundary rather than mid-recurrence; it resumes later
        through the decode path with no wake-up token re-run through the full-depth prefill.
        """

        # Freeing a request's own KV only helps if another request keeps draining and frees more. With
        # this request alone active it needs those blocks back plus one more, so preempting would stall
        # the restore instead of relieving the pressure.
        if len(self.scheduler.active_requests) <= 1:
            request_id = next(iter(self.stalled_decoders))
            raise CacheFullError(
                f"A single request's KV cache does not fit: request {request_id} could not allocate "
                "a decode block and no other request can be preempted (increase num_blocks or reduce "
                "max_model_len)."
            )
        # Preempt the stalled request with the fewest computed tokens, the same cost rule
        # ``pop_request_to_evict`` applies to the full-depth engine's active set, ties breaking toward
        # the most recent staller. Stall order is no more a proxy for investment than admission order
        # is: a restored request that stalls again moves to the back of this map carrying its whole KV.
        request_id = min(
            reversed(self.stalled_decoders), key=lambda req_id: self.stalled_decoders[req_id].current_len()
        )
        state = self.stalled_decoders.pop(request_id)
        self.offloading_manager.preempt_request(request_id, state)

    def _try_allocate_one_decode_token(self, state: RequestState) -> bool:
        """Ensure a KV slot for the next decode token; return False if the cache is full."""

        current_len = state.current_len()
        occupancy = state.allocated_blocks * self.cache.block_size - current_len
        if occupancy >= 1 and state.allocated_blocks > 0:
            return True
        # Allocate exactly the blocks needed to hold one more token (ceil), no spare - the same
        # exact-ceil rule as the scheduler's _allocate_blocks_if_needed, so a decode never claims a
        # block past what reserved_peak_blocks assumes. Only reached when the last block is exactly full.
        blocks_needed = -(-(current_len + 1) // self.cache.block_size) - state.allocated_blocks
        allocated = self.cache.allocate_blocks(blocks_needed, state.request_id, state.allocated_blocks)
        if allocated is None:
            return False
        state.allocated_blocks += allocated
        return True

    def _should_exit(
        self,
        item: DepthWorkItem,
        recurrent_step: int,
        exit_signals: torch.Tensor | None,
        batch_index: int,
    ) -> bool:
        max_steps = self.cdb_config.max_recurrent_steps
        assert max_steps is not None
        if recurrent_step == max_steps - 1:
            return True
        if not item.apply_exit_gate:
            return False
        if item.synthetic_exit_depth is not None:
            return recurrent_step >= item.synthetic_exit_depth
        if self.cdb_config.exit_threshold is None:
            return False
        if item.preloop_exit_step is not None:
            return recurrent_step >= item.preloop_exit_step
        if exit_signals is None:
            raise RuntimeError("exit_threshold requires the CDB model adapter to return exit signals")

        return self.exit_policy.should_exit(
            item.policy_state,
            source_step=recurrent_step,
            signal=exit_signals[0, batch_index, 0],
        )

    def _consume_delayed_gate(self, item: DepthWorkItem) -> bool:
        pending_gate = item.pending_gate
        if pending_gate is None:
            return False
        should_exit = self.exit_policy.should_exit(
            item.policy_state,
            source_step=pending_gate.recurrent_step,
            signal=pending_gate.signal(),
        )
        item.pending_gate = None
        return should_exit

    def _consume_delayed_synthetic_exit(self, item: DepthWorkItem) -> bool:
        # Pay the staged signal GPU->CPU readout (await the side-stream copy) even though the exit
        # decision comes from the recorded schedule; the value is discarded. This makes synthetic
        # replay time the same signal-sync cost a live policy would, instead of hiding it.
        if item.pending_gate is not None:
            item.pending_gate.signal()
            item.pending_gate = None
        pending_step = item.pending_synthetic_exit_step
        if pending_step is None:
            return False
        item.pending_synthetic_exit_step = None
        if item.synthetic_exit_depth is None:
            return False
        return pending_step >= item.synthetic_exit_depth

    def _uses_delayed_gate_consumption(self, recurrent_kwargs: dict[str, object]) -> bool:
        return self.exit_policy.uses_delayed_online_signal(
            signal_enabled=bool(recurrent_kwargs.get("use_early_exit_gate", True))
        )

    def _uses_preloop_gate(self) -> bool:
        """Whether exits come from a pre-loop distribution resolved at prelude time."""

        return self.exit_policy.decides_before_loop()

    def _uses_delayed_synthetic_exit(self) -> bool:
        return self.exit_policy.uses_delayed_synthetic_exit()

    def _uses_delayed_gate_readout(self, recurrent_kwargs: dict[str, object]) -> bool:
        """Whether this recurrent launch stages scalar signals for later routing."""

        return self.exit_policy.stages_delayed_signal(
            signal_enabled=bool(recurrent_kwargs.get("use_early_exit_gate", True))
        )

    def _free_finished_or_active(self) -> None:
        self.scheduler.free_unfinished_requests()

    def _validate_model_adapter(self) -> None:
        missing = [name for name in ("prelude", "recurrent_step", "lm_head") if not hasattr(self.model_adapter, name)]
        if missing:
            raise TypeError(f"Continuous depth batching requires staged model adapter methods: {missing}")

    def _validate_model_kwargs(self, model_kwargs: dict | None) -> None:
        if (
            self.cdb_config.exit_threshold is not None
            and model_kwargs is not None
            and model_kwargs.get("use_early_exit_gate") is False
        ):
            raise ValueError("exit_threshold CDB requires use_early_exit_gate to stay enabled")

    def _validate_exit_gate_serving(self) -> None:
        """Validate that adapter-declared offset-1 signals retain their overlap timeline."""

        self.exit_policy.validate_serving_timing()

    @staticmethod
    def _validate_generate_inputs(input_ids: list[list[int]], max_new_tokens: int | list[int]) -> None:
        """Validate token-level generation inputs before scheduling begins."""

        if not input_ids:
            raise ValueError("input_ids must contain at least one prompt")
        if isinstance(max_new_tokens, list):
            if len(max_new_tokens) != len(input_ids):
                raise ValueError(
                    f"max_new_tokens has {len(max_new_tokens)} entries but there are {len(input_ids)} prompts"
                )
            invalid = [idx for idx, value in enumerate(max_new_tokens) if value < 1]
            if invalid:
                raise ValueError(f"max_new_tokens must be >= 1 for every request, but got <1 at indices {invalid}")
        elif max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, but got {max_new_tokens}")
        empty_prompt_indices = [idx for idx, prompt_ids in enumerate(input_ids) if not prompt_ids]
        if empty_prompt_indices:
            raise ValueError(f"input_ids contains empty prompts at indices {empty_prompt_indices}")

    def _validate_exit_depths(
        self,
        input_ids: list[list[int]],
        max_new_tokens: list[int],
        exit_depths: list[list[int]] | None,
    ) -> None:
        """Validate a per-request replay schedule against the config and lengths."""

        if exit_depths is None:
            return
        if not self.cdb_config.synthetic_exit_replay:
            raise ValueError("exit_depths requires cdb_config.synthetic_exit_replay=True")
        if self.cdb_config.exit_threshold is not None:
            raise ValueError("exit_depths cannot be combined with exit_threshold")
        if len(exit_depths) != len(input_ids):
            raise ValueError(f"exit_depths has {len(exit_depths)} entries but there are {len(input_ids)} requests")
        max_steps = self.cdb_config.max_recurrent_steps
        assert max_steps is not None
        for idx, depths in enumerate(exit_depths):
            expected = max_new_tokens[idx] - 1
            if len(depths) != expected:
                raise ValueError(
                    f"exit_depths[{idx}] has {len(depths)} entries but the request stages {expected} decode tokens "
                    "(max_new_tokens - 1)"
                )
            for depth in depths:
                if not 0 <= depth < max_steps:
                    raise ValueError(
                        f"exit_depths[{idx}] value {depth} is out of range [0, {max_steps}); values are 0-based "
                        "recurrent steps"
                    )
        if copies_exit_kv(self.cdb_config.kv_policy):
            # Copy-on-exit routing fills the deeper slots at exit time, so any replay depth is safe.
            return
        slots_per_layer = resolve_kv_slots_per_layer(
            self.cdb_config.kv_policy,
            total_recurrent_steps=max_steps,
            requested_slots=self.cdb_config.kv_slots_per_layer,
        )
        delayed_extra_step = int(self.cdb_config.delay_gate_consumption and self.cdb_config.use_async_batching)
        for depths in exit_depths:
            for depth in depths:
                if depth + 1 + delayed_extra_step < slots_per_layer:
                    raise ValueError(
                        "Every replay exit depth must write all static KV slots before exit: "
                        f"kv_slots_per_layer={slots_per_layer}, exit step {depth + 1 + delayed_extra_step}"
                    )
