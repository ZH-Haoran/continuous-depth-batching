# Copyright 2024 The HuggingFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified for looped-model serving and continuous depth batching.
# See THIRD_PARTY_NOTICES.md and LICENSES/Apache-2.0.txt.

"""High-level token API for native continuous batching.

The engine wires together the native continuous batching components:
request state, paged KV cache, FIFO scheduler, static I/O tensors, and model
runner. It intentionally stays token-level for now. Tokenization, chat
templates, and benchmark-specific setup belong in callers/scripts.

Reference:
https://github.com/huggingface/transformers/blob/main/src/transformers/generation/continuous_batching/continuous_api.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from looped_cdb.arrivals import ArrivalQueue
from looped_cdb.benchmarks import nvtx
from looped_cdb.kv_cache_sizing import resolve_num_blocks
from looped_cdb.offloading_manager import OffloadingManager
from looped_cdb.paged_cache_geometry import PagedKVCacheGeometry
from looped_cdb.scheduler_base import resolve_max_num_seqs
from looped_cdb.utils import cap_max_new_tokens_to_model_len, normalize_max_new_tokens

from .cache import PagedAttentionCache
from .config import ContinuousBatchingConfig
from .input_outputs import ContinuousBatchingIOs, OutputCopyHandle
from .model_runner import ModelRunner
from .requests import FutureRequestState, GenerationOutput, RequestState, RequestStatus
from .scheduler import FIFOScheduler


class CacheFullError(RuntimeError):
    """Raised when the scheduler cannot place any request because the KV cache is exhausted.

    A subclass of :class:`RuntimeError`, so existing callers that catch ``RuntimeError`` still
    handle it; the distinct type lets a caller (e.g. the decode-latency measurement) tell "the
    batch does not fit" apart from other engine errors and report it clearly.
    """


@dataclass
class ContinuousBatchingStats:
    """Per-generation batch counters for continuous batching verification and profiling.

    A tick is a prefill batch unless every scheduled request advances by a single query token
    while decoding; a decode batch runs the whole batch one token forward at the model's full
    recurrent depth. The counters accumulate across a single ``generate_batch`` call and are
    reset alongside the engine's other per-generation state.
    """

    prefill_batches: int = 0
    decode_batches: int = 0
    prefill_requests: int = 0
    prefill_query_tokens: int = 0
    decode_tokens: int = 0

    @property
    def mean_prefill_batch_size(self) -> float:
        """Mean number of requests per prefill batch, or 0.0 when no prefill batch ran."""

        return self.prefill_requests / self.prefill_batches if self.prefill_batches else 0.0

    @property
    def mean_prefill_query_tokens(self) -> float:
        """Mean number of query tokens per prefill batch, or 0.0 when no prefill batch ran."""

        return self.prefill_query_tokens / self.prefill_batches if self.prefill_batches else 0.0

    @property
    def mean_decode_batch_size(self) -> float:
        """Mean number of sequences advanced per decode batch, or 0.0 when no decode batch ran."""

        return self.decode_tokens / self.decode_batches if self.decode_batches else 0.0


@dataclass
class _TickOutcome:
    """What one scheduler tick did, so a driving loop can react to it."""

    use_decode_fast_path: bool
    batch_size: int
    completed_requests: int


@dataclass
class _InFlightBatch:
    """A launched batch whose sampled tokens the host has not consumed yet."""

    futures: list[FutureRequestState]
    num_new_tokens: int  # rows with a sampled token, in logits-row order
    output_copy: OutputCopyHandle


@dataclass
class DecodeStepTiming:
    """Latency of one steady-state decode step, isolated from prefill.

    The timed decode window is bracketed by two full-device synchronizations (one when it
    opens, one when it closes) - graph-clean, no per-step sync. ``per_step_ms`` is the host
    wall clock over that window divided by the number of timed steps; it is the quantity that
    decides whether early exit saves wall clock, and is directly comparable to the older
    differencing measurement. ``gpu_per_step_ms`` is the same window timed with CUDA events on
    the engine's compute stream (the pure on-GPU floor, excluding host gaps); it is <= the wall
    time and the two agree when the decode loop keeps the GPU saturated. ``context_min`` /
    ``context_max`` bracket the KV length across the batch when timing opened; with
    ``safety_margin=0`` all prompts prefill before any decode, so they stay equal.
    """

    batch_size: int
    context_length: int
    context_min: int
    context_max: int
    warmup_decode_steps: int
    timed_decode_steps: int
    wall_elapsed_ms: float
    gpu_elapsed_ms: float
    per_step_ms: float
    gpu_per_step_ms: float


@dataclass
class ContinuousBatchingEngine:
    """Small native continuous batching engine for token-level generation.

    With ``use_async_batching`` (the default) the loop runs vLLM-style async scheduling: the host
    schedules, prepares, and launches batch k+1 before consuming batch k's sampled tokens, hiding
    one full step of host work behind the device. The scheduler can run ahead because a decoding
    request's pending token is a placeholder the device fills in (see
    :class:`~looped_cdb.continuous_batching.input_outputs.ContinuousBatchingIOs`); a request that
    reaches its length limit is simply not re-scheduled, while an EOS finish is only visible one
    step late, so it computes one extra token that is discarded on consumption. With
    ``use_async_batching=False`` the same machinery runs serialized: each batch's tokens are
    consumed before the next batch is scheduled. Both modes produce identical tokens.
    """

    model: nn.Module
    cb_config: ContinuousBatchingConfig
    cache: PagedAttentionCache
    scheduler: FIFOScheduler
    inputs_and_outputs: ContinuousBatchingIOs
    runner: ModelRunner
    offloading_manager: OffloadingManager
    last_stats: ContinuousBatchingStats = field(default_factory=ContinuousBatchingStats)

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        cb_config: ContinuousBatchingConfig,
        *,
        dtype: torch.dtype | None = None,
    ) -> ContinuousBatchingEngine:
        """Create a continuous batching engine for an already-loaded model."""

        model_config = model.config
        cb_config = cb_config.get_resolved(model_config)
        first_param = next(model.parameters())
        device = first_param.device
        dtype = first_param.dtype if dtype is None else dtype
        geometry = PagedKVCacheGeometry.from_model(model_config, block_size=cb_config.block_size, dtype=dtype)
        cb_config.num_blocks = resolve_num_blocks(cb_config.num_blocks, cb_config.mem_fraction_static, geometry, device)
        cache = PagedAttentionCache(
            config=model_config,
            continuous_batching_config=cb_config,
            device=device,
            dtype=dtype,
            geometry=geometry,
        )
        # Resolved onto the config, not just inside the scheduler: the model runner pads decode
        # batches and walks warmup shapes up to ``max_num_seqs``, and those buffers are sized by the
        # cache's token budget, so the two must be the same number.
        cb_config.max_num_seqs = resolve_max_num_seqs(cb_config.max_num_seqs, cache.max_num_batched_tokens)
        io_kwargs = {
            "cache": cache,
            "config": model_config,
            "device": device,
            "model_dtype": dtype,
        }
        inputs_and_outputs = ContinuousBatchingIOs(**io_kwargs)
        runner = ModelRunner(
            engine_config=cb_config,
            inputs_and_outputs=inputs_and_outputs,
            cache=cache,
        )
        runner.configure_sampled_row_gather(model)
        scheduler = FIFOScheduler(
            cache,
            safety_margin=cb_config.safety_margin,
            kv_pressure_mode=cb_config.kv_pressure_mode,
            max_model_len=cb_config.max_model_len,
            max_num_seqs=cb_config.max_num_seqs,
            min_free_slots=cb_config.min_free_slots,
        )
        offloading_manager = OffloadingManager(
            cache,
            scheduler,
            cpu_offload_space_gib=cb_config.cpu_offload_space,
            compute_stream=inputs_and_outputs.compute_stream,
            pin_memory=cache.device.type == "cuda",
        )
        return cls(
            model=model,
            cb_config=cb_config,
            cache=cache,
            scheduler=scheduler,
            inputs_and_outputs=inputs_and_outputs,
            runner=runner,
            offloading_manager=offloading_manager,
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
        model_kwargs: dict[str, Any] | None = None,
        arrival_offsets_s: list[float] | None = None,
    ) -> list[GenerationOutput]:
        """Generate from a batch of tokenized prompts.

        ``max_new_tokens`` may be a scalar applied to every request or a per-request
        list. Every token is decoded at the model's full recurrent depth.
        ``arrival_offsets_s`` optionally releases the requests open-loop at wall-clock
        offsets from the start of generation (one non-decreasing offset per request, see
        :mod:`looped_cdb.arrivals`) instead of submitting them all upfront; new arrivals
        are admitted at tick boundaries and the loop sleeps when the engine is idle.
        """

        self._validate_generate_inputs(input_ids, max_new_tokens)
        max_new_tokens = normalize_max_new_tokens(max_new_tokens, len(input_ids))
        max_new_tokens = cap_max_new_tokens_to_model_len(input_ids, max_new_tokens, self.cb_config.max_model_len)
        self.reset()
        if warmup:
            self.runner.warmup(self.model, model_kwargs)

        states = [
            RequestState(
                request_id=f"request-{idx}",
                initial_tokens=prompt_ids,
                max_new_tokens=max_new_tokens[idx],
                eos_token_id=eos_token_id,
                stop_sequences=[list(sequence) for sequence in stop_sequences] if stop_sequences else [],
                replay_eos_finishes=self.cb_config.replay_eos_finishes,
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

        completed_requests = 0
        # Collect each request's output when it finishes, keyed by id. Soft-reset preemption replaces a
        # request's state object with a fresh one (same id), so the original ``states`` entries can be
        # stale; the finishing object is the source of truth.
        finished_outputs: dict[str, GenerationOutput] = {}
        in_flight: _InFlightBatch | None = None
        while (
            (arrivals is not None and arrivals.pending)
            or self.scheduler.has_pending_requests()
            or in_flight is not None
        ):
            if arrivals is not None:
                for state in arrivals.release_due():
                    self.scheduler.add_waiting_request(state)
                if not self.scheduler.has_pending_requests() and in_flight is None:
                    if arrivals.pending:
                        # Nothing to run and nothing in flight: idle until the next arrival.
                        arrivals.wait_for_next_arrival()
                    continue
            in_flight, outcome = self._execute_tick(
                model_kwargs,
                in_flight,
                completed_requests=completed_requests,
                finished_outputs=finished_outputs,
            )
            completed_requests = outcome.completed_requests

        self.scheduler.close_steady_window()
        return [finished_outputs[state.request_id] for state in states]

    def _execute_tick(
        self,
        model_kwargs: dict[str, Any] | None,
        in_flight: _InFlightBatch | None,
        *,
        completed_requests: int,
        finished_outputs: dict[str, GenerationOutput] | None = None,
    ) -> tuple[_InFlightBatch | None, _TickOutcome]:
        """Run one engine tick: schedule and launch the next batch, then consume the lagged one.

        In pipelined mode (``use_async_batching``) the launch happens before ``in_flight`` (the batch
        launched by the previous tick) is consumed, so the host's scheduling and preparation work
        overlaps the device's compute. In serialized mode the tick consumes the batch it just
        launched and never carries one. Returns the batch left in flight and what the tick did;
        ``completed_requests`` is the running total coming in, updated in the outcome.
        """

        pipelined = self.cb_config.use_async_batching
        launched: _InFlightBatch | None = None
        cache_full = False
        requests_in_batch: list[FutureRequestState] = []
        use_decode_fast_path = False
        num_q_tokens = max_kv_read = padded_q = padded_kv = 0

        if self.scheduler.has_pending_requests():
            with nvtx.range("cb.schedule"):
                scheduled = self.scheduler.schedule_batch(
                    token_budget=self.cache.max_num_batched_tokens,
                    cache_budget=self.cache.num_pages,
                )
            use_decode_fast_path = scheduled.use_decode_fast_path
            num_q_tokens, max_kv_read = scheduled.num_q_tokens, scheduled.max_kv_read
            if scheduled.cache_exhausted:
                # The cache is full and nothing could be placed. Consuming the in-flight batch first may
                # finish requests and free their blocks, so the hard-failure handling below only runs
                # when there was nothing left to consume.
                cache_full = True
            elif not scheduled.requests and in_flight is None:
                # An empty batch that is not cache-exhausted, with nothing in flight and no decode
                # work, means a pending prefill request never fit for a soft reason -- its KV read
                # (the prompt's cached length) exceeds the cache budget -- so the prompt is too long
                # for this cache. With a batch in flight the empty schedule is transient: every
                # remaining request's pending token is about to be consumed.
                raise RuntimeError(
                    "No request could be scheduled: a pending prefill request's KV read exceeds the "
                    "cache budget (the prompt is too long for the KV cache)."
                )
            else:
                requests_in_batch = scheduled.requests

        if requests_in_batch:
            # Copy back the KV of any offloaded requests that were just re-scheduled, before compute reads it.
            self.offloading_manager.restore_scheduled_requests(requests_in_batch)

            # A tick is a decode batch only when every scheduled request advances by a single token while
            # decoding; anything else (a full or chunked prompt) is prefill work. Recorded per tick so the
            # benchmark can report prefill/decode batch counts and widths for the engine.
            is_decode_batch = all(
                future.query_length == 1 and future.state.status == RequestStatus.DECODING
                for future in requests_in_batch
            )
            if is_decode_batch:
                self.last_stats.decode_batches += 1
                self.last_stats.decode_tokens += len(requests_in_batch)
            else:
                self.last_stats.prefill_batches += 1
                self.last_stats.prefill_requests += len(requests_in_batch)
                self.last_stats.prefill_query_tokens += num_q_tokens

            with nvtx.range("cb.prepare"):
                with nvtx.range("cb.prepare.stage"):
                    padded_q, padded_kv = self.runner.maybe_pad_inputs(
                        num_q_tokens=num_q_tokens,
                        max_kv_read=max_kv_read,
                        use_decode_fast_path=use_decode_fast_path,
                    )
                    self.inputs_and_outputs.prepare_batch_tensors(
                        requests_in_batch=requests_in_batch,
                        use_decode_fast_path=use_decode_fast_path,
                        num_q_tokens=padded_q,
                        max_kv_read=padded_kv,
                    )
                with nvtx.range("cb.prepare.h2d"):
                    batch_data = self.inputs_and_outputs.get_model_kwargs(
                        use_padding=self.runner.pads_batch(use_decode_fast_path)
                    )
                    if model_kwargs is not None:
                        batch_data.update(model_kwargs)
            # In pipelined mode the compute and retrieve sections only enqueue device work; the host
            # blocks on this batch's outputs one tick later, in the update section.
            with nvtx.range("cb.compute.decode" if is_decode_batch else "cb.compute.prefill"):
                self.runner.compute_batch(self.model, batch_data)
            with nvtx.range("cb.retrieve"):
                output_copy = self.inputs_and_outputs.enqueue_output_copy()
            launched = _InFlightBatch(
                futures=requests_in_batch,
                num_new_tokens=sum(1 for future in requests_in_batch if future.has_new_token),
                output_copy=output_copy,
            )

        to_consume = in_flight if pipelined else launched
        carried = launched if pipelined else None
        if to_consume is not None:
            with nvtx.range("cb.update"):
                completed_requests = self._consume_batch(
                    to_consume,
                    completed_requests=completed_requests,
                    finished_outputs=finished_outputs,
                )

        if cache_full and to_consume is None:
            if self.cb_config.kv_pressure_mode == "none":
                # No pressure policy: the workload over-subscribed a cache it was assumed to fit.
                raise CacheFullError(
                    "KV cache full and kv_pressure_mode='none' does no preemption. Increase num_blocks, "
                    "reduce load, or select kv_pressure_mode='reserve'/'recompute'/'offload'."
                )
            if self.cb_config.kv_pressure_mode == "reserve":
                # Impossible by construction (a request is only admitted if its worst-case peak already
                # fits), so it signals a reservation-accounting bug rather than genuine pressure.
                raise CacheFullError(
                    "Reserve admission over-committed the KV cache: an admitted request could not "
                    "allocate. This is a bug in reservation accounting."
                )
            # recompute/offload: preempt one active request to free blocks, then retry on the next tick.
            # Nothing is in flight here (the lagged batch was consumed on the previous tick), so the
            # victim's state is complete and its KV is not being written. With only one active request
            # there is nothing to preempt: its KV alone does not fit.
            if len(self.scheduler.active_requests) > 1:
                self.offloading_manager.offload_one_request()
                return None, _TickOutcome(
                    use_decode_fast_path=False, batch_size=0, completed_requests=completed_requests
                )
            raise CacheFullError(
                "No request could be scheduled and no request can be preempted: a single request's KV "
                "cache does not fit (increase num_blocks or reduce max_model_len)."
            )

        return carried, _TickOutcome(
            use_decode_fast_path=use_decode_fast_path,
            batch_size=len(requests_in_batch),
            completed_requests=completed_requests,
        )

    def _consume_batch(
        self,
        batch: _InFlightBatch,
        *,
        completed_requests: int,
        finished_outputs: dict[str, GenerationOutput] | None,
    ) -> int:
        """Consume a launched batch's sampled tokens and apply them to the request states.

        Returns the updated completion total. A row whose request is already ``FINISHED``
        finished on EOS while this row was in flight; its token is discarded (its blocks were
        freed when it finished, and stale KV written by this row is harmless because any block
        reuse writes on the same stream afterwards).
        """

        new_tokens = self.inputs_and_outputs.consume_output_tokens(batch.output_copy, batch.num_new_tokens)
        new_token_idx = 0
        for future_state in batch.futures:
            if not future_state.has_new_token:
                continue

            token_id = new_tokens[new_token_idx]
            new_token_idx += 1

            state = future_state.state
            if state.status in (RequestStatus.FINISHED, RequestStatus.PENDING):
                continue

            is_finished = state.update_and_check_completion(token_id)
            if is_finished:
                self.scheduler.record_completion(state)
                self.scheduler.finish_request(state.request_id)
                completed_requests += 1
                # A finish frees blocks; lift the post-preemption admission block so waiting
                # requests (offloaded restores first) can be scheduled again.
                self.scheduler.block_new_requests = False
                if finished_outputs is not None:
                    finished_outputs[state.request_id] = state.to_generation_output()
        return completed_requests

    @torch.no_grad()
    def measure_decode_step_latency(
        self,
        input_ids: list[list[int]],
        *,
        max_new_tokens: int,
        warmup_decode_steps: int,
        timed_decode_steps: int,
        model_kwargs: dict[str, Any] | None = None,
    ) -> DecodeStepTiming:
        """Time one steady-state decode step at full batch ``B``, isolated from prefill.

        Submits ``B`` equal-length prompts, drives the scheduler until all of them are
        decoding together (a full-batch decode-fast-path tick), settles for
        ``warmup_decode_steps`` such ticks, then brackets the next ``timed_decode_steps``
        with CUDA events. Prefill happens before the event window and is never timed, so no
        prefill-cancelling second run is needed. This requires the engine's scheduler to run
        with ``safety_margin=0`` so all prefill drains before any decode (build the engine
        with ``ContinuousBatchingConfig(safety_margin=0.0)``); otherwise the batch never
        reaches ``B`` at a uniform context. The caller must also have warmed up the engine
        (so the decode graph for batch ``B`` is captured); ``max_new_tokens`` must be large
        enough that no request finishes before the window closes - otherwise the batch drops
        below ``B`` and this raises.
        """

        if warmup_decode_steps < 1:
            raise ValueError("warmup_decode_steps must be >= 1 to settle the decode graph before timing")
        if timed_decode_steps < 1:
            raise ValueError("timed_decode_steps must be >= 1")
        if max_new_tokens <= warmup_decode_steps + timed_decode_steps:
            raise ValueError(
                f"max_new_tokens ({max_new_tokens}) must exceed warmup+timed decode steps "
                f"({warmup_decode_steps + timed_decode_steps})"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("measure_decode_step_latency requires CUDA for event timing")
        self._validate_generate_inputs(input_ids, max_new_tokens)
        batch_size = len(input_ids)
        context_length = len(input_ids[0])
        if any(len(prompt) != context_length for prompt in input_ids):
            raise ValueError("measure_decode_step_latency expects equal-length prompts")

        self.reset()
        states = [
            RequestState(
                request_id=f"request-{idx}",
                initial_tokens=prompt_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=None,
            )
            for idx, prompt_ids in enumerate(input_ids)
        ]
        for state in states:
            self.scheduler.add_waiting_request(state)

        # The decode compute runs on this stream (``None`` in synchronous mode, i.e. the default
        # stream), so the timing events must be recorded on it to bracket the real work.
        compute_stream = self.inputs_and_outputs.compute_stream
        start_event = torch.cuda.Event(enable_timing=True)
        stop_event = torch.cuda.Event(enable_timing=True)
        full_batch_ticks = 0
        timing_open = False
        wall_start = 0.0
        context_min = context_length
        context_max = context_length
        completed_requests = 0

        in_flight: _InFlightBatch | None = None
        while self.scheduler.has_pending_requests() or in_flight is not None:
            try:
                in_flight, outcome = self._execute_tick(
                    model_kwargs,
                    in_flight,
                    completed_requests=completed_requests,
                )
            except CacheFullError as exc:
                # With safety_margin=0 the batch is assembled by draining all prefill first; if the
                # cache cannot hold every request resident, prefill stalls before decode can begin.
                # Fail loudly with the cause instead of leaving a half-assembled batch.
                raise CacheFullError(
                    f"could not assemble the full batch of {batch_size} requests at context "
                    f"{context_length}: the KV cache is exhausted before all requests are resident. "
                    "Increase num_blocks, or reduce the batch size or context length."
                ) from exc
            completed_requests = outcome.completed_requests

            is_full_batch_decode = (
                outcome.use_decode_fast_path
                and outcome.batch_size == batch_size
                and not self.scheduler.waiting_requests
            )
            if not is_full_batch_decode:
                if timing_open:
                    raise RuntimeError("decode batch dropped below B inside the timed window")
                continue

            full_batch_ticks += 1
            if full_batch_ticks < warmup_decode_steps:
                continue
            if full_batch_ticks == warmup_decode_steps:
                # The warmup ticks are enqueued; open the window just after them. A full-device
                # sync settles that work, then the wall clock and a compute-stream CUDA event
                # both start from the same point so they bracket exactly the timed ticks.
                torch.cuda.synchronize()
                lengths = [state.current_len() for state in states]
                context_min, context_max = min(lengths), max(lengths)
                wall_start = time.perf_counter()
                start_event.record(compute_stream)
                timing_open = True
                continue
            if full_batch_ticks == warmup_decode_steps + timed_decode_steps:
                # The engine runs the decode compute on ``compute_stream``, so the events must be
                # recorded on it; the closing full-device sync then bounds the wall clock too.
                stop_event.record(compute_stream)
                torch.cuda.synchronize()
                wall_ms = (time.perf_counter() - wall_start) * 1000.0
                gpu_ms = start_event.elapsed_time(stop_event)
                return DecodeStepTiming(
                    batch_size=batch_size,
                    context_length=context_length,
                    context_min=context_min,
                    context_max=context_max,
                    warmup_decode_steps=warmup_decode_steps,
                    timed_decode_steps=timed_decode_steps,
                    wall_elapsed_ms=wall_ms,
                    gpu_elapsed_ms=gpu_ms,
                    per_step_ms=wall_ms / timed_decode_steps,
                    gpu_per_step_ms=gpu_ms / timed_decode_steps,
                )

        raise RuntimeError(
            f"generation ended after {full_batch_ticks} full-batch decode ticks before timing completed "
            f"(need {warmup_decode_steps + timed_decode_steps}); the batch may never reach B={batch_size} "
            "at this context, or max_new_tokens/block budget is too small"
        )

    def reset(self) -> None:
        """Reset engine-owned scheduler, I/O, cache, and offloading state."""

        self.offloading_manager.free_all_waiting_cpu_caches()
        self.scheduler.reset()
        self.inputs_and_outputs.reset()
        self.cache.free_all_requests()
        self.cache.reset_peak_usage()
        self.offloading_manager.reset()
        self.last_stats = ContinuousBatchingStats()

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
            raise ValueError("max_new_tokens must be positive")
        empty_prompt_indices = [idx for idx, prompt_ids in enumerate(input_ids) if not prompt_ids]
        if empty_prompt_indices:
            raise ValueError(f"input_ids contains empty prompts at indices {empty_prompt_indices}")
