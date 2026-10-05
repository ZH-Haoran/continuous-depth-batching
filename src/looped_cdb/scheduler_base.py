# Copyright 2025 The HuggingFace Inc. team
# SPDX-License-Identifier: Apache-2.0
# Modified for looped-model serving and continuous depth batching.
# See THIRD_PARTY_NOTICES.md and LICENSES/Apache-2.0.txt.

"""Serving-scheduler machinery shared by continuous batching and continuous depth batching.

Both engines phase-separate their work: a batch is either all-prefill or all-decode, never mixed. That
makes prompt admission a policy decision taken once per scheduler tick, and the two engines must take it
the same way or a throughput comparison between them measures the admission policy rather than the
feature under test. :meth:`BaseServingScheduler.admit_prefill` is that decision, and it lives here so it
has exactly one definition.

The base also owns the request bookkeeping the two schedulers implement identically: block allocation,
the waiting queue, the reserve-mode admission prefix, and preemption-victim selection. What stays in the
subclasses is the shape of a scheduled batch, which differs: continuous batching returns a flat list of
requests, while continuous depth batching returns one homogeneous bucket of prefill, recurrent, or coda
work.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from itertools import islice
from typing import TYPE_CHECKING, Any, Protocol

from looped_cdb.benchmarks import nvtx
from looped_cdb.request_status import RequestStatus
from looped_cdb.utils import (
    DEFAULT_MAX_NUM_SEQS,
    KVPressureMode,
    reservation_admissible_prefix,
    reserved_peak_blocks,
    resolve_min_free_slots,
)

logger = logging.getLogger("ContinuousBatchingLogger")

# NVTX range over the steady-state window, so a Nsight trace clips to the window the row reports.
STEADY_NVTX_LABEL = "benchmark.steady"


if TYPE_CHECKING:
    from collections.abc import Sequence


def resolve_max_num_seqs(max_num_seqs: int, max_num_batched_tokens: int) -> int:
    """Clamp the resident cap to the batch buffers a decode launch writes into.

    A decode batch advances one query token per resident request into buffers sized by
    ``max_num_batched_tokens``, so the prefill budget is also the ceiling on the decode batch. The
    budget is only known once the cache has sized itself to the device, so engines resolve the cap
    here and write the result back onto their config: the decode padding cap, the warmup shapes and
    the benchmark row then all read the batch that actually runs.
    """

    if max_num_seqs <= max_num_batched_tokens:
        return max_num_seqs
    logger.warning(
        f"max_num_seqs={max_num_seqs} exceeds max_num_batched_tokens={max_num_batched_tokens}, "
        f"which sizes the batch buffers a decode launch writes into; the decode batch runs at "
        f"{max_num_batched_tokens}."
    )
    return max_num_batched_tokens


class PagedCache(Protocol):
    """The paged-KV surface a scheduler needs to place requests."""

    num_blocks: int
    block_size: int
    max_blocks_per_request: int
    max_num_batched_tokens: int

    def get_num_free_blocks(self) -> int: ...

    def allocate_blocks(self, n_blocks: int, request_id: str, current_blocks: int) -> int | None: ...

    def free_blocks(self, request_id: str) -> None: ...


class SchedulableRequest(Protocol):
    """The request surface a scheduler reads while placing work."""

    request_id: str
    initial_tokens: list[int]
    max_new_tokens: int | None
    tokens_to_process: list[int]
    remaining_prefill_tokens: list[int]
    allocated_blocks: int
    position_offset: int
    is_cpu_offloaded: bool

    @property
    def status(self) -> RequestStatus: ...

    def current_len(self) -> int: ...

    def total_generated_len(self) -> int: ...


class BaseServingScheduler[RequestT: SchedulableRequest]:
    """Admission, block allocation, and waiting-queue bookkeeping shared by both serving schedulers.

    Subclasses own the scheduling entry point and the batch shape. They must call ``super().__init__``,
    assign their own fields, and only then call :meth:`reset` - the base deliberately does not call it,
    because a subclass ``reset`` may touch fields the subclass has not assigned yet.
    """

    def __init__(
        self,
        cache: PagedCache,
        *,
        safety_margin: float = 0.2,
        kv_pressure_mode: KVPressureMode = "recompute",
        max_model_len: int | None = None,
        max_num_seqs: int = DEFAULT_MAX_NUM_SEQS,
        min_free_slots: int | None = None,
    ) -> None:
        self.cache = cache
        if max_num_seqs <= 0:
            raise ValueError(f"max_num_seqs must be positive, got {max_num_seqs}")
        # Engines resolve this against the sized cache before building the scheduler, so the call is a
        # no-op there; it is what bounds a scheduler constructed directly against a cache.
        self.max_num_seqs = resolve_max_num_seqs(max_num_seqs, cache.max_num_batched_tokens)
        # Reserve admission is exact - an admitted set provably never over-commits the pool - so the
        # free-block safety margin (a heuristic that stops admitting new prefill near capacity) would
        # only throttle reserve below what it can safely run, biasing the reserve-vs-preemption
        # comparison against reserve. It is disabled under reserve; the other modes keep it.
        self.safety_margin = 0.0 if kv_pressure_mode == "reserve" else safety_margin
        self.kv_pressure_mode = kv_pressure_mode
        self.max_model_len = max_model_len
        self.max_decode_fast_path_length = self.cache.max_blocks_per_request * self.cache.block_size
        self.max_num_batched_tokens = self.cache.max_num_batched_tokens
        # Open resident slots required before a prefill launch.
        if min_free_slots is not None and min_free_slots <= 0:
            raise ValueError(f"min_free_slots must be positive, got {min_free_slots}")
        if min_free_slots is not None and min_free_slots > self.max_num_seqs:
            logger.warning(
                f"min_free_slots={min_free_slots} exceeds the resident cap of {self.max_num_seqs}; prefill "
                f"admission waits for {self.max_num_seqs} open slots, an empty resident set."
            )
        self.min_free_slots = resolve_min_free_slots(min_free_slots, self.max_num_seqs)
        # Completions before the steady-state window opens (see :meth:`steady_state_summary`); ``None``
        # leaves the window off. A measurement setting, so the benchmark runner sets it.
        self.steady_warmup_requests: int | None = None
        if kv_pressure_mode == "reserve":
            self._validate_reserve_capacity()

    # ----------------------------------------------------------------- admission policy

    def has_prefill_headroom(self) -> bool:
        """Whether the KV cache has enough free blocks to admit prompt prefill."""

        return self.cache.get_num_free_blocks() > self.safety_margin * self.cache.num_blocks

    def has_decode_work(self) -> bool:
        """Whether any admitted request is decoding.

        Read from ``active_requests`` rather than from any queue. In the depth engine a ``DECODING``
        request holds its single token in the depth queue, the coda queue, the engine's in-flight coda
        results, the batch currently executing, or - when its next token could not allocate a KV block -
        nowhere at all, stalled in the engine awaiting a free block. That last case is exactly the one
        where prefill must be withheld, and it is invisible to the queues.
        """

        return any(state.status == RequestStatus.DECODING for state in self.active_requests.values())

    def admit_prefill(self, has_decode_work: bool) -> bool:
        """Whether a prompt-prefill batch may run this tick.

        Prefill is admitted only while the KV cache holds free blocks above the safety margin, so
        admitting new work never starves the requests already decoding: below the margin the decoders
        run instead and free blocks. With no decode work at all the margin is ignored, otherwise a cache
        held below the margin by nothing but its own fragmentation could never admit the request that
        would drain it.

        Every serving loop routes prompt admission through this one predicate. The loops differ in what
        they run when it returns ``False`` - a decode batch, a recurrent batch, or a coda batch - but not
        in when they stop admitting, so a comparison between them does not measure admission policy.

        Admission has a second half, :meth:`admit_prefill_batch`, which decides whether the prompts this
        one lets through launch this tick or wait for more open slots.
        """

        return not has_decode_work or self.has_prefill_headroom()

    def admit_prefill_batch(self, candidates: list[RequestT]) -> bool:
        """Whether the prefill bucket runs this tick: once ``min_free_slots`` resident slots stand open.

        Sparse arrivals and cache pressure can leave more slots open.
        An in-progress chunked prefill runs regardless because its KV is already allocated.
        """

        if not candidates:
            return False
        if any(state.status == RequestStatus.PREFILLING for state in candidates):
            return True
        return self.free_residency_slots() >= self.min_free_slots

    def free_residency_slots(self) -> int:
        """Resident slots the cap leaves open, which is how far the admitted set has fallen below it."""

        return max(0, self.max_num_seqs - len(self.active_requests))

    def _validate_reserve_capacity(self) -> None:
        """Ensure a single worst-case request can fit, so reserve admission cannot starve it forever."""

        if self.max_model_len is None:
            raise ValueError("kv_pressure_mode='reserve' requires max_model_len to bound each request's KV footprint")
        peak = reserved_peak_blocks(0, self.max_model_len, self.max_model_len, self.cache.block_size)
        if peak > self.cache.num_blocks:
            raise ValueError(
                f"kv_pressure_mode='reserve' cannot serve a max-length request: its worst case needs {peak} blocks "
                f"but the cache has only {self.cache.num_blocks} (increase num_blocks or reduce max_model_len)."
            )

    # ----------------------------------------------------------------- request bookkeeping

    def reset(self) -> None:
        """Reset scheduler state for a new generation loop."""

        self.active_requests: dict[str, RequestT] = {}
        self.waiting_requests: dict[str, RequestT] = {}
        self.waiting_requests_order: deque[str] = deque()
        # Set after a preemption to stop admitting new waiting requests until an active request
        # finishes, so the batch drains and frees blocks instead of immediately re-admitting and
        # thrashing. Reset to False by the engine when a request completes.
        self.block_new_requests = False
        # Size of the admitted set, sampled per scheduler tick, so a run can show whether
        # ``max_num_seqs`` bound it or merely sat above it.
        self.max_resident_requests = 0
        self.resident_samples = 0
        self.resident_sum = 0
        self.record_queue_samples = False
        self.waiting_queue_samples: list[tuple[float, int]] = []
        self.kv_usage_samples: list[tuple[float, int]] = []
        # Requests that finished generating, and the tokens they generated; preemption does not count.
        self.completed_requests = 0
        self.completed_generated_tokens = 0
        # Steady-state window: opened on the tick the warm-up completion count is reached, closed on the
        # first later tick whose waiting queue is empty. Each end records the clock and a progress
        # snapshot; the ticks between them carry the window's residency mean.
        self._steady_start_time: float | None = None
        self._steady_end_time: float | None = None
        self._steady_start_progress = (0, 0)
        self._steady_end_progress = (0, 0)
        self._steady_ticks = 0
        self._steady_resident_sum = 0
        self._steady_nvtx_handle: int | None = None

    def record_completion(self, state: RequestT) -> None:
        """Count a request that finished generating, with what it generated.

        Called where the engine retires a finished request, not from ``finish_request``, which preemption
        also routes through.
        """

        self.completed_requests += 1
        self.completed_generated_tokens += state.total_generated_len()

    def record_resident_sample(self) -> None:
        """Sample the admitted set, once per scheduler tick, before this tick admits anything.

        Residency, not the achieved batch: a resident request can sit anywhere in the pipeline, and in
        the depth engine one parked in the coda queue or stalled on a block contributes nothing to a
        recurrent launch. ``mean_recurrent_batch_size`` and ``mean_coda_batch_size`` are the achieved
        batches. What this answers is whether admission held the resident set near its cap, which is
        what ``min_free_slots`` exists to do.

        Every loop samples at the same point in its tick, so the three are comparable. They are still
        tick means rather than time means, and a depth tick is one stage launch while a full-depth tick
        is one token, so the two count over different clocks.
        """

        resident = len(self.active_requests)
        self.max_resident_requests = max(self.max_resident_requests, resident)
        self.resident_samples += 1
        self.resident_sum += resident
        if self.record_queue_samples:
            self.sample_waiting_queue()
        self._advance_steady_window(resident)

    def sample_waiting_queue(self) -> None:
        """Capture waiting and KV occupancy on the request-latency clock."""

        if self.record_queue_samples:
            stamp = time.perf_counter()
            self.waiting_queue_samples.append((stamp, len(self.waiting_requests)))
            self.kv_usage_samples.append((stamp, self.cache.num_blocks - self.cache.get_num_free_blocks()))

    @property
    def mean_resident_requests(self) -> float:
        """Mean residency over the sampled ticks; 0.0 for a scheduler that never ran."""

        return self.resident_sum / self.resident_samples if self.resident_samples else 0.0

    # ----------------------------------------------------------------- steady-state window

    @property
    def steady_window_open(self) -> bool:
        """Whether the measured steady-state window is open."""

        return self._steady_start_time is not None and self._steady_end_time is None

    def _advance_steady_window(self, resident: int) -> None:
        """Open or close the window at this tick's sample, and count the tick if it lies inside.

        The opening tick counts; the closing tick, the first after the last admission, belongs to the drain.
        A warm-up count first reached with nothing left waiting never opens a window: the run is already
        draining.
        """

        if self.steady_warmup_requests is None or self._steady_end_time is not None:
            return
        if self._steady_start_time is None:
            if self.completed_requests < self.steady_warmup_requests or not self.waiting_requests:
                return
            self._steady_start_time = time.perf_counter()
            self._steady_start_progress = self._progress_snapshot()
            # A start/end range: this tick runs inside a stage range, whose pop would close a push/pop range.
            self._steady_nvtx_handle = nvtx.range_start(STEADY_NVTX_LABEL, registered=True)
        elif not self.waiting_requests:
            self.close_steady_window()
            return
        self._steady_ticks += 1
        self._steady_resident_sum += resident

    def _progress_snapshot(self) -> tuple[int, int]:
        """Requests completed and tokens generated so far, including tokens held by resident and
        preempted (re-queued) requests, so two snapshots differ by exactly what was generated between them."""

        generated = self.completed_generated_tokens
        generated += sum(state.total_generated_len() for state in self.active_requests.values())
        generated += sum(state.total_generated_len() for state in self.waiting_requests.values())
        return self.completed_requests, generated

    def close_steady_window(self) -> None:
        """Close the window now if it is open: on the first tick after the last admission, or when the
        engine ends a generation whose queue never emptied after the window opened."""

        if self._steady_start_time is None or self._steady_end_time is not None:
            return
        self._steady_end_time = time.perf_counter()
        self._steady_end_progress = self._progress_snapshot()
        if self._steady_nvtx_handle is not None:
            nvtx.range_end(self._steady_nvtx_handle)
            self._steady_nvtx_handle = None

    def steady_state_summary(self) -> dict[str, Any] | None:
        """Throughput inside the steady-state window, or ``None`` when the run held no such window.

        A closed-loop run fills the resident set to its cap, holds it there while admission replaces each
        retiring request, and drains once the waiting queue is empty. Only the middle phase runs at the
        operating point the cap names; whole-run throughput folds in the other two. The window opens on
        the tick ``steady_warmup_requests`` requests have completed, after the first cohort's correlated
        retirement, and closes on the first tick after the last waiting prompt was admitted. Both marks are
        scheduler events. Throughput is what completed and was generated between the two snapshots over
        the wall-clock between them; ``mean_resident_requests`` is over the window's ticks. ``None`` when
        the window never opened (the warm-up count was never reached, or reached only while draining),
        never closed, or saw no completion.
        """

        if self._steady_start_time is None or self._steady_end_time is None:
            return None
        window_s = self._steady_end_time - self._steady_start_time
        completed = self._steady_end_progress[0] - self._steady_start_progress[0]
        generated = self._steady_end_progress[1] - self._steady_start_progress[1]
        if window_s <= 0.0 or completed <= 0:
            return None
        return {
            "warmup_requests": self.steady_warmup_requests,
            "window_s": window_s,
            "ticks": self._steady_ticks,
            "completed_requests": completed,
            "generated_tokens": generated,
            "requests_per_second": completed / window_s,
            "generated_tokens_per_second": generated / window_s,
            "mean_resident_requests": self._steady_resident_sum / self._steady_ticks if self._steady_ticks else 0.0,
        }

    def add_waiting_request(self, state: RequestT) -> None:
        """Adds a request to the waiting list."""

        self.waiting_requests[state.request_id] = state
        self.waiting_requests_order.append(state.request_id)

    def pop_request_to_evict(self) -> tuple[str, RequestT]:
        """Remove and return the active request whose KV is cheapest to give up.

        The shortest KV state minimizes recomputation and transfer costs.
        Ties select the most recently added request.
        """

        request_id = min(reversed(self.active_requests), key=lambda req_id: self.active_requests[req_id].current_len())
        return request_id, self.active_requests.pop(request_id)

    def _allocate_blocks_if_needed(self, state: RequestT, len_next_tokens: int) -> bool:
        """Allocate additional cache blocks for a request if the current allocation is insufficient."""

        current_len = state.current_len()
        occupancy = state.allocated_blocks * self.cache.block_size - current_len
        if occupancy < len_next_tokens:
            # Allocate exactly enough blocks to hold the tokens present after this step (ceil), with no
            # spare headroom block. This diverges from the HF-CB formula
            # ``((len_next - occupancy + 1) // block_size) + 1``, which rounds up one whole block past
            # what the tokens need; the exact form matches vLLM (v1 allocates ``ceil(tokens/block_size)``
            # on demand and holds no spare block) and makes ``reserved_peak_blocks`` tight.
            blocks_needed = -(-(current_len + len_next_tokens) // self.cache.block_size) - state.allocated_blocks
            allocated = self.cache.allocate_blocks(blocks_needed, state.request_id, state.allocated_blocks)
            if allocated is None:
                return False
            state.allocated_blocks += allocated
        return True

    def admit_offloaded_restores(self) -> list[RequestT]:
        """Move offloaded waiting requests back into the active set and re-allocate their blocks.

        Each restored request keeps the status it was preempted at (decoding, or a chunked prefill in
        progress) and has its blocks re-allocated for its restored length; the engine copies its KV back
        before compute, so a restored request never re-enters as a prefill. Offloaded requests are
        cheaper to resume than fresh prompts are to prefill, so they are admitted first. Blocked while
        draining after a preemption (``block_new_requests``), bounded by the open resident slots like
        every other admission, and stops at the first request whose blocks no longer fit. Returns the
        admitted requests in admission order.
        """

        # Only the offload mode parks requests on the CPU, so no other mode scans the queue for them.
        if self.block_new_requests or self.kv_pressure_mode != "offload":
            return []
        admitted: list[RequestT] = []
        for req_id in list(self.waiting_requests_order):
            state = self.waiting_requests[req_id]
            if not state.is_cpu_offloaded:
                continue
            if self.free_residency_slots() == 0 or not self._allocate_blocks_if_needed(state, 1):
                break
            self.active_requests[req_id] = state
            del self.waiting_requests[req_id]
            self.waiting_requests_order.remove(req_id)
            admitted.append(state)
        return admitted

    # ----------------------------------------------------------------- prefill candidates

    def _prefill_admits(self, state: RequestT) -> bool:  # noqa: ARG002
        """Whether an in-progress chunked prefill may continue this tick.

        Always true here. Continuous depth batching overrides it to hold back a request whose previous
        chunk is still mid-forward.
        """

        return True

    def get_prefill_bucket(self) -> list[RequestT]:
        """The prompts eligible to prefill this tick, before the launch gate.

        In-progress chunked prefills first, then waiting prompts up to the open resident slots. While
        ``block_new_requests`` is set (just after a preemption), no waiting request is admitted and only
        in-progress prefills continue, so the batch drains and frees blocks.

        Kept separate from :meth:`get_prefill_candidates` so a caller diagnosing a stalled scheduler can
        see the prompt that cannot be placed, rather than an empty list that the launch gate produced.
        """

        prefilling = [
            state
            for state in self.active_requests.values()
            if state.status == RequestStatus.PREFILLING and self._prefill_admits(state)
        ]
        if self.block_new_requests:
            return prefilling
        admissible = self._get_waiting_candidates(limit=self.free_residency_slots())
        if self.kv_pressure_mode == "reserve":
            # Admit only the prefix of waiting prompts whose worst-case peak KV fits alongside the
            # running set's reservations, so an admitted request can always allocate as it decodes.
            admissible = reservation_admissible_prefix(
                self.active_requests.values(),
                admissible,
                num_blocks=self.cache.num_blocks,
                block_size=self.cache.block_size,
                max_model_len=self.max_model_len,
            )
        return prefilling + admissible

    def get_prefill_candidates(self) -> list[RequestT]:
        """The prefill bucket, or nothing while admission holds it for more open slots.

        ``max_num_seqs`` caps the resident set. It bounds a count, not a KV footprint, so it is a thrash
        guard rather than a memory guarantee: it stops the engine prefilling prompts it has no near-term
        intent to advance, only to preempt them later and recompute the same prefill. The memory
        guarantee is ``safety_margin`` plus the preemption policy, or exact reserve admission.
        """

        candidates = self.get_prefill_bucket()
        if not self.admit_prefill_batch(candidates):
            return []
        return candidates

    def _get_waiting_candidates(self, limit: int | None = None) -> list[RequestT]:
        """Return the first ``limit`` fresh (never-offloaded) waiting prompts in arrival order.

        Offloaded requests are not admitted here: they resume through the decode path, so a restored
        request never re-enters as a prefill. The scan stops once ``limit`` prompts are found, so a tick
        that can admit few or none touches that little of the queue.
        """

        fresh = (
            self.waiting_requests[req_id]
            for req_id in self.waiting_requests_order
            if not self.waiting_requests[req_id].is_cpu_offloaded
        )
        return list(islice(fresh, limit))

    def _cleanup_waiting_queue(self, request_ids_to_remove_from_waiting: Sequence[str] | set[str]) -> None:
        """Removes processed requests from the waiting queue order.

        Admitted prompts are the head of the order, so they pop off the front; only an id further back
        (behind an offloaded entry) costs a rebuild.
        """

        remaining = set(request_ids_to_remove_from_waiting)
        order = self.waiting_requests_order
        while remaining and order and order[0] in remaining:
            remaining.discard(order.popleft())
        if remaining:
            self.waiting_requests_order = deque(req_id for req_id in order if req_id not in remaining)
