"""Unified scheduler for continuous depth batching."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from looped_cdb.scheduler_base import BaseServingScheduler
from looped_cdb.utils import DEFAULT_MAX_NUM_SEQS, KVPressureMode

from .requests import DepthWorkItem, FutureRequestState, RequestState, RequestStatus, logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .cache import PagedAttentionCache


ScheduledKind = Literal["prefill", "coda", "recurrent"]


@dataclass
class ScheduledCDBBatch:
    """One homogeneous CDB scheduler tick."""

    kind: ScheduledKind
    requests: list[FutureRequestState] | None = None
    depth_items: list[DepthWorkItem] | None = None
    use_decode_fast_path: bool = False
    num_q_tokens: int = 0
    max_kv_read: int = 0


class CDBScheduler(BaseServingScheduler[RequestState]):
    """Unified scheduler for prompt prefill, recurrent depth work, and coda work.

    Decode prelude is intentionally not a separate bucket: when coda or prompt
    prefill produces a token, the engine runs its prelude immediately and enqueues the
    resulting recurrent work item.
    """

    def __init__(
        self,
        cache: PagedAttentionCache,
        max_recurrent_steps: int,
        safety_margin: float = 0.2,
        kv_pressure_mode: KVPressureMode = "recompute",
        max_model_len: int | None = None,
        max_num_seqs: int = DEFAULT_MAX_NUM_SEQS,
        min_free_slots: int | None = None,
        min_coda_batch_size: int = 1,
    ) -> None:
        if max_recurrent_steps <= 0:
            raise ValueError(f"max_recurrent_steps must be positive, got {max_recurrent_steps}")
        if min_coda_batch_size <= 0:
            raise ValueError(f"min_coda_batch_size must be positive, got {min_coda_batch_size}")
        super().__init__(
            cache,
            safety_margin=safety_margin,
            kv_pressure_mode=kv_pressure_mode,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            min_free_slots=min_free_slots,
        )
        self.max_recurrent_steps = max_recurrent_steps
        # Resolved against the clamped cap, not the requested one: the config rejects a minimum above
        # the resident cap it was given, but a cap the token budget then clamps can leave the minimum
        # above the resident set the coda queue is fed from. Clamping is schedule-preserving, since a
        # minimum at the cap already runs codas only when nothing else can.
        if min_coda_batch_size > self.max_num_seqs:
            logger.warning(
                f"min_coda_batch_size={min_coda_batch_size} exceeds the resident cap of "
                f"{self.max_num_seqs}, which the coda queue is fed from; it runs at "
                f"{self.max_num_seqs}, where codas launch once no other bucket can run."
            )
        self.min_coda_batch_size = min(min_coda_batch_size, self.max_num_seqs)
        # ``reset`` is called last, after every field it touches exists. The base class deliberately
        # does not call it, since dispatching to this override from the base constructor would run it
        # before the assignments above.
        self.reset()

    def reset(self) -> None:
        """Reset all request and depth queues."""

        super().reset()
        self.ready_queue: deque[DepthWorkItem] = deque()
        self.coda_queue: deque[DepthWorkItem] = deque()
        self.mixed_batches_scheduled = 0
        self.refill_events = 0
        self.max_depth_queue_size = 0
        self.in_flight_request_ids: set[str] = set()
        self.max_in_flight_requests = 0
        self.max_live_requests = 0

    def has_pending_work(self) -> bool:
        """Return whether any serving, depth, or coda work remains."""

        return bool(self.coda_queue or self.ready_queue or self.active_requests or self.waiting_requests)

    def schedule_next(
        self,
        token_budget: int,
        cache_budget: int,
        *,
        allow_coda: bool = True,
    ) -> ScheduledCDBBatch | None:
        """Choose the next homogeneous CDB bucket.

        Ready coda work has priority once it reaches ``min_coda_batch_size``.
        Prefill follows when admission permits, then recurrent work, then a final undersized coda flush.
        ``None`` means that no scheduler-visible bucket can run.
        """

        self.record_resident_sample()
        if allow_coda and len(self.coda_queue) >= self.min_coda_batch_size:
            coda_batch = self._schedule_coda_batch()
            if coda_batch:
                return ScheduledCDBBatch(kind="coda", depth_items=coda_batch)

        # A request whose token sits in the engine's in-flight coda results is still active and
        # DECODING, so suppressing the coda bucket via ``allow_coda`` never weakens this gate.
        if self.admit_prefill(self.has_decode_work()):
            prefill_batch = self.schedule_prefill_batch(token_budget=token_budget, cache_budget=cache_budget)
            if prefill_batch is not None:
                return prefill_batch

        recurrent_batch = self._schedule_recurrent_batch()
        if recurrent_batch:
            return ScheduledCDBBatch(kind="recurrent", depth_items=recurrent_batch)

        # Nothing else can run: flush a held-back under-size coda rather than stalling on it. This is
        # what keeps ``min_coda_batch_size`` deadlock-free at the drain tail and through exit-starved
        # phases, and it is a no-op at the default of 1, where any pending coda already ran above.
        if allow_coda:
            coda_batch = self._schedule_coda_batch()
            if coda_batch:
                return ScheduledCDBBatch(kind="coda", depth_items=coda_batch)

        return None

    def enqueue_depth(self, item: DepthWorkItem, *, front: bool = False) -> None:
        """Queue one recurrent-step work item, at the tail by default.

        ``front`` returns a request that already holds a launch slot to the head, so the next recurrent
        batch picks it up again instead of rotating it behind every other resident request. See
        :meth:`enqueue_depth_front`.
        """

        if not 0 <= item.recurrent_step < self.max_recurrent_steps:
            raise ValueError(f"recurrent_step must be in [0, {self.max_recurrent_steps}), got {item.recurrent_step}")
        self._mark_request_in_flight(item.state.request_id)
        # A refill is a fresh token (step 0) joining a depth queue that already holds other work - the
        # interleaving this scheduler exists to create. Survivors returning mid-depth are not refills,
        # and neither is a token entering an empty queue, which any scheduler would run alone.
        if item.recurrent_step == 0 and self.ready_queue:
            self.refill_events += 1
        if front:
            self.ready_queue.appendleft(item)
        else:
            self.ready_queue.append(item)
        self.max_depth_queue_size = max(self.max_depth_queue_size, len(self.ready_queue))

    def enqueue_depth_front(self, items: Sequence[DepthWorkItem]) -> None:
        """Return work items to the head of the queue, preserving their relative order.

        A recurrent batch pops up to ``max_num_seqs`` items off the head, and the queue holds at most
        one item per resident request, so ordinarily the whole queue runs and the order changes
        nothing. It decides which items run first only when a launch cannot carry the queue: a
        depth-indexed KV layout admits one slot per launch, and the items of the other slots wait.
        Head insertion keeps a bounded working set advancing there, and the requests behind it stay
        parked at prompt length.

        This is the discipline the full-depth scheduler gets for free: it walks ``active_requests`` in
        a stable insertion order, so it re-picks the same head of the decode set every tick. The
        no-refill loop takes the head of ``active_requests`` as its cohort for the same reason.
        """

        for item in reversed(items):
            self.enqueue_depth(item, front=True)

    def enqueue_coda(self, item: DepthWorkItem) -> None:
        """Queue one token whose recurrent stage has finished."""

        self._mark_request_in_flight(item.state.request_id)
        self.coda_queue.append(item)

    def finish_request(self, request_id: str) -> None:
        """Complete a request and free its cache blocks."""

        self.cache.free_blocks(request_id)
        self.active_requests.pop(request_id, None)
        self.in_flight_request_ids.discard(request_id)
        if self.waiting_requests.pop(request_id, None) is not None:
            self.waiting_requests_order.remove(request_id)

    def free_unfinished_requests(self) -> None:
        """Free cache blocks for any requests left alive after generation aborts or validation failures."""

        for request_id in list(self.active_requests):
            self.finish_request(request_id)
        for request_id in list(self.waiting_requests):
            self.finish_request(request_id)

    def schedule_prefill_batch(self, token_budget: int, cache_budget: int) -> ScheduledCDBBatch | None:
        """Pack one prompt-prefill batch, or return ``None`` when nothing could be placed.

        Callers must gate this on :meth:`admit_prefill`. ``None`` covers both "no candidate" and
        "a candidate could not allocate": the engine distinguishes them when it has to, since only it
        can decide between preempting a decoder and raising.
        """

        candidates = self.get_prefill_candidates()
        if not candidates:
            return None

        request_ids_to_remove_from_waiting: set[str] = set()
        scheduled_requests, _, decode_fast_path, num_q_tokens, max_kv_read = self._process_prefill_candidates(
            candidates=candidates,
            token_budget=token_budget,
            cache_budget=cache_budget,
            request_ids_to_remove_from_waiting=request_ids_to_remove_from_waiting,
        )
        self._cleanup_waiting_queue(request_ids_to_remove_from_waiting)
        if not scheduled_requests:
            return None

        return ScheduledCDBBatch(
            kind="prefill",
            requests=scheduled_requests,
            use_decode_fast_path=decode_fast_path,
            num_q_tokens=num_q_tokens,
            max_kv_read=max_kv_read,
        )

    def _process_prefill_candidates(
        self,
        candidates: list[RequestState],
        token_budget: int,
        cache_budget: int,
        request_ids_to_remove_from_waiting: set[str],
    ) -> tuple[list[FutureRequestState], bool, bool, int, int]:
        scheduled_requests = []
        one_allocation_failed = False
        # The batch takes the decode fast path only if every scheduled request is a single-token decode.
        decode_fast_path = True
        safety_margin_blocks = self.safety_margin * self.cache.num_blocks
        original_token_budget, original_cache_budget = token_budget, cache_budget

        for state in candidates:
            num_free_blocks = self.cache.get_num_free_blocks()
            if num_free_blocks < safety_margin_blocks and scheduled_requests:
                logger.info(
                    f"Outside safety margin, breaking out of prefill scheduling loop. "
                    f"{num_free_blocks = } {safety_margin_blocks = }"
                )
                break

            request_tokens = state.remaining_prefill_tokens
            request_len = min(len(request_tokens), token_budget)
            if request_len == 0:
                continue

            is_decode_eligible = request_len == 1 and state.position_offset < self.max_decode_fast_path_length
            read_cache_needed = state.current_len()
            if not (decode_fast_path and is_decode_eligible) and cache_budget < read_cache_needed:
                continue

            if not self._allocate_blocks_if_needed(state, request_len):
                one_allocation_failed = True
                if num_free_blocks == 0 and state.request_id in self.waiting_requests:
                    logger.info(f"Breaking mid-loop for request {state.request_id} because the cache is full")
                    break
                continue

            self._schedule_prefill_request(state, request_tokens, request_len, request_ids_to_remove_from_waiting)
            self._mark_request_in_flight(state.request_id)
            request_len = len(state.tokens_to_process)
            decode_fast_path &= request_len == 1 and state.position_offset < self.max_decode_fast_path_length
            token_budget -= request_len
            cache_budget -= read_cache_needed

            has_new_token = not state.remaining_prefill_tokens
            scheduled_requests.append(FutureRequestState(state, has_new_token, query_length=request_len))

            req_id = state.request_id
            was_waiting = self.waiting_requests.pop(req_id, None) is not None
            if was_waiting:
                request_ids_to_remove_from_waiting.add(req_id)

            if token_budget == 0 or (cache_budget <= 0 and not decode_fast_path):
                break

        num_q_tokens = original_token_budget - token_budget
        max_kv_read = original_cache_budget - cache_budget
        return scheduled_requests, one_allocation_failed, decode_fast_path, num_q_tokens, max_kv_read

    def _schedule_prefill_request(
        self,
        state: RequestState,
        request_tokens: list[int],
        request_len: int,
        request_ids_to_remove_from_waiting: set[str],
    ) -> None:
        if len(request_tokens) <= request_len:
            if state.status == RequestStatus.PENDING:
                self.active_requests[state.request_id] = state
                request_ids_to_remove_from_waiting.add(state.request_id)
            if state.status <= RequestStatus.PREFILLING:
                state.tokens_to_process = state.remaining_prefill_tokens
                state.remaining_prefill_tokens = []
                state.status = RequestStatus.DECODING
            return

        if state.status == RequestStatus.PENDING:
            self.active_requests[state.request_id] = state
            state.status = RequestStatus.PREFILLING
            request_ids_to_remove_from_waiting.add(state.request_id)
        state.remaining_prefill_tokens = request_tokens[request_len:]
        state.tokens_to_process = request_tokens[:request_len]

    def _schedule_recurrent_batch(self) -> list[DepthWorkItem]:
        batch_size = min(self.max_num_seqs, len(self.ready_queue))
        if batch_size == 0:
            return []

        first_item = self.ready_queue.popleft()
        batch = [first_item]
        if self.cache.requires_slot_homogeneous_recurrent_batches:
            target_slot = self.cache.kv_slot_for_recurrent_step(first_item.recurrent_step)
            deferred: deque[DepthWorkItem] = deque()
            while self.ready_queue and len(batch) < batch_size:
                item = self.ready_queue.popleft()
                if self.cache.kv_slot_for_recurrent_step(item.recurrent_step) == target_slot:
                    batch.append(item)
                else:
                    deferred.append(item)
            deferred.extend(self.ready_queue)
            self.ready_queue = deferred
        else:
            while self.ready_queue and len(batch) < batch_size:
                batch.append(self.ready_queue.popleft())

        if batch:
            self.mixed_batches_scheduled += 1
        return batch

    def _schedule_coda_batch(self) -> list[DepthWorkItem]:
        return [self.coda_queue.popleft() for _ in range(min(self.max_num_seqs, len(self.coda_queue)))]

    def _prefill_admits(self, state: RequestState) -> bool:
        """Hold back an in-progress chunked prefill whose previous chunk is still mid-forward."""

        return state.request_id not in self.in_flight_request_ids

    def release_in_flight_request(self, request_id: str) -> None:
        """Mark a request as ready to schedule another token."""

        self.in_flight_request_ids.discard(request_id)

    def _mark_request_in_flight(self, request_id: str) -> None:
        self.in_flight_request_ids.add(request_id)
        self.max_in_flight_requests = max(self.max_in_flight_requests, len(self.in_flight_request_ids))
        self.max_live_requests = max(
            self.max_live_requests,
            len(self.active_requests) + len(self.waiting_requests),
        )
