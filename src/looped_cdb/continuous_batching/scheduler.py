# Copyright 2025 The HuggingFace Inc. team
# SPDX-License-Identifier: Apache-2.0
# Modified for looped-model serving and continuous depth batching.
# See THIRD_PARTY_NOTICES.md and LICENSES/Apache-2.0.txt.

"""Scheduling requests for continuous batching.

This module mirrors Hugging Face's continuous batching scheduler while keeping
the implementation focused on FIFO scheduling, chunked prefill,
full attention, and explicit cache-budget accounting.

Reference:
https://github.com/huggingface/transformers/blob/main/src/transformers/generation/continuous_batching/scheduler.py
"""

from dataclasses import dataclass, field

from looped_cdb.scheduler_base import BaseServingScheduler
from looped_cdb.utils import DEFAULT_MAX_NUM_SEQS, KVPressureMode

from .cache import PagedAttentionCache
from .requests import FutureRequestState, RequestState, RequestStatus, logger


@dataclass
class ScheduledBatch:
    """One scheduler decision: the requests to launch and the shape the runner needs for them.

    ``requests`` is empty when nothing could be placed, and ``cache_exhausted`` separates the two
    reasons that can happen. Set, an allocation actually failed and no work can run until blocks are
    freed. Unset, the bucket simply placed nothing this tick, for instance a prompt whose KV read
    exceeds the cache budget, which is transient while another batch is still in flight. The depth
    engine returns its decision as an object too, in
    :class:`~looped_cdb.continuous_depth_batching.scheduler.ScheduledCDBBatch`, though it signals
    the hard case by returning nothing at all and leaving the engine to interpret it.
    """

    requests: list[FutureRequestState] = field(default_factory=list)
    use_decode_fast_path: bool = False
    num_q_tokens: int = 0
    max_kv_read: int = 0
    cache_exhausted: bool = False


class FIFOScheduler(BaseServingScheduler[RequestState]):
    """FIFO scheduler with phase-separated prefill and decode batches.

    Eligible prefill runs before decode while the cache has sufficient headroom.
    """

    def __init__(
        self,
        cache: PagedAttentionCache,
        safety_margin: float = 0.2,
        kv_pressure_mode: KVPressureMode = "recompute",
        max_model_len: int | None = None,
        max_num_seqs: int = DEFAULT_MAX_NUM_SEQS,
        min_free_slots: int | None = None,
    ) -> None:
        """Initialize the FIFO scheduler.

        The safety margin is the percentage of free blocks under which we stop scheduling new prefill requests, so
        safety_margin = 0.1 means that when there is less than 10% of free blocks, or equivalently when more than 90%
        of blocks are already allocated, we stop scheduling new prefill requests.

        ``kv_pressure_mode`` selects the cache-pressure policy. In ``"reserve"`` mode admission is
        gated on each request's worst-case peak KV fitting alongside the running reservations (see
        :func:`reservation_admissible_prefix`), which requires ``max_model_len`` to bound generation.
        The ``"recompute"`` and ``"offload"`` modes admit greedily and rely on preemption instead.
        """

        super().__init__(
            cache,
            safety_margin=safety_margin,
            kv_pressure_mode=kv_pressure_mode,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            min_free_slots=min_free_slots,
        )
        self.reset()

    def schedule_batch(
        self,
        token_budget: int,
        cache_budget: int,
    ) -> ScheduledBatch:
        """Schedule requests for the next batch based on available token and cache budgets.

        The token budget is the maximum number of query tokens that can be processed in a prefill
        batch, and the cache budget is the maximum number of KV cache entries that can be read in a
        variable-length batch. A decode-only batch is bounded instead by ``max_num_seqs``, since it
        advances one query token per resident request. The decision comes back as a
        :class:`ScheduledBatch`.
        """

        # Batches contain only prefill or decode work.
        # Prefill runs while cache headroom and residency admission permit it.
        # Bring any offloaded requests back into the active set first, so they resume as decode work
        # (their KV is restored before compute) rather than re-entering through the prefill path.
        self.admit_offloaded_restores()
        self.record_resident_sample()
        # A request whose in-flight token already meets its length limit is guaranteed to finish when
        # that token is consumed, so scheduling it again would only compute a wasted extra token.
        decode_candidates = [
            state
            for state in self.active_requests.values()
            if state.status == RequestStatus.DECODING and not state.will_finish_on_pending_token()
        ]
        prefill_candidates = self.get_prefill_candidates()

        if prefill_candidates and self.admit_prefill(bool(decode_candidates)):
            request_ids_to_remove_from_waiting: set[str] = set()
            scheduled_requests, one_allocation_failed, decode_fast_path, num_q_tokens, max_kv_read = (
                self._process_candidates(
                    prefill_candidates,
                    token_budget,
                    cache_budget,
                    request_ids_to_remove_from_waiting,
                    safety_margin=self.safety_margin,
                )
            )
            self._cleanup_waiting_queue(request_ids_to_remove_from_waiting)
            if scheduled_requests:
                return ScheduledBatch(scheduled_requests, decode_fast_path, num_q_tokens, max_kv_read)
            if not decode_candidates:
                # Prefill was the only option and nothing fit: a hard cache-full signal only when
                # an allocation actually failed, otherwise an empty batch (e.g. cache-budget reject).
                return ScheduledBatch(use_decode_fast_path=decode_fast_path, cache_exhausted=one_allocation_failed)

        if not decode_candidates:
            return ScheduledBatch(use_decode_fast_path=True)

        scheduled_requests, one_allocation_failed, decode_fast_path, num_q_tokens, max_kv_read = (
            self._process_candidates(
                decode_candidates,
                self.max_num_seqs,
                cache_budget,
                set(),
                safety_margin=0.0,
            )
        )
        if not scheduled_requests and one_allocation_failed:
            return ScheduledBatch(use_decode_fast_path=decode_fast_path, cache_exhausted=True)

        return ScheduledBatch(scheduled_requests, decode_fast_path, num_q_tokens, max_kv_read)

    def has_pending_requests(self) -> bool:
        """Checks if there are requests ready to be processed."""

        return bool(len(self.active_requests) or len(self.waiting_requests))

    def finish_request(self, request_id: str) -> None:
        """Complete processing of a request and free its allocated cache blocks."""

        self.cache.free_blocks(request_id)
        self.active_requests.pop(request_id, None)

    def _infer_request_tokens(self, state: RequestState) -> list[int]:
        """Prepare the tokens that would be present in the batch if the request is scheduled."""

        if state.status == RequestStatus.DECODING:
            return state.tokens_to_process
        return state.remaining_prefill_tokens

    def _schedule_request(
        self,
        state: RequestState,
        request_tokens: list[int],
        token_budget: int,
        request_ids_to_remove_from_waiting: set[str],
    ) -> None:
        """Schedule a request for the current batch, updating the request status according to the budget left."""

        # Case: we can process the entire prompt/remainder
        if len(request_tokens) <= token_budget:
            if state.status == RequestStatus.PENDING:
                self.active_requests[state.request_id] = state
                request_ids_to_remove_from_waiting.add(state.request_id)
            if state.status <= RequestStatus.PREFILLING:
                state.tokens_to_process = state.remaining_prefill_tokens
                state.remaining_prefill_tokens = []
                # Although prefill will only be done after the batch being scheduled now, we set the status to DECODING
                # to stay coherent when using asynchronous batching.
                state.status = RequestStatus.DECODING

        # Otherwise: we need to split the request
        else:
            if state.status == RequestStatus.PENDING:
                self.active_requests[state.request_id] = state
                state.status = RequestStatus.PREFILLING
                request_ids_to_remove_from_waiting.add(state.request_id)
            state.remaining_prefill_tokens = request_tokens[token_budget:]
            state.tokens_to_process = request_tokens[:token_budget]

    def _process_candidates(
        self,
        candidates: list[RequestState],
        token_budget: int,
        cache_budget: int,
        request_ids_to_remove_from_waiting: set[str],
        safety_margin: float = 0.0,
    ) -> tuple[list[FutureRequestState], bool, bool, int, int]:
        """Schedule candidate requests for the current batch."""

        scheduled_requests = []
        one_allocation_failed = False
        # The batch takes the decode fast path only if every scheduled request is a single-token decode.
        decode_fast_path = True
        safety_margins = safety_margin * self.cache.num_blocks
        original_token_budget, original_cache_budget = token_budget, cache_budget

        for state in candidates:
            num_free_blocks = self.cache.get_num_free_blocks()
            # If we are out the safety margin, we only accept decoding requests or the first prefill request.
            outside_safety_margin = num_free_blocks < safety_margins
            if outside_safety_margin and scheduled_requests and state.status != RequestStatus.DECODING:
                logger.info(
                    f"Outside safety margin, breaking out of scheduling loop. {num_free_blocks = } {safety_margins = }"
                )
                break

            request_tokens = self._infer_request_tokens(state)
            request_len = min(len(request_tokens), token_budget)
            if request_len == 0:
                continue

            # Decode batches use the block table path and do not consume the varlen read-index budget. A request that
            # would change the batch from decode to varlen is rejected if the cache budget is too low.
            is_decode_eligible = request_len == 1 and state.position_offset < self.max_decode_fast_path_length
            read_cache_needed = state.current_len()
            if not (decode_fast_path and is_decode_eligible) and cache_budget < read_cache_needed:
                continue

            allocation_successful = self._allocate_blocks_if_needed(state, request_len)
            if not allocation_successful:
                one_allocation_failed = True
                # If we reached a waiting request and the cache is full, all subsequent waiting requests will need
                # allocation as well, so we can safely break out of the scheduling loop.
                if num_free_blocks == 0 and state.request_id in self.waiting_requests:
                    logger.info(f"Breaking mid-loop for request {state.request_id} because the cache is full")
                    break
                continue

            self._schedule_request(state, request_tokens, token_budget, request_ids_to_remove_from_waiting)
            request_len = len(state.tokens_to_process)

            # The decode fast path is only used if every scheduled request processes a single token and its length fits
            # in the block table allocated for one request.
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
