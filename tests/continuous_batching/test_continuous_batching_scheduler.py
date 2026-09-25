import torch
from transformers import PreTrainedConfig

from looped_cdb.continuous_batching.cache import PagedAttentionCache
from looped_cdb.continuous_batching.config import ContinuousBatchingConfig
from looped_cdb.continuous_batching.requests import RequestState, RequestStatus
from looped_cdb.continuous_batching.scheduler import FIFOScheduler


def _config(**overrides: int | str | list[str] | None) -> PreTrainedConfig:
    values = {
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "hidden_size": 4,
        "vocab_size": 16,
        "sliding_window": None,
        "layer_types": None,
        "_attn_implementation": "paged|flash_attention_3",
    }
    values.update(overrides)
    return PreTrainedConfig(**values)


def _cb_config(
    num_blocks: int = 8,
    block_size: int = 4,
    max_num_batched_tokens: int = 16,
    max_blocks_per_request: int = 32,
) -> ContinuousBatchingConfig:
    return ContinuousBatchingConfig(
        num_blocks=num_blocks,
        block_size=block_size,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_blocks_per_request * block_size,
    )


def _cache(cb_config: ContinuousBatchingConfig | None = None) -> PagedAttentionCache:
    return PagedAttentionCache(
        config=_config(),
        continuous_batching_config=cb_config or _cb_config(),
        device="cpu",
        dtype=torch.float32,
    )


def test_fifo_scheduler_chunks_prefill_and_finishes_remainder() -> None:
    scheduler = FIFOScheduler(_cache(), safety_margin=0.0)
    state = RequestState(request_id="req", initial_tokens=[1, 2, 3, 4, 5])
    scheduler.add_waiting_request(state)

    first = scheduler.schedule_batch(
        token_budget=3,
        cache_budget=16,
    )

    assert first.requests
    assert not first.cache_exhausted
    assert [future.state.request_id for future in first.requests] == ["req"]
    assert first.requests[0].query_length == 3
    assert not first.requests[0].has_new_token
    assert state.status == RequestStatus.PREFILLING
    assert state.tokens_to_process == [1, 2, 3]
    assert state.remaining_prefill_tokens == [4, 5]
    assert state.allocated_blocks == 1  # 3 tokens -> ceil(3/4) = 1 block (exact, no spare block)
    assert not first.use_decode_fast_path
    assert first.num_q_tokens == 3
    assert first.max_kv_read == 0
    assert list(scheduler.waiting_requests_order) == []

    state.position_offset += first.requests[0].query_length
    second = scheduler.schedule_batch(
        token_budget=3,
        cache_budget=16,
    )

    assert second.requests
    assert second.requests[0].query_length == 2
    assert second.requests[0].has_new_token
    assert state.status == RequestStatus.DECODING
    assert state.tokens_to_process == [4, 5]
    assert state.remaining_prefill_tokens == []
    assert state.allocated_blocks == 2
    assert not second.use_decode_fast_path
    assert second.num_q_tokens == 2
    assert second.max_kv_read == 3


def _decode_and_prefill_scheduler(safety_margin: float) -> tuple[FIFOScheduler, RequestState, RequestState]:
    scheduler = FIFOScheduler(_cache(), safety_margin=safety_margin)
    decoding = RequestState(request_id="decode", initial_tokens=[10])
    decoding.status = RequestStatus.DECODING
    decoding.tokens_to_process = [11]
    decoding.remaining_prefill_tokens = []
    decoding.position_offset = 1
    decoding.allocated_blocks = scheduler.cache.allocate_blocks(1, decoding.request_id, decoding.allocated_blocks) or 0
    prefilling = RequestState(request_id="prefill", initial_tokens=[20, 21, 22])
    prefilling.status = RequestStatus.PREFILLING
    prefilling.remaining_prefill_tokens = [21, 22]
    prefilling.tokens_to_process = [20]
    prefilling.position_offset = 1
    prefilling.allocated_blocks = (
        scheduler.cache.allocate_blocks(1, prefilling.request_id, prefilling.allocated_blocks) or 0
    )
    scheduler.active_requests[decoding.request_id] = decoding
    scheduler.active_requests[prefilling.request_id] = prefilling
    return scheduler, decoding, prefilling


def test_fifo_scheduler_admits_prefill_only_batch_when_cache_has_headroom() -> None:
    # Phase-separated scheduling: with cache headroom a prefill-only batch is admitted so decode
    # never rides along in a mixed batch. The active decode is left for a later iteration.
    scheduler, _, _ = _decode_and_prefill_scheduler(safety_margin=0.0)
    waiting = RequestState(request_id="waiting", initial_tokens=[30, 31])
    scheduler.add_waiting_request(waiting)

    scheduled = scheduler.schedule_batch(token_budget=3, cache_budget=16)

    assert scheduled.requests
    scheduled_ids = [future.state.request_id for future in scheduled.requests]
    assert "decode" not in scheduled_ids
    assert scheduled_ids[0] == "prefill"
    assert not scheduled.use_decode_fast_path
    assert scheduled.num_q_tokens == 3


def test_fifo_scheduler_runs_decode_only_batch_without_headroom() -> None:
    # Without cache headroom (past the safety margin), decode runs alone on the fast path and the
    # in-progress prefill is excluded, so the two phases never share a batch.
    scheduler, _, prefilling = _decode_and_prefill_scheduler(safety_margin=1.0)

    scheduled = scheduler.schedule_batch(token_budget=3, cache_budget=16)

    assert scheduled.requests
    assert [future.state.request_id for future in scheduled.requests] == ["decode"]
    assert scheduled.use_decode_fast_path
    assert scheduled.num_q_tokens == 1
    assert prefilling.remaining_prefill_tokens == [21, 22]


def test_fifo_scheduler_uses_decode_fast_path_for_single_token_batch() -> None:
    scheduler = FIFOScheduler(_cache(_cb_config(max_blocks_per_request=4)), safety_margin=0.0)
    for request_id, token_id in [("req-0", 10), ("req-1", 20)]:
        state = RequestState(request_id=request_id, initial_tokens=[token_id])
        state.status = RequestStatus.DECODING
        state.tokens_to_process = [token_id + 1]
        state.position_offset = 1
        state.allocated_blocks = scheduler.cache.allocate_blocks(1, request_id, state.allocated_blocks) or 0
        scheduler.active_requests[request_id] = state

    scheduled = scheduler.schedule_batch(token_budget=2, cache_budget=0)

    assert scheduled.requests
    assert [future.state.request_id for future in scheduled.requests] == ["req-0", "req-1"]
    assert [future.query_length for future in scheduled.requests] == [1, 1]
    assert scheduled.use_decode_fast_path
    assert scheduled.num_q_tokens == 2
    assert scheduled.max_kv_read == 2


def _decode_only_scheduler(num_requests: int) -> FIFOScheduler:
    scheduler = FIFOScheduler(_cache(_cb_config(max_blocks_per_request=4)), safety_margin=0.0)
    for idx in range(num_requests):
        request_id = f"req-{idx}"
        state = RequestState(request_id=request_id, initial_tokens=[10 * idx])
        state.status = RequestStatus.DECODING
        state.tokens_to_process = [10 * idx + 1]
        state.position_offset = 1
        state.allocated_blocks = scheduler.cache.allocate_blocks(1, request_id, state.allocated_blocks) or 0
        scheduler.active_requests[request_id] = state
    return scheduler


def test_fifo_scheduler_caps_the_decode_batch_at_the_resident_cap() -> None:
    # One query token per decoding request, so the resident cap is what bounds a decode batch. The
    # prefill token budget is larger and does not widen it; the overflow decodes on a later tick.
    scheduler = _decode_only_scheduler(num_requests=3)
    scheduler.max_num_seqs = 2

    scheduled = scheduler.schedule_batch(token_budget=8, cache_budget=0)

    assert [future.state.request_id for future in scheduled.requests] == ["req-0", "req-1"]
    assert scheduled.use_decode_fast_path
    assert scheduled.num_q_tokens == 2


def test_fifo_scheduler_decodes_every_ready_request_that_fits_the_cap() -> None:
    # With the cap above the ready set, all three enter one batch.
    scheduler = _decode_only_scheduler(num_requests=3)

    scheduled = scheduler.schedule_batch(token_budget=8, cache_budget=0)

    assert [future.state.request_id for future in scheduled.requests] == ["req-0", "req-1", "req-2"]
    assert scheduled.num_q_tokens == 3


def test_fifo_scheduler_rejects_varlen_request_that_exceeds_cache_budget() -> None:
    scheduler = FIFOScheduler(_cache(), safety_margin=0.0)
    state = RequestState(request_id="req", initial_tokens=[1, 2, 3, 4, 5])
    state.status = RequestStatus.PREFILLING
    state.remaining_prefill_tokens = [4, 5]
    state.tokens_to_process = [1, 2, 3]
    state.position_offset = 3
    state.allocated_blocks = scheduler.cache.allocate_blocks(1, state.request_id, state.allocated_blocks) or 0
    scheduler.active_requests[state.request_id] = state

    scheduled = scheduler.schedule_batch(token_budget=2, cache_budget=2)

    assert scheduled.requests == []
    assert not scheduled.cache_exhausted
    assert scheduled.use_decode_fast_path
    assert scheduled.num_q_tokens == 0
    assert scheduled.max_kv_read == 0
    assert state.remaining_prefill_tokens == [4, 5]


def test_fifo_scheduler_reports_cache_exhausted_when_allocation_fails_for_waiting_request() -> None:
    scheduler = FIFOScheduler(_cache(_cb_config(num_blocks=1)), safety_margin=0.0)
    scheduler.cache.allocate_blocks(1, "existing", allocated_blocks=0)
    state = RequestState(request_id="waiting", initial_tokens=[1])
    scheduler.add_waiting_request(state)

    scheduled = scheduler.schedule_batch(token_budget=1, cache_budget=16)

    assert scheduled.requests == []
    assert scheduled.cache_exhausted
    assert scheduled.use_decode_fast_path
    assert scheduled.num_q_tokens == 0
    assert scheduled.max_kv_read == 0
    assert list(scheduler.waiting_requests_order) == ["waiting"]
    assert scheduler.waiting_requests[state.request_id] is state


# --- reserve cache-pressure mode ----------------------------------------------------------------


def _reserve_scheduler() -> FIFOScheduler:
    # num_blocks=8, block_size=4 -> 32 tokens of KV; max_model_len=16 -> a worst-case request needs
    # ceil(16/4)=4 blocks, so the cache can always hold one (the reserve capacity check passes).
    cache = _cache(_cb_config(num_blocks=8, block_size=4, max_blocks_per_request=4))
    return FIFOScheduler(cache, safety_margin=0.0, kv_pressure_mode="reserve", max_model_len=16)


def test_reserve_mode_admits_only_the_reservation_fitting_prefix() -> None:
    scheduler = _reserve_scheduler()
    # Peaks (block_size=4): a -> ceil(8/4)=2, b -> ceil(12/4)=3, c -> ceil(16/4)=4 blocks (2+3+4 = 9 > 8).
    a = RequestState(request_id="a", initial_tokens=[1, 2, 3, 4], max_new_tokens=4)
    b = RequestState(request_id="b", initial_tokens=[1, 2, 3, 4], max_new_tokens=8)
    c = RequestState(request_id="c", initial_tokens=list(range(8)), max_new_tokens=8)
    for state in (a, b, c):
        scheduler.add_waiting_request(state)

    candidates = scheduler.get_prefill_candidates()

    # 2 + 3 = 5 blocks fit; c (+4 -> 9) overflows and stops admission at the head of line.
    assert [state.request_id for state in candidates] == ["a", "b"]


def test_reserve_mode_counts_running_reservations_before_admitting() -> None:
    scheduler = _reserve_scheduler()
    running = RequestState(request_id="run", initial_tokens=list(range(8)), max_new_tokens=8)  # peak 4 blocks
    running.status = RequestStatus.DECODING
    scheduler.active_requests["run"] = running
    small = RequestState(request_id="small", initial_tokens=[1, 2, 3, 4], max_new_tokens=4)  # peak 2 blocks
    big = RequestState(request_id="big", initial_tokens=[1, 2, 3, 4], max_new_tokens=8)  # peak 3 blocks
    scheduler.add_waiting_request(small)
    scheduler.add_waiting_request(big)

    candidates = scheduler.get_prefill_bucket()

    # 4 committed + 2 = 6 fits; big (+3 -> 9) would overcommit, so only the small prompt is admitted.
    assert [state.request_id for state in candidates] == ["small"]


def test_recompute_mode_admits_greedily_without_a_reservation_gate() -> None:
    cache = _cache(_cb_config(num_blocks=8, block_size=4, max_blocks_per_request=4))
    scheduler = FIFOScheduler(cache, safety_margin=0.0, kv_pressure_mode="recompute", max_model_len=16)
    a = RequestState(request_id="a", initial_tokens=[1, 2, 3, 4], max_new_tokens=4)
    b = RequestState(request_id="b", initial_tokens=[1, 2, 3, 4], max_new_tokens=8)
    c = RequestState(request_id="c", initial_tokens=list(range(8)), max_new_tokens=8)
    for state in (a, b, c):
        scheduler.add_waiting_request(state)

    # No reservation gate: every waiting prompt is a candidate (pressure is handled by preemption).
    assert [state.request_id for state in scheduler.get_prefill_candidates()] == ["a", "b", "c"]


def test_reserve_mode_rejects_cache_too_small_for_a_max_length_request() -> None:
    # max_model_len=64 needs ceil(64/4)=16 blocks, but the cache only has 8.
    cache = _cache(_cb_config(num_blocks=8, block_size=4, max_blocks_per_request=4))
    try:
        FIFOScheduler(cache, safety_margin=0.0, kv_pressure_mode="reserve", max_model_len=64)
    except ValueError as exc:
        assert "cannot serve a max-length request" in str(exc)
    else:
        raise AssertionError("expected a ValueError for an undersized reserve cache")


def test_recompute_resume_keeps_the_requests_timing_identity() -> None:
    # A soft-reset request is the same request resuming, not a new arrival: its creation and
    # first-schedule times must survive the rebuild, and re-scheduling must not restamp them.
    state = RequestState(request_id="r", initial_tokens=[1, 2], max_new_tokens=5)
    state.status = RequestStatus.PREFILLING  # leaves PENDING, stamping the first-schedule time
    first_scheduled = state.lifespan[0]
    state.status = RequestStatus.DECODING
    state.generated_tokens.append(101)

    fresh = state.create_equivalent_initial_request()

    assert fresh.created_time == state.created_time
    assert fresh.lifespan[0] == first_scheduled
    fresh.status = RequestStatus.PREFILLING  # resumed schedule must not restamp the start
    assert fresh.lifespan[0] == first_scheduled
