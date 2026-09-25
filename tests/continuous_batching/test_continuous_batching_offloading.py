"""Preemption (CPU offload + soft-reset recompute) for continuous batching.

The soft-reset path and all scheduler/request bookkeeping are exercised on CPU with a real paged
cache and ``cpu_offload_space=None`` (no pinned pool). The GPU->CPU block copy of the swap path
requires pinned host memory and a CUDA device, so it is covered by the GPU test suite instead.
"""

import pytest
import torch
from transformers import PreTrainedConfig

from looped_cdb.continuous_batching.cache import PagedAttentionCache
from looped_cdb.continuous_batching.config import ContinuousBatchingConfig
from looped_cdb.continuous_batching.requests import FutureRequestState, RequestState, RequestStatus
from looped_cdb.continuous_batching.scheduler import FIFOScheduler
from looped_cdb.offloading_manager import OffloadingManager


def _config() -> PreTrainedConfig:
    return PreTrainedConfig(
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        hidden_size=4,
        vocab_size=16,
        sliding_window=None,
        layer_types=None,
        _attn_implementation="paged|flash_attention_3",
    )


def _cache(num_blocks: int = 8, block_size: int = 4) -> PagedAttentionCache:
    cb_config = ContinuousBatchingConfig(
        num_blocks=num_blocks,
        block_size=block_size,
        max_num_batched_tokens=16,
        max_model_len=num_blocks * block_size,
    )
    return PagedAttentionCache(
        config=_config(), continuous_batching_config=cb_config, device="cpu", dtype=torch.float32
    )


def _decoding_request(
    scheduler: FIFOScheduler, cache: PagedAttentionCache, request_id: str, prompt_len: int
) -> RequestState:
    """Register a request as actively decoding with real cache blocks allocated to it."""

    state = RequestState(request_id=request_id, initial_tokens=list(range(prompt_len)), max_new_tokens=10)
    state.generated_tokens = [100, 101]
    state.tokens_to_process = [101]
    state.position_offset = prompt_len + 2
    state._status = RequestStatus.DECODING
    scheduler.active_requests[request_id] = state
    blocks_needed = -(-state.position_offset // cache.block_size)
    cache.allocate_blocks(blocks_needed, request_id, 0)
    state.allocated_blocks = blocks_needed
    return state


# --- request soft-reset bookkeeping -------------------------------------------------------------


def test_create_equivalent_initial_request_folds_generated_and_reduces_budget() -> None:
    state = RequestState(request_id="r", initial_tokens=[1, 2, 3], max_new_tokens=10)
    state.generated_tokens = [7, 8]

    fresh = state.create_equivalent_initial_request()

    assert fresh.request_id == "r"
    assert fresh.initial_tokens == [1, 2, 3, 7, 8]  # generated folded onto the prompt
    assert fresh.remaining_prefill_tokens == [1, 2, 3, 7, 8]  # re-prefilled from scratch
    assert fresh.max_new_tokens == 8  # 10 - 2 already generated
    assert fresh._true_initial_tokens == 3  # original prompt boundary preserved
    assert fresh.status == RequestStatus.PENDING


def test_to_generation_output_recovers_true_prompt_and_full_generation() -> None:
    state = RequestState(request_id="r", initial_tokens=[1, 2, 3], max_new_tokens=10)
    state.generated_tokens = [7, 8]
    fresh = state.create_equivalent_initial_request()
    # Simulate more decoding after the reset.
    fresh.generated_tokens = [9]

    output = fresh.to_generation_output()

    assert output.prompt_ids == [1, 2, 3]  # true prompt, not the folded one
    assert output.generated_tokens == [7, 8, 9]  # pre-reset tokens + post-reset tokens


def test_soft_reset_is_idempotent_across_repeated_preemptions() -> None:
    state = RequestState(request_id="r", initial_tokens=[1, 2], max_new_tokens=10)
    state.generated_tokens = [5]
    once = state.create_equivalent_initial_request()
    once.generated_tokens = [6]

    twice = once.create_equivalent_initial_request()

    assert twice.initial_tokens == [1, 2, 5, 6]
    assert twice._true_initial_tokens == 2  # still the original prompt length, not 3
    assert twice.max_new_tokens == 8  # 10 - 2 generated so far


# --- scheduler eviction / admission -------------------------------------------------------------


def test_pop_request_to_evict_takes_the_least_computed_request() -> None:
    """The victim is the request whose KV costs least to give up, not the one in the last dict slot.

    Preemption re-runs (recompute) or copies (offload) every token the victim has computed, so cost
    scales with ``current_len``. The rule must not depend on the drain flag either: a request finishing
    clears it, which would otherwise make the next preemption fall on a different request entirely.
    """

    scheduler = FIFOScheduler(_cache(), safety_margin=0.0)
    for req_id, computed in [("veteran", 20), ("rookie", 2), ("middling", 9)]:
        state = RequestState(request_id=req_id, initial_tokens=[1])
        state.position_offset = computed
        scheduler.active_requests[req_id] = state

    assert scheduler.pop_request_to_evict()[0] == "rookie"

    scheduler.block_new_requests = True
    assert scheduler.pop_request_to_evict()[0] == "middling"


def test_a_restored_request_is_not_the_standing_preemption_victim() -> None:
    """Restoring re-inserts at the tail of ``active_requests``, so position cannot select the victim.

    A restored request carries every token it had computed. Picking by dict position would evict it
    again immediately, swapping the largest KV in the batch back out.
    """

    scheduler = FIFOScheduler(_cache(), safety_margin=0.0, kv_pressure_mode="offload")

    rookie = RequestState(request_id="rookie", initial_tokens=[1, 2], max_new_tokens=64)
    rookie.status = RequestStatus.DECODING
    rookie.remaining_prefill_tokens = []
    rookie.tokens_to_process = [3]
    rookie.position_offset = 2
    rookie.allocated_blocks = scheduler.cache.allocate_blocks(1, "rookie", 0) or 0
    scheduler.active_requests["rookie"] = rookie

    veteran = RequestState(request_id="veteran", initial_tokens=[1, 2], max_new_tokens=64)
    veteran.status = RequestStatus.DECODING
    veteran.remaining_prefill_tokens = []
    veteran.tokens_to_process = [3]
    veteran.position_offset = 22
    veteran.is_cpu_offloaded = True
    scheduler.add_waiting_request(veteran)

    scheduler.admit_offloaded_restores()

    # The restored request lands at the tail, and is nonetheless the last one to be preempted.
    assert list(scheduler.active_requests) == ["rookie", "veteran"]
    assert scheduler.pop_request_to_evict()[0] == "rookie"


def test_waiting_candidates_exclude_offloaded_requests() -> None:
    scheduler = FIFOScheduler(_cache(), safety_margin=0.0)
    for req_id, offloaded in [("f1", False), ("o1", True), ("f2", False)]:
        state = RequestState(request_id=req_id, initial_tokens=[1])
        state.is_cpu_offloaded = offloaded
        scheduler.add_waiting_request(state)

    # Offloaded requests resume through the decode path, so they are not prefill (waiting) candidates.
    assert [state.request_id for state in scheduler._get_waiting_candidates()] == ["f1", "f2"]


def test_admit_offloaded_restores_moves_them_to_active_and_allocates() -> None:
    cache = _cache()
    scheduler = FIFOScheduler(cache, safety_margin=0.0, kv_pressure_mode="offload", max_num_seqs=1)
    fresh = RequestState(request_id="fresh", initial_tokens=[1, 2])
    scheduler.add_waiting_request(fresh)
    for req_id in ("off", "off2"):
        off = RequestState(request_id=req_id, initial_tokens=[1, 2, 3])
        off.is_cpu_offloaded = True
        off._status = RequestStatus.DECODING
        off.tokens_to_process = [9]
        off.position_offset = 3
        scheduler.add_waiting_request(off)

    assert [state.request_id for state in scheduler.admit_offloaded_restores()] == ["off"]

    # The offloaded request rejoins the active (decode) set with blocks; the fresh prompt stays waiting,
    # and so does the second restore, since the resident cap bounds restores like any admission.
    assert "off" in scheduler.active_requests
    assert scheduler.active_requests["off"].allocated_blocks > 0
    assert scheduler.active_requests["off"].status == RequestStatus.DECODING
    assert "off" not in scheduler.waiting_requests
    assert "fresh" in scheduler.waiting_requests
    assert "off2" in scheduler.waiting_requests


def test_admit_offloaded_restores_blocked_while_draining() -> None:
    scheduler = FIFOScheduler(_cache(), safety_margin=0.0, kv_pressure_mode="offload")
    off = RequestState(request_id="off", initial_tokens=[1])
    off.is_cpu_offloaded = True
    off._status = RequestStatus.DECODING
    scheduler.add_waiting_request(off)
    scheduler.block_new_requests = True

    scheduler.admit_offloaded_restores()

    # While draining after a preemption, restores wait too, so the batch shrinks before refilling.
    assert "off" in scheduler.waiting_requests
    assert "off" not in scheduler.active_requests


def test_block_new_requests_stops_admitting_waiting_prompts() -> None:
    scheduler = FIFOScheduler(_cache(), safety_margin=0.0)
    scheduler.add_waiting_request(RequestState(request_id="w", initial_tokens=[1, 2]))

    assert [s.request_id for s in scheduler.get_prefill_candidates()] == ["w"]

    scheduler.block_new_requests = True
    assert scheduler.get_prefill_candidates() == []


# --- manager preemption end to end (soft-reset path, CPU) ----------------------------------------


def test_manager_reports_offloading_disabled_without_a_pool() -> None:
    cache = _cache()
    scheduler = FIFOScheduler(cache, safety_margin=0.0)
    manager = OffloadingManager(cache, scheduler, cpu_offload_space_gib=None)

    assert manager.offloading_enabled is False


def test_offload_one_request_soft_resets_victim_and_frees_its_blocks() -> None:
    cache = _cache()
    scheduler = FIFOScheduler(cache, safety_margin=0.0)
    manager = OffloadingManager(cache, scheduler, cpu_offload_space_gib=None)
    victim = _decoding_request(scheduler, cache, "victim", prompt_len=6)
    free_before = cache.get_num_free_blocks()
    victim_blocks = victim.allocated_blocks

    manager.offload_one_request()

    # Victim left the active set and its GPU blocks were freed.
    assert "victim" not in scheduler.active_requests
    assert "victim" not in cache.cache_allocator.block_table
    assert cache.get_num_free_blocks() == free_before + victim_blocks

    # It is re-queued as a fresh (soft-reset) prompt, not offloaded to CPU.
    assert "victim" in scheduler.waiting_requests
    requeued = scheduler.waiting_requests["victim"]
    assert requeued.is_cpu_offloaded is False
    assert requeued.initial_tokens == [*range(6), 100, 101]
    assert requeued.max_new_tokens == 8  # 10 - 2 generated
    assert requeued._true_initial_tokens == 6

    # Admissions are blocked so the batch drains before refilling.
    assert scheduler.block_new_requests is True


def test_offload_one_request_uses_lifo_victim_while_draining() -> None:
    cache = _cache()
    scheduler = FIFOScheduler(cache, safety_margin=0.0)
    manager = OffloadingManager(cache, scheduler, cpu_offload_space_gib=None)
    _decoding_request(scheduler, cache, "old", prompt_len=4)
    _decoding_request(scheduler, cache, "new", prompt_len=4)
    scheduler.block_new_requests = True  # already draining

    manager.offload_one_request()

    # The newest active request is the one preempted.
    assert "new" not in scheduler.active_requests
    assert "old" in scheduler.active_requests


def test_preemption_counters_track_soft_resets_and_reset() -> None:
    cache = _cache()
    scheduler = FIFOScheduler(cache, safety_margin=0.0)
    manager = OffloadingManager(cache, scheduler, cpu_offload_space_gib=None)  # no pool -> soft reset
    assert manager.preemption_stats() == {
        "preemptions": 0,
        "offload_preemptions": 0,
        "recompute_preemptions": 0,
        "restores": 0,
    }

    _decoding_request(scheduler, cache, "victim", prompt_len=6)
    manager.offload_one_request()

    # Without a CPU pool the victim is recomputed (soft reset), not offloaded.
    assert manager.num_preemptions == 1
    assert manager.preemption_stats() == {
        "preemptions": 1,
        "offload_preemptions": 0,
        "recompute_preemptions": 1,
        "restores": 0,
    }

    manager.reset()
    assert manager.num_preemptions == 0


# --- manager swap path end to end (offload -> restore round-trip, CPU) ---------------------------


def _block_views(cache: PagedAttentionCache) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    shape = (-1, cache.block_size, cache.num_key_value_heads, cache.head_dim)
    return [k.view(*shape) for k in cache.key_cache], [v.view(*shape) for v in cache.value_cache]


def _fill_request_blocks(cache: PagedAttentionCache, request_id: str) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Write a distinct pattern into each of a request's KV blocks; return per-request-block snapshots."""

    key_views, value_views = _block_views(cache)
    blocks = cache.cache_allocator.block_table[request_id]
    key_snapshots, value_snapshots = [], []
    for slot, block_id in enumerate(blocks):
        for layer in range(len(cache.key_cache)):
            key_views[layer][block_id] = float(block_id) + 0.5 + 10 * layer + 100 * slot
            value_views[layer][block_id] = -(float(block_id) + 0.5 + 10 * layer + 100 * slot)
    for block_id in blocks:
        key_snapshots.append(torch.stack([key_views[layer][block_id].clone() for layer in range(len(cache.key_cache))]))
        value_snapshots.append(
            torch.stack([value_views[layer][block_id].clone() for layer in range(len(cache.key_cache))])
        )
    return key_snapshots, value_snapshots


def test_swap_offload_then_restore_round_trips_kv_blocks_on_cpu() -> None:
    # The swap copy runs on CPU when the pool is unpinned (``pin_memory=False``), so the exact same
    # offload -> clobber -> restore round-trip the CUDA test asserts byte-for-byte runs here too,
    # covering the block-copy transfer path without a GPU.
    cache = _cache()
    scheduler = FIFOScheduler(cache, safety_margin=0.0)
    manager = OffloadingManager(cache, scheduler, cpu_offload_space_gib=0.001, pin_memory=False)
    assert manager.offloading_enabled

    state = _decoding_request(scheduler, cache, "r", prompt_len=6)  # decodes into two KV blocks
    key_snapshots, value_snapshots = _fill_request_blocks(cache, "r")
    free_cpu_before = len(manager._free_cpu_blocks)
    offloaded_blocks = state.allocated_blocks

    manager.offload_one_request()

    assert state.is_cpu_offloaded is True
    assert "r" in scheduler.waiting_requests
    assert "r" not in cache.cache_allocator.block_table  # GPU blocks freed
    assert len(manager._free_cpu_blocks) == free_cpu_before - offloaded_blocks

    # Re-admit: allocate fresh GPU blocks and clobber them, so a correct restore must repopulate them.
    cache.allocate_blocks(offloaded_blocks, "r", 0)
    key_views, value_views = _block_views(cache)
    for block_id in cache.cache_allocator.block_table["r"]:
        for layer in range(len(cache.key_cache)):
            key_views[layer][block_id] = 0
            value_views[layer][block_id] = 0

    manager.restore_scheduled_requests([FutureRequestState(state, has_new_token=True, query_length=1)])

    assert state.is_cpu_offloaded is False
    assert len(manager._free_cpu_blocks) == free_cpu_before  # CPU blocks returned to the pool
    for slot, block_id in enumerate(cache.cache_allocator.block_table["r"]):
        for layer in range(len(cache.key_cache)):
            assert torch.equal(key_views[layer][block_id], key_snapshots[slot][layer])
            assert torch.equal(value_views[layer][block_id], value_snapshots[slot][layer])


def test_offload_refuses_to_offload_an_already_offloaded_request() -> None:
    cache = _cache()
    scheduler = FIFOScheduler(cache, safety_margin=0.0)
    manager = OffloadingManager(cache, scheduler, cpu_offload_space_gib=0.001, pin_memory=False)
    state = _decoding_request(scheduler, cache, "r", prompt_len=6)
    state.is_cpu_offloaded = True

    with pytest.raises(RuntimeError, match="already offloaded"):
        manager._offload_to_cpu("r", state)


def test_reserve_admission_zeroes_the_effective_safety_margin() -> None:
    # Reserve admission is exact, so the free-block margin is disabled. A caller that reads back the
    # requested margin would report a throttle the scheduler never applied.
    scheduler = FIFOScheduler(_cache(), safety_margin=0.2, kv_pressure_mode="reserve", max_model_len=32)

    assert scheduler.safety_margin == 0.0


def test_preempting_modes_keep_the_requested_safety_margin() -> None:
    scheduler = FIFOScheduler(_cache(), safety_margin=0.2, kv_pressure_mode="recompute")

    assert scheduler.safety_margin == 0.2
