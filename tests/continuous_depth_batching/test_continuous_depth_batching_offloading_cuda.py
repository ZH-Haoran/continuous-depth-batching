"""GPU coverage for the CPU-offload (swap) path of CDB preemption.

Identical in spirit to the CB swap round-trip test, but on a CDB cache whose per-layer KV is split
across multiple static recurrent-depth slots (``kv_slots_per_layer > 1``). The offloading manager
copies every cache layer, so this asserts that a preempted looped-model request comes back
byte-for-byte across all of its depth slots - the property that lets swap preserve each token's
exit-depth KV exactly.
"""

from types import SimpleNamespace

import pytest
import torch

from looped_cdb.continuous_depth_batching.cache import PagedAttentionCache
from looped_cdb.continuous_depth_batching.config import ContinuousDepthBatchingConfig
from looped_cdb.continuous_depth_batching.requests import FutureRequestState, RequestState, RequestStatus
from looped_cdb.continuous_depth_batching.scheduler import CDBScheduler
from looped_cdb.offloading_manager import OffloadingManager

pytestmark = pytest.mark.cuda

BLOCK_SIZE = 4
NUM_BLOCKS = 8
NUM_HIDDEN_LAYERS = 2
KV_SLOTS_PER_LAYER = 2  # two static recurrent-depth KV slots per layer


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        hidden_size=4,
        vocab_size=16,
        sliding_window=None,
        layer_types=["full_attention"] * NUM_HIDDEN_LAYERS,
        _attn_implementation="paged|flash_attention_3",
        total_ut_steps=3,
    )


def _cache() -> PagedAttentionCache:
    cdb_config = ContinuousDepthBatchingConfig(
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        max_num_batched_tokens=16,
        max_model_len=NUM_BLOCKS * BLOCK_SIZE,
        max_recurrent_steps=3,
        kv_policy="first_then_shared",
        kv_slots_per_layer=KV_SLOTS_PER_LAYER,
    )
    return PagedAttentionCache(
        config=_config(), continuous_batching_config=cdb_config, device="cuda", dtype=torch.float16
    )


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


def test_cdb_swap_offload_then_restore_round_trips_all_depth_slots() -> None:
    # The multi-slot cache has more layers than hidden layers, one per (layer, depth slot).
    cache = _cache()
    assert len(cache.key_cache) == NUM_HIDDEN_LAYERS * KV_SLOTS_PER_LAYER
    scheduler = CDBScheduler(cache=cache, max_recurrent_steps=3, max_num_seqs=4, safety_margin=0.0)
    manager = OffloadingManager(cache, scheduler, cpu_offload_space_gib=0.001)  # ~16k tiny blocks: ample
    assert manager.offloading_enabled

    # A decoding request occupying two KV blocks (positions 0..7).
    state = RequestState(request_id="r", initial_tokens=[1, 2, 3, 4, 5, 6], max_new_tokens=10)
    state.generated_tokens = [7, 8]
    state.tokens_to_process = [8]
    state.position_offset = 8
    state._status = RequestStatus.DECODING
    scheduler.active_requests["r"] = state
    cache.allocate_blocks(2, "r", 0)
    state.allocated_blocks = 2
    key_snapshots, value_snapshots = _fill_request_blocks(cache, "r")
    free_cpu_before = len(manager._free_cpu_blocks)

    # Offload: KV goes to the pinned CPU pool, GPU blocks are freed, request is re-queued.
    manager.offload_one_request()
    torch.cuda.synchronize()
    assert state.is_cpu_offloaded is True
    assert "r" in scheduler.waiting_requests
    assert "r" not in cache.cache_allocator.block_table
    assert len(manager._free_cpu_blocks) == free_cpu_before - 2

    # Re-admit: allocate fresh GPU blocks and clobber them, so a correct restore must repopulate them.
    cache.allocate_blocks(2, "r", 0)
    key_views, value_views = _block_views(cache)
    for block_id in cache.cache_allocator.block_table["r"]:
        for layer in range(len(cache.key_cache)):
            key_views[layer][block_id] = 0
            value_views[layer][block_id] = 0

    manager.restore_scheduled_requests([FutureRequestState(state, has_new_token=True, query_length=1)])
    torch.cuda.synchronize()

    assert state.is_cpu_offloaded is False
    assert len(manager._free_cpu_blocks) == free_cpu_before
    new_blocks = cache.cache_allocator.block_table["r"]
    for slot, block_id in enumerate(new_blocks):
        for layer in range(len(cache.key_cache)):
            assert torch.equal(key_views[layer][block_id], key_snapshots[slot][layer])
            assert torch.equal(value_views[layer][block_id], value_snapshots[slot][layer])
