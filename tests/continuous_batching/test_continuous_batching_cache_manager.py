import pytest
import torch

from looped_cdb.continuous_batching.cache_manager import BlockManager, FullAttentionCacheAllocator


def test_block_manager_allocates_fifo_and_tracks_capacity() -> None:
    manager = BlockManager(num_blocks=3, block_size=4)

    first = manager.get_free_blocks(2)
    second = manager.get_free_blocks(1)

    assert first == [0, 1]
    assert second == [2]
    assert manager.num_free_blocks == 0
    assert manager.get_free_blocks(1) is None


def test_block_manager_reuses_freed_blocks_after_live_blocks() -> None:
    manager = BlockManager(num_blocks=3, block_size=4)
    blocks = manager.get_free_blocks(2)

    manager.free_blocks([blocks[0]])
    next_blocks = manager.get_free_blocks(2)

    assert next_blocks == [2, 0]
    assert manager.num_free_blocks == 0


def test_block_manager_rejects_freeing_unowned_block() -> None:
    manager = BlockManager(num_blocks=1, block_size=4)

    with pytest.raises(ValueError, match="not currently allocated"):
        manager.free_blocks([0])


def test_full_attention_allocator_extends_request_block_table() -> None:
    manager = BlockManager(num_blocks=4, block_size=4)
    allocator = FullAttentionCacheAllocator(index=0, block_size=4)

    assert allocator.allocate_blocks(1, "req", manager) == 1
    assert allocator.allocate_blocks(2, "req", manager) == 2

    assert allocator.block_table["req"] == [0, 1, 2]
    assert manager.num_free_blocks == 1


def test_full_attention_allocator_returns_read_indices_across_partial_block() -> None:
    manager = BlockManager(num_blocks=3, block_size=4)
    allocator = FullAttentionCacheAllocator(index=0, block_size=4)
    allocator.allocate_blocks(2, "req", manager)

    assert allocator.get_read_indices("req", past_length=3, query_length=3) == [0, 1, 2, 3, 4, 5]


def test_full_attention_allocator_returns_write_indices_across_block_boundary() -> None:
    manager = BlockManager(num_blocks=3, block_size=4)
    allocator = FullAttentionCacheAllocator(index=0, block_size=4)
    allocator.allocate_blocks(2, "req", manager)

    assert allocator.get_write_indices("req", past_length=3, query_length=3) == [3, 4, 5]


def test_full_attention_allocator_fills_flash_attention_block_table() -> None:
    manager = BlockManager(num_blocks=4, block_size=4)
    allocator = FullAttentionCacheAllocator(index=0, block_size=4)
    allocator.allocate_blocks(3, "req", manager)
    block_table = torch.full((4,), -1, dtype=torch.int32)

    allocator.fill_block_table("req", past_length=4, query_length=5, block_table=block_table)

    assert block_table.tolist() == [0, 1, 2, -1]


def test_full_attention_allocator_frees_all_request_blocks() -> None:
    manager = BlockManager(num_blocks=2, block_size=4)
    allocator = FullAttentionCacheAllocator(index=0, block_size=4)
    allocator.allocate_blocks(2, "req", manager)

    allocator.free_blocks("req", manager)

    assert "req" not in allocator.block_table
    assert manager.num_free_blocks == 2


def test_full_attention_allocator_rejects_unallocated_length() -> None:
    manager = BlockManager(num_blocks=1, block_size=4)
    allocator = FullAttentionCacheAllocator(index=0, block_size=4)
    allocator.allocate_blocks(1, "req", manager)

    with pytest.raises(ValueError, match="needs 2 blocks"):
        allocator.get_write_indices("req", past_length=3, query_length=2)
