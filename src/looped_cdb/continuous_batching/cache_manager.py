# Copyright 2025 The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0
# Modified for looped-model serving and continuous depth batching.
# See THIRD_PARTY_NOTICES.md and LICENSES/Apache-2.0.txt.

"""Paged KV block ownership for continuous batching.

This module mirrors the first cache-management layer from Hugging Face's
continuous batching implementation, but keeps the initial scope focused on the
target path for this repo: full-attention paged KV cache with FlashAttention
block tables.

The ``BlockManager`` owns physical page IDs. Allocators then maintain the
per-request block tables that map logical sequence positions to those physical
pages. Scheduling can allocate pages before a CUDA launch, and the pages remain
owned by the request while that launch is in flight. Async H2D/D2H and stream
ordering belong in ``input_outputs.py`` and ``model_runner.py``; this file stays
CPU-side and torch-light except when materializing block tables for paged
attention.

Reference:
https://github.com/huggingface/transformers/blob/main/src/transformers/generation/continuous_batching/cache_manager.py
"""

from abc import ABC, abstractmethod
from collections import deque

import torch

from .requests import logger


class BlockManager:
    """Owns the pool of physical blocks backing the paged KV cache.

    Blocks are either free or allocated. This is the non-prefix-sharing subset
    of HF's block manager: no block metadata, no initialized reusable prefix
    blocks, and no request forking.
    """

    def __init__(self, num_blocks: int, block_size: int) -> None:
        """Initializes the block manager with a given number of blocks (num_blocks) of size (block_size)."""

        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free_block_ids: deque[int] = deque(range(num_blocks))
        self._allocated_block_ids: set[int] = set()
        self.peak_num_allocated_blocks = 0

    @property
    def num_free_blocks(self) -> int:
        """Returns the number of free blocks left."""

        return len(self._free_block_ids)

    @property
    def num_allocated_blocks(self) -> int:
        """Returns the number of currently allocated blocks."""

        return len(self._allocated_block_ids)

    def reset_peak_usage(self) -> None:
        """Reset peak block usage to the current allocation."""

        self.peak_num_allocated_blocks = self.num_allocated_blocks

    def has_enough_free_blocks(self, n_blocks: int) -> bool:
        """Checks if there are enough free blocks to allocate the requested number of blocks (n_blocks)."""

        return len(self._free_block_ids) >= n_blocks

    def get_free_blocks(self, n_blocks: int) -> list[int] | None:
        """Allocate physical block IDs, returning ``None`` when capacity is insufficient."""

        if not self.has_enough_free_blocks(n_blocks):
            return None

        allocated_block_ids = [self._free_block_ids.popleft() for _ in range(n_blocks)]
        self._allocated_block_ids.update(allocated_block_ids)
        self.peak_num_allocated_blocks = max(self.peak_num_allocated_blocks, self.num_allocated_blocks)
        return allocated_block_ids

    def free_blocks(self, blocks: list[int]) -> None:
        """Release block IDs back to the free pool."""

        for block_id in blocks:
            if block_id not in self._allocated_block_ids:
                raise ValueError(f"Cannot free block {block_id}: block is not currently allocated")
            self._allocated_block_ids.remove(block_id)
            self._free_block_ids.append(block_id)


class CacheAllocator(ABC):
    """Abstract base class for cache managers. Cache managers keep track of per-request cache allocations, determine
    when a new physical block needs to be allocated and compute physical indices for reading or writing to the cache."""

    _index: int
    block_size: int
    block_table: dict[str, list[int]]  # request_id -> list of block_ids allocated to the request

    @abstractmethod
    def allocate_blocks(self, n_blocks: int, request_id: str, block_manager: BlockManager) -> int | None:
        """Allocates (n_blocks) for a given (request_id) using the (block_manager). Returns the num of blocks allocated
        if successful and None otherwise."""

    def free_blocks(self, request_id: str, block_manager: BlockManager) -> None:
        """Frees all blocks associated with a (request_id) using the (block_manager)."""

        if request_id in self.block_table:
            block_manager.free_blocks(self.block_table.pop(request_id))
        else:
            logger.warning(
                f"CacheAllocator {self._index} attempted to free blocks for non-existent request_id: {request_id}"
            )

    @abstractmethod
    def get_read_indices(self, request_id: str, past_length: int, query_length: int) -> list[int]:
        """Returns the physical indices of where to read request_id's cache in the cache tensor."""

    @abstractmethod
    def get_write_indices(self, request_id: str, past_length: int, query_length: int) -> list[int]:
        """Returns the physical indices of where to write request_id's cache in the cache tensor."""

    @abstractmethod
    def fill_block_table(
        self,
        request_id: str,
        past_length: int,
        query_length: int,
        block_table: torch.Tensor,
    ) -> None:
        """Fills the block table for a given request_id, past_length and query_length."""


class FullAttentionCacheAllocator(CacheAllocator):
    """Cache allocator for full-attention layers."""

    def __init__(self, index: int, block_size: int) -> None:
        """Initializes the cache allocator for full-attention layers.

        Args:
            index: The allocator index used in diagnostics.
            block_size: The size of the blocks in the cache.
        """

        self._index = index
        self.block_size = block_size
        self.block_table: dict[str, list[int]] = {}

    def allocate_blocks(self, n_blocks: int, request_id: str, block_manager: BlockManager) -> int | None:
        """Allocate (n_blocks) for a given (request_id) using the (block_manager). Returns the number of blocks
        allocated if successful and None otherwise. For full-attention layers, we always allocate the number of
        requested blocks."""

        request_blocks = self.block_table.get(request_id)
        if request_blocks is None:
            request_blocks = []
            self.block_table[request_id] = request_blocks

        allocated_blocks = block_manager.get_free_blocks(n_blocks)
        if allocated_blocks is None:
            return None

        request_blocks.extend(allocated_blocks)
        return n_blocks

    def get_read_indices(self, request_id: str, past_length: int, query_length: int) -> list[int]:
        """Returns the physical indices of where to read request_id's cache. For full-attention layers, we
        first write the new cache to the cache tensor and then read the entire cache from the beginning to the end."""

        # Retrieve the block table for the request and raise an error if it doesn't exist
        request_blocks = self._get_request_blocks(request_id)

        total_length = past_length + query_length
        self._validate_length(total_length, request_blocks, request_id)

        num_full_blocks = total_length // self.block_size
        remainder = total_length % self.block_size

        # Physical block IDs index fixed-size pages in the flat KV cache.
        physical_indices = []
        for block_index in range(num_full_blocks):
            start = request_blocks[block_index] * self.block_size
            physical_indices.extend(range(start, start + self.block_size))
        if remainder:
            start = request_blocks[num_full_blocks] * self.block_size
            physical_indices.extend(range(start, start + remainder))
        return physical_indices

    def get_write_indices(self, request_id: str, past_length: int, query_length: int) -> list[int]:
        """Returns the physical indices for writing to the cache. For full-attention layers, we write the new
        cache as a continuation of the existing cache for the same request."""

        if query_length <= 0:
            return []

        request_blocks = self._get_request_blocks(request_id)
        total_length = past_length + query_length
        self._validate_length(total_length, request_blocks, request_id)

        start_block = past_length // self.block_size
        start_offset = past_length % self.block_size
        end_pos = total_length
        end_block = (end_pos - 1) // self.block_size

        # Physical block IDs index fixed-size pages in the flat KV cache.
        physical_indices = []
        for block_index in range(start_block, end_block + 1):
            block_start = request_blocks[block_index] * self.block_size
            local_start = start_offset if block_index == start_block else 0
            local_end = (end_pos - 1) % self.block_size + 1 if block_index == end_block else self.block_size
            physical_indices.extend(range(block_start + local_start, block_start + local_end))
        return physical_indices

    def fill_block_table(
        self,
        request_id: str,
        past_length: int,
        query_length: int,
        block_table: torch.Tensor,
    ) -> None:
        """Fills the block table for a given request_id, past_length and query_length."""

        request_blocks = self._get_request_blocks(request_id)
        total_length = past_length + query_length
        self._validate_length(total_length, request_blocks, request_id)

        num_blocks_needed = (total_length + self.block_size - 1) // self.block_size
        block_table[:num_blocks_needed] = torch.tensor(
            request_blocks[:num_blocks_needed],
            device=block_table.device,
            dtype=block_table.dtype,
        )

    def _get_request_blocks(self, request_id: str) -> list[int]:
        """Returns the block table for a request, or raises if the request has no allocated cache blocks."""

        request_blocks = self.block_table.get(request_id)
        if request_blocks is None:
            raise ValueError(f"No block table found for request {request_id}")
        return request_blocks

    def _validate_length(self, total_length: int, request_blocks: list[int], request_id: str) -> None:
        """Checks that a request has enough allocated blocks to cover a sequence length."""

        if total_length < 0:
            raise ValueError(f"total_length must be non-negative, got {total_length}")
        blocks_needed = (total_length + self.block_size - 1) // self.block_size
        if blocks_needed > len(request_blocks):
            raise ValueError(
                f"Request {request_id} needs {blocks_needed} blocks for length {total_length}, "
                f"but only has {len(request_blocks)} allocated"
            )
