# Copyright 2025 The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0
# Modified for looped-model serving and continuous depth batching.
# See THIRD_PARTY_NOTICES.md and LICENSES/Apache-2.0.txt.

"""Paged attention cache for continuous batching.

This module ports the cache layer from Hugging Face's continuous batching
implementation while keeping it focused on the target path:
single-GPU, full-attention, paged KV cache management.

Reference:
https://github.com/huggingface/transformers/blob/main/src/transformers/generation/continuous_batching/cache.py
"""

import inspect
from typing import Any

import torch
from transformers.configuration_utils import PreTrainedConfig

from looped_cdb.kv_cache_policy import (
    DEPTH_INDEXED,
    kv_slot_for_step,
    resolve_kv_slots_per_layer,
)
from looped_cdb.paged_cache_geometry import (
    PagedCacheConfig,
    PagedKVCacheGeometry,
    infer_num_cache_layers,
    looped_recurrent_steps,
)

from .cache_manager import BlockManager, FullAttentionCacheAllocator
from .config import is_supported_attention_implementation
from .requests import logger


class PagedAttentionCache:
    """
    Manages the cache for a paged attention mechanism.

    The cache uses a three-level hierarchy:
    - Pages: The smallest unit of cache, a page has a size of [num_heads, head_size], which is the space needed to
        store the key or value states for one token and one layer. For a model with only full-attention layers, to store
        the KV cache of one token, we need `2 * num_layers` pages: key and values each take `num_layers` pages.
        Pages are grouped into blocks:
    - Blocks: A block is a collection of `block_size` pages, serving as the allocation unit to reduce management
        complexity and fragmentation. Cache is allocated and freed block by block, not page by page.
    - Cache tensors: The physical supports for the cache. There is one key tensor and one value tensor per layer, and
        each tensor has shape `[num_blocks * block_size, num_heads, head_size]`.

    This implementation keeps one full-attention cache allocator containing all model layers.
    """

    def __init__(
        self,
        config: PreTrainedConfig,
        continuous_batching_config: PagedCacheConfig,
        device: torch.device | str,
        dtype: torch.dtype = torch.float16,
        *,
        geometry: PagedKVCacheGeometry | None = None,
    ) -> None:
        """Initialize a paged attention cache for efficient memory usage.

        Args:
            config: Model configuration.
            continuous_batching_config: Continuous batching configuration containing cache parameters.
            device: Device for the cache tensors.
            dtype: Data type of the cache.
        """

        self.config = config
        self.continuous_batching_config = continuous_batching_config
        self.dtype = dtype
        self.device = torch.device(device)

        self.block_size = continuous_batching_config.block_size
        if self.block_size <= 0:
            raise ValueError(f"Block size must be positive, but got {self.block_size}")

        # HF CB has separate paths for mixed attention layouts. This implementation supports full-attention models only.
        if not is_supported_attention_implementation(config):
            attn_implementation = getattr(config, "_attn_implementation", None)
            raise NotImplementedError(
                "Continuous batching currently supports FlashAttention-backed paged attention only, "
                f"but got attn_implementation={attn_implementation!r}"
            )
        layer_types = getattr(config, "layer_types", None)
        has_sliding_layers = layer_types is not None and any(
            layer_type != "full_attention" for layer_type in layer_types
        )
        if getattr(config, "sliding_window", None) is not None or has_sliding_layers:
            raise NotImplementedError("Continuous batching currently supports full-attention models only")

        self.geometry = geometry or PagedKVCacheGeometry.from_model(
            config,
            block_size=self.block_size,
            dtype=dtype,
            kv_policy=getattr(continuous_batching_config, "kv_policy", None),
            kv_slots_per_layer=getattr(continuous_batching_config, "kv_slots_per_layer", None),
            total_recurrent_steps=getattr(continuous_batching_config, "max_recurrent_steps", None),
            assume_all_layers_recurrent=getattr(continuous_batching_config, "max_recurrent_steps", None) is not None,
        )
        self.kv_policy = self.geometry.kv_policy
        self.kv_slots_per_layer = self.geometry.kv_slots_per_layer
        self.num_layers = self.geometry.num_layers
        self.num_key_value_heads = self.geometry.num_key_value_heads
        self.head_dim = self.geometry.head_dim

        page_size = self.head_dim * self.num_key_value_heads
        assert continuous_batching_config.num_blocks is not None
        self.num_blocks = continuous_batching_config.num_blocks
        self.max_num_batched_tokens = continuous_batching_config.max_num_batched_tokens
        self.num_pages = self.num_blocks * self.block_size
        logger.info(
            f"PagedAttentionCache initialized with {self.num_blocks = }, {self.block_size = }, {page_size = }, "
            f"{self.max_num_batched_tokens = }"
        )

        self.max_blocks_per_request = continuous_batching_config.decode_block_table_width()
        self.layer_index_to_group_indices = [(0, layer_idx) for layer_idx in range(self.num_layers)]

        # Initialize the cache
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
        # We add two extra blocks to the cache as a padding zone that no BlockManager ever allocates from: one for the
        # sentinel index (marks the spot of a new token in the read indices) and one for the trash index (for padding,
        # block is never used so writes are silently discarded)
        self.cache_shape = (
            (self.num_blocks + self.geometry.extra_blocks) * self.block_size,
            self.num_key_value_heads,
            self.head_dim,
        )
        self.sentinel_index = self.cache_shape[0] - 1
        self.trash_index = self.sentinel_index - 1
        for _ in range(self.num_layers):
            new_layer_key_cache = torch.empty(self.cache_shape, dtype=self.dtype, device=self.device)
            new_layer_value_cache = torch.empty(self.cache_shape, dtype=self.dtype, device=self.device)
            torch._dynamo.mark_static_address(new_layer_key_cache)
            torch._dynamo.mark_static_address(new_layer_value_cache)
            self.key_cache.append(new_layer_key_cache)
            self.value_cache.append(new_layer_value_cache)
        logger.info(f"{self.cache_shape = } {self.key_cache[0].shape = } {self.key_cache[0].numel() = }")

        # Block management data structures
        self.cache_allocator = FullAttentionCacheAllocator(0, self.block_size)
        self._block_manager = BlockManager(self.num_blocks, self.block_size)

        # For block table support, we lazy init the name of the block table key
        self._block_table_key = None

    def will_allocation_be_successful(self, num_requested_blocks: int, allocated_blocks: int) -> bool:  # noqa: ARG002
        """Returns a boolean indicating if the allocation of (num_requested_blocks) blocks will be successful."""

        return num_requested_blocks <= self.get_num_free_blocks()

    @staticmethod
    def _kv_cache_policy(config: PreTrainedConfig) -> str:
        """The recurrent KV layout, defaulting to the one a checkpoint is trained under."""

        return str(getattr(config, "_cdb_kv_policy", DEPTH_INDEXED))

    @staticmethod
    def _kv_slots_per_layer(config: PreTrainedConfig, kv_policy: str) -> int:
        requested_slots = getattr(config, "_cdb_kv_slots_per_layer", None)
        return resolve_kv_slots_per_layer(
            kv_policy,
            total_recurrent_steps=looped_recurrent_steps(config) or 1,
            requested_slots=None if requested_slots is None else int(requested_slots),
        )

    @staticmethod
    def _infer_num_cache_layers(
        config: PreTrainedConfig,
        layer_types: list[str] | None,
        kv_policy: str = DEPTH_INDEXED,
        kv_slots_per_layer: int = 1,
    ) -> int:
        """Return the number of cache layers to allocate.

        Looped models replicate their core layers once per KV slot, while prelude
        and coda layers execute once per token and hold one slot each.
        """

        return infer_num_cache_layers(
            config,
            layer_types,
            policy=kv_policy,
            slots_per_layer=kv_slots_per_layer,
            total_recurrent_steps=looped_recurrent_steps(config) or 1,
        )

    def kv_slot_for_recurrent_step(self, recurrent_step: int) -> int:
        return kv_slot_for_step(
            recurrent_step,
            policy=self.kv_policy,
            slots_per_layer=self.kv_slots_per_layer,
        )

    @property
    def requires_slot_homogeneous_recurrent_batches(self) -> bool:
        """Whether one recurrent launch may hold only steps that write the same slot.

        A launch writes a single slot index, so any layout mapping steps to more than
        one slot constrains what can share a batch.
        """

        return self.kv_slots_per_layer > 1

    def allocate_blocks(self, n_blocks: int, request_id: str, allocated_blocks: int) -> int | None:
        """Allocate cache blocks for a given request. Actual allocation is done by the cache manager, and this method
        returns the number of blocks allocated."""

        if not self.will_allocation_be_successful(n_blocks, allocated_blocks):
            return None

        num_allocated_blocks = self.cache_allocator.allocate_blocks(n_blocks, request_id, self._block_manager)
        if num_allocated_blocks is None:
            raise ValueError(f"Failed to allocate {n_blocks} blocks for request {request_id}")
        return num_allocated_blocks

    def free_blocks(self, request_id: str) -> None:
        """Free all allocated cache blocks for a given request."""

        self.cache_allocator.free_blocks(request_id, self._block_manager)

    def get_num_free_blocks(self) -> int:
        """Get the current number of unallocated blocks available for new requests."""

        return self._block_manager.num_free_blocks

    def get_num_allocated_blocks(self) -> int:
        """Get the current number of allocated cache blocks."""

        return self._block_manager.num_allocated_blocks

    def reset_peak_usage(self) -> None:
        """Reset peak cache-block usage accounting."""

        self._block_manager.reset_peak_usage()

    @property
    def peak_num_allocated_blocks(self) -> int:
        """Peak number of allocated cache blocks since the last reset."""

        return self._block_manager.peak_num_allocated_blocks

    def extend_read_and_write_indices(
        self,
        request_id: str,
        past_length: int,
        query_length: int,
        read_index: list[int] | None,
        write_index: list[int],
    ) -> None:
        """Retrieve physical cache indices for reading KV states in the cache. This method
        coordinates with the cache manager to build the complete set of read indices needed for attention computation.
        When read_index is None, the batch has no cache reads and we only compute the write indices.
        """

        write_index.extend(self.cache_allocator.get_write_indices(request_id, past_length, query_length))

        if read_index is not None:
            read_index.extend(self.cache_allocator.get_read_indices(request_id, past_length, query_length))

    def fill_block_table(
        self,
        request_id: str,
        past_length: int,
        query_length: int,
        block_table: torch.Tensor,
    ) -> None:
        """Fill the block table for a request."""

        self.cache_allocator.fill_block_table(request_id, past_length, query_length, block_table)

    def get_seqlens_k(self, past_length: int, query_length: int) -> dict[str, int]:
        """Retrieve the key sequence length for full attention."""

        return {"full_attention": past_length + query_length}

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        read_index: torch.Tensor,
        write_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update the cache with new key-value states for a specific layer, and retrieves the relevant KV states from
        the cache for attention computation.

        When the layer's read index is empty, the batch has no cache reads (all requests are non-chunked prefills): we
        only write to the cache and return the input KV states directly, skipping the index_select read-back.
        """

        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(f"layer_idx must be in [0, {self.num_layers}), got {layer_idx}")
        layer_read_index = read_index
        layer_write_index = write_index
        k_cache = self.key_cache[layer_idx]
        v_cache = self.value_cache[layer_idx]

        key_states = key_states.transpose(1, 2).squeeze(0)
        value_states = value_states.transpose(1, 2).squeeze(0)

        if layer_read_index.numel() == 0:
            k_cache.index_copy_(0, layer_write_index, key_states)
            v_cache.index_copy_(0, layer_write_index, value_states)
            return key_states, value_states

        k_cache.index_copy_(0, layer_write_index, key_states)
        v_cache.index_copy_(0, layer_write_index, value_states)
        key_states_with_cache = torch.index_select(k_cache, 0, layer_read_index)
        value_states_with_cache = torch.index_select(v_cache, 0, layer_read_index)
        return key_states_with_cache, value_states_with_cache

    def get_block_table_key(self, flash_attn_with_kvcache_fn: Any) -> str:
        """A function to get the name of the block table key for the given flash_attn_with_kvcache_fn. The function's
        signature is only inspected once. This is necessary because different version of flash have different names for
        the block table key."""

        if self._block_table_key is None:
            kwarg_names = inspect.signature(flash_attn_with_kvcache_fn).parameters.keys()
            if "block_table" in kwarg_names:
                self._block_table_key = "block_table"
            elif "page_table" in kwarg_names:
                self._block_table_key = "page_table"
            else:
                raise ValueError(
                    "flash_attn_with_kvcache_fn does not have a block_table or page_table argument: "
                    f"{inspect.signature(flash_attn_with_kvcache_fn)}"
                )
        return self._block_table_key

    def free_all_requests(self) -> None:
        """Free all blocks allocated to requests across all cache managers."""

        for request_id in list(self.cache_allocator.block_table):
            self.free_blocks(request_id)
