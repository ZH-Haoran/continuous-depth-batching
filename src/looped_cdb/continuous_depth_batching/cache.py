# Copyright 2025 The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0
# Modified for looped-model serving and continuous depth batching.
# See THIRD_PARTY_NOTICES.md and LICENSES/Apache-2.0.txt.

"""CDB-specific block-table and exit-copy operations on the shared paged cache."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from looped_cdb.continuous_batching.cache import PagedAttentionCache as BasePagedAttentionCache
from looped_cdb.paged_cache_geometry import PagedKVCacheGeometry

if TYPE_CHECKING:
    from transformers.configuration_utils import PreTrainedConfig

    from .config import ContinuousDepthBatchingConfig


class PagedAttentionCache(BasePagedAttentionCache):
    def __init__(
        self,
        config: PreTrainedConfig,
        continuous_batching_config: ContinuousDepthBatchingConfig,
        device: torch.device | str,
        dtype: torch.dtype = torch.float16,
        *,
        geometry: PagedKVCacheGeometry | None = None,
    ) -> None:
        super().__init__(config, continuous_batching_config, device, dtype, geometry=geometry)
        self._request_block_tables: dict[str, torch.Tensor] = {}
        self._host_block_rows: dict[str, tuple[torch.Tensor, int]] = {}

    def allocate_blocks(self, n_blocks: int, request_id: str, allocated_blocks: int) -> int | None:
        allocated = super().allocate_blocks(n_blocks, request_id, allocated_blocks)
        if allocated is not None:
            self._refresh_cached_block_table(request_id)
        return allocated

    def free_blocks(self, request_id: str) -> None:
        super().free_blocks(request_id)
        self._request_block_tables.pop(request_id, None)
        self._host_block_rows.pop(request_id, None)

    def fill_block_table(
        self,
        request_id: str,
        past_length: int,
        query_length: int,
        block_table: torch.Tensor,
    ) -> None:
        cached_block_table = self._request_block_tables.get(request_id)
        if cached_block_table is None:
            super().fill_block_table(request_id, past_length, query_length, block_table)
            return
        total_length = past_length + query_length
        request_blocks = self.cache_allocator.block_table.get(request_id)
        if request_blocks is None:
            raise ValueError(f"No block table found for request {request_id}")
        self.cache_allocator._validate_length(total_length, request_blocks, request_id)
        block_table.copy_(cached_block_table)

    def gather_block_table_host_rows(
        self,
        request_ids: list[str],
        total_lengths: list[int],
        out: torch.Tensor,
    ) -> None:
        """Stack validated cached block-table rows for a decode launch."""

        if out.device.type != "cpu":
            raise ValueError("gather_block_table_host_rows expects a CPU tensor")
        rows = []
        for request_id, total_length in zip(request_ids, total_lengths, strict=True):
            entry = self._host_block_rows.get(request_id)
            if entry is None:
                raise ValueError(
                    f"No cached block-table row for request {request_id} (no allocation, or the request "
                    f"needs more than max_blocks_per_request={self.max_blocks_per_request} blocks)"
                )
            row, num_blocks = entry
            if total_length > num_blocks * self.block_size:
                raise ValueError(
                    f"Request {request_id} has {num_blocks} blocks allocated "
                    f"({num_blocks * self.block_size} tokens) but needs {total_length}"
                )
            rows.append(row)
        torch.stack(rows, dim=0, out=out)

    def get_token_indices(self, request_id: str, token_position: int) -> list[torch.Tensor]:
        if token_position < 0:
            raise ValueError(f"token_position must be non-negative, got {token_position}")
        token_indices = self.cache_allocator.get_write_indices(request_id, token_position, 1)
        return [torch.tensor(token_indices, dtype=torch.long, device=self.device)]

    def gather_token_kv_rows(self, positions: list[tuple[str, int]]) -> torch.Tensor:
        rows = [
            self.cache_allocator.get_write_indices(request_id, position, 1)[0] for request_id, position in positions
        ]
        return torch.tensor(rows, dtype=torch.long, device=self.device)

    def copy_kv_rows(self, source_layer_idx: int, target_layer_idxs: list[int], rows: torch.Tensor) -> None:
        for layer_idx in (source_layer_idx, *target_layer_idxs):
            if not 0 <= layer_idx < self.num_layers:
                raise IndexError(f"layer_idx must be in [0, {self.num_layers}), got {layer_idx}")
        source_keys = self.key_cache[source_layer_idx].index_select(0, rows)
        source_values = self.value_cache[source_layer_idx].index_select(0, rows)
        for target_layer_idx in target_layer_idxs:
            self.key_cache[target_layer_idx].index_copy_(0, rows, source_keys)
            self.value_cache[target_layer_idx].index_copy_(0, rows, source_values)

    def _refresh_cached_block_table(self, request_id: str) -> None:
        if self.max_blocks_per_request <= 0:
            return
        request_blocks = self.cache_allocator.block_table.get(request_id)
        if request_blocks is None:
            return
        if len(request_blocks) > self.max_blocks_per_request:
            self._request_block_tables.pop(request_id, None)
            self._host_block_rows.pop(request_id, None)
            return

        host_row = torch.full((self.max_blocks_per_request,), -1, dtype=torch.int32)
        if request_blocks:
            host_row[: len(request_blocks)] = torch.tensor(request_blocks, dtype=torch.int32)
        self._host_block_rows[request_id] = (host_row, len(request_blocks))

        cached_block_table = self._request_block_tables.get(request_id)
        if cached_block_table is None:
            cached_block_table = torch.full(
                (self.max_blocks_per_request,),
                -1,
                dtype=torch.int32,
                device=self.device,
            )
            torch._dynamo.mark_static_address(cached_block_table)
            self._request_block_tables[request_id] = cached_block_table
        else:
            cached_block_table.fill_(-1)
        if request_blocks:
            cached_block_table[: len(request_blocks)].copy_(host_row[: len(request_blocks)])
