# Copyright 2026 The HuggingFace Inc. team
# SPDX-License-Identifier: Apache-2.0
# Modified for looped-model serving and continuous depth batching.
# See THIRD_PARTY_NOTICES.md and LICENSES/Apache-2.0.txt.

"""Utility helpers for native continuous batching.

Reference:
https://github.com/huggingface/transformers/blob/main/src/transformers/generation/continuous_batching/utils.py
"""

import gc
from contextlib import contextmanager
from math import ceil
from typing import Any

import torch

from .requests import FutureRequestState, RequestState, RequestStatus


@contextmanager
def gc_paused():
    """Pause Python garbage collection across a CUDA graph capture.

    While a capture is registered with the caching allocator, any finalizer the
    cycle collector happens to run that reaches ``emptyCache()`` - for example a
    ``torch.cuda.MemPool`` destructor, or a ``__del__`` calling
    ``torch.cuda.empty_cache()`` - trips the allocator's
    ``captures_underway.empty()`` assert and terminates the process from a
    noexcept destructor (SIGABRT inside whatever op happened to be running).
    Collecting up front keeps the pause short and the capture-time heap quiet.
    """

    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


class CudaGraphBuffer:
    """Captured CUDA graphs by shape key; every graph stays resident for the runner's life."""

    def __init__(self) -> None:
        self._storage: dict[tuple[int, ...], torch.cuda.CUDAGraph] = {}

    def __del__(self) -> None:
        for graph in self._storage.values():
            graph.reset()

    def __len__(self) -> int:
        return len(self._storage)

    def get_graph(self, key: tuple[int, ...]) -> torch.cuda.CUDAGraph | None:
        return self._storage.get(key)

    def set_graph(self, key: tuple[int, ...], graph: torch.cuda.CUDAGraph) -> None:
        self._storage[key] = graph


def decode_graph_buckets(max_num_seqs: int) -> list[int]:
    """Decode batch sizes a launch pads to, one graph each; the cap is always the last bucket.

    SGLang's decode capture list: every small batch, then steps of 8 to 256, 16 to 512 and 32
    beyond, so a launch carries at most a few percent of padded rows once batches are wide enough
    for padding to cost compute.
    """

    if max_num_seqs <= 0:
        raise ValueError(f"max_num_seqs must be positive, got {max_num_seqs}")
    sizes = [1, 2, 4, 8, 12, *range(16, 257, 8), *range(272, 512, 16), *range(512, max_num_seqs + 1, 32)]
    return sorted({size for size in sizes if size <= max_num_seqs} | {max_num_seqs})


def pad_to_bucket(num_tokens: int, buckets: list[int]) -> int:
    """The smallest bucket holding ``num_tokens``."""

    for bucket in buckets:
        if bucket >= num_tokens:
            return bucket
    raise ValueError(f"{num_tokens} tokens exceed the widest bucket {buckets[-1] if buckets else None}")


def create_warmup_future_states(
    num: int,
    status: RequestStatus,
    num_q_tokens: int,
    max_kv_read: int,
    cache: Any,
) -> list[FutureRequestState]:
    """Create fake request states for continuous batching warmup."""

    return create_warmup_states([num_q_tokens] * num, status, max_kv_read, cache)


def create_warmup_states(
    query_lengths: list[int],
    status: RequestStatus,
    max_kv_read: int,
    cache: Any,
) -> list[FutureRequestState]:
    """Create one fake request per query length, each with ``max_kv_read`` cached tokens, allocating its blocks.

    Stops at the first request whose blocks do not fit, returning the ones allocated so far.
    """

    future_states = []
    for index, num_q_tokens in enumerate(query_lengths):
        total_tokens = num_q_tokens + max_kv_read
        blocks_needed = ceil(total_tokens / cache.block_size)
        state = RequestState(
            request_id=f"__warmup_{status.name}_{index}__", initial_tokens=[0] * total_tokens, max_new_tokens=1
        )
        state._status = status
        state.tokens_to_process = [0] * num_q_tokens
        state.position_offset = max_kv_read
        allocated = cache.allocate_blocks(blocks_needed, state.request_id, 0)
        if allocated is None:
            return future_states
        state.allocated_blocks = allocated
        future_states.append(FutureRequestState(state, has_new_token=True, query_length=num_q_tokens))
    return future_states


def aligned_divide(value: int, divisor: int, alignment: int) -> int:
    """Divide and round up to a multiple of ``alignment``."""

    divided = (value + divisor - 1) // divisor
    return ((divided + alignment - 1) // alignment) * alignment
