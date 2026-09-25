# Copyright 2026 The HuggingFace Inc. team
# SPDX-License-Identifier: Apache-2.0
# Modified for looped-model serving and continuous depth batching.
# See THIRD_PARTY_NOTICES.md and LICENSES/Apache-2.0.txt.

"""KV-cache preemption shared by the CB and CDB engines.

Ported from transformers ``generation/continuous_batching/offloading_manager.py``

Requests use a pinned CPU swap pool when space is available.
Otherwise they restart from a prompt containing their generated tokens.
"""

from __future__ import annotations

import logging
from collections import deque
from contextlib import nullcontext
from typing import Any

import torch

logger = logging.getLogger("ContinuousBatchingLogger")


class OffloadingManager:
    """Manages request preemption (CPU offload + soft-reset fallback) and restoration.

    Owns a static pinned CPU swap pool mirroring the GPU cache's per-layer block layout, performs
    the GPU<->CPU block copies, and picks between offloading and soft reset based on pool capacity.
    """

    def __init__(
        self,
        cache: Any,
        scheduler: Any,
        cpu_offload_space_gib: float | None,
        *,
        compute_stream: torch.cuda.Stream | None = None,
        pin_memory: bool = True,
        allow_recompute_fallback: bool = True,
    ) -> None:
        self.cache = cache
        self.scheduler = scheduler
        # A soft reset re-prefills the victim at full depth, which is only equivalent to the offload
        # swap when full-depth KV is what the cache would hold anyway. A policy that rewrites KV on
        # early exit (``last_exited``) breaks that equivalence, so its engine disables the fallback
        # and a full pool becomes a hard error instead of a silent change of KV semantics.
        self._allow_recompute_fallback = allow_recompute_fallback
        # All offloading transfers run on the compute stream (stream-ordered with generation), so a
        # preempt/restore copy never races the forward pass that reads or writes the same blocks.
        self._compute_stream = compute_stream
        # Pinned host blocks let the GPU<->CPU copies DMA asynchronously. Pinning requires CUDA, so the
        # engines derive this from the cache device (``cache.device.type == "cuda"``): a CUDA engine
        # pins, and a CPU-device engine (only reachable in tests, which exercise the preempt/restore
        # bookkeeping and block copies, not real DMA) falls back to an ordinary pageable pool.
        self._pin_memory = pin_memory

        # request_id -> the CPU pool block ids holding its offloaded KV, in GPU-block order.
        self._request_id_to_cpu_blocks: dict[str, list[int]] = {}

        # Per-generation preemption counters (zeroed by ``reset``). ``offload`` counts victims swapped to
        # the CPU pool; ``recompute`` counts victims soft-reset (including the pool-full fallback);
        # ``restores`` counts offloaded requests copied back to the GPU.
        self.num_offload_preemptions = 0
        self.num_recompute_preemptions = 0
        self.num_restores = 0

        self._num_cpu_blocks = self._compute_num_cpu_blocks(cpu_offload_space_gib)
        self._cpu_key_cache: list[torch.Tensor] = []
        self._cpu_value_cache: list[torch.Tensor] = []
        self._gpu_key_views: list[torch.Tensor] = []
        self._gpu_value_views: list[torch.Tensor] = []
        self._free_cpu_blocks: deque[int] = deque()

        if self._num_cpu_blocks == 0:
            if cpu_offload_space_gib:
                logger.warning(
                    f"cpu_offload_space={cpu_offload_space_gib:.2f} GiB is too small for even one block; "
                    "preemption will use soft reset (recompute) only."
                )
            return

        # Per-layer pinned CPU blocks, shaped like one GPU block: (num_cpu_blocks, block_size, heads, dim).
        block_shape = (self._num_cpu_blocks, cache.block_size, cache.num_key_value_heads, cache.head_dim)
        for _ in range(len(cache.key_cache)):
            self._cpu_key_cache.append(torch.empty(block_shape, dtype=cache.dtype, pin_memory=self._pin_memory))
            self._cpu_value_cache.append(torch.empty(block_shape, dtype=cache.dtype, pin_memory=self._pin_memory))

        # Pre-view the flat GPU cache tensors as block-shaped so the hot copy paths avoid per-op .view().
        # Each layer tensor is ((num_blocks + 2) * block_size, heads, dim); block b occupies rows
        # [b * block_size, (b + 1) * block_size), so a contiguous reshape indexes blocks directly.
        self._gpu_key_views = [
            k.view(-1, cache.block_size, cache.num_key_value_heads, cache.head_dim) for k in cache.key_cache
        ]
        self._gpu_value_views = [
            v.view(-1, cache.block_size, cache.num_key_value_heads, cache.head_dim) for v in cache.value_cache
        ]

        self._free_cpu_blocks = deque(range(self._num_cpu_blocks))
        size_gib = self._cpu_key_cache[0].numel() * self._cpu_key_cache[0].element_size() * 2 * len(self._cpu_key_cache)
        logger.info(f"CPU swap pool initialized: {self._num_cpu_blocks} blocks ({size_gib / 1024**3:.2f} GiB pinned)")

    @property
    def offloading_enabled(self) -> bool:
        """Whether a CPU swap pool exists (else preemption is soft-reset only)."""

        return self._num_cpu_blocks > 0

    @property
    def num_preemptions(self) -> int:
        """Total requests preempted this generation (offload swaps plus soft resets)."""

        return self.num_offload_preemptions + self.num_recompute_preemptions

    def preemption_stats(self) -> dict[str, int]:
        """Per-generation preemption counters for benchmark reporting."""

        return {
            "preemptions": self.num_preemptions,
            "offload_preemptions": self.num_offload_preemptions,
            "recompute_preemptions": self.num_recompute_preemptions,
            "restores": self.num_restores,
        }

    def _compute_num_cpu_blocks(self, cpu_offload_space_gib: float | None) -> int:
        """Number of blocks that fit in a ``cpu_offload_space_gib`` pinned pool (0 when unset)."""

        if not cpu_offload_space_gib or cpu_offload_space_gib <= 0:
            return 0
        bytes_per_block = (
            2  # key and value
            * len(self.cache.key_cache)
            * self.cache.block_size
            * self.cache.num_key_value_heads
            * self.cache.head_dim
            * self.cache.dtype.itemsize
        )
        if bytes_per_block == 0:
            raise ValueError("bytes per KV block is 0; cannot size the CPU swap pool")
        return int(cpu_offload_space_gib * 1024**3) // bytes_per_block

    def _stream_ctx(self):
        """Run enclosed ops on the compute stream, or a no-op context when none is set."""

        return torch.cuda.stream(self._compute_stream) if self._compute_stream is not None else nullcontext()

    def offload_one_request(self) -> None:
        """Preempt one active request to free GPU cache. Tries CPU offload first, else soft reset.

        The victim is re-queued and further fresh admissions are blocked until a request finishes, so
        the batch drains and refills instead of thrashing. The scheduler chooses the least costly victim.
        Used by the batch-synchronous CB engine, where any active request is at a clean token boundary.
        """

        request_id, state = self.scheduler.pop_request_to_evict()
        self._preempt(request_id, state)

    def preempt_request(self, request_id: str, state: Any) -> None:
        """Preempt a specific request that is already at a clean token boundary.

        Used by the pipelined CDB engine for self-preemption: the request that cannot allocate its next
        decode token yields its own KV. Choosing the failing request as the victim guarantees the victim
        is not mid-recurrence (its previous token's recurrence has fully completed), which a scheduler
        general scheduler pick could not. The caller must have removed ``state`` from any in-flight queue.
        """

        self._preempt(request_id, state)

    def _preempt(self, request_id: str, state: Any) -> None:
        """Offload (or soft-reset) one request's KV, free its GPU blocks, and re-queue it draining."""

        logger.info(
            f"Preempting request {request_id}: {len(state.initial_tokens)} prompt + "
            f"{len(state.generated_tokens)} generated tokens."
        )

        if self._offload_to_cpu(request_id, state):
            state.prepare_for_offload_requeue()
            new_state = state
            self.num_offload_preemptions += 1
        else:
            if not self._allow_recompute_fallback:
                needed = len(self.cache.cache_allocator.block_table.get(request_id, []))
                raise RuntimeError(
                    f"CPU swap pool cannot hold request {request_id} ({needed} blocks needed, "
                    f"{len(self._free_cpu_blocks)} free) and the soft-reset fallback is disabled: "
                    "re-prefilling at full depth would replace the KV this cache policy rewrote on "
                    "early exit. Increase cpu_offload_space or reduce load."
                )
            new_state = state.create_equivalent_initial_request()
            self.num_recompute_preemptions += 1

        self.scheduler.finish_request(request_id)
        self.scheduler.add_waiting_request(new_state)
        self.scheduler.block_new_requests = True

    def restore_scheduled_requests(self, requests_in_batch: list[Any]) -> None:
        """Copy KV back from CPU for any offloaded requests in the scheduled batch (before compute)."""

        all_cpu_indices: list[int] = []
        all_gpu_indices: list[int] = []
        for future_state in requests_in_batch:
            state = future_state.state
            if not state.is_cpu_offloaded:
                continue
            cpu_indices = self._request_id_to_cpu_blocks.pop(state.request_id)
            gpu_blocks = self.cache.cache_allocator.block_table.get(state.request_id, [])
            # The re-scheduled request may hold extra freshly-allocated blocks for the next token;
            # restore only the blocks that were offloaded, keeping GPU-block order aligned with CPU.
            all_cpu_indices.extend(cpu_indices)
            all_gpu_indices.extend(gpu_blocks[: len(cpu_indices)])
            state.is_cpu_offloaded = False
            state.allocated_blocks = len(gpu_blocks)
            self.num_restores += 1

        if not all_cpu_indices:
            return

        # Copy each block directly between its GPU view and its pinned CPU view. Gathering through an
        # intermediate ``index_select(...).to(device)`` would land the H2D transfer in a fresh pageable
        # temporary, taking the pinned pool off the DMA path (no async overlap, an extra host copy). The
        # per-block ``copy_`` keeps the pinned block as the host end of every transfer.
        # ``all_gpu_indices`` was truncated to ``len(cpu_indices)`` per request, so the two index lists
        # must be equal length; ``strict=True`` turns a future accounting slip into a loud error instead
        # of silently dropping a block's KV.
        with self._stream_ctx():
            for cpu_k, gpu_k in zip(self._cpu_key_cache, self._gpu_key_views):
                for cpu_b, gpu_b in zip(all_cpu_indices, all_gpu_indices, strict=True):
                    gpu_k[gpu_b].copy_(cpu_k[cpu_b], non_blocking=True)
            for cpu_v, gpu_v in zip(self._cpu_value_cache, self._gpu_value_views):
                for cpu_b, gpu_b in zip(all_cpu_indices, all_gpu_indices, strict=True):
                    gpu_v[gpu_b].copy_(cpu_v[cpu_b], non_blocking=True)
        # Returning the pool blocks now is safe only because every pool access is stream-ordered on the
        # compute stream: the non-blocking H2D reads above are enqueued before the next user of these
        # blocks. Moving the transfers to a dedicated copy stream (for real overlap) would make this a
        # race - the free would have to wait on a recorded event on that stream first.
        self._free_cpu_blocks.extend(all_cpu_indices)

    def _offload_to_cpu(self, request_id: str, state: Any) -> bool:
        """Copy a request's KV blocks GPU->CPU and free its GPU blocks. False if the pool is full."""

        if state.is_cpu_offloaded:
            # Re-offloading an already-offloaded request would overwrite its pool-block record and leak
            # the previous blocks. No reachable path reaches here today (a restored request always
            # allocates before it can be re-preempted); the guard keeps it that way.
            raise RuntimeError(f"Request {request_id} is already offloaded; refusing to offload it twice.")

        gpu_indices = self.cache.cache_allocator.block_table.get(request_id, [])
        total = len(gpu_indices)
        if total == 0 or len(self._free_cpu_blocks) < total:
            return False

        cpu_indices = [self._free_cpu_blocks.popleft() for _ in range(total)]
        with self._stream_ctx():
            for cpu_k, gpu_k in zip(self._cpu_key_cache, self._gpu_key_views):
                for cpu_b, gpu_b in zip(cpu_indices, gpu_indices, strict=True):
                    cpu_k[cpu_b].copy_(gpu_k[gpu_b], non_blocking=True)
            for cpu_v, gpu_v in zip(self._cpu_value_cache, self._gpu_value_views):
                for cpu_b, gpu_b in zip(cpu_indices, gpu_indices, strict=True):
                    cpu_v[cpu_b].copy_(gpu_v[gpu_b], non_blocking=True)

        # The victim's GPU blocks are freed by the caller's ``scheduler.finish_request``; the copy
        # above ran while they were still allocated, so their contents are now safe on the host.
        self._request_id_to_cpu_blocks[request_id] = cpu_indices
        state.is_cpu_offloaded = True
        return True

    def free_request_cpu_cache(self, state: Any) -> None:
        """Return a single request's CPU blocks to the pool without copying (e.g. on cancellation)."""

        if state.is_cpu_offloaded:
            self._free_cpu_blocks.extend(self._request_id_to_cpu_blocks.pop(state.request_id, []))
            state.is_cpu_offloaded = False

    def free_all_waiting_cpu_caches(self) -> None:
        """Return every waiting offloaded request's CPU blocks to the pool (e.g. on reset)."""

        for state in self.scheduler.waiting_requests.values():
            self.free_request_cpu_cache(state)

    def reset(self) -> None:
        """Drop all offloading bookkeeping for a new generation session."""

        self._request_id_to_cpu_blocks.clear()
        self._free_cpu_blocks = deque(range(self._num_cpu_blocks))
        self.num_offload_preemptions = 0
        self.num_recompute_preemptions = 0
        self.num_restores = 0
