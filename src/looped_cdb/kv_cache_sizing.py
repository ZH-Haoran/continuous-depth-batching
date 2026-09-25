"""Size the paged KV cache from free GPU memory after model loading.

Budgeting follows SGLang v0.5.20 (94602c9c2b7c), ``_profile_available_bytes``:
https://github.com/sgl-project/sglang/blob/94602c9c2b7c/python/sglang/srt/mem_cache/kv_cache_configurator.py#L2148-L2201

The model is already loaded when these engines are constructed, so total device memory
is the baseline for the runtime reserve. The reserve also covers later CUDA graph
captures, activations, and allocations outside the KV tensors.
"""

from __future__ import annotations

import gc
import logging
import math

import torch

from looped_cdb.paged_cache_geometry import PagedKVCacheGeometry

DEFAULT_MEM_FRACTION_STATIC = 0.8
logger = logging.getLogger(__name__)


def validate_cache_size_request(num_blocks: int | None, mem_fraction_static: float | None) -> None:
    if num_blocks is not None and num_blocks <= 0:
        raise ValueError(f"num_blocks must be positive, but got {num_blocks}")
    if mem_fraction_static is not None and (not math.isfinite(mem_fraction_static) or not 0 < mem_fraction_static < 1):
        raise ValueError(f"mem_fraction_static must be finite and in (0, 1), but got {mem_fraction_static}")
    if num_blocks is not None and mem_fraction_static is not None:
        raise ValueError("Set either num_blocks or mem_fraction_static, not both")


def resolve_num_blocks(
    num_blocks: int | None,
    mem_fraction_static: float | None,
    geometry: PagedKVCacheGeometry,
    device: torch.device | str,
) -> int:
    """Return the explicit block count or fit usable blocks into the post-load budget."""

    validate_cache_size_request(num_blocks, mem_fraction_static)
    if num_blocks is not None:
        return num_blocks
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("Automatic KV cache sizing requires CUDA; set num_blocks explicitly for other devices")

    gc.collect()
    torch.cuda.empty_cache()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    fraction = DEFAULT_MEM_FRACTION_STATIC if mem_fraction_static is None else mem_fraction_static
    budget_bytes = int(free_bytes - total_bytes * (1 - fraction))
    blocks = geometry.blocks_for_budget(budget_bytes)
    if blocks <= 0:
        raise ValueError(
            f"No usable KV blocks fit: free={free_bytes} bytes, total={total_bytes} bytes, "
            f"mem_fraction_static={fraction}, bytes_per_block={geometry.bytes_per_block}. "
            "Increase mem_fraction_static or reduce the model/cache size."
        )
    gib = 1024**3
    logger.info(
        "Auto KV cache: policy=%s, slots/layer=%d, cache_layers=%d, "
        "free_before=%.2f GiB, total=%.2f GiB, static_fraction=%.3f, "
        "runtime_reserve=%.2f GiB, budget=%.2f GiB, "
        "bytes/block=%d, usable_blocks=%d, usable_tokens=%d, kv_tensor_bytes=%.2f GiB",
        geometry.kv_policy,
        geometry.kv_slots_per_layer,
        geometry.num_layers,
        free_bytes / gib,
        total_bytes / gib,
        fraction,
        total_bytes * (1 - fraction) / gib,
        budget_bytes / gib,
        geometry.bytes_per_block,
        blocks,
        blocks * geometry.block_size,
        geometry.allocation_bytes(blocks) / gib,
    )
    return blocks
