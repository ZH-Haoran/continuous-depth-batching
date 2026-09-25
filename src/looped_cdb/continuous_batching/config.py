# Copyright 2022 The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0
# Modified for looped-model serving and continuous depth batching.
# See THIRD_PARTY_NOTICES.md and LICENSES/Apache-2.0.txt.

"""Configuration for native continuous batching.

Reference:
https://github.com/huggingface/transformers/blob/main/src/transformers/generation/configuration_utils.py
"""

from copy import deepcopy
from dataclasses import dataclass

from transformers.configuration_utils import PreTrainedConfig

from looped_cdb.kv_cache_sizing import validate_cache_size_request
from looped_cdb.utils import (
    DEFAULT_MAX_NUM_SEQS,
    KVPressureMode,
    validate_kv_pressure_config,
)


def is_supported_attention_implementation(config: PreTrainedConfig) -> bool:
    """Return whether the model config requests our supported attention backend."""

    attn_implementation = getattr(config, "_attn_implementation", None)
    if not isinstance(attn_implementation, str):
        return False
    prefix = "paged|"
    if not attn_implementation.startswith(prefix):
        return False
    flash_attention_backend = attn_implementation.removeprefix(prefix)
    return flash_attention_backend == "flash_attention" or (
        flash_attention_backend.startswith("flash_attention_")
        and flash_attention_backend.removeprefix("flash_attention_").isdigit()
    )


@dataclass
class ContinuousBatchingConfig:
    """Configuration for native continuous batching."""

    # FlashAttention 2 requires a multiple of 256. FlashAttention 3 accepts this smaller default.
    block_size: int = 16

    # Usable KV blocks. None sizes the cache from free GPU memory after loading the model.
    num_blocks: int | None = None
    # Fraction of device memory available to weights and static allocations; unset uses 0.8.
    mem_fraction_static: float | None = None
    # Query-token budget for prefill and the static decode buffers.
    max_num_batched_tokens: int = 8192

    # Maximum resident requests and decode launch width.
    max_num_seqs: int = DEFAULT_MAX_NUM_SEQS

    # Open resident slots required before admitting a prefill batch.
    min_free_slots: int | None = None

    # Stop admitting prefill below this fraction of free KV blocks.
    safety_margin: float = 0.2

    # Discover replay length finishes at consume time to model asynchronous EOS handling.
    replay_eos_finishes: bool = False

    # KV pressure policy: raise, reserve capacity, recompute a victim, or offload a victim.
    kv_pressure_mode: KVPressureMode = "recompute"

    # Pinned CPU swap-pool budget (GiB) for the "offload" pressure mode; unused by the other modes.
    # Must be set (> 0) when ``kv_pressure_mode == "offload"`` and left ``None`` otherwise.
    cpu_offload_space: float | None = None

    # Maximum prompt plus generation length and the basis for per-request KV sizing.
    max_model_len: int = 16384

    # Runs the host one batch ahead of the device (vLLM-style async scheduling), hiding the loop's host
    # work behind the device. False runs the same machinery serialized; tokens are identical either way.
    use_async_batching: bool = True

    # Graphs the decode fast path.
    use_cuda_graph: bool = True

    # Also graphs the varlen (prefill) forward piecewise, per padded token bucket; attention stays eager.
    use_cuda_graph_prefill: bool = True

    def decode_block_table_width(self) -> int:
        """Return ``ceil(max_model_len / block_size)`` for decode block tables."""

        return -(-self.max_model_len // self.block_size)

    def get_resolved(
        self,
        config: PreTrainedConfig,
    ) -> "ContinuousBatchingConfig":
        """Return a deep-copied, validated, and fully resolved config."""

        cb_config = deepcopy(self)
        cb_config.validate(config)
        return cb_config

    def validate(self, config: PreTrainedConfig) -> None:
        """Validate settings and model features."""

        if not is_supported_attention_implementation(config):
            attn_implementation = getattr(config, "_attn_implementation", None)
            raise NotImplementedError(
                "Native continuous batching currently supports FlashAttention-backed paged attention only, "
                f"but got attn_implementation={attn_implementation!r}"
            )
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, but got {self.block_size}")
        backend = (getattr(config, "_attn_implementation", "") or "").removeprefix("paged|")
        if backend == "flash_attention_2" and self.block_size % 256 != 0:
            raise ValueError(
                "FlashAttention-2's paged flash_attn_with_kvcache requires block_size to be a multiple of 256, "
                f"but got block_size={self.block_size}. Use paged|flash_attention_3 (which allows arbitrary block "
                "sizes) or make block_size a multiple of 256."
            )
        validate_cache_size_request(self.num_blocks, self.mem_fraction_static)
        if self.max_num_batched_tokens <= 0:
            raise ValueError(f"max_num_batched_tokens must be positive, but got {self.max_num_batched_tokens}")
        if self.max_num_seqs <= 0:
            raise ValueError(f"max_num_seqs must be positive, but got {self.max_num_seqs}")
        if self.min_free_slots is not None and self.min_free_slots <= 0:
            raise ValueError(f"min_free_slots must be positive, but got {self.min_free_slots}")
        if not (0 <= self.safety_margin < 1):
            raise ValueError(f"safety_margin must be in [0, 1), but got {self.safety_margin}")
        if self.max_model_len <= 0:
            raise ValueError(f"max_model_len must be positive, but got {self.max_model_len}")
        if self.replay_eos_finishes and not self.use_async_batching:
            # The synchronous loop consumes every result before scheduling more, so no token is ever
            # in flight when the finish prediction runs and the flag would change nothing while the
            # row still records it.
            raise ValueError("replay_eos_finishes requires use_async_batching")
        validate_kv_pressure_config(self.kv_pressure_mode, self.cpu_offload_space)
        layer_types = getattr(config, "layer_types", None)
        has_sliding_layers = layer_types is not None and any(
            layer_type != "full_attention" for layer_type in layer_types
        )
        if getattr(config, "sliding_window", None) is not None or has_sliding_layers:
            raise NotImplementedError("Native continuous batching currently supports full-attention models only")
