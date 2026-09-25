# Copyright 2022 The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0
# Modified for looped-model serving and continuous depth batching.
# See THIRD_PARTY_NOTICES.md and LICENSES/Apache-2.0.txt.

"""Configuration for native continuous depth batching.

Reference:
https://github.com/huggingface/transformers/blob/main/src/transformers/generation/configuration_utils.py
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

from looped_cdb.kv_cache_policy import (
    KV_CACHE_POLICIES,
    KvCachePolicy,
    copies_exit_kv,
    resolve_kv_slots_per_layer,
)
from looped_cdb.kv_cache_sizing import validate_cache_size_request
from looped_cdb.utils import (
    DEFAULT_MAX_NUM_SEQS,
    KVPressureMode,
    validate_kv_pressure_config,
)

if TYPE_CHECKING:
    from transformers.configuration_utils import PreTrainedConfig


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
class ContinuousDepthBatchingConfig:
    """Configuration for native continuous depth batching."""

    # FlashAttention 2 requires a multiple of 256. FlashAttention 3 accepts this smaller default.
    block_size: int = 16

    # Usable KV blocks. None sizes the cache from free GPU memory after loading the model.
    num_blocks: int | None = None
    # Fraction of device memory available to weights and static allocations; unset uses 0.8.
    mem_fraction_static: float | None = None
    # Query-token budget for prefill and the static decode buffers.
    max_num_batched_tokens: int = 8192

    # Stop admitting prefill below this fraction of free KV blocks.
    safety_margin: float = 0.2

    # Discover replay length finishes at consume time to model asynchronous EOS handling.
    replay_eos_finishes: bool = False

    # Maximum prompt plus generation length and the basis for per-request KV sizing.
    max_model_len: int = 16384

    # Runs the engine's async paths (async coda readback, delayed gate readout, on-device token
    # re-entry), hiding host work behind the device. False runs every stage synchronously.
    use_async_batching: bool = True

    # Graphs the decode fast path.
    use_cuda_graph: bool = True

    # Also graphs the varlen (prefill) forward piecewise, per padded token bucket; attention stays eager.
    use_cuda_graph_prefill: bool = True

    # Number of recurrent block applications. CDB requires an explicit value.
    max_recurrent_steps: int | None = None

    # Online early-exit policy. ``exit_threshold`` is interpreted by the
    # adapter-declared policy: Ouro accumulates per-step hazard logits, while
    # Huginn compares each latent-difference score directly. ``synthetic_exit_replay``
    # instead drives exits from a per-request recorded schedule attached to each
    # ``RequestState``. The two are mutually exclusive.
    exit_threshold: float | None = None
    synthetic_exit_replay: bool = False
    min_recurrent_steps: int = 1
    delay_gate_consumption: bool = True

    # Pending coda tokens required for an ordinary launch. Idle ticks flush smaller batches.
    min_coda_batch_size: int = 1

    # Maximum resident requests and recurrent or coda launch width.
    max_num_seqs: int = DEFAULT_MAX_NUM_SEQS

    # Open resident slots required before admitting a prefill batch.
    min_free_slots: int | None = None

    # Refill freed recurrent slots. False runs fixed cohorts in lockstep.
    refill: bool = True

    # Static recurrent KV policy.
    kv_policy: KvCachePolicy = "single"
    kv_slots_per_layer: int | None = None

    # KV pressure policy: raise, reserve capacity, recompute a victim, or offload a victim.
    kv_pressure_mode: KVPressureMode = "recompute"

    # Pinned CPU swap-pool budget (GiB) for the "offload" pressure mode; unused by the other modes.
    # Must be set (> 0) when ``kv_pressure_mode == "offload"`` and left ``None`` otherwise.
    cpu_offload_space: float | None = None

    def decode_block_table_width(self) -> int:
        """Return ``ceil(max_model_len / block_size)`` for decode block tables."""

        return -(-self.max_model_len // self.block_size)

    def get_resolved(
        self,
        config: PreTrainedConfig,
    ) -> ContinuousDepthBatchingConfig:
        """Return a deep-copied, validated, and fully resolved config."""

        cdb_config = deepcopy(self)
        cdb_config.validate(config)
        cdb_config.normalize_derived_values()
        cdb_config.validate_resolved()
        return cdb_config

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
        if self.max_model_len <= 0:
            raise ValueError(f"max_model_len must be positive, but got {self.max_model_len}")
        if self.max_recurrent_steps is None:
            raise ValueError("max_recurrent_steps must be explicitly set for CDB")
        if self.max_recurrent_steps <= 1:
            raise ValueError(f"max_recurrent_steps must be greater than 1, but got {self.max_recurrent_steps}")
        if self.max_num_seqs <= 0:
            raise ValueError(f"max_num_seqs must be positive, but got {self.max_num_seqs}")
        if self.min_free_slots is not None and self.min_free_slots <= 0:
            raise ValueError(f"min_free_slots must be positive, but got {self.min_free_slots}")
        if self.min_coda_batch_size <= 0:
            raise ValueError(f"min_coda_batch_size must be positive, but got {self.min_coda_batch_size}")
        if self.min_coda_batch_size > self.max_num_seqs:
            # The coda queue is fed from the resident set, so a minimum above it can never be reached:
            # the batch would be held back on every ordinary tick and released only by the drain flush.
            # Equality is the limit point of that same behaviour and stays legal; it reproduces the
            # no-refill wave schedule.
            raise ValueError(
                f"min_coda_batch_size must not exceed max_num_seqs, but got "
                f"{self.min_coda_batch_size} > {self.max_num_seqs}"
            )
        if self.replay_eos_finishes and not self.use_async_batching:
            # Eager re-entry, the path that stages the doomed successor, only exists on the async
            # path; without it the flag would change nothing while the row still records it.
            raise ValueError("replay_eos_finishes requires use_async_batching")
        if not (0 <= self.safety_margin < 1):
            raise ValueError(f"safety_margin must be in [0, 1), but got {self.safety_margin}")
        if self.kv_policy not in KV_CACHE_POLICIES:
            raise ValueError(f"Unsupported kv_policy={self.kv_policy!r}; expected one of {KV_CACHE_POLICIES}")
        if copies_exit_kv(self.kv_policy) and self.kv_pressure_mode == "recompute":
            raise ValueError(
                "kv_policy='last_exited' cannot run with kv_pressure_mode='recompute': recompute-preemption "
                "re-prefills the whole sequence at full depth, replacing previously exited tokens' "
                "exit-step KV with full-depth KV. Use 'reserve', 'offload', or 'none', which preserve KV."
            )
        if self.min_recurrent_steps <= 0:
            raise ValueError(f"min_recurrent_steps must be positive, but got {self.min_recurrent_steps}")
        validate_kv_pressure_config(self.kv_pressure_mode, self.cpu_offload_space)
        active_exit_policies = sum((self.exit_threshold is not None, self.synthetic_exit_replay))
        if active_exit_policies > 1:
            raise ValueError("Only one CDB exit policy can be active")
        layer_types = getattr(config, "layer_types", None)
        has_sliding_layers = layer_types is not None and any(
            layer_type != "full_attention" for layer_type in layer_types
        )
        if getattr(config, "sliding_window", None) is not None or has_sliding_layers:
            raise NotImplementedError("Native continuous batching currently supports full-attention models only")

    def normalize_derived_values(self) -> None:
        """Resolve CDB settings that are derived from explicit config values."""

        assert self.max_recurrent_steps is not None
        self.kv_slots_per_layer = resolve_kv_slots_per_layer(
            self.kv_policy,
            total_recurrent_steps=self.max_recurrent_steps,
            requested_slots=self.kv_slots_per_layer,
        )

    def validate_resolved(self) -> None:
        """Validate that sentinel values were resolved before components are created."""

        if self.max_recurrent_steps is None:
            raise ValueError("max_recurrent_steps must be resolved")
        if self.max_recurrent_steps <= 1:
            raise ValueError("max_recurrent_steps must be greater than 1")
        if self.min_recurrent_steps > self.max_recurrent_steps:
            raise ValueError(
                f"min_recurrent_steps={self.min_recurrent_steps} exceeds max_recurrent_steps={self.max_recurrent_steps}"
            )
        self._validate_kv_policy_exit_safety()

    def _validate_kv_policy_exit_safety(self) -> None:
        assert self.max_recurrent_steps is not None
        if copies_exit_kv(self.kv_policy):
            # Copy-on-exit routing fills the deeper slots at exit time, so any exit depth is safe.
            return
        slots_per_layer = resolve_kv_slots_per_layer(
            self.kv_policy,
            total_recurrent_steps=self.max_recurrent_steps,
            requested_slots=self.kv_slots_per_layer,
        )
        if self.exit_threshold is not None and self.min_recurrent_steps < slots_per_layer:
            raise ValueError(
                "kv_slots_per_layer must be <= min_recurrent_steps for gated early exit, "
                f"got kv_slots_per_layer={slots_per_layer}, min_recurrent_steps={self.min_recurrent_steps}"
            )
        # Per-request synthetic-replay depths are only known at generate time, so
        # their static-KV safety is validated in ``_validate_exit_depths`` there.
        # The default single-slot policy makes any exit depth safe.
