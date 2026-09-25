"""Physical storage geometry shared by continuous batching caches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch

from looped_cdb.kv_cache_policy import DEPTH_INDEXED, LoopedKvLayout, resolve_kv_slots_per_layer

if TYPE_CHECKING:
    from transformers.configuration_utils import PreTrainedConfig


class PagedCacheConfig(Protocol):
    block_size: int
    num_blocks: int | None
    max_num_batched_tokens: int

    def decode_block_table_width(self) -> int: ...


def looped_recurrent_steps(config: PreTrainedConfig) -> int | None:
    for attr in ("total_recurrent_steps", "total_ut_steps"):
        value = getattr(config, attr, None)
        if value:
            return int(value)
    return None


def looped_stage_structure(
    config: PreTrainedConfig, *, total_recurrent_steps: int | None = None
) -> tuple[int, int, int, int] | None:
    steps = total_recurrent_steps or looped_recurrent_steps(config)
    if steps is None:
        return None
    num_core_layers = getattr(config, "num_core_layers", None)
    if num_core_layers is not None:
        return (
            int(getattr(config, "num_prelude_layers", 0)),
            int(num_core_layers),
            int(getattr(config, "num_coda_layers", 0)),
            steps,
        )
    if getattr(config, "model_type", None) == "ouro":
        return (0, int(config.num_hidden_layers), 0, steps)
    return None


def infer_num_cache_layers(
    config: PreTrainedConfig,
    layer_types: list[str] | None,
    *,
    policy: str,
    slots_per_layer: int,
    total_recurrent_steps: int,
    assume_all_layers_recurrent: bool = False,
) -> int:
    structure = looped_stage_structure(config, total_recurrent_steps=total_recurrent_steps)
    if structure is None:
        if assume_all_layers_recurrent:
            return config.num_hidden_layers * slots_per_layer
        return len(layer_types) if layer_types is not None else config.num_hidden_layers
    prelude, core, coda, steps = structure
    return LoopedKvLayout(
        num_prelude_layers=prelude,
        num_core_layers=core,
        num_coda_layers=coda,
        total_recurrent_steps=steps,
        policy=policy,
        slots_per_layer=slots_per_layer,
    ).num_cache_layers


@dataclass(frozen=True)
class PagedKVCacheGeometry:
    num_layers: int
    block_size: int
    num_key_value_heads: int
    head_dim: int
    dtype: torch.dtype
    kv_policy: str
    kv_slots_per_layer: int

    extra_blocks: int = 2

    @classmethod
    def from_model(
        cls,
        config: PreTrainedConfig,
        *,
        block_size: int,
        dtype: torch.dtype,
        kv_policy: str | None = None,
        kv_slots_per_layer: int | None = None,
        total_recurrent_steps: int | None = None,
        assume_all_layers_recurrent: bool = False,
    ) -> PagedKVCacheGeometry:
        policy = kv_policy or str(getattr(config, "_cdb_kv_policy", DEPTH_INDEXED))
        steps = total_recurrent_steps or looped_recurrent_steps(config) or 1
        requested_slots = kv_slots_per_layer
        if requested_slots is None:
            requested_slots = getattr(config, "_cdb_kv_slots_per_layer", None)
        slots = resolve_kv_slots_per_layer(
            policy,
            total_recurrent_steps=steps,
            requested_slots=requested_slots,
        )
        kv_heads = getattr(config, "num_key_value_heads", None)
        head_dim = getattr(config, "head_dim", None)
        return cls(
            num_layers=infer_num_cache_layers(
                config,
                getattr(config, "layer_types", None),
                policy=policy,
                slots_per_layer=slots,
                total_recurrent_steps=steps,
                assume_all_layers_recurrent=assume_all_layers_recurrent,
            ),
            block_size=block_size,
            num_key_value_heads=kv_heads if kv_heads is not None else config.num_attention_heads,
            head_dim=head_dim if head_dim is not None else config.hidden_size // config.num_attention_heads,
            dtype=dtype,
            kv_policy=policy,
            kv_slots_per_layer=slots,
        )

    @property
    def bytes_per_block(self) -> int:
        return (
            2
            * self.num_layers
            * self.block_size
            * self.num_key_value_heads
            * self.head_dim
            * torch.empty((), dtype=self.dtype).element_size()
        )

    def allocation_bytes(self, num_blocks: int) -> int:
        return (num_blocks + self.extra_blocks) * self.bytes_per_block

    def blocks_for_budget(self, budget_bytes: int) -> int:
        return budget_bytes // self.bytes_per_block - self.extra_blocks
