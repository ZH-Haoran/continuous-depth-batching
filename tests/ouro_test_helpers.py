"""Shared test helpers for Ouro model tests."""

from __future__ import annotations

from typing import Any

from looped_cdb.models.ouro.configuration_ouro import OuroConfig


def tiny_ouro_config(**overrides: Any) -> OuroConfig:
    """Return a tiny Ouro config for CPU tests."""
    values: dict[str, Any] = {
        "vocab_size": 32,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "max_position_embeddings": 16,
        "total_ut_steps": 3,
        "layer_types": ["full_attention", "full_attention"],
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }
    values.update(overrides)
    return OuroConfig(**values)
