from __future__ import annotations

from typing import Any

import pytest
import torch
from ouro_test_helpers import tiny_ouro_config

import looped_cdb.models.ouro.modeling_ouro as modeling_ouro
from looped_cdb.models.ouro.modeling_ouro import OuroAttention, OuroForCausalLM


def test_paged_attention_uses_recurrent_step_virtual_layer_index(monkeypatch) -> None:
    config = tiny_ouro_config()
    config._attn_implementation = "paged|eager"
    attention = OuroAttention(config=config, layer_idx=1)
    hidden_states = torch.randn(1, 3, config.hidden_size)
    cos = torch.ones(1, 3, attention.head_dim)
    sin = torch.zeros(1, 3, attention.head_dim)
    captured: dict[str, int] = {}

    def fake_attention_forward(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        del key, value, attention_mask, scaling, kwargs
        captured["layer_idx"] = module.layer_idx
        return torch.zeros(
            query.shape[0],
            query.shape[2],
            query.shape[1],
            query.shape[3],
            dtype=query.dtype,
            device=query.device,
        ), None

    monkeypatch.setattr(modeling_ouro.ALL_ATTENTION_FUNCTIONS, "get_interface", lambda *_args: fake_attention_forward)

    attention(
        hidden_states=hidden_states,
        position_embeddings=(cos, sin),
        attention_mask=None,
        current_ut=2,
        cache=object(),
        read_index=[],
        write_index=[],
    )

    assert captured["layer_idx"] == 2 * config.num_hidden_layers + 1


def test_paged_attention_uses_physical_layer_index_for_single_slot_kv(monkeypatch) -> None:
    config = tiny_ouro_config()
    config._attn_implementation = "paged|eager"
    attention = OuroAttention(config=config, layer_idx=1)
    hidden_states = torch.randn(1, 3, config.hidden_size)
    cos = torch.ones(1, 3, attention.head_dim)
    sin = torch.zeros(1, 3, attention.head_dim)
    captured: dict[str, int] = {}

    def fake_attention_forward(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        del key, value, attention_mask, scaling, kwargs
        captured["layer_idx"] = module.layer_idx
        return torch.zeros(
            query.shape[0],
            query.shape[2],
            query.shape[1],
            query.shape[3],
            dtype=query.dtype,
            device=query.device,
        ), None

    monkeypatch.setattr(modeling_ouro.ALL_ATTENTION_FUNCTIONS, "get_interface", lambda *_args: fake_attention_forward)

    attention(
        hidden_states=hidden_states,
        position_embeddings=(cos, sin),
        attention_mask=None,
        current_ut=2,
        kv_policy="single",
        cache=object(),
        read_index=[],
        write_index=[],
    )

    assert captured["layer_idx"] == 1


@pytest.mark.parametrize(
    ("kv_policy", "kv_slots_per_layer", "current_ut", "expected_layer_idx"),
    [
        ("first_then_shared", 2, 3, 3),
        ("first_then_shared", 3, 1, 3),
        ("single", 1, 3, 1),  # every step collapses onto slot 0 -> 0 * 1 + layer_idx(1)
    ],
)
def test_paged_attention_uses_kv_policy_layer_index(
    monkeypatch,
    kv_policy: str,
    kv_slots_per_layer: int,
    current_ut: int,
    expected_layer_idx: int,
) -> None:
    config = tiny_ouro_config()
    config._attn_implementation = "paged|eager"
    attention = OuroAttention(config=config, layer_idx=1)
    hidden_states = torch.randn(1, 3, config.hidden_size)
    cos = torch.ones(1, 3, attention.head_dim)
    sin = torch.zeros(1, 3, attention.head_dim)
    captured: dict[str, int] = {}

    def fake_attention_forward(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        del key, value, attention_mask, scaling, kwargs
        captured["layer_idx"] = module.layer_idx
        return torch.zeros(
            query.shape[0],
            query.shape[2],
            query.shape[1],
            query.shape[3],
            dtype=query.dtype,
            device=query.device,
        ), None

    monkeypatch.setattr(modeling_ouro.ALL_ATTENTION_FUNCTIONS, "get_interface", lambda *_args: fake_attention_forward)

    attention(
        hidden_states=hidden_states,
        position_embeddings=(cos, sin),
        attention_mask=None,
        current_ut=current_ut,
        kv_policy=kv_policy,
        kv_slots_per_layer=kv_slots_per_layer,
        cache=object(),
        read_index=[],
        write_index=[],
    )

    assert captured["layer_idx"] == expected_layer_idx


def test_paged_attention_keeps_physical_layer_types() -> None:
    model = OuroForCausalLM(tiny_ouro_config())

    model.set_attn_implementation("paged|eager")

    assert model.config.num_hidden_layers == 2
    assert model.config.layer_types == ["full_attention", "full_attention"]

    model.set_attn_implementation("eager")

    assert model.config.num_hidden_layers == 2
    assert model.config.layer_types == ["full_attention", "full_attention"]
