# SPDX-License-Identifier: Apache-2.0
# Adapted from tomg-group-umd/huginn-0125 for paged attention and per-token depth control.
# See THIRD_PARTY_NOTICES.md and LICENSES/Apache-2.0.txt.

"""Paged-attention Huginn implementation for continuous (depth) batching.

This is a reimplementation of ``tomg-group-umd/huginn-0125`` against the
Hugging Face attention interface, so the model runs under the same
``paged|flash_attention_3`` backend as Ouro. The upstream release is not
vendored; ``tests/fixtures/huginn_golden.pt``, recorded from it before it was
removed, pins this implementation to its numerics.

The forward pass is deliberately split into the three stages the scheduler
cares about. :meth:`HuginnForCausalLM.run_prelude` embeds and runs the
non-recurrent prefix, :meth:`HuginnForCausalLM.run_core_step` advances one
recurrent step, and :meth:`HuginnForCausalLM.run_coda` produces logits.
:meth:`HuginnForCausalLM.forward` composes them at a fixed depth for standard
continuous batching; the depth scheduler drives the same three methods
directly.

Module and parameter names match the upstream checkpoint exactly
(``transformer.prelude``, ``transformer.core_block``, ``transformer.coda``,
``transformer.adapter``, ``transformer.ln_f``, ``lm_head``), so checkpoints
load without a key remap.

Numerics that must not drift from upstream, and are therefore reproduced
verbatim rather than replaced with the closest Hugging Face equivalent:

* Rotary embeddings rotate *interleaved* pairs via a real-valued complex
  formulation, not the split-half ``rotate_half`` used by Llama-style models.
* Attention adds a learned per-head ``qk_bias`` to queries and keys before
  rotation.
* Blocks are sandwich-normed: normalization is applied to each residual *sum*,
  not only to the branch input.
* Token embeddings are scaled by ``sqrt(n_embd)`` before the prelude, and the
  recurrent state is initialized at the same scale.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Any, ClassVar

import torch
from torch import Tensor, nn
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel

from looped_cdb.kv_cache_policy import DEPTH_INDEXED, LoopedKvLayout

from .configuration_huginn import HuginnConfig

Stage = str  # "prelude" | "core" | "coda"


# --------------------------------------------------------------------------- #
# KV layout
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=32)
def _build_layout(
    num_prelude_layers: int,
    num_core_layers: int,
    num_coda_layers: int,
    total_recurrent_steps: int,
    policy: str,
    requested_slots: int | None,
) -> LoopedKvLayout:
    return LoopedKvLayout.build(
        num_prelude_layers=num_prelude_layers,
        num_core_layers=num_core_layers,
        num_coda_layers=num_coda_layers,
        total_recurrent_steps=total_recurrent_steps,
        policy=policy,
        requested_slots=requested_slots,
    )


def cache_layout(config: HuginnConfig, kwargs: dict[str, Any] | None = None) -> LoopedKvLayout:
    """Resolve the KV layout from the config and per-batch scheduler kwargs.

    The scheduler passes ``kv_policy``/``kv_slots_per_layer`` per batch so the
    model indexes the cache the runtime actually allocated; the config carries
    the same values under ``_cdb_*`` for paths that build a model standalone.
    """

    kwargs = kwargs or {}
    policy = kwargs.get("kv_policy") or getattr(config, "_cdb_kv_policy", None) or DEPTH_INDEXED
    slots = kwargs.get("kv_slots_per_layer")
    if slots is None:
        slots = getattr(config, "_cdb_kv_slots_per_layer", None)
    return _build_layout(
        config.n_layers_in_prelude,
        config.n_layers_in_recurrent_block,
        config.n_layers_in_coda,
        int(config.total_recurrent_steps),
        str(policy),
        None if slots is None else int(slots),
    )


def cache_layer_index(
    config: HuginnConfig,
    stage: Stage,
    stage_layer_idx: int,
    recurrent_step: int,
    kwargs: dict[str, Any] | None = None,
) -> int:
    """Cache layer for one physical layer at one recurrent step."""

    layout = cache_layout(config, kwargs)
    if stage == "prelude":
        return layout.prelude_layer_index(stage_layer_idx)
    if stage == "coda":
        return layout.coda_layer_index(stage_layer_idx)
    explicit_slot = (kwargs or {}).get("kv_slot")
    return layout.core_layer_index(
        stage_layer_idx,
        recurrent_step,
        explicit_slot=None if explicit_slot is None else int(explicit_slot),
    )


# --------------------------------------------------------------------------- #
# Layers
# --------------------------------------------------------------------------- #


class HuginnRMSNorm(nn.Module):
    """RMSNorm computed in float32 regardless of the autocast context."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        device_type = x.device.type if x.device.type != "meta" else "cuda"
        with torch.autocast(enabled=False, device_type=device_type):
            xf = x.float()
            normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
            return normed.type_as(x) * self.weight


def rotary_inv_freqs(dim: int, theta: float) -> Tensor:
    """Inverse frequencies for one rotary half-dimension."""

    return 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))


def freqs_cis_for_positions(positions: Tensor, inv_freqs: Tensor) -> Tensor:
    """Cosine/sine values of shape ``(batch, seq, 1, dim // 2, 2)``.

    ``positions`` is ``(seq,)`` or ``(batch, seq)``; the batch dimension is
    preserved so sequences with different position offsets rotate correctly. It
    broadcasts against queries and keys shaped ``(batch, seq, heads, head_dim)``.

    Values are computed from the positions rather than gathered from a
    precomputed table. A table would have to be bounded by the trained context
    length, and continuous batching pads decode batches with out-of-range
    positions, which an indexed lookup rejects with a device-side assert.
    Computing them keeps padded lanes harmless and removes the context ceiling.
    """

    if positions.dim() == 1:
        positions = positions.unsqueeze(0)
    elif positions.dim() != 2:
        raise ValueError(f"positions must be (seq,) or (batch, seq), got {tuple(positions.shape)}")

    freqs = positions.float().unsqueeze(-1) * inv_freqs
    return torch.stack([freqs.cos().unsqueeze(2), freqs.sin().unsqueeze(2)], dim=4)


def apply_rotary_emb_complex_like(q: Tensor, k: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    """Rotate interleaved ``(even, odd)`` pairs of ``q`` and ``k``.

    Inputs are ``(batch, seq, heads, head_dim)``; ``freqs_cis`` broadcasts over
    batch and heads. This is the real-valued form of multiplying by
    ``exp(i * theta)`` and is *not* interchangeable with the split-half rotation
    used elsewhere in Hugging Face.
    """

    with torch.autocast(enabled=False, device_type=q.device.type if q.device.type != "meta" else "cuda"):
        qk_r2 = torch.cat([q, k], dim=2).unflatten(dim=-1, sizes=(-1, 2)).float()
        rotated = torch.stack(
            [
                qk_r2[..., 0] * freqs_cis[..., 0] - qk_r2[..., 1] * freqs_cis[..., 1],
                qk_r2[..., 1] * freqs_cis[..., 0] + qk_r2[..., 0] * freqs_cis[..., 1],
            ],
            dim=-1,
        ).flatten(3)
        return torch.split(rotated.type_as(q), q.shape[2], dim=2)  # type: ignore[return-value]


class _PagedHuginnAttentionView:
    """Proxy exposing a cache-layer index to Hugging Face paged attention.

    The paged kernels read ``module.layer_idx`` to pick the cache tensor. A
    looped model needs that index to depend on the recurrent step as well as the
    physical layer, so the attention module is wrapped rather than mutated,
    keeping the view safe under concurrent capture.
    """

    def __init__(self, module: HuginnAttention, layer_idx: int) -> None:
        self._module = module
        self.layer_idx = layer_idx

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)


class HuginnAttention(nn.Module):
    """Multi-head attention with a learned pre-rotation query/key bias."""

    def __init__(self, config: HuginnConfig, stage: Stage, stage_layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.stage = stage
        self.stage_layer_idx = stage_layer_idx
        self.layer_idx = stage_layer_idx
        self.n_head = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.n_head // self.n_kv_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0.0
        self.is_causal = True
        self.sliding_window = None

        self.chunks = [
            config.n_embd,
            self.n_kv_heads * self.head_dim,
            self.n_kv_heads * self.head_dim,
        ]
        self.Wqkv = nn.Linear(config.n_embd, sum(self.chunks), bias=False)
        if config.qk_bias:
            self.qk_bias = nn.Parameter(torch.zeros(2, 1, self.n_head, self.head_dim))
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(
        self,
        hidden_states: Tensor,
        freqs_cis: Tensor,
        attention_mask: Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: Tensor | None = None,
        recurrent_step: int = 0,
        **kwargs: Any,
    ) -> Tensor:
        batch, seq_len, _ = hidden_states.shape
        q, k, v = self.Wqkv(hidden_states).split(self.chunks, dim=2)
        q = q.view(batch, seq_len, self.n_head, self.head_dim)
        k = k.view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = v.view(batch, seq_len, self.n_kv_heads, self.head_dim)

        if self.config.qk_bias:
            q_bias, k_bias = self.qk_bias.split(1, dim=0)
            q = (q + q_bias).to(q.dtype)
            k = (k + k_bias).to(q.dtype)

        q, k = apply_rotary_emb_complex_like(q, k, freqs_cis=freqs_cis)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        layer_index = cache_layer_index(self.config, self.stage, self.stage_layer_idx, recurrent_step, kwargs)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, layer_index, {"cache_position": cache_position})

        attn_impl = self.config._attn_implementation
        attention_interface: Callable[..., tuple[Tensor, Tensor]] | None = ALL_ATTENTION_FUNCTIONS.get_interface(
            attn_impl, None
        )
        if attention_interface is None:
            # The eager reference path never builds a causal mask for this model, so
            # falling back to it would silently attend to future tokens.
            raise ValueError(
                f"attn_implementation {attn_impl!r} is not a registered attention backend; "
                "use 'sdpa' or a flash/paged kernel."
            )
        attention_module: Any = self
        if isinstance(attn_impl, str) and attn_impl.startswith("paged|"):
            attention_module = _PagedHuginnAttentionView(self, layer_index)

        attn_output, _ = attention_interface(
            attention_module,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(batch, seq_len, -1).contiguous()
        return self.proj(attn_output)


class HuginnMLP(nn.Module):
    """Gated MLP with gate and up projections fused into one matmul."""

    def __init__(self, config: HuginnConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.intermediate_size * 2, bias=False)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=False)
        self.nonlin = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        gate, up = self.fc(x).chunk(2, dim=-1)
        return self.proj(self.nonlin(gate) * up)


class HuginnBlock(nn.Module):
    """Sandwich-normalized transformer block.

    Both residual sums are normalized, which is what keeps the recurrent core
    stable when the same block is applied many times.
    """

    def __init__(self, config: HuginnConfig, stage: Stage, stage_layer_idx: int) -> None:
        super().__init__()
        self.norm_1 = HuginnRMSNorm(config.n_embd, eps=config.norm_eps)
        self.attn = HuginnAttention(config, stage, stage_layer_idx)
        self.norm_2 = HuginnRMSNorm(config.n_embd, eps=config.norm_eps)
        self.mlp = HuginnMLP(config)
        self.norm_3 = HuginnRMSNorm(config.n_embd, eps=config.norm_eps)
        self.norm_4 = HuginnRMSNorm(config.n_embd, eps=config.norm_eps)

    def forward(self, x: Tensor, freqs_cis: Tensor, **kwargs: Any) -> Tensor:
        attn_out = self.attn(self.norm_1(x), freqs_cis, **kwargs)
        x = self.norm_2(attn_out + x)
        return self.norm_4(self.mlp(self.norm_3(x)) + x)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


class HuginnPreTrainedModel(PreTrainedModel):
    config_class = HuginnConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = False
    _no_split_modules: ClassVar[list[str]] = ["HuginnBlock"]
    _skip_keys_device_placement: ClassVar[list[str]] = ["past_key_values"]
    # Rotary values are recomputed from the config; the checkpoint's cached table is redundant.
    _keys_to_ignore_on_load_unexpected: ClassVar[list[str]] = [r"freqs_cis"]
    _tied_weights_keys: ClassVar[dict[str, str]] = {"lm_head.weight": "transformer.wte.weight"}
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_attention_backend = True
    _can_compile_fullgraph = True


class HuginnForCausalLM(HuginnPreTrainedModel, GenerationMixin):
    """Looped decoder with a non-recurrent prelude and coda."""

    def __init__(self, config: HuginnConfig) -> None:
        super().__init__(config)
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.padded_vocab_size, config.n_embd),
                "prelude": nn.ModuleList(HuginnBlock(config, "prelude", i) for i in range(config.n_layers_in_prelude)),
                "adapter": nn.Linear(config.n_embd * 2, config.n_embd, bias=config.bias),
                "core_block": nn.ModuleList(
                    HuginnBlock(config, "core", i) for i in range(config.n_layers_in_recurrent_block)
                ),
                "coda": nn.ModuleList(HuginnBlock(config, "coda", i) for i in range(config.n_layers_in_coda)),
                "ln_f": HuginnRMSNorm(config.n_embd, eps=config.norm_eps),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)

        # Rotary inverse frequencies are derived from the config, so the
        # checkpoint's cached freqs_cis table is redundant and ignored on load.
        # They are cached lazily per device rather than registered as a buffer:
        # a non-persistent buffer absent from the checkpoint is left as
        # uninitialized memory by the weight loader.
        self._inv_freqs_cache: dict[torch.device, Tensor] = {}
        self.post_init()

    def rotary_inv_freqs(self, device: torch.device) -> Tensor:
        """Inverse rotary frequencies for ``device``, computed once and cached."""

        cached = self._inv_freqs_cache.get(device)
        if cached is None:
            cached = rotary_inv_freqs(self.config.head_dim, self.config.rope_base).to(device)
            self._inv_freqs_cache[device] = cached
        return cached

    def get_input_embeddings(self) -> nn.Module:
        return self.transformer["wte"]

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.transformer["wte"] = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    # -- stages ------------------------------------------------------------- #

    def select_freqs_cis(self, seq_len: int, position_ids: Tensor | None) -> Tensor:
        """Rotary cosine/sine values for the positions in this batch."""

        device = self.lm_head.weight.device
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device)
        return freqs_cis_for_positions(position_ids, self.rotary_inv_freqs(device))

    def initialize_state(self, input_embeds: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """Initial recurrent state, matching the upstream initialization scale."""

        if self.config.state_init == "zero":
            return torch.zeros_like(input_embeds)
        std = self.config.state_init_std
        state = torch.empty_like(input_embeds)
        state.normal_(mean=0.0, std=std, generator=generator)
        state.clamp_(min=-3 * std, max=3 * std)
        return state * self.config.embed_scale

    def run_prelude(
        self,
        input_ids: Tensor,
        freqs_cis: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        """Embed tokens and run the non-recurrent prelude."""

        x = self.transformer["wte"](input_ids) * self.config.embed_scale
        for block in self.transformer["prelude"]:
            x = block(x, freqs_cis, **kwargs)
        return x

    def run_core_step(
        self,
        state: Tensor,
        input_embeds: Tensor,
        freqs_cis: Tensor,
        recurrent_step: int,
        **kwargs: Any,
    ) -> Tensor:
        """Advance the recurrent state by one step."""

        x = self.transformer["adapter"](torch.cat([state, input_embeds], dim=-1))
        for block in self.transformer["core_block"]:
            x = block(x, freqs_cis, recurrent_step=recurrent_step, **kwargs)
        return x

    def run_coda(self, state: Tensor, freqs_cis: Tensor, **kwargs: Any) -> Tensor:
        """Run the non-recurrent coda and return final hidden states."""

        x = self.transformer["ln_f"](state)
        for block in self.transformer["coda"]:
            x = block(x, freqs_cis, **kwargs)
        return self.transformer["ln_f"](x)

    # -- composed forward --------------------------------------------------- #

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: Tensor | None = None,
        num_steps: int | None = None,
        logits_to_keep: int | slice = 0,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        """Run prelude, a fixed number of recurrent steps, then the coda."""

        steps = int(self.config.total_recurrent_steps if num_steps is None else num_steps)
        if steps <= 0:
            raise ValueError(f"num_steps must be positive, got {steps}")

        freqs_cis = self.select_freqs_cis(input_ids.shape[1], position_ids)
        stage_kwargs: dict[str, Any] = {
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "cache_position": cache_position,
            **kwargs,
        }

        input_embeds = self.run_prelude(input_ids, freqs_cis, **stage_kwargs)
        state = self.initialize_state(input_embeds)
        for step in range(steps):
            state = self.run_core_step(state, input_embeds, freqs_cis, step, **stage_kwargs)
        hidden_states = self.run_coda(state, freqs_cis, **stage_kwargs)

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :]).float()
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )
