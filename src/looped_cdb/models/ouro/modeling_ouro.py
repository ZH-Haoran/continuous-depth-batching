# SPDX-License-Identifier: Apache-2.0
# Modified from ByteDance/Ouro-1.4B for paged attention, staged execution, and exit gates.
# See THIRD_PARTY_NOTICES.md and LICENSES/Apache-2.0.txt.

import logging
from collections.abc import Callable
from typing import Any, ClassVar

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple
from transformers.utils.generic import merge_with_config_defaults

from looped_cdb.kv_cache_policy import DEPTH_INDEXED, kv_layer_index, resolve_kv_slots_per_layer

from .configuration_ouro import OuroConfig
from .exit_gates import EXIT_GATE_TYPES, exit_pdf_from_hazards, stack_gate_hazards

logger = logging.getLogger(__name__)


def qexit_steps_from_pdf(
    stacked_exit_pdf: torch.Tensor,
    *,
    threshold: float | torch.Tensor,
    min_exit_step: int = 1,
    exit_delay_steps: int = 0,
) -> torch.Tensor:
    cumulative_probs = torch.cumsum(stacked_exit_pdf, dim=2)
    threshold_value = threshold.to(cumulative_probs.device) if isinstance(threshold, torch.Tensor) else threshold
    threshold_mask = cumulative_probs >= threshold_value
    min_depth = max(1, int(min_exit_step or 1))
    if min_depth > 1:
        threshold_mask[..., : min(min_depth - 1, threshold_mask.shape[2])] = False
    exit_steps = torch.argmax(threshold_mask.float(), dim=2)
    last_step_idx = stacked_exit_pdf.shape[2] - 1
    if last_step_idx >= 0:
        never_exceeded = ~threshold_mask.any(dim=2)
        exit_steps[never_exceeded] = last_step_idx
    delay_steps = max(0, int(exit_delay_steps or 0))
    if delay_steps:
        exit_steps = torch.clamp(exit_steps + delay_steps, max=last_step_idx)
    return exit_steps


def _base_layer_types(config: OuroConfig) -> list[str]:
    """Return the physical decoder layer types before UT-depth expansion."""

    stored_base = getattr(config, "_ouro_base_layer_types", None)
    if stored_base is not None:
        return list(stored_base)

    layer_types = list(config.layer_types)
    base_depth = config.num_hidden_layers
    if len(layer_types) >= base_depth:
        layer_types = layer_types[:base_depth]
    config._ouro_base_layer_types = list(layer_types)
    return layer_types


def _set_paged_layer_types(config: OuroConfig, attn_implementation: str | dict) -> None:
    if not isinstance(attn_implementation, str):
        return

    base_layer_types = _base_layer_types(config)
    config.layer_types = base_layer_types


def _kv_cache_policy(kwargs: dict[str, Any]) -> str:
    """The KV layout this forward writes under, defaulting to depth-indexed."""

    policy = kwargs.get("kv_policy")
    return policy if isinstance(policy, str) else DEPTH_INDEXED


def _kv_slots(config: OuroConfig, kwargs: dict[str, Any]) -> int:
    requested_slots = kwargs.get("kv_slots_per_layer")
    return resolve_kv_slots_per_layer(
        _kv_cache_policy(kwargs),
        total_recurrent_steps=getattr(config, "total_ut_steps", 1) or 1,
        requested_slots=None if requested_slots is None else int(requested_slots),
    )


def _ouro_cache_layer_index(config: OuroConfig, layer_idx: int, current_ut: int, kwargs: dict[str, Any]) -> int:
    explicit_slot = kwargs.get("kv_slot")
    return kv_layer_index(
        physical_layer_idx=layer_idx,
        num_hidden_layers=config.num_hidden_layers,
        recurrent_step=current_ut,
        policy=_kv_cache_policy(kwargs),
        slots_per_layer=_kv_slots(config, kwargs),
        explicit_slot=None if explicit_slot is None else int(explicit_slot),
    )


class _PagedOuroAttentionView:
    """Proxy exposing a UT-depth virtual layer index to HF paged attention."""

    def __init__(self, module: "OuroAttention", layer_idx: int) -> None:
        self._module = module
        self.layer_idx = layer_idx

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)


def needs_universal_cache(cache: Cache | None, max_cache_size: int | None) -> bool:
    if cache is None:
        return True
    if isinstance(cache, UniversalTransformerCache):
        return False
    if not isinstance(cache, Cache):
        return False
    can_grow = getattr(cache, "layer_class_to_replicate", None) is not None
    if can_grow:
        # Dynamic caches can extend to any index, so let them be
        return False
    cache_layers = getattr(cache, "layers", [])
    if max_cache_size is not None and len(cache_layers) < max_cache_size:
        try:
            cached_tokens = cache.get_seq_length()
        except Exception:
            cached_tokens = 0
        if cached_tokens > 0:
            raise ValueError(
                "The provided cache cannot store all Universal Transformer iterations. Please "
                "instantiate Ouro.modeling_ouro.UniversalTransformerCache and pass it as past_key_values."
            )
        return True
    return False


class OuroMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    del position_ids
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class UniversalTransformerCache(Cache):
    """Cache implementation that supports Ouro's multi-step Universal Transformer loops."""

    def __init__(self, max_cache_size: int | None = None):
        # We intentionally don't call super().__init__ because the parent assumes static cache sizes.
        self.key_cache: list[torch.Tensor | None] = []
        self.value_cache: list[torch.Tensor | None] = []
        self.layers: list[Any] = []  # attribute expected by HF Cache utilities
        self._seen_tokens = 0
        self.max_cache_size = max_cache_size

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cache_kwargs
        if layer_idx < 0:
            raise ValueError(f"layer_idx must be non-negative, got {layer_idx}")

        if self.max_cache_size is not None and layer_idx >= self.max_cache_size:
            raise IndexError(
                f"Cache index {layer_idx} exceeds configured max_cache_size={self.max_cache_size}. "
                "Check total_ut_steps and num_hidden_layers."
            )

        # Expand cache storage so the requested index is available.
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)

        cached_key = self.key_cache[layer_idx]
        cached_value = self.value_cache[layer_idx]

        if cached_key is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            if (
                key_states.shape[0] != cached_key.shape[0]
                or key_states.shape[1] != cached_key.shape[1]
                or key_states.shape[3] != cached_key.shape[3]
            ):
                raise ValueError(
                    "Cached and incoming key/value tensors must match on batch, head, and head_dim dimensions."
                )
            assert cached_value is not None
            self.key_cache[layer_idx] = torch.cat([cached_key, key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([cached_value, value_states], dim=2)

        result_key = self.key_cache[layer_idx]
        result_value = self.value_cache[layer_idx]
        assert result_key is not None and result_value is not None

        # Track sequence length using the first populated cache entry.
        self._seen_tokens = result_key.shape[2]
        return result_key, result_value

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        if layer_idx is None:
            layer_idx = 0
        if layer_idx < 0 or len(self.key_cache) <= layer_idx:
            return 0
        cached = self.key_cache[layer_idx]
        if cached is None:
            return 0
        return cached.shape[2]

    def get_max_length(self) -> int | None:
        return None

    def get_usable_length(self, new_seq_length: int, layer_idx: int | None = 0) -> int:
        del new_seq_length
        return self.get_seq_length(layer_idx)

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        for idx, (key_entry, value_entry) in enumerate(zip(self.key_cache, self.value_cache)):
            if key_entry is None:
                continue
            assert value_entry is not None
            device = key_entry.device
            self.key_cache[idx] = key_entry.index_select(0, beam_idx.to(device))
            self.value_cache[idx] = value_entry.index_select(0, beam_idx.to(device))

    def get_mask_sizes(self, cache_position_or_q_length: torch.Tensor | int, layer_idx: int = 0) -> tuple[int, int]:
        """Return (kv_length, kv_offset) accounting for cached tokens.

        The inherited Cache.get_mask_sizes checks ``self.layers`` which is
        always empty for UniversalTransformerCache, causing it to return
        ``(query_length, 0)`` instead of ``(cached_length + query_length, 0)``
        during autoregressive decoding.
        """
        query_length = (
            cache_position_or_q_length
            if isinstance(cache_position_or_q_length, int)
            else cache_position_or_q_length.shape[0]
        )
        seq_length = self.get_seq_length(layer_idx)
        return seq_length + query_length, 0

    @property
    def is_compileable(self) -> bool:
        return False

    def clear(self) -> None:
        logger.debug("Clearing UniversalTransformerCache")
        self.key_cache = []
        self.value_cache = []
        self._seen_tokens = 0


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    del kwargs
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class OuroAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: OuroConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_value: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        current_ut: int = 0,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                _ouro_cache_layer_index(self.config, self.layer_idx, current_ut, kwargs),
                cache_kwargs,
            )

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attention_module: nn.Module | _PagedOuroAttentionView = self
        if isinstance(self.config._attn_implementation, str) and self.config._attn_implementation.startswith("paged|"):
            paged_layer_idx = _ouro_cache_layer_index(self.config, self.layer_idx, current_ut, kwargs)
            attention_module = _PagedOuroAttentionView(
                self,
                paged_layer_idx,
            )

        attn_output, attn_weights = attention_interface(
            attention_module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # main diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


@use_kernel_forward_from_hub("RMSNorm")
class OuroRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        OuroRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class OuroDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: OuroConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = OuroAttention(config=config, layer_idx=layer_idx)

        self.mlp = OuroMLP(config)
        self.input_layernorm = OuroRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_2 = OuroRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = OuroRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_2 = OuroRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = self.input_layernorm_2(hidden_states)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_attention_layernorm_2(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class OuroPreTrainedModel(PreTrainedModel):
    config: OuroConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules: ClassVar[list[str]] = ["OuroDecoderLayer"]
    _skip_keys_device_placement: ClassVar[list[str]] = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs: ClassVar[dict[str, type[nn.Module]]] = {
        "hidden_states": OuroDecoderLayer,
        "attentions": OuroAttention,
    }

    def set_attn_implementation(self, attn_implementation: str | dict, allow_all_kernels: bool = False) -> None:
        super().set_attn_implementation(attn_implementation, allow_all_kernels=allow_all_kernels)
        _set_paged_layer_types(self.config, attn_implementation)


class OuroRotaryEmbedding(nn.Module):
    def __init__(self, config: OuroConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_type = self.config.rope_parameters["rope_type"]
        self.rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config: OuroConfig | None = None,
        device: torch.device | None = None,
        seq_len: int | None = None,
    ) -> tuple[torch.Tensor, float]:
        del seq_len
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        attention_factor = 1.0

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class OuroModel(OuroPreTrainedModel):
    def __init__(self, config: OuroConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [OuroDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = OuroRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = OuroRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        self.total_ut_steps = getattr(self.config, "total_ut_steps", 4)
        self.early_exit_gate = nn.Linear(config.hidden_size, 1)
        self.lookahead_exit_gate = nn.Linear(config.hidden_size, 1)
        self.preloop_exit_gate = nn.Linear(config.hidden_size, self.total_ut_steps)
        # Initialize weights and apply final processing
        self.post_init()

    def initialize_lookahead_exit_gate_from_frozen_gate(self) -> None:
        """Initialize the lookahead gate from the frozen early-exit gate."""
        self.lookahead_exit_gate.load_state_dict(self.early_exit_gate.state_dict())

    def _run_recurrent_step(
        self,
        hidden_states: torch.Tensor,
        *,
        current_ut: int,
        causal_mask_mapping: dict[str, torch.Tensor],
        position_ids: torch.LongTensor,
        past_key_values: Cache | None,
        use_cache: bool | None,
        cache_position: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        kwargs: dict[str, Any],
    ) -> torch.Tensor:
        """Run one Ouro recurrent step through the shared decoder stack."""
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                current_ut=current_ut,
                **kwargs,
            )
        return self.norm(hidden_states)

    @merge_with_config_defaults
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        use_early_exit_gate: bool = False,
        return_preloop_hidden: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        cache_position = kwargs.pop("cache_position", None)
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache is None:
            use_cache = self.config.use_cache

        max_cache_size: int | None = None
        if use_cache:
            total_layers = getattr(self.config, "num_hidden_layers", None)
            if total_layers is not None:
                max_cache_size = total_layers * _kv_slots(self.config, kwargs)

            if needs_universal_cache(past_key_values, max_cache_size):
                past_key_values = UniversalTransformerCache(max_cache_size)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        preloop_hidden = hidden_states

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        hidden_states_list: list[torch.Tensor] = []
        gate_list: list[torch.Tensor] = []

        for current_ut in range(self.total_ut_steps):
            hidden_states = self._run_recurrent_step(
                hidden_states,
                current_ut=current_ut,
                causal_mask_mapping=causal_mask_mapping,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                kwargs=kwargs,
            )
            if use_early_exit_gate:
                hidden_states_list.append(hidden_states)
                gate_list.append(self.early_exit_gate(hidden_states))

        outputs = (
            BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values if use_cache else None,
            ),
            hidden_states_list,
            gate_list,
        )
        if return_preloop_hidden:
            return (*outputs, preloop_hidden)
        return outputs


class _FP32LMHead(nn.Module):
    """Wraps a linear lm_head to perform the matmul in fp32.

    Weights stay in their original dtype (bf16); hidden states and weights are
    upcast to fp32 only for the matmul (ScaleRL / MiniMax approach).
    Exposes .weight and .bias so that save_pretrained / tied-weights still work.
    """

    def __init__(self, original: nn.Linear) -> None:
        super().__init__()
        self.original = original

    @property
    def weight(self) -> torch.Tensor:
        return self.original.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.original.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(
            x.float(),
            self.original.weight.float(),
            self.original.bias.float() if self.original.bias is not None else None,
        )


class OuroForCausalLM(OuroPreTrainedModel, GenerationMixin):
    _tied_weights_keys: ClassVar[dict[str, str]] = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan: ClassVar[dict[str, str]] = {"lm_head": "colwise_rep"}
    _pp_plan: ClassVar[dict[str, tuple[list[str], list[str]]]] = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = OuroModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 分块大小配置
        self.chunk_size = getattr(config, "chunk_size", 2)  # 默认分块大小为2
        self.early_exit_step = getattr(config, "early_exit_step", None)
        self.early_exit_threshold = getattr(config, "early_exit_threshold", None)
        self.min_exit_step = getattr(config, "min_exit_step", 1)
        self.exit_delay_steps = getattr(config, "exit_delay_steps", 0)
        self.exit_gate_type = getattr(config, "exit_gate_type", "early_exit")

        # Initialize weights and apply final processing
        self.post_init()

    def enable_fp32_lm_head(self) -> None:
        """Wrap lm_head to compute the final matmul in fp32 (ScaleRL fix).

        Weights remain bf16; only the matmul is upcast. Call after from_pretrained().
        """
        if not isinstance(self.lm_head, _FP32LMHead):
            self.lm_head = _FP32LMHead(self.lm_head)

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        use_early_exit_gate: bool = False,
        use_weighted_exit: bool | None = False,  # 控制是否使用加权 early exit
        exit_at_step: int | None = None,
        exit_threshold: float | None = None,
        min_exit_step: int | None = None,
        exit_delay_steps: int | None = None,
        return_exit_steps: bool = False,
        return_exit_pdf: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Args:
            use_weighted_exit (`bool`, *optional*, defaults to `False`):
                Whether to use weighted early exit. If `True`, the logits from all UT steps will be
                averaged according to the exit probability distribution.
            exit_at_step (`int`, *optional*):
                Specifies which UT step to exit at. If set, the model will directly use the hidden states
                from this step to generate logits, ignoring other exit strategies.
            exit_threshold (`float`, *optional*):
                The cumulative probability threshold for early exit. When the cumulative exit probability
                reaches this threshold, the model will exit at that step.
            min_exit_step (`int`, *optional*):
                One-based minimum recurrent depth for threshold-based early exit.
            exit_delay_steps (`int`, *optional*):
                Additional recurrent steps to run after the threshold-selected exit depth.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OuroForCausalLM

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        cache_position = kwargs.pop("cache_position", None)
        exit_at_step = exit_at_step if exit_at_step is not None else self.early_exit_step
        exit_threshold = exit_threshold if exit_threshold is not None else self.early_exit_threshold
        min_exit_step = min_exit_step if min_exit_step is not None else self.min_exit_step
        exit_delay_steps = exit_delay_steps if exit_delay_steps is not None else self.exit_delay_steps
        exit_gate_type = getattr(self.config, "exit_gate_type", self.exit_gate_type)
        if exit_gate_type not in EXIT_GATE_TYPES:
            raise ValueError(f"Unsupported exit_gate_type={exit_gate_type!r}")
        use_threshold_exit = exit_threshold is not None and float(exit_threshold) < 1.0
        needs_recurrent_outputs = bool(
            use_early_exit_gate
            and (
                use_weighted_exit
                or exit_at_step is not None
                or use_threshold_exit
                or return_exit_steps
                or return_exit_pdf
                or labels is not None
            )
        )
        needs_preloop_hidden = needs_recurrent_outputs and exit_gate_type == "preloop"

        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            use_early_exit_gate=needs_recurrent_outputs,
            return_preloop_hidden=needs_preloop_hidden,
            **kwargs,
        )
        if needs_preloop_hidden:
            outputs, hidden_states_list, gate_list, preloop_hidden = model_outputs
        else:
            outputs, hidden_states_list, gate_list = model_outputs
            preloop_hidden = None
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

        def _select_token_positions(tensor: torch.Tensor) -> torch.Tensor:
            if isinstance(slice_indices, slice):
                return tensor[:, slice_indices, ...]
            if isinstance(slice_indices, torch.Tensor):
                return tensor.index_select(1, slice_indices.to(tensor.device))
            raise TypeError(f"Unsupported index type for logits_to_keep: {type(slice_indices)}")

        stacked_exit_pdf = None
        if hidden_states_list:
            total_steps = len(hidden_states_list)
            if exit_gate_type == "early_exit" and gate_list:
                stacked_exit_pdf = exit_pdf_from_hazards(stack_gate_hazards(gate_list), total_steps=total_steps)
            elif exit_gate_type == "lookahead" and total_steps > 1:
                lookahead_gate_logits = [self.model.lookahead_exit_gate(hidden) for hidden in hidden_states_list[:-1]]
                stacked_exit_pdf = exit_pdf_from_hazards(
                    stack_gate_hazards(lookahead_gate_logits),
                    first_step_index=1,
                    total_steps=total_steps,
                )
            elif exit_gate_type == "preloop":
                if preloop_hidden is None:
                    raise RuntimeError("preloop exit gate requires return_preloop_hidden=True")
                gate_dtype = self.model.preloop_exit_gate.weight.dtype
                stacked_exit_pdf = torch.softmax(self.model.preloop_exit_gate(preloop_hidden.to(gate_dtype)), dim=-1)

        expected_logits_cache: torch.Tensor | None = None

        def compute_expected_logits() -> torch.Tensor | None:
            nonlocal expected_logits_cache
            if expected_logits_cache is not None:
                return expected_logits_cache
            if stacked_exit_pdf is None or not hidden_states_list:
                return None
            token_exit_pdf = _select_token_positions(stacked_exit_pdf)
            expected_logits = None
            for step_idx, hidden in enumerate(hidden_states_list):
                step_hidden = _select_token_positions(hidden)
                step_logits = self.lm_head(step_hidden)
                weight = token_exit_pdf[..., step_idx].unsqueeze(-1).to(step_logits.dtype)
                expected_logits = (
                    step_logits * weight if expected_logits is None else expected_logits + step_logits * weight
                )
            expected_logits_cache = expected_logits
            return expected_logits_cache

        logits: torch.Tensor | None = None
        loss: torch.Tensor | None = None
        exit_steps: torch.Tensor | None = None

        if labels is not None:
            logits = compute_expected_logits()
            if logits is None:
                hidden_states = outputs.last_hidden_state
                logits = self.lm_head(_select_token_positions(hidden_states))
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )
        else:
            if stacked_exit_pdf is not None and hidden_states_list:
                if exit_at_step is not None and 0 <= exit_at_step < len(hidden_states_list):
                    selected_hidden = hidden_states_list[exit_at_step]
                    logits = self.lm_head(_select_token_positions(selected_hidden))
                elif use_threshold_exit:
                    exit_steps = qexit_steps_from_pdf(
                        stacked_exit_pdf,
                        threshold=exit_threshold,
                        min_exit_step=min_exit_step,
                        exit_delay_steps=exit_delay_steps,
                    )
                    stacked_hidden = torch.stack(hidden_states_list, dim=2)
                    gather_index = exit_steps.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, stacked_hidden.size(-1))
                    final_hidden_states = torch.gather(stacked_hidden, 2, gather_index).squeeze(2)
                    logits = self.lm_head(_select_token_positions(final_hidden_states))
                elif use_weighted_exit:
                    logits = compute_expected_logits()

            if logits is None:
                hidden_states = outputs.last_hidden_state
                logits = self.lm_head(_select_token_positions(hidden_states))

        result = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        if return_exit_steps and exit_steps is not None:
            result.ouro_exit_steps = exit_steps
        if return_exit_pdf and stacked_exit_pdf is not None:
            # Raw per-position exit PDF over recurrent steps ([batch, seq, total_steps]);
            # a threshold-free record from which any exit depth can be derived offline.
            result.ouro_exit_pdf = stacked_exit_pdf

        return result


class OuroForSequenceClassification(GenericForSequenceClassification, OuroPreTrainedModel):
    pass


class OuroForTokenClassification(GenericForTokenClassification, OuroPreTrainedModel):
    pass


class OuroForQuestionAnswering(GenericForQuestionAnswering, OuroPreTrainedModel):
    base_model_prefix = "transformer"  # For BC, where `transformer` was used instead of `model`


__all__ = [
    "OuroForCausalLM",
    "OuroForQuestionAnswering",
    "OuroForSequenceClassification",
    "OuroForTokenClassification",
    "OuroModel",
    "OuroPreTrainedModel",
    "UniversalTransformerCache",
]
