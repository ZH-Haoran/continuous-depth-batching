# SPDX-License-Identifier: Apache-2.0
# Adapted from tomg-group-umd/huginn-0125 for paged attention and per-token depth control.
# See THIRD_PARTY_NOTICES.md and LICENSES/Apache-2.0.txt.

"""Configuration for the paged-attention Huginn implementation.

This is an inference-only reduction of the upstream ``RavenConfig``. Training
knobs (depth sampling, backprop depth, activation checkpointing, init strategy)
are dropped; only fields that change the forward pass or the KV layout remain,
and their defaults are the ``tomg-group-umd/huginn-0125`` release values, the
checkpoint whose numerics the modeling code reproduces.
"""

from __future__ import annotations

from math import sqrt
from typing import Any, ClassVar

from transformers import PretrainedConfig


class HuginnConfig(PretrainedConfig):
    """Config for a prelude/recurrent-core/coda looped decoder.

    ``num_hidden_layers`` is the number of *physical* layers
    (``prelude + core + coda``), not the unrolled depth, so paged-cache sizing
    and layer bookkeeping operate on real parameter blocks. The unrolled depth
    at inference is ``total_recurrent_steps``, which defaults to the training
    mean recurrence.

    ``state_init`` defaults to a zeroed recurrent state so that a run is
    reproducible and two runs are comparable. Upstream draws the state randomly
    per forward, which is available as ``state_init="random"`` but makes every
    measurement carry run-to-run variance.
    """

    model_type = "huginn"
    keys_to_ignore_at_inference: ClassVar[list[str]] = ["latent_states"]
    attribute_map: ClassVar[dict[str, str]] = {
        "num_attention_heads": "n_heads",
        "hidden_size": "n_embd",
        "num_hidden_layers": "n_layers",
    }

    def __init__(
        self,
        n_embd: int = 5280,
        n_heads: int = 55,
        n_layers: int = 8,
        n_layers_in_prelude: int = 2,
        n_layers_in_recurrent_block: int = 4,
        n_layers_in_coda: int = 2,
        mean_recurrence: int = 32,
        total_recurrent_steps: int | None = None,
        block_size: int = 4096,
        vocab_size: int = 65536,
        intermediate_size: int = 17920,
        norm_eps: float = 1e-6,
        rope_base: float = 50_000.0,
        qk_bias: bool = True,
        bias: bool = False,
        tie_embeddings: bool = True,
        state_init: str = "zero",
        bos_token_id: int = 65504,
        eos_token_id: int = 65505,
        pad_token_id: int = 65509,
        **kwargs: Any,
    ) -> None:
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_layers_in_prelude = n_layers_in_prelude
        self.n_layers_in_recurrent_block = n_layers_in_recurrent_block
        self.n_layers_in_coda = n_layers_in_coda
        self.mean_recurrence = mean_recurrence
        self.total_recurrent_steps = mean_recurrence if total_recurrent_steps is None else int(total_recurrent_steps)
        self.block_size = block_size
        self.vocab_size = self.padded_vocab_size = vocab_size
        self.intermediate_size = intermediate_size
        self.norm_eps = norm_eps
        self.rope_base = rope_base
        self.qk_bias = qk_bias
        self.bias = bias
        self.tie_embeddings = tie_embeddings
        self.state_init = state_init

        # Derived attributes the shared serving code expects.
        self.num_key_value_heads = n_heads
        self.head_dim = n_embd // n_heads
        self.attention_dropout = 0.0
        self.sliding_window = None
        self.layer_types = ["full_attention"] * n_layers

        expected = n_layers_in_prelude + n_layers_in_recurrent_block + n_layers_in_coda
        if expected != n_layers:
            raise ValueError(f"n_layers={n_layers} must equal prelude + core + coda = {expected}")
        if self.total_recurrent_steps <= 0:
            raise ValueError(f"total_recurrent_steps must be positive, got {self.total_recurrent_steps}")

        super().__init__(
            tie_word_embeddings=tie_embeddings,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )

    # Stage structure, read by the paged cache to size and index KV slots.
    @property
    def num_prelude_layers(self) -> int:
        return self.n_layers_in_prelude

    @property
    def num_core_layers(self) -> int:
        return self.n_layers_in_recurrent_block

    @property
    def num_coda_layers(self) -> int:
        return self.n_layers_in_coda

    @property
    def embed_scale(self) -> float:
        """Scale applied to token embeddings before the prelude."""

        return sqrt(self.n_embd)

    @property
    def state_init_std(self) -> float:
        """Standard deviation of the initial recurrent state."""

        return sqrt(2 / (5 * self.n_embd))

    @property
    def effective_depth(self) -> int:
        """Unrolled layer count at the configured recurrence."""

        return (
            self.n_layers_in_prelude
            + self.n_layers_in_recurrent_block * self.total_recurrent_steps
            + self.n_layers_in_coda
        )
