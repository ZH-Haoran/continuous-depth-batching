"""Continuous-depth-batching adapter for Ouro models.

CDB runs Ouro as three explicit stages: a prelude that is a bare token-embedding
lookup, repeated recurrent block applications, and a coda that is the LM-head
projection. The adapter passes the KV cache
layout to Ouro attention so the cache layer index follows the slot mapping the
runtime allocated for.
"""

from contextlib import suppress
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch import nn

from looped_cdb.continuous_depth_batching.exit_policy import (
    DecisionHorizon,
    ExitDecisionRule,
    ExitPolicySpec,
)
from looped_cdb.continuous_depth_batching.model_adapter import CDBModelAdapter
from looped_cdb.kv_cache_policy import DEPTH_INDEXED, LoopedKvLayout
from looped_cdb.models.ouro.exit_gates import EXIT_GATE_TYPES

#: Gates whose decision comes from the pre-loop hidden state, before any recurrent step,
#: as a full distribution over exit depths rather than a per-step hazard.
PRELOOP_EXIT_GATES = ("preloop",)


def normalize_cdb_exit_gate_type(exit_gate_type: str) -> str:
    """Normalize a gate-type string to its CDB-served form, or raise for unknown ones.

    CDB serves every gate Ouro implements, so this only resolves the ``same_step``
    alias for ``early_exit`` and rejects names the model does not define.
    """

    effective = "early_exit" if exit_gate_type == "same_step" else exit_gate_type
    if effective not in EXIT_GATE_TYPES:
        raise NotImplementedError(
            f"CDB serving supports exit_gate_type in {sorted(EXIT_GATE_TYPES)} "
            f"(plus the 'same_step' alias); got {exit_gate_type!r}."
        )
    return effective


def resolve_cdb_exit_gate_type(config: Any) -> str:
    """Return the CDB-served gate type for an Ouro config, or raise for unsupported ones."""

    return normalize_cdb_exit_gate_type(getattr(config, "exit_gate_type", "early_exit"))


def configure_ouro_exit_gate(
    model: nn.Module,
    *,
    exit_gate_type: str | None = None,
    exit_gate_path: str | Path | None = None,
) -> nn.Module:
    """Load a trained exit gate into an Ouro model for CDB serving.

    ``early_exit`` (alias ``same_step``), ``lookahead`` and ``preloop`` are supported.
    When ``exit_gate_path`` is given, its safetensors weights are loaded into the matching
    gate module. The resolved gate type is written back onto the config so the adapter
    selects the same gate.

    ``exit_gate_type=None`` (the caller did not pass one) falls back to the checkpoint's
    own ``config.exit_gate_type`` rather than a hardcoded default, so a path-only request
    never silently loads e.g. a lookahead checkpoint into the early-exit module and forces
    the config to ``early_exit``. Requesting ``lookahead`` or ``preloop`` without a
    checkpoint raises, because only ``early_exit`` ships trained weights in the base model
    and the others would otherwise be randomly initialized and served under a name implying
    training; the one exception is a checkpoint whose config already declares the requested
    gate (it ships its own trained weights for it).
    """

    ouro_model = getattr(model, "model", model)
    config_gate_type = getattr(getattr(model, "config", None), "exit_gate_type", None)
    # Fall back to the checkpoint's declared gate type when the caller passes none, so a
    # default never overwrites (and mislabels) a value the user did not choose.
    requested = exit_gate_type if exit_gate_type is not None else (config_gate_type or "early_exit")
    effective = normalize_cdb_exit_gate_type(requested)

    # The checkpoint's own (supported) gate type, used only to permit a path-less trained
    # gate for checkpoints that already ship those weights. A config naming a gate this
    # code does not implement simply does not count as a match.
    config_effective: str | None = None
    if config_gate_type is not None:
        with suppress(NotImplementedError):
            config_effective = normalize_cdb_exit_gate_type(config_gate_type)
    # Only ``early_exit`` ships trained weights in the base checkpoint; the other heads are
    # freshly initialized by the model, so serving one without a checkpoint would emit
    # garbage decisions under a gate name that implies training.
    if effective in ("lookahead", "preloop") and exit_gate_path is None and config_effective != effective:
        raise ValueError(
            f"exit_gate_type={effective!r} requires exit_gate_path to a trained {effective} gate; "
            f"without one the {effective}_exit_gate is randomly initialized and would serve garbage "
            f"decisions. (Only a checkpoint whose config already declares {effective!r} may omit the path.)"
        )

    gate_attr = {"early_exit": "early_exit_gate", "lookahead": "lookahead_exit_gate", "preloop": "preloop_exit_gate"}[
        effective
    ]
    if exit_gate_path is not None:
        gate = getattr(ouro_model, gate_attr, None)
        if gate is None:
            raise TypeError(f"Ouro model has no {gate_attr} to load exit_gate_path into")
        gate.load_state_dict(load_file(str(exit_gate_path)))
    for target in (model, ouro_model):
        config = getattr(target, "config", None)
        if config is not None:
            config.exit_gate_type = effective
    return model


class OuroCDBAdapter(CDBModelAdapter):
    """Expose an Ouro causal LM through the staged CDB adapter interface.

    The wrapped model may be either a causal-LM wrapper with ``get_decoder`` or
    a decoder-like module directly. The adapter keeps references to both the
    outer model, for ``lm_head``, and the decoder, for embeddings, recurrent
    layers, normalization, rotary embeddings, and the early-exit gate.
    """

    def __init__(self, model: nn.Module) -> None:
        """Create an adapter around an Ouro model and validate required hooks."""

        self.model = model
        self.decoder = model.get_decoder() if hasattr(model, "get_decoder") else getattr(model, "model", model)
        self._validate()
        # Bind the exit gate once: gate loading (configure_ouro_exit_gate) runs before the
        # adapter is built, and CUDA-graph capture freezes the gate anyway, so re-resolving
        # it from the config on every recurrent step is pure hot-path overhead. The gate type
        # is bound from the same read, so the bound module and the served semantics cannot
        # disagree if the config is mutated afterwards.
        self._exit_gate_type = resolve_cdb_exit_gate_type(self.decoder.config)
        self._exit_gate_module = self._resolve_exit_gate_module()

    @staticmethod
    def supports(model: nn.Module) -> bool:
        """Return whether ``model`` exposes the Ouro surfaces CDB needs."""

        decoder = model.get_decoder() if hasattr(model, "get_decoder") else getattr(model, "model", model)
        model_type = getattr(getattr(model, "config", None), "model_type", None)
        decoder_model_type = getattr(getattr(decoder, "config", None), "model_type", None)
        if "ouro" not in {model_type, decoder_model_type}:
            return False
        return all(hasattr(decoder, name) for name in ("embed_tokens", "layers", "norm", "rotary_emb")) and hasattr(
            model, "lm_head"
        )

    def configure_recurrent_steps(self, max_recurrent_steps: int) -> None:
        """Map CDB's recurrent-step count to Ouro.

        A preloop gate emits one probability per trained depth, so it cannot express an
        exit deeper than the head it was trained with. Serving a larger budget would cap
        every token at the trained depth while reporting the larger one, so it raises
        instead. The check reads the head's own width, which the assignment below
        overwrites on the config.
        """

        if self.decides_exit_before_loop():
            trained_steps = self.decoder.preloop_exit_gate.out_features
            if max_recurrent_steps > trained_steps:
                raise ValueError(
                    f"exit_gate_type='preloop' emits one probability per trained depth, so it cannot route "
                    f"beyond {trained_steps} recurrent steps; got max_recurrent_steps={max_recurrent_steps}."
                )
        for target in (self.model, self.decoder):
            config = getattr(target, "config", None)
            if config is not None:
                config.total_ut_steps = max_recurrent_steps
                self._restore_physical_layer_types(config)
            with suppress(AttributeError):
                target.total_ut_steps = max_recurrent_steps

    def configure_kv_policy(self, kv_policy: str, kv_slots_per_layer: int | None) -> None:
        """Record the CDB recurrent KV policy on the wrapped Ouro configs."""

        for target in (self.model, self.decoder):
            config = getattr(target, "config", None)
            if config is not None:
                config._cdb_kv_policy = kv_policy
                config._cdb_kv_slots_per_layer = kv_slots_per_layer

    def kv_cache_layout(self) -> LoopedKvLayout:
        """The layout Ouro attention addresses the cache with: fully looped, no prelude or coda."""

        config = self.decoder.config
        return LoopedKvLayout.build(
            num_prelude_layers=0,
            num_core_layers=config.num_hidden_layers,
            num_coda_layers=0,
            total_recurrent_steps=int(config.total_ut_steps),
            policy=getattr(config, "_cdb_kv_policy", DEPTH_INDEXED),
            requested_slots=getattr(config, "_cdb_kv_slots_per_layer", None),
        )

    def prelude(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run CDB's prelude stage, which for Ouro is the token-embedding lookup.

        Ouro is fully looped, so this is a pure lookup: it neither reads nor
        writes KV, and the stage's paged-cache arguments do not apply.
        """

        del position_ids, kwargs
        return self.decoder.embed_tokens(input_ids)

    def recurrent_step(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        recurrent_steps: torch.LongTensor,
        cache_position: torch.LongTensor | None = None,
        use_early_exit_gate: bool = True,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run one scheduled recurrent stage for a mixed-depth CDB batch.

        ``recurrent_steps`` is part of the CDB scheduling contract. The current
        Ouro block implementation still executes the same physical layers for
        every row, but CDB validates the scheduled depths here and returns gate
        logits so the engine can decide whether each row exits or gets
        re-enqueued for another recurrent step.
        """

        max_recurrent_steps = getattr(self.decoder, "total_ut_steps", getattr(self.decoder.config, "total_ut_steps", 4))
        if recurrent_steps.device.type == "cpu" and torch.any(
            (recurrent_steps < 0) | (recurrent_steps >= max_recurrent_steps)
        ):
            raise ValueError(f"recurrent_steps must be in [0, {max_recurrent_steps})")
        if position_ids is None:
            raise ValueError("position_ids are required for recurrent_step")
        if cache_position is None:
            cache_position = position_ids.squeeze(0)

        kwargs.pop("attention_mask", None)
        kwargs.pop("use_cache", None)
        kwargs.setdefault("kv_policy", getattr(self.decoder.config, "_cdb_kv_policy", DEPTH_INDEXED))
        kwargs.setdefault("kv_slots_per_layer", getattr(self.decoder.config, "_cdb_kv_slots_per_layer", 1))
        kwargs.setdefault("kv_slot", 0)

        position_embeddings = self.decoder.rotary_emb(hidden_states, position_ids)
        for decoder_layer in self.decoder.layers[: self.decoder.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                past_key_value=None,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                current_ut=0,
                **kwargs,
            )

        hidden_states = self.decoder.norm(hidden_states)
        exit_signals = self._exit_gate(hidden_states) if use_early_exit_gate else None
        return hidden_states, exit_signals

    def _resolve_exit_gate_module(self) -> nn.Module | None:
        """Select the exit-gate module for the configured gate type.

        For ``lookahead`` this is ``lookahead_exit_gate``, whose logit at hidden state
        ``hidden_t`` is the hazard for the *next* recurrent step. The CDB engine's
        delayed-gate-consumption path consumes it one step later, which is exactly the
        lookahead contract (a gate computed at step ``t`` decides the exit at ``t+1``;
        step 0 cannot exit). Lookahead therefore requires the delayed-consumption path,
        which the engine enforces in ``_validate_exit_gate_serving``.

        ``preloop`` has no per-step module: it reads the pre-loop hidden state once, so
        the recurrent stage returns no gate logits and ``preloop_exit_pdf`` serves it.
        """

        if self.decides_exit_before_loop():
            return None
        if self._exit_gate_type == "lookahead":
            return self.decoder.lookahead_exit_gate
        return self.decoder.early_exit_gate

    def _exit_gate(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        """Apply the gate module bound at construction (see ``_resolve_exit_gate_module``)."""

        if self._exit_gate_module is None:
            return None
        return self._exit_gate_module(hidden_states)

    def exit_policy_spec(self) -> ExitPolicySpec:
        """Describe the bound Ouro gate's output and decision horizon."""

        if self._exit_gate_type in PRELOOP_EXIT_GATES:
            return ExitPolicySpec(ExitDecisionRule.PRELOOP_DISTRIBUTION)
        horizon = DecisionHorizon.OFFSET_ONE if self._exit_gate_type == "lookahead" else DecisionHorizon.SAME_STEP
        return ExitPolicySpec(ExitDecisionRule.CUMULATIVE_HAZARD, horizon)

    def decides_exit_before_loop(self) -> bool:
        """Whether the gate bound at construction is the pre-loop head."""

        return self.exit_policy_spec().rule is ExitDecisionRule.PRELOOP_DISTRIBUTION

    def preloop_exit_pdf(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        """Return the pre-loop exit distribution over recurrent depths, or None.

        The preloop gate reads the prelude-stage hidden state, before any recurrent
        block runs, and emits one probability per recurrent depth. The whole exit
        decision is therefore available up front: there is no per-step hazard to
        accumulate and nothing for the scheduler to consume a step late. Returns None
        for the hazard gates, which decide during the loop instead.
        """

        if not self.decides_exit_before_loop():
            return None
        gate = self.decoder.preloop_exit_gate
        return torch.softmax(gate(hidden_states.to(gate.weight.dtype)), dim=-1)

    def lm_head(self, hidden_states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Run CDB's coda stage and project hidden states to logits.

        Ouro has no coda layers, so this is a pure projection and the stage's
        paged-cache arguments do not apply.
        """

        del kwargs
        return self.model.lm_head(hidden_states)

    def _validate(self) -> None:
        """Raise if the wrapped model lacks the Ouro hooks required by CDB."""

        if not self.supports(self.model):
            raise TypeError(f"OuroCDBAdapter does not support model type {type(self.model)!r}")
        if not hasattr(self.decoder, "early_exit_gate"):
            raise TypeError("OuroCDBAdapter requires the decoder to expose early_exit_gate")
        effective_gate_type = resolve_cdb_exit_gate_type(self.model.config)
        required_gate_attr = {"lookahead": "lookahead_exit_gate", "preloop": "preloop_exit_gate"}.get(
            effective_gate_type
        )
        if required_gate_attr is not None and not hasattr(self.decoder, required_gate_attr):
            raise TypeError(f"OuroCDBAdapter requires {required_gate_attr} for exit_gate_type={effective_gate_type!r}")

    @staticmethod
    def _restore_physical_layer_types(config: Any) -> None:
        """Collapse Ouro's paged-attention layer types back to physical layers."""

        layer_types = list(getattr(config, "_ouro_base_layer_types", getattr(config, "layer_types", [])))
        num_hidden_layers = getattr(config, "num_hidden_layers", len(layer_types))
        config.layer_types = layer_types[:num_hidden_layers]
        config._ouro_base_layer_types = list(config.layer_types)
