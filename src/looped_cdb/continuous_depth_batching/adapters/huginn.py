"""Continuous-depth-batching adapter for Huginn models.

Huginn differs from Ouro in two ways that matter to the depth scheduler.

**The core consumes two tensors.** Ouro's recurrent block is a function of one
hidden state. Huginn injects the prelude output at every step,
``adapter(cat([state, input_embeds]))``, so a work item has to carry both. The
engine tracks a single hidden vector per item, so this adapter packs them into
one ``2 * n_embd`` vector: ``prelude`` returns the pair concatenated, every
recurrent step rewrites the state half and passes the injection half through
unchanged, and the coda splits the state back out. The engine sizes its hidden
bank from whatever width ``prelude`` returns, so this needs no scheduler change,
at the cost of doubling the bank.

**The prelude and coda contain attention.** Ouro's prelude is a pure
lookup and its coda is a pure projection, so the engine invokes both bare.
Huginn's prelude and coda are two transformer layers each and write their own
KV, so they need the same cache arguments the recurrent stage receives. Setting
``stages_use_attention`` makes the engine supply them.

Exits are decided by state convergence rather than a trained gate. The
adapter returns the relative latent difference directly as its normalized
scalar signal. The direct-threshold policy compares each step independently on
the criterion's own scale.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from looped_cdb.continuous_depth_batching.exit_policy import ExitDecisionRule, ExitPolicySpec
from looped_cdb.continuous_depth_batching.model_adapter import CDBModelAdapter
from looped_cdb.kv_cache_policy import LoopedKvLayout
from looped_cdb.models.huginn.exit_gates import latent_diff
from looped_cdb.models.huginn.modeling_huginn import cache_layout


class HuginnCDBAdapter(CDBModelAdapter):
    """Expose a Huginn causal LM through the staged CDB adapter interface."""

    #: The prelude and coda are transformer layers that write their own KV.
    stages_use_attention = True

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.config = model.config
        if not self.supports(model):
            raise TypeError(f"HuginnCDBAdapter does not support model type {type(model)!r}")
        self._exit_signal_enabled = False
        self.hidden_size = int(self.config.n_embd)

    @staticmethod
    def supports(model: nn.Module) -> bool:
        """Return whether ``model`` exposes the Huginn stages CDB needs."""

        if getattr(getattr(model, "config", None), "model_type", None) != "huginn":
            return False
        return all(hasattr(model, name) for name in ("run_prelude", "run_core_step", "run_coda", "lm_head"))

    def exit_policy_spec(self) -> ExitPolicySpec:
        """Declare direct, non-accumulating latent-difference decisions."""

        return ExitPolicySpec(ExitDecisionRule.DIRECT_THRESHOLD)

    def configure_exit_signal(self, enabled: bool) -> None:
        """Enable latent-difference output when serving or timing an exit policy."""

        self._exit_signal_enabled = enabled

    def configure_recurrent_steps(self, max_recurrent_steps: int) -> None:
        """Map CDB's recurrent-step count onto the model config."""

        self.config.total_recurrent_steps = int(max_recurrent_steps)

    def configure_kv_policy(self, kv_policy: str, kv_slots_per_layer: int | None) -> None:
        """Record the CDB recurrent KV policy so attention indexes the allocated cache."""

        self.config._cdb_kv_policy = kv_policy
        self.config._cdb_kv_slots_per_layer = kv_slots_per_layer

    def kv_cache_layout(self) -> LoopedKvLayout:
        """The prelude/core/coda layout Huginn attention addresses the cache with."""

        return cache_layout(self.config)

    # -- stage packing ------------------------------------------------------ #

    def pack(self, state: torch.Tensor, injection: torch.Tensor) -> torch.Tensor:
        """Pack the recurrent state and the injected prelude output into one vector."""

        return torch.cat([state, injection], dim=-1)

    def unpack(self, packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split a packed vector back into ``(state, injection)``."""

        if packed.size(-1) != 2 * self.hidden_size:
            raise ValueError(f"expected width {2 * self.hidden_size}, got {packed.size(-1)}")
        return packed[..., : self.hidden_size], packed[..., self.hidden_size :]

    # -- stages ------------------------------------------------------------- #

    def prelude(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run the prelude and return the packed initial state and injection."""

        if position_ids is None:
            raise ValueError("position_ids are required for the prelude stage")

        freqs_cis = self.model.select_freqs_cis(input_ids.shape[-1], position_ids)
        injection = self.model.run_prelude(input_ids, freqs_cis, **kwargs)
        state = self.model.initialize_state(injection)
        return self.pack(state, injection)

    def recurrent_step(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        recurrent_steps: torch.LongTensor,
        cache_position: torch.LongTensor | None = None,
        use_early_exit_gate: bool = True,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Advance one recurrent step and report whether each token has converged.

        ``recurrent_steps`` carries the per-token depth. Under a static KV
        policy every step of a launch writes the same slot, which the engine
        selects with ``kv_slot``, so the step index does not change addressing;
        it is validated rather than used for indexing.
        """

        if position_ids is None:
            raise ValueError("position_ids are required for recurrent_step")
        max_steps = int(self.config.total_recurrent_steps)
        if recurrent_steps.device.type == "cpu" and torch.any((recurrent_steps < 0) | (recurrent_steps >= max_steps)):
            raise ValueError(f"recurrent_steps must be in [0, {max_steps})")

        state, injection = self.unpack(hidden_states)
        kwargs.pop("attention_mask", None)
        kwargs.pop("use_cache", None)
        if "kv_slot" not in kwargs:
            raise ValueError(
                "recurrent_step requires kv_slot: the scheduler picks the static KV slot for the "
                "launch, and reading it from recurrent_steps would need a device-to-host copy that "
                "CUDA graph capture forbids."
            )

        freqs_cis = self.model.select_freqs_cis(state.shape[-2], position_ids)
        next_state = self.model.run_core_step(
            state,
            injection,
            freqs_cis,
            # Addressing comes from the scheduler's kv_slot, so the per-token depth
            # in ``recurrent_steps`` never reaches cache indexing and this stays 0.
            # Reading the real value here would synchronize and break graph capture.
            0,
            cache_position=cache_position,
            **kwargs,
        )

        exit_signal = None
        if use_early_exit_gate and self._exit_signal_enabled:
            exit_signal = self.exit_signal(next_state, state)
        return self.pack(next_state, injection), exit_signal

    @staticmethod
    def exit_signal(state: torch.Tensor, prev_state: torch.Tensor) -> torch.Tensor:
        """Return relative latent difference as a direct threshold score."""

        return latent_diff(state, prev_state).to(state.dtype).unsqueeze(-1)

    def lm_head(self, hidden_states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Run the coda on the state half and project to logits."""

        state, _ = self.unpack(hidden_states)
        position_ids = kwargs.pop("position_ids", None)
        if position_ids is None:
            # ``select_freqs_cis`` would fall back to ``arange`` over the batch axis,
            # rotating request ``i`` as if it sat at position ``i`` and writing that
            # wrong K into the request's correct paged slot.
            raise ValueError("position_ids are required for the coda stage")
        freqs_cis = self.model.select_freqs_cis(state.shape[-2], position_ids)
        return self.model.lm_head(self.model.run_coda(state, freqs_cis, **kwargs))
