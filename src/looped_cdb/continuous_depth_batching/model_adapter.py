"""Generic model adapter interface for continuous depth batching."""

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn

from looped_cdb.kv_cache_policy import LoopedKvLayout

from .exit_policy import DecisionHorizon, ExitDecisionRule, ExitPolicySpec


class CDBModelAdapter(ABC):
    """Model-specific staged decode operations used by the CDB scheduler.

    ``stages_use_attention`` declares whether the prelude and coda stages
    contain attention. A fully looped model's prelude is a lookup and its coda
    a matmul, so the engine calls those stages bare. A prelude/core/coda
    model runs transformer layers in both and needs the same paged-cache
    arguments the recurrent stage receives, which the engine supplies only when
    this is set.
    """

    stages_use_attention: bool = False

    @abstractmethod
    def configure_recurrent_steps(self, max_recurrent_steps: int) -> None:
        """Map CDB's explicit recurrent-step count onto the wrapped model."""

    @abstractmethod
    def prelude(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run the prelude stage: a token-embedding lookup plus, where the model has them, prelude layers."""

    @abstractmethod
    def recurrent_step(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        recurrent_steps: torch.LongTensor,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run one mixed-depth recurrent block application."""

    @abstractmethod
    def lm_head(self, hidden_states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Run the coda/logits stage."""

    def exit_policy_spec(self) -> ExitPolicySpec:
        """Describe the normalized scalar returned by ``recurrent_step``.

        The default serves trained per-step gates whose scalar output is a
        hazard logit for the hidden state just produced. Adapters with direct
        scores, lookahead signals, or preloop distributions override it.
        """

        return ExitPolicySpec(
            rule=ExitDecisionRule.CUMULATIVE_HAZARD,
            decision_horizon=DecisionHorizon.SAME_STEP,
        )

    def configure_exit_signal(self, enabled: bool) -> None:
        """Configure whether ``recurrent_step`` should produce an exit signal.

        Most trained-gate adapters already honor ``use_early_exit_gate`` per
        launch and need no additional configuration.
        """

        del enabled

    def decides_exit_before_loop(self) -> bool:
        """Whether this adapter's gate fixes each token's exit depth before the loop runs."""

        return self.exit_policy_spec().rule is ExitDecisionRule.PRELOOP_DISTRIBUTION

    def preloop_exit_pdf(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        """Return a pre-loop exit distribution over recurrent depths, or None.

        A pre-loop gate decides a token's exit depth from the prelude-stage hidden
        state, before any recurrent step, so it yields a whole distribution rather than
        the per-step hazard ``recurrent_step`` returns. Adapters without such a gate keep
        the default and the engine drives exits from the per-step hazards instead.
        """

        del hidden_states
        return None

    def kv_cache_layout(self) -> LoopedKvLayout:
        """Return the KV layout the wrapped model addresses the paged cache with.

        The engine needs it only for policies that route KV between slots
        (``last_exited``), so adapters without such a use may leave it
        unimplemented.
        """

        raise NotImplementedError(f"{type(self).__name__} does not expose its KV cache layout")


def resolve_cdb_model_adapter(model: nn.Module, adapter: CDBModelAdapter | None = None) -> CDBModelAdapter:
    """Return an explicit or inferred CDB model adapter."""

    if adapter is not None:
        if not isinstance(adapter, CDBModelAdapter):
            raise TypeError(f"model_adapter must inherit CDBModelAdapter, got {type(adapter)!r}")
        return adapter

    from looped_cdb.continuous_depth_batching.adapters import HuginnCDBAdapter, OuroCDBAdapter

    if OuroCDBAdapter.supports(model):
        return OuroCDBAdapter(model)
    if HuginnCDBAdapter.supports(model):
        return HuginnCDBAdapter(model)

    raise TypeError(
        f"Continuous depth batching requires an explicit model adapter or a supported looped model; got {type(model)!r}"
    )
