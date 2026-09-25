"""Typed exit-decision policies for continuous depth batching."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum
from math import isfinite

import torch


class ExitDecisionRule(StrEnum):
    """How a model adapter's normalized exit signal is interpreted."""

    CUMULATIVE_HAZARD = "cumulative_hazard"
    DIRECT_THRESHOLD = "direct_threshold"
    PRELOOP_DISTRIBUTION = "preloop_distribution"


class DecisionHorizon(IntEnum):
    """Number of recurrent launches between signal production and exit routing."""

    SAME_STEP = 0
    OFFSET_ONE = 1


@dataclass(frozen=True, slots=True)
class ExitPolicySpec:
    """Adapter-declared signal semantics and native decision horizon.

    Per-step adapters return one scalar per token with shape ``[1, batch, 1]``.
    A cumulative-hazard scalar is an unnormalized logit whose sigmoid is the
    current step's hazard. A direct-threshold scalar is a score that exits when
    it is strictly below the configured threshold. A preloop adapter returns a
    probability distribution from ``preloop_exit_pdf`` instead of per-step
    signals.
    """

    rule: ExitDecisionRule
    decision_horizon: DecisionHorizon = DecisionHorizon.SAME_STEP


@dataclass(slots=True)
class TokenExitPolicyState:
    """Mutable decision state for one token's recurrent work item."""

    exit_cdf: float = 0.0
    survival: float = 1.0


@dataclass(frozen=True, slots=True)
class ExitPolicy:
    """Resolved decision semantics and serving-time signal timing."""

    spec: ExitPolicySpec
    threshold: float | None
    min_recurrent_steps: int
    max_recurrent_steps: int
    delay_gate_consumption: bool
    use_async_batching: bool
    synthetic_exit_replay: bool

    def __post_init__(self) -> None:
        """Validate the threshold on the adapter's signal scale."""

        if self.threshold is None:
            return
        if not isfinite(self.threshold):
            raise ValueError("exit_threshold must be finite")
        if self.spec.rule is ExitDecisionRule.DIRECT_THRESHOLD:
            if self.threshold < 0:
                raise ValueError("exit_threshold must be nonnegative for direct-threshold policies")
        elif not 0 < self.threshold <= 1:
            raise ValueError("exit_threshold must be in (0, 1] for probability policies")

    @property
    def delayed_transport_enabled(self) -> bool:
        """Whether the compatibility delayed-consumption path is active."""

        return self.delay_gate_consumption and self.use_async_batching

    @property
    def decision_horizon(self) -> DecisionHorizon:
        """Effective horizon for live per-step signals.

        Adapter-declared offset-1 signals retain that horizon. The public
        delayed-consumption setting promotes native same-step signals to offset
        1 on the async path, preserving its existing serving behavior.
        """

        if self.spec.rule is ExitDecisionRule.PRELOOP_DISTRIBUTION:
            return DecisionHorizon.SAME_STEP
        if self.spec.decision_horizon is DecisionHorizon.OFFSET_ONE or self.delayed_transport_enabled:
            return DecisionHorizon.OFFSET_ONE
        return DecisionHorizon.SAME_STEP

    @property
    def synthetic_decision_horizon(self) -> DecisionHorizon:
        """Effective horizon for recorded synthetic exit decisions."""

        if self.delayed_transport_enabled:
            return DecisionHorizon.OFFSET_ONE
        return DecisionHorizon.SAME_STEP

    def validate_serving_timing(self) -> None:
        """Reject a native offset-1 signal when overlap transport is disabled."""

        if self.spec.decision_horizon is DecisionHorizon.OFFSET_ONE and not self.delayed_transport_enabled:
            raise ValueError(
                "An adapter-declared offset-1 exit signal requires the delayed-gate-consumption path "
                "(use_async_batching=True and delay_gate_consumption=True)."
            )

    def decides_before_loop(self) -> bool:
        """Whether the policy chooses a depth from the prelude output."""

        return self.spec.rule is ExitDecisionRule.PRELOOP_DISTRIBUTION

    def uses_delayed_online_signal(self, *, signal_enabled: bool) -> bool:
        """Whether a live signal is consumed after the following launch."""

        return (
            signal_enabled
            and self.threshold is not None
            and not self.decides_before_loop()
            and self.decision_horizon is DecisionHorizon.OFFSET_ONE
        )

    def uses_delayed_synthetic_exit(self) -> bool:
        """Whether replayed exits are applied after the following launch."""

        return self.synthetic_exit_replay and self.synthetic_decision_horizon is DecisionHorizon.OFFSET_ONE

    def stages_delayed_signal(self, *, signal_enabled: bool) -> bool:
        """Whether this launch must stage its scalar signal for later consumption."""

        return self.uses_delayed_online_signal(signal_enabled=signal_enabled) or (
            signal_enabled and self.uses_delayed_synthetic_exit()
        )

    def needs_synchronous_signal(self, *, signal_enabled: bool) -> bool:
        """Whether the current signal must be host-visible before routing."""

        if not signal_enabled:
            return False
        live_signal = self.threshold is not None and not self.decides_before_loop()
        measured_replay_signal = self.synthetic_exit_replay
        return (live_signal or measured_replay_signal) and not self.stages_delayed_signal(signal_enabled=True)

    def should_exit(
        self,
        state: TokenExitPolicyState,
        *,
        source_step: int,
        signal: torch.Tensor,
    ) -> bool:
        """Consume one scalar model signal and return its token-local decision."""

        if self.threshold is None:
            return False
        can_exit = source_step + 1 >= self.min_recurrent_steps
        if self.spec.rule is ExitDecisionRule.CUMULATIVE_HAZARD:
            hazard = torch.sigmoid(signal).float().item()
            state.exit_cdf += hazard * state.survival
            state.survival *= 1.0 - hazard
            return can_exit and state.exit_cdf >= self.threshold
        if self.spec.rule is ExitDecisionRule.DIRECT_THRESHOLD:
            return can_exit and signal.float().item() < self.threshold
        raise RuntimeError("Preloop exit distributions do not consume per-step signals")

    def select_preloop_steps(self, exit_pdf: torch.Tensor) -> torch.Tensor:
        """Select one zero-based exit step from each preloop distribution row."""

        if self.threshold is None:
            raise RuntimeError("A preloop exit policy requires an exit threshold")
        exit_cdf = exit_pdf.reshape(-1, exit_pdf.size(-1))[:, : self.max_recurrent_steps].float().cumsum(dim=-1)
        reached = exit_cdf >= self.threshold
        steps = torch.where(
            reached.any(dim=-1),
            reached.float().argmax(dim=-1),
            torch.full_like(reached[:, 0], self.max_recurrent_steps - 1, dtype=torch.long),
        )
        return steps.clamp(min=self.min_recurrent_steps - 1, max=self.max_recurrent_steps - 1)
