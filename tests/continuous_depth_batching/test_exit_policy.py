"""Behavior tests for model-independent CDB exit decisions."""

from __future__ import annotations

import pytest
import torch

from looped_cdb.continuous_depth_batching.exit_policy import (
    DecisionHorizon,
    ExitDecisionRule,
    ExitPolicy,
    ExitPolicySpec,
    TokenExitPolicyState,
)


def _policy(
    rule: ExitDecisionRule,
    *,
    threshold: float = 0.5,
    min_steps: int = 1,
    native_horizon: DecisionHorizon = DecisionHorizon.SAME_STEP,
    delay: bool = False,
    async_io: bool = False,
) -> ExitPolicy:
    return ExitPolicy(
        spec=ExitPolicySpec(rule, native_horizon),
        threshold=threshold,
        min_recurrent_steps=min_steps,
        max_recurrent_steps=4,
        delay_gate_consumption=delay,
        use_async_batching=async_io,
        synthetic_exit_replay=False,
    )


def test_direct_threshold_does_not_accumulate_across_steps() -> None:
    policy = _policy(ExitDecisionRule.DIRECT_THRESHOLD, threshold=0.3)
    state = TokenExitPolicyState()

    assert not policy.should_exit(state, source_step=0, signal=torch.tensor(0.4))
    assert not policy.should_exit(state, source_step=1, signal=torch.tensor(0.4))
    assert not policy.should_exit(state, source_step=2, signal=torch.tensor(0.4))
    assert state == TokenExitPolicyState()


def test_direct_preminimum_convergence_does_not_latch_after_divergence() -> None:
    policy = _policy(ExitDecisionRule.DIRECT_THRESHOLD, threshold=0.3, min_steps=2)
    state = TokenExitPolicyState()

    assert not policy.should_exit(state, source_step=0, signal=torch.tensor(0.1))
    assert not policy.should_exit(state, source_step=1, signal=torch.tensor(0.8))
    assert policy.should_exit(state, source_step=2, signal=torch.tensor(0.2))


def test_cumulative_hazard_accumulates_before_minimum_depth() -> None:
    policy = _policy(ExitDecisionRule.CUMULATIVE_HAZARD, threshold=0.7, min_steps=2)
    state = TokenExitPolicyState()
    hazard_half_logit = torch.tensor(0.0)

    assert not policy.should_exit(state, source_step=0, signal=hazard_half_logit)
    assert policy.should_exit(state, source_step=1, signal=hazard_half_logit)
    assert state.exit_cdf == pytest.approx(0.75)
    assert state.survival == pytest.approx(0.25)


def test_compatibility_delay_promotes_native_same_step_to_offset_one() -> None:
    immediate = _policy(ExitDecisionRule.DIRECT_THRESHOLD)
    delayed = _policy(ExitDecisionRule.DIRECT_THRESHOLD, delay=True, async_io=True)

    assert immediate.decision_horizon is DecisionHorizon.SAME_STEP
    assert delayed.decision_horizon is DecisionHorizon.OFFSET_ONE


def test_native_offset_one_requires_async_delayed_transport() -> None:
    policy = _policy(
        ExitDecisionRule.CUMULATIVE_HAZARD,
        native_horizon=DecisionHorizon.OFFSET_ONE,
    )

    with pytest.raises(ValueError, match="offset-1"):
        policy.validate_serving_timing()


@pytest.mark.parametrize("rule", list(ExitDecisionRule))
@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), float("-inf"), -0.1])
def test_invalid_thresholds_are_rejected(rule: ExitDecisionRule, threshold: float) -> None:
    with pytest.raises(ValueError, match="exit_threshold"):
        _policy(rule, threshold=threshold)


@pytest.mark.parametrize("rule", [ExitDecisionRule.CUMULATIVE_HAZARD, ExitDecisionRule.PRELOOP_DISTRIBUTION])
@pytest.mark.parametrize("threshold", [0.0, 1.01])
def test_probability_thresholds_require_unit_interval(rule: ExitDecisionRule, threshold: float) -> None:
    with pytest.raises(ValueError, match="probability policies"):
        _policy(rule, threshold=threshold)


@pytest.mark.parametrize("rule", list(ExitDecisionRule))
@pytest.mark.parametrize("threshold", [0.01, 1.0])
def test_valid_probability_thresholds_are_accepted(rule: ExitDecisionRule, threshold: float) -> None:
    assert _policy(rule, threshold=threshold).threshold == threshold


def test_zero_direct_threshold_never_exits_on_nonnegative_scores() -> None:
    policy = _policy(ExitDecisionRule.DIRECT_THRESHOLD, threshold=0.0)
    for score in [0.0, 0.1, 2.0]:
        assert not policy.should_exit(TokenExitPolicyState(), source_step=0, signal=torch.tensor(score))


def test_direct_threshold_can_exceed_one() -> None:
    policy = _policy(ExitDecisionRule.DIRECT_THRESHOLD, threshold=2.0)
    assert policy.should_exit(TokenExitPolicyState(), source_step=0, signal=torch.tensor(1.5))
    assert not policy.should_exit(TokenExitPolicyState(), source_step=0, signal=torch.tensor(2.0))


def test_direct_threshold_preserves_python_float_boundary_for_bfloat16_signal() -> None:
    policy = _policy(ExitDecisionRule.DIRECT_THRESHOLD, threshold=0.28)
    signal = torch.tensor(0.279296875, dtype=torch.bfloat16)
    assert policy.should_exit(TokenExitPolicyState(), source_step=0, signal=signal)
