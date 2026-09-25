"""Tests for Huginn's state-only convergence exit criterion."""

from __future__ import annotations

import pytest
import torch

from looped_cdb.models.huginn.exit_gates import (
    ExitTracker,
    latent_diff,
    should_exit,
)


def test_identical_states_have_zero_change() -> None:
    state = torch.randn(2, 3, 8)

    assert torch.allclose(latent_diff(state, state), torch.zeros(2, 3), atol=1e-6)


def test_values_are_per_token_not_sequence_averaged() -> None:
    """A depth scheduler needs one value per token so tokens can exit separately."""

    prev = torch.ones(1, 3, 4)
    state = prev.clone()
    state[0, 1] *= 4.0  # only the middle token moves

    values = latent_diff(state, prev)

    assert values.shape == (1, 3)
    assert values[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert values[0, 2] == pytest.approx(0.0, abs=1e-6)
    assert values[0, 1] > 0.5


def test_latent_diff_is_scale_relative() -> None:
    """Doubling both states leaves the relative change unchanged."""

    prev, state = torch.randn(1, 4, 16), torch.randn(1, 4, 16)

    torch.testing.assert_close(latent_diff(state, prev), latent_diff(2 * state, 2 * prev))


def test_should_exit_uses_explicit_threshold() -> None:
    prev = torch.ones(1, 2, 4)
    converged = prev * (1 + 1e-6)
    moving = prev * 2.0

    exited, _ = should_exit(converged, prev, threshold=0.03)
    still_running, _ = should_exit(moving, prev, threshold=0.03)

    assert exited.all()
    assert not still_running.any()


def test_should_exit_requires_explicit_threshold() -> None:
    with pytest.raises(ValueError, match="explicitly set"):
        should_exit(torch.ones(1, 1, 4), torch.ones(1, 1, 4))


def test_mismatched_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="must match"):
        latent_diff(torch.randn(1, 2, 4), torch.randn(1, 3, 4))
    with pytest.raises(ValueError, match="batch, seq, hidden"):
        latent_diff(torch.randn(2, 4), torch.randn(2, 4))


def test_tracker_records_first_converged_step() -> None:
    """Each token's exit depth is the first step at which it stopped moving."""

    initial = torch.zeros(1, 2, 4)
    tracker = ExitTracker(threshold=0.03)
    tracker.reset(initial)

    # Token 0 converges at step 1; token 1 keeps moving throughout.
    states = [
        torch.tensor([[[1.0, 0, 0, 0], [1.0, 0, 0, 0]]]),
        torch.tensor([[[1.0, 0, 0, 0], [5.0, 0, 0, 0]]]),
        torch.tensor([[[1.0, 0, 0, 0], [50.0, 0, 0, 0]]]),
    ]
    for state in states:
        tracker.update(state)

    assert tracker.finalize().tolist() == [[1, 2]]


def test_tracker_requires_reset() -> None:
    with pytest.raises(RuntimeError, match="reset"):
        ExitTracker(threshold=0.03).update(torch.zeros(1, 1, 4))


def test_tracker_requires_explicit_threshold() -> None:
    with pytest.raises(ValueError, match="explicitly set"):
        ExitTracker()
