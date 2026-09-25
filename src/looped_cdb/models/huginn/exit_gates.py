# SPDX-License-Identifier: Apache-2.0
# Adapted from tomg-group-umd/huginn-0125 for paged attention and per-token depth control.
# See THIRD_PARTY_NOTICES.md and LICENSES/Apache-2.0.txt.

"""Convergence exit criterion for Huginn.

Huginn has no trained exit head, so exits are decided by comparing consecutive
recurrent states. The criterion is the relative L2 change between them: it costs
two elementwise reductions and needs nothing but the recurrent state, so a
token's exit decision is available the moment its core step finishes.

Prediction-space criteria (KL divergence, entropy change, argmax stability)
compare output *distributions*, which requires running the coda and LM head at
every recurrent step. That turns the coda from a stage that runs once per token
into one that runs once per step, so they are deliberately not supported.

The criterion is stateless: it takes the current and previous recurrent state
and returns one exit value per token. The caller owns the previous state, which
is what the depth scheduler already tracks per work item.

Values are computed per token rather than averaged over the sequence. The
upstream implementation reduces over the sequence dimension because it only ever
inspects the final position during decoding; per-token values are what a depth
scheduler needs in order to let tokens in one batch exit at different depths.
"""

from __future__ import annotations

import torch
from torch import Tensor


def latent_diff(state: Tensor, prev_state: Tensor) -> Tensor:
    """Relative L2 change between consecutive recurrent states.

    Returns ``||s_r - s_{r-1}|| / ||s_r||`` per token, shape ``(batch, seq)``.
    """

    _check_shapes(state, prev_state)
    return (state - prev_state).norm(dim=-1) / state.norm(dim=-1).clamp_min(1e-9)


def should_exit(
    state: Tensor,
    prev_state: Tensor,
    *,
    threshold: float | None = None,
) -> tuple[Tensor, Tensor]:
    """Evaluate the exit criterion.

    Returns ``(exit_mask, exit_values)``, both shaped ``(batch, seq)``. A token
    exits when its value falls below ``threshold``, meaning the recurrent state
    has stopped changing.
    """

    if threshold is None:
        raise ValueError("threshold must be explicitly set for Huginn early exit")
    limit = float(threshold)
    values = latent_diff(state, prev_state)
    return values < limit, values


def _check_shapes(state: Tensor, prev_state: Tensor) -> None:
    if state.shape != prev_state.shape:
        raise ValueError(f"state and prev_state must match, got {tuple(state.shape)} and {tuple(prev_state.shape)}")
    if state.ndim != 3:
        raise ValueError(f"expected (batch, seq, hidden) recurrent states, got {tuple(state.shape)}")


class ExitTracker:
    """Stateful wrapper for eval loops that run a fixed recurrence.

    The depth scheduler calls the stateless criterion directly; this exists for
    offline analysis, where it is convenient to feed states in step by step and
    read back the depth at which each token first converged.
    """

    def __init__(self, threshold: float | None = None) -> None:
        if threshold is None:
            raise ValueError("threshold must be explicitly set for Huginn early exit")
        self.threshold = float(threshold)
        self._prev: Tensor | None = None
        self._step = 0
        self.exit_step: Tensor | None = None

    def reset(self, initial_state: Tensor) -> None:
        self._prev = initial_state.detach().clone()
        self._step = 0
        self.exit_step = torch.full(
            initial_state.shape[:2],
            fill_value=-1,
            dtype=torch.long,
            device=initial_state.device,
        )

    def update(self, state: Tensor) -> Tensor:
        """Record one recurrent step and return this step's exit values."""

        if self._prev is None or self.exit_step is None:
            raise RuntimeError("reset() must be called with the initial state before update()")
        exited, values = should_exit(state, self._prev, threshold=self.threshold)
        newly_exited = exited & (self.exit_step < 0)
        self.exit_step = torch.where(newly_exited, self._step, self.exit_step)
        self._prev = state.detach().clone()
        self._step += 1
        return values

    def finalize(self, total_steps: int | None = None) -> Tensor:
        """Exit depth per token; tokens that never converged get the last step."""

        if self.exit_step is None:
            raise RuntimeError("reset() must be called before finalize()")
        last = (self._step if total_steps is None else total_steps) - 1
        return torch.where(self.exit_step < 0, last, self.exit_step)
