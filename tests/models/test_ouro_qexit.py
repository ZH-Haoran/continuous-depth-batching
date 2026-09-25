from __future__ import annotations

import torch

from looped_cdb.models.ouro.modeling_ouro import qexit_steps_from_pdf


def test_qexit_min_exit_step_masks_shallow_exits() -> None:
    exit_pdf = torch.tensor([[[0.30, 0.20, 0.10, 0.40], [0.10, 0.20, 0.60, 0.10]]])

    assert torch.equal(
        qexit_steps_from_pdf(exit_pdf, threshold=0.25, min_exit_step=1),
        torch.tensor([[0, 1]]),
    )
    assert torch.equal(
        qexit_steps_from_pdf(exit_pdf, threshold=0.25, min_exit_step=2),
        torch.tensor([[1, 1]]),
    )


def test_qexit_exit_delay_steps_shifts_exits_deeper() -> None:
    exit_pdf = torch.tensor([[[0.30, 0.20, 0.10, 0.40], [0.10, 0.20, 0.60, 0.10]]])

    # Baseline exits (delay 0): first step whose cumulative mass clears the threshold.
    assert torch.equal(
        qexit_steps_from_pdf(exit_pdf, threshold=0.25),
        torch.tensor([[0, 1]]),
    )
    # A delay pushes each exit deeper by exit_delay_steps.
    assert torch.equal(
        qexit_steps_from_pdf(exit_pdf, threshold=0.25, exit_delay_steps=1),
        torch.tensor([[1, 2]]),
    )
    # The delayed exit is clamped to the final recurrent step.
    assert torch.equal(
        qexit_steps_from_pdf(exit_pdf, threshold=0.25, exit_delay_steps=5),
        torch.tensor([[3, 3]]),
    )
