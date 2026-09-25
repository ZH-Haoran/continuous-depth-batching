"""Exit-gate helpers for Ouro models."""

from __future__ import annotations

import torch

EXIT_GATE_TYPES = {"early_exit", "lookahead", "preloop"}


def exit_pdf_from_hazards(
    hazards: torch.Tensor,
    *,
    first_step_index: int = 0,
    total_steps: int | None = None,
) -> torch.Tensor:
    """Convert recurrent exit hazards to an exit-depth PDF."""
    if hazards.ndim != 3:
        raise ValueError(f"hazards must have shape (batch, seq, steps), got {tuple(hazards.shape)}")
    if first_step_index < 0:
        raise ValueError("first_step_index must be non-negative")

    batch_size, seq_len, predicted_steps = hazards.shape
    if total_steps is None:
        total_steps = first_step_index + predicted_steps
    if total_steps <= first_step_index:
        raise ValueError("total_steps must be greater than first_step_index")
    if predicted_steps > total_steps - first_step_index:
        raise ValueError("hazards do not fit into total_steps")

    pdf = hazards.new_zeros(batch_size, seq_len, total_steps)
    survival = torch.ones(batch_size, seq_len, dtype=hazards.dtype, device=hazards.device)
    last_predicted = min(predicted_steps, total_steps - first_step_index)
    for offset in range(last_predicted):
        step_idx = first_step_index + offset
        if step_idx >= total_steps - 1:
            break
        hazard = hazards[..., offset]
        pdf[..., step_idx] = hazard * survival
        survival = survival * (1.0 - hazard)

    pdf[..., -1] = survival
    return pdf


def stack_gate_hazards(gate_tensors: list[torch.Tensor]) -> torch.Tensor:
    """Stack scalar gate logits into a batch-by-sequence-by-step hazard tensor."""
    if not gate_tensors:
        raise ValueError("gate_tensors must contain at least one recurrent step")
    return torch.sigmoid(torch.stack([gate_tensor.squeeze(-1) for gate_tensor in gate_tensors], dim=2))
