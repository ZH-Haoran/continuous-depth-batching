from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from looped_cdb.models.ouro.exit_gates import exit_pdf_from_hazards as compute_exit_pdf_from_hazards
from looped_cdb.models.ouro.modeling_ouro import qexit_steps_from_pdf


@dataclass(frozen=True)
class LookaheadBatchMetrics:
    """Scalar training metrics for one hazard-gate batch."""

    loss: torch.Tensor
    hazard_mae: torch.Tensor
    student_mean_hazard: torch.Tensor
    teacher_mean_hazard: torch.Tensor


@dataclass(frozen=True)
class PreloopBatchMetrics:
    """Scalar training metrics for one pre-loop distribution batch."""

    loss: torch.Tensor
    pdf_mae: torch.Tensor
    cdf_mae: torch.Tensor
    kl: torch.Tensor


def _masked_mean(values: torch.Tensor, loss_mask: torch.Tensor | None) -> torch.Tensor:
    """Average token-level values over selected loss positions."""
    mask = torch.ones_like(values) if loss_mask is None else loss_mask.to(device=values.device, dtype=values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def _stack_step_tensors(step_tensors: list[torch.Tensor], *, squeeze_last: bool = True) -> torch.Tensor:
    """Stack per-loop tensors into a batch-by-sequence-by-step tensor."""
    values = [tensor.squeeze(-1) if squeeze_last else tensor for tensor in step_tensors]
    return torch.stack(values, dim=2)


def _fold_pdf_to_min_exit_step(pdf: torch.Tensor, *, min_exit_step: int) -> torch.Tensor:
    """Move probability mass before the minimum exit depth onto that first valid step."""
    min_depth = max(1, int(min_exit_step or 1))
    first_valid_idx = min(min_depth - 1, pdf.shape[-1] - 1)
    if first_valid_idx <= 0:
        return pdf
    folded_pdf = pdf.clone()
    early_mass = folded_pdf[..., :first_valid_idx].sum(dim=-1)
    folded_pdf[..., :first_valid_idx] = 0.0
    folded_pdf[..., first_valid_idx] = folded_pdf[..., first_valid_idx] + early_mass
    return folded_pdf


def _usable_hazard_step_count(*, total_steps: int, first_step_index: int) -> int:
    """Return how many predicted hazards can affect a finite-depth exit PDF."""
    return max(total_steps - first_step_index - 1, 0)


def compute_teacher_exit_pdf(teacher_gate_list: list[torch.Tensor]) -> torch.Tensor:
    """Convert frozen per-loop gate logits into a teacher exit PDF."""
    if not teacher_gate_list:
        raise ValueError("teacher_gate_list must contain at least one recurrent step")
    teacher_hazards = torch.sigmoid(_stack_step_tensors(teacher_gate_list))
    return compute_exit_pdf_from_hazards(teacher_hazards, total_steps=len(teacher_gate_list))


def lookahead_gate_loss(
    *,
    hidden_states_list: list[torch.Tensor],
    teacher_gate_list: list[torch.Tensor],
    lookahead_gate: nn.Module,
    loss_mask: torch.Tensor | None = None,
) -> LookaheadBatchMetrics:
    """Match each hidden state to the frozen gate hazard from the next loop."""
    return hazard_gate_loss(
        hidden_states_list=hidden_states_list,
        teacher_gate_list=teacher_gate_list,
        hazard_gate=lookahead_gate,
        target_step_offset=1,
        loss_mask=loss_mask,
    )


def hazard_gate_loss(
    *,
    hidden_states_list: list[torch.Tensor],
    teacher_gate_list: list[torch.Tensor],
    hazard_gate: nn.Module,
    target_step_offset: int,
    loss_mask: torch.Tensor | None = None,
) -> LookaheadBatchMetrics:
    """Match frozen hazards and the induced q-exit CDF at a fixed step offset."""
    if len(hidden_states_list) != len(teacher_gate_list):
        raise ValueError("hidden_states_list and teacher_gate_list must have the same length")
    if target_step_offset < 0:
        raise ValueError("target_step_offset must be non-negative")
    usable_steps = _usable_hazard_step_count(
        total_steps=len(teacher_gate_list),
        first_step_index=target_step_offset,
    )
    if usable_steps == 0:
        raise ValueError("hazard distillation target offset leaves no trainable steps")

    student_hidden_states = hidden_states_list[:usable_steps]
    target_gate_outputs = teacher_gate_list[target_step_offset : target_step_offset + usable_steps]

    student_logits = _stack_step_tensors([hazard_gate(hidden.float()) for hidden in student_hidden_states])
    with torch.no_grad():
        teacher_hazards = torch.sigmoid(_stack_step_tensors(target_gate_outputs)).to(student_logits.dtype)

    loss_by_token = nn.functional.binary_cross_entropy_with_logits(student_logits, teacher_hazards, reduction="none")
    student_hazards = torch.sigmoid(student_logits)
    hazard_error = (student_hazards - teacher_hazards).abs()

    if loss_mask is None:
        step_mask = torch.ones_like(loss_by_token)
        token_mask = torch.ones_like(loss_by_token[..., 0])
    else:
        token_mask = loss_mask.to(device=loss_by_token.device, dtype=loss_by_token.dtype)
        step_mask = token_mask.unsqueeze(-1).expand_as(loss_by_token)

    denom = step_mask.sum().clamp_min(1.0)
    loss = (loss_by_token * step_mask).sum() / denom
    hazard_mae = (hazard_error * step_mask).sum() / denom
    student_mean_hazard = (student_hazards * step_mask).sum() / denom
    teacher_mean_hazard = (teacher_hazards * step_mask).sum() / denom

    full_teacher_hazards = torch.sigmoid(_stack_step_tensors(teacher_gate_list)).to(student_logits.dtype)
    student_pdf = compute_exit_pdf_from_hazards(
        student_hazards,
        first_step_index=target_step_offset,
        total_steps=len(teacher_gate_list),
    )
    teacher_pdf = compute_exit_pdf_from_hazards(full_teacher_hazards, total_steps=len(teacher_gate_list))
    student_pdf = _fold_pdf_to_min_exit_step(student_pdf, min_exit_step=2)
    teacher_pdf = _fold_pdf_to_min_exit_step(teacher_pdf, min_exit_step=2)
    cdf_error_by_token = (student_pdf.cumsum(dim=-1) - teacher_pdf.cumsum(dim=-1)).abs().mean(dim=-1)
    loss = loss + _masked_mean(cdf_error_by_token, token_mask)

    return LookaheadBatchMetrics(
        loss=loss,
        hazard_mae=hazard_mae,
        student_mean_hazard=student_mean_hazard,
        teacher_mean_hazard=teacher_mean_hazard,
    )


def preloop_gate_loss(
    *,
    preloop_hidden: torch.Tensor,
    teacher_gate_list: list[torch.Tensor],
    preloop_gate: nn.Module,
    loss_mask: torch.Tensor | None = None,
) -> PreloopBatchMetrics:
    """Match the pre-loop gate distribution to the frozen full exit PDF."""
    with torch.no_grad():
        teacher_pdf = compute_teacher_exit_pdf(teacher_gate_list)
    student_logits = preloop_gate(preloop_hidden.float())
    return preloop_pdf_loss(
        student_logits=student_logits,
        teacher_pdf=teacher_pdf,
        loss_mask=loss_mask,
    )


def preloop_pdf_loss(
    *,
    student_logits: torch.Tensor,
    teacher_pdf: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
) -> PreloopBatchMetrics:
    """Compare student pre-loop distribution logits against a teacher exit PDF and CDF."""
    if student_logits.shape != teacher_pdf.shape:
        raise ValueError(
            f"student logits must have shape {tuple(teacher_pdf.shape)}, got {tuple(student_logits.shape)}"
        )

    log_student_pdf = nn.functional.log_softmax(student_logits, dim=-1)
    student_pdf = log_student_pdf.exp()
    teacher_pdf = teacher_pdf.to(dtype=student_pdf.dtype, device=student_pdf.device)
    log_teacher_pdf = teacher_pdf.clamp_min(1e-12).log()

    kl_by_token = (teacher_pdf * (log_teacher_pdf - log_student_pdf)).sum(dim=-1)
    pdf_error_by_token = (student_pdf - teacher_pdf).abs().mean(dim=-1)
    cdf_error_by_token = (student_pdf.cumsum(dim=-1) - teacher_pdf.cumsum(dim=-1)).abs().mean(dim=-1)
    kl = _masked_mean(kl_by_token, loss_mask)
    pdf_mae = _masked_mean(pdf_error_by_token, loss_mask)
    cdf_mae = _masked_mean(cdf_error_by_token, loss_mask)
    loss = kl + cdf_mae
    return PreloopBatchMetrics(loss=loss, pdf_mae=pdf_mae, cdf_mae=cdf_mae, kl=kl)


@torch.no_grad()
def shifted_qexit_metrics(
    *,
    hidden_states_list: list[torch.Tensor],
    teacher_gate_list: list[torch.Tensor],
    lookahead_gate: nn.Module,
    thresholds: tuple[float, ...],
    loss_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Report q-exit agreement after shifting student hazards one loop later."""
    return hazard_qexit_metrics(
        hidden_states_list=hidden_states_list,
        teacher_gate_list=teacher_gate_list,
        hazard_gate=lookahead_gate,
        first_step_index=1,
        thresholds=thresholds,
        loss_mask=loss_mask,
    )


@torch.no_grad()
def hazard_qexit_metrics(
    *,
    hidden_states_list: list[torch.Tensor],
    teacher_gate_list: list[torch.Tensor],
    hazard_gate: nn.Module,
    first_step_index: int,
    thresholds: tuple[float, ...],
    loss_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Report q-exit agreement for a hazard gate aligned to the step timeline."""
    if first_step_index < 0:
        raise ValueError("first_step_index must be non-negative")
    if len(hidden_states_list) != len(teacher_gate_list):
        raise ValueError("hidden_states_list and teacher_gate_list must have the same length")
    usable_steps = _usable_hazard_step_count(
        total_steps=len(teacher_gate_list),
        first_step_index=first_step_index,
    )
    if usable_steps == 0:
        raise ValueError("first_step_index leaves no predicted hazards")

    predicted_hidden_states = hidden_states_list[:usable_steps]
    student_hazards = torch.sigmoid(
        _stack_step_tensors([hazard_gate(hidden.float()) for hidden in predicted_hidden_states])
    )
    teacher_hazards = torch.sigmoid(_stack_step_tensors(teacher_gate_list))
    total_steps = len(teacher_gate_list)
    student_pdf = compute_exit_pdf_from_hazards(
        student_hazards,
        first_step_index=first_step_index,
        total_steps=total_steps,
    )
    teacher_pdf = compute_exit_pdf_from_hazards(teacher_hazards, total_steps=total_steps)
    return qexit_metrics_from_pdfs(
        student_pdf=student_pdf,
        teacher_pdf=teacher_pdf,
        thresholds=thresholds,
        loss_mask=loss_mask,
    )


@torch.no_grad()
def preloop_qexit_metrics(
    *,
    preloop_hidden: torch.Tensor,
    teacher_gate_list: list[torch.Tensor],
    preloop_gate: nn.Module,
    thresholds: tuple[float, ...],
    loss_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Report q-exit agreement from a pre-loop full exit distribution."""
    student_logits = preloop_gate(preloop_hidden.float())
    student_pdf = torch.softmax(student_logits, dim=-1)
    teacher_pdf = compute_teacher_exit_pdf(teacher_gate_list)
    if student_pdf.shape != teacher_pdf.shape:
        raise ValueError(f"student PDF must have shape {tuple(teacher_pdf.shape)}, got {tuple(student_pdf.shape)}")
    return qexit_metrics_from_pdfs(
        student_pdf=student_pdf,
        teacher_pdf=teacher_pdf,
        thresholds=thresholds,
        loss_mask=loss_mask,
    )


@torch.no_grad()
def qexit_metrics_from_pdfs(
    *,
    student_pdf: torch.Tensor,
    teacher_pdf: torch.Tensor,
    thresholds: tuple[float, ...],
    loss_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Report q-exit agreement for two exit-step distributions."""
    if student_pdf.shape != teacher_pdf.shape:
        raise ValueError(f"student PDF must have shape {tuple(teacher_pdf.shape)}, got {tuple(student_pdf.shape)}")
    if loss_mask is None:
        valid_mask = torch.ones_like(teacher_pdf[..., 0], dtype=torch.bool)
    else:
        valid_mask = loss_mask.to(device=teacher_pdf.device, dtype=torch.bool)

    metrics: dict[str, float] = {}
    valid_count = valid_mask.sum().clamp_min(1)
    for threshold in thresholds:
        student_steps = qexit_steps_from_pdf(student_pdf, threshold=threshold, min_exit_step=2)
        teacher_steps = qexit_steps_from_pdf(teacher_pdf, threshold=threshold, min_exit_step=2)
        agreement = ((student_steps == teacher_steps) & valid_mask).sum().float() / valid_count
        student_depth = ((student_steps.float() + 1.0) * valid_mask.float()).sum() / valid_count
        slug = str(threshold).replace(".", "p")
        metrics[f"qexit/agreement_q{slug}"] = float(agreement.item())
        metrics[f"qexit/student_mean_depth_q{slug}"] = float(student_depth.item())
        for step_idx in range(student_pdf.shape[-1]):
            depth = step_idx + 1
            student_exit_frac = ((student_steps == step_idx) & valid_mask).sum().float() / valid_count
            teacher_exit_frac = ((teacher_steps == step_idx) & valid_mask).sum().float() / valid_count
            metrics[f"qexit/student_exit_frac_depth{depth}_q{slug}"] = float(student_exit_frac.item())
            metrics[f"qexit/teacher_exit_frac_depth{depth}_q{slug}"] = float(teacher_exit_frac.item())

    return metrics
