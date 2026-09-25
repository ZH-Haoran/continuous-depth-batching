"""Exit-gate distillation for Ouro models."""

from .metrics import (
    LookaheadBatchMetrics,
    PreloopBatchMetrics,
    compute_exit_pdf_from_hazards,
    compute_teacher_exit_pdf,
    hazard_gate_loss,
    hazard_qexit_metrics,
    lookahead_gate_loss,
    preloop_gate_loss,
    preloop_pdf_loss,
    preloop_qexit_metrics,
    qexit_metrics_from_pdfs,
    shifted_qexit_metrics,
)

__all__ = [
    "LookaheadBatchMetrics",
    "PreloopBatchMetrics",
    "compute_exit_pdf_from_hazards",
    "compute_teacher_exit_pdf",
    "hazard_gate_loss",
    "hazard_qexit_metrics",
    "lookahead_gate_loss",
    "preloop_gate_loss",
    "preloop_pdf_loss",
    "preloop_qexit_metrics",
    "qexit_metrics_from_pdfs",
    "shifted_qexit_metrics",
]
