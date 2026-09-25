from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from safetensors.torch import save_file
from torch.optim import AdamW
from transformers import AutoTokenizer, set_seed

from looped_cdb.models.ouro import OuroForCausalLM
from looped_cdb.train.common_data import (
    NEMOTRON_CATEGORIES,
    NEMOTRON_DATASET_NAME,
    NEMOTRON_DATASET_REVISION,
    VALIDATION_PACKED_SEQUENCES,
    VALIDATION_SEED,
)
from looped_cdb.utils import require_flash_attention

from .data import ID_TO_SOURCE, create_packed_dataloaders
from .metrics import (
    LookaheadBatchMetrics,
    PreloopBatchMetrics,
    hazard_gate_loss,
    hazard_qexit_metrics,
    preloop_gate_loss,
    preloop_qexit_metrics,
)

GateSelection = Literal["both", "lookahead", "preloop"]
ControlGateSelection = Literal["none", "same_step"]
LearningRateSchedule = Literal["constant", "warmup_cosine"]
QEXIT_THRESHOLDS = (0.1, 0.2, 0.25, 0.3, 0.5, 0.7)
HAZARD_GATE_TARGET_OFFSETS = {
    "lookahead": 1,
    "same_step": 0,
}


@dataclass(frozen=True)
class GateTrainingConfig:
    """Configuration for Ouro gate training."""

    model_name_or_path: str
    output_dir: Path
    train_gates: GateSelection = "both"
    control_gates: ControlGateSelection = "same_step"
    sequence_length: int = 4096
    batch_size: int = 1
    eval_batch_size: int | None = None
    max_steps: int = 20_000
    lookahead_learning_rate: float = 1e-3
    preloop_learning_rate: float = 1e-3
    learning_rate_schedule: LearningRateSchedule = "warmup_cosine"
    warmup_steps: int = 500
    min_lr_ratio: float = 0.05
    max_grad_norm: float = 1.0
    log_every: int = 20
    eval_every: int = 500
    save_every: int = 0
    seed: int = 79
    wandb_project: str | None = "looped-cdb"
    wandb_run_name: str | None = None
    wandb_mode: str | None = None

    def __post_init__(self) -> None:
        """Validate scalar configuration early, before loading data or models."""
        if self.sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.eval_batch_size is not None and self.eval_batch_size < 1:
            raise ValueError("eval_batch_size must be positive when set")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.lookahead_learning_rate < 0.0:
            raise ValueError("lookahead_learning_rate must be non-negative")
        if self.preloop_learning_rate < 0.0:
            raise ValueError("preloop_learning_rate must be non-negative")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.warmup_steps > self.max_steps:
            raise ValueError("warmup_steps must be less than or equal to max_steps")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be between 0.0 and 1.0")
        if self.max_grad_norm < 0.0:
            raise ValueError("max_grad_norm must be non-negative")
        if self.log_every < 1:
            raise ValueError("log_every must be positive")
        if self.eval_every < 0:
            raise ValueError("eval_every must be non-negative")
        if self.save_every < 0:
            raise ValueError("save_every must be non-negative")


def _selected_gate_names(train_gates: GateSelection) -> tuple[str, ...]:
    """Return the concrete real gate names selected for training."""
    if train_gates == "both":
        return ("lookahead", "preloop")
    if train_gates in {"lookahead", "preloop"}:
        return (train_gates,)
    raise ValueError(f"Unsupported train_gates value: {train_gates}")


def _selected_control_gate_names(control_gates: ControlGateSelection) -> tuple[str, ...]:
    """Return the concrete diagnostic control gate names selected for training."""
    if control_gates == "none":
        return ()
    if control_gates == "same_step":
        return ("same_step",)
    raise ValueError(f"Unsupported control_gates value: {control_gates}")


def _require_gate(model: torch.nn.Module, gate_name: str) -> torch.nn.Module:
    """Return a trainable model-owned gate or fail with a clear model-shape error."""
    ouro_model = getattr(model, "model", None)
    attr_name = f"{gate_name}_exit_gate"
    if ouro_model is None or not hasattr(ouro_model, attr_name):
        raise TypeError(f"Model must expose model.{attr_name}; use the repo-local Ouro implementation")
    return getattr(ouro_model, attr_name)


def _ouro_core(model: torch.nn.Module) -> torch.nn.Module:
    """Return the wrapped Ouro core from a causal-LM-style module."""
    unwrapped = getattr(model, "module", model)
    ouro_model = getattr(unwrapped, "model", None)
    if ouro_model is None:
        raise TypeError("Model must expose a .model Ouro core")
    return ouro_model


def _initialize_trainable_gates(model: torch.nn.Module, train_gates: GateSelection) -> dict[str, torch.nn.Module]:
    """Initialize selected real gates and freeze all unselected model parameters."""
    ouro_model = _ouro_core(model)
    selected_gates = _selected_gate_names(train_gates)
    if "lookahead" in selected_gates:
        if not hasattr(ouro_model, "initialize_lookahead_exit_gate_from_frozen_gate"):
            raise TypeError("Model must expose initialize_lookahead_exit_gate_from_frozen_gate()")
        ouro_model.initialize_lookahead_exit_gate_from_frozen_gate()

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    gates = {gate_name: _require_gate(model, gate_name) for gate_name in selected_gates}
    for gate in gates.values():
        gate.float()
        gate.train()
        for parameter in gate.parameters():
            parameter.requires_grad_(True)
    return gates


def _initialize_control_gates(
    model: torch.nn.Module,
    control_gates: ControlGateSelection,
    *,
    device: torch.device,
) -> dict[str, torch.nn.Module]:
    """Create training-only diagnostic gates."""
    hidden_size = int(_ouro_core(model).config.hidden_size)
    gates: dict[str, torch.nn.Module] = {}
    if "same_step" in _selected_control_gate_names(control_gates):
        gate = torch.nn.Linear(hidden_size, 1, device=device)
        gate.float()
        gate.train()
        gates["same_step"] = gate
    return gates


def _build_gate_optimizer(gates: dict[str, torch.nn.Module], config: GateTrainingConfig) -> AdamW:
    """Build AdamW with independent parameter groups for selected gates."""
    param_groups = []
    if "lookahead" in gates:
        param_groups.append(
            {"params": gates["lookahead"].parameters(), "lr": config.lookahead_learning_rate, "name": "lookahead"}
        )
    if "preloop" in gates:
        param_groups.append(
            {"params": gates["preloop"].parameters(), "lr": config.preloop_learning_rate, "name": "preloop"}
        )
    if "same_step" in gates:
        param_groups.append(
            {"params": gates["same_step"].parameters(), "lr": config.lookahead_learning_rate, "name": "same_step"}
        )
    if not param_groups:
        raise ValueError("At least one gate must be selected for training")
    return AdamW(param_groups)


def _learning_rate_scale(
    *,
    step: int,
    max_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
    schedule: LearningRateSchedule,
) -> float:
    """Return the multiplicative LR scale for a one-indexed optimizer step."""
    if step < 1:
        raise ValueError("step must be one-indexed")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be between 0.0 and 1.0")
    if schedule == "constant":
        return 1.0
    if schedule != "warmup_cosine":
        raise ValueError(f"Unsupported learning rate schedule: {schedule}")

    bounded_step = min(step, max_steps)
    if warmup_steps > 0 and bounded_step <= warmup_steps:
        return bounded_step / warmup_steps
    cosine_steps = max(max_steps - warmup_steps, 1)
    progress = min(max(bounded_step - warmup_steps, 0) / cosine_steps, 1.0)
    cosine_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_scale


def _set_optimizer_learning_rates(
    optimizer: AdamW,
    *,
    step: int,
    config: GateTrainingConfig,
) -> dict[str, float]:
    """Apply the configured LR schedule and return current group LRs."""
    scale = _learning_rate_scale(
        step=step,
        max_steps=config.max_steps,
        warmup_steps=config.warmup_steps,
        min_lr_ratio=config.min_lr_ratio,
        schedule=config.learning_rate_schedule,
    )
    learning_rates: dict[str, float] = {}
    for group in optimizer.param_groups:
        group_name = str(group.get("name", len(learning_rates)))
        base_lr = float(group.get("initial_lr", group["lr"]))
        group["initial_lr"] = base_lr
        group["lr"] = base_lr * scale
        learning_rates[group_name] = group["lr"]
    return learning_rates


def _clip_gate_gradients(gates: dict[str, torch.nn.Module], max_grad_norm: float) -> dict[str, float]:
    """Clip each gate independently so joint training matches separate runs."""
    if max_grad_norm <= 0.0:
        return {}

    grad_norms: dict[str, float] = {}
    for gate_name, gate in gates.items():
        parameters = [
            parameter for parameter in gate.parameters() if parameter.requires_grad and parameter.grad is not None
        ]
        if not parameters:
            continue
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
        grad_norms[gate_name] = float(grad_norm.item())
    return grad_norms


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Move a token batch to the training device."""
    return {key: value.to(device=device, non_blocking=True) for key, value in batch.items()}


def _maybe_init_wandb(config: GateTrainingConfig) -> object | None:
    """Start a W&B run unless logging has been disabled."""
    if config.wandb_project is None:
        return None
    import wandb

    return wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        mode=config.wandb_mode,
        config={key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
    )


def _save_gate(
    *,
    model: torch.nn.Module,
    control_gates: dict[str, torch.nn.Module] | None = None,
    output_dir: Path,
    config: GateTrainingConfig,
    step: int,
) -> None:
    """Save selected gate weights and shared training metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for gate_name in _selected_gate_names(config.train_gates):
        gate = _require_gate(model, gate_name)
        state_dict = {key: value.detach().cpu() for key, value in gate.state_dict().items()}
        save_file(state_dict, output_dir / f"{gate_name}_exit_gate.safetensors")
    control_gates = {} if control_gates is None else control_gates
    for gate_name in _selected_control_gate_names(config.control_gates):
        gate = control_gates[gate_name]
        state_dict = {key: value.detach().cpu() for key, value in gate.state_dict().items()}
        save_file(state_dict, output_dir / f"{gate_name}_exit_gate.safetensors")
    metadata = asdict(config)
    metadata["output_dir"] = str(metadata["output_dir"])
    metadata["step"] = step
    metadata["data"] = {
        "dataset": NEMOTRON_DATASET_NAME,
        "dataset_revision": NEMOTRON_DATASET_REVISION,
        "categories": list(NEMOTRON_CATEGORIES),
        "validation_packed_sequences": VALIDATION_PACKED_SEQUENCES,
        "validation_seed": VALIDATION_SEED,
    }
    (output_dir / "gate_training_config.json").write_text(json.dumps(metadata, indent=2) + "\n")


def _prefixed_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    """Prefix a dictionary of scalar metrics for W&B grouping."""
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


def _selected_hazard_gate_names(gates: dict[str, torch.nn.Module]) -> tuple[str, ...]:
    """Return selected scalar hazard gate names."""
    return tuple(gate_name for gate_name in HAZARD_GATE_TARGET_OFFSETS if gate_name in gates)


def _selected_preloop_gate_names(gates: dict[str, torch.nn.Module]) -> tuple[str, ...]:
    """Return selected full-distribution pre-loop gate names."""
    return ("preloop",) if "preloop" in gates else ()


def _compute_gate_losses(
    *,
    gates: dict[str, torch.nn.Module],
    hazard_gate_names: tuple[str, ...],
    preloop_gate_names: tuple[str, ...],
    hidden_states_list: list[torch.Tensor],
    teacher_gate_list: list[torch.Tensor],
    preloop_hidden: torch.Tensor,
    loss_mask: torch.Tensor,
) -> tuple[dict[str, LookaheadBatchMetrics], dict[str, PreloopBatchMetrics], list[torch.Tensor]]:
    """Compute all selected gate losses for one batch."""
    step_losses: list[torch.Tensor] = []
    hazard_metrics: dict[str, LookaheadBatchMetrics] = {}
    for gate_name in hazard_gate_names:
        metrics = hazard_gate_loss(
            hidden_states_list=hidden_states_list,
            teacher_gate_list=teacher_gate_list,
            hazard_gate=gates[gate_name],
            target_step_offset=HAZARD_GATE_TARGET_OFFSETS[gate_name],
            loss_mask=loss_mask,
        )
        hazard_metrics[gate_name] = metrics
        step_losses.append(metrics.loss)

    preloop_metrics: dict[str, PreloopBatchMetrics] = {}
    for gate_name in preloop_gate_names:
        metrics = preloop_gate_loss(
            preloop_hidden=preloop_hidden,
            teacher_gate_list=teacher_gate_list,
            preloop_gate=gates[gate_name],
            loss_mask=loss_mask,
        )
        preloop_metrics[gate_name] = metrics
        step_losses.append(metrics.loss)

    if not step_losses:
        raise ValueError("No gate losses were selected")
    return hazard_metrics, preloop_metrics, step_losses


def _batch_data_counts(batch: dict[str, torch.Tensor]) -> dict[str, int]:
    """Return raw data-count diagnostics for a batch."""
    source_counts = batch["source_counts"].detach().cpu().sum(dim=0)
    counts = {f"source_tokens/{ID_TO_SOURCE[index]}": int(value.item()) for index, value in enumerate(source_counts)}
    counts["assistant_loss_tokens"] = int(batch["loss_mask"].sum().item())
    counts["tokens"] = int(batch["attention_mask"].sum().item())
    return counts


def _data_logs_from_counts(counts: dict[str, int], *, prefix: str = "data") -> dict[str, float | int]:
    """Format raw count diagnostics for W&B logging."""
    total_tokens = max(int(counts.get("tokens", 0)), 1)
    logs: dict[str, float | int] = {
        f"{prefix}/assistant_loss_tokens": int(counts.get("assistant_loss_tokens", 0)),
        f"{prefix}/assistant_loss_fraction": int(counts.get("assistant_loss_tokens", 0)) / total_tokens,
    }
    for key, value in sorted(counts.items()):
        if not key.startswith("source_tokens/"):
            continue
        source = key.rsplit("/", maxsplit=1)[-1]
        logs[f"{prefix}/source_tokens/{source}"] = int(value)
        logs[f"{prefix}/source_fraction/{source}"] = int(value) / total_tokens
    return logs


@torch.no_grad()
def _run_validation(
    *,
    eval_loader: torch.utils.data.DataLoader,
    device: torch.device,
    ouro_model: torch.nn.Module,
    gates: dict[str, torch.nn.Module],
    hazard_gate_names: tuple[str, ...],
    preloop_gate_names: tuple[str, ...],
) -> dict[str, float | int]:
    """Run one full fixed-validation pass and return weighted metrics."""
    weighted_logs: dict[str, float] = {}
    total_weight = 0.0

    for batch in eval_loader:
        batch = _move_batch(batch, device)
        loss_mask = batch["loss_mask"]
        weight = float(loss_mask.sum().clamp_min(1).item())
        _, hidden_states_list, teacher_gate_list, preloop_hidden = ouro_model(
            input_ids=batch["input_ids"],
            attention_mask=None,
            position_ids=batch["position_ids"],
            use_cache=False,
            use_early_exit_gate=True,
            return_preloop_hidden=True,
        )
        hazard_metrics, preloop_metrics, _step_losses = _compute_gate_losses(
            gates=gates,
            hazard_gate_names=hazard_gate_names,
            preloop_gate_names=preloop_gate_names,
            hidden_states_list=hidden_states_list,
            teacher_gate_list=teacher_gate_list,
            preloop_hidden=preloop_hidden,
            loss_mask=loss_mask,
        )
        batch_logs: dict[str, float] = {}
        for gate_name, metrics in hazard_metrics.items():
            batch_logs[f"{gate_name}/eval/loss"] = float(metrics.loss.item())
            batch_logs[f"{gate_name}/eval/hazard_mae"] = float(metrics.hazard_mae.item())
            batch_logs.update(
                _prefixed_metrics(
                    gate_name,
                    hazard_qexit_metrics(
                        hidden_states_list=hidden_states_list,
                        teacher_gate_list=teacher_gate_list,
                        hazard_gate=gates[gate_name],
                        first_step_index=HAZARD_GATE_TARGET_OFFSETS[gate_name],
                        thresholds=QEXIT_THRESHOLDS,
                        loss_mask=loss_mask,
                    ),
                )
            )
        for gate_name, metrics in preloop_metrics.items():
            batch_logs[f"{gate_name}/eval/loss"] = float(metrics.loss.item())
            batch_logs[f"{gate_name}/eval/pdf_mae"] = float(metrics.pdf_mae.item())
            batch_logs[f"{gate_name}/eval/cdf_mae"] = float(metrics.cdf_mae.item())
            batch_logs[f"{gate_name}/eval/kl"] = float(metrics.kl.item())
            batch_logs.update(
                _prefixed_metrics(
                    gate_name,
                    preloop_qexit_metrics(
                        preloop_hidden=preloop_hidden,
                        teacher_gate_list=teacher_gate_list,
                        preloop_gate=gates[gate_name],
                        thresholds=QEXIT_THRESHOLDS,
                        loss_mask=loss_mask,
                    ),
                )
            )

        for key, value in batch_logs.items():
            weighted_logs[key] = weighted_logs.get(key, 0.0) + value * weight
        total_weight += weight

    if total_weight == 0.0:
        raise RuntimeError(
            "Validation produced no assistant-loss tokens; check the validation split and chat template."
        )

    logs: dict[str, float | int] = {key: value / max(total_weight, 1.0) for key, value in weighted_logs.items()}
    return logs


def _format_log_lines(logs: dict[str, float | int], *, step: int, max_steps: int) -> list[str]:
    """Format structured metrics into compact stdout progress lines."""
    lines: list[str] = []
    train_parts: list[str] = []
    train_fields = (
        ("train/tokens", "tokens", ".3g"),
        ("time/tokens_per_second", "tok/s", ".0f"),
        ("time/seconds_per_step", "s/step", ".2f"),
        ("time/eta_hours", "eta", ".2f"),
        ("learning_rate/lookahead", "lookahead lr", ".3g"),
        ("learning_rate/preloop", "preloop lr", ".3g"),
        ("learning_rate/same_step", "same_step lr", ".3g"),
    )
    for key, label, spec in train_fields:
        value = logs.get(key)
        if not isinstance(value, int | float) or not math.isfinite(value):
            continue
        if not train_parts:
            train_parts.append(f"train step {step}/{max_steps}")
        suffix = "h" if key == "time/eta_hours" else ""
        train_parts.append(f"{label} {value:{spec}}{suffix}")
    metric_labels = (
        ("lookahead/train/loss", "lookahead loss", ".4f"),
        ("lookahead/train/hazard_mae", "lookahead mae", ".4f"),
        ("same_step/train/loss", "same_step loss", ".4f"),
        ("same_step/train/hazard_mae", "same_step mae", ".4f"),
        ("preloop/train/loss", "preloop loss", ".4f"),
        ("preloop/train/pdf_mae", "preloop pdf_mae", ".4f"),
        ("preloop/train/cdf_mae", "preloop cdf_mae", ".4f"),
        ("preloop/train/kl", "preloop kl", ".4f"),
        ("data/assistant_loss_fraction", "assistant", ".1%"),
    )
    for key, label, spec in metric_labels:
        value = logs.get(key)
        if not isinstance(value, int | float) or not math.isfinite(value):
            continue
        if not train_parts:
            train_parts.append(f"train step {step}/{max_steps}")
        train_parts.append(f"{label} {value:{spec}}")
    source_parts: list[str] = []
    for key in sorted(logs):
        if not key.startswith("data/source_fraction/"):
            continue
        value = logs[key]
        if isinstance(value, int | float) and math.isfinite(value):
            source = key.rsplit("/", maxsplit=1)[-1]
            source_parts.append(f"{source} {100.0 * value:.1f}%")
    if source_parts:
        if not train_parts:
            train_parts.append(f"train step {step}/{max_steps}")
        train_parts.append(f"source {', '.join(source_parts)}")
    if train_parts:
        lines.append(" | ".join(train_parts))

    eval_parts: list[str] = []
    for key in sorted(logs):
        if "/qexit/" not in key:
            continue
        if "exit_frac_depth" in key:
            continue
        value = logs[key]
        if isinstance(value, int | float) and math.isfinite(value):
            if not eval_parts:
                eval_parts.append(f"eval step {step}/{max_steps}")
            label = key.replace("/qexit/", " ").removeprefix("lookahead/").removeprefix("preloop/")
            eval_parts.append(f"{label} {value:.3f}")
    if eval_parts:
        lines.append(" | ".join(eval_parts))

    return lines


def train_lookahead_gate(config: GateTrainingConfig) -> None:
    """Train selected Ouro exit gates by frozen-gate distillation."""
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = _maybe_init_wandb(config)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    train_loader, eval_loader = create_packed_dataloaders(
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        eval_batch_size=config.eval_batch_size or config.batch_size,
        sequence_length=config.sequence_length,
        seed=config.seed,
    )
    attn_implementation = require_flash_attention()
    print(f"Attention backend: {attn_implementation}", flush=True)
    model = OuroForCausalLM.from_pretrained(
        config.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    )
    model.config.use_cache = False
    model.to(device)
    real_gates = _initialize_trainable_gates(model, config.train_gates)
    control_gates = _initialize_control_gates(model, config.control_gates, device=device)
    gates = {**real_gates, **control_gates}
    optimizer = _build_gate_optimizer(gates, config)
    hazard_gate_names = _selected_hazard_gate_names(gates)
    preloop_gate_names = _selected_preloop_gate_names(gates)

    batches = iter(train_loader)
    epoch = 0
    running_hazard_loss = dict.fromkeys(hazard_gate_names, 0.0)
    running_hazard_mae = dict.fromkeys(hazard_gate_names, 0.0)
    running_preloop_loss = dict.fromkeys(preloop_gate_names, 0.0)
    running_preloop_pdf_mae = dict.fromkeys(preloop_gate_names, 0.0)
    running_preloop_cdf_mae = dict.fromkeys(preloop_gate_names, 0.0)
    running_preloop_kl = dict.fromkeys(preloop_gate_names, 0.0)
    running_weight = 0.0
    completed_steps = 0
    completed_tokens = 0
    train_start_time = time.monotonic()
    last_log_time = train_start_time
    last_log_step = 0
    last_log_tokens = 0

    ouro_model = _ouro_core(model)
    if config.eval_every > 0:
        model.eval()
        baseline_eval_metrics = _run_validation(
            eval_loader=eval_loader,
            device=device,
            ouro_model=ouro_model,
            gates=gates,
            hazard_gate_names=hazard_gate_names,
            preloop_gate_names=preloop_gate_names,
        )
        for line in _format_log_lines(baseline_eval_metrics, step=0, max_steps=config.max_steps):
            print(line, flush=True)
        if run is not None:
            run.log(baseline_eval_metrics, step=0)

    while completed_steps < config.max_steps:
        model.eval()
        for gate in gates.values():
            gate.train()
        try:
            batch = _move_batch(next(batches), device)
        except StopIteration:
            # The packed dataset is single-pass; restart it rather than dying
            # mid-run (with save_every=0 a StopIteration here would lose the
            # whole run: the only _save_gate call sits after this loop).
            epoch += 1
            print(f"train data exhausted after {completed_steps} steps; starting epoch {epoch + 1}", flush=True)
            batches = iter(train_loader)
            batch = _move_batch(next(batches), device)
        loss_mask = batch["loss_mask"]
        with torch.no_grad():
            _, hidden_states_list, teacher_gate_list, preloop_hidden = ouro_model(
                input_ids=batch["input_ids"],
                attention_mask=None,
                position_ids=batch["position_ids"],
                use_cache=False,
                use_early_exit_gate=True,
                return_preloop_hidden=True,
            )
        hazard_metrics, preloop_metrics, step_losses = _compute_gate_losses(
            gates=gates,
            hazard_gate_names=hazard_gate_names,
            preloop_gate_names=preloop_gate_names,
            hidden_states_list=hidden_states_list,
            teacher_gate_list=teacher_gate_list,
            preloop_hidden=preloop_hidden,
            loss_mask=loss_mask,
        )
        current_learning_rates = _set_optimizer_learning_rates(
            optimizer,
            step=completed_steps + 1,
            config=config,
        )
        torch.stack(step_losses).sum().backward()
        current_grad_norms = _clip_gate_gradients(gates, config.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        completed_steps += 1
        completed_tokens += int(batch["attention_mask"].sum().item())
        batch_weight = float(loss_mask.sum().clamp_min(1).item())
        for gate_name, metrics in hazard_metrics.items():
            running_hazard_loss[gate_name] += float(metrics.loss.detach().item()) * batch_weight
            running_hazard_mae[gate_name] += float(metrics.hazard_mae.detach().item()) * batch_weight
        for gate_name, metrics in preloop_metrics.items():
            running_preloop_loss[gate_name] += float(metrics.loss.detach().item()) * batch_weight
            running_preloop_pdf_mae[gate_name] += float(metrics.pdf_mae.detach().item()) * batch_weight
            running_preloop_cdf_mae[gate_name] += float(metrics.cdf_mae.detach().item()) * batch_weight
            running_preloop_kl[gate_name] += float(metrics.kl.detach().item()) * batch_weight
        running_weight += batch_weight

        should_log = completed_steps % config.log_every == 0 or completed_steps == 1
        should_eval = config.eval_every > 0 and completed_steps % config.eval_every == 0
        logs: dict[str, float | int] = {}
        if should_log:
            current_time = time.monotonic()
            interval_seconds = max(current_time - last_log_time, 1e-12)
            interval_steps = max(completed_steps - last_log_step, 1)
            interval_tokens = max(completed_tokens - last_log_tokens, 1)
            seconds_per_step = interval_seconds / interval_steps
            logs.update(
                {
                    "train/step": completed_steps,
                    "train/tokens": completed_tokens,
                    "time/seconds_per_step": seconds_per_step,
                    "time/tokens_per_second": interval_tokens / interval_seconds,
                    "time/elapsed_seconds": current_time - train_start_time,
                    "time/eta_hours": seconds_per_step * (config.max_steps - completed_steps) / 3600.0,
                }
            )
            logs.update(_data_logs_from_counts(_batch_data_counts(batch)))
            logs.update({f"learning_rate/{gate_name}": value for gate_name, value in current_learning_rates.items()})
            logs.update({f"grad_norm/{gate_name}": value for gate_name, value in current_grad_norms.items()})
            denom = max(running_weight, 1.0)
            for gate_name in hazard_gate_names:
                logs[f"{gate_name}/train/loss"] = running_hazard_loss[gate_name] / denom
                logs[f"{gate_name}/train/hazard_mae"] = running_hazard_mae[gate_name] / denom
            for gate_name in preloop_gate_names:
                logs[f"{gate_name}/train/loss"] = running_preloop_loss[gate_name] / denom
                logs[f"{gate_name}/train/pdf_mae"] = running_preloop_pdf_mae[gate_name] / denom
                logs[f"{gate_name}/train/cdf_mae"] = running_preloop_cdf_mae[gate_name] / denom
                logs[f"{gate_name}/train/kl"] = running_preloop_kl[gate_name] / denom
            running_hazard_loss = dict.fromkeys(hazard_gate_names, 0.0)
            running_hazard_mae = dict.fromkeys(hazard_gate_names, 0.0)
            running_preloop_loss = dict.fromkeys(preloop_gate_names, 0.0)
            running_preloop_pdf_mae = dict.fromkeys(preloop_gate_names, 0.0)
            running_preloop_cdf_mae = dict.fromkeys(preloop_gate_names, 0.0)
            running_preloop_kl = dict.fromkeys(preloop_gate_names, 0.0)
            running_weight = 0.0
            last_log_time = current_time
            last_log_step = completed_steps
            last_log_tokens = completed_tokens

        if should_eval:
            model.eval()
            logs.update(
                _run_validation(
                    eval_loader=eval_loader,
                    device=device,
                    ouro_model=ouro_model,
                    gates=gates,
                    hazard_gate_names=hazard_gate_names,
                    preloop_gate_names=preloop_gate_names,
                )
            )

        if logs:
            for line in _format_log_lines(logs, step=completed_steps, max_steps=config.max_steps):
                print(line, flush=True)
            if run is not None:
                run.log(logs, step=completed_steps)

        if config.save_every > 0 and completed_steps % config.save_every == 0:
            _save_gate(
                model=model,
                control_gates=control_gates,
                output_dir=config.output_dir / f"checkpoint-{completed_steps}",
                config=config,
                step=completed_steps,
            )

    _save_gate(
        model=model,
        control_gates=control_gates,
        output_dir=config.output_dir,
        config=config,
        step=completed_steps,
    )
    if run is not None:
        run.finish()
