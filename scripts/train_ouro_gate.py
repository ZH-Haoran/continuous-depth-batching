from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from looped_cdb.train.ouro_gate.train import (
    GateTrainingConfig,
    train_lookahead_gate,
)


def _default(field_name: str) -> Any:
    """Read the argparse default from the training config dataclass."""
    return GateTrainingConfig.__dataclass_fields__[field_name].default


def parse_args() -> argparse.Namespace:
    """Parse the user-facing Ouro gate training arguments."""
    parser = argparse.ArgumentParser(description="Train Ouro lookahead and/or pre-loop exit gates.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-gates", choices=("both", "lookahead", "preloop"), default=_default("train_gates"))
    parser.add_argument(
        "--control-gates",
        choices=("none", "same_step"),
        default=_default("control_gates"),
    )
    parser.add_argument("--sequence-length", type=int, default=_default("sequence_length"))
    parser.add_argument("--batch-size", type=int, default=_default("batch_size"))
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--max-steps", type=int, default=_default("max_steps"))
    parser.add_argument("--lookahead-learning-rate", type=float, default=_default("lookahead_learning_rate"))
    parser.add_argument("--preloop-learning-rate", type=float, default=_default("preloop_learning_rate"))
    parser.add_argument(
        "--learning-rate-schedule",
        choices=("constant", "warmup_cosine"),
        default=_default("learning_rate_schedule"),
    )
    parser.add_argument("--warmup-steps", type=int, default=_default("warmup_steps"))
    parser.add_argument("--min-lr-ratio", type=float, default=_default("min_lr_ratio"))
    parser.add_argument("--max-grad-norm", type=float, default=_default("max_grad_norm"))
    parser.add_argument("--log-every", type=int, default=_default("log_every"))
    parser.add_argument("--eval-every", type=int, default=_default("eval_every"))
    parser.add_argument("--save-every", type=int, default=_default("save_every"))
    parser.add_argument("--seed", type=int, default=_default("seed"))
    parser.add_argument("--wandb-project", default=_default("wandb_project"))
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-mode")
    return parser.parse_args()


def main() -> None:
    """Run Ouro gate training from CLI arguments."""
    args = parse_args()
    config = GateTrainingConfig(
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
        train_gates=args.train_gates,
        control_gates=args.control_gates,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        max_steps=args.max_steps,
        lookahead_learning_rate=args.lookahead_learning_rate,
        preloop_learning_rate=args.preloop_learning_rate,
        learning_rate_schedule=args.learning_rate_schedule,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        max_grad_norm=args.max_grad_norm,
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
        seed=args.seed,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
    )
    train_lookahead_gate(config)


if __name__ == "__main__":
    main()
