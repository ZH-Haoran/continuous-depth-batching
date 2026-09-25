"""Per-stage FLOP weights of a looped model, and the speed-up bound they imply.

A looped decoder pays two kinds of per-token cost. The recurrent core runs once per
recurrent step, so early exits remove it; the prelude, coda and output head run once per
generated token whatever the exit depth, so no exit can remove them. Writing ``F_r`` for the
FLOPs of one core application and ``F_0`` for the depth-independent remainder, decoding a
token at depth ``d`` costs ``F_0 + d * F_r`` against a full-depth ``F_0 + r_max * F_r``, so
the speed-up from adaptive depth is bounded by

    (F_0 + r_max * F_r) / (F_0 + d_bar * F_r)

for a mean exit depth ``d_bar``. The bound is the FLOP bound of the paper's eq. (1). Only
``F_0 / F_r`` matters, so both weights are counted in parameters: every stage's cost is
dominated by weight matmuls that touch each parameter once per token, and the ratio of
parameter counts is the ratio of FLOPs. Token embeddings are a lookup rather than a matmul
and contribute nothing.

The weights are read off the loaded model and not tabulated per checkpoint, because the
prelude-core-coda split is a configuration: serving Huginn as 0-4-0 or 1-4-1 changes ``F_0``
and therefore changes the bound the run should be measured against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import nn


def _parameter_count(module: nn.Module | None) -> int:
    """Parameters of ``module``, counting a tied weight once per module that applies it."""

    if module is None:
        return 0
    return sum(parameter.numel() for parameter in module.parameters())


@dataclass(frozen=True)
class StageFlops:
    """Depth-independent and per-recursion FLOP weights of one served model."""

    #: Prelude, coda and output head: paid once per generated token at any exit depth.
    f0: int
    #: One application of the recurrent core: paid once per recurrent step.
    fr: int

    def __post_init__(self) -> None:
        if self.fr <= 0:
            raise ValueError(f"fr must be positive, got {self.fr}")
        if self.f0 < 0:
            raise ValueError(f"f0 must be non-negative, got {self.f0}")

    @property
    def f0_over_fr(self) -> float:
        """Depth-independent cost in units of one recurrent step."""

        return self.f0 / self.fr

    def ideal_speedup(self, *, max_depth: int, mean_depth: float) -> float:
        """The eq. (1) FLOP bound for a schedule with this mean exit depth.

        Reduces to ``max_depth / mean_depth`` only when the boundary stages are free.
        Huginn's are not: its prelude and coda together cost more than one core
        application, so its bound falls well below the depth ratio.
        """

        if mean_depth <= 0:
            raise ValueError(f"mean_depth must be positive, got {mean_depth}")
        if max_depth <= 0:
            raise ValueError(f"max_depth must be positive, got {max_depth}")
        return (self.f0 + max_depth * self.fr) / (self.f0 + mean_depth * self.fr)

    def to_json_dict(self) -> dict[str, int]:
        return {"f0_params": self.f0, "fr_params": self.fr}


def stage_flops(model: nn.Module) -> StageFlops:
    """Derive the FLOP weights of a served looped model from its stage modules.

    Raises for models whose stage structure is unknown, rather than guessing a split: a
    wrong split silently publishes a wrong bound.
    """

    transformer = getattr(model, "transformer", None)
    if transformer is not None and "core_block" in transformer:
        # Huginn: the adapter mixes the injected prelude output into the state at the top of
        # every recurrent step, so it belongs to the core rather than to the boundary stages.
        core = _parameter_count(transformer["adapter"]) + _parameter_count(transformer["core_block"])
        boundary_stages = (
            transformer["prelude"],
            transformer["coda"],
            transformer["ln_f"],
            getattr(model, "lm_head", None),
        )
        return StageFlops(f0=sum(_parameter_count(stage) for stage in boundary_stages), fr=core)

    decoder = model.get_decoder() if hasattr(model, "get_decoder") else getattr(model, "model", model)
    layers = getattr(decoder, "layers", None)
    if layers is not None and getattr(getattr(decoder, "config", None), "model_type", None) == "ouro":
        # Ouro is fully looped: every physical layer is a core layer, and the only
        # depth-independent weights are the final norm and the output head. The layer list can
        # carry extra paged-attention entries, so it is sliced the way the adapter runs it.
        num_hidden_layers = int(decoder.config.num_hidden_layers)
        core = sum(_parameter_count(layer) for layer in layers[:num_hidden_layers])
        boundary = _parameter_count(getattr(decoder, "norm", None)) + _parameter_count(getattr(model, "lm_head", None))
        return StageFlops(f0=boundary, fr=core)

    raise TypeError(
        f"cannot derive prelude-core-coda FLOP weights for {type(model).__name__}; "
        "add its stage structure to looped_cdb.benchmarks.flop_bound.stage_flops"
    )


def stage_flops_from_json(payload: dict[str, Any]) -> StageFlops:
    """Rebuild the weights a run recorded."""

    return StageFlops(f0=int(payload["f0_params"]), fr=int(payload["fr_params"]))
