"""Per-stage FLOP weights read off a served model, and the bound they imply."""

import pytest
import torch
from torch import nn

from looped_cdb.benchmarks.flop_bound import StageFlops, stage_flops, stage_flops_from_json
from looped_cdb.models.huginn.configuration_huginn import HuginnConfig
from looped_cdb.models.huginn.modeling_huginn import HuginnForCausalLM


def _huginn(*, prelude: int, core: int, coda: int) -> HuginnForCausalLM:
    config = HuginnConfig(
        n_embd=32,
        n_heads=4,
        n_layers=prelude + core + coda,
        n_layers_in_prelude=prelude,
        n_layers_in_recurrent_block=core,
        n_layers_in_coda=coda,
        mean_recurrence=4,
        block_size=64,
        vocab_size=128,
        intermediate_size=16,
        state_init="zero",
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    return HuginnForCausalLM(config).eval()


def _params(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def test_huginn_weights_split_along_the_prelude_core_coda_boundary() -> None:
    model = _huginn(prelude=2, core=2, coda=2)

    flops = stage_flops(model)

    transformer = model.transformer
    assert flops.fr == _params(transformer["core_block"]) + _params(transformer["adapter"])
    assert flops.f0 == (
        _params(transformer["prelude"])
        + _params(transformer["coda"])
        + _params(transformer["ln_f"])
        + _params(model.lm_head)
    )
    # Token embeddings are a lookup, not a matmul, so they are in neither stage.
    assert _params(transformer["wte"]) not in (flops.f0, flops.fr)


def test_dropping_the_boundary_layers_lowers_f0_and_raises_the_bound() -> None:
    # The layer-split ablation serves the same recorded schedule as 2-4-2, 1-4-1 and 0-4-0, so
    # the weights (and the bound each run should be measured against) must follow the split.
    balanced = stage_flops(_huginn(prelude=2, core=4, coda=2))
    trimmed = stage_flops(_huginn(prelude=1, core=4, coda=1))
    bare = stage_flops(_huginn(prelude=0, core=4, coda=0))

    assert balanced.f0 > trimmed.f0 > bare.f0
    assert balanced.fr == trimmed.fr == bare.fr
    bounds = [flops.ideal_speedup(max_depth=16, mean_depth=8.0) for flops in (balanced, trimmed, bare)]
    assert bounds[0] < bounds[1] < bounds[2]
    # With no boundary layers left the head still costs something, so even 0-4-0 stays below
    # the depth ratio.
    assert bounds[2] < 16 / 8.0


def test_the_bound_reduces_to_the_depth_ratio_only_without_boundary_stages() -> None:
    assert StageFlops(f0=0, fr=100).ideal_speedup(max_depth=4, mean_depth=2.0) == pytest.approx(2.0)
    # Boundary stages worth one core application each: (100 + 4 * 100) / (100 + 2 * 100).
    assert StageFlops(f0=100, fr=100).ideal_speedup(max_depth=4, mean_depth=2.0) == pytest.approx(5 / 3)


def test_a_model_with_no_known_stage_structure_is_rejected() -> None:
    # Guessing a split would silently publish a bound for a structure nobody verified.
    with pytest.raises(TypeError, match="stage structure"):
        stage_flops(nn.Linear(4, 4))


def test_weights_round_trip_through_json() -> None:
    flops = StageFlops(f0=17, fr=23)
    assert stage_flops_from_json(flops.to_json_dict()) == flops
