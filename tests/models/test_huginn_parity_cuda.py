"""Pin the Huginn port's numerics to values recorded from the upstream release.

``tests/fixtures/huginn_golden.pt`` was recorded from the upstream
implementation while it was still vendored. The upstream files have since been
deleted, so this fixture is what keeps the port honest: it records per-stage
moments and fixed sampled elements for a deterministic input, plus the argmax
over every position.

Regenerating the fixture requires re-vendoring upstream and recording the same
stage outputs, so a failure here means the port changed behavior, not that the
fixture is stale.
"""

from __future__ import annotations

import os

import pytest
import torch
from repo_paths import FIXTURES_DIR

from looped_cdb.eval.model_loading import huginn_revision

pytestmark = pytest.mark.cuda

FIXTURE = FIXTURES_DIR / "huginn_golden.pt"
MODEL = os.environ.get("HUGINN_MODEL", "tomg-group-umd/huginn-0125")

# The port computes rotary values on device instead of loading the checkpoint's
# precomputed table, so agreement is to float32 round-off rather than bit-exact.
MOMENT_TOL = 1e-4
PROBE_TOL = 1e-3


def _probe(tensor: torch.Tensor, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flat = tensor.detach().float().flatten()
    moments = torch.stack([flat.mean(), flat.std(), flat.min(), flat.max()]).double().cpu()
    return moments, flat[indices.to(flat.device)].double().cpu()


@pytest.fixture(scope="module")
def golden() -> dict:
    if not FIXTURE.exists():
        pytest.skip(f"golden fixture missing: {FIXTURE}")
    return torch.load(FIXTURE, weights_only=False)


@pytest.fixture(scope="module")
def stage_outputs(golden: dict) -> dict[str, torch.Tensor]:
    """Run the port through the same stages the fixture recorded."""

    from looped_cdb.models.huginn import HuginnConfig, HuginnForCausalLM

    config = HuginnConfig(state_init="zero")
    config._attn_implementation = "sdpa"
    model = HuginnForCausalLM.from_pretrained(
        MODEL, config=config, revision=huginn_revision(MODEL), dtype=torch.float32
    )
    model = model.to("cuda").eval()

    input_ids = golden["input_ids"].to("cuda")
    outputs: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        freqs_cis = model.select_freqs_cis(input_ids.shape[1], position_ids=None)
        embeds = model.run_prelude(input_ids, freqs_cis)
        outputs["prelude"] = embeds

        state = torch.zeros_like(embeds)
        for step in range(golden["steps"]):
            state = model.run_core_step(state, embeds, freqs_cis, step)
            outputs[f"core_step_{step}"] = state

        hidden = model.run_coda(state, freqs_cis)
        outputs["logits"] = model.lm_head(hidden).float()
    return outputs


@pytest.mark.cuda
def test_all_recorded_stages_are_checked(golden: dict, stage_outputs: dict[str, torch.Tensor]) -> None:
    assert set(golden["stages"]) == set(stage_outputs)


@pytest.mark.cuda
@pytest.mark.parametrize("stage", ["prelude", "core_step_0", "core_step_1", "core_step_2", "core_step_3", "logits"])
def test_stage_matches_golden(golden: dict, stage_outputs: dict[str, torch.Tensor], stage: str) -> None:
    expected = golden["stages"][stage]
    moments, probes = _probe(stage_outputs[stage], expected["probe_indices"])

    torch.testing.assert_close(moments, expected["moments"], rtol=MOMENT_TOL, atol=MOMENT_TOL)
    torch.testing.assert_close(probes, expected["probe_values"], rtol=PROBE_TOL, atol=PROBE_TOL)


@pytest.mark.cuda
def test_argmax_matches_golden(golden: dict, stage_outputs: dict[str, torch.Tensor]) -> None:
    """Greedy decoding must be unchanged, which round-off tolerances alone would not catch."""

    argmax = stage_outputs["logits"].argmax(dim=-1).cpu()

    assert torch.equal(argmax, golden["argmax"])
