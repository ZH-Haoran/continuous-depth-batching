from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from ouro_test_helpers import tiny_ouro_config
from safetensors.torch import save_file

from looped_cdb.continuous_depth_batching.adapters import OuroCDBAdapter
from looped_cdb.continuous_depth_batching.adapters.ouro import configure_ouro_exit_gate
from looped_cdb.continuous_depth_batching.continuous_api import ContinuousDepthBatchingEngine
from looped_cdb.continuous_depth_batching.exit_policy import (
    DecisionHorizon,
    ExitDecisionRule,
    ExitPolicy,
    ExitPolicySpec,
)
from looped_cdb.models.ouro.modeling_ouro import OuroForCausalLM


def test_ouro_cdb_full_depth_staged_api_matches_forward_without_cache() -> None:
    torch.manual_seed(0)
    model = OuroForCausalLM(tiny_ouro_config())
    adapter = OuroCDBAdapter(model)
    model.eval()
    input_ids = torch.tensor([[4, 5, 6]])
    position_ids = torch.arange(input_ids.size(1)).unsqueeze(0)

    with torch.no_grad():
        forward_logits = model(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=False,
            use_early_exit_gate=True,
            exit_at_step=model.config.total_ut_steps - 1,
        ).logits

        hidden_states = adapter.prelude(input_ids=input_ids, position_ids=position_ids)
        for recurrent_step in range(model.config.total_ut_steps):
            hidden_states, exit_signals = adapter.recurrent_step(
                hidden_states=hidden_states,
                position_ids=position_ids,
                recurrent_steps=torch.full((input_ids.size(1),), recurrent_step, dtype=torch.long),
                attention_mask=None,
            )
            assert exit_signals is not None
        staged_logits = adapter.lm_head(hidden_states)

    assert torch.allclose(staged_logits, forward_logits, atol=1e-5)


def test_ouro_adapter_recurrent_step_uses_configured_loop_count() -> None:
    model = OuroForCausalLM(tiny_ouro_config(total_ut_steps=3))
    adapter = OuroCDBAdapter(model)
    adapter.configure_recurrent_steps(1)
    hidden_states = adapter.prelude(input_ids=torch.tensor([[4]]), position_ids=torch.tensor([[0]]))

    with torch.no_grad():
        adapter.recurrent_step(
            hidden_states=hidden_states,
            position_ids=torch.tensor([[0]]),
            recurrent_steps=torch.tensor([0]),
            attention_mask=None,
        )

    with torch.no_grad(), pytest.raises(ValueError, match="recurrent_step"):
        adapter.recurrent_step(
            hidden_states=hidden_states,
            position_ids=torch.tensor([[0]]),
            recurrent_steps=torch.tensor([1]),
            attention_mask=None,
        )


def test_ouro_adapter_recurrent_step_accepts_use_cache_kwarg() -> None:
    model = OuroForCausalLM(tiny_ouro_config(total_ut_steps=1))
    adapter = OuroCDBAdapter(model)
    hidden_states = adapter.prelude(input_ids=torch.tensor([[4]]), position_ids=torch.tensor([[0]]))

    with torch.no_grad():
        output, _ = adapter.recurrent_step(
            hidden_states=hidden_states,
            position_ids=torch.tensor([[0]]),
            recurrent_steps=torch.tensor([0]),
            attention_mask=None,
            use_cache=False,
        )

    assert output.shape == hidden_states.shape


def _distinct_gate_model() -> OuroForCausalLM:
    """An Ouro model whose early-exit and lookahead gates emit distinguishable logits."""
    model = OuroForCausalLM(tiny_ouro_config(total_ut_steps=2))
    with torch.no_grad():
        model.model.early_exit_gate.weight.zero_()
        model.model.early_exit_gate.bias.fill_(-5.0)
        model.model.lookahead_exit_gate.weight.zero_()
        model.model.lookahead_exit_gate.bias.fill_(5.0)
    return model


def _run_gate(adapter: OuroCDBAdapter, hidden_states: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        _, exit_signals = adapter.recurrent_step(
            hidden_states=hidden_states,
            position_ids=torch.tensor([[0]]),
            recurrent_steps=torch.tensor([0]),
            attention_mask=None,
        )
    return exit_signals


def test_ouro_adapter_recurrent_step_selects_gate_by_exit_gate_type() -> None:
    # The gate is bound at adapter construction (gate loading happens before the adapter is
    # built), so selection is exercised by building one adapter per configured gate type.
    early_model = _distinct_gate_model()
    early_model.config.exit_gate_type = "early_exit"
    lookahead_model = _distinct_gate_model()
    lookahead_model.config.exit_gate_type = "lookahead"

    early_adapter = OuroCDBAdapter(early_model)
    lookahead_adapter = OuroCDBAdapter(lookahead_model)
    hidden_states = early_adapter.prelude(input_ids=torch.tensor([[4]]), position_ids=torch.tensor([[0]]))

    early_logits = _run_gate(early_adapter, hidden_states)
    lookahead_logits = _run_gate(lookahead_adapter, hidden_states)

    # The two gates carry opposite biases, so selection is observable in the logit sign.
    assert torch.all(early_logits < 0)
    assert torch.all(lookahead_logits > 0)
    assert early_adapter.exit_policy_spec() == ExitPolicySpec(
        ExitDecisionRule.CUMULATIVE_HAZARD,
        DecisionHorizon.SAME_STEP,
    )
    assert lookahead_adapter.exit_policy_spec() == ExitPolicySpec(
        ExitDecisionRule.CUMULATIVE_HAZARD,
        DecisionHorizon.OFFSET_ONE,
    )


def test_ouro_adapter_rejects_an_unknown_gate_type() -> None:
    # A config naming a gate the model does not implement must fail at adapter
    # construction rather than falling back to some default gate at serving time.
    config = tiny_ouro_config()
    config.exit_gate_type = "not_a_gate"
    model = OuroForCausalLM(config)

    with pytest.raises(NotImplementedError, match="not_a_gate"):
        OuroCDBAdapter(model)


def test_ouro_adapter_rejects_a_budget_deeper_than_the_preloop_head() -> None:
    # The head emits one probability per trained depth, so a deeper budget has no
    # probability to route into and would silently cap every token at the trained depth.
    config = tiny_ouro_config(total_ut_steps=4)
    config.exit_gate_type = "preloop"
    adapter = OuroCDBAdapter(OuroForCausalLM(config))

    with pytest.raises(ValueError, match="cannot route beyond 4 recurrent steps"):
        adapter.configure_recurrent_steps(6)

    adapter.configure_recurrent_steps(4)  # at the trained depth, no raise
    adapter.configure_recurrent_steps(2)  # shorter budgets truncate, which is fine


def test_ouro_adapter_serves_the_preloop_gate_without_step_logits() -> None:
    # The preloop gate decides from the prelude output, so the recurrent stage produces no
    # gate logits and the exit distribution comes from preloop_exit_pdf instead.
    config = tiny_ouro_config(total_ut_steps=3)
    config.exit_gate_type = "preloop"
    adapter = OuroCDBAdapter(OuroForCausalLM(config))

    hidden_states = torch.randn(1, 2, config.hidden_size)
    exit_pdf = adapter.preloop_exit_pdf(hidden_states)

    assert exit_pdf is not None
    assert exit_pdf.shape[-1] == config.total_ut_steps
    assert torch.allclose(exit_pdf.sum(dim=-1), torch.ones(exit_pdf.shape[:-1]), atol=1e-5)
    assert adapter._exit_gate(hidden_states) is None


def test_ouro_adapter_gate_type_and_bound_module_cannot_disagree() -> None:
    # The adapter binds its gate module at construction, so the served semantics must be
    # bound from the same read. A later config edit that changed only one of them would
    # leave the engine expecting per-step logits the bound module never produces.
    config = tiny_ouro_config(total_ut_steps=3)
    config.exit_gate_type = "preloop"
    adapter = OuroCDBAdapter(OuroForCausalLM(config))

    adapter.decoder.config.exit_gate_type = "early_exit"

    hidden_states = torch.randn(1, 2, config.hidden_size)
    assert adapter.decides_exit_before_loop()
    assert adapter._exit_gate(hidden_states) is None
    assert adapter.preloop_exit_pdf(hidden_states) is not None


def _save_lookahead_gate(tmp_path: Path) -> tuple[OuroForCausalLM, Path]:
    """Build a source model with distinctive lookahead weights and save that gate."""
    source = OuroForCausalLM(tiny_ouro_config(total_ut_steps=2))
    with torch.no_grad():
        source.model.lookahead_exit_gate.weight.fill_(0.3)
        source.model.lookahead_exit_gate.bias.fill_(-0.7)
    gate_path = tmp_path / "lookahead_exit_gate.safetensors"
    save_file(dict(source.model.lookahead_exit_gate.state_dict()), gate_path)
    return source, gate_path


def test_configure_ouro_exit_gate_loads_lookahead_checkpoint(tmp_path: Path) -> None:
    source, gate_path = _save_lookahead_gate(tmp_path)

    target = OuroForCausalLM(tiny_ouro_config(total_ut_steps=2))
    configure_ouro_exit_gate(target, exit_gate_type="lookahead", exit_gate_path=gate_path)

    assert target.config.exit_gate_type == "lookahead"
    assert torch.allclose(target.model.lookahead_exit_gate.weight, source.model.lookahead_exit_gate.weight)
    assert torch.allclose(target.model.lookahead_exit_gate.bias, source.model.lookahead_exit_gate.bias)


def test_configure_ouro_exit_gate_path_without_type_uses_config_gate_type(tmp_path: Path) -> None:
    # A path-only request must not default to early_exit and load lookahead weights into the
    # wrong module: it falls back to the checkpoint's declared gate type.
    source, gate_path = _save_lookahead_gate(tmp_path)

    target = OuroForCausalLM(tiny_ouro_config(total_ut_steps=2))
    target.config.exit_gate_type = "lookahead"
    configure_ouro_exit_gate(target, exit_gate_path=gate_path)

    assert target.config.exit_gate_type == "lookahead"
    # Weights land in the lookahead module, and the early-exit module is left untouched.
    assert torch.allclose(target.model.lookahead_exit_gate.weight, source.model.lookahead_exit_gate.weight)
    assert not torch.allclose(target.model.early_exit_gate.weight, source.model.lookahead_exit_gate.weight)


def test_configure_ouro_exit_gate_rejects_lookahead_without_checkpoint() -> None:
    # Serving lookahead without a trained checkpoint would use a randomly initialized gate.
    model = OuroForCausalLM(tiny_ouro_config(total_ut_steps=2))
    model.config.exit_gate_type = "early_exit"
    with pytest.raises(ValueError, match=r"lookahead.*exit_gate_path"):
        configure_ouro_exit_gate(model, exit_gate_type="lookahead")


def test_configure_ouro_exit_gate_allows_lookahead_without_path_when_config_declares_it() -> None:
    # A checkpoint that already declares lookahead ships its own trained weights, so the
    # path may be omitted (no random-init risk).
    model = OuroForCausalLM(tiny_ouro_config(total_ut_steps=2))
    model.config.exit_gate_type = "lookahead"
    configure_ouro_exit_gate(model, exit_gate_type="lookahead")
    assert model.config.exit_gate_type == "lookahead"


def test_configure_ouro_exit_gate_normalizes_same_step_alias() -> None:
    model = OuroForCausalLM(tiny_ouro_config())
    configure_ouro_exit_gate(model, exit_gate_type="same_step")
    assert model.config.exit_gate_type == "early_exit"


def test_configure_ouro_exit_gate_rejects_an_unknown_gate_type() -> None:
    model = OuroForCausalLM(tiny_ouro_config())
    with pytest.raises(NotImplementedError, match="not_a_gate"):
        configure_ouro_exit_gate(model, exit_gate_type="not_a_gate")


def test_configure_ouro_exit_gate_requires_a_checkpoint_for_untrained_heads() -> None:
    # Only early_exit ships trained weights in the base checkpoint, so serving lookahead
    # or preloop without one would emit garbage under a name that implies training.
    for gate_type in ("lookahead", "preloop"):
        model = OuroForCausalLM(tiny_ouro_config())
        with pytest.raises(ValueError, match=f"{gate_type}.*requires exit_gate_path"):
            configure_ouro_exit_gate(model, exit_gate_type=gate_type)


def test_validate_exit_gate_serving_requires_delayed_consumption_for_offset_one() -> None:
    policy = ExitPolicy(
        spec=ExitPolicySpec(ExitDecisionRule.CUMULATIVE_HAZARD, DecisionHorizon.OFFSET_ONE),
        threshold=0.5,
        min_recurrent_steps=1,
        max_recurrent_steps=4,
        delay_gate_consumption=True,
        use_async_batching=False,
        synthetic_exit_replay=False,
    )
    engine = SimpleNamespace(exit_policy=policy)
    with pytest.raises(ValueError, match=r"offset-1.*delayed"):
        ContinuousDepthBatchingEngine._validate_exit_gate_serving(engine)

    engine.exit_policy = dataclasses.replace(policy, use_async_batching=True)
    ContinuousDepthBatchingEngine._validate_exit_gate_serving(engine)

    engine.exit_policy = dataclasses.replace(
        policy,
        spec=ExitPolicySpec(ExitDecisionRule.CUMULATIVE_HAZARD),
        use_async_batching=False,
    )
    ContinuousDepthBatchingEngine._validate_exit_gate_serving(engine)

    engine.exit_policy = dataclasses.replace(
        policy,
        spec=ExitPolicySpec(ExitDecisionRule.PRELOOP_DISTRIBUTION),
        use_async_batching=False,
    )
    ContinuousDepthBatchingEngine._validate_exit_gate_serving(engine)


def _assign_preloop_steps(exit_pdf, *, threshold, max_steps=4, min_steps=1, num_items=1):
    """Run the runner's preloop depth selection over a fixed exit distribution."""

    from looped_cdb.continuous_depth_batching.model_runner import ModelRunner

    policy = ExitPolicy(
        spec=ExitPolicySpec(ExitDecisionRule.PRELOOP_DISTRIBUTION),
        threshold=threshold,
        min_recurrent_steps=min_steps,
        max_recurrent_steps=max_steps,
        delay_gate_consumption=True,
        use_async_batching=True,
        synthetic_exit_replay=False,
    )
    runner = SimpleNamespace(
        exit_policy=policy,
        model_adapter=SimpleNamespace(preloop_exit_pdf=lambda hidden: exit_pdf),
    )
    items = [SimpleNamespace(preloop_exit_step=None) for _ in range(num_items)]
    ModelRunner._assign_preloop_exit_steps(runner, items, torch.zeros(1, num_items, 2))
    return [item.preloop_exit_step for item in items]


def test_preloop_exit_step_is_the_first_depth_reaching_the_threshold() -> None:
    # Cumulative mass is [0.2, 0.5, 0.9, 1.0], so the exit step is the first index at or
    # above the threshold, and a threshold no mass reaches falls back to the final step.
    exit_pdf = torch.tensor([[0.2, 0.3, 0.4, 0.1]])

    assert _assign_preloop_steps(exit_pdf, threshold=0.2) == [0]
    assert _assign_preloop_steps(exit_pdf, threshold=0.5) == [1]
    assert _assign_preloop_steps(exit_pdf, threshold=0.9) == [2]
    assert _assign_preloop_steps(exit_pdf, threshold=1.0) == [3]
    # min_recurrent_steps floors the choice even when the mass says exit sooner.
    assert _assign_preloop_steps(exit_pdf, threshold=0.2, min_steps=3) == [2]


def test_preloop_final_step_absorbs_mass_beyond_a_shorter_budget() -> None:
    # Serving fewer steps than the gate was trained for drops the tail, and the leftover
    # mass is absorbed by the final step rather than reweighted onto the earlier ones -
    # the same rule the hazard gates follow via their forced exit at the last step.
    # Truncated to two steps the cumulative mass is [0.2, 0.5].
    exit_pdf = torch.tensor([[0.2, 0.3, 0.4, 0.1]])

    assert _assign_preloop_steps(exit_pdf, threshold=0.2, max_steps=2) == [0]
    assert _assign_preloop_steps(exit_pdf, threshold=0.5, max_steps=2) == [1]
    # Above the surviving mass nothing reaches the threshold, so the token runs to the end.
    assert _assign_preloop_steps(exit_pdf, threshold=0.6, max_steps=2) == [1]
    # Truncation must not change a decision the budget can still express.
    assert _assign_preloop_steps(exit_pdf, threshold=0.2, max_steps=4) == [0]


def test_preloop_exit_steps_are_assigned_per_token() -> None:
    # Each row of the batch carries its own distribution, so tokens in one prelude batch
    # can be routed to different depths.
    exit_pdf = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

    assert _assign_preloop_steps(exit_pdf, threshold=0.5, num_items=2) == [0, 3]


def test_preloop_exit_steps_are_skipped_without_a_threshold_or_gate() -> None:
    exit_pdf = torch.tensor([[0.2, 0.3, 0.4, 0.1]])

    assert _assign_preloop_steps(exit_pdf, threshold=None) == [None]
    assert _assign_preloop_steps(None, threshold=0.5) == [None]


def test_preloop_gate_bypasses_the_delayed_consumption_machinery() -> None:
    # A preloop gate emits no per-step signals, so the delayed path must not arm itself
    # waiting for a readout that will never arrive.
    preloop_policy = ExitPolicy(
        spec=ExitPolicySpec(ExitDecisionRule.PRELOOP_DISTRIBUTION),
        threshold=0.5,
        min_recurrent_steps=1,
        max_recurrent_steps=4,
        delay_gate_consumption=True,
        use_async_batching=True,
        synthetic_exit_replay=False,
    )
    engine = SimpleNamespace(exit_policy=preloop_policy)
    engine._uses_delayed_gate_consumption = lambda kw: ContinuousDepthBatchingEngine._uses_delayed_gate_consumption(
        engine, kw
    )
    kwargs: dict[str, object] = {"use_early_exit_gate": True}

    assert not ContinuousDepthBatchingEngine._uses_delayed_gate_consumption(engine, kwargs)
    assert not ContinuousDepthBatchingEngine._uses_delayed_gate_readout(engine, kwargs)

    engine.exit_policy = dataclasses.replace(
        preloop_policy,
        spec=ExitPolicySpec(ExitDecisionRule.CUMULATIVE_HAZARD),
    )
    assert ContinuousDepthBatchingEngine._uses_delayed_gate_consumption(engine, kwargs)
