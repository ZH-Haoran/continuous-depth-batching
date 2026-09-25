from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from datasets import Dataset

import looped_cdb.utils
from looped_cdb.models.ouro.configuration_ouro import OuroConfig
from looped_cdb.models.ouro.exit_gates import EXIT_GATE_TYPES
from looped_cdb.models.ouro.modeling_ouro import OuroForCausalLM, OuroModel
from looped_cdb.train.common_data import strip_assistant_thinking_traces
from looped_cdb.train.ouro_gate import data as gate_data
from looped_cdb.train.ouro_gate.data import (
    PackedConversationDataset,
    _validate_assistant_mask_support,
    collate_packed_features,
    position_ids_from_seq_lengths,
    shifted_assistant_loss_mask,
)
from looped_cdb.train.ouro_gate.metrics import (
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
from looped_cdb.train.ouro_gate.train import (
    GateTrainingConfig,
    _build_gate_optimizer,
    _clip_gate_gradients,
    _format_log_lines,
    _initialize_control_gates,
    _initialize_trainable_gates,
    _learning_rate_scale,
    _save_gate,
    _set_optimizer_learning_rates,
)
from looped_cdb.utils import best_attention_backend, require_flash_attention

pytest.importorskip("trl")


class AssistantMaskTokenizer:
    eos_token_id = 0
    chat_template = "{% generation %}{{ content }}{% endgeneration %}"

    def __init__(self, *, mark_assistant: bool = True) -> None:
        self.mark_assistant = mark_assistant

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        return_dict: bool,
        return_assistant_tokens_mask: bool,
        add_generation_prompt: bool,
    ) -> dict[str, list[int]]:
        del tokenize, return_dict, return_assistant_tokens_mask, add_generation_prompt
        input_ids: list[int] = []
        assistant_masks: list[int] = []
        next_token_id = 1
        for message in messages:
            tokens = str(message["content"]).split()
            input_ids.extend(range(next_token_id, next_token_id + len(tokens)))
            next_token_id += len(tokens)
            mask_value = int(self.mark_assistant and message["role"] == "assistant")
            assistant_masks.extend([mask_value] * len(tokens))
        return {"input_ids": input_ids, "assistant_masks": assistant_masks}


class DummyOuroForCausalLM(torch.nn.Module):
    def __init__(self, model: OuroModel) -> None:
        super().__init__()
        self.model = model


def tiny_ouro_config(*, total_ut_steps: int = 3) -> OuroConfig:
    return OuroConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        total_ut_steps=total_ut_steps,
        layer_types=["full_attention"],
        pad_token_id=0,
    )


def test_lookahead_gate_initializes_from_frozen_gate() -> None:
    model = OuroModel(tiny_ouro_config(total_ut_steps=2))
    with torch.no_grad():
        model.early_exit_gate.weight.fill_(0.25)
        model.early_exit_gate.bias.fill_(-0.5)
        model.lookahead_exit_gate.weight.fill_(1.0)
        model.lookahead_exit_gate.bias.fill_(1.0)

    model.initialize_lookahead_exit_gate_from_frozen_gate()

    assert torch.equal(model.lookahead_exit_gate.weight, model.early_exit_gate.weight)
    assert torch.equal(model.lookahead_exit_gate.bias, model.early_exit_gate.bias)


def test_config_accepts_every_gate_the_model_implements() -> None:
    # The config validates against its own literal set rather than importing
    # EXIT_GATE_TYPES, which would pull torch into a module the package keeps
    # torch-free. Nothing else keeps the two in step. A config that rejects an
    # implemented gate fails at checkpoint load, so pin that direction; the reverse
    # fails safe, because the model rejects an unimplemented gate at forward.
    for gate_type in sorted(EXIT_GATE_TYPES):
        assert OuroConfig(exit_gate_type=gate_type).exit_gate_type == gate_type

    with pytest.raises(ValueError, match="exit_gate_type"):
        OuroConfig(exit_gate_type="not_a_gate")


def test_preloop_gate_shape_matches_total_steps() -> None:
    model = OuroModel(tiny_ouro_config(total_ut_steps=4))

    assert model.preloop_exit_gate.weight.shape == (4, model.config.hidden_size)
    assert model.preloop_exit_gate.bias.shape == (4,)


def test_preloop_exit_gate_accepts_fp32_hidden_with_bf16_weights() -> None:
    config = tiny_ouro_config(total_ut_steps=2)
    config.exit_gate_type = "preloop"
    config.early_exit_threshold = 0.5
    config.min_exit_step = 2
    model = OuroForCausalLM(config).to(dtype=torch.bfloat16)
    model.eval()

    with torch.no_grad():
        output = model(
            input_ids=torch.tensor([[1, 2, 3]]),
            use_early_exit_gate=True,
            return_exit_steps=True,
        )

    assert output.logits.dtype == torch.bfloat16
    assert output.ouro_exit_steps.shape == (1, 3)


def test_ouro_forward_optionally_returns_preloop_hidden() -> None:
    model = OuroModel(tiny_ouro_config(total_ut_steps=2))
    input_ids = torch.tensor([[1, 2, 3]])

    default_outputs = model(input_ids=input_ids, use_cache=False, use_early_exit_gate=True)
    extended_outputs = model(
        input_ids=input_ids,
        use_cache=False,
        use_early_exit_gate=True,
        return_preloop_hidden=True,
    )

    assert len(default_outputs) == 3
    assert len(extended_outputs) == 4
    assert extended_outputs[3].shape == (1, 3, model.config.hidden_size)


def test_ouro_model_forward_skips_gate_outputs_by_default() -> None:
    model = OuroModel(tiny_ouro_config(total_ut_steps=2))
    input_ids = torch.tensor([[1, 2, 3]])

    _outputs, hidden_states, gate_outputs = model(input_ids=input_ids, use_cache=False)

    assert hidden_states == []
    assert gate_outputs == []


def test_ouro_causal_lm_forward_does_not_call_exit_gate_by_default() -> None:
    class FailingGate(torch.nn.Module):
        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            raise AssertionError("early exit gate should not run on the default full-depth path")

    model = OuroForCausalLM(tiny_ouro_config(total_ut_steps=2))
    model.model.early_exit_gate = FailingGate()
    input_ids = torch.tensor([[1, 2, 3]])

    output = model(input_ids=input_ids, use_cache=False)

    assert output.logits.shape[:2] == input_ids.shape


def test_compute_exit_pdf_from_shifted_hazards_places_zero_mass_before_first_step() -> None:
    hazards = torch.tensor([[[0.25, 0.5]]])

    pdf = compute_exit_pdf_from_hazards(hazards, first_step_index=1, total_steps=4)

    assert torch.allclose(pdf, torch.tensor([[[0.0, 0.25, 0.375, 0.375]]]))


def test_lookahead_gate_loss_uses_next_step_teacher_hazard() -> None:
    gate = torch.nn.Linear(2, 1)
    with torch.no_grad():
        gate.weight.zero_()
        gate.bias.zero_()
    hidden_states = [
        torch.ones(1, 2, 2),
        torch.ones(1, 2, 2) * 2,
        torch.ones(1, 2, 2) * 3,
    ]
    teacher_gates = [
        torch.full((1, 2, 1), -10.0),
        torch.zeros(1, 2, 1),
        torch.full((1, 2, 1), 10.0),
    ]

    metrics = lookahead_gate_loss(
        hidden_states_list=hidden_states,
        teacher_gate_list=teacher_gates,
        lookahead_gate=gate,
        loss_mask=torch.tensor([[1, 0]]),
    )

    assert torch.isclose(metrics.teacher_mean_hazard, torch.tensor(0.5), atol=1e-4)
    assert torch.isclose(metrics.student_mean_hazard, torch.tensor(0.5), atol=1e-4)


def test_hazard_gate_loss_can_use_same_step_teacher_hazard() -> None:
    gate = torch.nn.Linear(2, 1)
    with torch.no_grad():
        gate.weight.zero_()
        gate.bias.zero_()
    hidden_states = [
        torch.ones(1, 2, 2),
        torch.ones(1, 2, 2) * 2,
        torch.ones(1, 2, 2) * 3,
    ]
    teacher_gates = [
        torch.full((1, 2, 1), -10.0),
        torch.zeros(1, 2, 1),
        torch.full((1, 2, 1), 10.0),
    ]

    metrics = hazard_gate_loss(
        hidden_states_list=hidden_states,
        teacher_gate_list=teacher_gates,
        hazard_gate=gate,
        target_step_offset=0,
        loss_mask=torch.tensor([[1, 0]]),
    )

    assert torch.isclose(metrics.teacher_mean_hazard, torch.tensor(0.25), atol=1e-4)
    assert torch.isclose(metrics.student_mean_hazard, torch.tensor(0.5), atol=1e-4)


def test_hazard_gate_loss_can_use_lookahead_teacher_hazard() -> None:
    gate = torch.nn.Linear(2, 1)
    with torch.no_grad():
        gate.weight.zero_()
        gate.bias.zero_()
    hidden_states = [
        torch.ones(1, 2, 2),
        torch.ones(1, 2, 2) * 2,
        torch.ones(1, 2, 2) * 3,
    ]
    teacher_gates = [
        torch.full((1, 2, 1), -10.0),
        torch.zeros(1, 2, 1),
        torch.full((1, 2, 1), 10.0),
    ]

    metrics = hazard_gate_loss(
        hidden_states_list=hidden_states,
        teacher_gate_list=teacher_gates,
        hazard_gate=gate,
        target_step_offset=1,
        loss_mask=torch.tensor([[1, 0]]),
    )

    assert torch.isclose(metrics.teacher_mean_hazard, torch.tensor(0.5), atol=1e-4)
    assert torch.isclose(metrics.student_mean_hazard, torch.tensor(0.5), atol=1e-4)


def test_hazard_gate_loss_ignores_final_forced_depth_hazard() -> None:
    gate = torch.nn.Linear(2, 1)
    with torch.no_grad():
        gate.weight.zero_()
        gate.bias.zero_()
    hidden_states = [
        torch.ones(1, 2, 2),
        torch.ones(1, 2, 2),
        torch.ones(1, 2, 2),
        torch.ones(1, 2, 2),
    ]
    teacher_gates = [
        torch.zeros(1, 2, 1),
        torch.full((1, 2, 1), -1.0),
        torch.full((1, 2, 1), 1.0),
        torch.full((1, 2, 1), -10.0),
    ]
    changed_final_teacher_gates = [
        teacher_gates[0],
        teacher_gates[1],
        teacher_gates[2],
        torch.full((1, 2, 1), 10.0),
    ]

    base_metrics = hazard_gate_loss(
        hidden_states_list=hidden_states,
        teacher_gate_list=teacher_gates,
        hazard_gate=gate,
        target_step_offset=1,
        loss_mask=torch.tensor([[1, 1]]),
    )
    changed_metrics = hazard_gate_loss(
        hidden_states_list=hidden_states,
        teacher_gate_list=changed_final_teacher_gates,
        hazard_gate=gate,
        target_step_offset=1,
        loss_mask=torch.tensor([[1, 1]]),
    )

    assert torch.isclose(changed_metrics.loss, base_metrics.loss)
    assert torch.isclose(changed_metrics.teacher_mean_hazard, base_metrics.teacher_mean_hazard)


def test_hazard_gate_loss_includes_exit_distribution_cdf() -> None:
    gate = torch.nn.Linear(2, 1)
    with torch.no_grad():
        gate.weight.zero_()
        gate.bias.zero_()
    hidden_states = [
        torch.ones(1, 2, 2),
        torch.ones(1, 2, 2) * 2,
        torch.ones(1, 2, 2) * 3,
    ]
    teacher_gates = [
        torch.full((1, 2, 1), -10.0),
        torch.full((1, 2, 1), 10.0),
        torch.full((1, 2, 1), -10.0),
    ]

    metrics = hazard_gate_loss(
        hidden_states_list=hidden_states,
        teacher_gate_list=teacher_gates,
        hazard_gate=gate,
        target_step_offset=1,
        loss_mask=torch.tensor([[1, 1]]),
    )
    teacher_hazards = torch.sigmoid(teacher_gates[1])
    hazard_only = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.zeros_like(teacher_hazards),
        teacher_hazards,
    )

    assert metrics.loss > hazard_only
    assert metrics.hazard_mae > 0


def test_preloop_gate_loss_is_zero_when_distribution_matches_teacher() -> None:
    teacher_gates = [
        torch.full((1, 2, 1), -1.3862944),
        torch.full((1, 2, 1), -0.2876821),
        torch.zeros(1, 2, 1),
    ]
    teacher_pdf = compute_teacher_exit_pdf(teacher_gates)
    gate = torch.nn.Linear(2, 3)
    with torch.no_grad():
        gate.weight.zero_()
        gate.bias.copy_(teacher_pdf[0, 0].log())

    metrics = preloop_gate_loss(
        preloop_hidden=torch.ones(1, 2, 2),
        teacher_gate_list=teacher_gates,
        preloop_gate=gate,
        loss_mask=torch.tensor([[1, 1]]),
    )

    assert torch.isclose(metrics.loss, torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(metrics.kl, torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(metrics.pdf_mae, torch.tensor(0.0), atol=1e-6)


def test_preloop_pdf_loss_masks_non_assistant_tokens() -> None:
    teacher_pdf = torch.tensor([[[0.25, 0.75], [0.95, 0.05]]])
    student_logits = torch.tensor([[[0.25, 0.75], [0.05, 0.95]]]).log()

    metrics = preloop_pdf_loss(
        student_logits=student_logits,
        teacher_pdf=teacher_pdf,
        loss_mask=torch.tensor([[1, 0]]),
    )

    assert torch.isclose(metrics.loss, torch.tensor(0.0), atol=1e-6)


def test_preloop_pdf_loss_includes_cdf_alignment() -> None:
    teacher_pdf = torch.tensor([[[0.2, 0.3, 0.5]]])
    student_logits = torch.tensor([[[0.6, 0.3, 0.1]]]).log()

    metrics = preloop_pdf_loss(
        student_logits=student_logits,
        teacher_pdf=teacher_pdf,
        loss_mask=torch.tensor([[1]]),
    )

    assert metrics.loss == pytest.approx(float((metrics.kl + metrics.cdf_mae).item()))
    assert metrics.pdf_mae > 0
    assert metrics.cdf_mae > 0


def test_shifted_qexit_metrics_reports_threshold_agreement() -> None:
    gate = torch.nn.Linear(2, 1)
    with torch.no_grad():
        gate.weight.zero_()
        gate.bias.fill_(10.0)
    hidden_states = [torch.ones(1, 2, 2), torch.ones(1, 2, 2), torch.ones(1, 2, 2)]
    teacher_gates = [
        torch.full((1, 2, 1), -10.0),
        torch.full((1, 2, 1), 10.0),
        torch.full((1, 2, 1), -10.0),
    ]

    metrics = shifted_qexit_metrics(
        hidden_states_list=hidden_states,
        teacher_gate_list=teacher_gates,
        lookahead_gate=gate,
        thresholds=(0.5,),
        loss_mask=torch.tensor([[1, 1]]),
    )

    assert metrics["qexit/agreement_q0p5"] == 1.0
    assert metrics["qexit/student_mean_depth_q0p5"] == 2.0
    assert "qexit/teacher_mean_depth_q0p5" not in metrics


def test_same_step_hazard_qexit_metrics_matches_teacher_pdf() -> None:
    gate = torch.nn.Linear(1, 1)
    with torch.no_grad():
        gate.weight.fill_(1.0)
        gate.bias.zero_()
    hidden_states = [
        torch.full((1, 2, 1), -10.0),
        torch.full((1, 2, 1), 10.0),
        torch.full((1, 2, 1), -10.0),
    ]
    teacher_gates = [
        torch.full((1, 2, 1), -10.0),
        torch.full((1, 2, 1), 10.0),
        torch.full((1, 2, 1), -10.0),
    ]

    metrics = hazard_qexit_metrics(
        hidden_states_list=hidden_states,
        teacher_gate_list=teacher_gates,
        hazard_gate=gate,
        first_step_index=0,
        thresholds=(0.5,),
        loss_mask=torch.tensor([[1, 1]]),
    )

    assert metrics["qexit/agreement_q0p5"] == 1.0
    assert metrics["qexit/student_mean_depth_q0p5"] == 2.0
    assert "qexit/teacher_mean_depth_q0p5" not in metrics


def test_preloop_qexit_metrics_reports_threshold_agreement() -> None:
    teacher_gates = [
        torch.full((1, 2, 1), -10.0),
        torch.full((1, 2, 1), 10.0),
        torch.full((1, 2, 1), -10.0),
    ]
    teacher_pdf = compute_teacher_exit_pdf(teacher_gates)
    gate = torch.nn.Linear(2, 3)
    with torch.no_grad():
        gate.weight.zero_()
        gate.bias.copy_(teacher_pdf[0, 0].log())

    metrics = preloop_qexit_metrics(
        preloop_hidden=torch.ones(1, 2, 2),
        teacher_gate_list=teacher_gates,
        preloop_gate=gate,
        thresholds=(0.5,),
        loss_mask=torch.tensor([[1, 1]]),
    )

    assert metrics["qexit/agreement_q0p5"] == 1.0
    assert metrics["qexit/student_mean_depth_q0p5"] == 2.0
    assert "qexit/teacher_mean_depth_q0p5" not in metrics


def test_qexit_metrics_from_pdfs_masks_non_assistant_tokens() -> None:
    student_pdf = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
    teacher_pdf = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]])

    metrics = qexit_metrics_from_pdfs(
        student_pdf=student_pdf,
        teacher_pdf=teacher_pdf,
        thresholds=(0.5,),
        loss_mask=torch.tensor([[1, 0]]),
    )

    assert metrics["qexit/agreement_q0p5"] == 1.0
    assert metrics["qexit/student_mean_depth_q0p5"] == 2.0
    assert metrics["qexit/student_exit_frac_depth2_q0p5"] == 1.0
    assert metrics["qexit/student_exit_frac_depth3_q0p5"] == 0.0
    assert "qexit/teacher_mean_depth_q0p5" not in metrics


def test_qexit_metrics_from_pdfs_reports_exit_depth_fractions() -> None:
    student_pdf = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
    teacher_pdf = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]])

    metrics = qexit_metrics_from_pdfs(
        student_pdf=student_pdf,
        teacher_pdf=teacher_pdf,
        thresholds=(0.5,),
        loss_mask=torch.tensor([[1, 1]]),
    )

    assert metrics["qexit/student_exit_frac_depth1_q0p5"] == 0.0
    assert metrics["qexit/student_exit_frac_depth2_q0p5"] == 0.5
    assert metrics["qexit/student_exit_frac_depth3_q0p5"] == 0.5
    assert metrics["qexit/teacher_exit_frac_depth3_q0p5"] == 1.0


def test_preloop_qexit_metrics_enforces_ouro_min_exit_depth() -> None:
    teacher_gates = [
        torch.full((1, 2, 1), 10.0),
        torch.full((1, 2, 1), -10.0),
        torch.full((1, 2, 1), -10.0),
    ]
    teacher_pdf = compute_teacher_exit_pdf(teacher_gates)
    gate = torch.nn.Linear(2, 3)
    with torch.no_grad():
        gate.weight.zero_()
        gate.bias.copy_(teacher_pdf[0, 0].log())

    metrics = preloop_qexit_metrics(
        preloop_hidden=torch.ones(1, 2, 2),
        teacher_gate_list=teacher_gates,
        preloop_gate=gate,
        thresholds=(0.5,),
        loss_mask=torch.tensor([[1, 1]]),
    )

    assert metrics["qexit/agreement_q0p5"] == 1.0
    assert metrics["qexit/student_mean_depth_q0p5"] == 2.0
    assert "qexit/teacher_mean_depth_q0p5" not in metrics


def test_strip_assistant_thinking_traces_preserves_user_think_text() -> None:
    messages = [
        {"role": "user", "content": "keep <think>visible</think>"},
        {"role": "assistant", "content": "<think>hidden</think>\nanswer"},
    ]

    stripped = strip_assistant_thinking_traces(messages)

    assert stripped[0]["content"] == "keep <think>visible</think>"
    assert stripped[1]["content"] == "answer"


def test_shifted_assistant_loss_mask_aligns_next_token() -> None:
    assert shifted_assistant_loss_mask([0, 1, 1, 0]) == [1, 1, 0, 0]
    assert shifted_assistant_loss_mask([0, 1, 1, 1], seq_lengths=[3, 1]) == [1, 1, 0, 0]
    assert shifted_assistant_loss_mask([]) == []


def test_position_ids_from_seq_lengths_resets_boundaries() -> None:
    assert position_ids_from_seq_lengths([3, 1, 2]) == [0, 1, 2, 0, 0, 1]


def test_validate_assistant_mask_support_rejects_zero_assistant_masks() -> None:
    _validate_assistant_mask_support(AssistantMaskTokenizer(mark_assistant=True))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="generation"):
        _validate_assistant_mask_support(AssistantMaskTokenizer(mark_assistant=False))  # type: ignore[arg-type]


def test_packed_conversation_dataset_carries_tails_and_counts_sources() -> None:
    dataset = Dataset.from_dict(
        {
            "input_ids": [[10, 11, 12, 20]],
            "assistant_mask": [[0, 1, 1, 1]],
            "source_ids": [[0, 0, 0, 1]],
            "seq_lengths": [[3, 1]],
        }
    )

    packed = list(PackedConversationDataset(dataset, sequence_length=4))

    assert len(packed) == 1
    assert packed[0]["input_ids"] == [10, 11, 12, 20]
    assert packed[0]["assistant_mask"] == [0, 1, 1, 1]
    assert packed[0]["loss_mask"] == [1, 1, 0, 0]
    assert packed[0]["position_ids"] == [0, 1, 2, 0]
    assert packed[0]["seq_lengths"] == [3, 1]
    assert packed[0]["source_counts"] == [3, 1, 0, 0]


def test_packed_conversation_dataset_skips_zero_loss_blocks() -> None:
    dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3, 4]],
            "assistant_mask": [[0, 0, 0, 0]],
            "source_ids": [[0, 0, 0, 0]],
            "seq_lengths": [[4]],
        }
    )

    assert list(PackedConversationDataset(dataset, sequence_length=4)) == []


def test_packed_conversation_dataset_requires_requested_validation_sequences() -> None:
    dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3, 4]],
            "assistant_mask": [[0, 1, 1, 0]],
            "source_ids": [[0, 0, 0, 0]],
            "seq_lengths": [[4]],
        }
    )

    with pytest.raises(RuntimeError, match="assistant-bearing sequences"):
        list(PackedConversationDataset(dataset, sequence_length=4, max_sequences=2))


def test_collate_packed_features_returns_expected_tensor_dtypes() -> None:
    batch = collate_packed_features(
        [
            {
                "input_ids": [1, 2],
                "attention_mask": [1, 1],
                "assistant_mask": [0, 1],
                "loss_mask": [1, 0],
                "position_ids": [0, 1],
                "seq_lengths": [2],
                "source_counts": [2, 0, 0, 0],
            }
        ]
    )

    assert batch["input_ids"].dtype == torch.long
    assert batch["attention_mask"].dtype == torch.bool
    assert batch["assistant_mask"].dtype == torch.bool
    assert batch["loss_mask"].dtype == torch.bool
    assert batch["position_ids"].tolist() == [[0, 1]]
    assert batch["seq_lengths"].tolist() == [2]
    assert batch["source_counts"].tolist() == [[2, 0, 0, 0]]


def test_create_packed_dataloaders_uses_eval_batch_size(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenized = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3, 4]] * 8,
            "assistant_mask": [[0, 1, 1, 0]] * 8,
            "source_ids": [[0, 0, 0, 0]] * 8,
            "seq_lengths": [[4]] * 8,
        }
    )
    monkeypatch.setattr(gate_data, "_validate_assistant_mask_support", lambda tokenizer: None)
    monkeypatch.setattr(gate_data, "_load_tokenized_splits", lambda **kwargs: (tokenized, tokenized))

    train_loader, eval_loader = gate_data.create_packed_dataloaders(
        tokenizer=AssistantMaskTokenizer(),  # type: ignore[arg-type]
        batch_size=2,
        eval_batch_size=3,
        sequence_length=4,
        seed=79,
    )

    assert train_loader.batch_size == 2
    assert train_loader.num_workers == gate_data.TRAIN_DATALOADER_NUM_WORKERS
    assert train_loader.prefetch_factor == gate_data.TRAIN_DATALOADER_PREFETCH_FACTOR
    assert eval_loader.batch_size == 3
    assert eval_loader.num_workers == 0


def test_format_log_lines_separates_train_and_eval_metrics() -> None:
    lines = _format_log_lines(
        {
            "lookahead/train/loss": 0.9111,
            "lookahead/train/hazard_mae": 0.1403,
            "train/step": 1,
            "train/tokens": 262144,
            "time/tokens_per_second": 1234.5,
            "time/seconds_per_step": 2.4,
            "time/eta_hours": 0.5,
            "lookahead/qexit/agreement_q0p5": 0.75,
            "lookahead/qexit/student_exit_frac_depth4_q0p5": 0.25,
        },
        step=1,
        max_steps=1000,
    )

    assert lines == [
        "train step 1/1000 | tokens 2.62e+05 | tok/s 1234 | s/step 2.40 | eta 0.50h | lookahead loss 0.9111 | lookahead mae 0.1403",
        "eval step 1/1000 | lookahead agreement_q0p5 0.750",
    ]
    assert all("lookahead/train/loss" not in line for line in lines)
    assert all("exit_frac_depth" not in line for line in lines)


def test_selected_gates_control_trainable_parameters_and_optimizer_groups() -> None:
    wrapped = DummyOuroForCausalLM(OuroModel(tiny_ouro_config(total_ut_steps=3)))
    config = GateTrainingConfig(
        model_name_or_path="dummy",
        output_dir=Path("unused"),
        train_gates="both",
        control_gates="none",
        lookahead_learning_rate=3e-5,
        preloop_learning_rate=1e-3,
    )

    gates = _initialize_trainable_gates(wrapped, "both")
    optimizer = _build_gate_optimizer(gates, config)

    assert set(gates) == {"lookahead", "preloop"}
    assert [group["lr"] for group in optimizer.param_groups] == [3e-5, 1e-3]
    assert all(parameter.requires_grad for gate in gates.values() for parameter in gate.parameters())
    assert not wrapped.model.early_exit_gate.weight.requires_grad


def test_no_control_gates_preserves_real_gate_optimizer_groups() -> None:
    wrapped = DummyOuroForCausalLM(OuroModel(tiny_ouro_config(total_ut_steps=3)))
    config = GateTrainingConfig(
        model_name_or_path="dummy",
        output_dir=Path("unused"),
        control_gates="none",
    )

    real_gates = _initialize_trainable_gates(wrapped, config.train_gates)
    control_gates = _initialize_control_gates(
        wrapped,
        config.control_gates,
        device=torch.device("cpu"),
    )
    optimizer = _build_gate_optimizer({**real_gates, **control_gates}, config)

    assert control_gates == {}
    assert [group["name"] for group in optimizer.param_groups] == ["lookahead", "preloop"]


def test_control_gates_add_optimizer_groups_with_lookahead_learning_rate() -> None:
    wrapped = DummyOuroForCausalLM(OuroModel(tiny_ouro_config(total_ut_steps=3)))
    config = GateTrainingConfig(
        model_name_or_path="dummy",
        output_dir=Path("unused"),
        control_gates="same_step",
        lookahead_learning_rate=3e-3,
        preloop_learning_rate=1e-4,
    )

    real_gates = _initialize_trainable_gates(wrapped, config.train_gates)
    control_gates = _initialize_control_gates(
        wrapped,
        config.control_gates,
        device=torch.device("cpu"),
    )
    optimizer = _build_gate_optimizer({**real_gates, **control_gates}, config)

    assert set(control_gates) == {"same_step"}
    assert [group["name"] for group in optimizer.param_groups] == [
        "lookahead",
        "preloop",
        "same_step",
    ]
    assert [group["lr"] for group in optimizer.param_groups] == [3e-3, 1e-4, 3e-3]


def test_standard_config_trains_all_gate_heads_with_shared_cosine_lr() -> None:
    wrapped = DummyOuroForCausalLM(OuroModel(tiny_ouro_config(total_ut_steps=3)))
    config = GateTrainingConfig(model_name_or_path="dummy", output_dir=Path("unused"))

    real_gates = _initialize_trainable_gates(wrapped, config.train_gates)
    control_gates = _initialize_control_gates(
        wrapped,
        config.control_gates,
        device=torch.device("cpu"),
    )
    optimizer = _build_gate_optimizer({**real_gates, **control_gates}, config)

    assert config.lookahead_learning_rate == 1e-3
    assert config.preloop_learning_rate == 1e-3
    assert config.learning_rate_schedule == "warmup_cosine"
    assert config.warmup_steps == 500
    assert config.min_lr_ratio == 0.05
    assert config.max_steps == 20000
    assert config.log_every == 20
    assert config.eval_every == 500
    assert [group["name"] for group in optimizer.param_groups] == [
        "lookahead",
        "preloop",
        "same_step",
    ]
    assert [group["lr"] for group in optimizer.param_groups] == [1e-3] * 3


def test_gate_training_config_rejects_invalid_standard_scalars() -> None:
    with pytest.raises(ValueError, match="warmup_steps"):
        GateTrainingConfig(model_name_or_path="dummy", output_dir=Path("unused"), max_steps=10, warmup_steps=11)

    with pytest.raises(ValueError, match="eval_batch_size"):
        GateTrainingConfig(model_name_or_path="dummy", output_dir=Path("unused"), eval_batch_size=0)

    with pytest.raises(ValueError, match="min_lr_ratio"):
        GateTrainingConfig(model_name_or_path="dummy", output_dir=Path("unused"), min_lr_ratio=1.1)


def test_clip_gate_gradients_clips_each_gate_independently() -> None:
    gate_a = torch.nn.Linear(1, 1, bias=False)
    gate_b = torch.nn.Linear(1, 1, bias=False)
    gate_a.weight.grad = torch.tensor([[3.0]])
    gate_b.weight.grad = torch.tensor([[4.0]])

    grad_norms = _clip_gate_gradients({"a": gate_a, "b": gate_b}, max_grad_norm=1.0)

    assert grad_norms == pytest.approx({"a": 3.0, "b": 4.0})
    assert gate_a.weight.grad.item() == pytest.approx(1.0)
    assert gate_b.weight.grad.item() == pytest.approx(1.0)


def test_warmup_cosine_learning_rate_schedule_scales_optimizer_groups() -> None:
    wrapped = DummyOuroForCausalLM(OuroModel(tiny_ouro_config(total_ut_steps=3)))
    config = GateTrainingConfig(
        model_name_or_path="dummy",
        output_dir=Path("unused"),
        train_gates="both",
        control_gates="none",
        max_steps=100,
        lookahead_learning_rate=1e-2,
        preloop_learning_rate=3e-2,
        learning_rate_schedule="warmup_cosine",
        warmup_steps=10,
    )
    gates = _initialize_trainable_gates(wrapped, "both")
    optimizer = _build_gate_optimizer(gates, config)

    warmup_rates = _set_optimizer_learning_rates(optimizer, step=5, config=config)
    peak_rates = _set_optimizer_learning_rates(optimizer, step=10, config=config)
    final_rates = _set_optimizer_learning_rates(optimizer, step=100, config=config)

    assert warmup_rates == {"lookahead": 5e-3, "preloop": 1.5e-2}
    assert peak_rates == {"lookahead": 1e-2, "preloop": 3e-2}
    assert final_rates == {"lookahead": 5e-4, "preloop": 1.5e-3}


def test_learning_rate_scale_rejects_invalid_step() -> None:
    with pytest.raises(ValueError, match="step must be one-indexed"):
        _learning_rate_scale(step=0, max_steps=100, warmup_steps=10, min_lr_ratio=0.05, schedule="warmup_cosine")


def test_selected_single_gate_only_trains_that_gate() -> None:
    wrapped = DummyOuroForCausalLM(OuroModel(tiny_ouro_config(total_ut_steps=3)))

    gates = _initialize_trainable_gates(wrapped, "preloop")

    assert set(gates) == {"preloop"}
    assert wrapped.model.preloop_exit_gate.weight.requires_grad
    assert not wrapped.model.lookahead_exit_gate.weight.requires_grad


def test_save_gate_writes_only_selected_artifacts(tmp_path: Path) -> None:
    wrapped = DummyOuroForCausalLM(OuroModel(tiny_ouro_config(total_ut_steps=3)))
    config = GateTrainingConfig(
        model_name_or_path="dummy",
        output_dir=tmp_path,
        train_gates="preloop",
        control_gates="none",
    )

    _save_gate(model=wrapped, output_dir=tmp_path, config=config, step=5)

    assert (tmp_path / "preloop_exit_gate.safetensors").exists()
    assert not (tmp_path / "lookahead_exit_gate.safetensors").exists()
    assert (tmp_path / "gate_training_config.json").exists()
    metadata = json.loads((tmp_path / "gate_training_config.json").read_text())
    assert metadata["data"]["dataset"] == "nvidia/Nemotron-Post-Training-Dataset-v2"
    assert metadata["data"]["dataset_revision"] == "5c89e01dd720ae0f4058445ed49c5fb68a03c76e"
    assert metadata["data"]["validation_packed_sequences"] == 128


def test_save_gate_writes_selected_control_artifacts(tmp_path: Path) -> None:
    wrapped = DummyOuroForCausalLM(OuroModel(tiny_ouro_config(total_ut_steps=3)))
    config = GateTrainingConfig(
        model_name_or_path="dummy",
        output_dir=tmp_path,
        train_gates="preloop",
        control_gates="same_step",
    )
    control_gates = _initialize_control_gates(
        wrapped,
        config.control_gates,
        device=torch.device("cpu"),
    )

    _save_gate(model=wrapped, control_gates=control_gates, output_dir=tmp_path, config=config, step=5)

    assert (tmp_path / "preloop_exit_gate.safetensors").exists()
    assert (tmp_path / "same_step_exit_gate.safetensors").exists()
    assert not (tmp_path / "lookahead_exit_gate.safetensors").exists()


def test_best_attention_backend_selects_by_compute_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))

    assert best_attention_backend() == "flash_attention_3"

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 0))

    assert best_attention_backend() == "flash_attention_2"


def test_best_attention_backend_rejects_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="FlashAttention"):
        best_attention_backend()


def test_gate_training_requires_flash_attention_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(looped_cdb.utils, "best_attention_backend", lambda: "sdpa")
    with pytest.raises(RuntimeError, match="flash-attention backend is required"):
        require_flash_attention()

    monkeypatch.setattr(looped_cdb.utils, "best_attention_backend", lambda: "flash_attention_2")
    assert require_flash_attention() == "flash_attention_2"
