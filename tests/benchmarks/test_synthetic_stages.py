"""CPU stage wiring and fixed-cap profiler configuration checks."""

from contextlib import nullcontext

import pytest
import torch

from looped_cdb.benchmarks.synthetic_stages import (
    SyntheticStageConfig,
    build_random_model,
    cpu_stage_smoke,
    profile_stages,
)


def test_real_random_weight_242_stages_cpu():
    result = cpu_stage_smoke()
    assert result["hidden_shape"] == [1, 2, 64]
    assert result["logits_shape"] == [1, 2, 32]
    assert not result["gate_enabled"]


def test_cpu_cannot_emit_gpu_costs():
    if torch.cuda.is_available():
        pytest.skip("CPU-only assertion")
    with pytest.raises(RuntimeError, match="requires CUDA"):
        profile_stages(SyntheticStageConfig())


@pytest.mark.parametrize(
    "kwargs", [{"num_heads": 23}, {"warmup": 1}, {"resident_cap": 0}, {"context_length": -1}, {"backend": "sdpa"}]
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        SyntheticStageConfig(**kwargs)


@pytest.mark.parametrize("backend", ["flash_attention_2", "flash_attention_3"])
def test_cuda_backend_selected_before_constructor_and_accepted_by_cdb(monkeypatch, backend):
    from transformers import modeling_utils

    from looped_cdb.continuous_depth_batching import ContinuousDepthBatchingConfig
    from looped_cdb.models import huginn

    real_model_type = huginn.HuginnForCausalLM

    class ConstructorProbe:
        def __init__(self, config):
            # No CUDA allocations or kernels here: inspect the constructor
            # boundary, where the real Transformers model selects its backend.
            assert config._attn_implementation == f"paged|{backend}"
            self.config = config

        def eval(self):
            return self

        def set_attn_implementation(self, *args):
            pytest.fail("profiler must not depend on Huginn's unsupported dynamic switch")

    original_dtype = torch.get_default_dtype()
    settings = SyntheticStageConfig(backend=backend, hidden_size=32, num_heads=2, intermediate_size=64, vocab_size=32)
    with monkeypatch.context() as probe:
        probe.setattr(huginn, "HuginnForCausalLM", ConstructorProbe)
        probe.setattr(torch, "device", lambda device: nullcontext())
        model = build_random_model(settings, device="cuda:0", dtype="bfloat16")
    assert model.config.dtype == torch.bfloat16
    assert torch.get_default_dtype() == original_dtype

    # Also exercise the real Transformers constructor with the captured config.
    # Only native FlashAttention availability/imports are stubbed on this CPU.
    monkeypatch.setattr(real_model_type, "_flash_attn_can_dispatch", lambda *args, **kwargs: True)
    monkeypatch.setattr(modeling_utils, "lazy_import_flash_attention", lambda *args, **kwargs: None)
    model = real_model_type(model.config)
    assert model.config._attn_implementation == f"paged|{backend}"
    resolved = ContinuousDepthBatchingConfig(
        block_size=256 if backend == "flash_attention_2" else 16, max_recurrent_steps=32, max_num_seqs=64
    ).get_resolved(model.config)
    assert resolved.max_num_seqs == 64


def test_profiler_rejects_silent_attention_fallback(monkeypatch):
    from looped_cdb.models import huginn

    class FallbackProbe:
        def __init__(self, config):
            self.config = config
            self.config._attn_implementation = "sdpa"

        def eval(self):
            return self

    monkeypatch.setattr(huginn, "HuginnForCausalLM", FallbackProbe)
    monkeypatch.setattr(torch, "device", lambda device: nullcontext())
    with pytest.raises(RuntimeError, match="requested attention 'paged\\|flash_attention_3'"):
        build_random_model(SyntheticStageConfig(), device="cuda:0", dtype="bfloat16")
