"""CPU stage wiring and fixed-cap profiler configuration checks."""

import pytest
import torch

from looped_cdb.benchmarks.synthetic_stages import SyntheticStageConfig, cpu_stage_smoke, profile_stages


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
