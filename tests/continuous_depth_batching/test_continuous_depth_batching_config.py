from types import SimpleNamespace

import pytest

from looped_cdb.continuous_depth_batching import ContinuousDepthBatchingConfig
from looped_cdb.continuous_depth_batching.config import is_supported_attention_implementation


def _model_config() -> SimpleNamespace:
    return SimpleNamespace(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        hidden_size=4,
        vocab_size=16,
        sliding_window=None,
        layer_types=["full_attention"] * 6,
        _attn_implementation="paged|flash_attention_3",
        total_ut_steps=3,
    )


def test_cdb_config_resolves_derived_values() -> None:
    config = ContinuousDepthBatchingConfig(
        num_blocks=8,
        max_num_batched_tokens=4,
        max_recurrent_steps=3,
        use_cuda_graph=False,
    ).get_resolved(_model_config())

    assert config.max_recurrent_steps == 3
    assert config.use_cuda_graph is False
    assert config.kv_policy == "single"
    assert config.kv_slots_per_layer == 1


def test_cdb_config_resolves_multi_slot_kv_policy() -> None:
    config = ContinuousDepthBatchingConfig(
        num_blocks=8,
        max_num_batched_tokens=4,
        max_recurrent_steps=4,
        min_recurrent_steps=2,
        exit_threshold=0.5,
        kv_policy="first_then_shared",
    ).get_resolved(_model_config())

    assert config.kv_policy == "first_then_shared"
    assert config.kv_slots_per_layer == 2


def test_cdb_config_allows_gated_exit_at_any_depth_under_last_exited() -> None:
    # Copy-on-exit routing fills the deeper slots at exit time, so the
    # min_recurrent_steps >= kv_slots_per_layer constraint of static layouts does not apply.
    config = ContinuousDepthBatchingConfig(
        num_blocks=8,
        max_num_batched_tokens=4,
        max_recurrent_steps=4,
        exit_threshold=0.5,
        kv_policy="last_exited",
        kv_pressure_mode="none",
    ).get_resolved(_model_config())

    assert config.kv_policy == "last_exited"
    assert config.kv_slots_per_layer == 4
    assert config.min_recurrent_steps == 1


def test_cdb_config_rejects_recompute_preemption_under_last_exited() -> None:
    config = ContinuousDepthBatchingConfig(
        num_blocks=8,
        max_num_batched_tokens=4,
        max_recurrent_steps=4,
        kv_policy="last_exited",
        kv_pressure_mode="recompute",
    )

    with pytest.raises(ValueError, match="recompute"):
        config.get_resolved(_model_config())


def test_cdb_config_resolves_synthetic_exit_replay() -> None:
    config = ContinuousDepthBatchingConfig(
        num_blocks=8,
        max_num_batched_tokens=4,
        max_recurrent_steps=4,
        synthetic_exit_replay=True,
        kv_policy="first_then_shared",
        kv_slots_per_layer=2,
        delay_gate_consumption=True,
        use_async_batching=True,
    ).get_resolved(_model_config())

    assert config.synthetic_exit_replay is True


def test_cdb_config_requires_paged_flash_attention() -> None:
    config = ContinuousDepthBatchingConfig(num_blocks=8, max_num_batched_tokens=4, max_recurrent_steps=3)
    model_config = _model_config()
    model_config._attn_implementation = "flash_attention_2"

    with pytest.raises(NotImplementedError, match="paged attention"):
        config.get_resolved(model_config)


@pytest.mark.parametrize(
    "attn_implementation",
    [
        "paged|flash_attention",
        "paged|flash_attention_2",
        "paged|flash_attention_3",
    ],
)
def test_cdb_config_accepts_paged_flash_attention(attn_implementation: str) -> None:
    model_config = _model_config()
    model_config._attn_implementation = attn_implementation

    assert is_supported_attention_implementation(model_config)


@pytest.mark.parametrize(
    "attn_implementation",
    [
        None,
        "flash_attention_2",
        "paged|eager",
        "eager|flash_attention_2",
        "paged|not_flash_attention",
        "paged|flash_attention_fake",
    ],
)
def test_cdb_config_rejects_non_paged_flash_attention(attn_implementation: str | None) -> None:
    model_config = _model_config()
    model_config._attn_implementation = attn_implementation

    assert not is_supported_attention_implementation(model_config)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_recurrent_steps": None}, "explicitly set"),
        ({"max_recurrent_steps": 0}, "max_recurrent_steps"),
        ({"max_recurrent_steps": 1}, "max_recurrent_steps"),
        ({"min_recurrent_steps": 0}, "min_recurrent_steps"),
        ({"kv_policy": "virtual"}, "kv_policy"),
        ({"kv_policy": "single", "kv_slots_per_layer": 2}, "exactly 1 KV slot"),
        ({"kv_policy": "ring"}, "Unsupported kv_policy"),
        ({"kv_policy": "first_then_shared", "kv_slots_per_layer": 4}, "exceeds total_recurrent_steps"),
        (
            {
                "kv_policy": "first_then_shared",
                "kv_slots_per_layer": 2,
                "exit_threshold": 0.5,
                "min_recurrent_steps": 1,
            },
            "kv_slots_per_layer",
        ),
        ({"exit_threshold": 0.5, "synthetic_exit_replay": True}, "Only one CDB exit policy"),
        ({"safety_margin": 1.0}, "safety_margin"),
        ({"safety_margin": -0.1}, "safety_margin"),
        ({"kv_pressure_mode": "swap"}, "kv_pressure_mode"),
        ({"kv_pressure_mode": "offload"}, "requires cpu_offload_space"),
        ({"kv_pressure_mode": "recompute", "cpu_offload_space": 1.0}, "only used by kv_pressure_mode='offload'"),
        ({"kv_pressure_mode": "reserve", "cpu_offload_space": 1.0}, "only used by kv_pressure_mode='offload'"),
    ],
)
def test_cdb_config_rejects_invalid_depth_settings(kwargs: dict[str, object], match: str) -> None:
    values = {"num_blocks": 8, "max_num_batched_tokens": 4, "max_recurrent_steps": 3}
    values.update(kwargs)
    config = ContinuousDepthBatchingConfig(**values)

    with pytest.raises(ValueError, match=match):
        config.get_resolved(_model_config())


def test_cdb_kv_pressure_mode_defaults_to_recompute_and_accepts_all_modes() -> None:
    assert ContinuousDepthBatchingConfig(max_recurrent_steps=3).kv_pressure_mode == "recompute"
    for mode in ("none", "reserve", "recompute"):
        ContinuousDepthBatchingConfig(
            num_blocks=8, max_num_batched_tokens=4, max_recurrent_steps=3, kv_pressure_mode=mode
        ).validate(_model_config())
    ContinuousDepthBatchingConfig(
        num_blocks=8, max_num_batched_tokens=4, max_recurrent_steps=3, kv_pressure_mode="offload", cpu_offload_space=1.0
    ).validate(_model_config())
