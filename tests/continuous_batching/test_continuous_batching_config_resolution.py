import pytest
from transformers import PreTrainedConfig

from looped_cdb.continuous_batching.config import ContinuousBatchingConfig, is_supported_attention_implementation


def _config(**overrides: int | str | list[str] | None) -> PreTrainedConfig:
    values = {
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "hidden_size": 4,
        "vocab_size": 16,
        "sliding_window": None,
        "layer_types": None,
        "_attn_implementation": "paged|flash_attention_3",
    }
    values.update(overrides)
    return PreTrainedConfig(**values)


def test_get_resolved_returns_resolved_copy() -> None:
    cb_config = ContinuousBatchingConfig(num_blocks=1024)

    resolved = cb_config.get_resolved(_config())

    assert resolved is not cb_config
    assert resolved.max_model_len == 16384
    assert resolved.decode_block_table_width() == 1024  # ceil(max_model_len 16384 / block_size 16)
    assert resolved.use_cuda_graph is True
    assert resolved.use_async_batching is True


def test_get_resolved_preserves_cuda_graph_bool() -> None:
    resolved = ContinuousBatchingConfig(num_blocks=1024, use_cuda_graph=False).get_resolved(_config())

    assert resolved.use_cuda_graph is False
    assert resolved.use_async_batching is True


def test_config_defaults_preserve_hf_values() -> None:
    cb_config = ContinuousBatchingConfig(num_blocks=1024)

    assert cb_config.max_model_len == 16384
    assert cb_config.use_cuda_graph is True
    assert cb_config.use_async_batching is True


def test_validate_native_cb_config_rejects_sliding_attention() -> None:
    with pytest.raises(NotImplementedError, match="full-attention"):
        ContinuousBatchingConfig(num_blocks=1024).validate(_config(sliding_window=4))


def test_validate_native_cb_config_rejects_non_flash_attention() -> None:
    with pytest.raises(NotImplementedError, match="FlashAttention"):
        ContinuousBatchingConfig(num_blocks=1024).validate(_config(_attn_implementation="paged|eager"))


@pytest.mark.parametrize(
    "attn_implementation",
    [
        "paged|flash_attention",
        "paged|flash_attention_2",
        "paged|flash_attention_3",
    ],
)
def test_native_cb_config_accepts_paged_flash_attention(attn_implementation: str) -> None:
    assert is_supported_attention_implementation(_config(_attn_implementation=attn_implementation))


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
def test_native_cb_config_rejects_non_paged_flash_attention(attn_implementation: str | None) -> None:
    assert not is_supported_attention_implementation(_config(_attn_implementation=attn_implementation))


def test_validate_native_cb_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="block_size"):
        ContinuousBatchingConfig(num_blocks=1024, block_size=0).validate(_config())
    with pytest.raises(ValueError, match="num_blocks must be positive"):
        ContinuousBatchingConfig(num_blocks=0).validate(_config())
    with pytest.raises(ValueError, match="max_num_seqs"):
        ContinuousBatchingConfig(num_blocks=1024, max_num_seqs=0).validate(_config())
    with pytest.raises(ValueError, match="safety_margin"):
        ContinuousBatchingConfig(num_blocks=1024, safety_margin=1.0).validate(_config())
    with pytest.raises(ValueError, match="safety_margin"):
        ContinuousBatchingConfig(num_blocks=1024, safety_margin=-0.1).validate(_config())


def test_fa2_rejects_block_size_not_multiple_of_256() -> None:
    # FA2's paged flash_attn_with_kvcache requires block_size % 256 == 0 when the fast decode path is on.
    with pytest.raises(ValueError, match="multiple of 256"):
        ContinuousBatchingConfig(num_blocks=1024, block_size=16).validate(
            _config(_attn_implementation="paged|flash_attention_2")
        )


def test_fa2_accepts_block_size_multiple_of_256() -> None:
    ContinuousBatchingConfig(num_blocks=1024, block_size=256).validate(
        _config(_attn_implementation="paged|flash_attention_2")
    )


def test_fa3_allows_small_block_size() -> None:
    # FA3's paged kernel allows arbitrary block sizes, so the default block_size=16 is valid.
    ContinuousBatchingConfig(num_blocks=1024, block_size=16).validate(
        _config(_attn_implementation="paged|flash_attention_3")
    )


def test_max_num_seqs_is_the_only_batch_setting_and_defaults_explicitly() -> None:
    # One number for residency and the decode launch built from it, given rather than derived, so no
    # engine has to resolve a sentinel into a batch the caller never named.
    assert ContinuousBatchingConfig().max_num_seqs == 256


def test_safety_margin_defaults_to_production_value_and_allows_zero() -> None:
    # 0.2 mirrors serving; 0.0 (full prefill drain) is what the decode-latency measurement builds with.
    assert ContinuousBatchingConfig().safety_margin == 0.2
    ContinuousBatchingConfig(num_blocks=1024, safety_margin=0.0).validate(_config())


def test_kv_pressure_mode_defaults_to_recompute() -> None:
    assert ContinuousBatchingConfig().kv_pressure_mode == "recompute"


def test_kv_pressure_mode_none_is_accepted() -> None:
    # "none" does no reservation or preemption, preserving the original raise-on-full behavior.
    ContinuousBatchingConfig(num_blocks=1024, kv_pressure_mode="none").validate(_config())


def test_kv_pressure_mode_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="kv_pressure_mode"):
        ContinuousBatchingConfig(num_blocks=1024, kv_pressure_mode="swap").validate(_config())


def test_offload_mode_requires_a_cpu_pool() -> None:
    with pytest.raises(ValueError, match="requires cpu_offload_space"):
        ContinuousBatchingConfig(num_blocks=1024, kv_pressure_mode="offload").validate(_config())
    # A positive pool is accepted.
    ContinuousBatchingConfig(num_blocks=1024, kv_pressure_mode="offload", cpu_offload_space=1.0).validate(_config())


def test_cpu_pool_rejected_outside_offload_mode() -> None:
    with pytest.raises(ValueError, match="only used by kv_pressure_mode='offload'"):
        ContinuousBatchingConfig(num_blocks=1024, kv_pressure_mode="recompute", cpu_offload_space=1.0).validate(
            _config()
        )
    with pytest.raises(ValueError, match="only used by kv_pressure_mode='offload'"):
        ContinuousBatchingConfig(num_blocks=1024, kv_pressure_mode="reserve", cpu_offload_space=1.0).validate(_config())
