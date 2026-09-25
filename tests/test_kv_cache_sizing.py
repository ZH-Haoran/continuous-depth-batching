"""Physical KV geometry and post-load block budgeting."""

import logging
from types import SimpleNamespace

import pytest
import torch

from looped_cdb.continuous_batching.cache import PagedAttentionCache as CBCache
from looped_cdb.continuous_batching.config import ContinuousBatchingConfig
from looped_cdb.continuous_depth_batching.cache import PagedAttentionCache as CDBCache
from looped_cdb.continuous_depth_batching.config import ContinuousDepthBatchingConfig
from looped_cdb.kv_cache_sizing import resolve_num_blocks, validate_cache_size_request
from looped_cdb.paged_cache_geometry import PagedKVCacheGeometry


def _huginn_config() -> SimpleNamespace:
    return SimpleNamespace(
        num_hidden_layers=8,
        num_prelude_layers=2,
        num_core_layers=4,
        num_coda_layers=2,
        total_recurrent_steps=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=16,
        head_dim=4,
        layer_types=["full_attention"] * 8,
        sliding_window=None,
        _attn_implementation="paged|flash_attention_3",
    )


@pytest.mark.parametrize(
    ("policy", "slots", "layers"),
    [("single", 1, 8), ("first_then_shared", 2, 12), ("depth_indexed", 4, 20), ("last_exited", 4, 20)],
)
def test_huginn_cache_geometry_and_allocation(policy: str, slots: int, layers: int) -> None:
    model_config = _huginn_config()
    config = ContinuousDepthBatchingConfig(
        block_size=2,
        num_blocks=3,
        max_num_batched_tokens=4,
        max_model_len=8,
        max_recurrent_steps=4,
        kv_policy=policy,
        kv_pressure_mode="none",
    ).get_resolved(model_config)
    geometry = PagedKVCacheGeometry.from_model(
        model_config,
        block_size=config.block_size,
        dtype=torch.float16,
        kv_policy=config.kv_policy,
        kv_slots_per_layer=config.kv_slots_per_layer,
        total_recurrent_steps=config.max_recurrent_steps,
        assume_all_layers_recurrent=True,
    )
    cache = CDBCache(model_config, config, "cpu", geometry=geometry)

    assert config.kv_slots_per_layer == slots
    assert len(cache.key_cache) == layers
    assert geometry.bytes_per_block == 2 * layers * 2 * 2 * 4 * 2
    actual_bytes = sum(t.numel() * t.element_size() for t in cache.key_cache + cache.value_cache)
    assert actual_bytes == geometry.allocation_bytes(config.num_blocks)


def test_cb_and_cdb_share_storage_for_single_slot() -> None:
    model_config = _huginn_config()
    model_config._cdb_kv_policy = "single"
    cb = CBCache(model_config, ContinuousBatchingConfig(block_size=2, num_blocks=3, max_model_len=8), "cpu")
    cdb = CDBCache(
        model_config,
        ContinuousDepthBatchingConfig(block_size=2, num_blocks=3, max_model_len=8, max_recurrent_steps=4),
        "cpu",
    )
    assert cb.cache_shape == cdb.cache_shape
    assert cb.num_layers == cdb.num_layers == 8


def test_budget_excludes_runtime_reserve_and_unallocatable_blocks(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    geometry = PagedKVCacheGeometry(2, 4, 2, 8, torch.float16, "single", 1)
    # One usable block costs 512 bytes; 2 additional blocks are allocated but unavailable.
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (8000, 10000))
    with caplog.at_level(logging.INFO, logger="looped_cdb.kv_cache_sizing"):
        blocks = resolve_num_blocks(None, 0.8, geometry, "cuda:0")
    assert blocks == 9
    assert "policy=single" in caplog.text
    assert "usable_blocks=9, usable_tokens=36" in caplog.text
    assert resolve_num_blocks(None, None, geometry, "cuda:0") == blocks
    assert geometry.allocation_bytes(blocks) <= 6000
    assert geometry.allocation_bytes(blocks + 1) > 6000


def test_manual_block_count_does_not_profile_device(monkeypatch: pytest.MonkeyPatch) -> None:
    geometry = PagedKVCacheGeometry(2, 4, 2, 8, torch.float16, "single", 1)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: pytest.fail("GPU memory should not be read"))
    assert resolve_num_blocks(7, None, geometry, "cpu") == 7


@pytest.mark.parametrize("fraction", [-0.1, 0, 1, float("nan"), float("inf")])
def test_invalid_fraction(fraction: float) -> None:
    with pytest.raises(ValueError, match="mem_fraction_static"):
        validate_cache_size_request(None, fraction)


def test_conflicting_settings_and_cpu_auto_sizing() -> None:
    geometry = PagedKVCacheGeometry(2, 4, 2, 8, torch.float16, "single", 1)
    with pytest.raises(ValueError, match="either num_blocks or mem_fraction_static"):
        resolve_num_blocks(7, 0.8, geometry, "cuda")
    with pytest.raises(ValueError, match="requires CUDA"):
        resolve_num_blocks(None, None, geometry, "cpu")


def test_both_engine_configs_accept_auto_and_reject_conflicting_overrides() -> None:
    model_config = _huginn_config()
    assert ContinuousBatchingConfig().get_resolved(model_config).num_blocks is None
    assert ContinuousDepthBatchingConfig(max_recurrent_steps=4).get_resolved(model_config).num_blocks is None
    for config in (
        ContinuousBatchingConfig(num_blocks=7, mem_fraction_static=0.8),
        ContinuousDepthBatchingConfig(num_blocks=7, mem_fraction_static=0.8, max_recurrent_steps=4),
    ):
        with pytest.raises(ValueError, match="either num_blocks or mem_fraction_static"):
            config.get_resolved(model_config)


def test_insufficient_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    geometry = PagedKVCacheGeometry(2, 4, 2, 8, torch.float16, "single", 1)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (2000, 10000))
    with pytest.raises(ValueError, match="No usable KV blocks fit"):
        resolve_num_blocks(None, None, geometry, "cuda")
