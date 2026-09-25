import pytest
import torch
from transformers import PreTrainedConfig

from looped_cdb.continuous_batching.cache import (
    PagedAttentionCache,
)
from looped_cdb.continuous_batching.config import ContinuousBatchingConfig


def _config(**overrides: int | str | list[str] | None) -> PreTrainedConfig:
    values = {
        "num_hidden_layers": 2,
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


def _cb_config(
    num_blocks: int | None = 2,
    block_size: int = 4,
    max_num_batched_tokens: int = 8,
    max_blocks_per_request: int = 32,
    use_async_batching: bool = False,
) -> ContinuousBatchingConfig:
    return ContinuousBatchingConfig(
        num_blocks=num_blocks,
        block_size=block_size,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_blocks_per_request * block_size,
        use_async_batching=use_async_batching,
    )


def test_paged_attention_cache_rejects_sliding_attention_for_now() -> None:
    with pytest.raises(NotImplementedError, match="full-attention"):
        PagedAttentionCache(
            config=_config(sliding_window=4),
            continuous_batching_config=_cb_config(),
            device="cpu",
            dtype=torch.float32,
        )


def test_paged_attention_cache_rejects_non_flash_attention() -> None:
    with pytest.raises(NotImplementedError, match="FlashAttention"):
        PagedAttentionCache(
            config=_config(_attn_implementation="paged|eager"),
            continuous_batching_config=_cb_config(),
            device="cpu",
            dtype=torch.float32,
        )


def test_paged_attention_cache_allocates_and_frees_request_blocks() -> None:
    cache = PagedAttentionCache(
        config=_config(),
        continuous_batching_config=_cb_config(num_blocks=2),
        device="cpu",
        dtype=torch.float32,
    )

    assert cache.allocate_blocks(1, "req-0", allocated_blocks=0) == 1
    assert cache.allocate_blocks(2, "req-1", allocated_blocks=0) is None
    assert cache.get_num_free_blocks() == 1

    cache.free_blocks("req-0")

    assert cache.get_num_free_blocks() == 2


def test_paged_attention_cache_allocates_virtual_ouro_layers() -> None:
    cache = PagedAttentionCache(
        config=_config(model_type="ouro", total_ut_steps=3, layer_types=["full_attention", "full_attention"]),
        continuous_batching_config=_cb_config(num_blocks=2, max_num_batched_tokens=4),
        device="cpu",
        dtype=torch.float32,
    )

    assert cache.num_layers == 6
    assert cache.layer_index_to_group_indices == [(0, layer_idx) for layer_idx in range(6)]


def test_paged_attention_cache_allocates_shared_ouro_layers_when_configured() -> None:
    config = _config(model_type="ouro", total_ut_steps=3, layer_types=["full_attention", "full_attention"])
    config._cdb_kv_policy = "single"

    cache = PagedAttentionCache(
        config=config,
        continuous_batching_config=_cb_config(num_blocks=2, max_num_batched_tokens=4),
        device="cpu",
        dtype=torch.float32,
    )

    assert cache.kv_policy == "single"
    assert cache.num_layers == 2
    assert len(cache.key_cache) == 2
    assert len(cache.value_cache) == 2
    assert cache.layer_index_to_group_indices == [(0, layer_idx) for layer_idx in range(2)]


def test_paged_attention_cache_allocates_multi_slot_static_ouro_layers() -> None:
    config = _config(model_type="ouro", total_ut_steps=4, layer_types=["full_attention", "full_attention"])
    config._cdb_kv_policy = "first_then_shared"
    config._cdb_kv_slots_per_layer = 2

    cache = PagedAttentionCache(
        config=config,
        continuous_batching_config=_cb_config(num_blocks=2, max_num_batched_tokens=4),
        device="cpu",
        dtype=torch.float32,
    )

    assert cache.kv_policy == "first_then_shared"
    assert cache.kv_slots_per_layer == 2
    assert cache.num_layers == 4
    assert cache.kv_slot_for_recurrent_step(3) == 1


def test_paged_attention_cache_extends_indices_and_fills_block_table() -> None:
    cache = PagedAttentionCache(
        config=_config(),
        continuous_batching_config=_cb_config(num_blocks=3),
        device="cpu",
        dtype=torch.float32,
    )
    cache.allocate_blocks(2, "req", allocated_blocks=0)

    read_index: list[int] = []
    write_index: list[int] = []
    cache.extend_read_and_write_indices(
        request_id="req",
        past_length=3,
        query_length=3,
        read_index=read_index,
        write_index=write_index,
    )

    block_table = torch.full((4,), -1, dtype=torch.int32)
    cache.fill_block_table("req", past_length=3, query_length=3, block_table=block_table)

    assert read_index == [0, 1, 2, 3, 4, 5]
    assert write_index == [3, 4, 5]
    assert block_table.tolist() == [0, 1, -1, -1]


def test_paged_attention_cache_update_writes_and_reads_full_attention_cache() -> None:
    cache = PagedAttentionCache(
        config=_config(num_hidden_layers=1),
        continuous_batching_config=_cb_config(num_blocks=2),
        device="cpu",
        dtype=torch.float32,
    )
    cache.allocate_blocks(1, "req", allocated_blocks=0)

    first_keys = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    first_values = first_keys + 10
    first_write = torch.tensor([0, 1], dtype=torch.int64)
    no_read = torch.empty((0,), dtype=torch.int64)

    returned_keys, returned_values = cache.update(
        first_keys, first_values, layer_idx=0, read_index=no_read, write_index=first_write
    )

    assert returned_keys.tolist() == [[[1.0, 2.0]], [[3.0, 4.0]]]
    assert returned_values.tolist() == [[[11.0, 12.0]], [[13.0, 14.0]]]

    next_keys = torch.tensor([[[[5.0, 6.0]]]])
    next_values = next_keys + 10
    read = torch.tensor([0, 1, 2], dtype=torch.int64)
    write = torch.tensor([2], dtype=torch.int64)

    returned_keys, returned_values = cache.update(
        next_keys, next_values, layer_idx=0, read_index=read, write_index=write
    )

    assert returned_keys.tolist() == [[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]]
    assert returned_values.tolist() == [[[11.0, 12.0]], [[13.0, 14.0]], [[15.0, 16.0]]]


def test_paged_attention_cache_detects_block_table_kwarg_name() -> None:
    cache = PagedAttentionCache(
        config=_config(),
        continuous_batching_config=_cb_config(num_blocks=1),
        device="cpu",
        dtype=torch.float32,
    )

    def fn_with_page_table(page_table: torch.Tensor) -> None:
        _ = page_table

    assert cache.get_block_table_key(fn_with_page_table) == "page_table"


def test_paged_attention_cache_frees_all_requests() -> None:
    cache = PagedAttentionCache(
        config=_config(),
        continuous_batching_config=_cb_config(num_blocks=2),
        device="cpu",
        dtype=torch.float32,
    )
    cache.allocate_blocks(1, "req-0", allocated_blocks=0)
    cache.allocate_blocks(1, "req-1", allocated_blocks=0)

    cache.free_all_requests()

    assert cache.get_num_free_blocks() == 2
