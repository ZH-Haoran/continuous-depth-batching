import pytest
import torch
from transformers import PreTrainedConfig

from looped_cdb.continuous_batching.cache import PagedAttentionCache
from looped_cdb.continuous_batching.config import ContinuousBatchingConfig
from looped_cdb.continuous_batching.input_outputs import ContinuousBatchingIOs
from looped_cdb.continuous_batching.requests import FutureRequestState, RequestState

pytestmark = pytest.mark.cuda


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


def _cb_config(
    num_blocks: int | None = 2,
    block_size: int = 4,
    max_num_batched_tokens: int = 8,
) -> ContinuousBatchingConfig:
    return ContinuousBatchingConfig(
        num_blocks=num_blocks,
        block_size=block_size,
        max_num_batched_tokens=max_num_batched_tokens,
    )


@pytest.mark.cuda
def test_paged_attention_cache_allocates_cuda_tensors() -> None:
    cache = PagedAttentionCache(
        config=_config(),
        continuous_batching_config=_cb_config(num_blocks=2),
        device="cuda",
        dtype=torch.float32,
    )

    assert cache.key_cache[0].is_cuda
    assert cache.value_cache[0].is_cuda
    assert cache.key_cache[0].shape == (16, 1, 2)
    assert cache.value_cache[0].shape == (16, 1, 2)


@pytest.mark.cuda
def test_paged_attention_cache_update_on_cuda() -> None:
    cache = PagedAttentionCache(
        config=_config(),
        continuous_batching_config=_cb_config(num_blocks=2),
        device="cuda",
        dtype=torch.float32,
    )
    cache.allocate_blocks(1, "req", allocated_blocks=0)

    first_keys = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], device="cuda")
    first_values = first_keys + 10
    no_read = torch.empty((0,), dtype=torch.int64, device="cuda")
    first_write = torch.tensor([0, 1], dtype=torch.int64, device="cuda")

    returned_keys, returned_values = cache.update(
        first_keys,
        first_values,
        layer_idx=0,
        read_index=no_read,
        write_index=first_write,
    )

    assert returned_keys.is_cuda
    assert returned_values.is_cuda
    torch.testing.assert_close(returned_keys.cpu(), torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]]))
    torch.testing.assert_close(returned_values.cpu(), torch.tensor([[[11.0, 12.0]], [[13.0, 14.0]]]))

    next_keys = torch.tensor([[[[5.0, 6.0]]]], device="cuda")
    next_values = next_keys + 10
    read = torch.tensor([0, 1, 2], dtype=torch.int64, device="cuda")
    write = torch.tensor([2], dtype=torch.int64, device="cuda")

    returned_keys, returned_values = cache.update(
        next_keys,
        next_values,
        layer_idx=0,
        read_index=read,
        write_index=write,
    )

    torch.testing.assert_close(returned_keys.cpu(), torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]]))
    torch.testing.assert_close(
        returned_values.cpu(),
        torch.tensor([[[11.0, 12.0]], [[13.0, 14.0]], [[15.0, 16.0]]]),
    )


@pytest.mark.cuda
def test_paged_attention_cache_update_isolates_layers_on_cuda() -> None:
    cache = PagedAttentionCache(
        config=_config(num_hidden_layers=2),
        continuous_batching_config=_cb_config(num_blocks=2),
        device="cuda",
        dtype=torch.float32,
    )
    cache.allocate_blocks(1, "req", allocated_blocks=0)

    keys = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], device="cuda")
    no_read = torch.empty((0,), dtype=torch.int64, device="cuda")
    write = torch.tensor([0, 1], dtype=torch.int64, device="cuda")

    cache.update(keys, keys + 10, layer_idx=0, read_index=no_read, write_index=write)
    cache.update(keys + 100, keys + 110, layer_idx=1, read_index=no_read, write_index=write)

    # Each layer owns its own physical slots: writing layer 1 must not perturb layer 0.
    torch.testing.assert_close(cache.key_cache[0][:2].cpu(), torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]]))
    torch.testing.assert_close(cache.key_cache[1][:2].cpu(), torch.tensor([[[101.0, 102.0]], [[103.0, 104.0]]]))
    torch.testing.assert_close(cache.value_cache[0][:2].cpu(), torch.tensor([[[11.0, 12.0]], [[13.0, 14.0]]]))
    torch.testing.assert_close(cache.value_cache[1][:2].cpu(), torch.tensor([[[111.0, 112.0]], [[113.0, 114.0]]]))


@pytest.mark.cuda
def test_ios_prepares_and_transfers_batch_to_cuda() -> None:
    cb_config = ContinuousBatchingConfig(
        num_blocks=2,
        block_size=4,
        max_num_batched_tokens=8,
        max_model_len=16,
        use_async_batching=True,
    )
    cache = PagedAttentionCache(
        config=_config(),
        continuous_batching_config=cb_config,
        device="cuda",
        dtype=torch.float32,
    )
    cache.allocate_blocks(1, "req", allocated_blocks=0)
    state = RequestState(request_id="req", initial_tokens=[10])
    state.position_offset = 1
    state.tokens_to_process = [20]
    ios = ContinuousBatchingIOs(
        cache=cache,
        config=_config(),
        device="cuda",
        model_dtype=torch.float32,
    )

    ios.prepare_batch_tensors(
        requests_in_batch=[FutureRequestState(state=state, has_new_token=True, query_length=1)],
        use_decode_fast_path=True,
        num_q_tokens=1,
        max_kv_read=1,
    )
    kwargs = ios.get_model_kwargs()
    ios.compute_stream.synchronize()

    assert kwargs["input_ids"].is_cuda
    assert kwargs["position_ids"].is_cuda
    assert kwargs["block_table"].is_cuda
    assert kwargs["input_ids"].cpu().tolist() == [[20]]
    assert kwargs["position_ids"].cpu().tolist() == [[1]]
    assert kwargs["block_table"].cpu().tolist() == [[[0, -1, -1, -1]]]


@pytest.mark.cuda
def test_ios_carries_over_previous_cuda_outputs() -> None:
    cb_config = ContinuousBatchingConfig(
        num_blocks=2,
        block_size=4,
        max_num_batched_tokens=8,
        use_async_batching=True,
    )
    cache = PagedAttentionCache(
        config=_config(),
        continuous_batching_config=cb_config,
        device="cuda",
        dtype=torch.float32,
    )
    ios = ContinuousBatchingIOs(
        cache=cache,
        config=_config(),
        device="cuda",
        model_dtype=torch.float32,
    )
    input_ids = torch.tensor([[100, 101, 102]], device="cuda", dtype=torch.int32)
    carry_over_ids = torch.tensor([-1, 0, 2], device="cuda", dtype=torch.int32)
    prev_output_ids = torch.zeros((1, 8), device="cuda", dtype=torch.int32)
    prev_output_ids[0, 0] = 200
    prev_output_ids[0, 2] = 202

    ios.carry_over_tokens(input_ids, carry_over_ids, prev_output_ids)

    assert input_ids.cpu().tolist() == [[100, 200, 202]]
