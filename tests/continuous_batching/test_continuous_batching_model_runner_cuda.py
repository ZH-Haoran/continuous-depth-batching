from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn
from transformers import PreTrainedConfig

from looped_cdb.continuous_batching.cache import PagedAttentionCache
from looped_cdb.continuous_batching.config import ContinuousBatchingConfig
from looped_cdb.continuous_batching.input_outputs import ContinuousBatchingIOs
from looped_cdb.continuous_batching.model_runner import ModelRunner
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


def _cache(cb_config: ContinuousBatchingConfig) -> PagedAttentionCache:
    return PagedAttentionCache(
        config=_config(),
        continuous_batching_config=cb_config,
        device="cuda",
        dtype=torch.float32,
    )


def _ios(cache: PagedAttentionCache) -> ContinuousBatchingIOs:
    return ContinuousBatchingIOs(cache=cache, config=_config(), device="cuda", model_dtype=torch.float32)


class TokenIncrementModel(nn.Module):
    def __init__(self, vocab_size: int = 16) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.register_buffer("vocab_ids", torch.arange(vocab_size, dtype=torch.long))

    def forward(self, input_ids: torch.Tensor, **_: Any) -> SimpleNamespace:
        next_tokens = (input_ids[0].to(dtype=torch.long) + 1) % self.vocab_size
        logits = -torch.abs(self.vocab_ids.view(1, 1, -1) - next_tokens.view(1, -1, 1)).to(dtype=torch.float32)
        return SimpleNamespace(logits=logits)


@pytest.mark.cuda
def test_model_runner_cuda_graph_capture_and_replay_decode_path() -> None:
    cb_config = ContinuousBatchingConfig(
        num_blocks=4,
        block_size=4,
        max_num_batched_tokens=4,
        max_model_len=16,
        use_cuda_graph=True,
    )
    cache = _cache(cb_config)
    cache.allocate_blocks(1, "req", allocated_blocks=0)
    ios = _ios(cache)
    runner = ModelRunner(engine_config=cb_config, inputs_and_outputs=ios, cache=cache)
    model = TokenIncrementModel().cuda()

    state = RequestState(request_id="req", initial_tokens=[1])
    state.position_offset = 1
    state.tokens_to_process = [5]
    ios.prepare_batch_tensors(
        requests_in_batch=[FutureRequestState(state=state, has_new_token=True, query_length=1)],
        use_decode_fast_path=True,
        num_q_tokens=1,
        max_kv_read=1,
    )
    runner.compute_batch(model, ios.get_model_kwargs(use_padding=True))
    assert ios.consume_output_tokens(ios.enqueue_output_copy(), 1) == [6]
    assert ios.get_graph() is not None

    # A different request id: a request appearing in consecutive batches would have its host-written
    # token replaced by the on-device carry-over, which is not what this replay check is after.
    cache.allocate_blocks(1, "req-2", allocated_blocks=0)
    state = RequestState(request_id="req-2", initial_tokens=[1])
    state.position_offset = 1
    state.tokens_to_process = [9]
    ios.prepare_batch_tensors(
        requests_in_batch=[FutureRequestState(state=state, has_new_token=True, query_length=1)],
        use_decode_fast_path=True,
        num_q_tokens=1,
        max_kv_read=1,
    )
    runner.compute_batch(model, ios.get_model_kwargs(use_padding=True))
    assert ios.consume_output_tokens(ios.enqueue_output_copy(), 1) == [10]


@pytest.mark.cuda
def test_model_runner_cuda_graph_decode_path_handles_padded_batch_rows() -> None:
    cb_config = ContinuousBatchingConfig(
        num_blocks=8,
        block_size=4,
        max_num_batched_tokens=4,
        max_model_len=16,
        use_cuda_graph=True,
    )
    cache = _cache(cb_config)
    ios = _ios(cache)
    runner = ModelRunner(engine_config=cb_config, inputs_and_outputs=ios, cache=cache)
    model = TokenIncrementModel().cuda()
    future_states = []
    for idx, token_id in enumerate([5, 7, 9]):
        request_id = f"req-{idx}"
        cache.allocate_blocks(1, request_id, allocated_blocks=0)
        state = RequestState(request_id=request_id, initial_tokens=[token_id])
        state.position_offset = 1
        state.tokens_to_process = [token_id]
        future_states.append(FutureRequestState(state=state, has_new_token=True, query_length=1))

    ios.prepare_batch_tensors(
        requests_in_batch=future_states,
        use_decode_fast_path=True,
        num_q_tokens=4,
        max_kv_read=0,
    )
    batch_data = ios.get_model_kwargs(use_padding=True)
    assert batch_data["input_ids"].shape == (1, 4)
    assert batch_data["cu_seq_lens_k"].tolist() == [0, 2, 4, 6, 6]
    assert batch_data["block_table"][0, 3].tolist() == [-1, -1, -1, -1]

    runner.compute_batch(model, batch_data)

    assert ios.consume_output_tokens(ios.enqueue_output_copy(), 3) == [6, 8, 10]
    assert ios.get_graph() is not None


@pytest.mark.cuda
def test_model_runner_warmup_captures_graphs_and_frees_blocks() -> None:
    cb_config = ContinuousBatchingConfig(
        num_blocks=16,
        block_size=4,
        max_num_batched_tokens=4,
        max_model_len=32,
        use_cuda_graph=True,
    )
    cache = _cache(cb_config)
    ios = _ios(cache)
    runner = ModelRunner(engine_config=cb_config, inputs_and_outputs=ios, cache=cache)

    runner.warmup(TokenIncrementModel().cuda())
    ios.compute_stream.synchronize()

    # Both the decode and the prefill warm-up return their fake requests' blocks.
    assert cache.get_num_free_blocks() == cache.num_blocks
    assert len(ios.device_buffers.graphs._storage) >= 1
    assert all(len(key) == 1 for key in ios.device_buffers.graphs._storage)
    assert len(runner.prefill_graphs) == len(runner.prefill_graph_buckets) == 1
    report = runner.warmup_report
    assert report is not None
    assert report.decode_graphs == len(ios.device_buffers.graphs._storage)
    assert report.prefill_graphs == 1
    assert report.device_allocated_bytes > 0
