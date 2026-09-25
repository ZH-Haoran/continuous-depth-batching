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
    num_blocks: int = 8,
    block_size: int = 4,
    max_num_batched_tokens: int = 8,
    max_blocks_per_request: int = 32,
    use_cuda_graph: bool = False,
) -> ContinuousBatchingConfig:
    return ContinuousBatchingConfig(
        num_blocks=num_blocks,
        block_size=block_size,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_blocks_per_request * block_size,
        use_cuda_graph=use_cuda_graph,
    )


def _cache(cb_config: ContinuousBatchingConfig | None = None) -> PagedAttentionCache:
    return PagedAttentionCache(
        config=_config(),
        continuous_batching_config=cb_config or _cb_config(),
        device="cpu",
        dtype=torch.float32,
    )


def _cache_for_config(
    config: PreTrainedConfig,
    cb_config: ContinuousBatchingConfig | None = None,
) -> PagedAttentionCache:
    return PagedAttentionCache(
        config=config,
        continuous_batching_config=cb_config or _cb_config(),
        device="cpu",
        dtype=torch.float32,
    )


def _ios(
    cache: PagedAttentionCache,
    *,
    config: PreTrainedConfig | None = None,
) -> ContinuousBatchingIOs:
    return ContinuousBatchingIOs(
        cache=cache,
        config=config or _config(),
        device="cpu",
        model_dtype=torch.float32,
    )


class TokenIncrementModel(nn.Module):
    def __init__(self, vocab_size: int = 16) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor, **_: Any) -> SimpleNamespace:
        logits = torch.full(
            (1, input_ids.size(1), self.vocab_size),
            -10.0,
            dtype=torch.float32,
            device=input_ids.device,
        )
        next_tokens = (input_ids[0].to(dtype=torch.long) + 1) % self.vocab_size
        logits[0, torch.arange(input_ids.size(1), device=input_ids.device), next_tokens] = 10.0
        return SimpleNamespace(logits=logits)


class SharedKvCaptureModel(TokenIncrementModel):
    def __init__(self) -> None:
        super().__init__()
        self.seen_kv_policy: list[str | None] = []
        self.seen_kv_slots_per_layer: list[int | None] = []

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> SimpleNamespace:
        self.seen_kv_policy.append(kwargs.get("kv_policy"))
        self.seen_kv_slots_per_layer.append(kwargs.get("kv_slots_per_layer"))
        return super().forward(input_ids, **kwargs)


def test_model_runner_computes_greedy_prefill_output_in_request_order() -> None:
    cb_config = _cb_config()
    cache = _cache(cb_config)
    cache.allocate_blocks(1, "req", allocated_blocks=0)
    state = RequestState(request_id="req", initial_tokens=[10, 11])
    state.tokens_to_process = [10, 11]
    future_state = FutureRequestState(state=state, has_new_token=True, query_length=2)
    ios = _ios(cache)
    ios.prepare_batch_tensors(
        requests_in_batch=[future_state],
        use_decode_fast_path=False,
        num_q_tokens=2,
        max_kv_read=0,
    )
    runner = ModelRunner(engine_config=cb_config, inputs_and_outputs=ios, cache=cache)

    runner.compute_batch(TokenIncrementModel(), ios.get_model_kwargs())
    new_tokens = ios.consume_output_tokens(ios.enqueue_output_copy(), 1)

    assert ios.requests_in_batch == [future_state]
    assert new_tokens == [12]


def test_model_runner_passes_kv_policy_for_shared_ouro_cache() -> None:
    config = _config(model_type="ouro", total_ut_steps=3)
    config._cdb_kv_policy = "single"
    cb_config = _cb_config()
    cache = _cache_for_config(config, cb_config)
    ios = _ios(cache, config=config)
    runner = ModelRunner(engine_config=cb_config, inputs_and_outputs=ios, cache=cache)
    model = SharedKvCaptureModel()

    runner.compute_batch(
        model,
        {
            "input_ids": torch.tensor([[3]], dtype=torch.int32),
            "logits_indices": torch.tensor([0], dtype=torch.int32),
        },
    )

    assert model.seen_kv_policy == ["single"]
    assert model.seen_kv_slots_per_layer == [1]


def test_model_runner_warmup_passes_kv_policy_for_shared_ouro_cache() -> None:
    config = _config(model_type="ouro", total_ut_steps=3)
    config._cdb_kv_policy = "single"
    cb_config = _cb_config(use_cuda_graph=False)
    cache = _cache_for_config(config, cb_config)
    ios = _ios(cache, config=config)
    runner = ModelRunner(engine_config=cb_config, inputs_and_outputs=ios, cache=cache)
    model = SharedKvCaptureModel()

    runner.run_one_warmup(model=model, num_requests=1)

    assert model.seen_kv_policy == ["single"]
    assert model.seen_kv_slots_per_layer == [1]


def test_model_runner_computes_greedy_decode_fast_path_batch() -> None:
    cb_config = _cb_config(max_blocks_per_request=4)
    cache = _cache(cb_config)
    cache.allocate_blocks(1, "req-0", allocated_blocks=0)
    cache.allocate_blocks(1, "req-1", allocated_blocks=0)
    state_0 = RequestState(request_id="req-0", initial_tokens=[1])
    state_0.position_offset = 1
    state_0.tokens_to_process = [4]
    state_1 = RequestState(request_id="req-1", initial_tokens=[2])
    state_1.position_offset = 1
    state_1.tokens_to_process = [7]
    future_states = [
        FutureRequestState(state=state_0, has_new_token=True, query_length=1),
        FutureRequestState(state=state_1, has_new_token=True, query_length=1),
    ]
    ios = _ios(cache)
    ios.prepare_batch_tensors(
        requests_in_batch=future_states,
        use_decode_fast_path=True,
        num_q_tokens=2,
        max_kv_read=1,
    )
    runner = ModelRunner(engine_config=cb_config, inputs_and_outputs=ios, cache=cache)

    runner.compute_batch(TokenIncrementModel(), ios.get_model_kwargs())
    new_tokens = ios.consume_output_tokens(ios.enqueue_output_copy(), 2)

    assert new_tokens == [5, 8]


def test_model_runner_pads_only_the_decode_fast_path() -> None:
    cb_config = _cb_config(use_cuda_graph=False)
    cache = _cache(cb_config)
    ios = _ios(cache)
    runner = ModelRunner(engine_config=cb_config, inputs_and_outputs=ios, cache=cache)
    runner.pad_inputs = True

    assert runner.maybe_pad_inputs(num_q_tokens=5, max_kv_read=9, use_decode_fast_path=False) == (5, 9)
    assert runner.maybe_pad_inputs(num_q_tokens=3, max_kv_read=9, use_decode_fast_path=True) == (4, 0)


def test_model_runner_rejects_sampling_for_now() -> None:
    cb_config = _cb_config()
    cache = _cache(cb_config)
    ios = _ios(cache)

    with pytest.raises(NotImplementedError, match="greedy"):
        ModelRunner(engine_config=cb_config, inputs_and_outputs=ios, cache=cache, do_sample=True)
