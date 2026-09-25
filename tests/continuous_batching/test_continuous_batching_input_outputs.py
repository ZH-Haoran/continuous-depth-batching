import pytest
import torch
from transformers import PreTrainedConfig

from looped_cdb.continuous_batching.cache import PagedAttentionCache
from looped_cdb.continuous_batching.config import ContinuousBatchingConfig
from looped_cdb.continuous_batching.input_outputs import ContinuousBatchingIOs
from looped_cdb.continuous_batching.requests import TMP_TOKEN_ID, FutureRequestState, RequestState


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
    num_blocks: int = 4,
    block_size: int = 4,
    max_num_batched_tokens: int = 8,
    max_blocks_per_request: int = 32,
) -> ContinuousBatchingConfig:
    return ContinuousBatchingConfig(
        num_blocks=num_blocks,
        block_size=block_size,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_blocks_per_request * block_size,
    )


def _cache(cb_config: ContinuousBatchingConfig | None = None) -> PagedAttentionCache:
    return PagedAttentionCache(
        config=_config(),
        continuous_batching_config=cb_config or _cb_config(),
        device="cpu",
        dtype=torch.float32,
    )


def test_sync_ios_prepare_prefill_chunk_without_cache_reads() -> None:
    cache = _cache()
    cache.allocate_blocks(1, "req", allocated_blocks=0)
    state = RequestState(request_id="req", initial_tokens=[10, 11, 12])
    state.tokens_to_process = [10, 11]
    future_state = FutureRequestState(state=state, has_new_token=False, query_length=2)
    ios = ContinuousBatchingIOs(
        cache=cache,
        config=_config(),
        device="cpu",
        model_dtype=torch.float32,
    )

    ios.prepare_batch_tensors(
        requests_in_batch=[future_state],
        use_decode_fast_path=False,
        num_q_tokens=2,
        max_kv_read=0,
    )
    kwargs = ios.get_model_kwargs()

    assert kwargs["input_ids"].tolist() == [[10, 11]]
    assert kwargs["position_ids"].tolist() == [[0, 1]]
    assert kwargs["cu_seq_lens_q"].tolist() == [0, 2]
    assert kwargs["cu_seq_lens_k"].tolist() == [0, 2]
    assert kwargs["max_seqlen_q"] == 2
    assert kwargs["max_seqlen_k"] == 2
    assert kwargs["write_index"].tolist() == [0, 1]
    assert kwargs["read_index"].numel() == 0
    assert kwargs["block_table"] is None
    assert state.position_offset == 2
    assert state.tokens_to_process == [10, 11]


def test_sync_ios_prepare_chunk_with_cache_reads() -> None:
    cache = _cache()
    cache.allocate_blocks(2, "req", allocated_blocks=0)
    state = RequestState(request_id="req", initial_tokens=[10, 11, 12, 13])
    state.position_offset = 2
    state.tokens_to_process = [12, 13]
    future_state = FutureRequestState(state=state, has_new_token=True, query_length=2)
    ios = ContinuousBatchingIOs(
        cache=cache,
        config=_config(),
        device="cpu",
        model_dtype=torch.float32,
    )

    ios.prepare_batch_tensors(
        requests_in_batch=[future_state],
        use_decode_fast_path=False,
        num_q_tokens=2,
        max_kv_read=2,
    )
    kwargs = ios.get_model_kwargs()

    assert kwargs["input_ids"].tolist() == [[12, 13]]
    assert kwargs["position_ids"].tolist() == [[2, 3]]
    assert kwargs["cu_seq_lens_q"].tolist() == [0, 2]
    assert kwargs["cu_seq_lens_k"].tolist() == [0, 4]
    assert kwargs["logits_indices"].tolist()[:2] == [1, 0]
    assert kwargs["write_index"].tolist() == [2, 3]
    assert kwargs["read_index"].tolist() == [0, 1, 2, 3]
    assert state.position_offset == 4
    assert state.tokens_to_process == [TMP_TOKEN_ID]
    assert state.tokens_in_flight == 1
    assert ios.host_buffers.req_id_to_new_token_position == {"req": 1}


def test_sync_ios_prepare_decode_fast_path_block_table() -> None:
    cb_config = _cb_config(max_blocks_per_request=4)
    cache = _cache(cb_config)
    cache.allocate_blocks(1, "req-0", allocated_blocks=0)
    cache.allocate_blocks(2, "req-1", allocated_blocks=0)
    state_0 = RequestState(request_id="req-0", initial_tokens=[10])
    state_0.position_offset = 1
    state_0.tokens_to_process = [20]
    state_1 = RequestState(request_id="req-1", initial_tokens=[30, 31, 32, 33])
    state_1.position_offset = 4
    state_1.tokens_to_process = [40]
    requests = [
        FutureRequestState(state=state_0, has_new_token=True, query_length=1),
        FutureRequestState(state=state_1, has_new_token=True, query_length=1),
    ]
    ios = ContinuousBatchingIOs(
        cache=cache,
        config=_config(),
        device="cpu",
        model_dtype=torch.float32,
    )

    ios.prepare_batch_tensors(
        requests_in_batch=requests,
        use_decode_fast_path=True,
        num_q_tokens=2,
        max_kv_read=4,
    )
    kwargs = ios.get_model_kwargs()

    assert kwargs["input_ids"].tolist() == [[20, 40]]
    assert kwargs["position_ids"].tolist() == [[1, 4]]
    assert kwargs["cu_seq_lens_q"].tolist() == [0, 1, 2]
    assert kwargs["cu_seq_lens_k"].tolist() == [0, 2, 7]
    assert kwargs["max_seqlen_q"] == 1
    assert kwargs["max_seqlen_k"] == 1
    assert kwargs["block_table"].tolist() == [[[0, -1, -1, -1], [1, 2, -1, -1]]]
    assert kwargs["read_index"].numel() == 0
    assert ios.use_block_table
    assert state_0.tokens_to_process == [TMP_TOKEN_ID]
    assert state_1.tokens_to_process == [TMP_TOKEN_ID]


def test_ios_carries_previous_outputs_into_placeholder_inputs() -> None:
    # Two consecutive decode batches for the same request: the second one's input holds the
    # placeholder, and the carry-over scatter must fill it from the first batch's output row.
    cache = _cache(_cb_config(max_blocks_per_request=4))
    cache.allocate_blocks(1, "req", allocated_blocks=0)
    state = RequestState(request_id="req", initial_tokens=[10])
    state.position_offset = 1
    state.tokens_to_process = [20]
    ios = ContinuousBatchingIOs(
        cache=cache,
        config=_config(),
        device="cpu",
        model_dtype=torch.float32,
    )

    ios.prepare_batch_tensors(
        requests_in_batch=[FutureRequestState(state=state, has_new_token=True, query_length=1)],
        use_decode_fast_path=True,
        num_q_tokens=1,
        max_kv_read=1,
    )
    ios.get_model_kwargs()
    ios.output_ids[0, 0] = 42  # the token the first batch sampled
    assert ios.consume_output_tokens(ios.enqueue_output_copy(), 1) == [42]

    ios.prepare_batch_tensors(
        requests_in_batch=[FutureRequestState(state=state, has_new_token=True, query_length=1)],
        use_decode_fast_path=True,
        num_q_tokens=1,
        max_kv_read=1,
    )
    kwargs = ios.get_model_kwargs()
    assert kwargs["input_ids"].tolist() == [[TMP_TOKEN_ID]]

    carry_over_ids, prev_output_ids, _ = ios.get_cb_kwargs()
    ios.carry_over_tokens(kwargs["input_ids"], carry_over_ids, prev_output_ids)
    assert kwargs["input_ids"].tolist() == [[42]]


def test_sync_ios_keeps_varlen_seqlens_exact() -> None:
    cache = _cache(_cb_config(num_blocks=300, max_num_batched_tokens=8))
    cache.allocate_blocks(2, "req", allocated_blocks=0)
    state = RequestState(request_id="req", initial_tokens=[10, 11, 12, 13])
    state.position_offset = 3
    state.tokens_to_process = [13]
    future_state = FutureRequestState(state=state, has_new_token=True, query_length=1)
    ios = ContinuousBatchingIOs(
        cache=cache,
        config=_config(),
        device="cpu",
        model_dtype=torch.float32,
    )

    ios.prepare_batch_tensors(
        requests_in_batch=[future_state],
        use_decode_fast_path=False,
        num_q_tokens=1,
        max_kv_read=3,
    )
    kwargs = ios.get_model_kwargs(use_padding=False)

    assert kwargs["cu_seq_lens_q"].tolist() == [0, 1]
    assert kwargs["cu_seq_lens_k"].tolist() == [0, 4]
    assert kwargs["max_seqlen_q"] == 1
    assert kwargs["max_seqlen_k"] == 4
    assert kwargs["write_index"].tolist() == [3]
    assert kwargs["read_index"].tolist()[:4] == [0, 1, 2, 3]
    assert kwargs["read_index"].numel() == 4
    # Varlen batches have no decode graph key; padded to a token bucket they keep their real lengths
    # for the graphed attention and pad only the query rows.
    with pytest.raises(RuntimeError, match="decode fast path"):
        ios.device_buffers._get_graph_key()
    state.position_offset = 3
    ios.prepare_batch_tensors(
        requests_in_batch=[future_state],
        use_decode_fast_path=False,
        num_q_tokens=4,
        max_kv_read=3,
    )
    kwargs = ios.get_model_kwargs(use_padding=True)
    context = ios.prefill_graph_context()
    assert kwargs["input_ids"].shape == (1, 4)
    assert kwargs["cu_seq_lens_q"].tolist() == [0, 1]
    assert kwargs["write_index"].tolist() == [3, cache.trash_index, cache.trash_index, cache.trash_index]
    assert context.num_tokens == 1
    assert context.cu_seq_lens_q.tolist() == [0, 1]
    assert context.cu_seq_lens_k.tolist() == [0, 4]
    assert (context.max_seqlen_q, context.max_seqlen_k) == (1, 4)
    assert context.write_index.tolist() == [3]
    assert context.read_index.tolist() == [0, 1, 2, 3]


def test_sync_ios_pads_decode_fast_path_for_cuda_graphs() -> None:
    cb_config = _cb_config(max_blocks_per_request=4)
    cache = _cache(cb_config)
    cache.allocate_blocks(1, "req-0", allocated_blocks=0)
    cache.allocate_blocks(1, "req-1", allocated_blocks=0)
    state_0 = RequestState(request_id="req-0", initial_tokens=[10])
    state_0.position_offset = 1
    state_0.tokens_to_process = [20]
    state_1 = RequestState(request_id="req-1", initial_tokens=[30])
    state_1.position_offset = 1
    state_1.tokens_to_process = [40]
    ios = ContinuousBatchingIOs(
        cache=cache,
        config=_config(),
        device="cpu",
        model_dtype=torch.float32,
    )

    ios.prepare_batch_tensors(
        requests_in_batch=[
            FutureRequestState(state=state_0, has_new_token=True, query_length=1),
            FutureRequestState(state=state_1, has_new_token=True, query_length=1),
        ],
        use_decode_fast_path=True,
        num_q_tokens=2,
        max_kv_read=1,
    )
    kwargs = ios.get_model_kwargs(use_padding=True)

    assert kwargs["cu_seq_lens_q"].tolist() == [0, 1, 2]
    assert kwargs["block_table"].shape == (1, 2, 4)
    assert kwargs["max_seqlen_q"] == 1
    assert kwargs["max_seqlen_k"] == 1
    assert ios.device_buffers._get_graph_key() == (2,)
