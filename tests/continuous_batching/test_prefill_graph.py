"""Host-side checks of the breakable prefill graph machinery: buckets, aliases, capture bookkeeping, dispatch."""

from __future__ import annotations

import gc
import weakref
from typing import Any, ClassVar

import pytest
import torch
from transformers import PreTrainedConfig
from transformers.integrations.flash_paged import paged_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from looped_cdb.continuous_batching.cache import PagedAttentionCache
from looped_cdb.continuous_batching.config import ContinuousBatchingConfig
from looped_cdb.continuous_batching.input_outputs import ContinuousBatchingIOs
from looped_cdb.continuous_batching.model_runner import ModelRunner
from looped_cdb.continuous_batching.prefill_graph import (
    BreakableCapture,
    BreakableGraph,
    PrefillGraphContext,
    eager_break,
    graphed_paged_attention_forward,
    install_graphed_paged_attention,
    prefill_graph_buckets,
    prefill_graph_context,
    weak_alias,
)
from looped_cdb.continuous_batching.requests import FutureRequestState, RequestState, RequestStatus
from looped_cdb.continuous_batching.utils import decode_graph_buckets, pad_to_bucket


def test_decode_graph_buckets_follow_the_capture_list_and_end_at_the_cap() -> None:
    assert decode_graph_buckets(64) == [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64]
    buckets = decode_graph_buckets(512)
    assert len(buckets) == 52 and buckets[-1] == 512
    assert pad_to_bucket(257, buckets) == 272 and pad_to_bucket(497, buckets) == 512
    # A cap off the grid is still the last bucket, so every launch has a graph.
    assert decode_graph_buckets(100)[-3:] == [88, 96, 100]


def test_prefill_graph_buckets_are_fine_then_coarse_and_end_at_the_token_budget() -> None:
    buckets = prefill_graph_buckets(8192)
    assert buckets[:4] == [16, 32, 48, 64]
    assert buckets[-1] == 8192
    assert buckets == sorted(set(buckets))
    assert pad_to_bucket(1, buckets) == 16
    assert pad_to_bucket(257, buckets) == 288
    assert pad_to_bucket(8192, buckets) == 8192
    # A budget off the grid is still the last bucket, so every launch has a graph.
    assert prefill_graph_buckets(300)[-2:] == [288, 300]
    with pytest.raises(ValueError, match="exceed"):
        pad_to_bucket(301, prefill_graph_buckets(300))


def test_weak_alias_shares_memory_without_owning_it() -> None:
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    alias = weak_alias(source[:, 1:3])
    assert alias.dtype == torch.float32
    assert alias.shape == (3, 2)
    assert alias.data_ptr() == source[:, 1:3].data_ptr()
    source[1, 1] = -1.0
    assert alias[1, 0].item() == -1.0
    assert weak_alias(torch.zeros(2, dtype=torch.bfloat16)).dtype == torch.bfloat16
    assert weak_alias(torch.zeros(2, dtype=torch.int64)).dtype == torch.int64
    # The alias references a small holder object only, so the source dies with its last owner.
    source_ref = weakref.ref(source)
    del source
    gc.collect()
    assert source_ref() is None


class FakeGraph:
    """Stands in for ``torch.cuda.CUDAGraph``: records the capture lifecycle and replays by counting."""

    log: ClassVar[list[str]] = []
    created: ClassVar[int] = 0

    def __init__(self) -> None:
        self.index = FakeGraph.created
        FakeGraph.created += 1
        self.replays = 0

    def capture_begin(self, pool: Any, capture_error_mode: str) -> None:
        FakeGraph.log.append(f"begin{self.index}")

    def capture_end(self) -> None:
        FakeGraph.log.append(f"end{self.index}")

    def replay(self) -> None:
        self.replays += 1
        FakeGraph.log.append(f"replay{self.index}")

    def reset(self) -> None:
        FakeGraph.log.append(f"reset{self.index}")


def test_breakable_capture_splits_segments_at_breaks_and_replays_them_with_fresh_aliases() -> None:
    FakeGraph.log = []
    FakeGraph.created = 0
    calls: list[tuple[int, float]] = []

    @eager_break
    def scale_into(source: torch.Tensor, factor: int, out: torch.Tensor) -> None:
        calls.append((factor, float(source[0])))
        out.copy_(source * factor)

    source = torch.ones(2)
    out_a = torch.zeros(2)
    out_b = torch.zeros(2)

    graph = BreakableGraph()
    with BreakableCapture(graph, pool=None, graph_factory=FakeGraph):
        FakeGraph.log.append("work")
        scale_into(source, 2, out_a)
        FakeGraph.log.append("work")
        scale_into(source, 3, out_b)
        FakeGraph.log.append("work")

    assert FakeGraph.log == ["begin0", "work", "end0", "begin1", "work", "end1", "begin2", "work", "end2"]
    assert len(graph.segments) == 3
    assert len(graph.breaks) == 2
    assert out_a.tolist() == [2.0, 2.0]
    assert out_b.tolist() == [3.0, 3.0]

    # Replay re-runs each break after its segment, reading the source through a non-owning alias, so
    # a new value at the same address is what the break sees.
    source.fill_(5.0)
    FakeGraph.log = []
    graph.replay()
    assert FakeGraph.log == ["replay0", "replay1", "replay2"]
    assert calls[-2:] == [(2, 5.0), (3, 5.0)]
    assert out_a.tolist() == [10.0, 10.0]
    assert out_b.tolist() == [15.0, 15.0]

    # Outside a capture the decorated function runs eagerly, and a break may not return a value.
    scale_into(source, 4, out_a)
    assert out_a.tolist() == [20.0, 20.0]

    @eager_break
    def returns_something(x: torch.Tensor) -> torch.Tensor:
        return x

    with (
        pytest.raises(TypeError, match="return None"),
        BreakableCapture(BreakableGraph(), pool=None, graph_factory=FakeGraph),
    ):
        returns_something(source)


def test_breakable_captures_do_not_nest() -> None:
    with (
        BreakableCapture(BreakableGraph(), pool=None, graph_factory=FakeGraph),
        pytest.raises(RuntimeError, match="do not nest"),
        BreakableCapture(BreakableGraph(), pool=None, graph_factory=FakeGraph),
    ):
        pass


def test_graphed_attention_delegates_to_the_paged_forward_outside_a_context(monkeypatch) -> None:
    seen: dict[str, Any] = {}

    def fake_paged(module, q, k, v, attention_mask, **kwargs):
        seen["module"] = module
        seen["kwargs"] = kwargs
        return q, None

    import looped_cdb.continuous_batching.prefill_graph as prefill_graph

    monkeypatch.setattr(prefill_graph, "paged_attention_forward", fake_paged)
    q = torch.zeros(1, 2, 3, 4)
    out, weights = graphed_paged_attention_forward("module", q, q, q, None, cache="c", block_table=None)
    assert out is q and weights is None
    assert seen == {"module": "module", "kwargs": {"cache": "c", "block_table": None}}


def test_install_graphed_paged_attention_routes_the_paged_flash_implementations() -> None:
    original = ALL_ATTENTION_FUNCTIONS.get_interface("paged|flash_attention_3", None)
    try:
        install_graphed_paged_attention("paged|flash_attention_3")
        assert ALL_ATTENTION_FUNCTIONS.get_interface("paged|flash_attention_3", None) is graphed_paged_attention_forward
        # Idempotent, and untouched keys still map to the stock forward.
        install_graphed_paged_attention("paged|flash_attention_3")
        assert ALL_ATTENTION_FUNCTIONS.get_interface("paged|flash_attention_2", None) is paged_attention_forward
        with pytest.raises(ValueError, match="paged attention"):
            install_graphed_paged_attention("sdpa")
        with pytest.raises(ValueError, match="paged flash forward"):
            install_graphed_paged_attention("paged|sdpa")
    finally:
        ALL_ATTENTION_FUNCTIONS["paged|flash_attention_3"] = original


# --------------------------------------------------------------------------- runner padding


def _config() -> PreTrainedConfig:
    return PreTrainedConfig(
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        hidden_size=4,
        vocab_size=16,
        sliding_window=None,
        layer_types=None,
        _attn_implementation="paged|flash_attention_3",
    )


def _runner(**overrides: Any) -> ModelRunner:
    cb_config = ContinuousBatchingConfig(
        num_blocks=64,
        block_size=4,
        max_num_batched_tokens=64,
        max_num_seqs=8,
        max_model_len=32,
        use_cuda_graph=False,
        **overrides,
    )
    cache = PagedAttentionCache(
        config=_config(), continuous_batching_config=cb_config, device="cpu", dtype=torch.float32
    )
    ios = ContinuousBatchingIOs(cache=cache, config=_config(), device="cpu", model_dtype=torch.float32)
    return ModelRunner(engine_config=cb_config, inputs_and_outputs=ios, cache=cache)


def test_runner_pads_varlen_batches_to_a_bucket_only_with_prefill_graphs() -> None:
    runner = _runner()
    assert runner.maybe_pad_inputs(num_q_tokens=5, max_kv_read=9, use_decode_fast_path=False) == (5, 9)
    assert runner.pads_batch(use_decode_fast_path=False) is False
    assert runner.prefill_graph_buckets == []

    # The switches are set here because a CPU runner cannot build the pool the graphs need.
    runner.use_cuda_graph_prefill = True
    runner.prefill_graph_buckets = prefill_graph_buckets(64)
    assert runner.maybe_pad_inputs(num_q_tokens=5, max_kv_read=9, use_decode_fast_path=False) == (16, 9)
    assert runner.maybe_pad_inputs(num_q_tokens=64, max_kv_read=0, use_decode_fast_path=False) == (64, 0)
    assert runner.pads_batch(use_decode_fast_path=False) is True
    # Decode padding is unchanged and still off without decode graphs.
    assert runner.maybe_pad_inputs(num_q_tokens=3, max_kv_read=9, use_decode_fast_path=True) == (3, 9)


def test_runner_gathers_a_fixed_logits_row_count_under_prefill_graphs() -> None:
    runner = _runner()
    ios = runner.inputs_and_outputs
    states = []
    for index, length in enumerate((3, 2)):
        state = RequestState(request_id=f"r{index}", initial_tokens=list(range(1, length + 1)))
        state._status = RequestStatus.PREFILLING
        state.tokens_to_process = list(state.initial_tokens)
        runner.cache.allocate_blocks(1, state.request_id, 0)
        states.append(FutureRequestState(state, has_new_token=True, query_length=length))
    ios.prepare_batch_tensors(states, use_decode_fast_path=False, num_q_tokens=8, max_kv_read=0)
    ios.get_model_kwargs(use_padding=True)

    assert runner._num_logits_rows() == 2
    runner.use_cuda_graph_prefill = True
    # At most one sampled row per resident request (the cap of 8), never more rows than the bucket.
    assert runner._num_logits_rows() == 8
    ios.device_buffers.num_q_tokens = 4
    assert runner._num_logits_rows() == 4


def test_prefill_graph_context_is_scoped() -> None:
    from looped_cdb.continuous_batching.prefill_graph import active_prefill_graph_context

    context = PrefillGraphContext(
        num_tokens=1,
        cu_seq_lens_q=torch.tensor([0, 1]),
        cu_seq_lens_k=torch.tensor([0, 1]),
        max_seqlen_q=1,
        max_seqlen_k=1,
        write_index=torch.tensor([0]),
        read_index=torch.empty(0, dtype=torch.int64),
        cache=None,  # type: ignore[arg-type]
    )
    assert active_prefill_graph_context() is None
    with prefill_graph_context(context):
        assert active_prefill_graph_context() is context
    assert active_prefill_graph_context() is None
