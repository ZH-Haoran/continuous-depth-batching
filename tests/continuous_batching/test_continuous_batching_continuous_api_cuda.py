"""End-to-end GPU verification for the phase-separated continuous-batching scheduler.

The scheduler keeps prefill and decode in separate iterations (a batch is never
mixed), so decode always runs on the paged block-table fast path. These tests run
the real paged-attention Ouro forward through the CB engine with several
different-length prompts and multiple decode steps, forcing the scheduler to
interleave prefill-only and decode-only batches.

The pipelined loop (host running one batch ahead, on-GPU carry-over, CUDA graphs) must
produce the same tokens as the serialized loop -- the invariant most at risk once
prefill and decode no longer share a batch. Run on a GPU node via
``shells/pytest_cuda.sh`` with ``LOOPED_CDB_RUN_CUDA_TESTS=1``.
"""

from __future__ import annotations

import pytest
import torch

from looped_cdb.continuous_batching import ContinuousBatchingEngine
from looped_cdb.continuous_batching.config import ContinuousBatchingConfig
from looped_cdb.models.ouro import OuroConfig, OuroForCausalLM
from looped_cdb.models.ouro.kv_cache_policy import configure_kv_cache_policy

pytestmark = pytest.mark.cuda

# Prompts of different lengths (each starting with BOS) so prefill batches vary in shape and the
# scheduler must interleave prefill-only and decode-only iterations.
PROMPTS = [[1, 5, 7, 3], [1, 9], [1, 3, 4, 6, 8], [1, 2]]
MAX_NEW_TOKENS = 5


@pytest.fixture(scope="module")
def ouro_model() -> OuroForCausalLM:
    """A small real Ouro causal LM on CUDA with paged FlashAttention-style serving."""

    config = OuroConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=512,
        total_ut_steps=4,
        layer_types=["full_attention", "full_attention"],
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    torch.manual_seed(0)
    model = OuroForCausalLM(config).to("cuda", dtype=torch.bfloat16).eval()
    model.set_attn_implementation("paged|flash_attention_3")
    configure_kv_cache_policy(model, kv_policy="single")
    return model


def _cb_engine(
    model: OuroForCausalLM, *, sync: bool, use_cuda_graph: bool, use_cuda_graph_prefill: bool = True
) -> ContinuousBatchingEngine:
    config = ContinuousBatchingConfig(
        num_blocks=256,
        max_num_batched_tokens=256,
        block_size=256,
        max_model_len=8192,
        use_async_batching=not sync,
        use_cuda_graph=use_cuda_graph,
        use_cuda_graph_prefill=use_cuda_graph_prefill,
    )
    return ContinuousBatchingEngine.from_model(model, cb_config=config, dtype=torch.bfloat16)


def _run(engine: ContinuousBatchingEngine) -> list[list[int]]:
    outputs = engine.generate_batch(
        input_ids=PROMPTS,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=None,
        warmup=True,
    )
    return [list(output.generated_tokens) for output in outputs]


def test_phase_separated_async_matches_sync(ouro_model: OuroForCausalLM) -> None:
    # The pipelined and serialized loops run the same forward on the same (homogeneous) batches, so
    # they must agree token-for-token. The pipelined loop additionally exercises the on-GPU
    # carry-over that phase separation stresses: a request prefilled in one batch decodes in a
    # later, non-adjacent batch, so the carry-over must fall back to the host token correctly.
    async_tokens = _run(_cb_engine(ouro_model, sync=False, use_cuda_graph=True))
    sync_tokens = _run(_cb_engine(ouro_model, sync=True, use_cuda_graph=False))

    assert all(len(tokens) == MAX_NEW_TOKENS for tokens in async_tokens)
    assert async_tokens == sync_tokens


def test_prefill_graphs_replay_every_prefill_and_match_eager_prefill(ouro_model: OuroForCausalLM) -> None:
    # Every varlen bucket is captured at warm-up, so the run only replays; the padded, piecewise
    # forward must sample the same tokens as the eager prefill with decode graphs alone.
    graphed = _cb_engine(ouro_model, sync=False, use_cuda_graph=True, use_cuda_graph_prefill=True)
    graphed_tokens = _run(graphed)
    eager_prefill = _cb_engine(ouro_model, sync=False, use_cuda_graph=True, use_cuda_graph_prefill=False)
    eager_tokens = _run(eager_prefill)

    runner = graphed.runner
    assert runner.prefill_graph_captures == len(runner.prefill_graph_buckets)
    assert runner.prefill_graph_hits >= graphed.last_stats.prefill_batches > 0
    assert eager_prefill.runner.prefill_graph_hits == 0
    assert graphed_tokens == eager_tokens
    report = runner.warmup_report
    assert report is not None
    assert report.prefill_graphs == len(runner.prefill_graph_buckets)
    assert report.decode_graphs == len(runner.decode_graph_buckets)
    assert report.prefill_seconds > 0 and report.device_allocated_bytes > 0


def test_phase_separated_generation_completes_for_all_prompts(ouro_model: OuroForCausalLM) -> None:
    # Every prompt must reach its full generation length even though prefill and decode never share
    # a batch, i.e. the prefill/decode interleaving makes progress and does not deadlock.
    tokens = _run(_cb_engine(ouro_model, sync=False, use_cuda_graph=True))

    assert len(tokens) == len(PROMPTS)
    assert all(len(prompt_tokens) == MAX_NEW_TOKENS for prompt_tokens in tokens)


# Equal-length prompts (B=2, L=4): the decode-latency measurement requires them so all requests
# reach the same context length once prefill drains.
MEASURE_PROMPTS = [[1, 5, 7, 3], [1, 9, 2, 4]]


def _measure_engine(model: OuroForCausalLM) -> ContinuousBatchingEngine:
    # safety_margin=0 makes the scheduler drain every prefill before any decode, which the
    # measurement depends on. Warm up so the decode graph for the full batch is captured.
    config = ContinuousBatchingConfig(
        num_blocks=256,
        max_num_batched_tokens=256,
        block_size=256,
        max_model_len=8192,
        use_async_batching=True,
        use_cuda_graph=True,
        safety_margin=0.0,
    )
    engine = ContinuousBatchingEngine.from_model(model, cb_config=config, dtype=torch.bfloat16)
    engine.runner.warmup(engine.model)
    return engine


def test_measure_decode_step_latency_times_uniform_full_batch(ouro_model: OuroForCausalLM) -> None:
    # The window brackets exactly ``timed_decode_steps`` full-batch decode ticks. Because
    # safety_margin=0 drains all prefill first, both requests decode together at one context length,
    # so context_min == context_max. Timings are positive and the GPU-event floor never exceeds the
    # wall clock.
    engine = _measure_engine(ouro_model)

    timing = engine.measure_decode_step_latency(
        MEASURE_PROMPTS,
        max_new_tokens=10,
        warmup_decode_steps=2,
        timed_decode_steps=3,
    )

    assert timing.batch_size == 2
    assert timing.warmup_decode_steps == 2
    assert timing.timed_decode_steps == 3
    assert timing.context_length == len(MEASURE_PROMPTS[0])
    assert timing.context_min == timing.context_max
    assert timing.context_max >= timing.context_length
    assert timing.wall_elapsed_ms > 0.0
    assert timing.per_step_ms > 0.0
    assert timing.gpu_per_step_ms > 0.0
    assert timing.gpu_per_step_ms <= timing.per_step_ms + 1e-6


def test_measure_decode_step_latency_rejects_invalid_arguments(ouro_model: OuroForCausalLM) -> None:
    # The four guard paths all raise before any compute, so the batch is never half-assembled.
    engine = _measure_engine(ouro_model)

    with pytest.raises(ValueError, match="warmup_decode_steps"):
        engine.measure_decode_step_latency(
            MEASURE_PROMPTS, max_new_tokens=10, warmup_decode_steps=0, timed_decode_steps=3
        )
    with pytest.raises(ValueError, match="timed_decode_steps"):
        engine.measure_decode_step_latency(
            MEASURE_PROMPTS, max_new_tokens=10, warmup_decode_steps=2, timed_decode_steps=0
        )
    with pytest.raises(ValueError, match="must exceed"):
        engine.measure_decode_step_latency(
            MEASURE_PROMPTS, max_new_tokens=5, warmup_decode_steps=2, timed_decode_steps=3
        )
    with pytest.raises(ValueError, match="equal-length"):
        engine.measure_decode_step_latency(
            [[1, 5, 7, 3], [1, 9]], max_new_tokens=10, warmup_decode_steps=2, timed_decode_steps=3
        )


def _cap_engine(model: OuroForCausalLM, *, max_model_len: int) -> ContinuousBatchingEngine:
    config = ContinuousBatchingConfig(
        num_blocks=256,
        max_num_batched_tokens=256,
        block_size=256,
        max_model_len=max_model_len,
        use_async_batching=True,
        use_cuda_graph=True,
    )
    return ContinuousBatchingEngine.from_model(model, cb_config=config, dtype=torch.bfloat16)


def test_generate_batch_caps_max_new_tokens_to_model_len(ouro_model: OuroForCausalLM) -> None:
    # max_model_len bounds prompt + generated (vLLM semantics), enforced inside generate_batch
    # rather than only in the helper unit test. With a 6-token budget, the length-4 prompt may
    # generate only 2 tokens and the length-2 prompt only 4, regardless of the requested
    # max_new_tokens=5. eos_token_id=None means the only stop is the length cap.
    outputs = _cap_engine(ouro_model, max_model_len=6).generate_batch(
        input_ids=[[1, 5, 7, 3], [1, 9]],
        max_new_tokens=5,
        eos_token_id=None,
        warmup=True,
    )

    assert [len(output.generated_tokens) for output in outputs] == [2, 4]


def test_generate_batch_rejects_prompt_with_no_room(ouro_model: OuroForCausalLM) -> None:
    # A prompt with no room to generate (len == max_model_len) is rejected by raising, aborting
    # the batch. The guard fires before any compute, so warmup is unnecessary.
    with pytest.raises(ValueError, match="max_model_len"):
        _cap_engine(ouro_model, max_model_len=4).generate_batch(
            input_ids=[[1, 2, 3, 4]],
            max_new_tokens=5,
            eos_token_id=None,
            warmup=False,
        )
