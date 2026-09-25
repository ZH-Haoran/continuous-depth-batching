import time
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn
from transformers import PreTrainedConfig

from looped_cdb.continuous_batching.config import ContinuousBatchingConfig
from looped_cdb.continuous_batching.continuous_api import CacheFullError, ContinuousBatchingEngine


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


class TokenIncrementModel(nn.Module):
    def __init__(self, config: PreTrainedConfig) -> None:
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids: torch.Tensor, **_: Any) -> SimpleNamespace:
        logits = torch.full(
            (1, input_ids.size(1), self.config.vocab_size),
            -10.0,
            dtype=torch.float32,
            device=input_ids.device,
        )
        next_tokens = (input_ids[0].to(dtype=torch.long) + 1) % self.config.vocab_size
        logits[0, torch.arange(input_ids.size(1), device=input_ids.device), next_tokens] = 10.0
        return SimpleNamespace(logits=logits)


class GatherAwareTokenIncrementModel(TokenIncrementModel):
    """TokenIncrementModel that honors HF's index-tensor ``logits_to_keep`` like the Ouro model."""

    def forward(
        self, input_ids: torch.Tensor, logits_to_keep: int | torch.Tensor = 0, **kwargs: Any
    ) -> SimpleNamespace:
        output = super().forward(input_ids, **kwargs)
        if isinstance(logits_to_keep, torch.Tensor):
            return SimpleNamespace(logits=output.logits.index_select(1, logits_to_keep))
        return output


@pytest.mark.parametrize("use_async_batching", [False, True])
def test_continuous_batching_engine_generates_token_batch(use_async_batching: bool) -> None:
    cb_config = ContinuousBatchingConfig(
        num_blocks=8,
        block_size=4,
        max_num_batched_tokens=4,
        max_model_len=16,
        use_async_batching=use_async_batching,
        use_cuda_graph=False,
    )
    engine = ContinuousBatchingEngine.from_model(TokenIncrementModel(_config()), cb_config, dtype=torch.float32)

    outputs = engine.generate_batch(
        input_ids=[[1, 2], [5]],
        max_new_tokens=3,
        eos_token_id=None,
        warmup=False,
    )

    assert [output.generated_tokens for output in outputs] == [[3, 4, 5], [6, 7, 8]]
    assert engine.cache.get_num_free_blocks() == engine.cache.num_blocks


def test_cb_eos_finishes_computes_one_discarded_token_per_request() -> None:
    # With finishes discovered at consume time, CB keeps scheduling each request one token past
    # its length limit and discards the extra sample; the outputs must not change.
    def run(replay_eos_finishes: bool) -> tuple[list[list[int]], int]:
        cb_config = ContinuousBatchingConfig(
            num_blocks=8,
            block_size=4,
            max_num_batched_tokens=4,
            max_model_len=16,
            use_async_batching=True,
            use_cuda_graph=False,
            replay_eos_finishes=replay_eos_finishes,
        )
        engine = ContinuousBatchingEngine.from_model(TokenIncrementModel(_config()), cb_config, dtype=torch.float32)
        outputs = engine.generate_batch(input_ids=[[1, 2], [5]], max_new_tokens=3, eos_token_id=None, warmup=False)
        return [output.generated_tokens for output in outputs], engine.last_stats.decode_tokens

    predicted_tokens, predicted_decode_tokens = run(False)
    late_tokens, late_decode_tokens = run(True)

    assert late_tokens == predicted_tokens
    assert late_decode_tokens == predicted_decode_tokens + 2, "one lagged token per request"


def test_the_resident_cap_is_resolved_against_the_batch_buffers() -> None:
    """A cap wider than the token budget is resolved on the config, not only inside the scheduler.

    The runner pads decode batches and walks warmup shapes up to the config's cap, while the buffers
    it pads into are sized by ``max_num_batched_tokens``, so a cap left unresolved there would pad a
    batch past the end of its own tensors.
    """

    cb_config = ContinuousBatchingConfig(
        num_blocks=8,
        block_size=4,
        max_num_batched_tokens=6,
        max_model_len=16,
        max_num_seqs=64,
        use_cuda_graph=False,
    )
    engine = ContinuousBatchingEngine.from_model(TokenIncrementModel(_config()), cb_config, dtype=torch.float32)

    assert engine.cb_config.max_num_seqs == 6
    assert engine.scheduler.max_num_seqs == 6
    assert engine.runner._decode_token_cap() <= engine.cache.max_num_batched_tokens


@pytest.mark.parametrize("use_async_batching", [False, True])
def test_timed_arrivals_match_drain_and_stamp_latencies(use_async_batching: bool) -> None:
    cb_config = ContinuousBatchingConfig(
        num_blocks=8,
        block_size=4,
        max_num_batched_tokens=4,
        max_model_len=16,
        use_async_batching=use_async_batching,
        use_cuda_graph=False,
    )
    engine = ContinuousBatchingEngine.from_model(TokenIncrementModel(_config()), cb_config, dtype=torch.float32)

    drained = engine.generate_batch(input_ids=[[1, 2], [5]], max_new_tokens=3, eos_token_id=None, warmup=False)
    # The second request arrives long after the first has fully drained, so the loop must go
    # idle and sleep until the arrival rather than exit or spin.
    timed = engine.generate_batch(
        input_ids=[[1, 2], [5]],
        max_new_tokens=3,
        eos_token_id=None,
        warmup=False,
        arrival_offsets_s=[0.0, 0.2],
    )

    assert [output.generated_tokens for output in timed] == [output.generated_tokens for output in drained]
    for output in timed:
        assert output.created_time <= output.lifespan[0] <= output.first_token_time <= output.lifespan[1]
    # Arrival stamps are the scheduled release instants, exactly one offset gap apart.
    assert timed[1].created_time - timed[0].created_time == pytest.approx(0.2)


def _delayed(fn: Any, delay_s: float) -> Any:
    """Wrap a callable with a fixed pre-call sleep, to slow a fake model down to wall-clock scale."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        time.sleep(delay_s)
        return fn(*args, **kwargs)

    return wrapper


def _arrived_mid_flight(outputs: list[Any]) -> bool:
    """Whether some request arrived while another was between first schedule and finish.

    This is the busy-release property from the recorded stamps: a fake model fast enough to
    drain each cohort before the next offset elapses turns a dense-arrival test back into an
    idle-path test without failing anything, so the overlap must be asserted, not assumed.
    """

    return any(
        other is not output and other.lifespan[0] <= output.created_time <= other.lifespan[1]
        for output in outputs
        for other in outputs
    )


@pytest.mark.parametrize("use_async_batching", [False, True])
def test_dense_arrivals_release_into_a_busy_engine_and_match_drain(use_async_batching: bool) -> None:
    # Dense offsets against a first cohort still generating: the later requests are released
    # into a scheduler that already has active work (and, on the async path, an in-flight
    # batch), the mid-run admission path a real serving sweep exercises on every tick. Only
    # the release timing changes, so tokens must stay drain-identical. The per-forward delay
    # keeps the first cohort resident well past the later offsets (>= 7 ticks x 5 ms against
    # 20/40 ms arrivals), so the overlap is guaranteed, not left to scheduling luck.
    cb_config = ContinuousBatchingConfig(
        num_blocks=16,
        block_size=4,
        max_num_batched_tokens=4,
        max_model_len=16,
        use_async_batching=use_async_batching,
        use_cuda_graph=False,
    )
    model = TokenIncrementModel(_config())
    model.forward = _delayed(model.forward, 0.005)
    engine = ContinuousBatchingEngine.from_model(model, cb_config, dtype=torch.float32)
    prompts = [[1, 2], [5], [3, 4], [7], [2]]

    drained = engine.generate_batch(input_ids=prompts, max_new_tokens=6, eos_token_id=None, warmup=False)
    timed = engine.generate_batch(
        input_ids=prompts,
        max_new_tokens=6,
        eos_token_id=None,
        warmup=False,
        arrival_offsets_s=[0.0, 0.0, 0.02, 0.02, 0.04],
    )

    assert [output.generated_tokens for output in timed] == [output.generated_tokens for output in drained]
    assert _arrived_mid_flight(timed)
    for output in timed:
        assert output.created_time <= output.lifespan[0] <= output.first_token_time <= output.lifespan[1]


@pytest.mark.parametrize(
    ("input_ids", "max_new_tokens", "error_match"),
    [
        ([], 1, "at least one prompt"),
        ([[1], []], 1, "empty prompts"),
        ([[1]], 0, "max_new_tokens"),
    ],
)
def test_continuous_batching_engine_validates_generation_inputs(
    input_ids: list[list[int]],
    max_new_tokens: int,
    error_match: str,
) -> None:
    cb_config = ContinuousBatchingConfig(
        num_blocks=8,
        block_size=4,
        max_num_batched_tokens=4,
        max_model_len=16,
        use_async_batching=False,
        use_cuda_graph=False,
    )
    engine = ContinuousBatchingEngine.from_model(TokenIncrementModel(_config()), cb_config, dtype=torch.float32)

    with pytest.raises(ValueError, match=error_match):
        engine.generate_batch(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=None,
            warmup=False,
        )


# --- KV-cache-pressure preemption end to end (recompute + offload, CPU) --------------------------

# Four requests that each grow to five 2-token blocks (prompt 2 + 8 generated = 10 tokens) share a
# 6-block cache: any two decoding at once over-subscribe it, while a single request still fits, so
# self-preemption can always make progress. safety_margin=0.0 admits every prompt immediately. This
# mirrors the CDB oversubscription fixture so the reference engine gets the same preemption coverage.
_CB_OVERSUBSCRIBED = {
    "num_blocks": 6,
    "block_size": 2,
    "max_model_len": 12,
    "max_num_batched_tokens": 4,
    "safety_margin": 0.0,
    "use_async_batching": False,
    "use_cuda_graph": False,
}
_CB_PROMPTS = [[1, 2], [3, 4], [5, 6], [7, 8]]


def _cb_generate(
    prompts: list[list[int]],
    max_new_tokens: int | list[int],
    eos_token_id: int | None = None,
    **config_overrides: Any,
) -> tuple[ContinuousBatchingEngine, list[list[int]]]:
    cb_config = ContinuousBatchingConfig(**{**_CB_OVERSUBSCRIBED, **config_overrides})
    engine = ContinuousBatchingEngine.from_model(TokenIncrementModel(_config()), cb_config, dtype=torch.float32)
    outputs = engine.generate_batch(
        input_ids=prompts, max_new_tokens=max_new_tokens, eos_token_id=eos_token_id, warmup=False
    )
    return engine, [output.generated_tokens for output in outputs]


def test_cb_none_mode_raises_when_the_cache_is_oversubscribed() -> None:
    # With kv_pressure_mode='none' the engine does no preemption and must surface the overflow. This
    # also establishes that the fixture genuinely over-subscribes the cache, so the recompute/offload
    # tests (same config) must preempt to succeed.
    with pytest.raises(CacheFullError):
        _cb_generate(_CB_PROMPTS, 8, kv_pressure_mode="none")


@pytest.mark.parametrize("use_async_batching", [False, True])
def test_cb_recompute_preemption_preserves_output(use_async_batching: bool) -> None:
    # The same over-subscribed cache (where 'none' raises) forces soft-reset preemption. Recompute
    # re-prefills the folded tokens, which for this token stream reproduces the unpreempted output.
    # In pipelined mode this also exercises that preemption only happens with no batch in flight, so
    # the victim's folded state includes every sampled token.
    _, reference = _cb_generate(_CB_PROMPTS, 8, num_blocks=64, kv_pressure_mode="none")

    engine, recomputed = _cb_generate(
        _CB_PROMPTS, 8, kv_pressure_mode="recompute", use_async_batching=use_async_batching
    )

    assert recomputed == reference
    assert engine.cache.get_num_free_blocks() == engine.cache.num_blocks  # all blocks released at the end
    assert engine.offloading_manager.num_recompute_preemptions > 0
    assert engine.offloading_manager.num_offload_preemptions == 0


@pytest.mark.parametrize("use_async_batching", [False, True])
def test_cb_offload_preemption_preserves_output(use_async_batching: bool) -> None:
    # Offload swaps the victim's KV to the (unpinned, CPU) pool and copies it back on restore, so the
    # output matches the unpreempted reference and the swap counters fire.
    _, reference = _cb_generate(_CB_PROMPTS, 8, num_blocks=64, kv_pressure_mode="none")

    engine, offloaded = _cb_generate(
        _CB_PROMPTS, 8, kv_pressure_mode="offload", cpu_offload_space=0.001, use_async_batching=use_async_batching
    )

    assert offloaded == reference
    assert engine.cache.get_num_free_blocks() == engine.cache.num_blocks
    assert engine.offloading_manager.num_offload_preemptions > 0
    assert engine.offloading_manager.num_restores > 0


def test_pipelined_loop_matches_serialized_outputs() -> None:
    # Mixed prompt lengths (forcing chunked prefill at max_num_batched_tokens=4), per-request length
    # limits, and EOS finishes: the pipelined loop must produce token-identical outputs. EOS requests
    # exercise the lagged finish (one discarded in-flight token); length-capped requests exercise the
    # will-finish scheduling skip.
    prompts = [[1, 2, 3, 4, 5, 6], [7], [2, 4], [9, 1, 3]]
    max_new_tokens = [5, 3, 8, 1]

    serialized_engine, serialized = _cb_generate(
        prompts, max_new_tokens, eos_token_id=5, num_blocks=64, use_async_batching=False
    )
    pipelined_engine, pipelined = _cb_generate(
        prompts, max_new_tokens, eos_token_id=5, num_blocks=64, use_async_batching=True
    )

    assert pipelined == serialized
    assert [tokens[-1] for tokens in pipelined[2:]] == [5, 4]  # EOS finish and length-capped finish
    assert pipelined_engine.cache.get_num_free_blocks() == pipelined_engine.cache.num_blocks
    # Only the EOS request pays one lagged (discarded) decode token; length-capped requests are not
    # re-scheduled once their in-flight token meets the limit, so they compute no extra work.
    assert pipelined_engine.last_stats.decode_tokens == serialized_engine.last_stats.decode_tokens + 1


def test_cb_reserve_admits_a_max_length_prompt_without_over_committing() -> None:
    # A prompt of max_model_len - 1 with a single generated token grows to exactly max_model_len tokens,
    # the block-boundary corner where a rounding-up allocator would over-allocate. The exact-ceil
    # allocator holds ceil(total / block_size) blocks here, so on a cache sized to the reserved peak
    # reserve must admit and complete it rather than raise its over-commit error.
    max_model_len, block_size = 16, 4
    reserved_peak = -(-max_model_len // block_size)  # ceil(16 / 4) = 4

    engine, outputs = _cb_generate(
        [[1] * (max_model_len - 1)],
        1,
        num_blocks=reserved_peak,
        block_size=block_size,
        max_model_len=max_model_len,
        kv_pressure_mode="reserve",
    )

    assert len(outputs[0]) == 1
    assert engine.cache.get_num_free_blocks() == engine.cache.num_blocks


def test_sampled_row_gather_matches_full_logits_outputs() -> None:
    # A model exposing logits_to_keep computes prefill logits only for the sampled rows; the tokens
    # must match the full-logits path exactly. max_num_batched_tokens=4 forces chunked prefill, so a
    # chunk with no sampled row (empty gather) is exercised too.
    def run(model_cls: type) -> tuple[ContinuousBatchingEngine, list[list[int]]]:
        cb_config = ContinuousBatchingConfig(
            num_blocks=8,
            block_size=4,
            max_num_batched_tokens=4,
            max_model_len=16,
            use_cuda_graph=False,
        )
        engine = ContinuousBatchingEngine.from_model(model_cls(_config()), cb_config, dtype=torch.float32)
        outputs = engine.generate_batch(
            input_ids=[[1, 2, 3, 4, 5, 6], [5]], max_new_tokens=3, eos_token_id=None, warmup=False
        )
        return engine, [output.generated_tokens for output in outputs]

    full_engine, full_tokens = run(TokenIncrementModel)
    gather_engine, gather_tokens = run(GatherAwareTokenIncrementModel)

    assert not full_engine.runner.gather_sampled_rows
    assert gather_engine.runner.gather_sampled_rows
    assert gather_tokens == full_tokens == [[7, 8, 9], [6, 7, 8]]
