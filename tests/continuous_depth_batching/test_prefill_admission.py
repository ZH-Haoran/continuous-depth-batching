"""The prompt-admission rule shared by the three serving loops.

Continuous batching, continuous depth batching, and the no-refill baseline are compared against each
other, so a throughput difference between them must not come from admitting prompts at different rates.
All three route prompt admission through ``BaseServingScheduler.admit_prefill``: a prefill batch runs
only when no request is decoding, or when free KV blocks exceed the safety margin.

These tests use a non-zero safety margin. With ``safety_margin=0.0`` the rule is vacuously true whenever
one block is free, so it never fires.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from test_continuous_batching_continuous_api import TokenIncrementModel
from test_continuous_depth_batching import TokenIncrementLoopAdapter, TokenIncrementLoopModel
from test_continuous_depth_batching import _config as _cdb_model_config
from transformers import PreTrainedConfig

from looped_cdb.continuous_batching.cache import PagedAttentionCache as CBCache
from looped_cdb.continuous_batching.config import ContinuousBatchingConfig
from looped_cdb.continuous_batching.continuous_api import ContinuousBatchingEngine
from looped_cdb.continuous_batching.requests import RequestState as CBRequestState
from looped_cdb.continuous_batching.scheduler import FIFOScheduler
from looped_cdb.continuous_depth_batching import ContinuousDepthBatchingConfig, ContinuousDepthBatchingEngine
from looped_cdb.continuous_depth_batching.cache import PagedAttentionCache as CDBCache
from looped_cdb.continuous_depth_batching.continuous_api import CacheFullError
from looped_cdb.continuous_depth_batching.requests import DepthWorkItem
from looped_cdb.continuous_depth_batching.requests import RequestState as CDBRequestState
from looped_cdb.continuous_depth_batching.scheduler import CDBScheduler
from looped_cdb.request_status import RequestStatus
from looped_cdb.utils import resolve_min_free_slots

# Eight blocks of four tokens, with a margin of half the pool: prefill stops once four blocks are used.
NUM_BLOCKS = 8
BLOCK_SIZE = 4
SAFETY_MARGIN = 0.5
MAX_MODEL_LEN = 16


def _cb_model_config(**overrides: Any) -> PreTrainedConfig:
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


def _cb_scheduler() -> FIFOScheduler:
    cb_config = ContinuousBatchingConfig(
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        max_num_batched_tokens=8,
        max_model_len=MAX_MODEL_LEN,
        safety_margin=SAFETY_MARGIN,
    )
    cache = CBCache(
        config=_cb_model_config(),
        continuous_batching_config=cb_config,
        device="cpu",
        dtype=torch.float32,
    )
    return FIFOScheduler(cache, safety_margin=SAFETY_MARGIN)


def _cdb_scheduler(**overrides: Any) -> CDBScheduler:
    cdb_config = ContinuousDepthBatchingConfig(
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        max_num_batched_tokens=8,
        max_model_len=MAX_MODEL_LEN,
        max_recurrent_steps=3,
    )
    cache = CDBCache(
        config=_cdb_model_config(),
        continuous_batching_config=cdb_config,
        device="cpu",
        dtype=torch.float32,
    )
    kwargs: dict[str, Any] = {"max_num_seqs": 8, "safety_margin": SAFETY_MARGIN}
    kwargs.update(overrides)
    return CDBScheduler(cache=cache, max_recurrent_steps=3, **kwargs)


def _admit_decoder(scheduler: FIFOScheduler | CDBScheduler, request_id: str, held_blocks: int) -> Any:
    """Place an active, decoding request holding ``held_blocks`` blocks of KV.

    The depth scheduler additionally gets the request's in-flight token queued as recurrent work, since
    a ``DECODING`` request there always holds its single token somewhere in the depth pipeline.
    """

    state_cls = CBRequestState if isinstance(scheduler, FIFOScheduler) else CDBRequestState
    state = state_cls(request_id=request_id, initial_tokens=[1, 2], max_new_tokens=8)
    state.status = RequestStatus.DECODING
    state.remaining_prefill_tokens = []
    state.tokens_to_process = [3]
    state.position_offset = 2
    allocated = scheduler.cache.allocate_blocks(held_blocks, request_id, 0)
    assert allocated == held_blocks
    state.allocated_blocks = allocated
    scheduler.active_requests[request_id] = state
    if isinstance(scheduler, CDBScheduler):
        scheduler.enqueue_depth(DepthWorkItem(state=state, token_id=3, token_position=2, recurrent_step=0))
    return state


def _schedule_once(scheduler: FIFOScheduler | CDBScheduler) -> str:
    """Run one scheduler tick and report which kind of work it placed."""

    if isinstance(scheduler, FIFOScheduler):
        scheduled = scheduler.schedule_batch(
            token_budget=scheduler.cache.max_num_batched_tokens,
            cache_budget=scheduler.cache.num_pages,
        )
        assert scheduled.requests, "the full-depth scheduler placed no work"
        is_decode = all(
            future.query_length == 1 and future.state.status == RequestStatus.DECODING for future in scheduled.requests
        )
        return "decode" if is_decode else "prefill"

    batch = scheduler.schedule_next(
        token_budget=scheduler.cache.max_num_batched_tokens,
        cache_budget=scheduler.cache.num_pages,
    )
    assert batch is not None, "the depth scheduler placed no work"
    return batch.kind


@pytest.fixture(params=["cb", "cdb"])
def scheduler(request: pytest.FixtureRequest) -> FIFOScheduler | CDBScheduler:
    return _cb_scheduler() if request.param == "cb" else _cdb_scheduler()


def _add_waiting_prompt(scheduler: FIFOScheduler | CDBScheduler, request_id: str = "waiting") -> Any:
    state_cls = CBRequestState if isinstance(scheduler, FIFOScheduler) else CDBRequestState
    state = state_cls(request_id=request_id, initial_tokens=[7, 8], max_new_tokens=4)
    scheduler.add_waiting_request(state)
    return state


def test_prefill_is_withheld_below_the_margin_while_a_request_decodes(
    scheduler: FIFOScheduler | CDBScheduler,
) -> None:
    """Below the margin, a decoding request runs and the waiting prompt stays queued."""

    decoder = _admit_decoder(scheduler, "decoder", held_blocks=5)
    waiting = _add_waiting_prompt(scheduler)
    assert scheduler.cache.get_num_free_blocks() == 3
    assert not scheduler.has_prefill_headroom()

    kind = _schedule_once(scheduler)

    assert kind in {"decode", "recurrent"}
    assert waiting.status == RequestStatus.PENDING
    assert "waiting" in scheduler.waiting_requests
    assert decoder.request_id in scheduler.active_requests


def test_prefill_is_admitted_below_the_margin_when_nothing_is_decoding(
    scheduler: FIFOScheduler | CDBScheduler,
) -> None:
    """With no decode work the margin is ignored, or a fragmented cache could never drain itself."""

    scheduler.cache.allocate_blocks(5, "filler", 0)
    waiting = _add_waiting_prompt(scheduler)
    assert scheduler.cache.get_num_free_blocks() == 3
    assert not scheduler.has_prefill_headroom()

    kind = _schedule_once(scheduler)

    assert kind == "prefill"
    assert waiting.status == RequestStatus.DECODING
    assert "waiting" in scheduler.active_requests


def test_prefill_is_admitted_above_the_margin_while_a_request_decodes(
    scheduler: FIFOScheduler | CDBScheduler,
) -> None:
    """With headroom the loops still prefer prefill, so the gate is what withholds it, not the tick order."""

    _admit_decoder(scheduler, "decoder", held_blocks=1)
    waiting = _add_waiting_prompt(scheduler)
    assert scheduler.has_prefill_headroom()

    kind = _schedule_once(scheduler)

    assert kind == "prefill"
    assert waiting.status == RequestStatus.DECODING


def test_min_free_slots_resolves_against_the_cap() -> None:
    """Unset waits for an eighth of the cap (at least one slot); a threshold above the cap is clamped to it."""

    # The fixture caps residency at eight requests.
    assert _cdb_scheduler().min_free_slots == 1
    assert _cdb_scheduler(min_free_slots=64).min_free_slots == 8
    assert resolve_min_free_slots(None, 64) == 8 and resolve_min_free_slots(3, 64) == 3


@pytest.mark.parametrize("overrides", [{"min_free_slots": 0}, {"max_num_seqs": 0}])
def test_admission_settings_are_validated(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _cdb_scheduler(**overrides)


def test_the_minimum_coda_batch_is_resolved_against_the_clamped_cap() -> None:
    """The coda queue is fed from the resident set, so its minimum cannot outrun the resolved cap.

    The config checks the minimum against the cap it was given, but the token budget can clamp that
    cap once the cache is sized, which would otherwise leave a minimum no coda queue could reach.
    """

    scheduler = _cdb_scheduler(max_num_seqs=64, min_coda_batch_size=64)

    assert scheduler.max_num_seqs == 8
    assert scheduler.min_coda_batch_size == 8


def test_recurrent_launch_width_is_capped_but_never_gates_the_bucket() -> None:
    """``max_num_seqs`` bounds a recurrent launch; it is not a queue-depth threshold.

    A depth queue far shallower than the cap still runs, rather than deferring to prefill until it
    fills, so no saturation-batch policy separates the depth scheduler from the full-depth baseline.
    """

    scheduler = _cdb_scheduler()
    assert scheduler.max_num_seqs == 8
    _admit_decoder(scheduler, "decoder", held_blocks=5)
    _add_waiting_prompt(scheduler)

    batch = scheduler.schedule_next(token_budget=8, cache_budget=scheduler.cache.num_pages)

    assert batch is not None
    assert batch.kind == "recurrent"
    assert batch.depth_items is not None
    assert len(batch.depth_items) == 1


# --------------------------------------------------------------------- engine-level


def _cb_engine(num_blocks: int, *, max_num_seqs: int = 256) -> ContinuousBatchingEngine:
    cb_config = ContinuousBatchingConfig(
        num_blocks=num_blocks,
        max_num_seqs=max_num_seqs,
        block_size=BLOCK_SIZE,
        max_num_batched_tokens=8,
        max_model_len=MAX_MODEL_LEN,
        use_async_batching=False,
        use_cuda_graph=False,
        safety_margin=0.2,
        kv_pressure_mode="recompute",
    )
    return ContinuousBatchingEngine.from_model(TokenIncrementModel(_cb_model_config()), cb_config, dtype=torch.float32)


def _cdb_engine(num_blocks: int, *, refill: bool) -> ContinuousDepthBatchingEngine:
    model = TokenIncrementLoopModel(_cdb_model_config())
    cdb_config = ContinuousDepthBatchingConfig(
        num_blocks=num_blocks,
        block_size=BLOCK_SIZE,
        max_num_batched_tokens=8,
        max_model_len=MAX_MODEL_LEN,
        use_async_batching=False,
        use_cuda_graph=False,
        safety_margin=0.2,
        kv_pressure_mode="recompute",
        max_recurrent_steps=3,
        refill=refill,
    )
    return ContinuousDepthBatchingEngine.from_model(
        model, cdb_config, dtype=torch.float32, model_adapter=TokenIncrementLoopAdapter(model)
    )


# Sixteen requests against twelve blocks of four tokens: every request grows to eight tokens (two
# blocks), so the pool holds at most six at once and admission has to throttle rather than preempt.
_PRESSURE_BLOCKS = 12
_PRESSURE_PROMPTS = [[1, 2]] * 16
_PRESSURE_NEW_TOKENS = 6


def test_full_depth_engine_serves_the_pressure_workload_without_preemption() -> None:
    """The reference point for the two depth engines: admission alone keeps the cache from filling."""

    engine = _cb_engine(_PRESSURE_BLOCKS)

    outputs = engine.generate_batch(
        input_ids=[list(prompt) for prompt in _PRESSURE_PROMPTS],
        max_new_tokens=_PRESSURE_NEW_TOKENS,
        eos_token_id=None,
        warmup=False,
    )

    assert len(outputs) == len(_PRESSURE_PROMPTS)
    assert all(len(output.generated_tokens) == _PRESSURE_NEW_TOKENS for output in outputs)
    assert engine.offloading_manager.num_preemptions == 0


@pytest.mark.parametrize("refill", [True, False], ids=["refill", "no_refill"])
def test_depth_engines_serve_the_pressure_workload_without_preemption(refill: bool) -> None:
    """Both depth loops throttle admission exactly as the full-depth engine does.

    Without the shared gate the refill loop admits a prompt on every tick where the depth queue is not
    saturated, drives the pool to zero free blocks, and soft-resets requests that then have to recompute.
    """

    engine = _cdb_engine(_PRESSURE_BLOCKS, refill=refill)

    outputs = engine.generate_batch(
        input_ids=[list(prompt) for prompt in _PRESSURE_PROMPTS],
        max_new_tokens=_PRESSURE_NEW_TOKENS,
        eos_token_id=None,
        warmup=False,
    )

    assert len(outputs) == len(_PRESSURE_PROMPTS)
    assert all(len(output.generated_tokens) == _PRESSURE_NEW_TOKENS for output in outputs)
    assert engine.offloading_manager.num_preemptions == 0


@pytest.mark.parametrize("refill", [True, False], ids=["refill", "no_refill"])
@pytest.mark.parametrize(
    "kv_pressure_mode,cpu_offload_space",
    [("recompute", None), ("offload", 0.05)],
    ids=["recompute", "offload"],
)
def test_depth_engines_serve_an_oversubscribed_cache_under_preemption(
    refill: bool, kv_pressure_mode: str, cpu_offload_space: float | None
) -> None:
    """Every request completes even when the cache is too small to hold the batch.

    The margin is disabled so admission cannot throttle, which forces the preemption path itself. Under
    ``offload`` a victim is swapped to the CPU pool and can only come back through the engine's restore
    step, so a loop that preempts without restoring strands it: the batch drains and the run dies with a
    full cache rather than finishing.
    """

    model = TokenIncrementLoopModel(_cdb_model_config())
    cdb_config = ContinuousDepthBatchingConfig(
        num_blocks=6,
        block_size=BLOCK_SIZE,
        max_num_batched_tokens=64,
        max_model_len=32,
        use_async_batching=False,
        use_cuda_graph=False,
        safety_margin=0.0,
        kv_pressure_mode=kv_pressure_mode,
        cpu_offload_space=cpu_offload_space,
        max_recurrent_steps=3,
        refill=refill,
    )
    engine = ContinuousDepthBatchingEngine.from_model(
        model, cdb_config, dtype=torch.float32, model_adapter=TokenIncrementLoopAdapter(model)
    )

    outputs = engine.generate_batch(
        input_ids=[[1, 2, 3, 4]] * 6,
        max_new_tokens=6,
        eos_token_id=None,
        warmup=False,
    )

    assert len(outputs) == 6
    assert all(len(output.generated_tokens) == 6 for output in outputs)
    assert engine.offloading_manager.num_preemptions > 0, "the workload did not exercise preemption"


@pytest.mark.parametrize("refill", [True, False], ids=["refill", "no_refill"])
@pytest.mark.parametrize(
    "kv_pressure_mode,cpu_offload_space",
    [("recompute", None), ("offload", 0.05)],
    ids=["recompute", "offload"],
)
def test_preemption_does_not_change_generated_tokens(
    refill: bool, kv_pressure_mode: str, cpu_offload_space: float | None
) -> None:
    """A preempted request resumes to the same tokens it would have produced uncontended."""

    prompts = [[1, 2, 3, 4]] * 6

    def run(num_blocks: int, use_async_batching: bool = False, **pressure: Any) -> list[list[int]]:
        model = TokenIncrementLoopModel(_cdb_model_config())
        cdb_config = ContinuousDepthBatchingConfig(
            num_blocks=num_blocks,
            block_size=BLOCK_SIZE,
            max_num_batched_tokens=64,
            max_model_len=32,
            use_async_batching=use_async_batching,
            use_cuda_graph=False,
            safety_margin=0.0,
            max_recurrent_steps=3,
            refill=refill,
            **pressure,
        )
        engine = ContinuousDepthBatchingEngine.from_model(
            model, cdb_config, dtype=torch.float32, model_adapter=TokenIncrementLoopAdapter(model)
        )
        outputs = engine.generate_batch(
            input_ids=[list(prompt) for prompt in prompts],
            max_new_tokens=6,
            eos_token_id=None,
            warmup=False,
        )
        return [output.generated_tokens for output in outputs]

    uncontended = run(64, kv_pressure_mode="none")
    contended = run(6, kv_pressure_mode=kv_pressure_mode, cpu_offload_space=cpu_offload_space)
    # The async loop drains its in-flight coda results before any preemption, so eager re-entry
    # must not change what a preempted request resumes to.
    contended_async = run(
        6, use_async_batching=True, kv_pressure_mode=kv_pressure_mode, cpu_offload_space=cpu_offload_space
    )

    assert contended == uncontended
    assert contended_async == uncontended


@pytest.mark.parametrize("refill", [True, False], ids=["refill", "no_refill"])
def test_depth_engine_raises_when_a_prompt_can_never_fit(refill: bool) -> None:
    """A prompt larger than the whole pool ends the run instead of spinning on an unplaceable tick."""

    engine = _cdb_engine(num_blocks=3, refill=refill)

    with pytest.raises(CacheFullError):
        engine.generate_batch(
            input_ids=[list(range(1, 15))],
            max_new_tokens=1,
            eos_token_id=None,
            warmup=False,
        )
