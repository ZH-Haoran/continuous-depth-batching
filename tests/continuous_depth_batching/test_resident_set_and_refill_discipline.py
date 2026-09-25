"""``max_num_seqs`` is the decode batch: it caps the resident set and the launch built from it.

Prompts the engine has no near-term intent to advance are not prefilled only to be preempted later,
and a launch advances at most the whole resident set. Both serving loops share it, so neither gets an
admission budget the other lacks.

A stepped depth item re-enters at the head of the queue. The queue holds at most one item per
resident request, so ordinarily the whole queue runs and the order changes nothing; it decides which
items run first only where a launch cannot carry the queue, as under a depth-indexed KV layout that
admits one slot per launch.
"""

from __future__ import annotations

import pytest
import torch
from test_continuous_depth_batching import TokenIncrementLoopAdapter, TokenIncrementLoopModel
from test_continuous_depth_batching import _config as _cdb_model_config
from test_prefill_admission import BLOCK_SIZE, MAX_MODEL_LEN, _admit_decoder, _cb_engine, _cb_scheduler, _cdb_scheduler

from looped_cdb.continuous_batching.requests import RequestState as CBRequestState
from looped_cdb.continuous_depth_batching import ContinuousDepthBatchingConfig, ContinuousDepthBatchingEngine
from looped_cdb.continuous_depth_batching import continuous_api as cdb_continuous_api
from looped_cdb.continuous_depth_batching.requests import DepthWorkItem
from looped_cdb.continuous_depth_batching.requests import RequestState as CDBRequestState
from looped_cdb.continuous_depth_batching.scheduler import CDBScheduler
from looped_cdb.request_status import RequestStatus

# --------------------------------------------------------------------- max_num_seqs


@pytest.fixture(params=["cb", "cdb"])
def scheduler(request: pytest.FixtureRequest):
    scheduler = _cb_scheduler() if request.param == "cb" else _cdb_scheduler()
    scheduler.max_num_seqs = 2
    # Admit whenever a slot stands open, so the cap alone is under test here; the admission tests
    # below raise the threshold themselves.
    scheduler.min_free_slots = 1
    return scheduler


# The scheduler fixtures budget eight query tokens per prefill batch, so four two-token prompts fill it.
_PREFILL_TOKEN_BUDGET = 8


def _add_waiting(scheduler, request_id: str, prompt_len: int = 2):
    """Queue a prompt on whichever scheduler is under test, using that engine's request type."""

    state_cls = CDBRequestState if isinstance(scheduler, CDBScheduler) else CBRequestState
    state = state_cls(request_id=request_id, initial_tokens=[1] * prompt_len, max_new_tokens=4)
    scheduler.add_waiting_request(state)
    return state


def test_prefill_candidates_stop_at_the_resident_cap(scheduler) -> None:
    """Both loops admit only up to ``max_num_seqs`` requests, however many prompts are waiting."""

    for index in range(5):
        _add_waiting(scheduler, f"r{index}")

    candidates = scheduler.get_prefill_candidates()

    assert [state.request_id for state in candidates] == ["r0", "r1"]


def test_the_resident_cap_counts_requests_already_admitted(scheduler) -> None:
    """A request occupying a slot consumes it: the cap is on the resident set, not on batch width."""

    _admit_decoder(scheduler, "decoder", held_blocks=1)
    for index in range(5):
        _add_waiting(scheduler, f"r{index}", prompt_len=_PREFILL_TOKEN_BUDGET)

    candidates = scheduler.get_prefill_candidates()

    assert [state.request_id for state in candidates] == ["r0"]


# --------------------------------------------------------------------- admission hysteresis


def test_admission_waits_for_min_free_slots(scheduler) -> None:
    """One slot freeing must not buy a prefill launch that carries one prompt.

    Decode runs while admission waits, so waiting is free; a near-empty prefill batch is not. Once the
    threshold is met the launch takes every waiting prompt that fits the open slots.
    """

    scheduler.max_num_seqs = 8
    scheduler.min_free_slots = 4
    for index in range(6):
        _admit_decoder(scheduler, f"decoder{index}", held_blocks=0)
    for index in range(10):
        _add_waiting(scheduler, f"r{index}")

    # Two open slots, below the threshold of four.
    assert scheduler.get_prefill_candidates() == []

    scheduler.finish_request("decoder0")
    scheduler.finish_request("decoder1")

    admitted = scheduler.get_prefill_candidates()
    assert [state.request_id for state in admitted] == ["r0", "r1", "r2", "r3"]


def test_admission_holds_a_lone_prompt_until_the_threshold_is_met(scheduler) -> None:
    """The threshold counts open slots, not waiting prompts: a lone arrival waits, then launches alone."""

    scheduler.max_num_seqs = 8
    scheduler.min_free_slots = 4
    for index in range(6):
        _admit_decoder(scheduler, f"decoder{index}", held_blocks=0)
    _add_waiting(scheduler, "r0")

    assert scheduler.get_prefill_candidates() == []

    scheduler.finish_request("decoder0")
    scheduler.finish_request("decoder1")

    assert [state.request_id for state in scheduler.get_prefill_candidates()] == ["r0"]


def test_admission_opens_when_nothing_is_resident(scheduler) -> None:
    """With nothing resident every slot stands open, so even a threshold at the cap is met."""

    scheduler.max_num_seqs = 2
    scheduler.min_free_slots = 2
    for index in range(5):
        _add_waiting(scheduler, f"r{index}")

    assert [state.request_id for state in scheduler.get_prefill_candidates()] == ["r0", "r1"]


def test_an_in_progress_chunk_runs_regardless_of_the_threshold(scheduler) -> None:
    """A truncated prompt from the previous batch continues even with no slot open: its KV is allocated."""

    scheduler.max_num_seqs = 8
    scheduler.min_free_slots = 4
    _admit_decoder(scheduler, "decoder", held_blocks=0)
    chunked = _add_waiting(scheduler, "chunked")
    scheduler.waiting_requests.pop("chunked")
    scheduler.waiting_requests_order.remove("chunked")
    chunked.status = RequestStatus.PREFILLING
    chunked.remaining_prefill_tokens = [1]
    scheduler.active_requests["chunked"] = chunked
    for index in range(10):
        _add_waiting(scheduler, f"r{index}")
    # Fill the cap so no waiting prompt can join the chunk, leaving it as the only candidate.
    for index in range(6):
        _admit_decoder(scheduler, f"filler{index}", held_blocks=0)
    assert scheduler.max_num_seqs == len(scheduler.active_requests)

    assert [state.request_id for state in scheduler.get_prefill_candidates()] == ["chunked"]


def test_a_cap_wider_than_the_prefill_budget_is_capped_to_it(scheduler) -> None:
    """A decode batch writes one query token per resident request into buffers sized by the prefill
    budget, so the budget is also the batch ceiling. The scheduler resolves the cap at construction
    and every caller reads the resolved value, so a run cannot report a batch it never ran."""

    cache = scheduler.cache
    over = cache.max_num_batched_tokens + 1
    resolved = (
        CDBScheduler(cache=cache, max_recurrent_steps=3, max_num_seqs=over)
        if isinstance(scheduler, CDBScheduler)
        else type(scheduler)(cache, max_num_seqs=over)
    )

    assert resolved.max_num_seqs == cache.max_num_batched_tokens


def test_a_soft_reset_request_still_counts_as_growing() -> None:
    """Recompute folds the generation onto the prompt and zeroes ``generated_tokens``.

    So a preempted request reports nothing generated while holding every token it ever computed. The
    growing-set metric must read its true prompt, not its generated-token count, or it undercounts
    exactly the requests an oversubscribed cache is straining under.
    """

    state = CDBRequestState(request_id="r", initial_tokens=[1, 2, 3], max_new_tokens=6)
    state.status = RequestStatus.DECODING
    state.position_offset = 3
    state.update_and_check_completion(4)
    state.update_and_check_completion(5)
    assert state.prompt_len() == 3

    resumed = state.create_equivalent_initial_request()
    resumed.position_offset = 5

    assert resumed.generated_tokens == []
    assert len(resumed.initial_tokens) == 5
    assert resumed.prompt_len() == 3
    assert resumed.has_grown_past_prompt()
    # The steady-state window counts generation through the reset: the two folded tokens stay counted.
    assert resumed.total_generated_len() == 2
    resumed.generated_tokens.append(6)
    assert resumed.total_generated_len() == 3


def test_the_prefill_bucket_shows_the_prompt_the_launch_gate_is_holding() -> None:
    """A stalled scheduler asks the bucket why it cannot place work; the gate's empty list cannot say.

    ``_raise_scheduler_stall`` reads the bucket for the leading candidate. Were it to read the gated
    list, a bucket merely held back as too small would look like a bookkeeping bug.
    """

    scheduler = _cdb_scheduler()
    scheduler.max_num_seqs = 8
    scheduler.min_free_slots = 4
    # Six residents leave two open slots, below the threshold, so the gate holds the bucket.
    for index in range(6):
        _admit_decoder(scheduler, f"decoder{index}", held_blocks=0)
    for index in range(5):
        _add_waiting(scheduler, f"r{index}")

    bucket = scheduler.get_prefill_bucket()

    assert [state.request_id for state in bucket] == ["r0", "r1"]
    assert scheduler.get_prefill_candidates() == []


# --------------------------------------------------------------------- steady-state window


def _complete(scheduler, request_id: str, generated: int) -> None:
    """Admit a decoder, have it generate ``generated`` tokens, and retire it as the engines do."""

    state = _admit_decoder(scheduler, request_id, held_blocks=0)
    state.generated_tokens.extend([1] * generated)
    scheduler.record_completion(state)
    scheduler.finish_request(request_id)


def test_the_steady_window_opens_after_the_warm_up_and_closes_at_the_last_admission(scheduler) -> None:
    """Both marks are scheduler events, and the window measures exactly what happened between them.

    The warm-up completion count opens it on the next tick; an empty waiting queue closes it on the
    tick that observes it; completions and tokens outside the window are not counted.
    """

    scheduler.max_num_seqs = 8
    scheduler.steady_warmup_requests = 2
    for index in range(3):
        _add_waiting(scheduler, f"r{index}")
    scheduler.record_resident_sample()
    assert scheduler.steady_state_summary() is None

    _complete(scheduler, "d0", generated=1)
    _complete(scheduler, "d1", generated=1)
    scheduler.record_resident_sample()
    # Open, not yet closed: the queue still holds prompts.
    assert scheduler.steady_state_summary() is None

    _complete(scheduler, "d2", generated=3)
    for request_id in list(scheduler.waiting_requests):
        scheduler.waiting_requests.pop(request_id)
        scheduler.waiting_requests_order.remove(request_id)
    scheduler.record_resident_sample()

    summary = scheduler.steady_state_summary()
    assert summary is not None
    assert summary["completed_requests"] == 1
    assert summary["generated_tokens"] == 3
    assert summary["ticks"] == 1
    assert summary["window_s"] > 0
    assert summary["requests_per_second"] == pytest.approx(1 / summary["window_s"])
    assert summary["generated_tokens_per_second"] == pytest.approx(3 / summary["window_s"])

    _complete(scheduler, "d3", generated=5)
    scheduler.record_resident_sample()
    assert scheduler.steady_state_summary()["completed_requests"] == 1


def test_the_steady_window_is_one_nvtx_range_across_ticks_and_stage_ranges(monkeypatch: pytest.MonkeyPatch) -> None:
    """Push/pop ranges are a per-thread stack, so the stage range around the opening tick would pop the
    window; it is a start/end range closed by handle instead."""

    events: list[tuple[str, str]] = []

    class FakeNvtx:
        def range_push(self, name: str) -> None:
            events.append(("push", name))

        def range_pop(self) -> None:
            events.append(("pop", ""))

        def registered_range_start(self, name: str) -> int:
            events.append(("start", name))
            return 42

        def range_end(self, handle: int) -> None:
            events.append(("end", str(handle)))

    from looped_cdb.benchmarks import nvtx

    monkeypatch.setattr(nvtx, "_cuda_nvtx", lambda: FakeNvtx())
    monkeypatch.setattr(nvtx, "_enabled", True)
    scheduler = _cdb_scheduler()
    scheduler.steady_warmup_requests = 0
    _add_waiting(scheduler, "r0")
    with nvtx.range("cdb.schedule"):
        scheduler.record_resident_sample()
    with nvtx.range("cdb.recurrent"):
        pass
    scheduler.waiting_requests.pop("r0")
    scheduler.waiting_requests_order.remove("r0")
    with nvtx.range("cdb.schedule"):
        scheduler.record_resident_sample()

    assert events == [
        ("push", "cdb.schedule"),
        ("start", "benchmark.steady"),
        ("pop", ""),
        ("push", "cdb.recurrent"),
        ("pop", ""),
        ("push", "cdb.schedule"),
        ("end", "42"),
        ("pop", ""),
    ]


def test_the_steady_window_is_off_unless_a_warm_up_is_set_and_absent_for_a_run_too_short_to_hold_one(
    scheduler,
) -> None:
    """No warm-up means no window; a warm-up the run never reaches, or reaches only once nothing is
    waiting any more, means no window either."""

    _add_waiting(scheduler, "r0")
    _complete(scheduler, "d0", generated=1)
    scheduler.record_resident_sample()
    assert scheduler.steady_state_summary() is None

    scheduler.steady_warmup_requests = 5
    scheduler.record_resident_sample()
    scheduler.close_steady_window()
    assert scheduler.steady_state_summary() is None

    scheduler.waiting_requests.pop("r0")
    scheduler.waiting_requests_order.remove("r0")
    scheduler.steady_warmup_requests = 1
    scheduler.record_resident_sample()
    _complete(scheduler, "d1", generated=1)
    scheduler.record_resident_sample()
    scheduler.close_steady_window()
    assert scheduler.steady_state_summary() is None


# --------------------------------------------------------------------- refill discipline


def _depth_item(scheduler, request_id: str, recurrent_step: int = 0) -> DepthWorkItem:
    state = CDBRequestState(request_id=request_id, initial_tokens=[1, 2], max_new_tokens=4)
    scheduler.active_requests[request_id] = state
    state.status = RequestStatus.DECODING
    return DepthWorkItem(state=state, token_id=1, token_position=2, recurrent_step=recurrent_step)


def test_head_insertion_re_serves_the_same_requests_and_tail_insertion_rotates() -> None:
    """The discipline decides which requests the next launch picks up, at equal launch width.

    Head insertion hands the launch back the items it just stepped. Tail insertion sends them behind
    every other resident request, so a different set advances next.
    """

    scheduler = _cdb_scheduler()
    stepped = [_depth_item(scheduler, f"w{index}") for index in range(2)]
    parked = [_depth_item(scheduler, f"p{index}") for index in range(2)]
    for item in parked:
        scheduler.enqueue_depth(item)

    scheduler.enqueue_depth_front(stepped)
    assert [item.state.request_id for item in scheduler.ready_queue] == ["w0", "w1", "p0", "p1"]

    scheduler.ready_queue.clear()
    for item in parked:
        scheduler.enqueue_depth(item)
    for item in stepped:
        scheduler.enqueue_depth(item)
    assert [item.state.request_id for item in scheduler.ready_queue] == ["p0", "p1", "w0", "w1"]


def test_head_insertion_preserves_the_relative_order_of_a_stepped_batch() -> None:
    """``appendleft`` per item would reverse the batch, flipping the head cohort on every launch."""

    scheduler = _cdb_scheduler()
    items = [_depth_item(scheduler, f"r{index}") for index in range(4)]

    scheduler.enqueue_depth_front(items)

    assert [item.state.request_id for item in scheduler.ready_queue] == ["r0", "r1", "r2", "r3"]


# --------------------------------------------------------------------- engine level


def _engine(*, max_num_seqs: int = 2, min_free_slots: int = 4, refill: bool = True) -> ContinuousDepthBatchingEngine:
    """Build a depth engine whose decode batch is ``max_num_seqs`` requests wide."""

    model = TokenIncrementLoopModel(_cdb_model_config())
    cdb_config = ContinuousDepthBatchingConfig(
        num_blocks=64,
        block_size=BLOCK_SIZE,
        max_num_batched_tokens=8,
        max_model_len=MAX_MODEL_LEN,
        use_async_batching=False,
        use_cuda_graph=False,
        safety_margin=0.2,
        kv_pressure_mode="recompute",
        max_recurrent_steps=3,
        refill=refill,
        max_num_seqs=max_num_seqs,
        min_free_slots=min_free_slots,
    )
    return ContinuousDepthBatchingEngine.from_model(
        model, cdb_config, dtype=torch.float32, model_adapter=TokenIncrementLoopAdapter(model)
    )


_PROMPTS = [[1, 2], [1, 3], [2, 3], [3, 1], [2, 1], [3, 2], [1, 1], [2, 2]]
_NEW_TOKENS = 5


@pytest.fixture
def sample_growing_every_tick(monkeypatch: pytest.MonkeyPatch) -> None:
    """The engine samples the growing set every 64 ticks; these runs are shorter than that."""

    monkeypatch.setattr(cdb_continuous_api, "_GROWING_SAMPLE_EVERY", 1)


@pytest.mark.parametrize("max_num_seqs", [2, 8])
def test_the_decode_batch_does_not_change_generated_tokens(max_num_seqs: int) -> None:
    """The decode batch is scheduling policy: it must not touch what is generated."""

    narrow = _engine(max_num_seqs=2).generate_batch(_PROMPTS, max_new_tokens=_NEW_TOKENS, eos_token_id=None)
    wide = _engine(max_num_seqs=max_num_seqs).generate_batch(_PROMPTS, max_new_tokens=_NEW_TOKENS, eos_token_id=None)

    assert [output.generated_tokens for output in narrow] == [output.generated_tokens for output in wide]


@pytest.mark.parametrize("max_num_seqs", [2, 8])
def test_the_decode_batch_bounds_the_growing_set(max_num_seqs: int, sample_growing_every_tick: None) -> None:
    """The growing set is what the cache absorbs, and the decode batch is what bounds it.

    Eight requests against a batch of two and of eight: only the admitted set grows its KV, and a
    launch carries the whole depth queue, so nothing outside the batch is advanced.
    """

    engine = _engine(max_num_seqs=max_num_seqs)
    engine.generate_batch(_PROMPTS, max_new_tokens=_NEW_TOKENS, eos_token_id=None)

    assert 0 < engine.last_stats.max_growing_requests <= max_num_seqs


@pytest.mark.parametrize("refill", [True, False])
def test_residency_is_sampled_per_tick_before_the_tick_admits_anything(refill: bool) -> None:
    """Both loops sample the admitted set at the same point in a tick, so their means compare.

    Sampling after admission instead would report the set at its fullest and read high against the
    same cap, which is exactly the comparison the metric exists to support.
    """

    engine = _engine(max_num_seqs=2, refill=refill)
    samples: list[int] = []
    scheduler = engine.scheduler
    record = scheduler.record_resident_sample

    def record_and_capture() -> None:
        samples.append(len(scheduler.active_requests))
        record()

    scheduler.record_resident_sample = record_and_capture  # type: ignore[method-assign]
    engine.generate_batch(_PROMPTS, max_new_tokens=_NEW_TOKENS, eos_token_id=None)

    assert samples[0] == 0
    assert max(samples) <= scheduler.max_num_seqs
    assert scheduler.mean_resident_requests == pytest.approx(sum(samples) / len(samples))


def test_a_lower_threshold_holds_the_decode_batch_nearer_its_cap() -> None:
    """The threshold trades launch size for residency, and must not touch what is generated.

    At the cap (every slot open) admission refills only once the resident set has drained, and the
    decode batch collapses with it; at one slot every retirement is refilled at once.
    """

    eager = _engine(max_num_seqs=3, min_free_slots=1)
    drained = _engine(max_num_seqs=3, min_free_slots=3)
    # Staggered lengths retire the residents one at a time, which is when the two thresholds differ.
    new_tokens = [2, 5, 3, 6, 2, 4, 5, 3]

    outputs = eager.generate_batch(_PROMPTS, max_new_tokens=new_tokens, eos_token_id=None)
    drained_outputs = drained.generate_batch(_PROMPTS, max_new_tokens=new_tokens, eos_token_id=None)

    assert eager.scheduler.mean_resident_requests > drained.scheduler.mean_resident_requests
    assert [output.generated_tokens for output in outputs] == [output.generated_tokens for output in drained_outputs]


def test_the_synchronous_loop_never_suppresses_the_coda_bucket() -> None:
    """No coda batch is in flight when a recurrent batch is built, so the working set needs no backfill.

    The async loop is the one that runs a recurrent batch while coda is outstanding, which is what lets
    the head-of-line working set drift above one launch width. ``mean_coda_latency_ticks`` measures it.
    Each coda is therefore read back in the tick that staged it, so the readback pipeline reaches a
    depth of one and ``CODA_PIPELINE_DEPTH`` never withholds the bucket.
    """

    engine = _engine()
    engine.generate_batch(_PROMPTS, max_new_tokens=_NEW_TOKENS, eos_token_id=None)

    assert engine.last_stats.coda_batches > 0
    assert engine.last_stats.coda_suppressed_ticks == 0
    assert engine.last_stats.mean_coda_latency_ticks == 0.0
    assert engine.last_stats.max_pending_coda_results == 1
    assert engine.last_stats.coda_pipeline_blocked_ticks == 0


@pytest.mark.parametrize("engine_kind", ["cb", "cdb-refill", "cdb-norefill"])
def test_every_engine_reports_a_steady_window_that_leaves_out_the_fill_and_the_drain(engine_kind: str) -> None:
    """The window holds only the middle of the run, and a warm-up the run cannot reach leaves no window."""

    def build() -> ContinuousDepthBatchingEngine:
        if engine_kind == "cb":
            return _cb_engine(64, max_num_seqs=2)
        return _engine(max_num_seqs=2, min_free_slots=1, refill=engine_kind == "cdb-refill")

    new_tokens = [2, 5, 3, 6, 2, 4, 5, 3]
    engine = build()
    # Set after construction, as the benchmark runner does; ``reset`` keeps it.
    engine.scheduler.steady_warmup_requests = 4
    outputs = engine.generate_batch(_PROMPTS, max_new_tokens=new_tokens, eos_token_id=None)
    summary = engine.scheduler.steady_state_summary()

    assert summary is not None
    assert summary["warmup_requests"] == 4
    # Four completions precede the window and the two residents at the last admission follow it.
    assert 1 <= summary["completed_requests"] <= len(_PROMPTS) - 4 - 2
    assert 0 < summary["generated_tokens"] < sum(len(output.generated_tokens) for output in outputs)
    assert summary["ticks"] >= 1
    assert 0 < summary["mean_resident_requests"] <= 2
    assert summary["window_s"] > 0

    short = build()
    short.scheduler.steady_warmup_requests = len(_PROMPTS)
    short.generate_batch(_PROMPTS, max_new_tokens=new_tokens, eos_token_id=None)
    assert short.scheduler.steady_state_summary() is None
