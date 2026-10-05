"""Tests for CDB stage traces."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import torch
from test_continuous_depth_batching import TokenIncrementLoopAdapter, TokenIncrementLoopModel, _cdb_config, _config

from looped_cdb.continuous_depth_batching import ContinuousDepthBatchingEngine
from looped_cdb.continuous_depth_batching.schedule_trace import TRACE_STAGES, ScheduleTrace, TraceEvent

PROMPTS = [[1, 2], [5], [3, 4, 5], [6]]
EXIT_DEPTHS = [[0, 2, 1], [2, 0, 2], [1, 1, 0], [2, 2, 1]]
MAX_NEW_TOKENS = 4


def _traced_engine(refill: bool, *, trace: ScheduleTrace | None = None) -> ContinuousDepthBatchingEngine:
    model = TokenIncrementLoopModel(_config())
    engine = ContinuousDepthBatchingEngine.from_model(
        model,
        _cdb_config(synthetic_exit_replay=True, refill=refill, max_num_seqs=3),
        dtype=torch.float32,
        model_adapter=TokenIncrementLoopAdapter(model),
    )
    engine.schedule_trace = ScheduleTrace() if trace is None else trace
    return engine


def _generate(engine: ContinuousDepthBatchingEngine) -> list[TraceEvent]:
    engine.generate_batch(
        input_ids=PROMPTS, max_new_tokens=MAX_NEW_TOKENS, eos_token_id=None, warmup=False, exit_depths=EXIT_DEPTHS
    )
    assert engine.schedule_trace is not None
    return engine.schedule_trace.events


def _assert_consistent_with_stats(engine: ContinuousDepthBatchingEngine, events: list[TraceEvent]) -> None:
    stats = engine.last_stats
    assert [event.launch for event in events] == list(range(1, len(events) + 1))
    assert all(a.tick <= b.tick for a, b in pairwise(events))
    assert {event.stage for event in events} == set(TRACE_STAGES)
    assert events[0].stage == "prefill"
    by_stage = {stage: [event for event in events if event.stage == stage] for stage in TRACE_STAGES}
    assert sum(event.size for event in by_stage["prefill"]) == stats.prefill_requests
    assert sum(event.size for event in by_stage["prelude"]) == sum(stats.prelude_tokens.values())
    assert sum(event.size for event in by_stage["coda"]) == stats.coda_tokens
    assert sum(event.exited or 0 for event in by_stage["recurrent"]) == stats.coda_tokens
    assert sum(event.size for event in by_stage["recurrent"]) == stats.recurrent_steps
    for event in by_stage["recurrent"]:
        assert event.depths is not None and sum(event.depths.values()) == event.size
        assert event.size <= engine.cdb_config.max_num_seqs
        assert event.exited is not None and 0 <= event.exited <= event.size
    for event in by_stage["prefill"]:
        assert event.tokens is not None and event.tokens >= event.size
    final = events[-1].queues
    assert (final.waiting, final.recurrent, final.coda) == (0, 0, 0)
    assert all(
        min(event.queues.waiting, event.queues.recurrent, event.queues.coda, event.queues.active) >= 0
        for event in events
    )
    assert all(0 <= event.queues.decode <= event.queues.active for event in events)


def test_refill_trace_agrees_with_engine_counters_and_mixes_depths() -> None:
    engine = _traced_engine(refill=True)
    events = _generate(engine)

    _assert_consistent_with_stats(engine, events)
    recurrent = [event for event in events if event.stage == "recurrent"]
    assert any(len(event.depths or {}) > 1 for event in recurrent)
    exiting = [event for event in recurrent if event.exited]
    assert exiting
    sites = {event.site for event in events if event.stage == "prelude"}
    assert sites <= {"after_prefill", "after_prefill_eager", "after_coda_staged", "after_coda_eager"}
    assert "after_coda_staged" in sites or "after_coda_eager" in sites


def test_trace_marks_the_launches_inside_the_run_steady_state_window() -> None:
    engine = _traced_engine(refill=True)
    engine.scheduler.steady_warmup_requests = 1
    events = _generate(engine)

    steady = [event.launch for event in events if event.steady]
    assert steady and steady == list(range(steady[0], steady[-1] + 1))
    assert not events[0].steady and not events[-1].steady


def test_no_refill_trace_is_a_sequence_of_shrinking_waves() -> None:
    engine = _traced_engine(refill=False)
    events = _generate(engine)

    _assert_consistent_with_stats(engine, events)
    assert all(event.queues.recurrent == 0 and event.queues.coda == 0 for event in events)
    waves: dict[int, list[TraceEvent]] = {}
    for event in events:
        waves.setdefault(event.tick, []).append(event)
    decode_waves = [wave for wave in waves.values() if any(event.stage == "recurrent" for event in wave)]
    assert decode_waves
    for wave in decode_waves:
        recurrent = [event for event in wave if event.stage == "recurrent"]
        sizes = [event.size for event in recurrent]
        assert sizes == sorted(sizes, reverse=True)
        assert [list((event.depths or {}).keys()) for event in recurrent] == [[step] for step in range(len(recurrent))]
        codas = [event for event in wave if event.stage == "coda"]
        assert len(codas) == 1 and codas[0].size == sum(event.exited or 0 for event in recurrent)
        preludes = [event for event in wave if event.stage == "prelude"]
        assert len(preludes) == 1
        assert preludes[0].site == "wave_cohort"
        assert preludes[0].size == recurrent[0].size
        assert wave.index(preludes[0]) < wave.index(recurrent[0])
        assert all(event.queues.decode == preludes[0].queues.decode for event in recurrent)
        assert codas[0].queues.decode == preludes[0].queues.decode + preludes[0].size


def test_trace_round_trips_through_jsonl_and_resets_per_generation(tmp_path: Path) -> None:
    trace = ScheduleTrace()
    engine = _traced_engine(refill=True, trace=trace)
    first = list(_generate(engine))
    second = list(_generate(engine))

    assert [event.to_record() for event in second] == [event.to_record() for event in first]

    path = tmp_path / "trace.jsonl"
    trace.write_jsonl(path)
    assert ScheduleTrace.load_jsonl(path) == second
    assert len(path.read_text().splitlines()) == len(second)


def test_serving_stage_timeline_records_launches_without_full_trace() -> None:
    engine = _traced_engine(refill=True)
    engine.schedule_trace = None
    engine.generate_batch(
        input_ids=PROMPTS,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=None,
        warmup=False,
        exit_depths=EXIT_DEPTHS,
        record_queue_samples=True,
    )

    assert {stage for _, stage, _ in engine.stage_timeline_samples} == set(TRACE_STAGES)
    assert all(stamp > 0 and size > 0 for stamp, _, size in engine.stage_timeline_samples)
