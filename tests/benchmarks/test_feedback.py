"""Work conservation, successor release and resident-budget behavior."""

import pytest

from looped_cdb.benchmarks.feedback import CostTable, RecurrenceTrace, generate_trace, simulate_static, transition


def costs(cap=2):
    return CostTable(cap, (0, *[1.0] * cap), (0, *[2.0] * cap), (0, *[3.0] * cap))


def test_successor_requires_boundary_and_exit_keeps_resident_slot():
    state = RecurrenceTrace(((1, 2), (3,))).initial_state
    state, _, _ = transition(state, "core", costs())
    assert state == ((0, 2), (2,))
    state, _, _ = transition(state, "core", costs())
    assert state == ((0, 2), (1,))  # pending sequence's successor has not advanced
    state, duration, released = transition(state, "boundary", costs())
    assert state == ((1,), (2,))
    assert duration == 5 and released == 1


def test_terminal_coda_does_not_pay_prelude():
    result = simulate_static(RecurrenceTrace(((1,),)), costs(1), 1)
    assert result.runtime_ms == 3 + 1 + 2
    assert result.steps[-1].successors == 0


@pytest.mark.parametrize("threshold", [1, 2, 3, 4])
def test_static_schedules_conserve_work_and_drain_below_threshold(threshold):
    trace = RecurrenceTrace(((1, 2), (4,), (2, 1)))
    result = simulate_static(trace, costs(4), threshold)
    assert result.core_visits == trace.core_visits
    assert result.coda_tokens == result.prelude_tokens == trace.token_count
    assert all(step.resident <= 4 for step in result.steps)
    assert result.core_hole_rounds == result.pending_coda_rounds + result.terminal_hole_rounds


def test_fixed_k_changes_successor_release_times():
    trace = RecurrenceTrace(((1, 1), (4, 1)))
    immediate = simulate_static(trace, costs(), 1)
    delayed = simulate_static(trace, costs(), 2)
    assert immediate.steps[2].action == "boundary"
    assert delayed.steps[2].action == "core"
    assert immediate.steps[2].successors == 1


def test_cost_schema_rejects_missing_batch_instead_of_interpolation():
    data = costs().to_dict()
    assert CostTable.from_dict(data) == costs()
    data["points"].pop()
    with pytest.raises(ValueError, match="every effective batch"):
        CostTable.from_dict(data)


def test_trace_generation_is_reproducible_and_uses_loop_counts():
    for distribution in ("geometric", "two_point"):
        first = generate_trace(seed=3, distribution=distribution)
        assert first == generate_trace(seed=3, distribution=distribution)
        assert first != generate_trace(seed=4, distribution=distribution)
        assert all(depth >= 1 for chain in first.chains for depth in chain)
    assert generate_trace(sequences=1, tokens_per_sequence=3, mean_depth=1).chains == ((1, 1, 1),)


@pytest.mark.parametrize("chains", [(), ((),), ((0,),), ((-1,),), ((1.5,),), ((True,),)])
def test_invalid_chains_are_rejected(chains):
    with pytest.raises(ValueError):
        RecurrenceTrace(chains)


def test_costs_reject_nonfinite_values():
    with pytest.raises(ValueError, match="finite"):
        CostTable(1, (0, float("nan")), (0, 1), (0, 1))
