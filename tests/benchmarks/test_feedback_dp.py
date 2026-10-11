"""Cross-check macro DP against independent stepwise enumeration and A*."""

import random
import sys

import pytest
from test_feedback_oracle import exhaustive_opt

from looped_cdb.benchmarks.feedback import CostTable, RecurrenceTrace, evaluate_actions, simulate_static, transition
from looped_cdb.benchmarks.feedback_dp import _segments, solve_dp_oracle
from looped_cdb.benchmarks.feedback_oracle import solve_oracle


def test_dp_matches_independent_exhaustive_opt_and_completed_astar():
    rng = random.Random(91)
    for _ in range(80):
        cap = rng.randint(1, 4)
        chains = tuple(tuple(rng.randint(1, 4) for _ in range(rng.randint(1, 3))) for _ in range(cap))
        # Arbitrary, nonmonotone costs: do not assume affine or increasing curves.
        costs = CostTable(cap, *((0.0, *(rng.uniform(0.1, 5) for _ in range(cap))) for _ in range(3)))
        trace = RecurrenceTrace(chains)
        dp = solve_dp_oracle(trace, costs)
        astar = solve_oracle(trace, costs, time_limit_s=10)
        assert astar.exact
        assert dp.runtime_ms == pytest.approx(exhaustive_opt(chains, costs))
        assert dp.runtime_ms == pytest.approx(astar.upper_bound_ms)
        actions = tuple(action for h in dp.core_segments for action in (*("core",) * h, "boundary"))
        assert evaluate_actions(trace, costs, actions) == dp.schedule
        assert dp.schedule.core_visits == trace.core_visits
        assert dp.schedule.coda_tokens == dp.schedule.prelude_tokens == trace.token_count


def test_macro_edges_match_stepwise_transitions_including_between_exit_rounds():
    trace = RecurrenceTrace(((1, 2), (4,), (6, 3)))
    costs = CostTable(5, (0, 2, 5, 3, 8, 1), (0, 5, 1, 3, 2, 4), (0, 3, 2, 5, 1, 4))
    edges = list(_segments(trace.initial_state, costs))
    assert [h for h, _, _ in edges] == list(range(1, 7))
    for h, cost, child in edges:
        state, elapsed = trace.initial_state, 0
        for _ in range(h):
            state, duration, _ = transition(state, "core", costs)
            elapsed += duration
        state, duration, _ = transition(state, "boundary", costs)
        assert cost == elapsed + duration
        assert child == state
        assert all(chain[0] > 0 for chain in child)


def test_pruning_is_exact_for_zero_nonmonotone_costs_and_unused_resident_slots():
    rng = random.Random(213)
    for _ in range(120):
        cap = rng.randint(1, 5)
        chains = tuple(
            tuple(rng.randint(1, 4) for _ in range(rng.randint(1, 3))) for _ in range(rng.randint(1, min(cap, 4)))
        )
        costs = CostTable(cap, *((0, *(rng.randint(0, 6) for _ in range(cap))) for _ in range(3)))
        result = solve_dp_oracle(RecurrenceTrace(chains), costs)
        assert result.runtime_ms == exhaustive_opt(chains, costs)


def test_dynamic_fixture_returns_30_versus_best_static_32():
    trace = RecurrenceTrace(((5, 4, 2), (1, 2, 2), (4, 1, 2), (3, 6, 2)))
    costs = CostTable(4, (0, 2, 2, 2, 2), (0, 1, 1, 1, 1), (0, 1, 1, 1, 1))
    result = solve_dp_oracle(trace, costs)
    assert result.runtime_ms == exhaustive_opt(trace.chains, costs) == 30
    assert min(simulate_static(trace, costs, k).runtime_ms for k in range(1, 5)) == 32
    assert result.states_evaluated > 0
    assert result.search_seconds >= 0


def test_long_single_chain_does_not_consume_python_recursion_depth():
    length = sys.getrecursionlimit() + 100
    trace = RecurrenceTrace(((1,) * length,))
    costs = CostTable(1, (0, 1), (0, 1), (0, 1))
    result = solve_dp_oracle(trace, costs)
    assert result.runtime_ms == 3 * length
    assert result.states_evaluated == length
    assert result.core_segments == (1,) * length


def test_canonicalization_merges_permutations_but_retains_future_chains():
    costs = CostTable(3, (0, 2, 2, 2), (0, 2, 2, 2), (0, 1, 1, 1))
    easy = RecurrenceTrace(((1, 1), (3, 1)))
    forward = solve_dp_oracle(easy, costs)
    reverse = solve_dp_oracle(RecurrenceTrace(tuple(reversed(easy.chains))), costs)
    hard = solve_dp_oracle(RecurrenceTrace(((1, 9), (3, 9))), costs)
    assert forward.runtime_ms == reverse.runtime_ms
    assert forward.states_evaluated == reverse.states_evaluated
    assert forward.runtime_ms < hard.runtime_ms


def test_initial_prelude_uses_initial_sequences_and_terminal_has_no_successor():
    costs = CostTable(4, (0, 1, 2, 3, 4), (0, 2, 3, 4, 5), (0, 3, 4, 5, 6))
    result = solve_dp_oracle(RecurrenceTrace(((1,),)), costs)
    assert result.runtime_ms == 6
    assert result.schedule.steps[0].duration_ms == 3
    assert result.schedule.steps[-1].successors == 0
    assert result.core_segments == (1,)


def test_identical_64_slot_chains_and_zero_costs():
    trace = RecurrenceTrace(((2, 1, 3),) * 64)
    costs = CostTable(64, *((0.0,) * 65 for _ in range(3)))
    result = solve_dp_oracle(trace, costs)
    assert result.runtime_ms == 0
    assert 1 <= result.states_evaluated <= 3
    assert result.pruned_segments > 0


def test_trace_exceeding_resident_cap_is_rejected():
    with pytest.raises(ValueError, match="resident cap"):
        solve_dp_oracle(RecurrenceTrace(((1,), (1,))), CostTable(1, (0, 1), (0, 1), (0, 1)))
