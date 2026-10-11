"""Compare oracle certificates against an independent labeled-state exhaustive DP."""

import random
from functools import cache

import pytest

from looped_cdb.benchmarks.feedback import CostTable, RecurrenceTrace, simulate_static
from looped_cdb.benchmarks.feedback_oracle import solve_oracle


def exhaustive_opt(chains, costs):
    # Independent reference retains sequence IDs and token positions. In
    # particular, it does not call the production transition or canonicalizer.
    @cache
    def visit(positions, remaining):
        live = [i for i, pos in enumerate(positions) if pos < len(chains[i])]
        if not live:
            return 0.0
        core = [i for i in live if remaining[i] > 0]
        pending = [i for i in live if remaining[i] == 0]
        choices = []
        if core:
            updated = list(remaining)
            for i in core:
                updated[i] -= 1
            choices.append(costs.core_ms[len(core)] + visit(positions, tuple(updated)))
        if pending:
            pos = list(positions)
            updated = list(remaining)
            successors = 0
            for i in pending:
                pos[i] += 1
                if pos[i] < len(chains[i]):
                    updated[i] = chains[i][pos[i]]
                    successors += 1
            choices.append(
                costs.coda_ms[len(pending)] + costs.prelude_ms[successors] + visit(tuple(pos), tuple(updated))
            )
        return min(choices)

    return costs.prelude_ms[len(chains)] + visit(tuple(0 for _ in chains), tuple(chain[0] for chain in chains))


def test_oracle_matches_exhaustive_dp_on_varied_costs_and_feedback_chains():
    rng = random.Random(91)
    for _ in range(40):
        cap = rng.randint(1, 4)
        chains = tuple(tuple(rng.randint(1, 4) for _ in range(rng.randint(1, 3))) for _ in range(cap))
        costs = CostTable(cap, *((0.0, *(rng.uniform(0.1, 5) for _ in range(cap))) for _ in range(3)))
        result = solve_oracle(RecurrenceTrace(chains), costs, time_limit_s=10)
        expected = exhaustive_opt(chains, costs)
        assert result.exact, chains
        assert result.lower_bound_ms == result.upper_bound_ms
        assert result.schedule.runtime_ms == pytest.approx(expected)
        assert result.schedule.core_visits == sum(map(sum, chains))


def test_interrupt_returns_valid_bounds_and_a_complete_schedule():
    trace = RecurrenceTrace(((1, 3), (2, 1), (4, 2)))
    costs = CostTable(3, (0, 2, 2, 2), (0, 3, 3, 3), (0, 1, 1, 1))
    exact = exhaustive_opt(trace.chains, costs)
    for budget in (0, 1, 3):
        result = solve_oracle(trace, costs, max_states=budget)
        assert result.lower_bound_ms <= exact + 1e-9 <= result.upper_bound_ms + 1e-9
        assert result.expanded_states <= budget
        assert result.schedule.coda_tokens == trace.token_count
        assert result.upper_bound_ms <= min(simulate_static(trace, costs, k).runtime_ms for k in (1, 2, 3))
    assert not solve_oracle(trace, costs, max_states=0).exact
    assert solve_oracle(trace, costs, time_limit_s=0).stop_reason == "time_limit"


def test_identical_64_slot_chains_solve_without_exponential_label_permutations():
    trace = RecurrenceTrace(((2, 1, 3),) * 64)
    costs = CostTable(64, *((0.0, *[float(batch + 1) for batch in range(1, 65)]) for _ in range(3)))
    result = solve_oracle(trace, costs, max_states=100)
    assert result.exact
    assert result.expanded_states < 20
    assert result.schedule.runtime_ms == simulate_static(trace, costs, 64).runtime_ms


def test_long_single_chain_does_not_hit_python_recursion_limit():
    trace = RecurrenceTrace(((1,) * 600,))
    costs = CostTable(1, (0, 1), (0, 1), (0, 1))
    result = solve_oracle(trace, costs, max_states=2000)
    assert result.exact
    assert result.upper_bound_ms == 1800


def test_chains_with_same_b_q_but_different_future_are_not_merged():
    costs = CostTable(2, (0, 2, 2), (0, 2, 2), (0, 1, 1))
    easy = solve_oracle(RecurrenceTrace(((1, 1), (3, 1))), costs)
    hard = solve_oracle(RecurrenceTrace(((1, 9), (3, 9))), costs)
    assert easy.exact and hard.exact
    assert easy.upper_bound_ms < hard.upper_bound_ms


def test_dynamic_schedule_can_beat_every_static_threshold():
    trace = RecurrenceTrace(((5, 4, 2), (1, 2, 2), (4, 1, 2), (3, 6, 2)))
    costs = CostTable(4, (0, 2, 2, 2, 2), (0, 1, 1, 1, 1), (0, 1, 1, 1, 1))
    result = solve_oracle(trace, costs)
    best_static = min(simulate_static(trace, costs, k).runtime_ms for k in range(1, 5))
    assert result.exact
    assert result.upper_bound_ms == exhaustive_opt(trace.chains, costs) == 30
    assert best_static == 32
