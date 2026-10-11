"""Bounded offline shortest-path search on scheduler-dependent feedback states.

The oracle chooses CORE or FLUSH-ALL; it never optimizes a fixed arrival trace.
A* uses unavoidable remaining stage work as an admissible lower bound, and the
best static threshold as an incumbent. An interrupted search reports bounds,
not an exact OPT. There is no recursion-depth limit for long chains.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass
from itertools import count

from .feedback import (
    Action,
    CostTable,
    RecurrenceTrace,
    ScheduleResult,
    State,
    evaluate_actions,
    occupancy,
    simulate_static,
    transition,
)


@dataclass(frozen=True)
class OracleResult:
    schedule: ScheduleResult
    lower_bound_ms: float
    upper_bound_ms: float
    exact: bool
    expanded_states: int
    stop_reason: str
    search_seconds: float


def unit_cost_bounds(costs: CostTable) -> tuple[float, float, float]:
    """Use minima over all measured widths, without assuming economies of scale."""
    return tuple(
        min(curve[batch] / batch for batch in range(1, costs.resident_cap + 1))
        for curve in (costs.core_ms, costs.coda_ms, costs.prelude_ms)
    )


def remaining_work_bound(state: State, unit_costs: tuple[float, float, float]) -> float:
    """Lower-bound unavoidable remaining visits, coda tokens and successor preludes."""
    core, coda, prelude = unit_costs
    result = math.fsum(
        (
            core * sum(sum(seq) for seq in state),
            coda * sum(map(len, state)),
            prelude * sum(len(seq) - 1 for seq in state),
        )
    )
    # Keep rounding of the ratio/products from turning an admissible bound into
    # a slightly optimistic prune. This changes neither the feasible schedules
    # nor their measured costs.
    return max(0.0, result - 1e-12 * max(1.0, result))


def solve_oracle(
    trace: RecurrenceTrace,
    costs: CostTable,
    *,
    thresholds: tuple[int, ...] | None = None,
    max_states: int = 100_000,
    time_limit_s: float = 30,
) -> OracleResult:
    """Return exact OPT if certified, otherwise a feasible schedule and [L,U].

    State canonicalization exchanges only homogeneous sequence labels, retaining
    each pending token's entire successor chain. It does not merge states by b,q.
    ``max_states`` bounds expansions (and hence at most two generated edges per
    expansion). Both budgets include seed schedule construction in wall time.
    """
    if type(max_states) is not int or max_states < 0:
        raise ValueError("max_states must be a nonnegative integer")
    if not math.isfinite(time_limit_s) or time_limit_s < 0:
        raise ValueError("time_limit_s must be finite and nonnegative")
    candidates = tuple(range(1, costs.resident_cap + 1)) if thresholds is None else thresholds
    if not candidates:
        raise ValueError("at least one static threshold is required")
    started = time.perf_counter()
    incumbent = min((simulate_static(trace, costs, k) for k in candidates), key=lambda result: result.runtime_ms)
    upper = incumbent.runtime_ms
    root = trace.initial_state
    initial = costs.prelude_ms[len(root)]
    units = unit_cost_bounds(costs)
    serial = count()
    # Each entry holds its own immutable action path, so a cheaper revisit of an
    # ancestor cannot silently mutate a previously saved incumbent's schedule.
    frontier = [(initial + remaining_work_bound(root, units), next(serial), initial, root, ())]
    best_cost = {root: initial}
    expanded = 0
    reason = "optimal"

    def discard_stale() -> None:
        while frontier and frontier[0][2] != best_cost.get(frontier[0][3]):
            heapq.heappop(frontier)

    while frontier:
        discard_stale()
        if not frontier or frontier[0][0] >= upper:
            break
        if expanded >= max_states:
            reason = "state_limit"
            break
        if time.perf_counter() - started >= time_limit_s:
            reason = "time_limit"
            break
        _, _, spent, state, path = heapq.heappop(frontier)
        expanded += 1
        _, b, q = occupancy(state)
        actions: tuple[Action, ...] = tuple(action for action, valid in (("core", b > 0), ("boundary", q > 0)) if valid)
        for action in actions:
            successor, duration, _ = transition(state, action, costs)
            candidate_cost = spent + duration
            if candidate_cost >= upper:
                continue
            candidate_path = (*path, action)
            if not successor:
                incumbent = evaluate_actions(trace, costs, candidate_path)
                upper = incumbent.runtime_ms
                continue
            if candidate_cost >= best_cost.get(successor, math.inf):
                continue
            bound = candidate_cost + remaining_work_bound(successor, units)
            if bound >= upper:
                continue
            best_cost[successor] = candidate_cost
            heapq.heappush(frontier, (bound, next(serial), candidate_cost, successor, candidate_path))

    discard_stale()
    lower = min(upper, frontier[0][0]) if frontier else upper
    exact = lower >= upper
    if exact:
        reason = "optimal"
    return OracleResult(incumbent, lower, upper, exact, expanded, reason, time.perf_counter() - started)
