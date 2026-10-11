"""Exact offline DP over post-Boundary states of the flush-all feedback model."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from math import fsum
from time import perf_counter

from .feedback import Action, CostTable, RecurrenceTrace, ScheduleResult, State, evaluate_actions


@dataclass(frozen=True)
class DPOracleResult:
    runtime_ms: float
    schedule: ScheduleResult
    states_evaluated: int
    search_seconds: float
    core_segments: tuple[int, ...]
    pruned_segments: int = 0


def _successor(state: State, rounds: int) -> State:
    return tuple(
        sorted(
            (chain[0] - rounds, *chain[1:]) if chain[0] > rounds else chain[1:]
            for chain in state
            if chain[0] > rounds or len(chain) > 1
        )
    )


def _segment_costs(state: State, costs: CostTable) -> tuple[int, list[float]]:
    """Compute every segment cost with one pass over the Core prefix."""
    minimum = min(chain[0] for chain in state)
    maximum = max(chain[0] for chain in state)
    exits = [0] * (maximum + 1)
    releases = [0] * (maximum + 1)
    for chain in state:
        exits[chain[0]] += 1
        releases[chain[0]] += len(chain) > 1
    prefix_core_cost = [0.0] * (maximum + 1)
    segment_costs = [0.0] * (maximum + 1)
    batch = len(state)
    pending = successors = 0
    for h in range(1, maximum + 1):
        # Before round h, the runnable heads are exactly d_i > h-1.
        prefix_core_cost[h] = prefix_core_cost[h - 1] + costs.core_ms[batch]
        batch -= exits[h]
        pending += exits[h]
        successors += releases[h]
        segment_costs[h] = prefix_core_cost[h] + costs.coda_ms[pending] + costs.prelude_ms[successors]
    return minimum, segment_costs


def _segments(state: State, costs: CostTable) -> Iterator[tuple[int, float, State]]:
    """Enumerate all integer h, including rounds between distinct exit depths."""
    minimum, segment_costs = _segment_costs(state, costs)
    for h in range(minimum, len(segment_costs)):
        yield h, segment_costs[h], _successor(state, h)


@dataclass
class _Frame:
    state: State
    edges: Iterator[tuple[int, float, State]]
    best_cost: float
    best_rounds: int
    awaiting: tuple[int, float, State] | None = None

    def consider(self, rounds: int, candidate: float) -> None:
        if candidate < self.best_cost:
            self.best_cost, self.best_rounds = candidate, rounds


def solve_dp_oracle(trace: RecurrenceTrace, costs: CostTable) -> DPOracleResult:
    """Solve the Bellman recurrence exactly, including initial Prelude.

    A Boundary flushes all pending tokens, so its successor state has only
    positive heads. The next Boundary follows h uninterrupted Core rounds,
    for every integer h in [min(heads), max(heads)]. Every edge completes at
    least one token, making this state graph acyclic. An explicit postorder
    stack memoizes V(S) without consuming Python recursion depth. A feasible
    wait-for-all suffix supplies an upper bound; unavoidable stage work and
    each sequence's serial chain supply lower bounds to prune macro edges.
    Only fully solved suffix values enter the exact memo.
    """
    started = perf_counter()
    if len(trace.chains) > costs.resident_cap:
        raise ValueError("trace exceeds the resident cap")
    initial = trace.initial_state
    # Store both the optimal suffix value and the chosen macro action.
    memo: dict[State, tuple[float, int]] = {(): (0.0, 0)}
    # Feasible suffixes are kept separate from exact Bellman values. Their
    # policies also let us reconstruct an edge pruned at equality with its UB.
    upper_memo: dict[State, tuple[float, int]] = {(): (0.0, 0)}
    unit_minima = [(0.0, 0.0, 0.0)]
    launch_minima = [(0.0, 0.0, 0.0)]
    units = launches = (float("inf"),) * 3
    curves = costs.core_ms, costs.coda_ms, costs.prelude_ms
    for n in range(1, len(initial) + 1):
        units = tuple(min(previous, curve[n] / n) for previous, curve in zip(units, curves))
        launches = tuple(min(previous, curve[n]) for previous, curve in zip(launches, curves))
        unit_minima.append(units)
        launch_minima.append(launches)

    def remaining_bound(state: State) -> float:
        if not state:
            return 0.0
        n = len(state)
        visits = [sum(chain) for chain in state]
        sizes = [len(chain) for chain in state]
        # The resident count can only fall; future launches have width <= n.
        # A single chain needs sum(depths) Core rounds, len(chain) Boundaries
        # with coda, and len(chain)-1 nonempty successor Prelude launches.
        work = sum(visits), sum(sizes), sum(sizes) - n
        serial = max(visits), max(sizes), max(sizes) - 1
        bound = fsum(
            max(unit * total, launch * count)
            for unit, total, launch, count in zip(unit_minima[n], work, launch_minima[n], serial)
        )
        return max(0.0, bound - 1e-12 * max(1.0, bound))

    def feasible_suffix(state: State) -> tuple[float, int]:
        path = []
        cursor = state
        while cursor not in memo and cursor not in upper_memo:
            _, segment_costs = _segment_costs(cursor, costs)
            h = len(segment_costs) - 1
            path.append((cursor, h, segment_costs[h]))
            cursor = _successor(cursor, h)
        suffix = memo[cursor][0] if cursor in memo else upper_memo[cursor][0]
        for parent, h, cost in reversed(path):
            suffix = cost + suffix
            upper_memo[parent] = suffix, h
        return memo[state] if state in memo else upper_memo[state]

    def make_frame(state: State) -> _Frame:
        upper, choice = feasible_suffix(state)
        edges = sorted(
            _segments(state, costs),
            key=lambda edge: (edge[1] + (memo[edge[2]][0] if edge[2] in memo else remaining_bound(edge[2])), -edge[0]),
        )
        return _Frame(state, iter(edges), upper, choice)

    pruned = 0
    stack = [make_frame(initial)]
    while stack:
        frame = stack[-1]
        if frame.awaiting is not None:
            h, cost, child = frame.awaiting
            frame.consider(h, cost + memo[child][0])
            frame.awaiting = None
        edge = next(frame.edges, None)
        if edge is None:
            memo[frame.state] = frame.best_cost, frame.best_rounds
            stack.pop()
            continue
        h, cost, child = edge
        if child in memo:
            frame.consider(h, cost + memo[child][0])
        elif cost + remaining_bound(child) >= frame.best_cost:
            pruned += 1
        else:
            frame.awaiting = edge
            stack.append(make_frame(child))

    actions: list[Action] = []
    segments = []
    state = initial
    while state:
        h = (memo[state] if state in memo else upper_memo[state])[1]
        segments.append(h)
        actions.extend(("core",) * h)
        actions.append("boundary")
        state = _successor(state, h)
    schedule = evaluate_actions(trace, costs, tuple(actions))
    # Report the replay's summation order, shared with all other schedules.
    return DPOracleResult(
        runtime_ms=schedule.runtime_ms,
        schedule=schedule,
        states_evaluated=len(memo) - 1,
        search_seconds=perf_counter() - started,
        core_segments=tuple(segments),
        pruned_segments=pruned,
    )
