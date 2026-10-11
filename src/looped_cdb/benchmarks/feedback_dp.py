"""Exact offline DP over post-Boundary states of the flush-all feedback model."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from time import perf_counter

from .feedback import Action, CostTable, RecurrenceTrace, ScheduleResult, State, evaluate_actions


@dataclass(frozen=True)
class DPOracleResult:
    runtime_ms: float
    schedule: ScheduleResult
    states_evaluated: int
    search_seconds: float
    core_segments: tuple[int, ...]


def _successor(state: State, rounds: int) -> State:
    return tuple(
        sorted(
            (chain[0] - rounds, *chain[1:]) if chain[0] > rounds else chain[1:]
            for chain in state
            if chain[0] > rounds or len(chain) > 1
        )
    )


def _segments(state: State, costs: CostTable) -> Iterator[tuple[int, float, State]]:
    """Enumerate every legal Core^h -> Boundary edge, computing prefixes once."""
    minimum = min(chain[0] for chain in state)
    maximum = max(chain[0] for chain in state)
    exits = [0] * (maximum + 1)
    releases = [0] * (maximum + 1)
    for chain in state:
        exits[chain[0]] += 1
        releases[chain[0]] += len(chain) > 1
    prefix_core_cost = [0.0] * (maximum + 1)
    batch = len(state)
    pending = successors = 0
    for h in range(1, maximum + 1):
        # Before round h, the runnable heads are exactly d_i > h-1.
        prefix_core_cost[h] = prefix_core_cost[h - 1] + costs.core_ms[batch]
        batch -= exits[h]
        pending += exits[h]
        successors += releases[h]
        if h >= minimum:
            segment_cost = prefix_core_cost[h] + costs.coda_ms[pending] + costs.prelude_ms[successors]
            yield h, segment_cost, _successor(state, h)


@dataclass
class _Frame:
    state: State
    edges: Iterator[tuple[int, float, State]]
    best_cost: float = float("inf")
    best_rounds: int = 0
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
    stack memoizes V(S) without consuming Python recursion depth.
    """
    started = perf_counter()
    if len(trace.chains) > costs.resident_cap:
        raise ValueError("trace exceeds the resident cap")
    initial = trace.initial_state
    # Store both the optimal suffix value and the chosen macro action.
    memo: dict[State, tuple[float, int]] = {(): (0.0, 0)}
    stack = [_Frame(initial, _segments(initial, costs))]
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
        else:
            frame.awaiting = edge
            stack.append(_Frame(child, _segments(child, costs)))

    actions: list[Action] = []
    segments = []
    state = initial
    while state:
        h = memo[state][1]
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
    )
