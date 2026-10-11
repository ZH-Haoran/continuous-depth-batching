"""CPU-only feedback batching model with a fixed resident-request budget.

Each sequence is a finite recurrence-depth chain. Successors become runnable only
after their predecessor's boundary service; no scheduler-independent exit times
are replayed. All stage costs refer to effective batch sizes at one fixed cap.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Action = Literal["core", "boundary"]
# A sequence holds (remaining current loops, *successor depths). A zero head
# means pending coda; finished sequences disappear. Sorting quotients out labels
# because this model gives all sequences identical context and stage costs.
State = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class CostTable:
    """Measured, complete stage curves indexed by effective batch size, in ms."""

    resident_cap: int
    core_ms: tuple[float, ...]
    coda_ms: tuple[float, ...]
    prelude_ms: tuple[float, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.resident_cap) is not int or self.resident_cap < 1:
            raise ValueError("resident_cap must be a positive integer")
        for name in ("core_ms", "coda_ms", "prelude_ms"):
            values = getattr(self, name)
            if len(values) != self.resident_cap + 1 or values[0] != 0:
                raise ValueError(f"{name} must cover batches 0..resident_cap, with zero cost at batch 0")
            if any(not math.isfinite(value) or value < 0 for value in values):
                raise ValueError(f"{name} must contain finite nonnegative costs")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CostTable:
        if data.get("format") != "looped-cdb-stage-costs-v1":
            raise ValueError("expected looped-cdb-stage-costs-v1")
        cap = data["resident_cap"]
        if type(cap) is not int or cap < 1:
            raise ValueError("resident_cap must be a positive integer")
        rows: dict[int, dict[str, Any]] = {}
        for point in data["points"]:
            batch = point["batch_size"]
            if type(batch) is not int or not 1 <= batch <= cap or batch in rows:
                raise ValueError("batch_size must be a unique integer in 1..resident_cap")
            rows[batch] = point
        if set(rows) != set(range(1, cap + 1)):
            raise ValueError("measure every effective batch 1..resident_cap; do not interpolate across graph buckets")
        curves = {
            name: (0.0, *(float(rows[batch][name]) for batch in range(1, cap + 1)))
            for name in ("core_ms", "coda_ms", "prelude_ms")
        }
        return cls(cap, **curves, metadata=data.get("metadata", {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "looped-cdb-stage-costs-v1",
            "resident_cap": self.resident_cap,
            "metadata": self.metadata,
            "points": [
                {
                    "batch_size": batch,
                    "core_ms": self.core_ms[batch],
                    "coda_ms": self.coda_ms[batch],
                    "prelude_ms": self.prelude_ms[batch],
                }
                for batch in range(1, self.resident_cap + 1)
            ],
        }


@dataclass(frozen=True)
class RecurrenceTrace:
    """Positive loop demands, one finite chain per initially resident sequence."""

    chains: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not self.chains or any(not chain for chain in self.chains):
            raise ValueError("trace must contain nonempty sequence chains")
        if any(type(depth) is not int or depth < 1 for chain in self.chains for depth in chain):
            raise ValueError("recurrence depths must be positive integers (loop counts, not zero-based exit steps)")

    @property
    def initial_state(self) -> State:
        return tuple(sorted(self.chains))

    @property
    def token_count(self) -> int:
        return sum(map(len, self.chains))

    @property
    def core_visits(self) -> int:
        return sum(sum(chain) for chain in self.chains)


def generate_trace(
    *,
    sequences: int = 64,
    tokens_per_sequence: int = 4,
    mean_depth: float = 4,
    seed: int = 0,
    distribution: Literal["geometric", "two_point"] = "geometric",
    high_depth: int = 16,
) -> RecurrenceTrace:
    """Generate demands once, independent of when a scheduler releases tokens.

    ``two_point`` mixes depths 1 and high_depth with the requested population
    mean. At mean 4/high 16 it has higher variance than the geometric trace.
    """
    if type(sequences) is not int or type(tokens_per_sequence) is not int:
        raise ValueError("sequences and tokens_per_sequence must be integers")
    if sequences < 1 or tokens_per_sequence < 1 or not math.isfinite(mean_depth) or mean_depth < 1:
        raise ValueError("positive chain sizes and a finite mean_depth >= 1 are required")
    if distribution not in ("geometric", "two_point"):
        raise ValueError(f"unknown distribution: {distribution}")
    if distribution == "two_point" and (type(high_depth) is not int or high_depth <= 1 or mean_depth > high_depth):
        raise ValueError("two_point requires integer high_depth > 1 and mean_depth <= high_depth")
    rng = random.Random(seed)

    def draw() -> int:
        if mean_depth == 1:
            return 1
        if distribution == "two_point":
            return high_depth if rng.random() < (mean_depth - 1) / (high_depth - 1) else 1
        return 1 + math.floor(math.log1p(-rng.random()) / math.log1p(-1 / mean_depth))

    return RecurrenceTrace(tuple(tuple(draw() for _ in range(tokens_per_sequence)) for _ in range(sequences)))


def occupancy(state: State) -> tuple[int, int, int]:
    """Return resident requests, effective core batch, pending coda tokens."""
    b = sum(sequence[0] > 0 for sequence in state)
    return len(state), b, len(state) - b


def transition(state: State, action: Action, costs: CostTable) -> tuple[State, float, int]:
    """Advance one stage; return successor state, cost and released successors."""
    n, b, q = occupancy(state)
    if n > costs.resident_cap:
        raise ValueError("state exceeds the resident cap")
    if action == "core":
        if not b:
            raise ValueError("cannot execute an empty core batch")
        next_state = tuple(sorted((max(0, seq[0] - 1), *seq[1:]) for seq in state))
        return next_state, costs.core_ms[b], 0
    if action != "boundary":
        raise ValueError(f"unknown action: {action}")
    if not q:
        raise ValueError("cannot execute an empty boundary batch")
    successors = sum(seq[0] == 0 and len(seq) > 1 for seq in state)
    next_state = tuple(sorted(seq if seq[0] > 0 else seq[1:] for seq in state if seq[0] > 0 or len(seq) > 1))
    return next_state, costs.coda_ms[q] + costs.prelude_ms[successors], successors


@dataclass(frozen=True)
class ScheduleStep:
    action: str
    resident: int
    core_batch: int
    pending_coda: int
    successors: int
    duration_ms: float
    end_ms: float


@dataclass(frozen=True)
class ScheduleResult:
    runtime_ms: float
    steps: tuple[ScheduleStep, ...]
    core_hole_rounds: int
    pending_coda_rounds: int
    terminal_hole_rounds: int
    boundary_batches: int
    core_visits: int
    coda_tokens: int
    prelude_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_actions(trace: RecurrenceTrace, costs: CostTable, actions: tuple[Action, ...]) -> ScheduleResult:
    """Replay a complete schedule, including common initial prelude and final coda."""
    if len(trace.chains) > costs.resident_cap:
        raise ValueError("trace exceeds the resident cap")
    state = trace.initial_state
    elapsed = costs.prelude_ms[len(state)]
    steps = [ScheduleStep("initial_prelude", len(state), len(state), 0, len(state), elapsed, elapsed)]
    holes = pending = terminal = boundaries = visits = codas = 0
    preludes = len(state)
    for action in actions:
        n, b, q = occupancy(state)
        state, duration, successors = transition(state, action, costs)
        elapsed += duration
        steps.append(ScheduleStep(action, n, b, q, successors, duration, elapsed))
        if action == "core":
            holes += costs.resident_cap - b
            pending += q
            terminal += costs.resident_cap - n
            visits += b
        else:
            boundaries += 1
            codas += q
            preludes += successors
    if state:
        raise ValueError("schedule ended before all chains completed")
    return ScheduleResult(elapsed, tuple(steps), holes, pending, terminal, boundaries, visits, codas, preludes)


def simulate_static(trace: RecurrenceTrace, costs: CostTable, threshold: int) -> ScheduleResult:
    """Flush all pending tokens at q >= K, or when no core work remains."""
    if type(threshold) is not int or not 1 <= threshold <= costs.resident_cap:
        raise ValueError("threshold must be an integer in 1..resident_cap")
    if len(trace.chains) > costs.resident_cap:
        raise ValueError("trace exceeds the resident cap")
    actions: list[Action] = []
    state = trace.initial_state
    while state:
        _, b, q = occupancy(state)
        action: Action = "boundary" if q >= threshold or not b else "core"
        actions.append(action)
        state, _, _ = transition(state, action, costs)
    return evaluate_actions(trace, costs, tuple(actions))
