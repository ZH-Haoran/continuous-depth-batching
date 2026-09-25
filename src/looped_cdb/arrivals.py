"""Open-loop request arrivals for serving benchmarks.

A drain benchmark submits every request upfront and measures offline batch
throughput. An open-loop serving benchmark instead releases requests at recorded
wall-clock offsets (typically a seeded Poisson process) and measures per-request
latency; the engine's admission and batching then see realistic, load-dependent
queue depths. :class:`ArrivalQueue` is the release mechanism shared by both
engines: the same offsets replayed against each backend give every system an
identical arrival trace.
"""

from __future__ import annotations

import time
from itertools import pairwise
from typing import Protocol

import numpy as np


class ArrivingState(Protocol):
    """The one attribute an arriving request state must expose: its arrival stamp."""

    created_time: float


def poisson_arrival_offsets(num_requests: int, rate_rps: float, *, seed: int) -> list[float]:
    """Seeded Poisson-process arrival offsets in seconds, one per request, non-decreasing.

    Offsets are relative to the benchmark's start: the first request arrives after one
    exponential inter-arrival gap (mean ``1 / rate_rps``), not at time zero. Deterministic
    given ``seed``, so every backend in a comparison replays the identical trace.
    """

    if num_requests < 1:
        raise ValueError(f"num_requests must be >= 1, got {num_requests}")
    if rate_rps <= 0:
        raise ValueError(f"rate_rps must be positive, got {rate_rps}")
    rng = np.random.default_rng(seed)
    return rng.exponential(1.0 / rate_rps, size=num_requests).cumsum().tolist()


class ArrivalQueue[StateT: ArrivingState]:
    """Releases pre-built request states at wall-clock offsets relative to :meth:`start`.

    ``release_due`` returns every not-yet-released state whose offset has elapsed and
    stamps its ``created_time`` with the scheduled arrival instant (not the poll instant),
    so the queueing delay between a request's arrival and the tick that notices it is
    measured rather than hidden. The engine polls at its natural boundary (a scheduler
    tick or a decode wave); when it has nothing to run, :meth:`wait_for_next_arrival`
    sleeps exactly until the next offset instead of spinning.
    """

    def __init__(self, states: list[StateT], offsets_s: list[float]) -> None:
        if len(states) != len(offsets_s):
            raise ValueError(f"got {len(states)} states but {len(offsets_s)} arrival offsets")
        if any(offset < 0 for offset in offsets_s):
            raise ValueError("arrival offsets must be non-negative")
        if any(later < earlier for earlier, later in pairwise(offsets_s)):
            raise ValueError("arrival offsets must be non-decreasing (requests are released in order)")
        self._states = states
        self._offsets_s = offsets_s
        self._next = 0
        self._start_time: float | None = None

    def start(self) -> None:
        """Anchor the offsets to the current wall clock; must be called before any release."""

        self._start_time = time.perf_counter()

    @property
    def pending(self) -> bool:
        """Whether any request has not been released yet."""

        return self._next < len(self._states)

    def release_due(self) -> list[StateT]:
        """Return the states whose arrival offset has elapsed, stamping their arrival time."""

        if self._start_time is None:
            raise RuntimeError("ArrivalQueue.start() must be called before releasing requests")
        elapsed = time.perf_counter() - self._start_time
        released: list[StateT] = []
        while self._next < len(self._states) and self._offsets_s[self._next] <= elapsed:
            state = self._states[self._next]
            state.created_time = self._start_time + self._offsets_s[self._next]
            released.append(state)
            self._next += 1
        return released

    def wait_for_next_arrival(self) -> None:
        """Sleep until the next arrival offset elapses (a no-op if it already has)."""

        if self._start_time is None:
            raise RuntimeError("ArrivalQueue.start() must be called before waiting on arrivals")
        if not self.pending:
            raise RuntimeError("no arrivals left to wait for")
        remaining = self._offsets_s[self._next] - (time.perf_counter() - self._start_time)
        if remaining > 0:
            time.sleep(remaining)
