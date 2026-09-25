from __future__ import annotations

import time
from itertools import pairwise
from types import SimpleNamespace

import pytest

from looped_cdb.arrivals import ArrivalQueue, poisson_arrival_offsets


def _states(count: int) -> list[SimpleNamespace]:
    return [SimpleNamespace(request_id=f"request-{i}", created_time=-1.0) for i in range(count)]


def test_poisson_arrival_offsets_are_deterministic_and_ordered() -> None:
    offsets = poisson_arrival_offsets(1000, 5.0, seed=7)
    assert offsets == poisson_arrival_offsets(1000, 5.0, seed=7)
    assert offsets != poisson_arrival_offsets(1000, 5.0, seed=8)
    assert all(later >= earlier for earlier, later in pairwise(offsets))
    assert all(offset > 0 for offset in offsets)
    # Mean inter-arrival gap of a rate-5 Poisson process is 0.2 s (loose bound over 1000 samples).
    assert offsets[-1] / len(offsets) == pytest.approx(0.2, rel=0.2)


@pytest.mark.parametrize(
    ("num_requests", "rate", "error_match"),
    [(0, 1.0, "num_requests"), (1, 0.0, "rate_rps"), (1, -2.0, "rate_rps")],
)
def test_poisson_arrival_offsets_validates_inputs(num_requests: int, rate: float, error_match: str) -> None:
    with pytest.raises(ValueError, match=error_match):
        poisson_arrival_offsets(num_requests, rate, seed=0)


def test_arrival_queue_releases_due_requests_and_stamps_arrival_time() -> None:
    states = _states(3)
    queue = ArrivalQueue(states, [0.0, 0.0, 60.0])
    queue.start()

    released = queue.release_due()
    assert released == states[:2]
    assert queue.pending
    # The stamp is the scheduled arrival instant, so both zero-offset requests share it exactly
    # and the delay until the engine noticed them counts as queueing time.
    assert states[0].created_time == states[1].created_time
    assert states[0].created_time <= time.perf_counter()
    assert states[2].created_time == -1.0
    assert queue.release_due() == []


def test_arrival_queue_wait_sleeps_until_the_next_offset() -> None:
    states = _states(2)
    offset = 0.05
    queue = ArrivalQueue(states, [0.0, offset])
    queue.start()
    start = time.perf_counter()
    assert queue.release_due() == states[:1]

    queue.wait_for_next_arrival()
    assert time.perf_counter() - start >= offset
    assert queue.release_due() == states[1:]
    assert not queue.pending
    with pytest.raises(RuntimeError, match="no arrivals left"):
        queue.wait_for_next_arrival()


def test_arrival_queue_requires_start_and_valid_offsets() -> None:
    queue = ArrivalQueue(_states(1), [0.0])
    with pytest.raises(RuntimeError, match="start"):
        queue.release_due()
    with pytest.raises(RuntimeError, match="start"):
        queue.wait_for_next_arrival()
    with pytest.raises(ValueError, match="offsets"):
        ArrivalQueue(_states(2), [0.0])
    with pytest.raises(ValueError, match="non-negative"):
        ArrivalQueue(_states(1), [-0.5])
    with pytest.raises(ValueError, match="non-decreasing"):
        ArrivalQueue(_states(2), [1.0, 0.5])
