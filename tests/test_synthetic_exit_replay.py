"""Per-request synthetic exit-depth replay building blocks for CDB (CPU only)."""

import pytest

from looped_cdb.continuous_depth_batching.requests import RequestState


def test_next_synthetic_exit_depth_returns_schedule_in_order() -> None:
    state = RequestState(request_id="r", initial_tokens=[1], synthetic_exit_depths=[0, 2, 1])
    assert [state.next_synthetic_exit_depth() for _ in range(3)] == [0, 2, 1]


def test_next_synthetic_exit_depth_is_none_without_schedule() -> None:
    state = RequestState(request_id="r", initial_tokens=[1])
    assert state.next_synthetic_exit_depth() is None


def test_next_synthetic_exit_depth_raises_on_exhaustion() -> None:
    state = RequestState(request_id="r", initial_tokens=[1], synthetic_exit_depths=[0])
    assert state.next_synthetic_exit_depth() == 0
    with pytest.raises(RuntimeError, match="exhausted"):
        state.next_synthetic_exit_depth()


def test_synthetic_exit_cursors_are_per_request() -> None:
    a = RequestState(request_id="a", initial_tokens=[1], synthetic_exit_depths=[0, 1])
    b = RequestState(request_id="b", initial_tokens=[1], synthetic_exit_depths=[2, 3])
    assert a.next_synthetic_exit_depth() == 0
    assert b.next_synthetic_exit_depth() == 2
    assert a.next_synthetic_exit_depth() == 1
    assert b.next_synthetic_exit_depth() == 3
