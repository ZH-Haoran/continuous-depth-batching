"""End-to-end: recorded Workload PDF -> materialized depths -> replay schedule."""

import numpy as np
import pytest

from looped_cdb.benchmarks.runner import materialized_to_replay
from looped_cdb.benchmarks.workload import Workload


def _toy_workload() -> Workload:
    # Two requests with 3 and 2 output tokens, 3 recurrent steps each. Each PDF row
    # is chosen so the cumulative-threshold depth at 0.5 is unambiguous.
    exit_pdf = np.array(
        [
            [1.0, 0.0, 0.0],  # req0 t0 -> depth 1
            [0.0, 1.0, 0.0],  # req0 t1 -> depth 2
            [0.0, 0.0, 1.0],  # req0 t2 -> depth 3
            [0.4, 0.4, 0.2],  # req1 t0 -> cumsum crosses 0.5 at step 1 -> depth 2
            [0.0, 0.0, 1.0],  # req1 t1 -> depth 3
        ],
        dtype=np.float16,
    )
    return Workload(
        ids=["r0", "r1"],
        input_lens=np.array([5, 7], dtype=np.int32),
        output_lens=np.array([3, 2], dtype=np.int32),
        exit_pdf=exit_pdf,
        offsets=np.array([0, 3, 5], dtype=np.int64),
        meta={"dataset": "toy"},
    )


def test_materialize_depths_matches_designed_pdf() -> None:
    depths = _toy_workload().materialize_depths(threshold=0.5, min_exit_step=1)
    assert depths == [[1, 2, 3], [2, 3]]


def test_replay_drops_first_token_and_offsets_immediate_policy() -> None:
    depths = _toy_workload().materialize_depths(threshold=0.5, min_exit_step=1)
    replay = materialized_to_replay(depths, delayed=False)
    # Drop each request's first (full-depth) output token, then 0-index (depth - 1).
    assert replay == [[1, 2], [2]]
    # Each replay list has output_len - 1 entries.
    assert [len(row) for row in replay] == [2, 1]


def test_replay_respects_min_exit_step_for_delayed_policy() -> None:
    # min_exit_step=2 floors depths at 2, so the delayed (-2) policy stays non-negative.
    depths = _toy_workload().materialize_depths(threshold=0.5, min_exit_step=2)
    replay = materialized_to_replay(depths, delayed=True)
    assert all(value >= 0 for row in replay for value in row)


def test_mean_requested_depth_counts_first_tokens_at_full_depth() -> None:
    # Depths at threshold 0.5 are [[1, 2, 3], [2, 3]], but each request's first token is
    # produced by the full-depth prefill: [[3, 2, 3], [3, 3]] -> mean depth 14/5.
    assert _toy_workload().mean_requested_depth_at(threshold=0.5) == pytest.approx(14 / 5)
    # A delay of one step shifts every non-saturated depth: [[3, 3, 3], [3, 3]] -> full depth.
    assert _toy_workload().mean_requested_depth_at(threshold=0.5, exit_delay_steps=1) == pytest.approx(3.0)


def test_mean_requested_depth_from_convergence_values() -> None:
    # A token exits at the first step whose criterion value falls below the threshold:
    # depths [3, 2] and [1, 3] at 0.5, first tokens forced to 3 -> [[3, 2], [3, 3]].
    workload = Workload(
        ids=["r0", "r1"],
        input_lens=np.array([4, 4], dtype=np.int32),
        output_lens=np.array([2, 2], dtype=np.int32),
        offsets=np.array([0, 2, 4], dtype=np.int64),
        exit_values=np.array([[0.9, 0.9, 0.9], [0.9, 0.4, 0.1], [0.4, 0.3, 0.1], [0.9, 0.9, 0.1]], dtype=np.float16),
    )
    assert workload.mean_requested_depth_at(threshold=0.5) == pytest.approx(11 / 4)


def test_mean_requested_depth_of_explicit_depths_needs_no_threshold() -> None:
    workload = Workload(
        ids=["r0", "r1"],
        input_lens=np.array([4, 4], dtype=np.int32),
        output_lens=np.array([2, 2], dtype=np.int32),
        offsets=np.array([0, 2, 4], dtype=np.int64),
        exit_depths=np.array([1, 2, 3, 1], dtype=np.int32),
        max_depth=3,
    )
    # First tokens forced to 3 -> [[3, 2], [3, 1]] -> mean depth 9/4.
    assert workload.mean_requested_depth_at() == pytest.approx(9 / 4)
