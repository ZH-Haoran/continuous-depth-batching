"""Tests for synthetic (explicit-depth) workloads and request slicing."""

from __future__ import annotations

import numpy as np
import pytest

from looped_cdb.benchmarks.workload import Workload, depths_from_values


def test_synthetic_fixed_lengths_and_shapes() -> None:
    wl = Workload.synthetic(num_requests=8, input_len=512, output_len=128, exit_dist="all_full", max_depth=4, seed=0)
    assert wl.num_requests == 8
    assert wl.total_steps == 4
    assert wl.input_lens.tolist() == [512] * 8
    assert wl.output_lens.tolist() == [128] * 8
    assert int(wl.offsets[-1]) == 8 * 128
    # all_full -> every token at max depth; first token stays full downstream anyway.
    depths = wl.materialize_depths(min_exit_step=1)
    assert all(d == 4 for row in depths for d in row)
    assert [len(row) for row in depths] == [128] * 8


def test_synthetic_uniform_length_range_is_seeded() -> None:
    a = Workload.synthetic(
        num_requests=32,
        input_len=16,
        output_len=8,
        exit_dist="bimodal_50",
        max_depth=4,
        seed=5,
        input_len_high=64,
        output_len_high=32,
    )
    b = Workload.synthetic(
        num_requests=32,
        input_len=16,
        output_len=8,
        exit_dist="bimodal_50",
        max_depth=4,
        seed=5,
        input_len_high=64,
        output_len_high=32,
    )
    np.testing.assert_array_equal(a.input_lens, b.input_lens)
    np.testing.assert_array_equal(a.output_lens, b.output_lens)
    assert a.input_lens.min() >= 16 and a.input_lens.max() <= 64
    assert a.output_lens.min() >= 8 and a.output_lens.max() <= 32


def test_synthetic_materialize_applies_min_exit_step_floor() -> None:
    wl = Workload.synthetic(num_requests=4, input_len=10, output_len=5, exit_dist="all_shallow1", max_depth=4, seed=0)
    # all_shallow1 wants depth 1, but min_exit_step=2 floors it (matches recorded path).
    depths = wl.materialize_depths(min_exit_step=2)
    assert all(d == 2 for row in depths for d in row)


def test_synthetic_roundtrip_through_disk(tmp_path) -> None:
    wl = Workload.synthetic(
        num_requests=6, input_len=20, output_len=10, exit_dist="gate_shallow_heavy", max_depth=4, seed=1
    )
    json_path, npz_path = wl.save(tmp_path / "syn.json")
    assert json_path.exists() and npz_path.exists()
    loaded = Workload.load(json_path)
    assert loaded.exit_pdf is None
    np.testing.assert_array_equal(loaded.exit_depths, wl.exit_depths)
    np.testing.assert_array_equal(loaded.offsets, wl.offsets)
    assert loaded.total_steps == 4
    assert loaded.materialize_depths(min_exit_step=1) == wl.materialize_depths(min_exit_step=1)


def test_take_slices_requests_and_rows() -> None:
    wl = Workload.synthetic(num_requests=10, input_len=4, output_len=3, exit_dist="clustered_20", max_depth=4, seed=2)
    sliced = wl.take(3)
    assert sliced.num_requests == 3
    assert int(sliced.offsets[-1]) == 3 * 3
    np.testing.assert_array_equal(sliced.exit_depths, wl.exit_depths[: 3 * 3])
    # take(None) and take(>=N) return the whole workload unchanged.
    assert wl.take(None) is wl
    assert wl.take(999) is wl


def test_materialize_requires_threshold_only_for_recorded() -> None:
    recorded = Workload(
        ids=["r0"],
        input_lens=np.array([4], dtype=np.int32),
        output_lens=np.array([2], dtype=np.int32),
        offsets=np.array([0, 2], dtype=np.int64),
        exit_pdf=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16),
    )
    with pytest.raises(ValueError, match="threshold is required"):
        recorded.materialize_depths()


def test_exactly_one_exit_source_required() -> None:
    with pytest.raises(ValueError, match="exactly one of exit_pdf, exit_values or exit_depths"):
        Workload(
            ids=["r0"],
            input_lens=np.array([4], dtype=np.int32),
            output_lens=np.array([2], dtype=np.int32),
            offsets=np.array([0, 2], dtype=np.int64),
        )


def test_depths_from_values_takes_the_first_step_below_threshold() -> None:
    """A convergence criterion decays, so the crossing is inverted relative to a gate PDF."""

    values = np.array([[0.9, 0.5, 0.02, 0.01], [0.9, 0.8, 0.7, 0.6]], dtype=np.float16)

    depths = depths_from_values(values, threshold=0.03)

    # Row 0 converges at step 3 (1-indexed); row 1 never does and runs to the budget.
    assert depths.tolist() == [3, 4]
    # The comparison is strict, so 0.0 never fires on non-negative values: every token runs
    # full depth. That makes q=0.0 the no-exit control of a convergence-threshold sweep.
    assert depths_from_values(values, threshold=0.0).tolist() == [4, 4]


def test_depths_from_values_respects_min_exit_step_and_delay() -> None:
    values = np.array([[0.01, 0.01, 0.01, 0.01]], dtype=np.float16)

    assert depths_from_values(values, threshold=0.03).tolist() == [1]
    assert depths_from_values(values, threshold=0.03, min_exit_step=3).tolist() == [3]
    assert depths_from_values(values, threshold=0.03, exit_delay_steps=2).tolist() == [3]
    # Never past the depth budget.
    assert depths_from_values(values, threshold=0.03, exit_delay_steps=10).tolist() == [4]


def test_recorded_values_workload_materializes_depths() -> None:
    workload = Workload(
        ids=["r0", "r1"],
        input_lens=np.array([4, 4], dtype=np.int32),
        output_lens=np.array([2, 1], dtype=np.int32),
        offsets=np.array([0, 2, 3], dtype=np.int64),
        exit_values=np.array([[0.9, 0.01], [0.9, 0.9], [0.005, 0.001]], dtype=np.float16),
    )

    assert workload.total_steps == 2
    assert workload.materialize_depths(threshold=0.03) == [[2, 2], [1]]


def test_shuffle_is_keyed_on_ids() -> None:
    wl = Workload.synthetic(num_requests=10, input_len=4, output_len=3, exit_dist="clustered_20", max_depth=4, seed=2)
    # The same requests stored in reverse order shuffle into the same replay order.
    reversed_wl = wl._reorder(np.arange(wl.num_requests)[::-1])
    a, b = wl.shuffle(3), reversed_wl.shuffle(3)
    assert a.ids == b.ids
    assert a.ids != wl.ids and sorted(a.ids) == sorted(wl.ids)
    np.testing.assert_array_equal(a.exit_depths, b.exit_depths)
    # Each request keeps its own rows.
    k = wl.ids.index(a.ids[0])
    np.testing.assert_array_equal(a.exit_depths[:3], wl.exit_depths[3 * k : 3 * k + 3])
