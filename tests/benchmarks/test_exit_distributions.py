"""Tests for the shared synthetic exit-depth distributions."""

from __future__ import annotations

import numpy as np
import pytest

from looped_cdb.benchmarks.exit_distributions import (
    DISTRIBUTION_KINDS,
    depth_weights,
    sample_token_depths,
)


def test_gate_ramps_match_canonical_paper_fractions_at_depth4() -> None:
    assert depth_weights("gate_shallow_heavy", 4) == {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.1}
    assert depth_weights("gate_deep_heavy", 4) == {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}


def test_constant_kinds_are_single_valued_and_ignore_seed() -> None:
    depths = sample_token_depths("all_shallow2", 100, 4, seed=7)
    assert depths.dtype == np.int32
    assert depths.tolist() == [2] * 100


def test_all_full_is_max_depth() -> None:
    assert sample_token_depths("all_full", 50, 4).tolist() == [4] * 50


def test_sampling_is_deterministic_and_in_range() -> None:
    a = sample_token_depths("gate_shallow_heavy", 500, 4, seed=3)
    b = sample_token_depths("gate_shallow_heavy", 500, 4, seed=3)
    np.testing.assert_array_equal(a, b)
    assert a.min() >= 1 and a.max() <= 4
    # Shallow-heavy should have mean depth below the midpoint.
    assert a.mean() < 2.5


def test_empty_count_returns_empty() -> None:
    assert sample_token_depths("bimodal_50", 0, 4).shape == (0,)


def test_all_shallow2_requires_depth2() -> None:
    with pytest.raises(ValueError, match="max_depth >= 2"):
        depth_weights("all_shallow2", 1)


def test_unknown_kind_raises() -> None:
    with pytest.raises(ValueError, match="unknown exit distribution"):
        depth_weights("nonsense", 4)


def test_all_kinds_produce_valid_depths() -> None:
    for kind in DISTRIBUTION_KINDS:
        depths = sample_token_depths(kind, 64, 4, seed=1)
        assert depths.min() >= 1 and depths.max() <= 4
