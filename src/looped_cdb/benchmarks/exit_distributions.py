"""Named synthetic exit-depth distributions.

Shared source of truth for the per-token recurrent-depth distributions used by
synthetic workloads (:meth:`looped_cdb.benchmarks.workload.Workload.synthetic`).
Each kind maps to a categorical distribution over depths ``1..max_depth`` that
tokens are sampled from; constant kinds (``all_*``) collapse to a single depth.

The gate-like ramps generalize to any ``max_depth`` as linear weights, and at
``max_depth == 4`` reproduce the canonical paper fractions
(shallow-heavy ``{1:0.4, 2:0.3, 3:0.2, 4:0.1}``, deep-heavy the mirror).
"""

from __future__ import annotations

import numpy as np

DISTRIBUTION_KINDS = (
    "all_full",
    "all_shallow1",
    "all_shallow2",
    "bimodal_50",
    "clustered_20",
    "gate_shallow_heavy",
    "gate_deep_heavy",
)


def depth_weights(kind: str, max_depth: int) -> dict[int, float]:
    """Return the normalized categorical weights over depths ``1..max_depth``."""

    if max_depth < 1:
        raise ValueError(f"max_depth must be >= 1, got {max_depth}")

    if kind == "all_full":
        weights = {max_depth: 1.0}
    elif kind == "all_shallow1":
        weights = {1: 1.0}
    elif kind == "all_shallow2":
        if max_depth < 2:
            raise ValueError("all_shallow2 requires max_depth >= 2")
        weights = {2: 1.0}
    elif kind == "bimodal_50":
        weights = {1: 0.5, max_depth: 0.5}
    elif kind == "clustered_20":
        weights = {1: 0.8, max_depth: 0.2}
    elif kind == "gate_shallow_heavy":
        weights = {depth: float(max_depth - depth + 1) for depth in range(1, max_depth + 1)}
    elif kind == "gate_deep_heavy":
        weights = {depth: float(depth) for depth in range(1, max_depth + 1)}
    else:
        raise ValueError(f"unknown exit distribution kind: {kind!r} (known: {', '.join(DISTRIBUTION_KINDS)})")

    total = sum(weights.values())
    return {depth: value / total for depth, value in weights.items()}


def sample_token_depths(kind: str, count: int, max_depth: int, *, seed: int = 0) -> np.ndarray:
    """Sample ``count`` per-token 1-indexed exit depths from a named distribution.

    Deterministic given ``seed``. Constant kinds return a filled array without
    consuming the RNG; multi-valued kinds draw each token independently from the
    categorical distribution defined by :func:`depth_weights`.
    """

    if count < 0:
        raise ValueError(f"count must be >= 0, got {count}")
    if count == 0:
        return np.empty(0, dtype=np.int32)

    weights = depth_weights(kind, max_depth)
    depths = np.array(sorted(weights), dtype=np.int32)
    if depths.shape[0] == 1:
        return np.full(count, int(depths[0]), dtype=np.int32)
    probs = np.array([weights[int(depth)] for depth in depths], dtype=np.float64)
    rng = np.random.default_rng(seed)
    return rng.choice(depths, size=count, p=probs).astype(np.int32)
