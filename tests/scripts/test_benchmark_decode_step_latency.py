"""Behavior tests for the decode-step latency benchmark's pure helpers.

The GPU measurement path is exercised on the cluster; here we pin the forward-agnostic
pieces the reported numbers rest on: the depth regression that turns per-depth
decode-step latencies into a per-recurrent-step slope, the ``max_new_tokens`` budget
that keeps the batch full through the timed window, and the KV-cache sizing that must
keep every request resident for its whole span.
"""

from benchmark_decode_step_latency import (
    DECODE_MARGIN,
    _cache_blocks,
    _fit_core_overhead,
)


def test_fit_recovers_slope_and_intercept_from_linear_data() -> None:
    # decode_step(S) = 0.3 (fixed per-token overhead) + 4.0 * S (per recurrent step)
    depths = [1, 2, 3, 4]
    decode_ms = [0.3 + 4.0 * s for s in depths]

    slope, intercept, r2 = _fit_core_overhead(depths, decode_ms)

    assert abs(slope - 4.0) < 1e-9
    assert abs(intercept - 0.3) < 1e-9
    assert abs(r2 - 1.0) < 1e-12


def test_fit_is_robust_to_small_jitter() -> None:
    depths = [1, 2, 3, 4]
    clean = [0.3 + 4.0 * s for s in depths]
    jittered = [y + j for y, j in zip(clean, (0.05, -0.04, 0.03, -0.02), strict=True)]

    slope, _intercept, r2 = _fit_core_overhead(depths, jittered)

    # A few percent of per-point noise must not move the slope far or break linearity.
    assert abs(slope - 4.0) < 0.1
    assert r2 > 0.99


def test_cache_blocks_keep_all_requests_resident() -> None:
    batch, context, max_new, block_size = 64, 2048, 120, 256
    num_blocks, blocks_per_request = _cache_blocks(batch, context, max_new, block_size)

    # A request's full (context + max_new) span fits in its block budget, which also
    # keeps it on the decode fast path...
    assert blocks_per_request * block_size >= context + max_new
    # ...and the pool holds all batch sequences at once with headroom to spare.
    assert num_blocks >= batch * blocks_per_request


def test_decode_margin_outlasts_the_window() -> None:
    # The per-request budget is warmup + timed + DECODE_MARGIN; it must outlast the one token a
    # request emits from prefill plus the warmup and timed decode steps, so none drops out early.
    warmup, timed = 3, 16
    max_new = warmup + timed + DECODE_MARGIN
    assert max_new > 1 + warmup + timed
