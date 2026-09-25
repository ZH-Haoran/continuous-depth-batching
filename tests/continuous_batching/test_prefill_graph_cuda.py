"""GPU checks of the breakable graph primitives: non-owning device aliases and segment capture/replay.

Run on a GPU node via ``shells/pytest_cuda.sh`` with ``LOOPED_CDB_RUN_CUDA_TESTS=1``.
"""

from __future__ import annotations

import pytest
import torch

from looped_cdb.continuous_batching.prefill_graph import BreakableCapture, BreakableGraph, eager_break, weak_alias
from looped_cdb.continuous_batching.utils import gc_paused

pytestmark = pytest.mark.cuda


def test_weak_alias_on_cuda_shares_memory_and_keeps_dtype() -> None:
    source = torch.arange(24, dtype=torch.bfloat16, device="cuda").reshape(4, 6)
    view = source[:, 1:4]
    alias = weak_alias(view)
    assert alias.device == view.device
    assert alias.dtype == torch.bfloat16
    assert alias.shape == view.shape
    assert alias.data_ptr() == view.data_ptr()
    source.fill_(2.0)
    torch.cuda.synchronize()
    assert torch.equal(alias, view)
    # The alias does not own the memory: dropping the source returns its block to the allocator.
    allocated_with_source = torch.cuda.memory_allocated()
    del source, view
    assert torch.cuda.memory_allocated() < allocated_with_source


def test_breakable_capture_replays_segments_around_an_eager_break() -> None:
    # A two-segment toy: segment 1 scales the input into graph memory, the eager break squares that
    # intermediate into an output buffer allocated by the segment, segment 2 adds one on top. Replay
    # with a new input must see the new intermediate through the alias and recompute the break.
    # During the capture itself the segments are recorded, not executed, so the break runs on an
    # unexecuted input; only the replays carry real values.
    stream = torch.cuda.Stream()
    pool = torch.cuda.graph_pool_handle()
    static_in = torch.ones(8, device="cuda")
    static_out = torch.zeros(8, device="cuda")
    seen: list[float] = []

    @eager_break
    def square_into(value: torch.Tensor, out: torch.Tensor) -> None:
        seen.append(float(value[0].item()))
        out.copy_(value * value)

    def forward() -> None:
        intermediate = static_in * 3.0
        out = torch.empty_like(intermediate)
        square_into(intermediate, out)
        static_out.copy_(out + 1.0)

    graph = BreakableGraph()
    with torch.cuda.stream(stream):
        forward()
        torch.cuda.synchronize()
        with gc_paused(), BreakableCapture(graph, pool=pool):
            forward()
    torch.cuda.synchronize()
    assert len(graph.segments) == 2
    assert len(graph.breaks) == 1
    assert seen[0] == 3.0 and len(seen) == 2

    static_in.fill_(2.0)
    with torch.cuda.stream(stream):
        graph.replay()
    torch.cuda.synchronize()
    assert seen[-1] == 6.0
    assert torch.equal(static_out, torch.full((8,), 37.0, device="cuda"))

    graph.reset()
    assert graph.segments == [] and graph.breaks == []
