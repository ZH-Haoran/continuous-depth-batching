"""Piecewise CUDA graphs for the variable-length prefill forward.

A prompt-prefill forward is one full-depth variable-length pass over every prompt token of the
launch. Its shape is set by the total query tokens, which pads to a bucket, except inside attention:
``flash_attn_varlen_func`` takes the per-request lengths and the maximum sequence length as Python
ints, so a graph of the whole forward would be specific to the prompt mix of the launch it was
captured from. The forward is therefore captured as a *breakable* graph: one CUDA graph segment per
stretch between attention calls, with attention itself running eagerly between segments, reading the
launch's real lengths from :class:`PrefillGraphContext` at call time. Everything else in the forward
(norms, projections, MLPs, the LM head) replays from the graph.

Mechanics, as in SGLang's breakable CUDA graph:

* :class:`BreakableCapture` runs the forward once on the capture stream, beginning a new
  ``torch.cuda.CUDAGraph`` segment whenever the previous one ends. A function decorated with
  :func:`eager_break` ends the current segment, runs eagerly, and begins the next segment; the call and
  its arguments are recorded so :meth:`BreakableGraph.replay` can re-run it between the replayed
  segments.
* The recorded arguments alias the tensors the graph segments produce without owning them
  (:func:`weak_alias`), so the graph memory pool stays free to reuse those blocks; the addresses are
  stable because a segment writes the same addresses on every replay.
* The break's inputs come from graph memory and its result goes into a buffer the enclosing segment
  allocated, so no bridge buffers are held between launches.

The attention function registered for the paged implementations (:func:`graphed_paged_attention_forward`)
delegates to the stock paged forward whenever no prefill-graph context is active, so decode launches and
eager prefill are unchanged.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers.integrations.flash_paged import paged_attention_forward
from transformers.modeling_flash_attention_utils import lazy_import_paged_flash_attention
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .cache import PagedAttentionCache

# ----------------------------------------------------------------------------- token buckets


def prefill_graph_buckets(max_num_batched_tokens: int) -> list[int]:
    """Token counts a prefill launch pads to, one graph each.

    Steps of 16 while a launch is bound by the weight reads, where the padded rows are free, then
    steps of about a tenth of the size toward the token budget, which is always the last bucket.
    """

    if max_num_batched_tokens <= 0:
        raise ValueError(f"max_num_batched_tokens must be positive, got {max_num_batched_tokens}")
    sizes = (
        list(range(16, 257, 16))
        + list(range(288, 513, 32))
        + list(range(576, 1025, 64))
        + list(range(1280, 4097, 256))
        + list(range(4608, max_num_batched_tokens + 1, 512))
    )
    buckets = sorted({size for size in sizes if size <= max_num_batched_tokens} | {max_num_batched_tokens})
    return buckets


# ----------------------------------------------------------------------------- weak aliases

_TYPESTR_BY_ITEMSIZE = {1: "|i1", 2: "<i2", 4: "<i4", 8: "<i8"}
_VIEW_DTYPE_BY_ITEMSIZE = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


class _ArrayInterfaceHolder:
    """Exposes a raw device or host pointer through the array interface protocols."""

    def __init__(self, interface: dict[str, Any], cuda: bool) -> None:
        if cuda:
            self.__cuda_array_interface__ = interface
        else:
            self.__array_interface__ = interface


def weak_alias(tensor: torch.Tensor) -> torch.Tensor:
    """A tensor over the same memory as ``tensor`` that does not keep that memory alive.

    Built through the array interface, whose consumer (``torch.as_tensor``) references the small
    holder object rather than the storage. The alias is typed through a same-size integer dtype,
    since the interface has no spelling for bfloat16, and viewed back.
    """

    itemsize = tensor.element_size()
    interface: dict[str, Any] = {
        "shape": tuple(tensor.shape),
        "typestr": _TYPESTR_BY_ITEMSIZE[itemsize],
        "data": (tensor.data_ptr(), False),
        "strides": None if tensor.is_contiguous() else tuple(stride * itemsize for stride in tensor.stride()),
        "version": 3,
    }
    if tensor.device.type == "cuda":
        alias = torch.as_tensor(_ArrayInterfaceHolder(interface, cuda=True), device=tensor.device)
    else:
        # numpy reads the host interface into a non-owning view; the torch tensor then shares that view.
        alias = torch.from_numpy(np.asarray(_ArrayInterfaceHolder(interface, cuda=False)))
    if alias.dtype != _VIEW_DTYPE_BY_ITEMSIZE[itemsize]:
        raise RuntimeError(f"array interface produced {alias.dtype} for an itemsize of {itemsize}")
    return alias.view(tensor.dtype)


def _alias_argument(value: Any) -> Any:
    return weak_alias(value) if isinstance(value, torch.Tensor) else value


# ----------------------------------------------------------------------------- breakable graphs


class BreakableGraph:
    """CUDA graph segments with an eager break between consecutive segments."""

    def __init__(self) -> None:
        self.segments: list[Any] = []
        self.breaks: list[Callable[[], None]] = []

    def replay(self) -> None:
        """Replay every segment on the current stream, running the recorded break after each."""

        for index, segment in enumerate(self.segments):
            segment.replay()
            if index < len(self.breaks):
                self.breaks[index]()

    def reset(self) -> None:
        for segment in self.segments:
            segment.reset()
        self.segments = []
        self.breaks = []


_ACTIVE_CAPTURE: ContextVar[BreakableCapture | None] = ContextVar("breakable_capture", default=None)


class BreakableCapture:
    """Capture the enclosed code as a :class:`BreakableGraph` on the current stream.

    Every segment is captured into ``pool``; ``graph_factory`` makes the per-segment graph objects
    (``torch.cuda.CUDAGraph`` in use, a fake in tests). The caller owns the stream context and the
    garbage-collection pause around the capture.
    """

    def __init__(
        self,
        graph: BreakableGraph,
        *,
        pool: Any,
        capture_error_mode: str = "thread_local",
        graph_factory: Callable[[], Any] = torch.cuda.CUDAGraph,
    ) -> None:
        self.graph = graph
        self.pool = pool
        self.capture_error_mode = capture_error_mode
        self.graph_factory = graph_factory
        self._segment: Any | None = None
        self._token: Any = None

    def __enter__(self) -> BreakableCapture:
        if _ACTIVE_CAPTURE.get() is not None:
            raise RuntimeError("breakable graph captures do not nest")
        self._token = _ACTIVE_CAPTURE.set(self)
        self._begin_segment()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        # The open segment is ended even when the body raised, as ``torch.cuda.graph`` does, so no
        # capture is left in progress on the stream; the exception propagates.
        try:
            if self._segment is not None:
                self._end_segment()
        finally:
            _ACTIVE_CAPTURE.reset(self._token)
            self._segment = None

    def _begin_segment(self) -> None:
        segment = self.graph_factory()
        segment.capture_begin(pool=self.pool, capture_error_mode=self.capture_error_mode)
        self._segment = segment

    def _end_segment(self) -> None:
        segment = self._segment
        if segment is None:
            raise RuntimeError("no segment is being captured")
        segment.capture_end()
        self.graph.segments.append(segment)
        self._segment = None

    def run_break(self, fn: Callable[..., None], args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        """End the current segment, run ``fn`` eagerly, record it for replay, begin the next segment."""

        self._end_segment()
        result = fn(*args, **kwargs)
        if result is not None:
            raise TypeError("an eager break must write its result into a tensor it was given and return None")
        aliased_args = tuple(_alias_argument(value) for value in args)
        aliased_kwargs = {key: _alias_argument(value) for key, value in kwargs.items()}
        self.graph.breaks.append(functools.partial(fn, *aliased_args, **aliased_kwargs))
        self._begin_segment()


def eager_break(fn: Callable[..., None]) -> Callable[..., None]:
    """Run ``fn`` eagerly between graph segments while a :class:`BreakableCapture` is active.

    Outside a capture the function runs as is (eager launches, and the replays that re-run it).
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        capture = _ACTIVE_CAPTURE.get()
        if capture is None:
            return fn(*args, **kwargs)
        return capture.run_break(fn, args, kwargs)

    return wrapper


# ----------------------------------------------------------------------------- attention break


@dataclass
class PrefillGraphContext:
    """The launch's real variable-length metadata, read by the attention break at call time.

    ``num_tokens`` is the real query-token count inside the padded bucket; the index tensors are
    views on the static device buffers sliced to the launch.
    """

    num_tokens: int
    cu_seq_lens_q: torch.Tensor
    cu_seq_lens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    write_index: torch.Tensor
    read_index: torch.Tensor
    cache: PagedAttentionCache


_ACTIVE_CONTEXT: ContextVar[PrefillGraphContext | None] = ContextVar("prefill_graph_context", default=None)


@contextmanager
def prefill_graph_context(context: PrefillGraphContext) -> Iterator[None]:
    """Make ``context`` the launch the graphed attention reads, for a capture or a replay."""

    token = _ACTIVE_CONTEXT.set(context)
    try:
        yield
    finally:
        _ACTIVE_CONTEXT.reset(token)


def active_prefill_graph_context() -> PrefillGraphContext | None:
    return _ACTIVE_CONTEXT.get()


@eager_break
def _attention_break(module: Any, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor) -> None:
    """Attention over the launch's real tokens, written into the padded ``out`` rows.

    ``q``, ``k``, ``v`` are ``(1, heads, bucket, head_dim)``; ``out`` is ``(bucket, heads, head_dim)``.
    The padded tail of ``out`` is zeroed, and padded rows never reach the KV cache.

    This is the varlen branch of transformers' ``paged_attention_forward`` for the kwargs this
    engine passes. It drops the per-layer-type dict forms of ``cu_seq_lens_k`` / ``max_seqlen_k``
    (mixed sliding/full models, which
    :class:`~looped_cdb.continuous_batching.cache.PagedAttentionCache` rejects at construction)
    and the ``s_aux`` attention-sink kwarg, which no served model passes.
    """

    context = _ACTIVE_CONTEXT.get()
    if context is None:
        raise RuntimeError("the attention break runs only inside a prefill graph context")
    num_tokens = context.num_tokens
    flash_attn_varlen_func, _ = lazy_import_paged_flash_attention(module.config._attn_implementation)
    sliding_window = (-1, -1) if not getattr(module, "sliding_window", False) else (module.sliding_window - 1, 0)

    keys, values = context.cache.update(
        key_states=k[:, :, :num_tokens],
        value_states=v[:, :, :num_tokens],
        layer_idx=module.layer_idx,
        read_index=context.read_index,
        write_index=context.write_index,
    )
    attn_output = flash_attn_varlen_func(
        q[0, :, :num_tokens].transpose(0, 1).contiguous(),
        keys.contiguous(),
        values.contiguous(),
        context.cu_seq_lens_q.to(torch.int32),
        context.cu_seq_lens_k.to(torch.int32).clone(),
        context.max_seqlen_q,
        context.max_seqlen_k,
        softmax_scale=module.scaling,
        causal=True,
        window_size=sliding_window,
    )
    if isinstance(attn_output, tuple):
        attn_output = attn_output[0]
    out[:num_tokens].copy_(attn_output)
    if num_tokens < out.shape[0]:
        out[num_tokens:].zero_()


def graphed_paged_attention_forward(
    module: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Paged attention that runs as an eager break inside a prefill graph, and as the stock paged forward otherwise.

    The output buffer is allocated here, inside the graph segment, so the break writes into graph
    memory and the segment after it reads the result without a copy.
    """

    if _ACTIVE_CONTEXT.get() is None:
        return paged_attention_forward(module, q, k, v, attention_mask, **kwargs)
    out = q.new_empty((q.shape[2], q.shape[1], q.shape[3]))
    _attention_break(module, q, k, v, out)
    return out, None


def install_graphed_paged_attention(attn_implementation: str) -> None:
    """Route ``attn_implementation`` (a ``paged|...`` flash implementation) through the graphed forward.

    Idempotent; the stock forward keeps serving every call made outside a prefill graph context.
    """

    if not attn_implementation.startswith("paged|"):
        raise ValueError(f"prefill graphs need a paged attention implementation, got {attn_implementation!r}")
    if ALL_ATTENTION_FUNCTIONS.get_interface(attn_implementation, None) is not paged_attention_forward:
        registered = ALL_ATTENTION_FUNCTIONS.get_interface(attn_implementation, None)
        if registered is graphed_paged_attention_forward:
            return
        raise ValueError(f"{attn_implementation!r} is not served by the paged flash forward, got {registered!r}")
    ALL_ATTENTION_FUNCTIONS[attn_implementation] = graphed_paged_attention_forward
