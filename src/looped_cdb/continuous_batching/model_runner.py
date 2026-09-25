# Copyright 2026 The HuggingFace Inc. team
# SPDX-License-Identifier: Apache-2.0
# Modified for looped-model serving and continuous depth batching.
# See THIRD_PARTY_NOTICES.md and LICENSES/Apache-2.0.txt.

"""Model execution for continuous batching.

This module mirrors Hugging Face's continuous batching runner while keeping the
implementation focused on greedy generation, paged attention inputs, async
I/O compatibility, and CUDA graph hooks.

Reference:
https://github.com/huggingface/transformers/blob/main/src/transformers/generation/continuous_batching/model_runner.py
"""

import inspect
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from looped_cdb.benchmarks import nvtx

from .cache import PagedAttentionCache
from .config import ContinuousBatchingConfig
from .input_outputs import ContinuousBatchingIOs
from .prefill_graph import (
    BreakableCapture,
    BreakableGraph,
    install_graphed_paged_attention,
    prefill_graph_buckets,
    prefill_graph_context,
)
from .requests import RequestStatus, logger
from .utils import (
    create_warmup_future_states,
    create_warmup_states,
    decode_graph_buckets,
    gc_paused,
    pad_to_bucket,
)


@dataclass(frozen=True)
class WarmupReport:
    """What a warm-up captured and what it cost.

    The memory figures are deltas over the warm-up, as the caching allocators see them: device
    ``memory_allocated`` (the graph pools and the per-bucket static buffers the captures pin down) and
    ``memory_reserved`` (that plus the cache the eager pre-passes leave behind), and on the host the
    pinned pool's ``allocated_bytes.current``, which is the pinned memory taken from the OS (active and
    cached blocks, each rounded up to a power of two) rather than the bytes the host buffers hold. A
    delta can be negative when a warm-up releases more than it takes. A warm-up on a runner whose graphs
    already exist captures nothing and reports the same counts, but it still replays one launch per
    bucket, so its seconds are the replay pass.
    """

    decode_graphs: int
    decode_seconds: float
    prefill_graphs: int
    prefill_seconds: float
    device_allocated_bytes: int
    device_reserved_bytes: int
    pinned_host_bytes: int

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)

    def summary(self) -> str:
        mib = 1 << 20
        return (
            f"Warmup captured {self.decode_graphs} decode graphs in {self.decode_seconds:.1f}s and "
            f"{self.prefill_graphs} prefill graphs in {self.prefill_seconds:.1f}s; "
            f"device {self.device_allocated_bytes / mib:+.0f} MiB allocated "
            f"({self.device_reserved_bytes / mib:+.0f} MiB reserved), "
            f"pinned host {self.pinned_host_bytes / mib:+.0f} MiB"
        )


def _memory_snapshot() -> tuple[int, int, int]:
    """Current device allocated / reserved bytes and pinned host bytes held by the caching allocators."""

    if not torch.cuda.is_available():
        return 0, 0, 0
    pinned = int(torch.cuda.host_memory_stats().get("allocated_bytes.current", 0))
    return torch.cuda.memory_allocated(), torch.cuda.memory_reserved(), pinned


class ModelRunner:
    """Continuous batching entry point for running the model on the device.

    The implementation keeps the same high-level shape as Hugging Face's runner: it retrieves the
    static tensors prepared by ``ContinuousBatchingIOs``, optionally carries over async tokens, runs the model, and
    writes sampled token IDs into ``output_ids``. Sampling is intentionally greedy-only for now.
    """

    def __init__(
        self,
        engine_config: ContinuousBatchingConfig,
        inputs_and_outputs: ContinuousBatchingIOs,
        cache: PagedAttentionCache,
        do_sample: bool = False,
    ) -> None:
        if do_sample:
            raise NotImplementedError("Native continuous batching currently supports greedy decoding only")

        self.engine_config = engine_config
        self.inputs_and_outputs = inputs_and_outputs
        self.cache = cache
        self.do_sample = do_sample
        self.use_cuda_graph_decode = bool(self.engine_config.use_cuda_graph)
        # ``use_cuda_graph`` is the master switch; the prefill graphs are its second stage.
        self.use_cuda_graph_prefill = self.use_cuda_graph_decode and bool(self.engine_config.use_cuda_graph_prefill)

        if (self.use_cuda_graph_decode or self.use_cuda_graph_prefill) and not torch.cuda.is_available():
            raise RuntimeError("CUDA graphs require CUDA")
        if self.use_cuda_graph_prefill:
            # Installed before any capture can happen, so a prefill graph always breaks at attention.
            # A model served through a paged flash implementation breaks its graph there; any other
            # model (the tests' attention-free stand-ins) captures as one segment.
            attn_implementation = getattr(self.cache.config, "_attn_implementation", None)
            if isinstance(attn_implementation, str) and attn_implementation.startswith("paged|"):
                install_graphed_paged_attention(attn_implementation)

        self.gather_sampled_rows = False
        # Decode batches pad to a batch bucket, varlen batches to a token bucket; one graph each.
        self.pad_inputs = self.use_cuda_graph_decode
        self.decode_graph_buckets = decode_graph_buckets(self._decode_token_cap())
        self.prefill_graph_buckets = (
            prefill_graph_buckets(self.cache.max_num_batched_tokens) if self.use_cuda_graph_prefill else []
        )
        # One breakable graph per token bucket, all captured at warm-up.
        self.prefill_graphs: dict[int, BreakableGraph] = {}
        self.prefill_graph_hits = 0
        self.prefill_graph_captures = 0
        self.warmup_report: WarmupReport | None = None
        # A plain pool handle, not a torch.cuda.MemPool: pool memory is owned by the graphs
        # captured into it, so a dead runner leaves no destructor behind for the GC to run.
        self.graph_pool_id = (
            torch.cuda.graph_pool_handle() if self.use_cuda_graph_decode or self.use_cuda_graph_prefill else None
        )

    def maybe_pad_inputs(self, num_q_tokens: int, max_kv_read: int, use_decode_fast_path: bool) -> tuple[int, int]:
        """Pad a batch to its graph shape: decode batches to a batch bucket, varlen batches to a token bucket."""

        if use_decode_fast_path:
            if not self.pad_inputs:
                return num_q_tokens, max_kv_read
            return pad_to_bucket(num_q_tokens, self.decode_graph_buckets), 0
        if not self.use_cuda_graph_prefill:
            return num_q_tokens, max_kv_read
        return pad_to_bucket(num_q_tokens, self.prefill_graph_buckets), max_kv_read

    def pads_batch(self, use_decode_fast_path: bool) -> bool:
        """Whether :meth:`maybe_pad_inputs` pads this kind of batch, so its tensors are finalized padded."""

        return self.pad_inputs if use_decode_fast_path else self.use_cuda_graph_prefill

    def _decode_token_cap(self) -> int:
        """The widest decode batch the scheduler can produce, which bounds decode padding and warmup shapes.

        One query token per resident request, so the resident cap is the batch cap. The engine
        resolves that cap against ``max_num_batched_tokens``, which sizes the buffers this pads into,
        so a padded batch always fits.
        """

        return self.engine_config.max_num_seqs

    def compute_batch(self, model: nn.Module, batch_data: dict[str, Any]) -> None:
        """Run the forward pass and write next-token IDs into the static output tensor."""

        self._ensure_cache_policy_kwargs(batch_data)
        carry_over_ids, prev_output_ids, output_ids = self.inputs_and_outputs.get_cb_kwargs()
        compute_stream = self.inputs_and_outputs.compute_stream
        use_block_table = self.inputs_and_outputs.use_block_table
        forward_fn, use_cuda_graph = self._get_forward_fn(use_block_table=use_block_table)

        if use_cuda_graph and not use_block_table:
            self._run_prefill_graph(forward_fn, model, batch_data, carry_over_ids, prev_output_ids, output_ids)
            return
        if not use_cuda_graph:
            maybe_stream = torch.cuda.stream(compute_stream) if compute_stream is not None else nullcontext()
            with maybe_stream:
                forward_fn(model, batch_data, carry_over_ids, prev_output_ids, output_ids)
            return

        graph = self.inputs_and_outputs.get_graph()
        if graph is not None:
            with nvtx.range("runner.graph_replay"):
                if compute_stream is None:
                    graph.replay()
                else:
                    with torch.cuda.stream(compute_stream):
                        graph.replay()
            return

        if compute_stream is None:
            raise RuntimeError("CUDA graph capture requires a CUDA compute stream")
        args = (model, batch_data, carry_over_ids, prev_output_ids, output_ids)
        with nvtx.range("runner.graph_capture"):
            self._capture_graph(forward_fn, compute_stream, *args)

    def _get_forward_fn(self, use_block_table: bool) -> tuple[Callable[..., None], bool]:
        """Return the forward function and whether this batch should use CUDA graphs."""

        use_cuda_graph = self.use_cuda_graph_decode if use_block_table else self.use_cuda_graph_prefill
        return self._forward_process_and_sample, use_cuda_graph

    def _ensure_cache_policy_kwargs(self, batch_data: dict[str, Any]) -> None:
        """Keep model-side cache indexing consistent with the allocated cache layout."""

        batch_data["kv_policy"] = self.cache.kv_policy
        batch_data["kv_slots_per_layer"] = self.cache.kv_slots_per_layer

    def _capture_graph(
        self,
        forward_fn: Callable[..., None],
        compute_stream: torch.cuda.Stream,
        *args: object,
    ) -> None:
        """Capture and store a CUDA graph for the current static tensor shapes."""

        with gc_paused():
            with torch.cuda.stream(compute_stream):
                forward_fn(*args)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                graph, stream=compute_stream, pool=self.graph_pool_id, capture_error_mode="thread_local"
            ):
                forward_fn(*args)
        self.inputs_and_outputs.set_graph(graph)

    def _run_prefill_graph(
        self,
        forward_fn: Callable[..., None],
        model: nn.Module,
        batch_data: dict[str, Any],
        carry_over_ids: torch.Tensor,
        prev_output_ids: torch.Tensor,
        output_ids: torch.Tensor,
    ) -> None:
        """Replay the bucket's breakable prefill graph, capturing it first if warm-up did not.

        A capture records the segments without executing them, so its eager breaks run on
        unexecuted inputs and write garbage K/V at the launch's write indices; the replay that
        follows every capture rewrites those rows and produces the launch's real outputs. The
        eager pre-pass before the capture samples into ``output_ids``, which the carry-over
        scatter reads as the previous batch's tokens, so the replay sees the pre-pass's input
        again.
        """

        io = self.inputs_and_outputs
        compute_stream = io.compute_stream
        if compute_stream is None:
            raise RuntimeError("prefill CUDA graphs require a CUDA compute stream")
        num_kept = self._num_logits_rows()
        if self.gather_sampled_rows and io.true_num_logits > num_kept:
            raise RuntimeError(
                f"prefill launch samples {io.true_num_logits} rows, more than the {num_kept} its graph gathers"
            )
        bucket = io.device_buffers.num_q_tokens
        graph = self.prefill_graphs.get(bucket)
        with torch.cuda.stream(compute_stream), prefill_graph_context(io.prefill_graph_context()):
            if graph is not None:
                self.prefill_graph_hits += 1
                with nvtx.range("runner.prefill_graph_replay"):
                    graph.replay()
                return
            self.prefill_graph_captures += 1
            logger.debug(f"Capturing prefill graph for {bucket = }")
            with nvtx.range("runner.prefill_graph_capture"):
                graph = BreakableGraph()
                previous_tokens = output_ids.clone()
                with gc_paused():
                    forward_fn(model, batch_data, carry_over_ids, prev_output_ids, output_ids)
                    torch.cuda.synchronize()
                    with BreakableCapture(graph, pool=self.graph_pool_id):
                        forward_fn(model, batch_data, carry_over_ids, prev_output_ids, output_ids)
                output_ids.copy_(previous_tokens)
            self.prefill_graphs[bucket] = graph
            with nvtx.range("runner.prefill_graph_replay"):
                graph.replay()

    def _num_logits_rows(self) -> int:
        """Rows the varlen LM head gathers: the sampled rows, padded to a fixed count under prefill graphs.

        A graph is one shape, so the gather cannot follow the launch's sampled-row count. Each launch
        samples at most one row per resident request, and never more rows than query tokens. The
        scheduler keeps a prefill launch within the resident cap (the admissible prompts are bounded
        by the free residency slots, the already-prefilling ones are resident), and
        :meth:`_run_prefill_graph` raises if a launch ever samples more rows than this keeps.
        """

        io = self.inputs_and_outputs
        if self.use_cuda_graph_prefill:
            return min(self._decode_token_cap(), io.device_buffers.num_q_tokens)
        return io.true_num_logits

    def configure_sampled_row_gather(self, model: nn.Module) -> None:
        """Restrict varlen LM-head work to the sampled rows when the model supports it.

        A prompt-prefill batch samples only the last position of each finishing prompt, yet the
        monolithic forward computes vocabulary logits for every query token. Models exposing HF's
        ``logits_to_keep`` (index form) can gather the sampled rows before the LM head instead. The
        decode fast path samples every row, so it is unaffected either way.
        """

        self.gather_sampled_rows = "logits_to_keep" in inspect.signature(model.forward).parameters

    def _forward_process_and_sample(
        self,
        model: nn.Module,
        batch_data: dict[str, Any],
        carry_over_ids: torch.Tensor,
        prev_output_ids: torch.Tensor,
        output_ids: torch.Tensor,
    ) -> None:
        """Run model forward, greedily select next tokens, and write the static outputs."""

        self.inputs_and_outputs.carry_over_tokens(batch_data["input_ids"], carry_over_ids, prev_output_ids)
        logits_indices = batch_data["logits_indices"]
        if self.gather_sampled_rows and not self.inputs_and_outputs.use_block_table:
            num_kept = self._num_logits_rows()
            batch_data = {**batch_data, "logits_to_keep": logits_indices[:num_kept].to(torch.long)}
            self._sample_pregathered(model(**batch_data).logits, output_ids)
            return
        self._sample(model(**batch_data).logits, logits_indices, output_ids)

    def _sample_pregathered(self, scores: torch.Tensor, output_ids: torch.Tensor) -> None:
        """Greedily select next tokens from logits the model already restricted to the sampled rows."""

        next_tokens = torch.argmax(scores[0], dim=-1)
        output_ids[0, : next_tokens.size(0)].copy_(next_tokens.to(dtype=output_ids.dtype))

    def _sample(self, scores: torch.Tensor, logits_indices: torch.Tensor, output_ids: torch.Tensor) -> None:
        """Greedily select next tokens in the model dtype (an fp32 upcast cannot change the argmax)."""

        scores_2d = scores[0]
        next_tokens = torch.argmax(scores_2d, dim=-1)

        num_tokens = next_tokens.size(0)
        indices = logits_indices[:num_tokens].to(dtype=torch.long)
        next_tokens = next_tokens[indices]
        output_ids[0, :num_tokens].copy_(next_tokens.to(dtype=output_ids.dtype))

    @torch.no_grad()
    def warmup(self, model: nn.Module, model_kwargs: dict[str, Any] | None = None) -> None:
        """Pre-capture the decode and prefill CUDA graphs when graphing is enabled.

        The captures, their seconds and the memory they pinned down are summarized in
        :attr:`warmup_report`. ``model_kwargs`` are the kwargs the run will decode with, and every decode batch
        merges them into its forward. A replay ignores them, so a graph captured without
        them would stand in for a forward it never saw; warming with the same values keeps
        capture and replay the same forward.
        """

        if not (self.use_cuda_graph_decode or self.use_cuda_graph_prefill):
            return

        torch.cuda.synchronize()
        allocated_before, reserved_before, pinned_before = _memory_snapshot()
        decode_seconds = self._warmup_decode_graphs(model, dict(model_kwargs or {}))
        prefill_seconds = self._warmup_prefill_graphs(model, self._prefill_model_kwargs(model_kwargs))
        torch.cuda.synchronize()
        allocated_after, reserved_after, pinned_after = _memory_snapshot()

        self.inputs_and_outputs.clear_batch_history()
        self.warmup_report = WarmupReport(
            decode_graphs=self._num_decode_graphs(),
            decode_seconds=decode_seconds,
            prefill_graphs=len(self.prefill_graphs),
            prefill_seconds=prefill_seconds,
            device_allocated_bytes=allocated_after - allocated_before,
            device_reserved_bytes=reserved_after - reserved_before,
            pinned_host_bytes=pinned_after - pinned_before,
        )
        logger.info(self.warmup_report.summary())

    def _num_decode_graphs(self) -> int:
        """Decode-side graphs the runner holds; the depth engine counts its three stages."""

        return len(self.inputs_and_outputs.device_buffers.graphs)

    def _prefill_model_kwargs(self, model_kwargs: dict[str, Any] | None) -> dict[str, Any]:
        """The kwargs the engine merges into a prefill forward, so capture and replay run the same forward."""

        return dict(model_kwargs or {})

    def _warmup_prefill_graphs(self, model: nn.Module, model_kwargs: dict[str, Any]) -> float:
        """Capture the breakable prefill graph of every token bucket on fake prompts."""

        if not self.use_cuda_graph_prefill:
            return 0.0
        start = time.perf_counter()
        for bucket in self.prefill_graph_buckets:
            if not self._warmup_one_prefill_bucket(model, bucket, model_kwargs):
                logger.warning(
                    f"Cache too small to warm up the prefill graph at {bucket = } tokens; "
                    "this bucket and wider ones capture inside the run."
                )
                break
        return time.perf_counter() - start

    def _warmup_one_prefill_bucket(self, model: nn.Module, bucket: int, model_kwargs: dict[str, Any]) -> bool:
        """Drive one bucket of fake prompts through the prefill path, so its graph is captured.

        The bucket is filled with prompts no longer than ``max_model_len``; the padded rows beyond the
        prompts exercise the same path a real launch pads with. Returns ``False`` when the cache cannot
        hold the prompts.
        """

        max_len = self.engine_config.max_model_len or bucket
        query_lengths = [max_len] * (bucket // max_len)
        if bucket % max_len:
            query_lengths.append(bucket % max_len)
        future_states = create_warmup_states(query_lengths, RequestStatus.PREFILLING, 0, self.cache)
        try:
            if len(future_states) != len(query_lengths):
                return False
            self.inputs_and_outputs.prepare_batch_tensors(
                future_states, use_decode_fast_path=False, num_q_tokens=bucket, max_kv_read=0
            )
            batch_data = self.inputs_and_outputs.get_model_kwargs(use_padding=True)
            batch_data.update(model_kwargs)
            self.compute_batch(model, batch_data)
            if self.inputs_and_outputs.compute_stream is not None:
                self.inputs_and_outputs.compute_stream.synchronize()
        finally:
            for future_state in future_states:
                self.cache.free_blocks(future_state.state.request_id)
        return True

    def _warmup_decode_graphs(self, model: nn.Module, model_kwargs: dict[str, Any]) -> float:
        """Capture the decode fast path at every batch bucket up to the decode cap."""

        if not self.use_cuda_graph_decode:
            return 0.0
        return sum(
            self.run_one_warmup(model=model, num_requests=bucket, model_kwargs=model_kwargs)
            for bucket in self.decode_graph_buckets
        )

    def run_one_warmup(self, model: nn.Module, num_requests: int, model_kwargs: dict[str, Any] | None = None) -> float:
        """Warm up one decode-fast-path shape."""

        query_length = 1
        max_kv_read = self.cache.block_size
        logger.debug(f"Warming up decode fast path for {num_requests = }.")

        future_states = create_warmup_future_states(
            num_requests, RequestStatus.DECODING, query_length, max_kv_read, self.cache
        )
        if not future_states:
            logger.warning(
                f"Failed to warm up: no blocks allocated for {num_requests = }, {query_length = }, {max_kv_read = }."
            )
            return 0.0
        if len(future_states) != num_requests:
            logger.debug(
                f"Warmup allocated {len(future_states)} fake requests for requested {num_requests}; "
                "the remaining rows will be padding."
            )

        padded_q, padded_kv = self.maybe_pad_inputs(
            num_q_tokens=num_requests, max_kv_read=max_kv_read, use_decode_fast_path=True
        )

        start = time.perf_counter()
        try:
            self.inputs_and_outputs.prepare_batch_tensors(
                future_states,
                use_decode_fast_path=True,
                num_q_tokens=padded_q,
                max_kv_read=padded_kv,
            )
            batch_data = self.inputs_and_outputs.get_model_kwargs(use_padding=True)
            self._ensure_cache_policy_kwargs(batch_data)
            # The same merge every decode batch does, so the captured forward is the one replayed.
            batch_data.update(model_kwargs or {})
            carry_over_ids, prev_output_ids, output_ids = self.inputs_and_outputs.get_cb_kwargs()
            forward_fn, use_cuda_graph = self._get_forward_fn(use_block_table=self.inputs_and_outputs.use_block_table)
            forward_fn_args = (model, batch_data, carry_over_ids, prev_output_ids, output_ids)
            if use_cuda_graph:
                compute_stream = self.inputs_and_outputs.compute_stream
                if compute_stream is None:
                    raise RuntimeError("CUDA graph warmup requires a CUDA compute stream")
                self._capture_graph(forward_fn, compute_stream, *forward_fn_args)
            else:
                maybe_stream = (
                    torch.cuda.stream(self.inputs_and_outputs.compute_stream)
                    if self.inputs_and_outputs.compute_stream is not None
                    else nullcontext()
                )
                with maybe_stream:
                    forward_fn(*forward_fn_args)
            duration = time.perf_counter() - start
            logger.debug(f"Warmup completed in {duration:.2f}s")
        except Exception as e:
            duration = 0.0
            logger.warning(f"Failed to warm up: {e}.\nGraph pool may fragment and OOM under load.")
        finally:
            for fs in future_states:
                self.cache.free_blocks(fs.state.request_id)
        return duration
