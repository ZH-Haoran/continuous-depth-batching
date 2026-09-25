"""Model execution for continuous depth batching.

Extends the continuous batching runner (``looped_cdb.continuous_batching.model_runner``),
whose full-depth varlen forward serves prompt prefill unchanged. Decode runs instead as
staged prelude/recurrent/coda launches that this module owns, each with its own static
buffers, CUDA graphs, and warmup captures.
"""

import time
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from looped_cdb.benchmarks import nvtx
from looped_cdb.continuous_batching.model_runner import ModelRunner as CBModelRunner
from looped_cdb.continuous_batching.utils import (
    CudaGraphBuffer,
    create_warmup_future_states,
    gc_paused,
    pad_to_bucket,
)

from .cache import PagedAttentionCache
from .config import ContinuousDepthBatchingConfig
from .exit_policy import ExitPolicy
from .input_outputs import ContinuousDepthBatchingIOs
from .model_adapter import CDBModelAdapter
from .requests import (
    TMP_TOKEN_ID,
    DepthWorkItem,
    FutureRequestState,
    PendingGateResult,
    RequestStatus,
    logger,
)

# Bound on in-flight coda batches. The engine stops scheduling codas once this many results
# are pending, and the rotating coda token buffers hold exactly this many turns, so a turn
# is never restaged before its D2H read is enqueued. Deepening the pipeline and sizing the
# buffers are one decision.
#
# Above one because a coda batch can be staged before the previous one's tokens have been read
# back to the host.
CODA_PIPELINE_DEPTH = 2


@dataclass
class RecurrentStageBuffers:
    """Static tensors for one CDB recurrent-stage graph bucket."""

    hidden_in: torch.Tensor
    hidden_out: torch.Tensor
    exit_signals: torch.Tensor
    position_ids: torch.Tensor
    recurrent_steps: torch.Tensor
    cache_position: torch.Tensor
    cu_seq_lens_q: torch.Tensor
    cu_seq_lens_k: torch.Tensor
    block_table: torch.Tensor
    slot_indices: torch.Tensor
    empty_index: torch.Tensor
    host_position_ids: torch.Tensor
    host_recurrent_steps: torch.Tensor
    host_cache_position: torch.Tensor
    host_cu_seq_lens_q: torch.Tensor
    host_cu_seq_lens_k: torch.Tensor
    host_block_table: torch.Tensor
    host_slot_indices: torch.Tensor
    metadata_copied_event: Any | None = None


@dataclass
class StageAttentionBuffers:
    """Paged-attention metadata for a one-token-per-request stage batch.

    The prelude and coda of a prelude/core/coda model contain attention and
    write their own KV, so those stages need the same cache arguments the
    recurrent stage receives. Fully looped models have no such stages and ignore
    these. The buffers mirror the recurrent stage's layout; they are separate so
    the recurrent CUDA-graph buckets keep their own static addresses.
    """

    cu_seq_lens_q: torch.Tensor
    cu_seq_lens_k: torch.Tensor
    block_table: torch.Tensor
    empty_index: torch.Tensor
    position_ids: torch.Tensor
    host_cu_seq_lens_q: torch.Tensor
    host_cu_seq_lens_k: torch.Tensor
    host_block_table: torch.Tensor
    host_position_ids: torch.Tensor
    #: Recorded after each batch's host-to-device copies are enqueued. Two batches
    #: of the same size share one buffer set, so the next batch's host writes must
    #: wait for the previous copies to land before overwriting pinned memory.
    copied_event: torch.cuda.Event | None


@dataclass
class PreludeGraphBuffers:
    """Static device tensors for one CUDA-graphed prelude-stage bucket.

    ``input_ids`` and ``position_ids`` are written eagerly before each replay (from the
    pinned staging buffers, or gathered from a staged coda's device tokens), so one graph
    per bucket serves both prelude entry points. ``hidden_out`` is allocated at capture
    time, once the stage's output width is known. Padding rows keep whatever ids they
    last held: the buffers are zero-initialized and only ever overwritten with real
    in-vocab ids, so a stale padding row is always a valid lookup whose output no one
    reads.
    """

    input_ids: torch.Tensor
    position_ids: torch.Tensor
    hidden_out: torch.Tensor | None = None


@dataclass
class CodaGraphBuffers:
    """Static device tensors for one CUDA-graphed coda-stage bucket.

    The coda's sampled tokens are read by the off-lane D2H copy while later coda
    batches pipeline behind them on the compute stream, so ``tokens_out`` rotates
    between two buffers, each with its own captured graph. ``reuse_events`` gates a
    buffer's overwrite on the previous D2H copy that read it.
    """

    slot_indices: torch.Tensor
    host_slot_indices: torch.Tensor
    hidden_in: torch.Tensor | None = None
    tokens_out: list[torch.Tensor] = field(default_factory=list)
    reuse_events: list[Any] = field(default_factory=list)
    turn: int = 0


@dataclass
class PreludeStageBuffers:
    """Reusable pinned host buffers for staging prelude-batch metadata to the device.

    ``copied_event`` guards buffer reuse: it is recorded after each batch's H2D copies and awaited
    before the set is next overwritten. Because the record lands after the prelude forward and bank
    write enqueued on the same stream, the wait covers that batch's prelude compute, not just its
    copies; the runner therefore rotates two buffer sets, so the awaited event is one prelude older
    than the latest and the wait stays off the hot path. That matters most for the refill loop,
    which runs one per coda and one per prefill batch, so back-to-back preludes would otherwise
    make each one wait on its immediate predecessor. Building device tensors from Python lists
    instead would synchronize the compute stream on every prelude.
    """

    host_input_ids: torch.Tensor
    host_position_ids: torch.Tensor
    host_slot_indices: torch.Tensor
    host_gather_indices: torch.Tensor
    copied_event: Any | None = None


@dataclass
class PendingCodaResult:
    """Sampled coda tokens for one coda batch: device-visible immediately, host-visible when ready.

    ``device_tokens`` lets the prelude re-enter the successor token without waiting for the
    host copy: the coda is staged on the compute stream, so the re-entry gather that follows it
    there is ordered by program order alone. ``eager_reentries`` marks which items the engine
    re-entered that way, so consumption applies their tokens as bookkeeping only.
    """

    items: list[DepthWorkItem]
    host_tokens: torch.Tensor
    device_tokens: torch.Tensor | None = None
    ready_event: Any | None = None
    keepalive: Any | None = None
    eager_reentries: list[bool] | None = None

    def is_ready(self) -> bool:
        if self.ready_event is None:
            return True
        query = getattr(self.ready_event, "query", None)
        return bool(query()) if query is not None else False


@dataclass
class PendingPrefillResult:
    """Sampled first decode tokens of one prefill batch: device-visible immediately, host-visible when ready.

    The prompt-to-decode counterpart of :class:`PendingCodaResult`: ``device_tokens`` lets the
    prelude enter each prompt's first decode token before the host has seen it, and
    ``eager_entries`` marks the new-token indices entered that way, so consumption applies their
    tokens as bookkeeping only.
    """

    futures: list[FutureRequestState]
    host_tokens: torch.Tensor
    device_tokens: torch.Tensor | None = None
    ready_event: Any | None = None
    eager_entries: list[bool] | None = None

    def is_ready(self) -> bool:
        if self.ready_event is None:
            return True
        query = getattr(self.ready_event, "query", None)
        return bool(query()) if query is not None else False


class ModelRunner(CBModelRunner):
    """Depth-batched model execution: the CB runner's prefill path plus staged recurrent decode."""

    def __init__(
        self,
        engine_config: ContinuousDepthBatchingConfig,
        inputs_and_outputs: ContinuousDepthBatchingIOs,
        cache: PagedAttentionCache,
        model_adapter: CDBModelAdapter,
        exit_policy: ExitPolicy,
        do_sample: bool = False,
    ) -> None:
        super().__init__(engine_config, inputs_and_outputs, cache, do_sample)
        self.model_adapter = model_adapter
        self.exit_policy = exit_policy
        # Prelude/coda stages of a prelude/core/coda model contain attention; fully
        # looped models run a lookup for the prelude and a matmul for the coda.
        self.model_adapter_stages_use_attention = bool(getattr(model_adapter, "stages_use_attention", False))
        self.recurrent_stage_buffers: dict[tuple[int, torch.dtype], RecurrentStageBuffers] = {}
        # Keyed by (stage, batch size): reuse of a set waits on its copied_event, so a set
        # shared between the prelude and coda stages would stall the host on the previous
        # stage's in-flight copy at every alternation between them.
        self.stage_attention_buffers: dict[tuple[str, int], StageAttentionBuffers] = {}
        self.prelude_stage_buffer_pool: list[PreludeStageBuffers | None] = [None, None]
        self.prelude_stage_buffer_turn = 0
        self.recurrent_graphs: CudaGraphBuffer = CudaGraphBuffer()
        self.recurrent_graph_hits = 0
        self.recurrent_graph_captures = 0
        self.prelude_graphs: CudaGraphBuffer = CudaGraphBuffer()
        self.coda_graphs: CudaGraphBuffer = CudaGraphBuffer()
        self.prelude_graph_buffers: dict[int, PreludeGraphBuffers] = {}
        self.coda_graph_buffers: dict[int, CodaGraphBuffers] = {}
        self.stage_graph_hits = 0
        self.stage_graph_captures = 0
        # The coda turn used by the most recent graphed coda launch; its reuse event is
        # recorded once the caller has enqueued the D2H copy that reads the buffer.
        self._last_coda_turn: tuple[CodaGraphBuffers, int] | None = None
        self.hidden_state_bank: torch.Tensor | None = None
        self.hidden_state_bank_meta: tuple[int, torch.dtype] | None = None
        self.free_hidden_slots: list[int] = []
        self.next_hidden_slot = 0
        self.pending_gate_copy_keepalives: list[tuple[torch.Tensor, object, torch.Tensor]] = []
        # The prelude and coda stages are graphed whenever decode graphs are on: they are
        # decode-side launches like the recurrent stage, and for prelude/coda models they
        # are eager multi-kernel transformer stages whose launch overhead a graph removes.
        # All captures share one graph pool: every graph copies its result into a static
        # buffer allocated outside the capture region, so no pool block is ever read
        # after another graph replays over it.
        self.use_stage_cuda_graphs = self.use_cuda_graph_decode and self.cache.device.type == "cuda"

    def _prefill_model_kwargs(self, model_kwargs: dict[str, Any] | None) -> dict[str, Any]:  # noqa: ARG002
        """The depth engine's prefill forward takes no engine kwargs; the gate runs in the staged decode."""

        return {}

    def _num_decode_graphs(self) -> int:
        return len(self.prelude_graphs) + len(self.recurrent_graphs) + len(self.coda_graphs)

    def _warmup_decode_graphs(self, model: nn.Module, model_kwargs: dict[str, Any]) -> float:  # noqa: ARG002
        """Capture the prelude, recurrent, and coda graphs at every bucket a decode launch can land in.

        These three are the depth engine's decode graphs, so the inherited fast path is not
        captured: it is never launched, and its shapes would only hold graph-pool memory.

        The recurrent key also carries the KV slot, and only slot 0 is captured here; a
        depth-indexed layout captures the other slots inside the run.
        """

        if not self.use_stage_cuda_graphs:
            return 0.0
        duration = 0.0
        for bucket_size in self.decode_graph_buckets:
            bucket_duration = self._warmup_one_stage_bucket(bucket_size, model_kwargs)
            if bucket_duration is None:
                # Each bucket frees its blocks before the next runs, so demand only grows: no
                # wider bucket can fit either.
                logger.warning(
                    f"Cache too small to warm up CDB stage graphs at {bucket_size = }; "
                    "this bucket and wider ones capture inside the run."
                )
                break
            duration += bucket_duration
        return duration

    def _warmup_one_stage_bucket(self, bucket_size: int, model_kwargs: dict[str, Any]) -> float | None:
        """Drive one bucket through all three stages on fake requests, so each captures its graph.

        Returns ``None`` when the cache cannot hold the bucket, which is the caller's signal to
        stop. The pass repeats ``CODA_PIPELINE_DEPTH`` times because the coda graph key carries
        the rotating output turn, so each turn holds a separate capture; prelude and recurrent
        are captured on the first pass and replayed on the rest.

        The fake requests read a single KV entry, so one block each is enough. Sizing them like
        the full-depth fast path, which reads a padded window, would double the demand and put
        the widest buckets out of reach on a small cache.
        """

        future_states = create_warmup_future_states(bucket_size, RequestStatus.DECODING, 1, 0, self.cache)
        if len(future_states) < bucket_size:
            for state in future_states:
                self.cache.free_blocks(state.state.request_id)
            return None
        items = [
            DepthWorkItem(state=state.state, token_id=0, token_position=0, recurrent_step=0) for state in future_states
        ]
        start = time.perf_counter()
        try:
            for _ in range(CODA_PIPELINE_DEPTH):
                self.compute_prelude_batch(items)
                self.compute_recurrent_batch(items, dict(model_kwargs))
                self.compute_coda_batch(items)
            duration = time.perf_counter() - start
        except Exception as exc:
            duration = 0.0
            logger.warning(f"Failed to warm up CDB stage graphs at {bucket_size = }: {exc}. Capture moves to the run.")
        finally:
            self.release_hidden_slots(items)
            for state in future_states:
                self.cache.free_blocks(state.state.request_id)
        return duration

    def stage_prefill_batch(
        self,
        model: nn.Module,
        requests: list[FutureRequestState],
        *,
        num_q_tokens: int,
        max_kv_read: int,
        allow_async: bool = True,
    ) -> PendingPrefillResult:
        """Run one full-depth monolithic prefill with an asynchronous token readback.

        The host does not block on the forward or the sampling: the sampled first decode tokens
        are snapshotted on the compute stream (program order protects the snapshot from the next
        batch's sampler) and copied out on the D2H stream, so the host applies them one consume
        later as bookkeeping.
        Without CUDA streams, or with ``allow_async`` unset (the engine's synchronous mode),
        the tokens are read synchronously and the result is born ready.
        """

        with nvtx.range("cdb.prefill.stage"):
            num_q_tokens, max_kv_read = self.maybe_pad_inputs(num_q_tokens, max_kv_read, use_decode_fast_path=False)
            self.inputs_and_outputs.prepare_batch_tensors(
                requests_in_batch=requests,
                use_decode_fast_path=False,
                num_q_tokens=num_q_tokens,
                max_kv_read=max_kv_read,
            )
        with nvtx.range("cdb.prefill.h2d"):
            batch_data = self.inputs_and_outputs.get_model_kwargs(
                use_padding=self.pads_batch(use_decode_fast_path=False)
            )
        with nvtx.range("cdb.prefill.launch"):
            self.compute_batch(model, batch_data)
        io = self.inputs_and_outputs
        futures = io.requests_in_batch
        num_new_tokens = sum(1 for future in requests if future.has_new_token)
        if num_new_tokens == 0:
            return PendingPrefillResult(futures=futures, host_tokens=torch.empty((0,), dtype=torch.int64))
        compute_stream = io.compute_stream
        if not allow_async or compute_stream is None or io.d2h_stream is None:
            new_tokens = io.consume_output_tokens(io.enqueue_output_copy(), num_new_tokens)
            host_tokens = torch.tensor(new_tokens, dtype=torch.int64)
            return PendingPrefillResult(futures=futures, host_tokens=host_tokens, device_tokens=host_tokens)
        host_tokens = torch.empty((num_new_tokens,), dtype=torch.int64, device="cpu", pin_memory=True)
        sampled_done = torch.cuda.Event()
        d2h_done = torch.cuda.Event()
        with torch.cuda.stream(compute_stream):
            device_tokens = io.device_buffers.output_ids[0, :num_new_tokens].detach().clone().to(torch.int64)
        compute_stream.record_event(sampled_done)
        io.d2h_stream.wait_event(sampled_done)
        with torch.cuda.stream(io.d2h_stream):
            host_tokens.copy_(device_tokens, non_blocking=True)
        io.d2h_stream.record_event(d2h_done)
        return PendingPrefillResult(
            futures=futures,
            host_tokens=host_tokens,
            device_tokens=device_tokens,
            ready_event=d2h_done,
        )

    def consume_prefill_result(self, pending: PendingPrefillResult) -> list[int]:
        """Wait for a pending prefill result and return its sampled first decode tokens."""

        if pending.ready_event is not None:
            with nvtx.range("cdb.prefill_wait"):
                pending.ready_event.synchronize()
        return pending.host_tokens.tolist()

    def compute_prelude_batch(
        self,
        items: list[DepthWorkItem],
        device_groups: Sequence[tuple[PendingCodaResult | PendingPrefillResult, list[int]]] = (),
    ) -> None:
        """Run the prelude stage for a depth-work batch.

        ``device_groups`` covers a prefix of ``items``: each group's input tokens are
        gathered on device from its staged result's sampled tokens at the given rows
        (in item order), so those tokens enter the prelude before the host has seen
        them. The staging forward and the gather share the compute stream, so program
        order alone carries the data dependency. The remaining items' ids come from
        ``item.token_id`` through the pinned staging row, and the whole batch runs as
        one prelude launch regardless of how its tokens were sourced.
        """

        if not items:
            return
        device = self.cache.device
        batch_size = len(items)
        num_device = sum(len(rows) for _, rows in device_groups)
        num_host = batch_size - num_device
        # Device-sourced items are staged with a placeholder id, so this pins the prefix contract:
        # a caller that grouped a suffix or interleaved the two sources would otherwise embed the
        # wrong tokens silently, since ``num_device`` alone decides where each source lands.
        assert all(item.token_id == TMP_TOKEN_ID for item in items[:num_device]), (
            "device_groups must cover a prefix of items"
        )
        stage = self._get_prelude_stage_buffers(batch_size)
        if num_host:
            stage.host_input_ids[:num_host] = torch.tensor(
                [item.token_id for item in items[num_device:]], dtype=torch.long
            )
        gather_offset = 0
        for result, rows in device_groups:
            if result.device_tokens is None:
                raise RuntimeError("Cannot gather prelude tokens from a staged result that carries none")
            stage.host_gather_indices[gather_offset : gather_offset + len(rows)] = torch.tensor(rows, dtype=torch.long)
            gather_offset += len(rows)
        compute_stream = self.inputs_and_outputs.compute_stream
        maybe_stream = torch.cuda.stream(compute_stream) if compute_stream is not None else nullcontext()
        with maybe_stream:
            non_blocking = device.type == "cuda"
            gathered_parts: list[torch.Tensor] = []
            if num_device:
                gather_indices = stage.host_gather_indices[:num_device].to(device, non_blocking=non_blocking)
                group_offset = 0
                for result, rows in device_groups:
                    assert result.device_tokens is not None
                    gathered_parts.append(
                        result.device_tokens.index_select(0, gather_indices[group_offset : group_offset + len(rows)])
                    )
                    group_offset += len(rows)
            bucket_size = self._stage_bucket_size(batch_size)
            if bucket_size is not None:
                graph_buffers = self._get_prelude_graph_buffers(bucket_size)
                input_row = graph_buffers.input_ids[0]
                row_offset = 0
                for part in gathered_parts:
                    input_row[row_offset : row_offset + part.numel()].copy_(part)
                    row_offset += part.numel()
                if num_host:
                    input_row[num_device:batch_size].copy_(stage.host_input_ids[:num_host], non_blocking=non_blocking)
                self._run_prelude_graphed(items, graph_buffers, bucket_size, stage)
                return
            if num_host:
                gathered_parts.append(stage.host_input_ids[:num_host].to(device, non_blocking=non_blocking))
            input_ids = gathered_parts[0] if len(gathered_parts) == 1 else torch.cat(gathered_parts)
            self._run_prelude(items, input_ids.view(1, batch_size), stage)

    def _run_prelude(
        self,
        items: list[DepthWorkItem],
        input_ids: torch.Tensor,
        stage: PreludeStageBuffers,
    ) -> None:
        """Embed one batch of decode tokens into freshly allocated hidden-bank slots.

        Runs inside the caller's stream context.
        """

        device = self.cache.device
        batch_size = len(items)
        non_blocking = device.type == "cuda"
        stage.host_position_ids[:batch_size] = torch.tensor([item.token_position for item in items], dtype=torch.long)
        position_ids = stage.host_position_ids[:batch_size].to(device, non_blocking=non_blocking).view(1, batch_size)
        hidden_states = self.model_adapter.prelude(
            input_ids=input_ids,
            position_ids=position_ids,
            **self.stage_attention_kwargs(items, stage="prelude"),
        )
        self._finish_prelude_batch(items, hidden_states, stage)

    def _run_prelude_graphed(
        self,
        items: list[DepthWorkItem],
        graph_buffers: PreludeGraphBuffers,
        bucket_size: int,
        stage: PreludeStageBuffers,
    ) -> None:
        """Replay (or capture) the prelude-stage graph for one bucket.

        Runs inside the caller's compute-stream context; the caller has already written
        this batch's token ids into the bucket's static ``input_ids``. Only the stage
        forward is captured: the position/metadata refresh runs eagerly before each
        replay, and the preloop gate, slot allocation, and hidden-bank write run eagerly
        on the graph's static output, so the bank may grow (reallocate) without
        invalidating captured graphs.
        """

        batch_size = len(items)
        non_blocking = self.cache.device.type == "cuda"
        stage.host_position_ids[:batch_size] = torch.tensor([item.token_position for item in items], dtype=torch.long)
        graph_buffers.position_ids[0, :batch_size].copy_(
            stage.host_position_ids[:batch_size], non_blocking=non_blocking
        )
        attention_kwargs = self.stage_attention_kwargs(items, stage="prelude", bucket_size=bucket_size)

        def forward() -> torch.Tensor:
            return self.model_adapter.prelude(
                input_ids=graph_buffers.input_ids,
                position_ids=graph_buffers.position_ids,
                **attention_kwargs,
            )

        graph_key = bucket_size
        graph = self.prelude_graphs.get_graph(graph_key)
        if graph is not None:
            self.stage_graph_hits += 1
            graph.replay()
        else:
            capture_stream = torch.cuda.current_stream()
            with gc_paused():
                warmup_out = forward()
                if graph_buffers.hidden_out is None:
                    graph_buffers.hidden_out = torch.empty_like(warmup_out)
                    torch._dynamo.mark_static_address(graph_buffers.hidden_out)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(
                    graph, stream=capture_stream, pool=self.graph_pool_id, capture_error_mode="thread_local"
                ):
                    graph_buffers.hidden_out.copy_(forward())
            self.prelude_graphs.set_graph(graph_key, graph)
            self.stage_graph_captures += 1
            graph.replay()

        assert graph_buffers.hidden_out is not None
        self._finish_prelude_batch(items, graph_buffers.hidden_out[:, :batch_size, :], stage)

    def _finish_prelude_batch(
        self,
        items: list[DepthWorkItem],
        hidden_states: torch.Tensor,
        stage: PreludeStageBuffers,
    ) -> None:
        """Route one prelude batch's hidden states into the bank and record slot assignments."""

        batch_size = len(items)
        non_blocking = self.cache.device.type == "cuda"
        self._assign_preloop_exit_steps(items, hidden_states)
        slots = self._allocate_hidden_slots(batch_size, hidden_states.size(-1), hidden_states.dtype)
        stage.host_slot_indices[:batch_size] = torch.tensor(slots, dtype=torch.long)
        slot_indices = stage.host_slot_indices[:batch_size].to(self.cache.device, non_blocking=non_blocking)
        assert self.hidden_state_bank is not None
        self.hidden_state_bank.index_copy_(1, slot_indices, hidden_states)
        if stage.copied_event is not None:
            # Records on the caller's stream, after this batch's H2D copies were enqueued on it.
            stage.copied_event.record()
        for item, slot in zip(items, slots, strict=True):
            item.hidden_slot = slot

    def _stage_bucket_size(self, batch_size: int) -> int | None:
        """Return the static stage-graph bucket for this launch, or None when stage graphs are off.

        A batch wider than the decode cap has no bucket and raises, like the recurrent stage.
        """

        if not self.use_stage_cuda_graphs:
            return None
        return pad_to_bucket(batch_size, self.decode_graph_buckets)

    def _get_prelude_graph_buffers(self, bucket_size: int) -> PreludeGraphBuffers:
        buffers = self.prelude_graph_buffers.get(bucket_size)
        if buffers is not None:
            return buffers
        device = self.cache.device
        # Zero-initialized so a padding row is always a valid (in-vocab, in-range)
        # lookup; per-batch writes only ever touch the real rows.
        buffers = PreludeGraphBuffers(
            input_ids=torch.zeros((1, bucket_size), dtype=torch.long, device=device),
            position_ids=torch.zeros((1, bucket_size), dtype=torch.long, device=device),
        )
        torch._dynamo.mark_static_address(buffers.input_ids)
        torch._dynamo.mark_static_address(buffers.position_ids)
        self.prelude_graph_buffers[bucket_size] = buffers
        return buffers

    def _assign_preloop_exit_steps(self, items: list[DepthWorkItem], hidden_states: torch.Tensor) -> None:
        """Choose each token's exit step from a pre-loop gate, before its first recurrent step.

        A pre-loop gate emits a distribution over exit depths from the embedding alone, so
        the exit step is the first depth whose cumulative probability reaches the exit
        threshold, floored at ``min_recurrent_steps``. Resolving it here rather than during
        the loop is what the gate is for: the decision needs no per-step readout and nothing
        is consumed a step late. Does nothing for the per-step hazard gates.

        A budget shorter than the gate's trained depth needs no reweighting: the mass past
        the budget simply never reaches the threshold, so those tokens fall through to the
        final step. That is the rule the hazard gates already follow, where the forced exit
        at the last step absorbs whatever survival probability is left. A budget longer than
        the trained depth is rejected when the adapter is configured, since the gate has no
        probability to assign that deep.
        """

        if self.exit_policy.threshold is None:
            return
        exit_pdf = self.model_adapter.preloop_exit_pdf(hidden_states)
        if exit_pdf is None:
            return
        steps = self.exit_policy.select_preloop_steps(exit_pdf)
        for item, step in zip(items, steps.tolist(), strict=True):
            item.preloop_exit_step = int(step)

    def _get_prelude_stage_buffers(self, batch_size: int) -> PreludeStageBuffers:
        """Return pinned prelude staging buffers, safe to overwrite, with room for ``batch_size`` items.

        Rotates between two buffer sets so the reuse guard awaits the event of the prelude before
        last, not the one just enqueued.
        """

        turn = self.prelude_stage_buffer_turn
        self.prelude_stage_buffer_turn = (turn + 1) % len(self.prelude_stage_buffer_pool)
        buffers = self.prelude_stage_buffer_pool[turn]
        if buffers is not None and buffers.host_input_ids.numel() >= batch_size:
            if buffers.copied_event is not None:
                with nvtx.range("cdb.prelude.buffer_wait"):
                    buffers.copied_event.synchronize()
            return buffers
        # A grown-out-of set may still have copies in flight; the pinned caching allocator defers
        # reusing its memory until they complete, so dropping the reference is safe.
        capacity = max(batch_size, 2 * (buffers.host_input_ids.numel() if buffers is not None else 0))
        buffers = PreludeStageBuffers(
            host_input_ids=self._new_host_tensor((capacity,), dtype=torch.long),
            host_position_ids=self._new_host_tensor((capacity,), dtype=torch.long),
            host_slot_indices=self._new_host_tensor((capacity,), dtype=torch.long),
            host_gather_indices=self._new_host_tensor((capacity,), dtype=torch.long),
            copied_event=torch.cuda.Event() if self.cache.device.type == "cuda" else None,
        )
        self.prelude_stage_buffer_pool[turn] = buffers
        return buffers

    def compute_recurrent_batch(
        self,
        items: list[DepthWorkItem],
        recurrent_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one mixed-depth recurrent step batch."""

        recurrent_kwargs = {} if recurrent_kwargs is None else dict(recurrent_kwargs)
        if not items:
            raise ValueError("items must not be empty")
        self._ensure_recurrent_cache_policy_kwargs(items, recurrent_kwargs)
        real_batch_size = len(items)
        bucket_size = self._recurrent_bucket_size(real_batch_size)
        hidden_bank = self._require_hidden_state_bank()
        buffers = self._get_recurrent_stage_buffers(bucket_size, hidden_bank.size(-1), hidden_bank.dtype)
        if buffers.metadata_copied_event is not None:
            with nvtx.range("cdb.recurrent.buffer_wait"):
                buffers.metadata_copied_event.synchronize()
        with nvtx.range("cdb.recurrent.stage"):
            self._prepare_recurrent_stage_buffers(
                buffers, items, real_batch_size=real_batch_size, bucket_size=bucket_size
            )

        compute_stream = self.inputs_and_outputs.compute_stream
        maybe_stream = torch.cuda.stream(compute_stream) if compute_stream is not None else nullcontext()
        non_blocking = self.cache.device.type == "cuda"
        with nvtx.range("cdb.recurrent.h2d"), maybe_stream:
            buffers.slot_indices[:real_batch_size].copy_(
                buffers.host_slot_indices[:real_batch_size],
                non_blocking=non_blocking,
            )
            buffers.position_ids.copy_(buffers.host_position_ids, non_blocking=non_blocking)
            buffers.recurrent_steps.copy_(buffers.host_recurrent_steps, non_blocking=non_blocking)
            buffers.cache_position.copy_(buffers.host_cache_position, non_blocking=non_blocking)
            buffers.cu_seq_lens_q.copy_(buffers.host_cu_seq_lens_q, non_blocking=non_blocking)
            buffers.cu_seq_lens_k.copy_(buffers.host_cu_seq_lens_k, non_blocking=non_blocking)
            buffers.block_table.copy_(buffers.host_block_table, non_blocking=non_blocking)
            if buffers.metadata_copied_event is not None:
                compute_stream.record_event(buffers.metadata_copied_event)
            buffers.hidden_in.zero_()
            selected_hidden = hidden_bank.index_select(1, buffers.slot_indices[:real_batch_size])
            buffers.hidden_in[:, :real_batch_size, :].copy_(selected_hidden)
        with nvtx.range("cdb.recurrent.launch"):
            self._run_recurrent_stage(buffers, recurrent_kwargs=recurrent_kwargs)
            with maybe_stream:
                hidden_bank.index_copy_(
                    1, buffers.slot_indices[:real_batch_size], buffers.hidden_out[:, :real_batch_size, :]
                )
        if (
            self._recurrent_batch_needs_cpu_exit_signals(recurrent_kwargs)
            and self.inputs_and_outputs.compute_stream is not None
        ):
            with nvtx.range("cdb.recurrent.gate_sync"):
                self.inputs_and_outputs.compute_stream.synchronize()
        hidden_outputs = buffers.hidden_out[:, :real_batch_size, :]
        exit_signals = buffers.exit_signals[:, :real_batch_size, :]
        return hidden_outputs, exit_signals

    def _prepare_recurrent_stage_buffers(
        self,
        buffers: RecurrentStageBuffers,
        items: list[DepthWorkItem],
        *,
        real_batch_size: int,
        bucket_size: int,
    ) -> None:
        """Fill host-side recurrent metadata before one bulk device transfer.

        Per-item metadata is collected into plain Python lists and written as whole tensors;
        per-element writes into pinned buffers would dominate the host side of a decode launch.
        """

        slots: list[int] = []
        positions: list[int] = []
        steps: list[int] = []
        request_ids: list[str] = []
        kv_lengths: list[int] = []
        for item in items:
            if item.hidden_slot is None:
                raise RuntimeError(f"Depth item for request {item.state.request_id} has no hidden_state slot")
            slots.append(item.hidden_slot)
            positions.append(item.token_position)
            steps.append(item.recurrent_step)
            request_ids.append(item.state.request_id)
            kv_lengths.append(item.token_position + 1)

        n = real_batch_size
        position_row = torch.tensor(positions, dtype=torch.long)
        buffers.host_slot_indices[:n] = torch.tensor(slots, dtype=torch.long)
        buffers.host_position_ids[0, :n] = position_row
        buffers.host_recurrent_steps[:n] = torch.tensor(steps, dtype=torch.long)
        buffers.host_cache_position[:n] = position_row
        torch.arange(n + 1, dtype=torch.int32, out=buffers.host_cu_seq_lens_q[: n + 1])
        k_cumsum = torch.tensor(kv_lengths, dtype=torch.int32).cumsum(0, dtype=torch.int32)
        buffers.host_cu_seq_lens_k[0] = 0
        buffers.host_cu_seq_lens_k[1 : n + 1] = k_cumsum
        self.cache.gather_block_table_host_rows(request_ids, kv_lengths, buffers.host_block_table[0, :n])

        if bucket_size > n:
            buffers.host_slot_indices[n:].zero_()
            buffers.host_position_ids[0, n:].zero_()
            buffers.host_recurrent_steps[n:].zero_()
            buffers.host_cache_position[n:].zero_()
            # Padding rows use the same convention as the CB decode fast path: a plateaued cumsum
            # (zero-length row) and an all ``-1`` block-table row, never a real block id that the
            # kernel could alias with an allocated request's cache.
            buffers.host_cu_seq_lens_q[n + 1 :].fill_(n)
            buffers.host_cu_seq_lens_k[n + 1 :].fill_(k_cumsum[-1])
            buffers.host_block_table[0, n:].fill_(-1)

    def _ensure_recurrent_cache_policy_kwargs(
        self,
        items: list[DepthWorkItem],
        recurrent_kwargs: dict[str, Any],
    ) -> None:
        target_slot = self.cache.kv_slot_for_recurrent_step(items[0].recurrent_step)
        for item in items[1:]:
            item_slot = self.cache.kv_slot_for_recurrent_step(item.recurrent_step)
            if item_slot != target_slot:
                raise RuntimeError(
                    "CDB recurrent batch contains multiple KV slots under a multi-slot KV policy: "
                    f"first_slot={target_slot}, item_slot={item_slot}"
                )
        recurrent_kwargs["kv_policy"] = self.cache.kv_policy
        recurrent_kwargs["kv_slots_per_layer"] = self.cache.kv_slots_per_layer
        recurrent_kwargs["kv_slot"] = target_slot

    def stage_delayed_exit_signals(
        self,
        items: list[DepthWorkItem],
        exit_signals: torch.Tensor,
    ) -> list[PendingGateResult]:
        """Queue normalized scalar exit signals for delayed CPU routing."""

        real_batch_size = len(items)
        if real_batch_size == 0:
            return []
        flat_signals = exit_signals[0, :real_batch_size, 0].detach()
        host_signals, ready_event, keepalive = self._copy_exit_signals_to_host(flat_signals)
        return [
            PendingGateResult(
                host_logits=host_signals,
                batch_index=idx,
                recurrent_step=item.recurrent_step,
                ready_event=ready_event,
                keepalive=keepalive,
            )
            for idx, item in enumerate(items)
        ]

    def _copy_exit_signals_to_host(
        self, exit_signals: torch.Tensor
    ) -> tuple[torch.Tensor, object | None, object | None]:
        if exit_signals.device.type != "cuda" or not self.engine_config.use_async_batching:
            return exit_signals.float().cpu().clone(), None, None

        compute_stream = self.inputs_and_outputs.compute_stream
        d2h_stream = self.inputs_and_outputs.d2h_stream
        if compute_stream is None:
            raise RuntimeError("Async delayed gate copy requires a CUDA compute stream")

        device_signals = torch.empty((exit_signals.numel(),), dtype=torch.float32, device=exit_signals.device)
        host_signals = torch.empty((exit_signals.numel(),), dtype=torch.float32, device="cpu", pin_memory=True)
        device_copy_done = torch.cuda.Event()
        host_copy_done = torch.cuda.Event()

        with torch.cuda.stream(compute_stream):
            device_signals.copy_(exit_signals.float())
        compute_stream.record_event(device_copy_done)
        d2h_stream.wait_event(device_copy_done)
        with torch.cuda.stream(d2h_stream):
            host_signals.copy_(device_signals, non_blocking=True)
        d2h_stream.record_event(host_copy_done)
        self._retire_finished_gate_copies()
        self.pending_gate_copy_keepalives.append((host_signals, host_copy_done, device_signals))
        return host_signals, host_copy_done, device_signals

    def _retire_finished_gate_copies(self) -> None:
        self.pending_gate_copy_keepalives = [
            keepalive for keepalive in self.pending_gate_copy_keepalives if not keepalive[1].query()
        ]

    def _recurrent_bucket_size(self, real_batch_size: int) -> int:
        """Return the static recurrent-stage bucket size for this launch."""

        if not self.pad_inputs:
            return real_batch_size
        return pad_to_bucket(real_batch_size, self.decode_graph_buckets)

    def _get_recurrent_stage_buffers(
        self,
        batch_size: int,
        hidden_size: int,
        dtype: torch.dtype,
    ) -> RecurrentStageBuffers:
        key = (batch_size, dtype)
        buffers = self.recurrent_stage_buffers.get(key)
        if buffers is not None:
            return buffers

        device = self.cache.device
        buffers = RecurrentStageBuffers(
            hidden_in=torch.empty((1, batch_size, hidden_size), dtype=dtype, device=device),
            hidden_out=torch.empty((1, batch_size, hidden_size), dtype=dtype, device=device),
            exit_signals=torch.empty((1, batch_size, 1), dtype=dtype, device=device),
            position_ids=torch.empty((1, batch_size), dtype=torch.long, device=device),
            recurrent_steps=torch.empty((batch_size,), dtype=torch.long, device=device),
            cache_position=torch.empty((batch_size,), dtype=torch.long, device=device),
            cu_seq_lens_q=torch.arange(batch_size + 1, dtype=torch.int32, device=device),
            cu_seq_lens_k=torch.empty((batch_size + 1,), dtype=torch.int32, device=device),
            block_table=torch.empty(
                (1, batch_size, self.cache.max_blocks_per_request),
                dtype=torch.int32,
                device=device,
            ),
            slot_indices=torch.empty((batch_size,), dtype=torch.long, device=device),
            empty_index=torch.empty((0,), dtype=torch.int64, device=device),
            host_position_ids=self._new_host_tensor((1, batch_size), dtype=torch.long),
            host_recurrent_steps=self._new_host_tensor((batch_size,), dtype=torch.long),
            host_cache_position=self._new_host_tensor((batch_size,), dtype=torch.long),
            host_cu_seq_lens_q=self._new_host_tensor((batch_size + 1,), dtype=torch.int32),
            host_cu_seq_lens_k=self._new_host_tensor((batch_size + 1,), dtype=torch.int32),
            host_block_table=self._new_host_tensor(
                (1, batch_size, self.cache.max_blocks_per_request),
                dtype=torch.int32,
            ),
            host_slot_indices=self._new_host_tensor((batch_size,), dtype=torch.long),
            metadata_copied_event=torch.cuda.Event() if device.type == "cuda" else None,
        )
        for tensor in (
            buffers.hidden_in,
            buffers.hidden_out,
            buffers.exit_signals,
            buffers.position_ids,
            buffers.recurrent_steps,
            buffers.cache_position,
            buffers.cu_seq_lens_q,
            buffers.cu_seq_lens_k,
            buffers.block_table,
            buffers.slot_indices,
            buffers.empty_index,
        ):
            torch._dynamo.mark_static_address(tensor)
        self.recurrent_stage_buffers[key] = buffers
        return buffers

    def _new_host_tensor(self, shape: tuple[int, ...], *, dtype: torch.dtype) -> torch.Tensor:
        pin_memory = self.cache.device.type == "cuda" and torch.cuda.is_available()
        return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=pin_memory)

    def _run_recurrent_stage(
        self,
        buffers: RecurrentStageBuffers,
        recurrent_kwargs: dict[str, Any],
    ) -> None:
        use_cuda_graph = self.use_cuda_graph_decode and self.cache.device.type == "cuda"
        graph_key = (
            2,
            buffers.hidden_in.size(1),
            int(bool(recurrent_kwargs.get("use_early_exit_gate", True))),
            int(recurrent_kwargs.get("kv_slot", 0)),
        )
        compute_stream = self.inputs_and_outputs.compute_stream

        if not use_cuda_graph:
            maybe_stream = torch.cuda.stream(compute_stream) if compute_stream is not None else nullcontext()
            with maybe_stream:
                self._forward_recurrent_stage(buffers, recurrent_kwargs)
            return

        graph = self.recurrent_graphs.get_graph(graph_key)
        if graph is not None:
            self.recurrent_graph_hits += 1
            if compute_stream is None:
                graph.replay()
            else:
                with torch.cuda.stream(compute_stream):
                    graph.replay()
            return

        if compute_stream is None:
            raise RuntimeError("CUDA graph capture requires a CUDA compute stream")

        with gc_paused():
            with torch.cuda.stream(compute_stream):
                self._forward_recurrent_stage(buffers, recurrent_kwargs)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                graph, stream=compute_stream, pool=self.graph_pool_id, capture_error_mode="thread_local"
            ):
                self._forward_recurrent_stage(buffers, recurrent_kwargs)
        self.recurrent_graphs.set_graph(graph_key, graph)
        self.recurrent_graph_captures += 1

    def _forward_recurrent_stage(
        self,
        buffers: RecurrentStageBuffers,
        recurrent_kwargs: dict[str, Any],
    ) -> None:
        hidden_outputs, exit_signals = self.model_adapter.recurrent_step(
            hidden_states=buffers.hidden_in,
            position_ids=buffers.position_ids,
            recurrent_steps=buffers.recurrent_steps,
            cache_position=buffers.cache_position,
            cache=self.cache,
            block_table=buffers.block_table,
            cu_seq_lens_q=buffers.cu_seq_lens_q,
            cu_seq_lens_k=buffers.cu_seq_lens_k,
            max_seqlen_q=1,
            max_seqlen_k=1,
            read_index=buffers.empty_index,
            write_index=buffers.empty_index,
            **recurrent_kwargs,
        )
        buffers.hidden_out.copy_(hidden_outputs)
        if exit_signals is None:
            buffers.exit_signals.zero_()
        else:
            buffers.exit_signals.copy_(exit_signals)

    def compute_coda_batch(self, items: list[DepthWorkItem]) -> list[int]:
        """Run the coda/LM-head stage and greedily sample next tokens."""

        pending = self.stage_coda_batch(items, allow_async=False)
        return self.consume_coda_result(pending)

    def stage_coda_batch(self, items: list[DepthWorkItem], *, allow_async: bool = True) -> PendingCodaResult:
        """Launch the coda/LM-head stage and return a host-visible token result.

        The coda launches on the compute stream, in program order with every other stage -
        the GPU runs one lane and only host work is hidden: the token readback goes out
        asynchronously on the D2H copy engine, so the host defers its wait past the next
        launches instead of draining the device per coda. The async path needs CUDA
        streams, so CPU runs (and ``allow_async=False``) fall through to the synchronous
        variant.
        """

        if not items:
            return PendingCodaResult(items=[], host_tokens=torch.empty((0,), dtype=torch.int64))
        for item in items:
            item.state.tokens_in_flight += 1
        io = self.inputs_and_outputs
        if allow_async and io.compute_stream is not None and io.d2h_stream is not None:
            return self._stage_coda_batch_async(items)
        return self._stage_coda_batch_sync(items)

    def _stage_coda_batch_sync(self, items: list[DepthWorkItem]) -> PendingCodaResult:
        compute_stream = self.inputs_and_outputs.compute_stream
        maybe_stream = torch.cuda.stream(compute_stream) if compute_stream is not None else nullcontext()
        with maybe_stream:
            next_tokens, _staging = self._run_coda_forward(items)

        if compute_stream is not None:
            compute_stream.synchronize()
        # The full-device sync above drains the launch, so a graphed output buffer needs
        # no reuse gate before its next overwrite.
        self._record_coda_buffer_reuse(None)
        host_tokens = next_tokens.detach().to(device="cpu", dtype=torch.int64)
        return PendingCodaResult(items=items, host_tokens=host_tokens, device_tokens=next_tokens.detach())

    def _stage_coda_batch_async(self, items: list[DepthWorkItem]) -> PendingCodaResult:
        compute_stream = self.inputs_and_outputs.compute_stream
        d2h_stream = self.inputs_and_outputs.d2h_stream
        assert compute_stream is not None and d2h_stream is not None

        batch_size = len(items)
        host_tokens = torch.empty((batch_size,), dtype=torch.int64, device="cpu", pin_memory=True)
        coda_done = torch.cuda.Event()
        d2h_done = torch.cuda.Event()

        with torch.cuda.stream(compute_stream):
            device_tokens, staging = self._run_coda_forward(items)
        compute_stream.record_event(coda_done)
        d2h_stream.wait_event(coda_done)
        with torch.cuda.stream(d2h_stream):
            host_tokens.copy_(device_tokens, non_blocking=True)
        d2h_stream.record_event(d2h_done)
        # The D2H copy is the last off-lane reader of a graphed coda's rotating output buffer, so
        # its completion event gates that buffer's next overwrite. The re-entry prelude gather runs
        # later on the compute stream, so program order already places it before any next coda.
        self._record_coda_buffer_reuse(d2h_done)
        return PendingCodaResult(
            items=items,
            host_tokens=host_tokens,
            device_tokens=device_tokens,
            ready_event=d2h_done,
            keepalive=staging,
        )

    def _run_coda_forward(self, items: list[DepthWorkItem]) -> tuple[torch.Tensor, Any | None]:
        """Run the coda/LM-head forward on the current stream and return sampled device tokens.

        Dispatches to the CUDA-graphed bucket path when stage graphs are on, else runs
        eagerly with per-call staging tensors (in-flight coda batches pipeline, so the
        eager path must not share mutable staging buffers between them). The second
        return value holds the eager path's staging tensors, which the caller must keep
        alive until the result is consumed: the pinned slot buffer is not stream-tracked,
        so freeing it while its H2D copy is in flight would let the host reuse it.
        """

        batch_size = len(items)
        bucket_size = self._stage_bucket_size(batch_size)
        if bucket_size is not None:
            return self._run_coda_forward_graphed(items, bucket_size), None

        hidden_bank = self._require_hidden_state_bank()
        non_blocking = self.cache.device.type == "cuda"
        host_slot_indices = self._new_host_tensor((batch_size,), dtype=torch.long)
        self._fill_host_slot_indices(host_slot_indices, items)
        slot_indices = torch.empty((batch_size,), dtype=torch.long, device=self.cache.device)
        coda_attention_kwargs = self.stage_attention_kwargs(items, stage="coda", include_position_ids=True)
        slot_indices.copy_(host_slot_indices, non_blocking=non_blocking)
        hidden_states = hidden_bank.index_select(1, slot_indices)
        logits = self.model_adapter.lm_head(hidden_states, **coda_attention_kwargs)
        self._last_coda_turn = None
        return torch.argmax(logits[0], dim=-1).to(dtype=torch.int64), (host_slot_indices, slot_indices)

    def _run_coda_forward_graphed(self, items: list[DepthWorkItem], bucket_size: int) -> torch.Tensor:
        """Replay (or capture) the coda-stage graph for one bucket on the current stream.

        The hidden-bank gather runs eagerly into the bucket's static input, so the bank
        may grow without invalidating graphs; the captured region is the coda forward,
        LM head, and argmax into the turn's static token buffer.
        """

        hidden_bank = self._require_hidden_state_bank()
        batch_size = len(items)
        non_blocking = self.cache.device.type == "cuda"
        buffers = self._get_coda_graph_buffers(bucket_size)
        turn = buffers.turn
        buffers.turn = (turn + 1) % len(buffers.tokens_out)
        reuse_event = buffers.reuse_events[turn]
        if reuse_event is not None:
            # The previous coda on this turn may still be flowing toward its D2H copy;
            # both its pinned staging rows and its token buffer are about to be reused.
            with nvtx.range("cdb.coda.buffer_wait"):
                reuse_event.synchronize()
            buffers.reuse_events[turn] = None

        host_slots = buffers.host_slot_indices[turn]
        self._fill_host_slot_indices(host_slots, items)
        if bucket_size > batch_size:
            # Padding rows gather bank row 0: always allocated, and their coda output is
            # never read.
            host_slots[batch_size:].fill_(0)
        attention_kwargs = self.stage_attention_kwargs(
            items, stage="coda", include_position_ids=True, bucket_size=bucket_size
        )
        buffers.slot_indices.copy_(host_slots, non_blocking=non_blocking)
        if buffers.hidden_in is None:
            buffers.hidden_in = torch.empty(
                (1, bucket_size, hidden_bank.size(-1)), dtype=hidden_bank.dtype, device=self.cache.device
            )
            torch._dynamo.mark_static_address(buffers.hidden_in)
        buffers.hidden_in.copy_(hidden_bank.index_select(1, buffers.slot_indices))

        def forward() -> torch.Tensor:
            logits = self.model_adapter.lm_head(buffers.hidden_in, **attention_kwargs)
            return torch.argmax(logits[0], dim=-1).to(dtype=torch.int64)

        graph = self.coda_graphs.get_graph((bucket_size, turn))
        if graph is not None:
            self.stage_graph_hits += 1
            graph.replay()
        else:
            capture_stream = torch.cuda.current_stream()
            with gc_paused():
                forward()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(
                    graph, stream=capture_stream, pool=self.graph_pool_id, capture_error_mode="thread_local"
                ):
                    buffers.tokens_out[turn].copy_(forward())
            self.coda_graphs.set_graph((bucket_size, turn), graph)
            self.stage_graph_captures += 1
            graph.replay()

        self._last_coda_turn = (buffers, turn)
        return buffers.tokens_out[turn][:batch_size]

    def _record_coda_buffer_reuse(self, ready_event: Any | None) -> None:
        """Gate the last graphed coda's turn buffer on its final reader, or clear the gate."""

        if self._last_coda_turn is None:
            return
        buffers, turn = self._last_coda_turn
        buffers.reuse_events[turn] = ready_event
        self._last_coda_turn = None

    def _get_coda_graph_buffers(self, bucket_size: int) -> CodaGraphBuffers:
        buffers = self.coda_graph_buffers.get(bucket_size)
        if buffers is not None:
            return buffers
        device = self.cache.device
        num_turns = CODA_PIPELINE_DEPTH
        buffers = CodaGraphBuffers(
            slot_indices=torch.zeros((bucket_size,), dtype=torch.long, device=device),
            host_slot_indices=self._new_host_tensor((num_turns, bucket_size), dtype=torch.long),
            tokens_out=[torch.zeros((bucket_size,), dtype=torch.int64, device=device) for _ in range(num_turns)],
            reuse_events=[None] * num_turns,
        )
        torch._dynamo.mark_static_address(buffers.slot_indices)
        for tokens in buffers.tokens_out:
            torch._dynamo.mark_static_address(tokens)
        self.coda_graph_buffers[bucket_size] = buffers
        return buffers

    def consume_coda_result(self, pending: PendingCodaResult) -> list[int]:
        """Wait for a pending coda result, release hidden slots, and return sampled token ids."""

        if pending.ready_event is not None:
            with nvtx.range("cdb.coda_wait"):
                pending.ready_event.synchronize()
        token_ids = pending.host_tokens.tolist()
        self._release_hidden_slots(pending.items)
        return token_ids

    def release_hidden_slots(self, items: list[DepthWorkItem]) -> None:
        """Release hidden-state-bank slots for work items that do not need coda."""

        self._release_hidden_slots(items)

    def reset_cdb_runtime_counters(self) -> None:
        """Reset CDB graph counters for a new generation run."""

        self.recurrent_graph_hits = 0
        self.recurrent_graph_captures = 0
        self.stage_graph_hits = 0
        self.stage_graph_captures = 0
        self.hidden_state_bank = None
        self.hidden_state_bank_meta = None
        if not self.use_stage_cuda_graphs:
            # Captured stage graphs hold raw pointers into these buffers, so under stage
            # graphs they must outlive the run exactly like the recurrent stage buffers.
            self.stage_attention_buffers.clear()
        # The syncs above drained every in-flight coda, so the rotating token buffers
        # need no reuse gate on the next run.
        for buffers in self.coda_graph_buffers.values():
            buffers.reuse_events = [None] * len(buffers.reuse_events)
        self._last_coda_turn = None
        self.prelude_stage_buffer_pool = [None, None]
        self.prelude_stage_buffer_turn = 0
        self.free_hidden_slots = []
        self.next_hidden_slot = 0
        self.pending_gate_copy_keepalives.clear()

    def _allocate_hidden_slots(self, count: int, hidden_size: int, dtype: torch.dtype) -> list[int]:
        self._ensure_hidden_state_bank(
            self.next_hidden_slot + max(0, count - len(self.free_hidden_slots)), hidden_size, dtype
        )
        slots = []
        while self.free_hidden_slots and len(slots) < count:
            slots.append(self.free_hidden_slots.pop())
        while len(slots) < count:
            slots.append(self.next_hidden_slot)
            self.next_hidden_slot += 1
        return slots

    def _ensure_hidden_state_bank(self, min_capacity: int, hidden_size: int, dtype: torch.dtype) -> None:
        meta = (hidden_size, dtype)
        if self.hidden_state_bank is not None and self.hidden_state_bank_meta != meta:
            raise RuntimeError(
                "CDB hidden-state bank cannot change hidden size or dtype within one generation: "
                f"existing={self.hidden_state_bank_meta}, requested={meta}"
            )
        if self.hidden_state_bank is not None and self.hidden_state_bank.size(1) >= min_capacity:
            return

        old_bank = self.hidden_state_bank
        old_capacity = 0 if old_bank is None else old_bank.size(1)
        # Two launches' worth of slots are live at once: the wave-boundary carry (and the refill
        # path's in-flight codas) hold the previous launch's slots across the next launch's prelude,
        # so the first allocation sizes for both up front and steady-state growth never triggers.
        new_capacity = max(2 * self.engine_config.max_num_seqs, min_capacity, old_capacity * 2, 1)
        new_bank = torch.empty((1, new_capacity, hidden_size), dtype=dtype, device=self.cache.device)
        if old_bank is not None and old_capacity:
            # Every bank write runs on the compute stream, so drain it before the copy-forward:
            # no in-flight write is lost and the old bank cannot be freed under a pending access.
            # Growth is rare (see the pre-sizing above), so the sync costs nothing in steady state.
            compute_stream = self.inputs_and_outputs.compute_stream
            if compute_stream is not None:
                compute_stream.synchronize()
            new_bank[:, :old_capacity, :].copy_(old_bank)
        self.hidden_state_bank = new_bank
        self.hidden_state_bank_meta = meta

    def _require_hidden_state_bank(self) -> torch.Tensor:
        if self.hidden_state_bank is None:
            raise RuntimeError("CDB hidden-state bank is empty; run compute_prelude_batch before recurrent/coda stages")
        return self.hidden_state_bank

    def _fill_host_slot_indices(self, host_slot_indices: torch.Tensor, items: list[DepthWorkItem]) -> None:
        if host_slot_indices.device.type != "cpu":
            raise ValueError("_fill_host_slot_indices expects a CPU tensor")
        if host_slot_indices.numel() < len(items):
            raise ValueError(f"host_slot_indices has {host_slot_indices.numel()} slots for {len(items)} items")
        slots = []
        for item in items:
            if item.hidden_slot is None:
                raise RuntimeError(f"Depth item for request {item.state.request_id} has no hidden_state slot")
            slots.append(item.hidden_slot)
        host_slot_indices[: len(items)] = torch.tensor(slots, dtype=host_slot_indices.dtype)

    def _get_stage_attention_buffers(self, stage: str, batch_size: int) -> StageAttentionBuffers:
        """Return paged-attention metadata buffers for one stage's batch, safe to overwrite.

        One buffer set per ``(stage, batch_size)``, so consecutive batches of the
        same size reuse it. Waiting on the previous batch's copy event keeps the
        host writes below from racing an in-flight copy out of pinned memory.
        """

        key = (stage, batch_size)
        buffers = self.stage_attention_buffers.get(key)
        if buffers is not None:
            if buffers.copied_event is not None:
                with nvtx.range("cdb.stage_attention.buffer_wait"):
                    buffers.copied_event.synchronize()
            return buffers
        device = self.cache.device
        buffers = StageAttentionBuffers(
            cu_seq_lens_q=torch.empty((batch_size + 1,), dtype=torch.int32, device=device),
            cu_seq_lens_k=torch.empty((batch_size + 1,), dtype=torch.int32, device=device),
            block_table=torch.empty(
                (1, batch_size, self.cache.max_blocks_per_request), dtype=torch.int32, device=device
            ),
            empty_index=torch.empty((0,), dtype=torch.int64, device=device),
            position_ids=torch.empty((1, batch_size), dtype=torch.long, device=device),
            host_cu_seq_lens_q=self._new_host_tensor((batch_size + 1,), dtype=torch.int32),
            host_cu_seq_lens_k=self._new_host_tensor((batch_size + 1,), dtype=torch.int32),
            host_block_table=self._new_host_tensor(
                (1, batch_size, self.cache.max_blocks_per_request), dtype=torch.int32
            ),
            host_position_ids=self._new_host_tensor((1, batch_size), dtype=torch.long),
            copied_event=torch.cuda.Event() if device.type == "cuda" else None,
        )
        self.stage_attention_buffers[key] = buffers
        return buffers

    def stage_attention_kwargs(
        self,
        items: list[DepthWorkItem],
        *,
        stage: str,
        include_position_ids: bool = False,
        bucket_size: int | None = None,
    ) -> dict[str, Any]:
        """Build paged-attention arguments for a stage that runs one token per request.

        Returns an empty mapping when the model has no attention outside the
        recurrent core, so fully looped models keep their current stage calls.

        ``bucket_size`` pads the batch to a static CUDA-graph shape. Padding rows use
        the recurrent stage's convention: a plateaued cumsum (zero-length row) and an
        all ``-1`` block-table row, never a real block id that the kernel could alias
        with an allocated request's cache.
        """

        if not self.model_adapter_stages_use_attention:
            return {}

        # Must be called inside the stream that runs the stage: the host-to-device
        # copies are asynchronous, so issuing them on another stream would let the
        # attention kernel read the buffers before they land. Same-stream ordering
        # covers the kernel, but not the host, which is why the buffer getter waits
        # on the previous batch's copy event before the writes below reuse the
        # pinned staging memory.

        n = len(items)
        size = n if bucket_size is None else bucket_size
        if size < n:
            raise ValueError(f"bucket_size {size} is smaller than the batch ({n} items)")
        device = self.cache.device
        non_blocking = device.type == "cuda"
        buffers = self._get_stage_attention_buffers(stage, size)

        kv_lengths = [item.token_position + 1 for item in items]
        request_ids = [item.state.request_id for item in items]
        torch.arange(n + 1, dtype=torch.int32, out=buffers.host_cu_seq_lens_q[: n + 1])
        k_cumsum = torch.tensor(kv_lengths, dtype=torch.int32).cumsum(0, dtype=torch.int32)
        buffers.host_cu_seq_lens_k[0] = 0
        buffers.host_cu_seq_lens_k[1 : n + 1] = k_cumsum
        self.cache.gather_block_table_host_rows(request_ids, kv_lengths, buffers.host_block_table[0, :n])
        if size > n:
            buffers.host_cu_seq_lens_q[n + 1 :].fill_(n)
            buffers.host_cu_seq_lens_k[n + 1 :].fill_(int(k_cumsum[-1]))
            buffers.host_block_table[0, n:].fill_(-1)

        buffers.cu_seq_lens_q.copy_(buffers.host_cu_seq_lens_q, non_blocking=non_blocking)
        buffers.cu_seq_lens_k.copy_(buffers.host_cu_seq_lens_k, non_blocking=non_blocking)
        buffers.block_table.copy_(buffers.host_block_table, non_blocking=non_blocking)

        kwargs: dict[str, Any] = {
            "cache": self.cache,
            "block_table": buffers.block_table,
            "cu_seq_lens_q": buffers.cu_seq_lens_q,
            "cu_seq_lens_k": buffers.cu_seq_lens_k,
            "max_seqlen_q": 1,
            "max_seqlen_k": 1,
            "read_index": buffers.empty_index,
            "write_index": buffers.empty_index,
        }
        if include_position_ids:
            buffers.host_position_ids[0, :n] = torch.tensor([item.token_position for item in items], dtype=torch.long)
            if size > n:
                buffers.host_position_ids[0, n:].zero_()
            buffers.position_ids.copy_(buffers.host_position_ids, non_blocking=non_blocking)
            kwargs["position_ids"] = buffers.position_ids
        if buffers.copied_event is not None:
            # Records on the caller's stream, after this batch's H2D copies were enqueued on it.
            buffers.copied_event.record()
        return kwargs

    def _release_hidden_slots(self, items: list[DepthWorkItem]) -> None:
        for item in items:
            if item.hidden_slot is not None:
                self.free_hidden_slots.append(item.hidden_slot)
                item.hidden_slot = None

    def _recurrent_batch_needs_cpu_exit_signals(self, recurrent_kwargs: dict[str, Any]) -> bool:
        """Whether routing needs the current normalized signal on the host now."""

        return self.exit_policy.needs_synchronous_signal(
            signal_enabled=bool(recurrent_kwargs.get("use_early_exit_gate", True))
        )
