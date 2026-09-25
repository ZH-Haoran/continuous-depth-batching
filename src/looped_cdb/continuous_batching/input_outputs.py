# Copyright 2026 The HuggingFace Inc. team
# Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Modified for looped-model serving and continuous depth batching.
# See THIRD_PARTY_NOTICES.md and LICENSES/Apache-2.0.txt.

"""Batch-shaped inputs and outputs for continuous batching.

This module owns the static tensors that connect scheduling, cache management, and model
execution. There is a single set of device-side buffers (and therefore a single CUDA-graph
set): batches are staged in pinned host memory and copied to the device with one bulk
transfer enqueued on the compute stream, so every device-side write is stream-ordered
against the forward passes that read it and no double buffering is needed. Sampled tokens
are copied back on a separate stream and consumed by the host one batch later, which lets
the engine run the scheduler one step ahead of the device (vLLM-style async scheduling).

Decode inputs whose token has not been sampled yet hold a placeholder id; the forward pass
starts with an on-device scatter (``carry_over_tokens``) that fills them from the previous
batch's output tensor, so the token feedback loop never round-trips through the host.

References:
https://github.com/huggingface/transformers/blob/main/src/transformers/generation/continuous_batching/input_outputs.py
https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_model_runner.py (async scheduling)
"""

from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
from transformers.configuration_utils import PreTrainedConfig

from looped_cdb.benchmarks import nvtx

from .cache import PagedAttentionCache
from .prefill_graph import PrefillGraphContext
from .requests import TMP_TOKEN_ID, FutureRequestState, logger
from .utils import CudaGraphBuffer, aligned_divide


@dataclass
class PagedAttentionArgs:
    """Dataclass containing the keyword arguments for a forward pass using paged attention.

    Attributes:
        input_ids: Input token IDs tensor of shape `(1, total_query_tokens)`.
        position_ids: Position IDs tensor of shape `(1, total_query_tokens)`.
        cu_seq_lens_q: Cumulative sequence lengths for queries, used for variable-length batching.
        cu_seq_lens_k: Cumulative sequence lengths for keys/values.
        max_seqlen_q: Maximum query sequence length in the batch.
        max_seqlen_k: Maximum key/value sequence length.
        write_index: Tensor indicating where to write new KV states in the cache.
        read_index: Tensor indicating which cache positions to read from.
        logits_indices: Tensor indicating which positions in the output should be used for next-token prediction.
        cache: The [`PagedAttentionCache`] instance managing the KV cache.
        block_table: Grouped block table for paged KV cache. If provided, uses `flash_attn_with_kvcache` for fused
            attention + cache update. More information in src/transformers/integrations/flash_paged.py.
        use_cache: Whether to use caching (always `False` in continuous batching as the cache is managed externally).
    """

    input_ids: torch.Tensor
    position_ids: torch.Tensor
    cu_seq_lens_q: torch.Tensor
    cu_seq_lens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    write_index: torch.Tensor
    read_index: torch.Tensor
    logits_indices: torch.Tensor
    cache: PagedAttentionCache
    block_table: torch.Tensor | None
    use_cache: bool = False

    def asdict(self) -> dict[str, Any]:
        return {
            "input_ids": self.input_ids,
            "position_ids": self.position_ids,
            "cu_seq_lens_q": self.cu_seq_lens_q,
            "cu_seq_lens_k": self.cu_seq_lens_k,
            "max_seqlen_q": self.max_seqlen_q,
            "max_seqlen_k": self.max_seqlen_k,
            "write_index": self.write_index,
            "read_index": self.read_index,
            "logits_indices": self.logits_indices,
            "cache": self.cache,
            "block_table": self.block_table,
            "use_cache": self.use_cache,
        }


class StaticIOBuffers:
    """One set of static batch tensors, usable as host staging or as the device-side set."""

    static_inputs: int = 6  # Number of static inputs always present in the bulk tensor

    def __init__(
        self,
        cache: PagedAttentionCache,
        config: PreTrainedConfig,
        device: torch.device | str,
        model_dtype: torch.dtype,
    ) -> None:
        """Initialize one static tensor set.

        Args:
            cache: The [`PagedAttentionCache`] instance managing the KV cache. Meant to be unique.
            config: The model's pretrained configuration.
            device: The device to allocate tensors on. If the device is CPU, then the memory is pinned when CUDA is
                available.
            model_dtype: The data type for model computations.
        """

        self.cache = cache
        self.device = torch.device(device)
        self.config = config
        self.model_dtype = model_dtype

        self.num_q_tokens = 0  # number of query tokens in the batch. Can be padded.
        self.max_kv_read = 0  # number of KV tokens read from cache. Can be padded.
        self.true_batch_size = 0
        self.true_read_size = 0
        self.true_write_size = 0
        self.true_num_logits = 0  # rows whose next-token logits are actually sampled
        self.use_block_table = False  # True if all requests in batch have query_length == 1

        self.requests_in_batch: list[FutureRequestState] = []
        self.req_id_to_new_token_position: dict[str, int] = {}  # maps request id -> row in output_ids
        self.graphs: CudaGraphBuffer = CudaGraphBuffer()
        self._trash_index = cache.trash_index

        self._setup_static_tensors()
        self._reset_static_tensors(full_reset=True)

    def _setup_static_tensors(self) -> None:
        """Allocates static tensors for generation inputs and outputs. This is called only once at init time, to avoid
        repeated allocations and enable CUDA graphs. All tensors are allocated with maximum possible sizes.
        The allocated tensors are:

        - `_bulk_input_tensor`: Storage for all the small inputs: `input_ids`, `position_ids`, `cumulative_seqlens_q`,
          `logits_indices`, `cumulative_seqlens_k`, `carry_over_ids`.
        - `write_index` and `read_index` storage: Cache indexing tensors.
        - `output_ids`: Storage for generated token IDs.
        """

        max_num_batched_tokens = self.cache.max_num_batched_tokens
        num_pages = self.cache.num_blocks * self.cache.block_size
        pin_memory = self.device.type == "cpu" and torch.cuda.is_available()

        # Small inputs are allocated as slices in a larger tensor aligned to 128 bytes (32 * 4b). This reduces
        # fragmentation, so it lowers the number of H2D transfers and speeds up transfers.
        bulk_lines = self.static_inputs
        bulk_columns = aligned_divide(max_num_batched_tokens + 1, 1, 32)
        self._bulk_input_tensor = torch.empty(
            (bulk_lines, bulk_columns),
            dtype=torch.int32,
            device=self.device,
            pin_memory=pin_memory,
        )

        self.input_ids = self._bulk_input_tensor[0, :max_num_batched_tokens]
        self.position_ids = self._bulk_input_tensor[1, :max_num_batched_tokens]
        self.cumulative_seqlens_q = self._bulk_input_tensor[2, : max_num_batched_tokens + 1]
        self.logits_indices = self._bulk_input_tensor[3, :max_num_batched_tokens]
        self.cumulative_seqlens_k = self._bulk_input_tensor[4, : max_num_batched_tokens + 1]
        self.carry_over_ids = self._bulk_input_tensor[5, :max_num_batched_tokens]

        self.output_ids = torch.empty(
            (1, max_num_batched_tokens + 1),
            dtype=torch.int32,
            device=self.device,
            pin_memory=pin_memory,
        )
        # The last output token is a sentinel that is never written: carry_over_ids entries of -1 index
        # it, and the carry mask then discards the read value. Zeroed once here, not per batch.
        self.output_ids.zero_()
        self.total_seqlen_q = 0
        self.total_seqlen_k = 0
        self.max_seqlen_q = 0
        self.max_seqlen_k = 0

        self.block_table = torch.empty(
            (1, max_num_batched_tokens, self.cache.max_blocks_per_request),
            dtype=torch.int32,
            device=self.device,
            pin_memory=pin_memory,
        )

        self.write_index_storage = torch.empty(
            (max_num_batched_tokens,),
            dtype=torch.int64,
            device=self.device,
            pin_memory=pin_memory,
        )
        self.read_index_storage = torch.empty(
            (num_pages + max_num_batched_tokens,),
            dtype=torch.int64,
            device=self.device,
            pin_memory=pin_memory,
        )

    def transfer_inputs_to(self, other: "StaticIOBuffers") -> None:
        """Copy this batch's metadata and used tensor regions into ``other`` (host staging to device).

        Only the regions the current batch reads are copied; the caller is responsible for enqueueing
        this on the stream that runs the forward pass, so the copy is ordered after the previous batch.
        """

        other.num_q_tokens = self.num_q_tokens
        other.max_kv_read = self.max_kv_read
        other.true_batch_size = self.true_batch_size
        other.true_read_size = self.true_read_size
        other.true_write_size = self.true_write_size
        other.use_block_table = self.use_block_table

        other.total_seqlen_q = self.total_seqlen_q
        other.total_seqlen_k = self.total_seqlen_k
        other.true_num_logits = self.true_num_logits
        other.max_seqlen_q = self.max_seqlen_q
        other.max_seqlen_k = self.max_seqlen_k
        other.requests_in_batch = self.requests_in_batch
        other.req_id_to_new_token_position = self.req_id_to_new_token_position

        q_len = self.num_q_tokens
        other._bulk_input_tensor[:, : q_len + 1].copy_(self._bulk_input_tensor[:, : q_len + 1], non_blocking=True)
        if self.use_block_table:
            other.block_table[:, :q_len].copy_(self.block_table[:, :q_len], non_blocking=True)
        else:
            other.write_index_storage[:q_len].copy_(self.write_index_storage[:q_len], non_blocking=True)
            if self.max_kv_read > 0:
                kv_len = self.max_kv_read + q_len
                other.read_index_storage[:kv_len].copy_(self.read_index_storage[:kv_len], non_blocking=True)

    @torch.no_grad()
    def _reset_static_tensors(self, full_reset: bool = False) -> None:
        """Reset static tensors for the next batch."""

        q_len = self.write_index_storage.size(-1) if full_reset else self.num_q_tokens
        kv_len = self.read_index_storage.size(-1) if full_reset else self.max_kv_read

        self._bulk_input_tensor[: self.static_inputs, : q_len + 1].zero_()
        self.max_seqlen_q = 0
        self.max_seqlen_k = 0

        self.logits_indices[:q_len].zero_()
        self.total_seqlen_k = 0

        if full_reset:
            self.block_table[:, :q_len].fill_(-1)
            self.write_index_storage[:q_len].fill_(self._trash_index)
            self.read_index_storage[: q_len + kv_len].fill_(self._trash_index)
        elif self.use_block_table:
            self.block_table[:, :q_len].fill_(-1)
        else:
            self.write_index_storage[:q_len].fill_(self._trash_index)
            self.read_index_storage[: q_len + kv_len].fill_(self._trash_index)

    def reset(self) -> None:
        """Reset all relevant states for a new generation loop."""

        self._reset_static_tensors(full_reset=True)
        self.requests_in_batch = []
        self.req_id_to_new_token_position = {}

    def get_cumulative_seqlens(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the cumulative sequence lengths for the current batch."""

        return self.cumulative_seqlens_q, self.cumulative_seqlens_k

    def prepare_batch_tensors(
        self,
        requests_in_batch: list[FutureRequestState],
        use_decode_fast_path: bool,
        num_q_tokens: int,
        max_kv_read: int,
    ) -> None:
        """Prepare tensors and metadata for the next model forward pass, using the given requests as data.

        This method:

        1. Resets the static tensors from the previous batch.
        2. Iterates through requests to accumulate input_ids, position_ids, and sequence lengths.
        3. Extends read/write indices for cache management.
        4. Converts accumulated lists to tensors and copies them to static storage.

        This method also modifies the `position_offset` attribute of each request to track progress and adds a
        temporary token at the end of the requests for which there will a new token.
        """

        if not requests_in_batch:
            raise ValueError("No requests in batch")

        self.use_block_table = use_decode_fast_path and self.block_table.numel() > 0
        self.num_q_tokens = num_q_tokens
        self.max_kv_read = 0 if self.use_block_table else max_kv_read
        self.true_batch_size = len(requests_in_batch)
        self._reset_static_tensors()

        self.true_read_size = 0
        self.true_write_size = 0
        self.requests_in_batch = []
        self.req_id_to_new_token_position = {}

        # Prepare accumulators. For batches with no past cache to read, we leave read_index empty: the cache.update
        # will detect the 0-size indices and skip the read.
        input_ids = []
        position_ids = []
        cumulative_seqlens_q = [0]
        logits_indices = []
        cumulative_seqlens_k = [0]
        write_index: list[int] = []
        read_index: list[int] | None = None if self.max_kv_read == 0 else []

        for i, future_state in enumerate(requests_in_batch):
            # First we retrieve the lengths related to the request
            state = future_state.state
            past_length = state.position_offset
            query_length = future_state.query_length
            seqlens_k = self.cache.get_seqlens_k(past_length, query_length)["full_attention"]

            # Update the internal state of the request
            state.position_offset += query_length

            # Then we accumulate for the object used in the kwargs
            input_ids.extend(state.tokens_to_process)
            position_ids.extend(range(past_length, past_length + query_length))
            cumulative_seqlens_q.append(cumulative_seqlens_q[-1] + query_length)
            self.max_seqlen_q = max(self.max_seqlen_q, query_length)

            # Accumulate the key sequence lengths for the current request
            cumulative_seqlens_k.append(cumulative_seqlens_k[-1] + seqlens_k)
            self.max_seqlen_k = max(self.max_seqlen_k, seqlens_k)

            # We extend the read and write indices for the cache, or fill the block table for decode-only batches
            if self.use_block_table:
                self.cache.fill_block_table(state.request_id, past_length, query_length, self.block_table[0, i])
            else:
                self.cache.extend_read_and_write_indices(
                    state.request_id,
                    past_length,
                    query_length,
                    read_index,
                    write_index,
                )

            # If the request has no remaining prefill tokens, it means the next token prediction is relevant
            if future_state.has_new_token:
                logits_indices.append(cumulative_seqlens_q[-1] - 1)
                state.tokens_to_process = [TMP_TOKEN_ID]
                state.tokens_in_flight += 1
                self.req_id_to_new_token_position[state.request_id] = logits_indices[-1]

            self.requests_in_batch.append(future_state)

        if len(input_ids) > num_q_tokens:
            raise ValueError(f"Expected at most {num_q_tokens} query tokens, but got {len(input_ids)}")

        # When looping over request is done, we can build the actual tensors. This is faster than modifying the static
        # tensors inside the loop.
        to_tensor = partial(torch.tensor, dtype=torch.int32, device=self.device)

        # Those kwargs always have the same type regardless of the model
        self.input_ids[: len(input_ids)] = to_tensor(input_ids)
        self.position_ids[: len(position_ids)] = to_tensor(position_ids)
        self.cumulative_seqlens_q[: len(cumulative_seqlens_q)] = to_tensor(cumulative_seqlens_q)
        self.logits_indices[: len(logits_indices)] = to_tensor(logits_indices)
        self.total_seqlen_q = cumulative_seqlens_q[-1]

        self.cumulative_seqlens_k[: len(cumulative_seqlens_k)] = to_tensor(cumulative_seqlens_k)
        self.total_seqlen_k = cumulative_seqlens_k[-1]
        self.true_num_logits = len(logits_indices)

        # If we are not using the block table, we populate the write indices (and maybe the read indices)
        if not self.use_block_table:
            to_index_tensor = partial(torch.tensor, dtype=torch.int64, device=self.device)
            self.write_index_storage[: len(write_index)] = to_index_tensor(write_index)
            self.true_write_size = len(write_index)
            if read_index is not None:
                self.read_index_storage[: len(read_index)] = to_index_tensor(read_index)
                self.true_read_size = len(read_index)

    def finalize_batch(self, use_padding: bool) -> None:
        """Apply padded lengths in place for graphed decode batches, before the tensors are transferred.

        Decode (block-table) batches pad with zero-length sequences (cumulative lengths plateau), which
        the fused decode kernel handles per lane. A padded varlen batch needs no change here: its padded
        query rows carry placeholder tokens and trash write slots, and its attention reads the real
        lengths from :meth:`prefill_graph_context`.
        """

        if not use_padding or not self.use_block_table:
            return
        batch_size = self.num_q_tokens
        self.cumulative_seqlens_q[self.true_batch_size + 1 : batch_size + 1] = self.total_seqlen_q
        self.cumulative_seqlens_k[self.true_batch_size + 1 : batch_size + 1] = self.total_seqlen_k

    def get_model_kwargs(self, use_padding: bool = False) -> dict[str, Any]:
        """Get model keyword arguments for the current batch as views on the static tensors. Padding must already
        have been applied by :meth:`finalize_batch`; this method performs no writes, so it is safe to call on the
        device-side buffer set while a previous batch is still in flight."""

        q_size = self.num_q_tokens
        kv_size = self.max_kv_read + self.num_q_tokens
        # Only decode padding pads the request axis; a padded varlen batch keeps its real requests.
        batch_size = self.num_q_tokens if use_padding and self.use_block_table else self.true_batch_size

        # When using block table, max_seqlen_q and max_seqlen_k are not used by flash_attn_with_kvcache, so we set
        # them to constant `1` to avoid dynamo guards on these changing integer values.
        max_seqlen_q = 1 if self.use_block_table else self.max_seqlen_q
        max_seqlen_k = 1 if self.use_block_table else self.max_seqlen_k

        write_index_size = q_size if use_padding else self.true_write_size
        read_index_size = 0 if self.max_kv_read == 0 else self.true_read_size
        if use_padding and self.max_kv_read > 0:
            read_index_size = kv_size

        kwargs = PagedAttentionArgs(
            input_ids=self.input_ids[:q_size].unsqueeze(0),
            position_ids=self.position_ids[:q_size].unsqueeze(0),
            cu_seq_lens_q=self.cumulative_seqlens_q[: batch_size + 1],
            cu_seq_lens_k=self.cumulative_seqlens_k[: batch_size + 1],
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            write_index=self.write_index_storage[:write_index_size],
            read_index=self.read_index_storage[:read_index_size],
            logits_indices=self.logits_indices[:q_size],
            cache=self.cache,
            block_table=self.block_table[:, :batch_size] if self.use_block_table else None,
            use_cache=False,
        )
        return kwargs.asdict()

    def prefill_graph_context(self) -> PrefillGraphContext:
        """The current varlen batch's real lengths and cache indices, for the graphed attention break."""

        if self.use_block_table:
            raise RuntimeError("prefill graph contexts describe varlen batches only")
        batch_size = self.true_batch_size
        return PrefillGraphContext(
            num_tokens=self.total_seqlen_q,
            cu_seq_lens_q=self.cumulative_seqlens_q[: batch_size + 1],
            cu_seq_lens_k=self.cumulative_seqlens_k[: batch_size + 1],
            max_seqlen_q=self.max_seqlen_q,
            max_seqlen_k=self.max_seqlen_k,
            write_index=self.write_index_storage[: self.true_write_size],
            read_index=self.read_index_storage[: self.true_read_size],
            cache=self.cache,
        )

    def _get_graph_key(self) -> tuple[int, ...]:
        # Only the decode fast path is graphed; its shape is fully described by the padded batch
        # size (per-request context lengths are tensor data the graph re-reads on replay).
        if not self.use_block_table:
            raise RuntimeError("CUDA graphs are only supported on the decode fast path")
        return (self.num_q_tokens,)

    def get_graph(self) -> torch.cuda.CUDAGraph | None:
        key = self._get_graph_key()
        graph = self.graphs.get_graph(key)
        if graph is None:
            logger.debug(f"Creating graph for {key = }")
        return graph

    def set_graph(self, graph: torch.cuda.CUDAGraph) -> None:
        key = self._get_graph_key()
        self.graphs.set_graph(key, graph)
        logger.debug(f"Setting graph for {key = }")


@dataclass
class OutputCopyHandle:
    """One launched batch's output copy: the host buffer it lands in and the event that says it did.

    Output copies ping-pong between two host buffers, so the copy of batch k+1 (enqueued before
    batch k is consumed) never overwrites the tokens batch k is about to read.
    """

    copy_done: torch.cuda.Event | None  # None when the copy was synchronous (CPU)
    host_output_ids: torch.Tensor


class ContinuousBatchingIOs:
    """Inputs and outputs for the continuous batching loop, with host-ahead pipelining.

    There is one device-side buffer set (so one CUDA-graph set and no doubled VRAM). Batches are
    staged in pinned host memory and copied to the device with one bulk transfer enqueued on the
    compute stream, so the copy is ordered after the previous forward pass by the stream itself -
    no events are needed on the input path, and there is no window in which a device buffer is
    written while a forward pass reads it. Sampled tokens are copied back on a dedicated stream and
    consumed by the host one batch later. This is vLLM's async-scheduling structure: the host
    schedules, prepares, and enqueues batch k+1 while the device still computes batch k, and only
    then blocks on batch k's output copy.

    CPU (k = batch index):

        CPU  |SCH k|PREP k|ENQ k|UPD k-1|SCH k+1|PREP k+1|ENQ k+1|UPD k| ...
        GPU        |H2D k|FWD k ........|H2D k+1|FWD k+1 ........|
        D2H                       |<-k|                     |<-k+1|

    Decode rows whose input token is still being computed hold ``TMP_TOKEN_ID``; the forward pass
    begins with :meth:`carry_over_tokens`, an on-device scatter that replaces them with the previous
    batch's sampled tokens (recorded inside the CUDA graph). The forward pass therefore never waits
    for the host to see the tokens it feeds back.

    On a CPU-only device the same code runs eagerly with a single buffer set and no streams.
    """

    def __init__(
        self,
        cache: PagedAttentionCache,
        config: PreTrainedConfig,
        device: torch.device | str,
        model_dtype: torch.dtype,
    ) -> None:
        device = torch.device(device)
        self.device = device
        self.max_num_batched_tokens = cache.max_num_batched_tokens

        self.device_buffers = StaticIOBuffers(cache, config, device, model_dtype)
        if device.type == "cuda":
            self.host_buffers = StaticIOBuffers(cache, config, torch.device("cpu"), model_dtype)
            self.compute_stream: torch.cuda.Stream | None = torch.cuda.Stream(device=device)
            self.d2h_stream: torch.cuda.Stream | None = torch.cuda.Stream(device=device)
            self._compute_over = torch.cuda.Event()
            copy_done_events: list[torch.cuda.Event | None] = [torch.cuda.Event(), torch.cuda.Event()]
        else:
            self.host_buffers = self.device_buffers
            self.compute_stream = None
            self.d2h_stream = None
            self._compute_over = None
            copy_done_events = [None, None]
        # Two output-copy slots (host buffer + completion event): the copy of batch k+1 is enqueued
        # before batch k is consumed, so it must land in the other buffer. On CPU the shared buffer set
        # means the next forward overwrites output_ids, so the snapshot is needed there too.
        self._output_copy_slots = [
            OutputCopyHandle(
                copy_done=event,
                host_output_ids=torch.empty_like(
                    self.device_buffers.output_ids,
                    device="cpu",
                    pin_memory=device.type == "cuda",
                ),
            )
            for event in copy_done_events
        ]
        self._output_copy_slot = 0

        # Mapping of the previously prepared (in-flight) batch, for the carry-over scatter.
        self._prev_req_id_to_new_token_position: dict[str, int] = {}

    # ------------------------------------------------------------------ batch preparation

    def prepare_batch_tensors(
        self,
        requests_in_batch: list[FutureRequestState],
        use_decode_fast_path: bool,
        num_q_tokens: int,
        max_kv_read: int,
    ) -> None:
        self._prev_req_id_to_new_token_position = self.host_buffers.req_id_to_new_token_position
        self.host_buffers.prepare_batch_tensors(requests_in_batch, use_decode_fast_path, num_q_tokens, max_kv_read)
        self._fill_carry_over_ids()

    def _fill_carry_over_ids(self) -> None:
        """Fill the host carry-over row: for each request with a new-token row in both the in-flight batch and the
        one being prepared, map its input position to its output row in the in-flight batch. The output row is the
        request's insertion rank in the in-flight mapping, because sampled tokens land contiguously in logits-row
        order."""

        carry_over_ids = [-1] * self.max_num_batched_tokens
        next_positions = self.host_buffers.req_id_to_new_token_position
        for output_row, req_id in enumerate(self._prev_req_id_to_new_token_position):
            input_position = next_positions.get(req_id)
            if input_position is not None:
                carry_over_ids[input_position] = output_row
        self.host_buffers.carry_over_ids.copy_(torch.tensor(carry_over_ids, dtype=torch.int32))

    def get_model_kwargs(self, use_padding: bool = False) -> dict[str, Any]:
        self.host_buffers.finalize_batch(use_padding)
        if self.compute_stream is not None:
            with torch.cuda.stream(self.compute_stream):
                # Ordered after the previous forward by the stream; the last output copy must also have
                # drained before this batch's sampler overwrites output_ids.
                last_copy_done = self._output_copy_slots[1 - self._output_copy_slot].copy_done
                self.compute_stream.wait_event(last_copy_done)
                self.host_buffers.transfer_inputs_to(self.device_buffers)
        return self.device_buffers.get_model_kwargs(use_padding=use_padding)

    def carry_over_tokens(
        self,
        input_ids: torch.Tensor,
        carry_over_ids: torch.Tensor,
        prev_output_ids: torch.Tensor,
    ) -> None:
        """Replace placeholder input tokens with the previous batch's sampled tokens, on device.

        Runs at the start of the forward pass (and is recorded in CUDA graphs if they are enabled), so the
        token feedback from batch k to batch k+1 never leaves the device.
        """

        # Compute tokens to carry over and the corresponding mask
        carried_over_ids = prev_output_ids[0, carry_over_ids]
        carried_over_mask = (carry_over_ids != -1).int()
        # Truncate everything to the right size
        carried_over_ids = carried_over_ids[: input_ids.size(1)]
        carried_over_mask = carried_over_mask[: input_ids.size(1)]
        # Perform the carry over
        input_ids[0] = carried_over_ids * carried_over_mask + input_ids[0] * (1 - carried_over_mask)

    def get_cb_kwargs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the tensors used inside the generation step that are not inputs to the model forward pass:
        the carry-over mapping, the previous output ids to carry from, and the output ids to sample into. With a
        single buffer set the previous and current output tensor are the same storage; the carry-over scatter
        reads it before the sampler overwrites it, sequentially on the compute stream."""

        output_ids = self.device_buffers.output_ids
        return self.device_buffers.carry_over_ids, output_ids, output_ids

    # ------------------------------------------------------------------ output consumption

    def enqueue_output_copy(self) -> OutputCopyHandle:
        """Enqueue the device-to-host copy of the sampled tokens and return its handle.

        The copy runs on a dedicated stream after the forward pass finishes, so it overlaps whatever the
        host enqueues next on the compute stream. The caller passes the returned handle back to
        :meth:`consume_output_tokens` when it is ready to block on this batch's tokens.
        """

        handle = self._output_copy_slots[self._output_copy_slot]
        self._output_copy_slot = 1 - self._output_copy_slot
        if self.compute_stream is None:
            handle.host_output_ids.copy_(self.device_buffers.output_ids)
            return handle
        self.compute_stream.record_event(self._compute_over)
        self.d2h_stream.wait_event(self._compute_over)
        with torch.cuda.stream(self.d2h_stream):
            handle.host_output_ids.copy_(self.device_buffers.output_ids, non_blocking=True)
        self.d2h_stream.record_event(handle.copy_done)
        return handle

    def consume_output_tokens(self, output_copy: OutputCopyHandle, num_new_tokens: int) -> list[int]:
        """Block until the given output copy has landed and return its sampled tokens."""

        if output_copy.copy_done is not None:
            with nvtx.range("cb.output_wait"):
                output_copy.copy_done.synchronize()
        return output_copy.host_output_ids[0, :num_new_tokens].tolist()

    # ------------------------------------------------------------------ pass-throughs and state

    @property
    def requests_in_batch(self) -> list[FutureRequestState]:
        return self.host_buffers.requests_in_batch

    @property
    def use_block_table(self) -> bool:
        return self.host_buffers.use_block_table

    @property
    def true_num_logits(self) -> int:
        return self.host_buffers.true_num_logits

    @property
    def output_ids(self) -> torch.Tensor:
        return self.device_buffers.output_ids

    def get_cumulative_seqlens(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.host_buffers.get_cumulative_seqlens()

    def get_graph(self) -> torch.cuda.CUDAGraph | None:
        return self.device_buffers.get_graph()

    def prefill_graph_context(self) -> PrefillGraphContext:
        return self.device_buffers.prefill_graph_context()

    def set_graph(self, graph: torch.cuda.CUDAGraph) -> None:
        self.device_buffers.set_graph(graph)

    def clear_batch_history(self) -> None:
        """Forget the previously prepared batch, so the next batch carries nothing over (used after warmup)."""

        self._prev_req_id_to_new_token_position = {}
        self.host_buffers.req_id_to_new_token_position = {}

    def reset(self) -> None:
        """Reset all state for a new generation loop."""

        if self.compute_stream is not None:
            self.compute_stream.synchronize()
            self.d2h_stream.synchronize()
        self.host_buffers.reset()
        if self.device_buffers is not self.host_buffers:
            self.device_buffers.reset()
        self._prev_req_id_to_new_token_position = {}
