"""Record every output token's exit PDF from a model into a workload bundle.

Each request's prompt and reference output are run through the model in a single
forward pass with the early-exit gate enabled (``return_exit_pdf=True``), which
reports, at every position, the gate's probability of exiting at each recurrent
step. The distribution at the position that predicts each output token (position
``i`` predicts token ``i + 1``) is packed into a
:class:`~looped_cdb.benchmarks.workload.Workload`. Exit depths for any threshold
are derived from these PDFs offline, so the pass is threshold-free and only runs
once per (dataset, recurrent depth, gate).

The PDFs are read from the full reference output, so every token attends to a
prefix computed at full recurrent depth. A real early-exit decode would instead
attend to KV cached at each token's own (shallower) exit depth, so the recorded
schedule approximates -- but does not exactly reproduce -- an autoregressive
early-exit decode.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

from .datasets import SampledRequest
from .workload import Workload, slice_output_pdf


def group_by_token_budget(
    requests: Sequence[SampledRequest],
    *,
    max_num_batched_tokens: int,
    max_batch_size: int = 64,
) -> Iterator[list[SampledRequest]]:
    """Greedily group requests into microbatches under a padded-token budget.

    A batch is padded to its longest sequence, so its cost is
    ``len(batch) * longest_seq_len``. Requests are added while that cost stays
    within ``max_num_batched_tokens`` and the count within ``max_batch_size``. A request
    longer than the budget forms its own single-element batch. Pre-sort by length
    so similar-length requests batch together and padding waste stays low.
    """

    if max_num_batched_tokens < 1:
        raise ValueError(f"max_num_batched_tokens must be >= 1, got {max_num_batched_tokens}")
    if max_batch_size < 1:
        raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}")

    batch: list[SampledRequest] = []
    batch_max_len = 0
    for request in requests:
        seq_len = request.input_len + request.output_len
        candidate_max = max(batch_max_len, seq_len)
        padded_cost = candidate_max * (len(batch) + 1)
        if batch and (padded_cost > max_num_batched_tokens or len(batch) >= max_batch_size):
            yield batch
            batch, batch_max_len = [], 0
            candidate_max = seq_len
        batch.append(request)
        batch_max_len = candidate_max
    if batch:
        yield batch


def record_exit_pdfs(
    model: Any,
    requests: Sequence[SampledRequest],
    *,
    max_num_batched_tokens: int = 16384,
    max_batch_size: int = 64,
    device: str = "cuda",
    sort_by_length: bool = True,
    meta: dict[str, Any] | None = None,
    progress: bool = False,
) -> Workload:
    """Run ``requests`` through ``model`` and pack every output token's exit PDF.

    Each microbatch is a single padded forward with the early-exit gate enabled
    and ``return_exit_pdf=True`` (no KV cache: the whole sequence is seen at
    once). Requests are sorted by length only to pack similar-length requests
    into microbatches (low padding waste); this batching order is internal and
    the returned :class:`Workload` is assembled in the original ``requests``
    order, so the length sort never leaks into the on-disk replay order.
    """

    import torch

    order = (
        sorted(range(len(requests)), key=lambda i: requests[i].input_len + requests[i].output_len)
        if sort_by_length
        else list(range(len(requests)))
    )
    ordered = [requests[i] for i in order]
    recorded_pdf: list[np.ndarray | None] = [None] * len(requests)

    batches = list(
        group_by_token_budget(ordered, max_num_batched_tokens=max_num_batched_tokens, max_batch_size=max_batch_size)
    )
    done = 0
    for batch_index, batch in enumerate(batches):
        seqs = [req.prompt_ids + req.output_ids for req in batch]
        max_len = max(len(seq) for seq in seqs)
        input_ids = torch.zeros((len(seqs), max_len), dtype=torch.long)
        attention_mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
        for row, seq in enumerate(seqs):
            input_ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            attention_mask[row, : len(seq)] = 1
        # Right padding: real tokens occupy positions 0..len-1, so a plain arange
        # gives correct position ids for every real token (pad positions are masked).
        position_ids = torch.arange(max_len, dtype=torch.long).unsqueeze(0).expand(len(seqs), -1)

        with torch.no_grad():
            output = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                position_ids=position_ids.to(device),
                use_cache=False,
                use_early_exit_gate=True,
                return_exit_pdf=True,
                logits_to_keep=1,  # logits are unused; keep a single position to save time and memory
            )
        exit_pdf = getattr(output, "ouro_exit_pdf", None)
        if exit_pdf is None:
            raise RuntimeError(
                "model did not return ouro_exit_pdf; ensure the early-exit gate is enabled and supported"
            )
        exit_pdf = exit_pdf.to(dtype=torch.float32, device="cpu").numpy()

        for row, req in enumerate(batch):
            request_pdf = slice_output_pdf(exit_pdf[row], req.input_len, req.output_len).astype(np.float16)
            # Batching consumes ``ordered`` in order and drops nothing, so the request at batching
            # position ``done + row`` came from ``requests[order[done + row]]``. Keying on the
            # position rather than on ``id(req)`` keeps this correct even if the caller passes the
            # same request object twice, which identity keying would silently collapse.
            recorded_pdf[order[done + row]] = request_pdf
        done += len(batch)
        if progress:
            print(f"  microbatch {batch_index + 1}/{len(batches)}: {done}/{len(requests)} requests", flush=True)

    if done != len(requests):
        raise RuntimeError(f"recorded {done} of {len(requests)} requests; microbatching dropped some")

    # Emit in the original request order so the batching sort does not shape the artifact.
    ids = [req.id for req in requests]
    input_lens = [req.input_len for req in requests]
    output_lens = [req.output_len for req in requests]
    offsets = [0]
    for output_len in output_lens:
        offsets.append(offsets[-1] + output_len)
    exit_pdf_array = (
        np.concatenate(recorded_pdf, axis=0)
        if recorded_pdf
        else np.zeros((0, int(getattr(model.config, "total_ut_steps", 1))), dtype=np.float16)
    )
    return Workload(
        ids=ids,
        input_lens=np.array(input_lens, dtype=np.int32),
        output_lens=np.array(output_lens, dtype=np.int32),
        exit_pdf=exit_pdf_array,
        offsets=np.array(offsets, dtype=np.int64),
        meta=dict(meta or {}),
    )
