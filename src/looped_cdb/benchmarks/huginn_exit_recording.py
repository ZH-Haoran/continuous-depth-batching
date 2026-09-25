"""Record every output token's convergence trajectory from Huginn into a workload bundle.

Ouro records an exit *PDF* from its trained gate, and depths are derived offline
at any threshold. Huginn has no gate: it exits when consecutive recurrent states
stop changing, so there is no distribution over depths. The threshold-free
analogue is the criterion's trajectory, the value it takes after each recurrent
step, from which the depth at any threshold is the first step that falls below
it.

Each request's prompt and reference output run through the model in one
teacher-forced pass at full depth, keeping the criterion value at every step.
The values at the position that predicts each output token (position ``i``
predicts token ``i + 1``) are packed into a
:class:`~looped_cdb.benchmarks.workload.Workload` as ``exit_values``.

As with the gate recording, every token attends to a prefix computed at full
recurrent depth. A real early-exit decode would instead attend to KV cached at
each token's own shallower exit depth, so the recorded schedule approximates,
but does not exactly reproduce, an autoregressive early-exit decode.

The recurrent state is zeroed rather than drawn randomly, so a bundle is
reproducible; upstream draws it per forward.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .datasets import SampledRequest
from .workload import Workload


@torch.no_grad()
def record_request_values(
    model: Any,
    request: SampledRequest,
    *,
    criterion: Any,
    max_depth: int,
) -> np.ndarray:
    """Return the ``(output_len, max_depth)`` criterion trajectory for one request.

    Row ``j`` holds the criterion after each recurrent step at the position that
    predicts output token ``j``.
    """

    device = next(model.parameters()).device
    token_ids = list(request.prompt_ids) + list(request.output_ids)
    input_ids = torch.tensor([token_ids], device=device)

    freqs_cis = model.select_freqs_cis(input_ids.shape[1], position_ids=None)
    injection = model.run_prelude(input_ids, freqs_cis)
    state = torch.zeros_like(injection)

    per_step: list[torch.Tensor] = []
    for step in range(max_depth):
        next_state = model.run_core_step(state, injection, freqs_cis, step)
        per_step.append(criterion(next_state, state)[0])
        state = next_state

    values = torch.stack(per_step, dim=-1)  # (seq, max_depth)
    # Position i predicts token i + 1, so the row for output token j is the
    # position just before it.
    first = len(request.prompt_ids) - 1
    sliced = values[first : first + len(request.output_ids)]
    return sliced.float().cpu().numpy().astype(np.float16)


def record_workload(
    model: Any,
    requests: list[SampledRequest],
    *,
    criterion: Any,
    max_depth: int,
    meta: dict[str, Any] | None = None,
    progress_every: int = 50,
) -> Workload:
    """Record a convergence-trajectory workload bundle for ``requests``."""

    if not requests:
        raise ValueError("requests must not be empty")

    rows: list[np.ndarray] = []
    offsets = [0]
    for index, request in enumerate(requests):
        rows.append(record_request_values(model, request, criterion=criterion, max_depth=max_depth))
        offsets.append(offsets[-1] + len(request.output_ids))
        if progress_every and (index + 1) % progress_every == 0:
            print(f"  recorded {index + 1}/{len(requests)} requests", flush=True)

    return Workload(
        ids=[request.id for request in requests],
        input_lens=np.array([request.input_len for request in requests], dtype=np.int32),
        output_lens=np.array([len(request.output_ids) for request in requests], dtype=np.int32),
        offsets=np.array(offsets, dtype=np.int64),
        exit_values=np.concatenate(rows, axis=0),
        meta=meta or {},
    )
