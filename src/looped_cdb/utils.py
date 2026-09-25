from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal, Protocol, get_args

import torch

# KV-cache-pressure policy shared by the CB and CDB engines. See each config's ``kv_pressure_mode``.
#   none      - greedy admission, no reservation and no preemption; raise if the cache fills (the
#               original behavior, for workloads sized to fit and for microbenchmarks).
#   reserve   - admit only if a request's worst-case peak KV fits; never preempt.
#   recompute - greedy admission; preempt by soft reset (recompute) on exhaustion.
#   offload   - greedy admission; preempt by CPU swap on exhaustion.
KVPressureMode = Literal["none", "reserve", "recompute", "offload"]

# Scheduler defaults shared by the engine configs, the schedulers and the benchmark CLI.
DEFAULT_MAX_NUM_SEQS = 256
# Default prefill admission threshold as a fraction of the resident cap: a launch waits for
# ``max_num_seqs // MIN_FREE_SLOTS_DIVISOR`` open slots.
MIN_FREE_SLOTS_DIVISOR = 8


def resolve_min_free_slots(min_free_slots: int | None, max_num_seqs: int) -> int:
    """The open resident slots a prefill launch waits for.

    ``None`` resolves to an eighth of ``max_num_seqs``, with a minimum of one.
    Explicit values are capped at ``max_num_seqs``.
    """

    if min_free_slots is None:
        return max(1, max_num_seqs // MIN_FREE_SLOTS_DIVISOR)
    return min(min_free_slots, max_num_seqs)


def validate_kv_pressure_config(kv_pressure_mode: str, cpu_offload_space: float | None) -> None:
    """Validate the KV-pressure mode and its CPU-pool knob, shared by the CB and CDB configs.

    ``cpu_offload_space`` sizes the pinned CPU swap pool and is meaningful only for ``"offload"``:
    that mode requires it (> 0), every other mode forbids it, so a stray value cannot silently size a
    pool that is never used. Raises :class:`ValueError` on any violation.
    """

    if kv_pressure_mode not in get_args(KVPressureMode):
        raise ValueError(f"kv_pressure_mode must be one of {get_args(KVPressureMode)}, but got {kv_pressure_mode!r}")
    if cpu_offload_space is not None and cpu_offload_space < 0:
        raise ValueError(f"cpu_offload_space must be non-negative when set, but got {cpu_offload_space}")
    if kv_pressure_mode == "offload":
        if not cpu_offload_space:
            raise ValueError("kv_pressure_mode='offload' requires cpu_offload_space > 0 (the pinned CPU pool size)")
    elif cpu_offload_space:
        raise ValueError(
            f"cpu_offload_space is only used by kv_pressure_mode='offload', but kv_pressure_mode="
            f"{kv_pressure_mode!r} was set with cpu_offload_space={cpu_offload_space}"
        )


def normalize_max_new_tokens(max_new_tokens: int | list[int], num_requests: int) -> list[int]:
    """Expand a scalar ``max_new_tokens`` to one value per request.

    A list is returned unchanged (callers validate its length separately); a scalar is
    broadcast to ``num_requests`` entries so downstream code can index per request without
    re-checking the ``int | list[int]`` union at every use site.
    """

    if isinstance(max_new_tokens, list):
        return max_new_tokens
    return [max_new_tokens] * num_requests


def cap_max_new_tokens_to_model_len(
    input_ids: list[list[int]],
    max_new_tokens: list[int],
    max_model_len: int | None,
) -> list[int]:
    """Reject over-length prompts and cap generation so prompt + generated <= ``max_model_len``.

    Mirrors vLLM's ``max_model_len`` semantics for the batch API: a prompt with no room for
    even one generated token (``len(prompt) >= max_model_len``) is rejected by raising
    ``ValueError`` (aborting the batch), and every other request has its ``max_new_tokens``
    clamped down to ``max_model_len - len(prompt)`` so decoding stops on reaching the limit (a
    length finish). ``max_new_tokens`` must already be per-request (see
    :func:`normalize_max_new_tokens`). Returns the clamped list; a no-op when ``max_model_len``
    is ``None``. The clamp never drops below 1 because rejected prompts are the only ones with
    zero room.
    """

    if max_model_len is None:
        return max_new_tokens
    too_long = [(idx, len(prompt)) for idx, prompt in enumerate(input_ids) if len(prompt) >= max_model_len]
    if too_long:
        preview = ", ".join(f"request {idx} (len {length})" for idx, length in too_long[:5])
        suffix = " ..." if len(too_long) > 5 else ""
        raise ValueError(
            f"{len(too_long)} prompt(s) have length >= max_model_len ({max_model_len}) and leave no room "
            f"for a generated token: {preview}{suffix}. Filter or truncate them before serving."
        )
    return [min(mnt, max_model_len - len(prompt)) for mnt, prompt in zip(max_new_tokens, input_ids, strict=True)]


class SupportsReservation(Protocol):
    """A request whose worst-case KV footprint can be bounded from its prompt and generation budget."""

    initial_tokens: list[int]
    max_new_tokens: int | None


def reserved_peak_blocks(initial_len: int, max_new_tokens: int | None, max_model_len: int, block_size: int) -> int:
    """Worst-case number of KV blocks a request can ever hold over its lifetime.

    A request grows to at most ``min(prompt + max_new_tokens, max_model_len)`` tokens (unbounded
    generation is capped by ``max_model_len``), whose KV occupies ``ceil(total / block_size)`` blocks.
    The scheduler's block allocator (:meth:`_allocate_blocks_if_needed`) allocates exactly this many
    on demand and holds no spare headroom block (matching vLLM v1, which allocates
    ``ceil(tokens / block_size)`` and grabs the next block only when the current one fills), so
    ``ceil(total / block_size)`` is the exact peak - not an over-estimate. This is the amount
    reservation-based admission commits up front so an admitted request can always allocate as it
    decodes, without ever preempting.

    The reservation horizon is the request's ``max_new_tokens``. In a recorded-workload replay that is
    set to the exact number of tokens the request will emit, so ``reserve`` reserves tightly - an
    oracle-length reservation, i.e. the optimistic bound for the policy (reserve with a perfect length
    predictor). A production ``reserve`` that does not know the output length would instead pass the
    request's declared cap or ``max_model_len`` here and reserve more conservatively (fewer concurrent
    requests). This benchmark deliberately keeps the oracle horizon rather than adding a separate
    reservation cap: it is the strongest form of ``reserve`` to compare preemption against.
    """

    total_len = max_model_len if max_new_tokens is None else min(initial_len + max_new_tokens, max_model_len)
    return -(-total_len // block_size)


def reservation_admissible_prefix(
    active_states: Iterable[SupportsReservation],
    waiting_states: Sequence[SupportsReservation],
    *,
    num_blocks: int,
    block_size: int,
    max_model_len: int,
) -> list[SupportsReservation]:
    """Return the longest arrival-order prefix of ``waiting_states`` that fits the free reservation.

    Every active request already commits its worst-case peak (:func:`reserved_peak_blocks`); the
    sum of those peaks is the committed reservation. Waiting prompts are admitted in order while the
    running total stays within ``num_blocks``, stopping at the first that does not fit (head-of-line,
    so a large prompt waits for the running set to drain rather than being skipped and starving it).
    Because no admitted set can ever over-commit the pool, a reserved request never needs preemption.
    """

    def peak(state: SupportsReservation) -> int:
        return reserved_peak_blocks(len(state.initial_tokens), state.max_new_tokens, max_model_len, block_size)

    committed = sum(peak(state) for state in active_states)
    admitted: list[SupportsReservation] = []
    for state in waiting_states:
        need = peak(state)
        if committed + need > num_blocks:
            break
        committed += need
        admitted.append(state)
    return admitted


def best_attention_backend() -> str:
    """Choose the FlashAttention backend for this CUDA host."""
    if not torch.cuda.is_available():
        raise RuntimeError("FlashAttention requires CUDA; no CPU or SDPA fallback is supported")
    major, _minor = torch.cuda.get_device_capability()
    if major >= 9:
        return "flash_attention_3"
    return "flash_attention_2"


def require_flash_attention() -> str:
    """Return the flash-attention backend string, or raise before expensive setup.

    Raises:
        RuntimeError: if no flash-attention backend is available.
    """
    attn_implementation = best_attention_backend()
    if not attn_implementation.startswith("flash_attention"):
        raise RuntimeError(
            "A flash-attention backend is required but none is available; "
            f"best available backend was {attn_implementation!r}. "
            "Install flash-attn (uv sync --extra fa2) or run on hardware that supports it."
        )
    return attn_implementation
