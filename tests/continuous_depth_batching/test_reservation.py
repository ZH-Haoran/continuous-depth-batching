"""Worst-case KV reservation used by the ``reserve`` cache-pressure mode.

``reserved_peak_blocks`` bounds the peak blocks a request can hold. The scheduler's block allocator
allocates exactly ``ceil(tokens / block_size)`` blocks on demand and holds no spare headroom block
(matching vLLM v1), so the peak is exactly ``ceil(total / block_size)`` - the reservation is tight.
"""

from dataclasses import dataclass

from looped_cdb.utils import reservation_admissible_prefix, reserved_peak_blocks


@dataclass
class _Req:
    initial_tokens: list[int]
    max_new_tokens: int | None


def test_reserved_peak_blocks_bounds_prompt_plus_generation() -> None:
    # 10 prompt + 6 generated = 16 tokens -> 4 blocks of 4.
    assert reserved_peak_blocks(10, 6, max_model_len=100, block_size=4) == 4
    # A partial block still counts as a whole block (ceil).
    assert reserved_peak_blocks(10, 7, max_model_len=100, block_size=4) == 5


def test_reserved_peak_blocks_caps_at_max_model_len() -> None:
    # Unbounded generation is bounded by max_model_len.
    assert reserved_peak_blocks(50, None, max_model_len=64, block_size=16) == 4
    # A budget that would exceed max_model_len is clamped to it.
    assert reserved_peak_blocks(50, 60, max_model_len=64, block_size=16) == 4


def test_reserved_peak_blocks_is_tight_at_a_block_boundary() -> None:
    # A prompt of max_model_len - 1 with a single generated token grows to exactly max_model_len tokens
    # on a cache whose size is a multiple of block_size. The exact-ceil allocator holds ceil(16/4) = 4
    # blocks here (no lookahead block), so the reservation is exactly 4 - the case that would have
    # over-committed under an allocator that rounds up one block past the tokens.
    assert reserved_peak_blocks(15, 1, max_model_len=16, block_size=4) == 4


def test_admissible_prefix_stops_head_of_line_when_next_would_overcommit() -> None:
    waiting = [
        _Req([1, 2, 3, 4], 4),  # 8 tokens -> 2 blocks
        _Req([1, 2, 3, 4], 8),  # 12 tokens -> 3 blocks
        _Req(list(range(12)), 8),  # 20 tokens -> 5 blocks (would overflow)
        _Req([1], 1),  # 2 tokens -> 1 block, but blocked behind the big one
    ]
    admitted = reservation_admissible_prefix(
        active_states=[], waiting_states=waiting, num_blocks=8, block_size=4, max_model_len=100
    )

    # 2 + 3 = 5 fit; the 5-block request overflows (>8) and stops admission at the head of line,
    # so the small request behind it waits too rather than jumping the queue.
    assert admitted == waiting[:2]


def test_admissible_prefix_counts_active_reservations_as_committed() -> None:
    active = [_Req(list(range(20)), 4)]  # 24 tokens -> 6 blocks already committed
    waiting = [_Req([1, 2, 3, 4], 4), _Req([1, 2, 3, 4], 8)]  # 2 blocks, then 3 blocks
    admitted = reservation_admissible_prefix(
        active_states=active, waiting_states=waiting, num_blocks=8, block_size=4, max_model_len=100
    )

    # 6 committed + 2 = 8 fits; the next (+3) would reach 11 > 8, so only the first is admitted.
    assert admitted == waiting[:1]


def test_admissible_prefix_admits_all_when_everything_fits() -> None:
    waiting = [_Req([1], 1), _Req([1, 2], 1)]
    admitted = reservation_admissible_prefix(
        active_states=[], waiting_states=waiting, num_blocks=8, block_size=4, max_model_len=100
    )
    assert admitted == waiting
