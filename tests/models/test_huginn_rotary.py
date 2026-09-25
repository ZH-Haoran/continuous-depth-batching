"""Tests for Huginn's rotary embedding construction.

Continuous batching packs decode work into a single sequence, so a rotary
implementation that silently assumes ``batch == 1`` passes every serving test
while breaking ordinary batched generation.
"""

from __future__ import annotations

import pytest
import torch

from looped_cdb.models.huginn.modeling_huginn import (
    apply_rotary_emb_complex_like,
    freqs_cis_for_positions,
    rotary_inv_freqs,
)

HEAD_DIM = 8
ROPE_BASE = 50_000.0


@pytest.fixture
def inv_freqs() -> torch.Tensor:
    return rotary_inv_freqs(HEAD_DIM, ROPE_BASE)


def test_shape_keeps_batch_dimension(inv_freqs: torch.Tensor) -> None:
    flat = freqs_cis_for_positions(torch.arange(6), inv_freqs)
    batched = freqs_cis_for_positions(torch.zeros(3, 6, dtype=torch.long), inv_freqs)

    assert flat.shape == (1, 6, 1, HEAD_DIM // 2, 2)
    assert batched.shape == (3, 6, 1, HEAD_DIM // 2, 2)


def test_one_dimensional_positions_match_a_single_row(inv_freqs: torch.Tensor) -> None:
    positions = torch.arange(6)

    assert torch.equal(
        freqs_cis_for_positions(positions, inv_freqs),
        freqs_cis_for_positions(positions.unsqueeze(0), inv_freqs),
    )


def test_each_sequence_rotates_by_its_own_positions(inv_freqs: torch.Tensor) -> None:
    """Sequences at different offsets must not share one sequence's rotation."""

    positions = torch.tensor([[0, 1, 2, 3], [10, 11, 12, 13]])

    batched = freqs_cis_for_positions(positions, inv_freqs)

    for row in range(positions.shape[0]):
        assert torch.equal(batched[row : row + 1], freqs_cis_for_positions(positions[row], inv_freqs))


def test_broadcasts_against_batched_queries_and_keys(inv_freqs: torch.Tensor) -> None:
    """The regression: batch > 1 previously flattened positions and failed to broadcast."""

    batch, seq, heads = 3, 4, 2
    q = torch.randn(batch, seq, heads, HEAD_DIM)
    k = torch.randn(batch, seq, heads, HEAD_DIM)
    positions = torch.arange(seq).expand(batch, seq)

    freqs = freqs_cis_for_positions(positions, inv_freqs)
    rotated_q, rotated_k = apply_rotary_emb_complex_like(q, k, freqs)

    assert rotated_q.shape == q.shape
    assert rotated_k.shape == k.shape


def test_rotation_preserves_norm(inv_freqs: torch.Tensor) -> None:
    """Rotary embedding is a rotation, so per-pair magnitude is unchanged."""

    q = torch.randn(2, 5, 3, HEAD_DIM)
    k = torch.randn(2, 5, 3, HEAD_DIM)
    freqs = freqs_cis_for_positions(torch.arange(5).expand(2, 5), inv_freqs)

    rotated_q, _ = apply_rotary_emb_complex_like(q, k, freqs)

    torch.testing.assert_close(rotated_q.norm(dim=-1), q.norm(dim=-1), rtol=1e-5, atol=1e-5)


def test_position_zero_is_the_identity(inv_freqs: torch.Tensor) -> None:
    q = torch.randn(1, 3, 2, HEAD_DIM)
    k = torch.randn(1, 3, 2, HEAD_DIM)
    freqs = freqs_cis_for_positions(torch.zeros(1, 3, dtype=torch.long), inv_freqs)

    rotated_q, rotated_k = apply_rotary_emb_complex_like(q, k, freqs)

    torch.testing.assert_close(rotated_q, q, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(rotated_k, k, rtol=1e-6, atol=1e-6)


def test_rejects_higher_rank_positions(inv_freqs: torch.Tensor) -> None:
    with pytest.raises(ValueError, match=r"\(seq,\) or \(batch, seq\)"):
        freqs_cis_for_positions(torch.zeros(2, 3, 4, dtype=torch.long), inv_freqs)
