import torch


def _make_shared_history_cache(batch_size: int, seq_len: int, hidden_size: int) -> torch.Tensor:
    values = torch.arange(batch_size * seq_len * hidden_size, dtype=torch.float32)
    return values.reshape(batch_size, seq_len, hidden_size) / 100.0


def _mixed_depth_shared_kv_step(
    cache: torch.Tensor,
    hidden: torch.Tensor,
    recurrent_steps: torch.Tensor,
    write_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_idx = torch.arange(hidden.size(0))
    history = cache[batch_idx]
    depth_bias = recurrent_steps.to(hidden.dtype).unsqueeze(1) / 10.0
    hidden_out = hidden + history.sum(dim=1) + depth_bias
    updated_cache = cache.clone()
    updated_cache[batch_idx, write_positions] = hidden_out
    return hidden_out, updated_cache


def _scalar_shared_kv_reference(
    cache: torch.Tensor,
    hidden: torch.Tensor,
    recurrent_steps: torch.Tensor,
    write_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    updated_cache = cache.clone()
    for row, recurrent_step in enumerate(recurrent_steps.tolist()):
        hidden_out = hidden[row] + cache[row].sum(dim=0) + recurrent_step / 10.0
        rows.append(hidden_out)
        updated_cache[row, int(write_positions[row])] = hidden_out
    return torch.stack(rows), updated_cache


def test_mixed_depth_shared_kv_launch_matches_scalar_reference() -> None:
    cache = _make_shared_history_cache(batch_size=4, seq_len=6, hidden_size=3)
    hidden = torch.tensor(
        [
            [0.1, 0.2, 0.3],
            [1.0, 1.1, 1.2],
            [2.0, 2.1, 2.2],
            [3.0, 3.1, 3.2],
        ],
        dtype=torch.float32,
    )
    recurrent_steps = torch.tensor([3, 0, 2, 1], dtype=torch.long)
    write_positions = torch.tensor([5, 4, 3, 2], dtype=torch.long)

    mixed_hidden, mixed_cache = _mixed_depth_shared_kv_step(cache, hidden, recurrent_steps, write_positions)
    reference_hidden, reference_cache = _scalar_shared_kv_reference(cache, hidden, recurrent_steps, write_positions)

    torch.testing.assert_close(mixed_hidden, reference_hidden)
    torch.testing.assert_close(mixed_cache, reference_cache)
