"""vLLM-style max_model_len: admission rejection, generation cap, and workload truncation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from transformers import PreTrainedConfig

from looped_cdb.benchmarks.workload import Workload
from looped_cdb.continuous_batching.config import ContinuousBatchingConfig
from looped_cdb.continuous_depth_batching import ContinuousDepthBatchingConfig
from looped_cdb.utils import cap_max_new_tokens_to_model_len

# --------------------------------------------------------------------------- util


def test_cap_none_is_noop() -> None:
    mnt = [4, 4, 4]
    assert cap_max_new_tokens_to_model_len([[1], [2], [3]], mnt, None) is mnt


def test_cap_clamps_to_remaining_budget() -> None:
    # prompts of length 2, 8, 3 against a 10-token budget leave 8, 2, 7 tokens.
    capped = cap_max_new_tokens_to_model_len([[0] * 2, [0] * 8, [0] * 3], [5, 5, 5], 10)
    assert capped == [5, 2, 5]


def test_cap_leaves_room_for_at_least_one_token_at_boundary() -> None:
    # A prompt one below the limit may generate exactly one token.
    assert cap_max_new_tokens_to_model_len([[0] * 9], [16], 10) == [1]


def test_cap_rejects_prompt_with_no_room() -> None:
    # len(prompt) == max_model_len leaves zero room, so it is rejected like an over-length prompt.
    with pytest.raises(ValueError, match="max_model_len"):
        cap_max_new_tokens_to_model_len([[0] * 5, [0] * 10], [4, 4], 10)
    with pytest.raises(ValueError, match="max_model_len"):
        cap_max_new_tokens_to_model_len([[0] * 11], [4], 10)


# --------------------------------------------------------------------------- config validation


def _cb_model_config(**overrides: object) -> PreTrainedConfig:
    values = {
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "hidden_size": 4,
        "vocab_size": 16,
        "sliding_window": None,
        "layer_types": None,
        "_attn_implementation": "paged|flash_attention_3",
    }
    values.update(overrides)
    return PreTrainedConfig(**values)


def _cdb_model_config() -> SimpleNamespace:
    return SimpleNamespace(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        hidden_size=4,
        vocab_size=16,
        sliding_window=None,
        layer_types=["full_attention"] * 6,
        _attn_implementation="paged|flash_attention_3",
        total_ut_steps=3,
    )


def test_cb_config_derives_block_table_width_from_max_model_len() -> None:
    # The fast decode path sizes the per-request block table as ceil(max_model_len / block_size).
    assert ContinuousBatchingConfig(block_size=256, max_model_len=1024).decode_block_table_width() == 4
    assert ContinuousBatchingConfig(block_size=256, max_model_len=1025).decode_block_table_width() == 5


def test_cb_config_rejects_non_positive_max_model_len() -> None:
    with pytest.raises(ValueError, match="max_model_len must be positive"):
        ContinuousBatchingConfig(num_blocks=1024, max_model_len=0).validate(_cb_model_config())


def test_cdb_config_derives_block_table_width_from_max_model_len() -> None:
    config = ContinuousDepthBatchingConfig(block_size=256, max_model_len=1025, max_recurrent_steps=3)
    assert config.decode_block_table_width() == 5


# --------------------------------------------------------------------------- workload truncation


def _explicit_workload() -> Workload:
    """Four requests with distinct per-token depths so slicing can be checked by value."""

    input_lens = np.array([2, 8, 3, 12], dtype=np.int32)
    output_lens = np.array([5, 4, 10, 6], dtype=np.int32)
    offsets = np.zeros(5, dtype=np.int64)
    np.cumsum(output_lens.astype(np.int64), out=offsets[1:])
    # request i emits (i + 1) at every one of its output tokens.
    exit_depths = np.concatenate([np.full(int(n), i + 1, dtype=np.int32) for i, n in enumerate(output_lens)])
    return Workload(
        ids=[f"r{i}" for i in range(4)],
        input_lens=input_lens,
        output_lens=output_lens,
        offsets=offsets,
        exit_depths=exit_depths,
        max_depth=4,
    )


def test_apply_max_model_len_none_is_noop() -> None:
    workload = _explicit_workload()
    assert workload.apply_max_model_len(None) is workload


def test_apply_max_model_len_returns_self_when_every_request_already_fits() -> None:
    # Callers apply this before slicing, so a bundle recorded at this context length would
    # otherwise rebuild a schedule of millions of rows to arrive at the same workload.
    workload = _explicit_workload()
    budget = int((workload.input_lens + workload.output_lens).max())

    assert workload.apply_max_model_len(budget) is workload
    assert workload.apply_max_model_len(budget - 1) is not workload  # one request now truncates


def test_apply_max_model_len_drops_and_truncates() -> None:
    # budget 10: r0 keeps 5, r1 truncates 4->2, r2 truncates 10->7, r3 (input 12) is dropped.
    out = _explicit_workload().apply_max_model_len(10)

    assert out.ids == ["r0", "r1", "r2"]
    assert out.output_lens.tolist() == [5, 2, 7]
    assert out.input_lens.tolist() == [2, 8, 3]
    assert out.offsets.tolist() == [0, 5, 7, 14]
    # every kept token has input + output within budget
    for input_len, output_len in zip(out.input_lens.tolist(), out.output_lens.tolist(), strict=True):
        assert input_len + output_len <= 10
    # the per-token schedule is sliced to the retained tokens, in order
    assert out.exit_depths.tolist() == [1] * 5 + [2] * 2 + [3] * 7


def test_apply_max_model_len_keeps_schedule_aligned_with_outputs() -> None:
    out = _explicit_workload().apply_max_model_len(10)
    depths = out.materialize_depths(min_exit_step=1)
    assert [len(row) for row in depths] == out.output_lens.tolist()


def test_apply_max_model_len_raises_when_all_dropped() -> None:
    with pytest.raises(ValueError, match="no request fits"):
        _explicit_workload().apply_max_model_len(1)
