"""Request selection for the workload recorder: filter first, then subsample."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from create_workload_json import select_requests


def whitespace_encode(text: str, add_special_tokens: bool) -> list[int]:
    """Trivial tokenizer: one id per word, with a leading BOS (id 1) when asked."""

    ids = [10 + (len(word) % 5) for word in text.split()]
    return [1, *ids] if add_special_tokens else ids


def _write_sharegpt(path: Path, count: int, long_every: int = 0) -> None:
    """Write ``count`` two-turn conversations; every ``long_every``-th reply is over-long."""

    records = []
    for i in range(count):
        reply = " ".join(["word"] * 40) if long_every and i % long_every == 0 else f"reply number {i} indeed"
        records.append(
            {
                "id": f"c{i}",
                "conversations": [
                    {"from": "human", "value": f"prompt number {i} here"},
                    {"from": "gpt", "value": reply},
                ],
            }
        )
    path.write_text(json.dumps(records))


def _select(path: Path, **overrides):
    kwargs = {
        "max_model_len": 20,
        "min_prompt_len": 1,
        "min_output_len": 1,
        "limit": None,
        "shuffle_seed": 0,
    }
    kwargs.update(overrides)
    return select_requests("sharegpt", path, whitespace_encode, **kwargs)


def test_limit_is_exact_even_when_the_context_filter_drops_requests(tmp_path: Path) -> None:
    # Subsampling before the context filter returns fewer requests than asked for, by an
    # amount that varies with max_model_len. The bundle size must depend on neither.
    path = tmp_path / "sharegpt.json"
    _write_sharegpt(path, count=30, long_every=3)  # a third of the replies exceed the context

    requests, dropped, sampled_from = _select(path, limit=10)

    assert len(requests) == 10
    assert all(r.input_len + r.output_len <= 20 for r in requests)
    assert dropped == 10
    assert sampled_from == 20  # the pool the 10 were drawn from


def test_limit_takes_a_prefix_of_the_unlimited_run(tmp_path: Path) -> None:
    # What makes a subsampled bundle reproducible: same seed and filters, same requests, so
    # the smaller bundle is a prefix of the larger one rather than an unrelated draw.
    path = tmp_path / "sharegpt.json"
    _write_sharegpt(path, count=30, long_every=3)

    everything, _, pool = _select(path, shuffle_seed=3)
    subsample, _, subsample_pool = _select(path, shuffle_seed=3, limit=7)

    assert [r.id for r in subsample] == [r.id for r in everything[:7]]
    assert subsample_pool == pool == len(everything)


def test_asking_for_more_requests_than_survive_is_an_error(tmp_path: Path) -> None:
    # Silently returning a smaller bundle would misreport the workload size downstream.
    path = tmp_path / "sharegpt.json"
    _write_sharegpt(path, count=30, long_every=3)

    with pytest.raises(ValueError, match="only 20 survive filtering"):
        _select(path, limit=21)


def test_without_a_limit_every_surviving_request_is_kept(tmp_path: Path) -> None:
    path = tmp_path / "sharegpt.json"
    _write_sharegpt(path, count=12, long_every=4)

    requests, dropped, sampled_from = _select(path)

    assert len(requests) == sampled_from == 9
    assert dropped == 3
