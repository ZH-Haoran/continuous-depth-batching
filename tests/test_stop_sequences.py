"""Multi-token stop sequences, from tokenizer strings down to the request state.

A stop string that tokenizes to more than one id cannot be an EOS id. In
few-shot prompting that is the common case: the answer ends when the model
opens the next exemplar (``Q:``), which is two tokens. Without suffix matching
the request runs to its length cap and the surplus is thrown away after
decoding, which wastes decode and skews any per-token statistic collected over
the generation.
"""

from __future__ import annotations

import pytest

from looped_cdb.continuous_batching.requests import RequestState as CBRequestState
from looped_cdb.continuous_depth_batching.requests import RequestState as CDBRequestState
from looped_cdb.continuous_depth_batching.requests import RequestStatus
from looped_cdb.eval.backends import (
    EngineBackend,
    engine_accepts,
    stop_string_eos_ids,
    stop_string_sequences,
)


class _FakeTokenizer:
    """Maps a few strings to fixed ids; everything else is one id per character."""

    eos_token_id = 99

    def __init__(self, table: dict[str, list[int]] | None = None) -> None:
        self.table = table or {}

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": self.table.get(text, [ord(c) for c in text])}

    def decode(self, token_ids, skip_special_tokens: bool = False) -> str:
        return "".join(chr(t) for t in token_ids)


#: Both engines carry their own RequestState; the stop rule must hold in each.
STATE_CLASSES = [CBRequestState, CDBRequestState]


@pytest.fixture(params=STATE_CLASSES, ids=["cb", "cdb"])
def state_cls(request):
    return request.param


def _decoding_state(cls, **kwargs):
    state = cls(request_id="r", initial_tokens=[1], max_new_tokens=64, **kwargs)
    state.status = RequestStatus.PENDING
    state.status = RequestStatus.PREFILLING
    state.status = RequestStatus.DECODING
    return state


def test_multi_token_stops_are_separated_from_eos_ids() -> None:
    tokenizer = _FakeTokenizer({"Q:": [81, 58], "<|end|>": [65505]})

    assert stop_string_sequences(tokenizer, ["Q:", "<|end|>"]) == [[81, 58]]
    assert 65505 in stop_string_eos_ids(tokenizer, ["Q:", "<|end|>"])


def test_generation_stops_when_the_tail_matches_a_stop_sequence(state_cls) -> None:
    state = _decoding_state(state_cls, stop_sequences=[[81, 58]])

    assert state.update_and_check_completion(5) is False
    assert state.update_and_check_completion(81) is False  # partial match, must not fire
    assert state.update_and_check_completion(58) is True
    assert state.status == RequestStatus.FINISHED


def test_a_stop_sequence_only_fires_as_a_suffix(state_cls) -> None:
    """The ids must be adjacent and at the end, not merely present."""

    state = _decoding_state(state_cls, stop_sequences=[[81, 58]])

    for token in (81, 7, 58, 7):
        assert state.update_and_check_completion(token) is False
    assert state.status == RequestStatus.DECODING


def test_without_stop_sequences_generation_runs_to_the_length_cap(state_cls) -> None:
    """The behavior this fixes: nothing ends the request but its cap."""

    state = state_cls(request_id="r", initial_tokens=[1], max_new_tokens=4)
    state.status = RequestStatus.PENDING
    state.status = RequestStatus.PREFILLING
    state.status = RequestStatus.DECODING

    finishes = [state.update_and_check_completion(token) for token in (81, 58, 81, 58)]

    assert finishes == [False, False, False, True]
    assert state.generated_len() == 4


def test_eos_still_finishes_when_stop_sequences_are_set(state_cls) -> None:
    state = _decoding_state(state_cls, eos_token_id=99, stop_sequences=[[81, 58]])

    assert state.update_and_check_completion(99) is True


def test_a_preempted_request_keeps_its_stop_sequences(state_cls) -> None:
    """Recompute preemption rebuilds the request from ``get_request_config``.

    A rebuilt request that lost its stop sequences would silently run to its
    length cap, so the loss shows up as wasted decode rather than as an error.
    """

    state = _decoding_state(state_cls, stop_sequences=[[81, 58]])
    state.update_and_check_completion(5)

    rebuilt = state.create_equivalent_initial_request()

    assert rebuilt.stop_sequences == [[81, 58]]
    rebuilt.status = RequestStatus.PENDING
    rebuilt.status = RequestStatus.PREFILLING
    rebuilt.status = RequestStatus.DECODING
    assert rebuilt.update_and_check_completion(81) is False
    assert rebuilt.update_and_check_completion(58) is True


class _FakeEngine:
    """Engine stub recording what the backend forwards to ``generate_batch``."""

    def __init__(self, accepts_stop_sequences: bool = True) -> None:
        self.calls: list[dict] = []
        if not accepts_stop_sequences:
            self.generate_batch = self._generate_without_stops

    def generate_batch(self, input_ids, *, max_new_tokens, eos_token_id, stop_sequences=None, **kwargs):
        self.calls.append({"stop_sequences": stop_sequences})
        return [[1] for _ in input_ids]

    def _generate_without_stops(self, input_ids, *, max_new_tokens, eos_token_id, **kwargs):
        self.calls.append({"stop_sequences": None})
        return [[1] for _ in input_ids]


def test_engine_accepts_probes_the_generate_batch_signature() -> None:
    assert engine_accepts(_FakeEngine(), "stop_sequences") is True
    assert engine_accepts(_FakeEngine(accepts_stop_sequences=False), "stop_sequences") is False
    assert engine_accepts(object(), "stop_sequences") is False


def test_backend_forwards_multi_token_stops_to_the_engine() -> None:
    engine = _FakeEngine()
    tokenizer = _FakeTokenizer({"Q:": [81, 58]})
    backend = EngineBackend(engine, tokenizer)

    backend.generate(["prompt"], max_new_tokens=8, stop_strings=["Q:"])

    assert engine.calls[0]["stop_sequences"] == [[81, 58]]


def test_backend_warns_when_the_engine_cannot_enforce_stops(caplog) -> None:
    """Failing silently here reproduces the exact bug stop sequences fix."""

    engine = _FakeEngine(accepts_stop_sequences=False)
    tokenizer = _FakeTokenizer({"Q:": [81, 58]})
    backend = EngineBackend(engine, tokenizer)

    with caplog.at_level("WARNING"):
        backend.generate(["prompt"], max_new_tokens=8, stop_strings=["Q:"])

    assert "will not" in caplog.text
    assert engine.calls[0]["stop_sequences"] is None


def test_duplicate_stop_strings_are_collapsed() -> None:
    tokenizer = _FakeTokenizer({"Q:": [81, 58]})

    assert stop_string_sequences(tokenizer, ["Q:", "Q:"]) == [[81, 58]]
