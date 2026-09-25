"""Unit tests for generation IO helpers (no GPU)."""

from looped_cdb.eval.backends import (
    eos_token_ids,
    generated_tokens,
    stop_string_eos_ids,
    tokenize,
    truncate_at_stop,
)


class FakeTokenizer:
    """Minimal tokenizer: whitespace ids, single-token words map to one id."""

    def __init__(self, eos_token_id=7):
        self.eos_token_id = eos_token_id
        self._vocab = {"Q:": 100, "</s>": 7, "<|im_end|>": 200, "hi": 5}

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [self._vocab.get(tok, 1) for tok in text.split()]}


def test_truncate_at_earliest_stop():
    text = "answer is 9.\nQ: next question"
    assert truncate_at_stop(text, ["Q:", "<|im_end|>"]) == "answer is 9.\n"


def test_truncate_no_stop_present_returns_unchanged():
    assert truncate_at_stop("just an answer", ["Q:"]) == "just an answer"


def test_truncate_empty_stops_returns_unchanged():
    assert truncate_at_stop("text", []) == "text"


def test_truncate_picks_first_of_multiple():
    text = "a </s> b Q: c"
    assert truncate_at_stop(text, ["Q:", "</s>"]) == "a "


def test_tokenize_returns_flat_list():
    assert tokenize(FakeTokenizer(), "hi Q:") == [5, 100]


def test_generated_tokens_from_attribute():
    class Out:
        def __init__(self):
            self.generated_tokens = [1, 2, 3]

    assert generated_tokens(Out()) == [1, 2, 3]


def test_eos_token_ids_scalar_list_and_none():
    assert eos_token_ids(FakeTokenizer(eos_token_id=7)) == [7]
    assert eos_token_ids(FakeTokenizer(eos_token_id=[7, 8])) == [7, 8]
    assert eos_token_ids(FakeTokenizer(eos_token_id=None)) == []


def test_stop_string_eos_ids_adds_single_token_stops_only():
    tok = FakeTokenizer(eos_token_id=7)
    # "Q:" -> single id 100 (added); multi-word stop -> multi token (ignored)
    ids = stop_string_eos_ids(tok, ["Q:", "some words"])
    assert 7 in ids and 100 in ids
    # deduplicated
    assert len(ids) == len(set(ids))
