"""Backend tests that run on CPU via fakes (real engines are CUDA-gated)."""

from typing import ClassVar

from looped_cdb.eval.backends import EngineBackend, decode_generations


class FakeTokenizer:
    """Reversible id<->word tokenizer for decode tests."""

    eos_token_id = 7
    _id2tok: ClassVar[dict[int, str]] = {5: "hi", 100: "Q:", 7: "</s>", 42: "answer", 9: "9."}

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(self._id2tok.get(i, "?") for i in ids)

    def __call__(self, text, add_special_tokens=False):
        rev = {v: k for k, v in self._id2tok.items()}
        return {"input_ids": [rev.get(tok, 1) for tok in text.split()]}


class FakeOutput:
    def __init__(self, generated_tokens):
        self.generated_tokens = generated_tokens


class FakeEngine:
    """Records the generate_batch call and returns canned outputs."""

    def __init__(self, outputs):
        self._outputs = outputs
        self.calls = []

    def generate_batch(self, input_ids, *, max_new_tokens, eos_token_id, warmup=True, model_kwargs=None):
        self.calls.append(
            {
                "input_ids": input_ids,
                "max_new_tokens": max_new_tokens,
                "eos": eos_token_id,
                "warmup": warmup,
                "model_kwargs": model_kwargs,
            }
        )
        return self._outputs


def test_decode_generations_truncates_and_appends_eos():
    tok = FakeTokenizer()
    # tokens decode to "answer 9. </s>"; eos text "</s>" should be truncated away
    out = decode_generations(tok, [[42, 9, 7]], stop_strings=["Q:"])
    assert out == ["answer 9. "]


def test_decode_generations_truncates_at_task_stop():
    tok = FakeTokenizer()
    out = decode_generations(tok, [[42, 100, 9]], stop_strings=["Q:"])
    assert out == ["answer "]


def test_engine_backend_generate_end_to_end():
    tok = FakeTokenizer()
    engine = FakeEngine([FakeOutput([42, 9, 7])])
    backend = EngineBackend(engine, tok)
    result = backend.generate(["hi Q:"], max_new_tokens=256, stop_strings=["Q:"])
    assert result == ["answer 9. "]
    # prompt was tokenized and eos ids (tokenizer eos + single-token "Q:") were passed
    call = engine.calls[0]
    assert call["input_ids"] == [[5, 100]]
    assert call["max_new_tokens"] == 256
    assert 7 in call["eos"] and 100 in call["eos"]
    assert call["warmup"] is False
    assert call["model_kwargs"] is None  # no early-exit -> no model_kwargs forwarded


def test_engine_backend_forwards_model_kwargs():
    tok = FakeTokenizer()
    engine = FakeEngine([FakeOutput([42])])
    exit_kwargs = {"use_early_exit_gate": True, "exit_threshold": 0.5, "min_exit_step": 2, "return_exit_steps": True}
    backend = EngineBackend(engine, tok, model_kwargs=exit_kwargs)
    backend.generate(["hi"], max_new_tokens=8, stop_strings=[])
    assert engine.calls[0]["model_kwargs"] == exit_kwargs


class FakeStats:
    def __init__(self, exit_depth_histogram: dict[int, int]) -> None:
        self.exit_depth_histogram = exit_depth_histogram


def test_engine_backend_exit_depth_counts_from_last_stats():
    # The CDB engine reports a 0-indexed exit-step histogram via last_stats; the
    # backend exposes it as a 1-indexed exit-depth -> token-count mapping.
    tok = FakeTokenizer()
    engine = FakeEngine([FakeOutput([42])])
    engine.last_stats = FakeStats(exit_depth_histogram={1: 5, 2: 1})
    backend = EngineBackend(engine, tok)
    assert backend.exit_depth_counts == {2: 5, 3: 1}


def test_engine_backend_exit_depth_counts_empty_without_stats():
    tok = FakeTokenizer()
    engine = FakeEngine([FakeOutput([42])])
    backend = EngineBackend(engine, tok)
    assert backend.exit_depth_counts == {}
