"""Runner orchestration tests (CPU, fake backend, injected docs)."""

import looped_cdb.eval.gsm8k  # noqa: F401
from looped_cdb.eval.runner import run_task
from looped_cdb.eval.tasks import get_task


class FakeBackend:
    def __init__(self, generations):
        self._generations = generations
        self.seen = {}

    def generate(self, prompts, *, max_new_tokens, stop_strings):
        self.seen = {"n": len(prompts), "max_new_tokens": max_new_tokens, "stop_strings": stop_strings}
        assert len(prompts) == len(self._generations)
        return self._generations


def test_run_task_aggregates_both_scorers():
    task = get_task("gsm8k_cot")
    docs = [
        {"question": "q1", "answer": "#### 9"},
        {"question": "q2", "answer": "#### 5"},
    ]
    generations = ["blah blah. The answer is 9.", "nope only 4 here"]
    result = run_task(task, FakeBackend(generations), num_fewshot=0, docs=docs)
    assert result.num_docs == 2
    assert result.metrics["strict-match"] == 0.5  # only doc1 has "The answer is 9."
    assert result.metrics["flexible-extract"] == 0.5  # doc1 -> 9 (ok), doc2 -> 4 (!=5)


def test_run_task_passes_generation_limits():
    task = get_task("gsm8k_cot")
    docs = [{"question": "q1", "answer": "#### 1"}]
    backend = FakeBackend(["The answer is 1."])
    run_task(task, backend, num_fewshot=0, docs=docs)
    assert backend.seen["max_new_tokens"] == 256
    assert backend.seen["stop_strings"] == ["Q:", "</s>", "<|im_end|>"]


def test_limit_truncates_docs():
    task = get_task("gsm8k_cot")
    docs = [{"question": f"q{i}", "answer": "#### 1"} for i in range(3)]
    backend = FakeBackend(["The answer is 1."])  # only one doc after limit
    result = run_task(task, backend, num_fewshot=0, limit=1, docs=docs)
    assert result.num_docs == 1


def test_log_samples_records_predictions():
    task = get_task("gsm8k_cot")
    docs = [{"question": "q1", "answer": "#### 9"}]
    result = run_task(task, FakeBackend(["The answer is 9."]), num_fewshot=0, log_samples=True, docs=docs)
    assert result.samples is not None
    sample = result.samples[0]
    assert sample["gold"] == "9"
    assert sample["generation"] == "The answer is 9."
    assert sample["scores"]["strict-match"] == 1
