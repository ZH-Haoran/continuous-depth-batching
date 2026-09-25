"""Tests for the gsm8k_cot task."""

import looped_cdb.eval.gsm8k  # noqa: F401  (registers gsm8k_cot)
from looped_cdb.eval.prompt import build_prompt
from looped_cdb.eval.tasks import get_task


def test_gold_extraction_takes_number_after_hashes():
    task = get_task("gsm8k_cot")
    assert task.doc_to_target({"question": "q", "answer": "reasoning\n#### 42"}) == "42"


def test_eight_exemplars_available():
    task = get_task("gsm8k_cot")
    assert len(task.fewshot_examples) == 8


def test_three_shot_prompt_shape():
    task = get_task("gsm8k_cot")
    doc = {"question": "TESTQ", "answer": "#### 5"}
    prompt = build_prompt(task, doc, num_fewshot=3)
    # three exemplars + the test document
    assert prompt.count("Q: ") == 4
    assert prompt.startswith("Q: There are 15 trees in the grove.")
    assert prompt.endswith("Q: TESTQ\nA:")
    assert "The answer is 6." in prompt  # first exemplar's answer


def test_exemplar_render_uses_space_delimiter():
    task = get_task("gsm8k_cot")
    doc = {"question": "TESTQ", "answer": "#### 5"}
    prompt = build_prompt(task, doc, num_fewshot=1)
    assert "\nA: There are 15 trees originally." in prompt


def test_scorers_are_strict_then_flexible():
    task = get_task("gsm8k_cot")
    assert [s.name for s in task.scorers] == ["strict-match", "flexible-extract"]


def test_generation_limits():
    task = get_task("gsm8k_cot")
    assert task.stop_strings == ["Q:", "</s>", "<|im_end|>"]
    assert task.max_gen_toks == 256


def test_strict_scorer_matches_gold():
    task = get_task("gsm8k_cot")
    strict = next(s for s in task.scorers if s.name == "strict-match")
    generation = "He had 5 and got 4 more. 5 + 4 = 9. The answer is 9."
    assert strict.score(generation, "9") == 1


def test_flexible_scorer_matches_last_number():
    task = get_task("gsm8k_cot")
    flexible = next(s for s in task.scorers if s.name == "flexible-extract")
    generation = "Step 12 then the result is 33"
    assert flexible.score(generation, "33") == 1
