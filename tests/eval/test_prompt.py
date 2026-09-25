"""Unit tests for few-shot prompt assembly."""

from looped_cdb.eval.prompt import build_prompt
from looped_cdb.eval.tasks import Task


def _toy_task() -> Task:
    return Task(
        name="toy",
        dataset_path="x",
        dataset_name=None,
        test_split="test",
        doc_to_text=lambda d: f"Q: {d['question']}\nA:",
        doc_to_target=lambda d: d["answer"].split("####")[-1].strip(),
        fewshot_examples=[
            {"question": "q1", "target": "r1. The answer is 1"},
            {"question": "q2", "target": "r2. The answer is 2"},
        ],
        target_delimiter=" ",
        fewshot_delimiter="\n\n",
        stop_strings=["Q:"],
        max_gen_toks=256,
        scorers=[],
        fewshot_target=lambda d: d["target"],
    )


def test_two_shot_prompt_exact():
    doc = {"question": "q3", "answer": "#### 3"}
    expected = "Q: q1\nA: r1. The answer is 1\n\nQ: q2\nA: r2. The answer is 2\n\nQ: q3\nA:"
    assert build_prompt(_toy_task(), doc, num_fewshot=2) == expected


def test_zero_shot_prompt():
    doc = {"question": "q3", "answer": "#### 3"}
    assert build_prompt(_toy_task(), doc, num_fewshot=0) == "Q: q3\nA:"


def test_num_fewshot_truncates_pool():
    doc = {"question": "q3", "answer": "#### 3"}
    prompt = build_prompt(_toy_task(), doc, num_fewshot=1)
    assert prompt == "Q: q1\nA: r1. The answer is 1\n\nQ: q3\nA:"
