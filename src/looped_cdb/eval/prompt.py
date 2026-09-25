"""Few-shot prompt assembly, shared across all tasks."""

from __future__ import annotations

from looped_cdb.eval.tasks import Doc, Task


def build_prompt(task: Task, doc: Doc, *, num_fewshot: int) -> str:
    """Render ``num_fewshot`` exemplars followed by the test document.

    Each exemplar is ``doc_to_text + target_delimiter + <exemplar target>`` and
    the pieces are joined by ``fewshot_delimiter``. The test document contributes
    only its ``doc_to_text`` (the model completes the answer).
    """

    render_target = task.fewshot_target or task.doc_to_target
    pieces = [
        task.doc_to_text(example) + task.target_delimiter + render_target(example)
        for example in task.fewshot_examples[:num_fewshot]
    ]
    pieces.append(task.doc_to_text(doc))
    return task.fewshot_delimiter.join(pieces)
