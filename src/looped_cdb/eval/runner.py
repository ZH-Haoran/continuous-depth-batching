"""Orchestrate a task + backend into scored metrics.

``run_task`` is the only component that touches task, backend, and scoring
together: it loads documents, builds few-shot prompts, generates, scores every
generation with every scorer, and aggregates per-scorer means.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from looped_cdb.eval.backends import Backend
from looped_cdb.eval.prompt import build_prompt
from looped_cdb.eval.tasks import Doc, Task


@dataclass(frozen=True)
class EvalResult:
    """Aggregate metrics (and optional per-doc samples) for one eval run."""

    task: str
    num_docs: int
    num_fewshot: int
    metrics: dict[str, float]
    samples: list[dict[str, Any]] | None = None
    # Early-exit decode-token histogram: 1-indexed exit depth -> count (None if not gated).
    exit_depth_counts: dict[int, int] | None = None
    mean_exit_depth: float | None = None


def load_task_docs(task: Task) -> list[Doc]:
    """Load the task's test documents from the HuggingFace hub."""

    from datasets import load_dataset

    dataset = load_dataset(task.dataset_path, task.dataset_name, split=task.test_split)
    return [dict(row) for row in dataset]


def run_task(
    task: Task,
    backend: Backend,
    *,
    num_fewshot: int,
    limit: int | None = None,
    log_samples: bool = False,
    docs: list[Doc] | None = None,
) -> EvalResult:
    """Evaluate ``task`` with ``backend`` and return aggregated metrics."""

    if docs is None:
        docs = load_task_docs(task)
    if limit is not None:
        docs = docs[:limit]

    prompts = [build_prompt(task, doc, num_fewshot=num_fewshot) for doc in docs]
    golds = [task.doc_to_target(doc) for doc in docs]
    generations = backend.generate(prompts, max_new_tokens=task.max_gen_toks, stop_strings=task.stop_strings)

    totals = {scorer.name: 0 for scorer in task.scorers}
    samples: list[dict[str, Any]] = []
    for doc, prompt, generation, gold in zip(docs, prompts, generations, golds, strict=True):
        scores = {scorer.name: scorer.score(generation, gold) for scorer in task.scorers}
        for name, value in scores.items():
            totals[name] += value
        if log_samples:
            samples.append(
                {
                    "doc": doc,
                    "prompt": prompt,
                    "generation": generation,
                    "gold": gold,
                    "scores": scores,
                }
            )

    num_docs = len(docs)
    metrics = {name: (total / num_docs if num_docs else 0.0) for name, total in totals.items()}

    exit_depth_counts = getattr(backend, "exit_depth_counts", None) or None
    mean_exit_depth = None
    if exit_depth_counts:
        total_tokens = sum(exit_depth_counts.values())
        mean_exit_depth = sum(d * c for d, c in exit_depth_counts.items()) / total_tokens if total_tokens else None

    return EvalResult(
        task=task.name,
        num_docs=num_docs,
        num_fewshot=num_fewshot,
        metrics=metrics,
        samples=samples if log_samples else None,
        exit_depth_counts=exit_depth_counts,
        mean_exit_depth=mean_exit_depth,
    )
