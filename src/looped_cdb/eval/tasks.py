"""Task model and registry for the eval harness.

A ``Task`` fully describes a generation eval: where the data comes from, how to
render a prompt and gold answer, the few-shot exemplar pool, decoding limits,
and the scorers. Adding a new dataset means constructing one ``Task`` and
registering it. No changes to the runner, backends, or scoring are needed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from looped_cdb.eval.scoring import RegexFilter, exact_match

Doc = dict[str, Any]


@dataclass(frozen=True)
class Scorer:
    """One named metric: a regex filter feeding exact_match against the gold."""

    name: str
    filter: RegexFilter
    regexes_to_ignore: list[str]
    ignore_case: bool = True

    def score(self, generation: str, gold: str) -> int:
        prediction = self.filter.apply(generation)
        return exact_match(
            prediction,
            gold,
            regexes_to_ignore=self.regexes_to_ignore,
            ignore_case=self.ignore_case,
        )


@dataclass(frozen=True)
class Task:
    """A generation eval task: prompts, few-shot pool, decoding, and scoring."""

    name: str
    dataset_path: str
    dataset_name: str | None
    test_split: str
    doc_to_text: Callable[[Doc], str]
    doc_to_target: Callable[[Doc], str]
    fewshot_examples: list[Doc]
    target_delimiter: str
    fewshot_delimiter: str
    stop_strings: list[str]
    max_gen_toks: int
    scorers: list[Scorer]
    # Few-shot exemplar targets are often pre-written (e.g. CoT strings) rather
    # than raw dataset rows; when set this renders the exemplar answer instead of
    # ``doc_to_target``. Defaults to ``doc_to_target``.
    fewshot_target: Callable[[Doc], str] | None = None


TASK_REGISTRY: dict[str, Callable[[], Task]] = {}


def register_task(name: str, factory: Callable[[], Task]) -> None:
    """Register a task factory under ``name``."""

    TASK_REGISTRY[name] = factory


def get_task(name: str) -> Task:
    """Return a fresh ``Task`` for ``name`` or raise with the available names."""

    if name not in TASK_REGISTRY:
        raise KeyError(f"unknown task '{name}'; available: {sorted(TASK_REGISTRY)}")
    return TASK_REGISTRY[name]()
