"""Evaluation harness for looped LMs."""

from looped_cdb.eval import gsm8k
from looped_cdb.eval.runner import EvalResult, load_task_docs, run_task
from looped_cdb.eval.tasks import TASK_REGISTRY, Task, get_task, register_task

__all__ = [
    "TASK_REGISTRY",
    "EvalResult",
    "Task",
    "get_task",
    "load_task_docs",
    "register_task",
    "run_task",
]
