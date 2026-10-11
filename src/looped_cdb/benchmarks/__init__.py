"""Benchmark helpers for CB/CDB measurement scripts."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .datasets import SampledRequest, filter_by_context, load_dataset
    from .metrics import BenchmarkConfig, BenchmarkSummary, ExitDistributionSummary
    from .workload import Workload, depths_from_pdf, slice_output_pdf

__all__ = [
    "BenchmarkConfig",
    "BenchmarkSummary",
    "ExitDistributionSummary",
    "SampledRequest",
    "Workload",
    "depths_from_pdf",
    "filter_by_context",
    "load_dataset",
    "slice_output_pdf",
]


def __getattr__(name: str) -> object:
    """Keep the CPU feedback simulator independent of serving dependencies."""
    if name in {"BenchmarkConfig", "BenchmarkSummary", "ExitDistributionSummary"}:
        module = ".metrics"
    elif name in {"SampledRequest", "filter_by_context", "load_dataset"}:
        module = ".datasets"
    elif name in {"Workload", "depths_from_pdf", "slice_output_pdf"}:
        module = ".workload"
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value
