"""Continuous depth batching experiments for looped language models."""

from typing import TYPE_CHECKING

__all__ = [
    "ContinuousBatchingConfig",
    "ContinuousBatchingEngine",
    "ContinuousDepthBatchingConfig",
    "ContinuousDepthBatchingEngine",
    "models",
]

if TYPE_CHECKING:
    from .continuous_batching import ContinuousBatchingConfig, ContinuousBatchingEngine
    from .continuous_depth_batching import ContinuousDepthBatchingConfig, ContinuousDepthBatchingEngine

_CB_EXPORTS = frozenset({"ContinuousBatchingConfig", "ContinuousBatchingEngine"})
_CDB_EXPORTS = frozenset({"ContinuousDepthBatchingConfig", "ContinuousDepthBatchingEngine"})


def __getattr__(name: str) -> object:
    """Load the engines lazily so importing the package does not pull in torch."""
    if name in _CB_EXPORTS:
        from . import continuous_batching

        return getattr(continuous_batching, name)
    if name in _CDB_EXPORTS:
        from . import continuous_depth_batching

        return getattr(continuous_depth_batching, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
