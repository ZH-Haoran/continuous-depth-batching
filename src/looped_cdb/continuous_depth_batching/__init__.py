from typing import TYPE_CHECKING

__all__ = [
    "CDBModelAdapter",
    "ContinuousDepthBatchingConfig",
    "ContinuousDepthBatchingEngine",
    "ContinuousDepthBatchingStats",
    "DecisionHorizon",
    "ExitDecisionRule",
    "ExitPolicy",
    "ExitPolicySpec",
]

if TYPE_CHECKING:
    from .config import ContinuousDepthBatchingConfig
    from .continuous_api import ContinuousDepthBatchingEngine, ContinuousDepthBatchingStats
    from .exit_policy import DecisionHorizon, ExitDecisionRule, ExitPolicy, ExitPolicySpec
    from .model_adapter import CDBModelAdapter


def __getattr__(name: str) -> object:
    if name == "ContinuousDepthBatchingConfig":
        from .config import ContinuousDepthBatchingConfig

        return ContinuousDepthBatchingConfig
    if name in {"ContinuousDepthBatchingEngine", "ContinuousDepthBatchingStats"}:
        from .continuous_api import ContinuousDepthBatchingEngine, ContinuousDepthBatchingStats

        return {
            "ContinuousDepthBatchingEngine": ContinuousDepthBatchingEngine,
            "ContinuousDepthBatchingStats": ContinuousDepthBatchingStats,
        }[name]
    if name in {"DecisionHorizon", "ExitDecisionRule", "ExitPolicy", "ExitPolicySpec"}:
        from .exit_policy import DecisionHorizon, ExitDecisionRule, ExitPolicy, ExitPolicySpec

        return {
            "DecisionHorizon": DecisionHorizon,
            "ExitDecisionRule": ExitDecisionRule,
            "ExitPolicy": ExitPolicy,
            "ExitPolicySpec": ExitPolicySpec,
        }[name]
    if name == "CDBModelAdapter":
        from .model_adapter import CDBModelAdapter

        return CDBModelAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
