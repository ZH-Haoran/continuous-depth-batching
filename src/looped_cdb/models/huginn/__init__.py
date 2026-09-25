"""Huginn model implementation.

A reimplementation of ``tomg-group-umd/huginn-0125`` against the Hugging Face
attention interface, so it runs on the same paged FlashAttention backend as Ouro
and exposes its prelude/core/coda stages to the scheduler. Checkpoints load
directly: parameter names match the upstream release.

The upstream implementation is not vendored. Its numerics are pinned by
``tests/fixtures/huginn_golden.pt``, recorded from it while it was still
vendored and checked by ``tests/test_huginn_parity_cuda.py``.

Heavy imports are lazy-loaded so importing this package does not pull in torch
until a model class is requested.
"""

__all__ = [
    "ExitTracker",
    "HuginnConfig",
    "HuginnForCausalLM",
    "cache_layer_index",
    "cache_layout",
    "latent_diff",
    "should_exit",
]

_MODELING = "looped_cdb.models.huginn.modeling_huginn"
_GATES = "looped_cdb.models.huginn.exit_gates"

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "HuginnConfig": ("looped_cdb.models.huginn.configuration_huginn", "HuginnConfig"),
    "HuginnForCausalLM": (_MODELING, "HuginnForCausalLM"),
    "cache_layer_index": (_MODELING, "cache_layer_index"),
    "cache_layout": (_MODELING, "cache_layout"),
    "ExitTracker": (_GATES, "ExitTracker"),
    "latent_diff": (_GATES, "latent_diff"),
    "should_exit": (_GATES, "should_exit"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module_path, attr = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
