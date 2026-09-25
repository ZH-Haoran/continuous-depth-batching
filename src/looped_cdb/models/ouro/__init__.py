"""Local Ouro model implementation.

The model files are vendored from ``KristianS7/Ouro-1.4B``.
Heavy imports are lazy-loaded so importing this package does not pull in torch until a model class is requested.
"""

__all__ = [
    "OuroConfig",
    "OuroForCausalLM",
    "OuroModel",
    "UniversalTransformerCache",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "OuroConfig": ("looped_cdb.models.ouro.configuration_ouro", "OuroConfig"),
    "OuroForCausalLM": ("looped_cdb.models.ouro.modeling_ouro", "OuroForCausalLM"),
    "OuroModel": ("looped_cdb.models.ouro.modeling_ouro", "OuroModel"),
    "UniversalTransformerCache": ("looped_cdb.models.ouro.modeling_ouro", "UniversalTransformerCache"),
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
