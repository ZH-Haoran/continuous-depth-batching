from types import SimpleNamespace

import pytest
from torch import nn

from looped_cdb.continuous_depth_batching.adapters import OuroCDBAdapter
from looped_cdb.continuous_depth_batching.model_adapter import CDBModelAdapter, resolve_cdb_model_adapter


def test_cdb_model_adapter_is_abstract() -> None:
    with pytest.raises(TypeError):
        CDBModelAdapter()


def test_resolve_cdb_model_adapter_rejects_non_base_adapter() -> None:
    with pytest.raises(TypeError, match="must inherit CDBModelAdapter"):
        resolve_cdb_model_adapter(nn.Linear(1, 1), adapter=object())  # type: ignore[arg-type]


def test_ouro_adapter_detection_rejects_lookalike_non_ouro_model() -> None:
    class LookalikeLoopModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            decoder_config = SimpleNamespace(model_type="huginn", num_hidden_layers=1)
            self.config = SimpleNamespace(model_type="huginn")
            self.model = SimpleNamespace(
                config=decoder_config,
                embed_tokens=object(),
                layers=[],
                norm=object(),
                rotary_emb=object(),
            )
            self.lm_head = nn.Linear(1, 1)

    assert not OuroCDBAdapter.supports(LookalikeLoopModel())
