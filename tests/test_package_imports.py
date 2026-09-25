from pathlib import Path

import pytest


def test_looped_cdb_ouro_lazy_import_surface() -> None:
    import looped_cdb
    import looped_cdb.models.ouro as ouro

    assert "models" in looped_cdb.__all__
    assert "OuroConfig" in ouro.__all__
    assert "OuroForCausalLM" in ouro.__all__
    assert "SubbatchedCache" not in ouro.__all__


def test_engines_are_reachable_from_the_package_root() -> None:
    import looped_cdb
    from looped_cdb.continuous_batching import ContinuousBatchingEngine
    from looped_cdb.continuous_depth_batching import ContinuousDepthBatchingEngine

    assert looped_cdb.ContinuousBatchingEngine is ContinuousBatchingEngine
    assert looped_cdb.ContinuousDepthBatchingEngine is ContinuousDepthBatchingEngine
    assert set(looped_cdb.__all__) == {
        "ContinuousBatchingConfig",
        "ContinuousBatchingEngine",
        "ContinuousDepthBatchingConfig",
        "ContinuousDepthBatchingEngine",
        "models",
    }


def test_unknown_package_root_attribute_raises() -> None:
    import looped_cdb

    with pytest.raises(AttributeError):
        _ = looped_cdb.NotAnEngine


def test_continuous_depth_batching_import_surface() -> None:
    import looped_cdb.continuous_depth_batching as cdb

    assert "ContinuousDepthBatchingEngine" in cdb.__all__
    assert "ContinuousDepthBatchingConfig" in cdb.__all__
    assert "ContinuousBatchingEngine" not in cdb.__all__


def test_ouro_model_package_contains_only_clean_hf_model_files() -> None:
    ouro_dir = Path(__file__).parents[1] / "src" / "looped_cdb" / "models" / "ouro"

    assert not (ouro_dir / "early_exit.py").exists()
    assert not (ouro_dir / "cdb_adapter.py").exists()


def test_ouro_attention_uses_transformers_attention_interface() -> None:
    from looped_cdb.models.ouro import OuroForCausalLM

    assert OuroForCausalLM._can_set_attn_implementation()
