"""CPU tests for looped-model loading."""

import pytest

import looped_cdb.continuous_depth_batching.adapters.ouro as cdb_adapters
import looped_cdb.models.ouro as ouro_pkg


class _FakeConfig:
    def __init__(self):
        self.total_ut_steps = 4

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        return cls()


class _FakeModel:
    def __init__(self, config):
        self.config = config

    @classmethod
    def from_pretrained(cls, model_id, *, config, **kwargs):
        return cls(config)

    def to(self, device):
        return self

    def eval(self):
        return self

    def set_attn_implementation(self, impl):
        pass


def test_load_configures_exit_gate_when_requested(monkeypatch):
    captured: dict = {}

    def fake_gate(model, *, exit_gate_type, exit_gate_path):
        captured["type"] = exit_gate_type
        captured["path"] = exit_gate_path

    monkeypatch.setattr(ouro_pkg, "OuroConfig", _FakeConfig)
    monkeypatch.setattr(ouro_pkg, "OuroForCausalLM", _FakeModel)
    monkeypatch.setattr(cdb_adapters, "configure_ouro_exit_gate", fake_gate)

    from looped_cdb.eval.model_loading import load_ouro_model

    load_ouro_model("some/ouro", exit_gate_type="lookahead", exit_gate_path="/g.safetensors", device="cpu", dtype="x")
    assert captured == {"type": "lookahead", "path": "/g.safetensors"}


def test_load_skips_exit_gate_by_default(monkeypatch):
    calls = {"n": 0}

    def fake_gate(model, **kwargs):
        calls["n"] += 1

    monkeypatch.setattr(ouro_pkg, "OuroConfig", _FakeConfig)
    monkeypatch.setattr(ouro_pkg, "OuroForCausalLM", _FakeModel)
    monkeypatch.setattr(cdb_adapters, "configure_ouro_exit_gate", fake_gate)

    from looped_cdb.eval.model_loading import load_ouro_model

    load_ouro_model("some/ouro", recur_steps=2, device="cpu", dtype="x")
    assert calls["n"] == 0


def test_load_applies_recur_steps_in_memory(monkeypatch):
    captured: dict = {}

    class FakeConfig:
        def __init__(self):
            self.total_ut_steps = 4

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            captured["config_id"] = model_id
            return cls()

    class FakeModel:
        def __init__(self, config):
            self.config = config

        @classmethod
        def from_pretrained(cls, model_id, *, config, **kwargs):
            captured["model_id"] = model_id
            captured["config_obj"] = config
            captured["kwargs"] = kwargs
            return cls(config)

        def to(self, device):
            captured["device"] = device
            return self

        def eval(self):
            captured["eval"] = True
            return self

        def set_attn_implementation(self, impl):
            captured["set_attn"] = impl

    monkeypatch.setattr(ouro_pkg, "OuroConfig", FakeConfig)
    monkeypatch.setattr(ouro_pkg, "OuroForCausalLM", FakeModel)

    from looped_cdb.eval.model_loading import load_ouro_model

    model = load_ouro_model(
        "some/ouro",
        recur_steps=6,
        dtype="BF16-SENTINEL",
        attn_impl="paged|flash_attention_2",
        device="cpu",
    )

    # recur_steps override applied to the in-memory config, not a rewritten file
    assert model.config.total_ut_steps == 6
    assert captured["config_obj"].total_ut_steps == 6
    assert captured["kwargs"]["attn_implementation"] == "paged|flash_attention_2"
    assert captured["kwargs"]["dtype"] == "BF16-SENTINEL"
    assert captured["device"] == "cpu"
    assert captured["eval"] is True
    # local class is used directly -> no trust_remote_code needed
    assert "trust_remote_code" not in captured["kwargs"]


def test_load_without_recur_steps_leaves_config(monkeypatch):
    class FakeConfig:
        def __init__(self):
            self.total_ut_steps = 4

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            return cls()

    class FakeModel:
        def __init__(self, config):
            self.config = config

        @classmethod
        def from_pretrained(cls, model_id, *, config, **kwargs):
            return cls(config)

        def to(self, device):
            return self

        def eval(self):
            return self

        def set_attn_implementation(self, impl):
            pass

    monkeypatch.setattr(ouro_pkg, "OuroConfig", FakeConfig)
    monkeypatch.setattr(ouro_pkg, "OuroForCausalLM", FakeModel)

    from looped_cdb.eval.model_loading import load_ouro_model

    model = load_ouro_model("some/ouro", recur_steps=None, dtype="x", attn_impl="sdpa", device="cpu")
    assert model.config.total_ut_steps == 4


class _FakeHuginn:
    """Captures the config load_huginn_model builds, skipping the checkpoint download."""

    def __init__(self, config):
        self.config = config

    @classmethod
    def from_pretrained(cls, model_id, *, config, **kwargs):
        return cls(config)

    def to(self, device):
        return self

    def eval(self):
        return self


def test_parse_layer_split_accepts_counts_and_rejects_malformed_specs():
    from looped_cdb.eval.model_loading import parse_layer_split

    assert parse_layer_split("2-4-2") == (2, 4, 2)
    assert parse_layer_split("0-4-0") == (0, 4, 0)
    for bad in ("2-4", "a-4-2", "2-0-2", "-1-4-1", ""):
        with pytest.raises(ValueError):
            parse_layer_split(bad)


def test_huginn_layer_split_overrides_the_in_memory_config(monkeypatch):
    import looped_cdb.models.huginn as huginn_pkg

    monkeypatch.setattr(huginn_pkg, "HuginnForCausalLM", _FakeHuginn)

    from looped_cdb.eval.model_loading import load_huginn_model

    model = load_huginn_model("some/huginn", layer_split="1-4-1", device="cpu")
    config = model.config
    split = (config.n_layers_in_prelude, config.n_layers_in_recurrent_block, config.n_layers_in_coda)
    assert split == (1, 4, 1)
    assert config.n_layers == 6

    default = load_huginn_model("some/huginn", device="cpu").config
    assert (default.n_layers_in_prelude, default.n_layers_in_coda) == (2, 2)


def test_layer_split_raises_for_ouro(monkeypatch):
    from looped_cdb.benchmarks import runner

    monkeypatch.setattr("looped_cdb.eval.model_loading.resolve_model_family", lambda model_id: "ouro")
    with pytest.raises(ValueError, match="only supported for Huginn"):
        runner.load_serving_model(
            "some/ouro",
            max_recurrent_depth=4,
            attn_implementation="sdpa",
            dtype="x",
            device="cpu",
            layer_split="1-4-1",
        )
