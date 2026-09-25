"""Load looped models and tokenizers for evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer

from looped_cdb.kv_cache_policy import resolve_kv_slots_per_layer, validate_kv_cache_policy

MODEL_FAMILIES = ("ouro", "huginn")

HUGINN_0125_ID = "tomg-group-umd/huginn-0125"
# The snapshot the paper's Huginn numbers were measured on; pinned so Hub-side
# updates cannot change the checkpoint.
HUGINN_0125_REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"


def huginn_revision(model_id: str) -> str | None:
    """Pinned snapshot for the published Huginn id; local paths and other ids float."""

    return HUGINN_0125_REVISION if model_id == HUGINN_0125_ID else None


def resolve_model_family(model_id: str) -> str:
    """Infer which looped model family ``model_id`` refers to.

    Reads ``model_type`` from the checkpoint config, falling back to the Hub for
    ids that are not local paths.
    """

    config_path = Path(model_id) / "config.json"
    if config_path.exists():
        model_type = json.loads(config_path.read_text()).get("model_type", "")
    else:
        config = AutoConfig.from_pretrained(model_id, revision=huginn_revision(model_id), trust_remote_code=True)
        model_type = getattr(config, "model_type", "")

    if model_type.startswith("huginn"):
        return "huginn"
    if model_type.startswith("ouro"):
        return "ouro"
    raise ValueError(
        f"Cannot infer a looped model family from model_type={model_type!r} for {model_id!r}; "
        f"the checkpoint config must name one of {', '.join(MODEL_FAMILIES)}."
    )


def load_ouro_model(
    model_id: str,
    *,
    recur_steps: int | None = None,
    exit_gate_type: str | None = None,
    exit_gate_path: str | None = None,
    kv_policy: str | None = None,
    kv_slots_per_layer: int | None = None,
    dtype: Any = torch.bfloat16,
    attn_impl: str = "paged|flash_attention_3",
    device: str = "cuda",
) -> Any:
    """Load an Ouro causal LM, setting the recurrent depth and (optionally) a trained exit gate.

    ``recur_steps`` overrides ``total_ut_steps``. ``exit_gate_type``/``exit_gate_path``
    load a trained gate checkpoint into the model; with neither, the model's built-in
    gate is used. The exit *threshold* is a serving policy and is set by the backend, not
    here.

    ``kv_policy`` selects the recurrent KV cache policy. Left ``None``, the model keeps the
    depth-indexed layout in which every recursion holds its own KV slot.
    """

    from looped_cdb.models.ouro import OuroConfig, OuroForCausalLM

    config = OuroConfig.from_pretrained(model_id)
    if recur_steps is not None:
        config.total_ut_steps = int(recur_steps)

    model = OuroForCausalLM.from_pretrained(
        model_id,
        config=config,
        dtype=dtype,
        attn_implementation=attn_impl,
    )
    model = model.to(device).eval()
    # OuroForCausalLM.set_attn_implementation also normalizes config.layer_types for the
    # paged backend, so apply it to finalize the "paged|..." attention setup after load.
    model.set_attn_implementation(attn_impl)

    if kv_policy is not None:
        from looped_cdb.models.ouro.kv_cache_policy import configure_kv_cache_policy

        configure_kv_cache_policy(model, kv_policy=kv_policy, kv_slots_per_layer=kv_slots_per_layer)

    if exit_gate_type is not None or exit_gate_path is not None:
        from looped_cdb.continuous_depth_batching.adapters.ouro import configure_ouro_exit_gate

        configure_ouro_exit_gate(model, exit_gate_type=exit_gate_type, exit_gate_path=exit_gate_path)
    return model


def load_ouro_tokenizer(model_id: str) -> Any:
    """Load the Ouro tokenizer."""

    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def parse_layer_split(spec: str) -> tuple[int, int, int]:
    """Parse a ``prelude-core-coda`` layer split such as ``"1-4-1"`` into its three counts."""

    parts = spec.split("-")
    try:
        counts = tuple(int(part) for part in parts)
    except ValueError:
        counts = ()
    if len(counts) != 3:
        raise ValueError(f"layer split must be three dash-separated integers 'prelude-core-coda', got {spec!r}")
    prelude, core, coda = counts
    if prelude < 0 or coda < 0 or core < 1:
        raise ValueError(f"layer split needs prelude/coda >= 0 and core >= 1, got {spec!r}")
    return prelude, core, coda


def load_huginn_model(
    model_id: str,
    *,
    recur_steps: int | None = None,
    kv_policy: str | None = None,
    kv_slots_per_layer: int | None = None,
    state_init: str | None = None,
    layer_split: str | None = None,
    dtype: Any = torch.bfloat16,
    attn_impl: str = "paged|flash_attention_3",
    device: str = "cuda",
) -> Any:
    """Load a Huginn causal LM for serving.

    ``model_id`` is a Hugging Face model id (``tomg-group-umd/huginn-0125``, which
    loads the :data:`HUGINN_0125_REVISION` snapshot) or a local checkpoint
    directory; only the weights are read from it, since :class:`HuginnConfig`
    defaults already encode the release's architecture.
    ``recur_steps`` sets the number of recurrent steps, defaulting to the training
    mean recurrence. ``kv_policy``/``kv_slots_per_layer`` select the recurrent KV
    layout; leaving them unset keeps ``depth_indexed``, the layout the checkpoint
    is trained under, which needs one KV slot per recurrent step and therefore far
    more memory. ``state_init`` overrides the config's recurrent-state
    initialization when given, and otherwise leaves the configured default in
    place. The exit *threshold* is a serving policy and is set by the backend.

    ``layer_split`` overrides the checkpoint's prelude-core-coda layer counts (e.g.
    ``"1-4-1"``). Checkpoint layers outside the split are dropped at load, so the
    model's outputs stop being meaningful.
    """

    from looped_cdb.models.huginn import HuginnConfig, HuginnForCausalLM

    overrides: dict[str, Any] = {}
    if state_init is not None:
        overrides["state_init"] = state_init
    if recur_steps is not None:
        overrides["total_recurrent_steps"] = int(recur_steps)
    if layer_split is not None:
        prelude, core, coda = parse_layer_split(layer_split)
        overrides["n_layers"] = prelude + core + coda
        overrides["n_layers_in_prelude"] = prelude
        overrides["n_layers_in_recurrent_block"] = core
        overrides["n_layers_in_coda"] = coda
    config = HuginnConfig(**overrides)
    if kv_policy is not None:
        # Resolve rather than store what the caller passed, so the config records the
        # slot count the cache will actually allocate.
        config._cdb_kv_policy = validate_kv_cache_policy(kv_policy)
        config._cdb_kv_slots_per_layer = resolve_kv_slots_per_layer(
            kv_policy,
            total_recurrent_steps=int(config.total_recurrent_steps),
            requested_slots=kv_slots_per_layer,
        )

    model = HuginnForCausalLM.from_pretrained(
        model_id,
        config=config,
        revision=huginn_revision(model_id),
        dtype=dtype,
        attn_implementation=attn_impl,
    )
    return model.to(device).eval()


def load_huginn_tokenizer(model_id: str) -> Any:
    """Load the Huginn tokenizer."""

    return AutoTokenizer.from_pretrained(model_id, revision=huginn_revision(model_id))
