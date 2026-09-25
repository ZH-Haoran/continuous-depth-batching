"""Select the recurrent KV cache layout for an Ouro model.

The paged engines read the layout from private ``_cdb_*`` attributes on the model
config rather than from a constructor argument, because the layout is a serving
choice and the same checkpoint is served under several of them. Leaving those
attributes unset selects ``depth_indexed``, the layout the checkpoint is trained
under.
"""

from __future__ import annotations

from typing import Any

from looped_cdb.kv_cache_policy import resolve_kv_slots_per_layer


def configure_kv_cache_policy(
    model: Any,
    *,
    kv_policy: str = "single",
    kv_slots_per_layer: int | None = None,
) -> None:
    """Record the recurrent KV layout an Ouro model's attention should write under.

    ``kv_policy`` names a layout from :data:`~looped_cdb.kv_cache_policy.KV_CACHE_POLICIES`;
    ``kv_slots_per_layer`` overrides the slot count the policy would derive, and
    only ``first_then_shared`` leaves that free.
    """

    for target in (model, getattr(model, "model", None)):
        config = getattr(target, "config", None)
        if config is None:
            continue
        resolved_slots = resolve_kv_slots_per_layer(
            kv_policy,
            total_recurrent_steps=getattr(config, "total_ut_steps", 1) or 1,
            requested_slots=kv_slots_per_layer,
        )
        config._cdb_kv_policy = kv_policy
        config._cdb_kv_slots_per_layer = resolved_slots
        # Slot reuse reduces the physical layer count the paged cache sizes against, so
        # the layer-type list is re-derived from the unshared baseline on every call.
        layer_types = list(getattr(config, "_ouro_base_layer_types", getattr(config, "layer_types", [])))
        num_hidden_layers = getattr(config, "num_hidden_layers", len(layer_types))
        config.layer_types = layer_types[:num_hidden_layers]
        config._ouro_base_layer_types = list(config.layer_types)
