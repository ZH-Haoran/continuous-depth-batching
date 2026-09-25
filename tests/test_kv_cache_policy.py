from typing import ClassVar

import pytest

from looped_cdb.kv_cache_policy import (
    DEPTH_INDEXED,
    KV_CACHE_POLICIES,
    LAST_EXITED,
    copies_exit_kv,
    kv_layer_index,
    kv_slot_for_step,
    resolve_kv_slots_per_layer,
    validate_kv_cache_policy,
)


def test_single_policy_maps_every_step_onto_one_slot() -> None:
    assert resolve_kv_slots_per_layer("single", total_recurrent_steps=4) == 1
    assert [kv_slot_for_step(step, policy="single", slots_per_layer=1) for step in range(4)] == [0, 0, 0, 0]


def test_depth_indexed_policy_gives_each_step_its_own_slot() -> None:
    assert resolve_kv_slots_per_layer(DEPTH_INDEXED, total_recurrent_steps=4) == 4
    assert [kv_slot_for_step(step, policy=DEPTH_INDEXED, slots_per_layer=4) for step in range(4)] == [0, 1, 2, 3]


def test_last_exited_keeps_the_depth_indexed_slot_arithmetic() -> None:
    assert resolve_kv_slots_per_layer(LAST_EXITED, total_recurrent_steps=4) == 4
    assert [kv_slot_for_step(step, policy=LAST_EXITED, slots_per_layer=4) for step in range(4)] == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="exactly 4 KV slot"):
        resolve_kv_slots_per_layer(LAST_EXITED, total_recurrent_steps=4, requested_slots=2)


def test_last_exited_is_the_only_policy_that_copies_exit_kv() -> None:
    assert copies_exit_kv(LAST_EXITED)
    assert [policy for policy in KV_CACHE_POLICIES if copies_exit_kv(policy)] == [LAST_EXITED]


def test_depth_indexed_layer_index_matches_unrolled_layer_numbering() -> None:
    # Step r of an H-layer block occupies cache layers [r * H, (r + 1) * H).
    assert (
        kv_layer_index(
            physical_layer_idx=1,
            num_hidden_layers=4,
            recurrent_step=3,
            policy=DEPTH_INDEXED,
            slots_per_layer=4,
        )
        == 13
    )


def test_retired_policy_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported KV cache policy"):
        validate_kv_cache_policy("shared")
    with pytest.raises(ValueError, match="Unsupported KV cache policy"):
        validate_kv_cache_policy("per_loop")


def test_first_then_shared_policy_keeps_prefix_then_reuses_last_slot() -> None:
    assert [kv_slot_for_step(step, policy="first_then_shared", slots_per_layer=3) for step in range(5)] == [
        0,
        1,
        2,
        2,
        2,
    ]


def test_kv_layer_index_offsets_physical_layer_by_slot() -> None:
    assert (
        kv_layer_index(
            physical_layer_idx=1,
            num_hidden_layers=4,
            recurrent_step=3,
            policy="first_then_shared",
            slots_per_layer=2,
        )
        == 5
    )


def test_single_policy_rejects_multi_slot_budget() -> None:
    with pytest.raises(ValueError, match="exactly 1 KV slot"):
        resolve_kv_slots_per_layer("single", total_recurrent_steps=4, requested_slots=2)


def test_configure_kv_cache_policy_records_the_layout_on_the_config() -> None:
    """Covers the config-mutating path on CPU; it previously only ran under CUDA."""

    from looped_cdb.models.ouro.kv_cache_policy import configure_kv_cache_policy

    class _Config:
        total_ut_steps = 4
        num_hidden_layers = 2
        layer_types: ClassVar[list[str]] = ["full_attention", "full_attention"]

    class _Model:
        def __init__(self) -> None:
            self.config = _Config()
            self.model = None

    model = _Model()
    configure_kv_cache_policy(model, kv_policy="first_then_shared")

    assert model.config._cdb_kv_policy == "first_then_shared"
    assert model.config._cdb_kv_slots_per_layer == 2
