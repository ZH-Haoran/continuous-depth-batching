"""Tests for the looped KV layout shared by Ouro and Huginn."""

from __future__ import annotations

import pytest

from looped_cdb.kv_cache_policy import LoopedKvLayout, kv_layer_index


def _huginn(policy: str, steps: int = 8) -> LoopedKvLayout:
    return LoopedKvLayout.build(
        num_prelude_layers=2,
        num_core_layers=4,
        num_coda_layers=2,
        total_recurrent_steps=steps,
        policy=policy,
    )


@pytest.mark.parametrize(
    ("policy", "expected_slots", "expected_cache_layers"),
    [
        ("depth_indexed", 8, 2 + 8 * 4 + 2),
        ("last_exited", 8, 2 + 8 * 4 + 2),
        ("single", 1, 2 + 1 * 4 + 2),
        ("first_then_shared", 2, 2 + 2 * 4 + 2),
    ],
)
def test_cache_layer_count(policy: str, expected_slots: int, expected_cache_layers: int) -> None:
    layout = _huginn(policy)

    assert layout.slots_per_layer == expected_slots
    assert layout.num_cache_layers == expected_cache_layers


def test_prelude_and_coda_hold_one_slot_each() -> None:
    layout = _huginn("first_then_shared")

    assert [layout.prelude_layer_index(i) for i in range(2)] == [0, 1]
    # Coda sits after every core slot, so it never collides with recurrent state.
    assert [layout.coda_layer_index(i) for i in range(2)] == [10, 11]


def test_stage_indices_are_disjoint_and_dense() -> None:
    layout = _huginn("first_then_shared")

    indices = {layout.prelude_layer_index(i) for i in range(2)}
    indices |= {layout.coda_layer_index(i) for i in range(2)}
    for step in range(layout.total_recurrent_steps):
        indices |= {layout.core_layer_index(i, step) for i in range(4)}

    assert indices == set(range(layout.num_cache_layers))


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        ("single", [2, 2, 2, 2, 2, 2, 2, 2]),
        ("first_then_shared", [2, 6, 6, 6, 6, 6, 6, 6]),
        ("depth_indexed", [2, 6, 10, 14, 18, 22, 26, 30]),
        ("last_exited", [2, 6, 10, 14, 18, 22, 26, 30]),
    ],
)
def test_core_slot_assignment_per_step(policy: str, expected: list[int]) -> None:
    layout = _huginn(policy)

    assert [layout.core_layer_index(0, step) for step in range(8)] == expected


@pytest.mark.parametrize("policy", ["single", "first_then_shared"])
def test_fully_looped_model_matches_ouro_indexing(policy: str) -> None:
    """A model with no prelude or coda must reproduce the existing Ouro layout."""

    num_layers, steps = 24, 4
    layout = LoopedKvLayout.build(
        num_prelude_layers=0,
        num_core_layers=num_layers,
        num_coda_layers=0,
        total_recurrent_steps=steps,
        policy=policy,
    )

    for step in range(steps):
        for layer in range(num_layers):
            assert layout.core_layer_index(layer, step) == kv_layer_index(
                physical_layer_idx=layer,
                num_hidden_layers=num_layers,
                recurrent_step=step,
                policy=policy,
                slots_per_layer=layout.slots_per_layer,
            )


def test_exit_copy_layer_groups_target_every_deeper_slot() -> None:
    layout = _huginn("last_exited", steps=4)

    # Exit at step 1: every core layer groups its step-1 slot with the step-2 and step-3 slots,
    # so one gather of the source serves all deeper targets.
    assert layout.exit_copy_layer_groups(1) == [
        (
            layout.core_layer_index(layer, 1),
            [layout.core_layer_index(layer, 1, explicit_slot=slot) for slot in (2, 3)],
        )
        for layer in range(4)
    ]
    # An exit at the final step has no deeper slot to fill.
    assert layout.exit_copy_layer_groups(3) == []


def test_exit_copy_layer_groups_are_empty_for_slot_sharing_layouts() -> None:
    # Deeper steps reuse the shared slot, so an exit there leaves nothing to copy.
    assert _huginn("single").exit_copy_layer_groups(0) == []
    assert _huginn("first_then_shared").exit_copy_layer_groups(1) == []


def test_rejects_out_of_range_stage_layers() -> None:
    layout = _huginn("single")

    with pytest.raises(ValueError, match="prelude layer index"):
        layout.prelude_layer_index(2)
    with pytest.raises(ValueError, match="coda layer index"):
        layout.coda_layer_index(2)
    with pytest.raises(ValueError, match="core layer index"):
        layout.core_layer_index(4, 0)


def test_rejects_out_of_range_recurrent_step() -> None:
    layout = _huginn("single", steps=4)

    with pytest.raises(ValueError, match="recurrent_step"):
        layout.core_layer_index(0, 4)


def test_rejects_more_slots_than_steps() -> None:
    with pytest.raises(ValueError, match="slots_per_layer"):
        LoopedKvLayout(
            num_prelude_layers=2,
            num_core_layers=4,
            num_coda_layers=2,
            total_recurrent_steps=2,
            policy="first_then_shared",
            slots_per_layer=4,
        )
