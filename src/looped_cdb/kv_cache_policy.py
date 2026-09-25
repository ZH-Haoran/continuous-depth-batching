"""Names and slot arithmetic for recurrent KV cache layouts.

A looped model runs the same physical layers once per recursion, so a layout says
which KV slot each recursion writes. ``depth_indexed`` gives every recursion its own
slot and is the layout a checkpoint is trained under; ``last_exited`` keeps that
per-recursion layout but propagates an early-exiting token's exit-step KV into its
deeper slots, so later tokens attend to the state the token actually exited with;
the others reuse a fixed number of slots, so the cache stops growing with
recurrent depth.

Every layout is a member of :data:`KV_CACHE_POLICIES`, including ``depth_indexed``,
and the arithmetic below covers all of them uniformly: a caller asks for slots and a
slot index without branching on which layout it holds.

The functions here address a fully looped model, whose every physical layer is a
core layer. :class:`LoopedKvLayout` places that arithmetic inside the wider
prelude/core/coda structure, where only the core repeats.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

type KvCachePolicy = Literal["depth_indexed", "last_exited", "single", "first_then_shared"]

#: Each recursion keeps a private KV slot. The layout a checkpoint is trained under.
DEPTH_INDEXED = "depth_indexed"

#: Depth-indexed slots, plus copy-on-exit routing: when a token exits early, its
#: exit-step KV is copied into its deeper slots, so later tokens at any recurrent
#: step attend to the state the token exited with. The copy is what makes gated
#: early exit KV-correct under a per-recursion layout; it frees no memory.
LAST_EXITED = "last_exited"

KV_CACHE_POLICIES: tuple[str, ...] = (DEPTH_INDEXED, LAST_EXITED, "single", "first_then_shared")


def validate_kv_cache_policy(policy: str) -> KvCachePolicy:
    """Return ``policy`` if it names a layout, else raise."""

    if policy not in KV_CACHE_POLICIES:
        raise ValueError(f"Unsupported KV cache policy {policy!r}; expected one of {KV_CACHE_POLICIES}")
    return policy  # type: ignore[return-value]


def copies_exit_kv(policy: str) -> bool:
    """Return whether ``policy`` copies a token's exit-step KV into its deeper slots on early exit."""

    return validate_kv_cache_policy(policy) == LAST_EXITED


def resolve_kv_slots_per_layer(
    policy: str,
    *,
    total_recurrent_steps: int,
    requested_slots: int | None = None,
) -> int:
    """Return the number of recurrent KV slots per physical layer.

    Only ``first_then_shared`` leaves the count free; the others derive it, and
    ``requested_slots`` must agree with what they derive.
    """

    if total_recurrent_steps <= 0:
        raise ValueError(f"total_recurrent_steps must be positive, got {total_recurrent_steps}")

    policy = validate_kv_cache_policy(policy)
    fixed = {DEPTH_INDEXED: total_recurrent_steps, LAST_EXITED: total_recurrent_steps, "single": 1}
    if policy in fixed:
        slots = fixed[policy]
        if requested_slots is not None and requested_slots != slots:
            raise ValueError(f"Policy {policy!r} uses exactly {slots} KV slot(s) per layer, got {requested_slots}")
        return slots

    slots = 2 if requested_slots is None else requested_slots
    if slots <= 0:
        raise ValueError(f"kv_slots_per_layer must be positive, got {slots}")
    if slots > total_recurrent_steps:
        raise ValueError(f"kv_slots_per_layer={slots} exceeds total_recurrent_steps={total_recurrent_steps}")
    return slots


def kv_slot_for_step(
    recurrent_step: int,
    *,
    policy: str,
    slots_per_layer: int,
) -> int:
    """Map a 0-based recurrent step to the recurrent KV slot it writes."""

    if recurrent_step < 0:
        raise ValueError(f"recurrent_step must be non-negative, got {recurrent_step}")
    if slots_per_layer <= 0:
        raise ValueError(f"slots_per_layer must be positive, got {slots_per_layer}")

    policy = validate_kv_cache_policy(policy)
    if policy in (DEPTH_INDEXED, LAST_EXITED):
        return recurrent_step
    if policy == "single":
        return 0
    # first_then_shared: a private slot for the first recursion, then one shared slot.
    return min(recurrent_step, slots_per_layer - 1)


def kv_layer_index(
    *,
    physical_layer_idx: int,
    num_hidden_layers: int,
    recurrent_step: int,
    policy: str,
    slots_per_layer: int,
    explicit_slot: int | None = None,
) -> int:
    """Return the cache-layer index for one physical layer and recurrent step."""

    if physical_layer_idx < 0:
        raise ValueError(f"physical_layer_idx must be non-negative, got {physical_layer_idx}")
    if num_hidden_layers <= 0:
        raise ValueError(f"num_hidden_layers must be positive, got {num_hidden_layers}")
    slot = (
        kv_slot_for_step(recurrent_step, policy=policy, slots_per_layer=slots_per_layer)
        if explicit_slot is None
        else explicit_slot
    )
    if not 0 <= slot < slots_per_layer:
        raise ValueError(f"explicit_slot must be in [0, {slots_per_layer}), got {slot}")
    return slot * num_hidden_layers + physical_layer_idx


@dataclass(frozen=True)
class LoopedKvLayout:
    """Maps a looped decoder's stages onto paged-cache layer indices.

    A looped decoder runs ``num_prelude_layers`` non-recurrent layers, then
    ``num_core_layers`` shared layers repeated ``total_recurrent_steps`` times,
    then ``num_coda_layers`` non-recurrent layers. Prelude and coda layers
    execute exactly once per token, so each holds a single KV slot; only core
    layers replicate their slots according to the KV cache policy.

    Cache layers are laid out as prelude, then core slot-major, then coda::

        [ prelude 0..P-1 | slot 0 core 0..C-1 | ... | slot S-1 core 0..C-1 | coda 0..D-1 ]

    A fully looped model such as Ouro is the degenerate case with no prelude and
    no coda, where this reduces to :func:`kv_layer_index`.
    """

    num_prelude_layers: int
    num_core_layers: int
    num_coda_layers: int
    total_recurrent_steps: int
    policy: str
    slots_per_layer: int

    def __post_init__(self) -> None:
        validate_kv_cache_policy(self.policy)
        if self.num_prelude_layers < 0 or self.num_coda_layers < 0:
            raise ValueError("prelude and coda layer counts must be non-negative")
        if self.num_core_layers <= 0:
            raise ValueError(f"num_core_layers must be positive, got {self.num_core_layers}")
        if self.total_recurrent_steps <= 0:
            raise ValueError(f"total_recurrent_steps must be positive, got {self.total_recurrent_steps}")
        if not 0 < self.slots_per_layer <= self.total_recurrent_steps:
            raise ValueError(
                f"slots_per_layer must be in [1, {self.total_recurrent_steps}], got {self.slots_per_layer}"
            )

    @classmethod
    def build(
        cls,
        *,
        num_prelude_layers: int,
        num_core_layers: int,
        num_coda_layers: int,
        total_recurrent_steps: int,
        policy: str = DEPTH_INDEXED,
        requested_slots: int | None = None,
    ) -> LoopedKvLayout:
        """Build a layout, deriving the core slot count from the policy."""

        return cls(
            num_prelude_layers=num_prelude_layers,
            num_core_layers=num_core_layers,
            num_coda_layers=num_coda_layers,
            total_recurrent_steps=total_recurrent_steps,
            policy=policy,
            slots_per_layer=resolve_kv_slots_per_layer(
                policy,
                total_recurrent_steps=total_recurrent_steps,
                requested_slots=requested_slots,
            ),
        )

    @property
    def num_cache_layers(self) -> int:
        """Total paged-cache layers this layout addresses."""

        return self.num_prelude_layers + self.slots_per_layer * self.num_core_layers + self.num_coda_layers

    def slot_for_step(self, recurrent_step: int) -> int:
        """Map a 0-based recurrent step to a core KV slot."""

        if not 0 <= recurrent_step < self.total_recurrent_steps:
            raise ValueError(f"recurrent_step must be in [0, {self.total_recurrent_steps}), got {recurrent_step}")
        return kv_slot_for_step(recurrent_step, policy=self.policy, slots_per_layer=self.slots_per_layer)

    def prelude_layer_index(self, physical_layer_idx: int) -> int:
        """Cache layer for a prelude layer, which is written once per token."""

        self._check_layer(physical_layer_idx, self.num_prelude_layers, "prelude")
        return physical_layer_idx

    def core_layer_index(
        self,
        physical_layer_idx: int,
        recurrent_step: int,
        *,
        explicit_slot: int | None = None,
    ) -> int:
        """Cache layer for one core layer at one recurrent step."""

        self._check_layer(physical_layer_idx, self.num_core_layers, "core")
        slot = self.slot_for_step(recurrent_step) if explicit_slot is None else explicit_slot
        if not 0 <= slot < self.slots_per_layer:
            raise ValueError(f"slot must be in [0, {self.slots_per_layer}), got {slot}")
        return self.num_prelude_layers + slot * self.num_core_layers + physical_layer_idx

    def coda_layer_index(self, physical_layer_idx: int) -> int:
        """Cache layer for a coda layer, which is written once per token."""

        self._check_layer(physical_layer_idx, self.num_coda_layers, "coda")
        return self.num_prelude_layers + self.slots_per_layer * self.num_core_layers + physical_layer_idx

    def exit_copy_layer_groups(self, exit_step: int) -> list[tuple[int, list[int]]]:
        """(source, targets) cache-layer groups that propagate an exit at ``exit_step`` to deeper slots.

        One group per core layer: the cache layer the exit step wrote, paired with the same
        physical layer in every deeper slot, so copying the source rows into the target rows
        makes any deeper recurrent step read the exit-step KV. Grouping by source lets a copy
        gather each source once and scatter it to all its targets. Empty when no deeper slot
        exists (an exit at the final step). Copying is how ``last_exited`` serves early exits;
        slot-sharing layouts leave nothing to copy.
        """

        source_slot = self.slot_for_step(exit_step)
        if source_slot + 1 == self.slots_per_layer:
            return []
        return [
            (
                self.core_layer_index(physical_layer_idx, exit_step),
                [
                    self.core_layer_index(physical_layer_idx, exit_step, explicit_slot=target_slot)
                    for target_slot in range(source_slot + 1, self.slots_per_layer)
                ],
            )
            for physical_layer_idx in range(self.num_core_layers)
        ]

    @staticmethod
    def _check_layer(physical_layer_idx: int, count: int, stage: str) -> None:
        if not 0 <= physical_layer_idx < count:
            raise ValueError(f"{stage} layer index must be in [0, {count}), got {physical_layer_idx}")
