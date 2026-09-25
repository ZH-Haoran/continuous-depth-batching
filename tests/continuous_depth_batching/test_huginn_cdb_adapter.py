"""Tests for the Huginn continuous-depth-batching adapter."""

from __future__ import annotations

import pytest
import torch

from looped_cdb.continuous_batching.cache import PagedAttentionCache
from looped_cdb.continuous_depth_batching.adapters import HuginnCDBAdapter, OuroCDBAdapter
from looped_cdb.continuous_depth_batching.exit_policy import ExitDecisionRule
from looped_cdb.continuous_depth_batching.model_adapter import resolve_cdb_model_adapter
from looped_cdb.kv_cache_policy import KV_CACHE_POLICIES, kv_slot_for_step, resolve_kv_slots_per_layer
from looped_cdb.models.huginn import HuginnConfig, HuginnForCausalLM

HIDDEN = 32
HEADS = 4


@pytest.fixture(scope="module")
def model() -> HuginnForCausalLM:
    config = HuginnConfig(
        n_embd=HIDDEN,
        n_heads=HEADS,
        n_layers=5,
        n_layers_in_prelude=1,
        n_layers_in_recurrent_block=2,
        n_layers_in_coda=2,
        mean_recurrence=4,
        block_size=64,
        vocab_size=128,
        intermediate_size=16,
        state_init="zero",
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    return HuginnForCausalLM(config).eval()


@pytest.fixture
def adapter(model: HuginnForCausalLM) -> HuginnCDBAdapter:
    adapter = HuginnCDBAdapter(model)
    adapter.configure_exit_signal(True)
    return adapter


def test_resolves_by_model_type(model: HuginnForCausalLM) -> None:
    resolved = resolve_cdb_model_adapter(model)

    assert isinstance(resolved, HuginnCDBAdapter)
    assert resolved.exit_policy_spec().rule is ExitDecisionRule.DIRECT_THRESHOLD
    assert not OuroCDBAdapter.supports(model)


def test_packing_round_trips(adapter: HuginnCDBAdapter) -> None:
    state = torch.randn(1, 3, HIDDEN)
    injection = torch.randn(1, 3, HIDDEN)

    packed = adapter.pack(state, injection)
    got_state, got_injection = adapter.unpack(packed)

    assert packed.shape == (1, 3, 2 * HIDDEN)
    assert torch.equal(got_state, state)
    assert torch.equal(got_injection, injection)


def test_unpack_rejects_unexpected_width(adapter: HuginnCDBAdapter) -> None:
    with pytest.raises(ValueError, match="expected width"):
        adapter.unpack(torch.randn(1, 2, HIDDEN))


def test_recurrent_step_updates_state_and_preserves_injection(adapter: HuginnCDBAdapter) -> None:
    """The injection is re-applied every step, so it must survive unchanged."""

    batch = 3
    state = torch.zeros(1, batch, HIDDEN)
    injection = torch.randn(1, batch, HIDDEN)
    position_ids = torch.arange(batch).view(1, batch)

    packed, _ = adapter.recurrent_step(
        hidden_states=adapter.pack(state, injection),
        position_ids=position_ids,
        recurrent_steps=torch.zeros(batch, dtype=torch.long),
        kv_slot=0,
    )
    next_state, next_injection = adapter.unpack(packed)

    assert torch.equal(next_injection, injection)
    assert not torch.allclose(next_state, state)


def test_exit_signals_have_the_shape_the_engine_reads(adapter: HuginnCDBAdapter) -> None:
    """The engine slices ``exit_signals[0, :batch, 0]``."""

    batch = 4
    injection = torch.randn(1, batch, HIDDEN)
    packed = adapter.pack(torch.zeros(1, batch, HIDDEN), injection)

    _, exit_signals = adapter.recurrent_step(
        hidden_states=packed,
        position_ids=torch.arange(batch).view(1, batch),
        recurrent_steps=torch.zeros(batch, dtype=torch.long),
        kv_slot=0,
    )

    assert exit_signals.shape == (1, batch, 1)


def test_exit_signal_is_the_direct_latent_difference(adapter: HuginnCDBAdapter) -> None:
    """Huginn exposes the convergence score without hazard encoding."""

    state = torch.randn(1, 2, HIDDEN)
    unchanged = state.clone()
    moved = state * 4.0

    converged = adapter.exit_signal(unchanged, state)
    still_moving = adapter.exit_signal(moved, state)

    assert torch.all(converged == 0)
    assert torch.all(still_moving > converged)


def test_exit_gate_can_be_disabled(adapter: HuginnCDBAdapter) -> None:
    packed = adapter.pack(torch.zeros(1, 2, HIDDEN), torch.randn(1, 2, HIDDEN))

    _, exit_signals = adapter.recurrent_step(
        hidden_states=packed,
        position_ids=torch.arange(2).view(1, 2),
        recurrent_steps=torch.zeros(2, dtype=torch.long),
        use_early_exit_gate=False,
        kv_slot=0,
    )

    assert exit_signals is None


def test_recurrent_step_validates_scheduled_depths(adapter: HuginnCDBAdapter) -> None:
    packed = adapter.pack(torch.zeros(1, 1, HIDDEN), torch.randn(1, 1, HIDDEN))
    kwargs = {"hidden_states": packed, "position_ids": torch.zeros(1, 1, dtype=torch.long), "kv_slot": 0}

    with pytest.raises(ValueError, match="recurrent_steps must be in"):
        adapter.recurrent_step(recurrent_steps=torch.tensor([99]), **kwargs)


def test_configure_recurrent_steps_and_kv_policy(adapter: HuginnCDBAdapter) -> None:
    adapter.configure_recurrent_steps(7)
    adapter.configure_kv_policy("first_then_shared", 2)

    assert adapter.config.total_recurrent_steps == 7
    assert adapter.config._cdb_kv_policy == "first_then_shared"
    assert adapter.config._cdb_kv_slots_per_layer == 2


def test_declares_that_its_stages_use_attention(adapter: HuginnCDBAdapter, model: HuginnForCausalLM) -> None:
    """The engine only supplies paged-cache arguments to stages that declare they need them."""

    assert adapter.stages_use_attention is True
    assert OuroCDBAdapter.stages_use_attention is False
    assert model.config.n_layers_in_prelude > 0
    assert model.config.n_layers_in_coda > 0


def test_prelude_returns_the_packed_state_pair(adapter: HuginnCDBAdapter) -> None:
    batch = 3
    input_ids = torch.randint(0, 128, (1, batch))
    position_ids = torch.arange(batch).view(1, batch)

    packed = adapter.prelude(input_ids=input_ids, position_ids=position_ids)
    state, injection = adapter.unpack(packed)

    assert packed.shape == (1, batch, 2 * HIDDEN)
    # state_init="zero" on this fixture, so only the injection carries prelude output.
    assert torch.count_nonzero(state) == 0
    assert torch.count_nonzero(injection) > 0


def test_lm_head_runs_the_coda_and_projects_the_state_half(adapter: HuginnCDBAdapter, model) -> None:
    batch = 2
    state = torch.randn(1, batch, HIDDEN)
    packed = adapter.pack(state, torch.randn(1, batch, HIDDEN))

    logits = adapter.lm_head(packed, position_ids=torch.arange(batch).view(1, batch))

    assert logits.shape == (1, batch, model.config.padded_vocab_size)


def test_boundary_free_split_runs_prelude_and_head() -> None:
    """A 0-core-0 split (no prelude or coda layers, the layer-split ablation's extreme) still serves."""

    config = HuginnConfig(
        n_embd=HIDDEN,
        n_heads=HEADS,
        n_layers=2,
        n_layers_in_prelude=0,
        n_layers_in_recurrent_block=2,
        n_layers_in_coda=0,
        mean_recurrence=4,
        block_size=64,
        vocab_size=128,
        intermediate_size=16,
        state_init="zero",
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    adapter = HuginnCDBAdapter(HuginnForCausalLM(config).eval())

    batch = 2
    position_ids = torch.arange(batch).view(1, batch)
    packed = adapter.prelude(input_ids=torch.randint(0, 128, (1, batch)), position_ids=position_ids)
    _, injection = adapter.unpack(packed)
    logits = adapter.lm_head(packed, position_ids=position_ids)

    # With no prelude layers the injection is the scaled embedding itself.
    assert torch.count_nonzero(injection) > 0
    assert logits.shape == (1, batch, config.padded_vocab_size)


def test_prelude_requires_position_ids(adapter: HuginnCDBAdapter) -> None:
    with pytest.raises(ValueError, match="position_ids are required"):
        adapter.prelude(input_ids=torch.zeros(1, 2, dtype=torch.long))


def test_exit_signal_is_disabled_by_default(model: HuginnForCausalLM) -> None:
    adapter = HuginnCDBAdapter(model)
    packed = adapter.pack(torch.ones(1, 1, HIDDEN), torch.ones(1, 1, HIDDEN))

    _, exit_signal = adapter.recurrent_step(
        hidden_states=packed,
        position_ids=torch.zeros(1, 1, dtype=torch.long),
        recurrent_steps=torch.zeros(1, dtype=torch.long),
        kv_slot=0,
    )

    assert exit_signal is None


def test_exit_signal_can_be_enabled(model: HuginnForCausalLM) -> None:
    adapter = HuginnCDBAdapter(model)
    adapter.configure_exit_signal(True)
    packed = adapter.pack(torch.ones(1, 1, HIDDEN), torch.ones(1, 1, HIDDEN))

    _, exit_signal = adapter.recurrent_step(
        hidden_states=packed,
        position_ids=torch.zeros(1, 1, dtype=torch.long),
        recurrent_steps=torch.zeros(1, dtype=torch.long),
        kv_slot=0,
    )

    assert exit_signal is not None


def test_recurrent_step_requires_the_scheduler_kv_slot() -> None:
    """Reading the slot from the per-token depths would synchronize and break graph capture."""

    config = HuginnConfig(
        n_embd=HIDDEN,
        n_heads=HEADS,
        n_layers=5,
        n_layers_in_prelude=1,
        n_layers_in_recurrent_block=2,
        n_layers_in_coda=2,
        mean_recurrence=4,
        block_size=64,
        vocab_size=128,
        intermediate_size=16,
        state_init="zero",
    )
    config._attn_implementation = "sdpa"
    adapter = HuginnCDBAdapter(HuginnForCausalLM(config).eval())
    packed = adapter.pack(torch.zeros(1, 1, HIDDEN), torch.randn(1, 1, HIDDEN))

    with pytest.raises(ValueError, match="requires kv_slot"):
        adapter.recurrent_step(
            hidden_states=packed,
            position_ids=torch.zeros(1, 1, dtype=torch.long),
            recurrent_steps=torch.zeros(1, dtype=torch.long),
        )


class _RecordingCache:
    """Fake paged cache that records the layer index of every KV write."""

    def __init__(self) -> None:
        self.layer_indices: list[int] = []

    def update(self, key, value, layer_idx, cache_kwargs=None):
        self.layer_indices.append(int(layer_idx))
        return key, value

    def take(self) -> list[int]:
        indices, self.layer_indices = self.layer_indices, []
        return indices


@pytest.mark.parametrize("policy", KV_CACHE_POLICIES)
def test_stages_write_the_cache_layers_the_allocator_sizes(model: HuginnForCausalLM, policy: str) -> None:
    """Each stage writes exactly the cache layers the layout assigns it.

    Prelude layers hold ``[0, P)``, core slot ``s`` holds ``[P + s*C, P + (s+1)*C)``,
    and the coda holds ``[P + S*C, P + S*C + D)``; across a full unroll the writes
    tile the allocator's cache-layer count with no gaps and no aliasing.
    """

    steps = 4
    adapter = HuginnCDBAdapter(model)
    adapter.configure_recurrent_steps(steps)
    slots = resolve_kv_slots_per_layer(policy, total_recurrent_steps=steps)
    adapter.configure_kv_policy(policy, slots)

    config = model.config
    prelude, core, coda = (
        config.n_layers_in_prelude,
        config.n_layers_in_recurrent_block,
        config.n_layers_in_coda,
    )

    cache = _RecordingCache()
    batch = 3
    input_ids = torch.randint(0, config.vocab_size, (1, batch))
    position_ids = torch.arange(batch).view(1, batch)

    packed = adapter.prelude(input_ids=input_ids, position_ids=position_ids, past_key_values=cache)
    written = cache.take()
    assert written == list(range(prelude))
    seen = set(written)

    for step in range(steps):
        slot = kv_slot_for_step(step, policy=policy, slots_per_layer=slots)
        packed, _ = adapter.recurrent_step(
            packed,
            position_ids,
            recurrent_steps=torch.full((1, batch), step),
            kv_slot=slot,
            past_key_values=cache,
        )
        written = cache.take()
        assert written == list(range(prelude + slot * core, prelude + (slot + 1) * core))
        seen.update(written)

    adapter.lm_head(packed, position_ids=position_ids, past_key_values=cache)
    written = cache.take()
    assert written == list(range(prelude + slots * core, prelude + slots * core + coda))
    seen.update(written)

    num_cache_layers = PagedAttentionCache._infer_num_cache_layers(
        config,
        None,
        kv_policy=policy,
        kv_slots_per_layer=slots,
    )
    assert seen == set(range(num_cache_layers))
