from __future__ import annotations

import time
from collections import Counter
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from looped_cdb.continuous_depth_batching import ContinuousDepthBatchingConfig, ContinuousDepthBatchingEngine
from looped_cdb.continuous_depth_batching.cache import PagedAttentionCache
from looped_cdb.continuous_depth_batching.continuous_api import PRELUDE_SITES, CacheFullError
from looped_cdb.continuous_depth_batching.exit_policy import (
    DecisionHorizon,
    ExitDecisionRule,
    ExitPolicy,
    ExitPolicySpec,
)
from looped_cdb.continuous_depth_batching.model_adapter import CDBModelAdapter
from looped_cdb.continuous_depth_batching.model_runner import ModelRunner, PendingCodaResult
from looped_cdb.continuous_depth_batching.requests import (
    TMP_TOKEN_ID,
    DepthWorkItem,
    PendingGateResult,
    RequestState,
    RequestStatus,
)
from looped_cdb.continuous_depth_batching.scheduler import CDBScheduler
from looped_cdb.kv_cache_policy import LoopedKvLayout


def _config(**overrides: Any) -> SimpleNamespace:
    values = {
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "hidden_size": 4,
        "vocab_size": 16,
        "sliding_window": None,
        "layer_types": ["full_attention", "full_attention", "full_attention"],
        "_attn_implementation": "paged|flash_attention_3",
        "total_ut_steps": 3,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TokenIncrementLoopModel(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(torch.zeros(1))
        self.seen_cu_seq_lens_k: list[list[int]] = []
        self.seen_recurrent_steps: list[list[int]] = []

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> SimpleNamespace:
        # The monolithic forward is the prefill path: it runs the full recurrent depth, so its
        # next-token prediction advances by ``total_ut_steps`` just like ``total_ut_steps`` staged
        # ``recurrent_step`` calls (each of which adds one) would.
        depth = self.config.total_ut_steps
        next_tokens = (input_ids[0].to(dtype=torch.long) + depth) % self.config.vocab_size
        logits = self._logits_for_tokens(next_tokens)
        return SimpleNamespace(logits=logits)

    def _logits_for_tokens(self, next_tokens: torch.Tensor) -> torch.Tensor:
        logits = torch.full(
            (1, next_tokens.numel(), self.config.vocab_size),
            -10.0,
            dtype=torch.float32,
            device=next_tokens.device,
        )
        logits[0, torch.arange(next_tokens.numel(), device=next_tokens.device), next_tokens] = 10.0
        return logits


class GatherAwareTokenIncrementLoopModel(TokenIncrementLoopModel):
    """TokenIncrementLoopModel that honors HF's index-tensor ``logits_to_keep`` like the Ouro model."""

    def forward(
        self, input_ids: torch.Tensor, logits_to_keep: int | torch.Tensor = 0, **kwargs: Any
    ) -> SimpleNamespace:
        output = super().forward(input_ids, **kwargs)
        if isinstance(logits_to_keep, torch.Tensor):
            return SimpleNamespace(logits=output.logits.index_select(1, logits_to_keep))
        return output


class TokenIncrementLoopAdapter(CDBModelAdapter):
    def __init__(
        self,
        model: TokenIncrementLoopModel,
        *,
        exit_policy_spec: ExitPolicySpec | None = None,
    ) -> None:
        self.model = model
        self.kv_policy = "depth_indexed"
        self.kv_slots_per_layer: int | None = None
        self._exit_policy_spec = exit_policy_spec or ExitPolicySpec(ExitDecisionRule.CUMULATIVE_HAZARD)

    def exit_policy_spec(self) -> ExitPolicySpec:
        return self._exit_policy_spec

    def configure_recurrent_steps(self, max_recurrent_steps: int) -> None:
        self.model.config.total_ut_steps = max_recurrent_steps

    def configure_kv_policy(self, kv_policy: str, kv_slots_per_layer: int | None) -> None:
        self.kv_policy = kv_policy
        self.kv_slots_per_layer = kv_slots_per_layer

    def kv_cache_layout(self) -> LoopedKvLayout:
        return LoopedKvLayout.build(
            num_prelude_layers=0,
            num_core_layers=self.model.config.num_hidden_layers,
            num_coda_layers=0,
            total_recurrent_steps=self.model.config.total_ut_steps,
            policy=self.kv_policy,
            requested_slots=self.kv_slots_per_layer,
        )

    def prelude(self, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        del position_ids
        hidden = input_ids.to(dtype=torch.float32).unsqueeze(-1).repeat(1, 1, self.model.config.hidden_size)
        return hidden + self.model.weight

    def recurrent_step(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        recurrent_steps: torch.Tensor,
        cache_position: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cache_position, position_ids
        self.model.seen_cu_seq_lens_k.append(kwargs["cu_seq_lens_k"].detach().cpu().tolist())
        self.model.seen_recurrent_steps.append(recurrent_steps.detach().cpu().tolist())
        hidden_states = hidden_states + 1.0
        gate_values = torch.where(recurrent_steps == 0, 10.0, -10.0).to(dtype=hidden_states.dtype)
        exit_signals = gate_values.view(1, -1, 1).to(device=hidden_states.device)
        return hidden_states, exit_signals

    def lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        next_tokens = hidden_states[0, :, 0].round().to(dtype=torch.long) % self.model.config.vocab_size
        return self.model._logits_for_tokens(next_tokens)


def _cdb_config(**overrides: Any) -> ContinuousDepthBatchingConfig:
    values = {
        "num_blocks": 16,
        "block_size": 4,
        "max_num_batched_tokens": 4,
        "max_model_len": 16,
        "use_async_batching": False,
        "use_cuda_graph": False,
        "max_recurrent_steps": 3,
    }
    values.update(overrides)
    return ContinuousDepthBatchingConfig(**values)


def _exit_policy(
    *,
    threshold: float | None = 0.5,
    delay: bool = False,
    async_io: bool = False,
    synthetic: bool = False,
    spec: ExitPolicySpec | None = None,
    min_steps: int = 1,
    max_steps: int = 3,
) -> ExitPolicy:
    return ExitPolicy(
        spec=spec or ExitPolicySpec(ExitDecisionRule.CUMULATIVE_HAZARD),
        threshold=threshold,
        min_recurrent_steps=min_steps,
        max_recurrent_steps=max_steps,
        delay_gate_consumption=delay,
        use_async_batching=async_io,
        synthetic_exit_replay=synthetic,
    )


@pytest.mark.parametrize("threshold", [0.0, 2.0])
def test_engine_validates_threshold_on_adapter_signal_scale(threshold: float) -> None:
    model = TokenIncrementLoopModel(_config())
    adapter = TokenIncrementLoopAdapter(model, exit_policy_spec=ExitPolicySpec(ExitDecisionRule.DIRECT_THRESHOLD))
    engine = ContinuousDepthBatchingEngine.from_model(
        model,
        _cdb_config(exit_threshold=threshold),
        dtype=torch.float32,
        model_adapter=adapter,
    )
    assert engine.exit_policy.threshold == threshold

    with pytest.raises(ValueError, match="probability policies"):
        ContinuousDepthBatchingEngine.from_model(
            model,
            _cdb_config(exit_threshold=threshold),
            dtype=torch.float32,
            model_adapter=TokenIncrementLoopAdapter(model),
        )


def _cache(**overrides: Any) -> PagedAttentionCache:
    return PagedAttentionCache(
        config=_config(),
        continuous_batching_config=_cdb_config(**overrides),
        device="cpu",
        dtype=torch.float32,
    )


def test_continuous_depth_batching_engine_skips_recurrent_work_with_synthetic_exit_sequence() -> None:
    model = TokenIncrementLoopModel(_config())
    engine = ContinuousDepthBatchingEngine.from_model(
        model,
        _cdb_config(synthetic_exit_replay=True),
        dtype=torch.float32,
        model_adapter=TokenIncrementLoopAdapter(model),
    )

    outputs = engine.generate_batch(
        input_ids=[[1, 2], [5]],
        max_new_tokens=3,
        eos_token_id=None,
        warmup=False,
        exit_depths=[[0, 0], [0, 0]],
    )

    assert [output.generated_tokens for output in outputs] == [[5, 6, 7], [8, 9, 10]]
    assert engine.last_stats.recurrent_steps == 4
    assert engine.last_stats.prefill_recurrent_steps == 9
    assert engine.last_stats.prefill_coda_tokens == 2
    assert engine.last_stats.coda_tokens == 4
    assert engine.last_stats.fixed_depth_recurrent_steps == 12
    assert engine.last_stats.skipped_recurrent_steps == 8
    assert engine.last_stats.exit_depth_histogram == {0: 4}
    assert engine.last_stats.recurrent_batches == engine.last_stats.mixed_recurrent_batches
    assert engine.cache.get_num_free_blocks() == engine.cache.num_blocks
    # Both prompts prefill in a single full-depth, shared-KV monolithic forward; decode never uses it.


def test_continuous_depth_batching_engine_uses_synthetic_exit_sequence() -> None:
    model = TokenIncrementLoopModel(_config())
    engine = ContinuousDepthBatchingEngine.from_model(
        model,
        _cdb_config(synthetic_exit_replay=True),
        dtype=torch.float32,
        model_adapter=TokenIncrementLoopAdapter(model),
    )

    outputs = engine.generate_batch(
        input_ids=[[1, 2], [5]],
        max_new_tokens=3,
        eos_token_id=None,
        warmup=False,
        exit_depths=[[0, 2], [1, 0]],
    )

    assert [len(output.generated_tokens) for output in outputs] == [3, 3]
    assert engine.last_stats.recurrent_steps == 7
    assert engine.last_stats.prefill_recurrent_steps == 9
    assert engine.last_stats.fixed_depth_recurrent_steps == 12
    assert engine.last_stats.skipped_recurrent_steps == 5
    assert engine.last_stats.exit_depth_histogram == {0: 2, 1: 1, 2: 1}
    assert engine.last_stats.mixed_recurrent_batches == engine.last_stats.recurrent_batches
    assert engine.last_stats.max_mixed_depths_per_batch >= 1
    assert engine.last_stats.scalar_split_count == 0
    assert engine.runner.next_hidden_slot <= engine.cache.max_num_batched_tokens
    assert engine.runner.hidden_state_bank is not None
    assert engine.runner.hidden_state_bank.size(1) >= 2


def _delayed(fn: Any, delay_s: float) -> Any:
    """Wrap a callable with a fixed pre-call sleep, to slow a fake model down to wall-clock scale."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        time.sleep(delay_s)
        return fn(*args, **kwargs)

    return wrapper


def _run_replay(
    refill: bool,
    prompts: list[list[int]],
    exit_depths: list[list[int]],
    max_new_tokens: int,
    *,
    force_delayed: bool = False,
    use_early_exit_gate: bool = False,
    eos_token_id: int | None = None,
    arrival_offsets_s: list[float] | None = None,
    step_delay_s: float = 0.0,
    min_coda_batch_size: int = 1,
    max_num_seqs: int | None = None,
    min_free_slots: int | None = None,
    replay_eos_finishes: bool = False,
    use_async_batching: bool = False,
):
    """Run one replay-driven generation and return (engine, per-request generated tokens).

    ``force_delayed`` exercises the async one-step-delayed exit path on CPU (real async
    batching needs CUDA), matching how the other delayed-path tests fake it.
    ``use_early_exit_gate`` runs the live gate alongside the recorded schedule so its
    readout is timed (the exit decision still comes from the schedule).
    ``step_delay_s`` slows every prefill forward and recurrent step by a fixed sleep, so
    open-loop arrival offsets land while earlier requests are still generating.
    """

    model = TokenIncrementLoopModel(_config())
    adapter = TokenIncrementLoopAdapter(model)
    if step_delay_s:
        model.forward = _delayed(model.forward, step_delay_s)
        adapter.recurrent_step = _delayed(adapter.recurrent_step, step_delay_s)
    engine = ContinuousDepthBatchingEngine.from_model(
        model,
        _cdb_config(
            synthetic_exit_replay=True,
            refill=refill,
            min_coda_batch_size=min_coda_batch_size,
            replay_eos_finishes=replay_eos_finishes,
            use_async_batching=use_async_batching,
            **({} if max_num_seqs is None else {"max_num_seqs": max_num_seqs}),
            **({} if min_free_slots is None else {"min_free_slots": min_free_slots}),
        ),
        dtype=torch.float32,
        model_adapter=adapter,
    )
    if force_delayed:
        engine._uses_delayed_synthetic_exit = lambda: True  # type: ignore[method-assign]
    outputs = engine.generate_batch(
        input_ids=prompts,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        warmup=False,
        exit_depths=exit_depths,
        model_kwargs={"use_early_exit_gate": use_early_exit_gate},
        arrival_offsets_s=arrival_offsets_s,
    )
    return engine, [output.generated_tokens for output in outputs]


@pytest.mark.parametrize("refill", [True, False])
def test_timed_arrivals_match_drain_for_both_refill_policies(refill: bool) -> None:
    # Open-loop release changes only when a request becomes schedulable, not its compute, so
    # the replay must produce drain-identical tokens through both serving loops. The last
    # request arrives long after the earlier ones drain, so each loop's idle-sleep path runs.
    prompts = [[1, 2], [5], [3, 4, 5]]
    exit_depths = [[0, 2], [1, 0], [2, 1]]

    _, drain_tokens = _run_replay(refill, prompts, exit_depths, max_new_tokens=3)
    engine, timed_tokens = _run_replay(
        refill, prompts, exit_depths, max_new_tokens=3, arrival_offsets_s=[0.0, 0.05, 0.2]
    )

    assert timed_tokens == drain_tokens
    outputs = [engine.finished_outputs[f"request-{idx}"] for idx in range(len(prompts))]
    for output in outputs:
        assert output.created_time <= output.lifespan[0] <= output.first_token_time <= output.lifespan[1]
    # Arrival stamps are the scheduled release instants, exactly the offset gap apart.
    assert outputs[2].created_time - outputs[0].created_time == pytest.approx(0.2)


@pytest.mark.parametrize("refill", [True, False])
def test_dense_arrivals_release_into_a_busy_engine_for_both_refill_policies(refill: bool) -> None:
    # Dense offsets against a first cohort still generating: the later requests are released
    # into a scheduler that already holds active depth work (refill loop) or mid-wave state
    # (no-refill loop), the mid-run admission path a real serving sweep exercises. Only the
    # release timing changes, so tokens must stay drain-identical. The per-step delay keeps
    # the first cohort resident well past the later offsets (>= 9 recurrent launches x 5 ms
    # against 20/40 ms arrivals), so the overlap is guaranteed, not left to scheduling luck.
    prompts = [[1, 2], [5], [3, 4, 5], [2], [1]]
    exit_depths = [
        [0, 1, 2, 0, 1],
        [2, 0, 1, 2, 0],
        [1, 2, 0, 1, 2],
        [0, 0, 2, 1, 0],
        [2, 1, 0, 2, 1],
    ]

    _, drain_tokens = _run_replay(refill, prompts, exit_depths, max_new_tokens=6, step_delay_s=0.005)
    engine, timed_tokens = _run_replay(
        refill,
        prompts,
        exit_depths,
        max_new_tokens=6,
        arrival_offsets_s=[0.0, 0.0, 0.02, 0.02, 0.04],
        step_delay_s=0.005,
    )

    assert timed_tokens == drain_tokens
    outputs = [engine.finished_outputs[f"request-{idx}"] for idx in range(len(prompts))]
    # The busy-release property, asserted from the stamps: some request arrived while another
    # was between first schedule and finish. Without it, a faster fake model would silently
    # turn this back into an idle-path test.
    assert any(
        other is not output and other.lifespan[0] <= output.created_time <= other.lifespan[1]
        for output in outputs
        for other in outputs
    )
    for output in outputs:
        assert output.created_time <= output.lifespan[0] <= output.first_token_time <= output.lifespan[1]


def test_no_refill_matches_refill_for_the_same_exit_schedule() -> None:
    # Refill changes only scheduling, not per-token compute, so replaying the same
    # exit depths through both policies must yield identical tokens and depth work.
    prompts = [[1, 2], [5], [3, 4, 5]]
    exit_depths = [[0, 2], [1, 0], [2, 1]]

    refill_engine, refill_tokens = _run_replay(True, prompts, exit_depths, max_new_tokens=3)
    no_refill_engine, no_refill_tokens = _run_replay(False, prompts, exit_depths, max_new_tokens=3)

    # The two policies must genuinely differ in scheduling: refill mixes work through
    # the depth queue, no-refill never does.
    assert refill_engine.last_stats.scheduler_refills > 0
    assert no_refill_engine.last_stats.scheduler_refills == 0

    assert no_refill_tokens == refill_tokens
    assert no_refill_engine.last_stats.exit_depth_histogram == refill_engine.last_stats.exit_depth_histogram
    assert no_refill_engine.last_stats.recurrent_steps == refill_engine.last_stats.recurrent_steps
    assert no_refill_engine.last_stats.coda_tokens == refill_engine.last_stats.coda_tokens
    assert no_refill_engine.last_stats.prefill_recurrent_steps == refill_engine.last_stats.prefill_recurrent_steps


def test_no_refill_matches_refill_on_the_async_delayed_path() -> None:
    # The headline baseline runs no-refill with the same async delayed-gate execution
    # as CDB (exits applied one recurrent step late), so both must agree token-for-token
    # while remaining genuinely different schedulers.
    prompts = [[1, 2], [5], [3, 4, 5]]
    exit_depths = [[0, 1], [1, 0], [0, 1]]

    refill_engine, refill_tokens = _run_replay(True, prompts, exit_depths, 3, force_delayed=True)
    no_refill_engine, no_refill_tokens = _run_replay(False, prompts, exit_depths, 3, force_delayed=True)

    assert refill_engine.last_stats.scheduler_refills > 0
    assert no_refill_engine.last_stats.scheduler_refills == 0
    assert no_refill_tokens == refill_tokens
    assert no_refill_engine.last_stats.exit_depth_histogram == refill_engine.last_stats.exit_depth_histogram
    assert no_refill_engine.last_stats.recurrent_steps == refill_engine.last_stats.recurrent_steps
    # No-refill is the strict wave with an async boundary: exited tokens wait for the cohort to
    # drain and exactly one wave-sized coda runs per decode wave with at least one exiter (every
    # wave in this fixture), never a per-step sliver. That coda is staged asynchronously with the
    # exiters' successors carried behind it, before their prelude, so the boundary does not serialize on the
    # host; the successors and any freshly prefilled entries join the next wave through its single
    # wave_cohort prelude launch, which gathers their tokens from the staged results' device
    # outputs - the wave never fragments its prelude into per-site slivers, and the mixed-depth
    # queue stays untouched. On CPU the staging falls back to synchronous and the refill
    # comparator runs without eager re-entry (its re-entry gates on use_async_batching), so this
    # test pins token/depth equality across the two mechanisms; the graphed async equivalence is
    # pinned by test_no_refill_matches_refill_on_gpu_with_real_graphs.
    waves = sum(1 for steps in no_refill_engine.model.seen_recurrent_steps if set(steps) == {0})
    assert no_refill_engine.last_stats.coda_batches == waves
    assert set(no_refill_engine.last_stats.prelude_batches) == {"wave_cohort"}
    assert no_refill_engine.last_stats.prelude_batches["wave_cohort"] == waves
    # Successors really are carried across the boundary rather than restaged from host tokens by
    # the next wave's cohort scan: the scan alone would also yield one launch per wave.
    assert no_refill_engine.last_stats.prelude_gathered_tokens["wave_cohort"] > 0
    assert no_refill_engine.last_stats.mixed_recurrent_batches == 0
    # A launch width below the prefill batch's prompt count caps the carried entries: prefill
    # admission is token-budget-gated, so the overflow must fall back to consume-time staging
    # instead of overflowing the recurrent launch buffers.
    _, narrow_refill = _run_replay(True, prompts, exit_depths, 3, force_delayed=True, max_num_seqs=2)
    narrow_engine, narrow_tokens = _run_replay(False, prompts, exit_depths, 3, force_delayed=True, max_num_seqs=2)
    assert narrow_tokens == narrow_refill
    assert narrow_tokens == no_refill_tokens
    assert set(narrow_engine.last_stats.prelude_batches) == {"wave_cohort"}
    for launch_steps in no_refill_engine.model.seen_recurrent_steps:
        assert len(set(launch_steps)) == 1, f"non-homogeneous recurrent launch: {launch_steps}"

    # More requests than the decode batch, with staggered lengths so requests retire mid-run and
    # the wave has to admit their replacements. Every wave still enters through one prelude launch.
    # What that launch computes when its rows take ids from two sources is pinned directly by
    # :func:`test_fused_prelude_matches_a_host_only_launch`: the wave loop no longer reaches the
    # mixed case on the ordinary path, since the eager budget covers every freed slot and a fresh
    # decoder is gathered from its prefill batch rather than staged from the host.
    fused_prompts = [[1, 2], [5], [3, 4, 5], [7, 8], [9], [2, 3], [6, 6], [4]]
    fused_lengths = [3 + (idx % 4) for idx in range(len(fused_prompts))]
    fused_depths = [[(idx + step) % 2 for step in range(n - 1)] for idx, n in enumerate(fused_lengths)]
    _, fused_refill = _run_replay(True, fused_prompts, fused_depths, fused_lengths, force_delayed=True, max_num_seqs=3)
    fused_engine, fused_tokens = _run_replay(
        False, fused_prompts, fused_depths, fused_lengths, force_delayed=True, max_num_seqs=3
    )
    assert fused_tokens == fused_refill
    assert set(fused_engine.last_stats.prelude_batches) == {"wave_cohort"}


@pytest.mark.parametrize("force_delayed", [False, True])
def test_no_refill_prefills_to_exhaustion_before_a_wave(force_delayed: bool) -> None:
    # Prefill keeps priority until it can no longer place a prompt, as the full-depth engine's
    # per-tick priority does and as alg:no-refill's CanPrefill loop specifies. One prefill batch
    # carries only max_num_batched_tokens, so admitting a single batch per wave would decode
    # cohorts that grow one batch at a time and hand the baseline an admission handicap the
    # full-depth engine never pays. On the delayed path the batches also queue several readbacks
    # and several carried entry groups into the one wave that follows them. Admission opens
    # whenever a slot is free, so the token budget alone splits the prompts into launches.
    prompts = [[1, 2], [3, 4], [5, 6], [7, 8]]
    exit_depths = [[0, 0]] * len(prompts)
    refill_engine, refill_tokens = _run_replay(
        True, prompts, exit_depths, 3, force_delayed=force_delayed, max_num_seqs=4, min_free_slots=1
    )
    engine, tokens = _run_replay(
        False, prompts, exit_depths, 3, force_delayed=force_delayed, max_num_seqs=4, min_free_slots=1
    )

    assert tokens == refill_tokens
    # Four two-token prompts against a four-token budget: two launches, both before the first wave.
    assert engine.last_stats.prefill_batches == 2
    assert refill_engine.last_stats.prefill_batches == 2
    # Every request then takes its two remaining tokens from one full-cohort wave each, so the
    # wave count is the per-request token count and no wave runs below the resident set.
    assert engine.last_stats.prelude_batches["wave_cohort"] == 2
    assert engine.last_stats.prelude_tokens["wave_cohort"] == 2 * len(prompts)
    # Both loops sample the admitted set, and both sample it after prefill has stopped placing
    # prompts, so a row cannot report a residency smaller than the set a wave actually decoded.
    assert engine.scheduler.max_resident_requests == len(prompts)
    assert refill_engine.scheduler.max_resident_requests == len(prompts)


def test_fused_prelude_matches_a_host_only_launch() -> None:
    # A launch whose leading rows take their ids from a staged batch's device tokens must compute
    # exactly what a launch carrying the same ids from the host computes. Driving this through the
    # scheduler only produces it when a wave happens to mix its two sources, so the primitive is
    # exercised directly: the gathered rows sit at non-contiguous rows of the device tokens, so a
    # wrong row or a wrong destination offset silently preludes the wrong token and shows up here.
    model = TokenIncrementLoopModel(_config())
    engine = ContinuousDepthBatchingEngine.from_model(
        model,
        _cdb_config(synthetic_exit_replay=True),
        dtype=torch.float32,
        model_adapter=TokenIncrementLoopAdapter(model),
    )
    runner = engine.runner
    token_ids = [5, 9, 3, 7]

    def prelude_rows(items: list[DepthWorkItem], device_groups: Any = ()) -> torch.Tensor:
        runner.compute_prelude_batch(items, device_groups=device_groups)
        bank = runner._require_hidden_state_bank()
        return torch.stack([bank[0, item.hidden_slot].clone() for item in items])

    def items_for(ids: list[int]) -> list[DepthWorkItem]:
        return [
            DepthWorkItem(
                state=RequestState(request_id=f"r{idx}", initial_tokens=[1]),
                token_id=token_id,
                token_position=idx,
                recurrent_step=0,
            )
            for idx, token_id in enumerate(ids)
        ]

    host_only = prelude_rows(items_for(token_ids))
    # The stage must actually depend on the token id, or the comparison below proves nothing.
    assert len({tuple(row.tolist()) for row in host_only}) == len(token_ids)

    # The first two ids arrive on device at rows 1 and 3 of a staged batch's sampled tokens.
    staged = PendingCodaResult(
        items=[],
        host_tokens=torch.tensor([0, token_ids[0], 0, token_ids[1]], dtype=torch.long),
        device_tokens=torch.tensor([0, token_ids[0], 0, token_ids[1]], dtype=torch.long),
    )
    mixed = items_for([TMP_TOKEN_ID, TMP_TOKEN_ID, *token_ids[2:]])
    fused = prelude_rows(mixed, device_groups=[(staged, [1, 3])])

    assert torch.equal(fused, host_only)


def test_prelude_launches_are_counted_per_site_and_refill_fragments_them() -> None:
    # Every generated token but a request's first needs a prelude launch, and where that launch
    # runs decides how wide it is: the no-refill wave takes a whole decode cohort at once, while
    # refill runs each prefill batch's first tokens and then each coda's successors separately.
    # The prelude reloads its weights per launch, so the split is a real cost and has to be
    # visible per site rather than hidden in a single total.
    prompts = [[1, 2], [5], [3, 4, 5], [7, 8], [9]]
    exit_depths = [[0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]]

    refill_engine, _ = _run_replay(True, prompts, exit_depths, 4)
    no_refill_engine, _ = _run_replay(False, prompts, exit_depths, 4)

    decode_tokens = len(prompts) * 4 - len(prompts)
    for engine in (refill_engine, no_refill_engine):
        stats = engine.last_stats
        assert sum(stats.prelude_tokens.values()) == decode_tokens
        assert set(stats.prelude_batches) <= set(PRELUDE_SITES)
        assert all(stats.prelude_batches[site] > 0 for site in stats.prelude_tokens)

    # The wave stages one launch per cohort; refill splits the same tokens across the prefill
    # boundary and the coda consumers, so it pays more launches and each carries fewer tokens.
    refill_launches = sum(refill_engine.last_stats.prelude_batches.values())
    wave_launches = sum(no_refill_engine.last_stats.prelude_batches.values())
    assert refill_launches > wave_launches
    assert set(no_refill_engine.last_stats.prelude_batches) == {"wave_cohort"}
    assert "after_prefill" in refill_engine.last_stats.prelude_batches


def test_no_refill_delayed_path_cancels_the_successor_of_a_late_eos() -> None:
    # An EOS is discovered only when the boundary coda is consumed, one wave after the successor
    # was eagerly re-entered. The successor must be cancelled instead of emitting an extra token,
    # and the outputs must match the delayed refill scheduler, which discovers its late EOS the
    # same way.
    # Request 2's very first sampled token is the EOS: its coda successor was carried into the
    # next wave, but the prefill consume retires the request before that wave launches, so the
    # successor is dropped at the wave entry instead of writing into reallocated KV.
    prompts = [[1], [4], [5]]
    exit_depths = [[0] * 5, [1] * 5, [0] * 5]

    refill_engine, refill_tokens = _run_replay(True, prompts, exit_depths, 6, force_delayed=True, eos_token_id=8)
    no_refill_engine, no_refill_tokens = _run_replay(False, prompts, exit_depths, 6, force_delayed=True, eos_token_id=8)

    assert no_refill_tokens == refill_tokens
    assert no_refill_tokens[2] == [8], "the first sampled token must already be the EOS"
    assert any(tokens and tokens[-1] == 8 for tokens in no_refill_tokens), "the schedule must hit the EOS token"
    assert no_refill_engine.last_stats.cancelled_depth_items > 0, (
        "a late EOS must cancel its eagerly re-entered successor"
    )
    # The wave charges its cancelled successor exactly one recurrent step (cancellation fires at
    # step 0, after the launch); the refill scheduler drops the cancelled item before the next
    # launch and charges none. Request 2's first-token EOS adds the wave's two recurrent steps
    # for that token (exit depth 0, applied one step late): the wave had already run them when
    # the readback revealed the EOS, while the refill consume cancels before the first launch.
    # Its carried successor is dropped at the next wave's entry and charges nothing.
    assert no_refill_engine.last_stats.recurrent_steps == refill_engine.last_stats.recurrent_steps + 3


def test_eos_finishes_cancels_each_final_tokens_successor() -> None:
    # With ``replay_eos_finishes`` the length-limit finish is no longer predicted host-side, so
    # every request's final token eagerly stages a successor that the consume must cancel, exactly
    # like a live-served EOS. Eager staging only exists on the async path; the generated tokens
    # must not change.
    prompts = [[1, 2], [5], [3, 4]]
    exit_depths = [[0, 1, 1], [1, 0, 1], [0, 1, 0]]

    predicted_engine, predicted_tokens = _run_replay(
        True, prompts, exit_depths, 4, force_delayed=True, use_async_batching=True
    )
    late_engine, late_tokens = _run_replay(
        True, prompts, exit_depths, 4, force_delayed=True, use_async_batching=True, replay_eos_finishes=True
    )

    assert late_tokens == predicted_tokens
    assert predicted_engine.last_stats.cancelled_depth_items == 0
    assert late_engine.last_stats.cancelled_depth_items == len(prompts), (
        "each final token's eager successor must be discovered late and cancelled"
    )


def test_synthetic_replay_times_gate_readout_only_when_gate_is_on() -> None:
    # The synthetic benchmark must pay the live gate's GPU->CPU readout cost instead of
    # hiding it: with the gate on, the delayed path stages a gate copy per recurrent step
    # (and awaits it a step later); with the gate off, no gate runs and nothing is staged.
    # The recorded schedule drives the same tokens either way.
    prompts = [[1, 2], [5], [3, 4, 5]]
    exit_depths = [[0, 1], [1, 0], [0, 1]]

    staged: dict[bool, int] = {}
    tokens: dict[bool, list[list[int]]] = {}
    for gate_on in (True, False):
        model = TokenIncrementLoopModel(_config())
        engine = ContinuousDepthBatchingEngine.from_model(
            model,
            _cdb_config(synthetic_exit_replay=True, refill=False, use_async_batching=True),
            dtype=torch.float32,
            model_adapter=TokenIncrementLoopAdapter(model),
        )
        calls = 0
        original = engine.runner.stage_delayed_exit_signals

        def counting_stage(items, exit_signals, *, _original=original):
            nonlocal calls
            calls += 1
            return _original(items, exit_signals)

        engine.runner.stage_delayed_exit_signals = counting_stage  # type: ignore[method-assign]
        outputs = engine.generate_batch(
            input_ids=prompts,
            max_new_tokens=3,
            eos_token_id=None,
            warmup=False,
            exit_depths=exit_depths,
            model_kwargs={"use_early_exit_gate": gate_on},
        )
        staged[gate_on] = calls
        tokens[gate_on] = [output.generated_tokens for output in outputs]

    assert staged[True] > 0, "gate-on synthetic replay must stage the gate readout to time it"
    assert staged[False] == 0, "gate-off synthetic replay must not run or read the gate"
    assert tokens[True] == tokens[False], "the recorded schedule must drive the same tokens either way"


def _needs_cpu_gate(*, synthetic: bool, threshold: float | None, delay: bool, async_io: bool, gate_on: bool) -> bool:
    runner = ModelRunner.__new__(ModelRunner)
    runner.exit_policy = _exit_policy(
        threshold=threshold,
        synthetic=synthetic,
        delay=delay,
        async_io=async_io,
    )
    return runner._recurrent_batch_needs_cpu_exit_signals({"use_early_exit_gate": gate_on})


def test_synthetic_replay_blocks_on_the_gate_only_on_the_synchronous_path() -> None:
    # The synchronous path must incur a blocking GPU->CPU gate readout after every recurrent
    # step under synthetic replay (so the sync ablation pays that stall), while the async
    # delayed path stages the copy instead of blocking. A disabled gate costs nothing.
    assert _needs_cpu_gate(synthetic=True, threshold=None, delay=False, async_io=False, gate_on=True)
    assert not _needs_cpu_gate(synthetic=True, threshold=None, delay=True, async_io=True, gate_on=True)
    assert not _needs_cpu_gate(synthetic=True, threshold=None, delay=False, async_io=False, gate_on=False)
    assert not _needs_cpu_gate(synthetic=False, threshold=None, delay=False, async_io=False, gate_on=True)


def test_no_refill_never_refills_and_keeps_recurrent_batches_homogeneous() -> None:
    # The no-refill baseline must (1) never touch the mixed-depth ready queue and
    # (2) launch each recurrent step on a single recurrent depth (the cohort shrinks
    # in lockstep), so freed depth slots are left empty rather than refilled.
    engine, _ = _run_replay(
        refill=False,
        prompts=[[1, 2], [5], [3, 4]],
        exit_depths=[[0, 2], [1, 0], [2, 1]],
        max_new_tokens=3,
    )

    assert engine.last_stats.scheduler_refills == 0
    assert engine.last_stats.mixed_recurrent_batches == 0
    assert engine.model.seen_recurrent_steps, "expected at least one recurrent launch"
    for launch_steps in engine.model.seen_recurrent_steps:
        assert len(set(launch_steps)) == 1, f"non-homogeneous recurrent launch: {launch_steps}"


def test_delayed_synthetic_exit_sequence_exits_one_recurrent_step_later() -> None:
    model = TokenIncrementLoopModel(_config())
    engine = ContinuousDepthBatchingEngine.from_model(
        model,
        _cdb_config(synthetic_exit_replay=True),
        dtype=torch.float32,
        model_adapter=TokenIncrementLoopAdapter(model),
    )
    engine._uses_delayed_synthetic_exit = lambda: True  # type: ignore[method-assign]
    state = RequestState(request_id="r0", initial_tokens=[1], synthetic_exit_depths=[0])
    state.status = RequestStatus.DECODING
    state.tokens_to_process = [1]
    item = engine._stage_decode_token(state)
    assert item is not None
    engine.runner.compute_prelude_batch([item])

    engine._run_mixed_recurrent_batch([item], recurrent_kwargs={})

    assert len(engine.scheduler.ready_queue) == 1
    delayed_item = engine.scheduler.ready_queue.popleft()
    assert delayed_item.recurrent_step == 1
    assert delayed_item.pending_synthetic_exit_step == 0
    assert len(engine.scheduler.coda_queue) == 0

    engine._run_mixed_recurrent_batch([delayed_item], recurrent_kwargs={})

    assert len(engine.scheduler.ready_queue) == 0
    assert len(engine.scheduler.coda_queue) == 1
    assert delayed_item.pending_synthetic_exit_step is None
    assert state.exit_depths == [1]
    assert engine.last_stats.exit_depth_histogram == {1: 1}


def test_cdb_scheduler_batches_mixed_recurrent_steps_fifo() -> None:
    cache = _cache()
    scheduler = CDBScheduler(cache=cache, max_recurrent_steps=3, max_num_seqs=2)
    state = RequestState(request_id="r0", initial_tokens=[1])
    scheduler.enqueue_depth(DepthWorkItem(state=state, token_id=1, token_position=0, recurrent_step=1))
    scheduler.enqueue_depth(DepthWorkItem(state=state, token_id=2, token_position=1, recurrent_step=2))
    scheduler.enqueue_depth(DepthWorkItem(state=state, token_id=3, token_position=2, recurrent_step=2))

    batch = scheduler.schedule_next(token_budget=cache.max_num_batched_tokens, cache_budget=cache.num_pages)

    assert batch is not None
    assert batch.kind == "recurrent"
    assert batch.depth_items is not None
    assert [item.token_id for item in batch.depth_items] == [1, 2]
    assert [item.recurrent_step for item in batch.depth_items] == [1, 2]
    assert scheduler.mixed_batches_scheduled == 1


def test_cdb_scheduler_prefills_waiting_prompts_before_underfilled_recurrent_batch() -> None:
    cache = _cache(max_num_batched_tokens=4)
    scheduler = CDBScheduler(cache=cache, max_recurrent_steps=3, max_num_seqs=2, safety_margin=0.0)
    depth_state = RequestState(request_id="decode", initial_tokens=[1])
    prefill_state = RequestState(request_id="prefill", initial_tokens=[4, 5])
    scheduler.enqueue_depth(DepthWorkItem(state=depth_state, token_id=1, token_position=0, recurrent_step=0))
    scheduler.add_waiting_request(prefill_state)

    batch = scheduler.schedule_next(token_budget=cache.max_num_batched_tokens, cache_budget=cache.num_pages)

    assert batch is not None
    assert batch.kind == "prefill"
    assert batch.requests is not None
    assert [future_state.state.request_id for future_state in batch.requests] == ["prefill"]
    # The whole prompt now prefills in one batch (no per-token cap), so it is ready to decode.
    assert prefill_state.status.name == "DECODING"
    assert prefill_state.tokens_to_process == [4, 5]
    assert prefill_state.remaining_prefill_tokens == []
    assert len(scheduler.ready_queue) == 1


def test_cdb_scheduler_does_not_prefill_request_with_in_flight_token() -> None:
    cache = _cache(max_num_batched_tokens=4)
    scheduler = CDBScheduler(cache=cache, max_recurrent_steps=3, max_num_seqs=4, safety_margin=0.0)
    state = RequestState(request_id="prefill", initial_tokens=[4, 5])
    state.status = RequestStatus.PREFILLING
    state.remaining_prefill_tokens = [5]
    state.tokens_to_process = [4]
    state.allocated_blocks = cache.allocate_blocks(1, state.request_id, state.allocated_blocks) or 0
    scheduler.active_requests[state.request_id] = state
    scheduler.enqueue_depth(DepthWorkItem(state=state, token_id=4, token_position=0, recurrent_step=0))

    batch = scheduler.schedule_next(token_budget=cache.max_num_batched_tokens, cache_budget=cache.num_pages)

    assert batch is not None
    assert batch.kind == "recurrent"
    assert batch.depth_items is not None
    assert [item.state.request_id for item in batch.depth_items] == ["prefill"]
    assert state.remaining_prefill_tokens == [5]

    scheduler.release_in_flight_request(state.request_id)
    next_batch = scheduler.schedule_next(token_budget=cache.max_num_batched_tokens, cache_budget=cache.num_pages)

    assert next_batch is not None
    assert next_batch.kind == "prefill"
    assert next_batch.requests is not None
    assert [future.state.request_id for future in next_batch.requests] == ["prefill"]


def test_cdb_scheduler_prioritizes_coda_over_recurrent_and_prefill() -> None:
    cache = _cache()
    scheduler = CDBScheduler(cache=cache, max_recurrent_steps=3, max_num_seqs=2, safety_margin=0.0)
    state = RequestState(request_id="r0", initial_tokens=[1])
    scheduler.add_waiting_request(RequestState(request_id="prefill", initial_tokens=[2]))
    scheduler.enqueue_depth(DepthWorkItem(state=state, token_id=1, token_position=0, recurrent_step=0))
    scheduler.enqueue_coda(DepthWorkItem(state=state, token_id=1, token_position=0, recurrent_step=1))

    batch = scheduler.schedule_next(token_budget=cache.max_num_batched_tokens, cache_budget=cache.num_pages)

    assert batch is not None
    assert batch.kind == "coda"
    assert batch.depth_items is not None
    assert [item.recurrent_step for item in batch.depth_items] == [1]


def test_async_token_reentry_matches_synchronous_outputs() -> None:
    # Depth 3 makes every token = previous + 3 (mod vocab), so the streams are exact. eos=13 stops
    # request 0 mid-stream: its successor was already eagerly staged from device tokens, so the EOS
    # is discovered one consume late and the staged item is cancelled before it runs any recurrent
    # step. Request 1 is length-capped: its in-flight token meets the limit, so it is never
    # re-entered. Request 3's very first sampled token is the EOS: its first decode item was
    # eagerly staged from the prefill batch's device tokens, so the finish is again discovered one
    # consume late and the item cancelled. Total recurrent work must match the synchronous loop.
    prompts = [[1], [2, 5], [7, 7, 1], [10]]
    max_new_tokens = [6, 2, 4, 3]

    def run(use_async_batching: bool) -> tuple[ContinuousDepthBatchingEngine, list[list[int]]]:
        model = TokenIncrementLoopModel(_config())
        engine = ContinuousDepthBatchingEngine.from_model(
            model,
            _cdb_config(
                num_blocks=64,
                max_num_batched_tokens=8,
                max_model_len=32,
                use_async_batching=use_async_batching,
            ),
            dtype=torch.float32,
            model_adapter=TokenIncrementLoopAdapter(model),
        )
        outputs = engine.generate_batch(
            input_ids=[list(prompt) for prompt in prompts],
            max_new_tokens=list(max_new_tokens),
            eos_token_id=13,
            warmup=False,
        )
        return engine, [output.generated_tokens for output in outputs]

    sync_engine, sync_tokens = run(False)
    async_engine, async_tokens = run(True)

    assert sync_tokens == [[4, 7, 10, 13], [8, 11], [4, 7, 10, 13], [13]]
    assert async_tokens == sync_tokens
    assert async_engine.last_stats.prelude_tokens["after_coda_eager"] > 0
    assert async_engine.last_stats.prelude_tokens["after_prefill_eager"] > 0
    # Request 0's post-EOS successor and request 3's post-EOS first decode item.
    assert async_engine.last_stats.cancelled_depth_items >= 2
    assert async_engine.last_stats.recurrent_steps == sync_engine.last_stats.recurrent_steps
    assert async_engine.cache.get_num_free_blocks() == async_engine.cache.num_blocks


def test_engine_runs_recurrent_work_while_coda_result_is_pending() -> None:
    class FakePendingCoda:
        def is_ready(self) -> bool:
            return False

    model = TokenIncrementLoopModel(_config())
    engine = ContinuousDepthBatchingEngine.from_model(
        model,
        _cdb_config(max_num_seqs=2),
        dtype=torch.float32,
        model_adapter=TokenIncrementLoopAdapter(model),
    )
    events: list[str] = []
    coda_state = RequestState(request_id="coda", initial_tokens=[1])
    recurrent_state = RequestState(request_id="recurrent", initial_tokens=[2])
    engine.scheduler.enqueue_coda(DepthWorkItem(state=coda_state, token_id=1, token_position=0, recurrent_step=0))
    engine.scheduler.enqueue_depth(DepthWorkItem(state=recurrent_state, token_id=2, token_position=0, recurrent_step=0))

    def stage_coda(items: list[DepthWorkItem], *, allow_async: bool = True) -> FakePendingCoda:
        del items, allow_async
        events.append("stage_coda")
        return FakePendingCoda()

    def run_recurrent(items: list[DepthWorkItem], *, recurrent_kwargs: dict[str, object]) -> None:
        del items, recurrent_kwargs
        events.append("recurrent")

    def consume_coda(pending: FakePendingCoda) -> None:
        del pending
        events.append("consume_coda")

    engine.runner.stage_coda_batch = stage_coda  # type: ignore[method-assign]
    engine._run_mixed_recurrent_batch = run_recurrent  # type: ignore[method-assign]
    engine._consume_coda_result = consume_coda  # type: ignore[method-assign]

    engine._run_scheduler_loop(model_kwargs=None)

    assert events == ["stage_coda", "recurrent", "consume_coda"]


def test_runner_reuses_hidden_state_bank_slots_after_coda() -> None:
    model = TokenIncrementLoopModel(_config())
    engine = ContinuousDepthBatchingEngine.from_model(
        model,
        _cdb_config(max_num_seqs=2),
        dtype=torch.float32,
        model_adapter=TokenIncrementLoopAdapter(model),
    )
    state_0 = RequestState(request_id="r0", initial_tokens=[1])
    state_1 = RequestState(request_id="r1", initial_tokens=[2])
    items = [
        DepthWorkItem(state=state_0, token_id=1, token_position=0, recurrent_step=0),
        DepthWorkItem(state=state_1, token_id=2, token_position=0, recurrent_step=0),
    ]

    engine.runner.compute_prelude_batch(items)
    first_slots = [item.hidden_slot for item in items]
    assert first_slots == [0, 1]

    engine.runner.compute_coda_batch(items)
    assert [item.hidden_slot for item in items] == [None, None]
    assert sorted(engine.runner.free_hidden_slots) == [0, 1]

    next_items = [
        DepthWorkItem(state=state_0, token_id=3, token_position=1, recurrent_step=0),
        DepthWorkItem(state=state_1, token_id=4, token_position=1, recurrent_step=0),
    ]
    engine.runner.compute_prelude_batch(next_items)

    assert sorted(item.hidden_slot for item in next_items if item.hidden_slot is not None) == [0, 1]
    assert engine.runner.next_hidden_slot == 2


def test_recurrent_stage_uses_static_bucket_shape_when_padding_is_enabled() -> None:
    model = TokenIncrementLoopModel(_config())
    engine = ContinuousDepthBatchingEngine.from_model(
        model,
        _cdb_config(max_num_seqs=4),
        dtype=torch.float32,
        model_adapter=TokenIncrementLoopAdapter(model),
    )
    engine.runner.pad_inputs = True
    items = []
    for idx in range(3):
        state = RequestState(request_id=f"r{idx}", initial_tokens=[idx + 1])
        state.status = RequestStatus.DECODING
        state.tokens_to_process = [idx + 1]
        item = engine._stage_decode_token(state)
        assert item is not None
        items.append(item)

    engine.runner.compute_prelude_batch(items)
    _, exit_signals = engine.runner.compute_recurrent_batch(items)

    assert exit_signals.shape == (1, 3, 1)
    assert engine.model.seen_recurrent_steps == [[0, 0, 0, 0]]

    buffers = engine.runner.recurrent_stage_buffers[(4, torch.float32)]
    assert buffers.host_slot_indices[:3].tolist() == [item.hidden_slot for item in items]
    assert buffers.host_position_ids[0].tolist() == [item.token_position for item in items] + [0]
    # Padding rows follow the CB decode fast path: plateaued cumsums and all -1 block-table rows.
    assert buffers.host_cu_seq_lens_q.tolist() == [0, 1, 2, 3, 3]
    expected_k = [0]
    for item in items:
        expected_k.append(expected_k[-1] + item.token_position + 1)
    assert buffers.host_cu_seq_lens_k.tolist() == [*expected_k, expected_k[-1]]
    cache = engine.runner.cache
    for idx, item in enumerate(items):
        blocks = cache.cache_allocator.block_table[item.state.request_id]
        row = buffers.host_block_table[0, idx].tolist()
        assert row == blocks + [-1] * (len(row) - len(blocks))
    assert buffers.host_block_table[0, 3].eq(-1).all()


def test_min_coda_batch_size_batches_codas_and_stays_token_neutral() -> None:
    """Coda batching changes launch counts, never tokens, and the flush keeps it live.

    Staggered exit depths make exits trickle in over several ticks, so coda-first
    scheduling launches many small codas; holding them until ``min_coda_batch_size``
    exits pool must produce the same generations in fewer, larger launches. A floor
    far above anything the workload can pool exercises the flush-when-idle path on
    every launch: the engine must still complete rather than deadlock.
    """

    prompts = [[1, 2], [3, 4], [5, 6], [7, 8]]
    exit_depths = [[0, 2, 1], [1, 0, 2], [2, 1, 0], [0, 1, 2]]
    outputs: dict[int, list[list[int]]] = {}
    stats: dict[int, Any] = {}
    for min_coda in (1, 3, 64):
        engine, tokens = _run_replay(True, prompts, exit_depths, max_new_tokens=4, min_coda_batch_size=min_coda)
        outputs[min_coda] = [list(t) for t in tokens]
        stats[min_coda] = engine.last_stats
    assert outputs[3] == outputs[1]
    assert outputs[64] == outputs[1]
    assert stats[3].coda_tokens == stats[1].coda_tokens
    assert stats[3].coda_batches < stats[1].coda_batches


def test_min_coda_batch_size_rejects_nonpositive_values() -> None:
    model = TokenIncrementLoopModel(_config())
    with pytest.raises(ValueError, match="min_coda_batch_size"):
        ContinuousDepthBatchingEngine.from_model(
            model,
            _cdb_config(min_coda_batch_size=0),
            dtype=torch.float32,
            model_adapter=TokenIncrementLoopAdapter(model),
        )


def test_stage_attention_kwargs_pads_bucket_rows_with_fast_path_convention() -> None:
    model = TokenIncrementLoopModel(_config())
    engine = ContinuousDepthBatchingEngine.from_model(
        model,
        _cdb_config(max_num_seqs=4),
        dtype=torch.float32,
        model_adapter=TokenIncrementLoopAdapter(model),
    )
    runner = engine.runner
    runner.model_adapter_stages_use_attention = True
    items = []
    for idx in range(3):
        state = RequestState(request_id=f"r{idx}", initial_tokens=[idx + 1])
        state.status = RequestStatus.DECODING
        state.tokens_to_process = [idx + 1]
        item = engine._stage_decode_token(state)
        assert item is not None
        items.append(item)

    kwargs = runner.stage_attention_kwargs(items, stage="coda", include_position_ids=True, bucket_size=4)

    buffers = runner.stage_attention_buffers[("coda", 4)]
    assert kwargs["cu_seq_lens_q"] is buffers.cu_seq_lens_q
    assert kwargs["position_ids"].shape == (1, 4)
    assert buffers.host_cu_seq_lens_q.tolist() == [0, 1, 2, 3, 3]
    expected_k = [0]
    for item in items:
        expected_k.append(expected_k[-1] + item.token_position + 1)
    assert buffers.host_cu_seq_lens_k.tolist() == [*expected_k, expected_k[-1]]
    assert buffers.host_block_table[0, 3].eq(-1).all()
    assert buffers.host_position_ids[0].tolist() == [item.token_position for item in items] + [0]

    # The stage bucket mirrors the recurrent stage's batch buckets, capped at the
    # decode launch width; a wider batch has no bucket in either stage.
    runner.use_stage_cuda_graphs = True
    assert runner._stage_bucket_size(3) == 4
    assert runner._stage_bucket_size(4) == 4
    with pytest.raises(ValueError, match="exceed the widest bucket"):
        runner._stage_bucket_size(5)
    runner.use_stage_cuda_graphs = False
    assert runner._stage_bucket_size(3) is None


def test_delayed_gate_consumption_exits_one_recurrent_step_later() -> None:
    model = TokenIncrementLoopModel(_config())
    engine = ContinuousDepthBatchingEngine.from_model(
        model,
        _cdb_config(exit_threshold=0.5, use_async_batching=True),
        dtype=torch.float32,
        model_adapter=TokenIncrementLoopAdapter(model),
    )
    state = RequestState(request_id="r0", initial_tokens=[1])
    state.status = RequestStatus.DECODING
    state.tokens_to_process = [1]
    item = engine._stage_decode_token(state)
    assert item is not None
    engine.runner.compute_prelude_batch([item])

    engine._run_mixed_recurrent_batch([item], recurrent_kwargs={})

    assert len(engine.scheduler.ready_queue) == 1
    delayed_item = engine.scheduler.ready_queue.popleft()
    assert delayed_item.recurrent_step == 1
    assert delayed_item.pending_gate is not None
    assert delayed_item.pending_gate.recurrent_step == 0
    assert len(engine.scheduler.coda_queue) == 0

    engine._run_mixed_recurrent_batch([delayed_item], recurrent_kwargs={})

    assert len(engine.scheduler.ready_queue) == 0
    assert len(engine.scheduler.coda_queue) == 1
    assert delayed_item.pending_gate is None
    assert state.exit_depths == [1]
    assert engine.last_stats.exit_depth_histogram == {1: 1}
    assert engine.model.seen_recurrent_steps == [[0], [1]]


@pytest.mark.parametrize("refill", [True, False])
@pytest.mark.parametrize(
    ("horizon", "use_async_batching", "delay_gate_consumption", "expected_depth"),
    [
        (DecisionHorizon.SAME_STEP, False, False, 0),
        (DecisionHorizon.OFFSET_ONE, True, True, 1),
    ],
)
def test_explicit_decision_horizon_controls_exact_exit_depth(
    refill: bool,
    horizon: DecisionHorizon,
    use_async_batching: bool,
    delay_gate_consumption: bool,
    expected_depth: int,
) -> None:
    model = TokenIncrementLoopModel(_config())
    adapter = TokenIncrementLoopAdapter(
        model,
        exit_policy_spec=ExitPolicySpec(ExitDecisionRule.CUMULATIVE_HAZARD, horizon),
    )
    engine = ContinuousDepthBatchingEngine.from_model(
        model,
        _cdb_config(
            exit_threshold=0.5,
            refill=refill,
            use_async_batching=use_async_batching,
            delay_gate_consumption=delay_gate_consumption,
        ),
        dtype=torch.float32,
        model_adapter=adapter,
    )

    engine.generate_batch(input_ids=[[1]], max_new_tokens=2, eos_token_id=None, warmup=False)

    assert engine.last_stats.exit_depth_histogram == {expected_depth: 1}
    assert engine.model.seen_recurrent_steps == [[step] for step in range(expected_depth + 1)]


def test_offset_one_consumes_step_n_signal_only_after_step_n_plus_one_launches() -> None:
    class OverlapEvent:
        def __init__(self, model: TokenIncrementLoopModel, source_step: int) -> None:
            self.model = model
            self.source_step = source_step

        def synchronize(self) -> None:
            assert self.model.seen_recurrent_steps[-1] == [self.source_step + 1]

    model = TokenIncrementLoopModel(_config())
    adapter = TokenIncrementLoopAdapter(
        model,
        exit_policy_spec=ExitPolicySpec(
            ExitDecisionRule.CUMULATIVE_HAZARD,
            DecisionHorizon.OFFSET_ONE,
        ),
    )
    engine = ContinuousDepthBatchingEngine.from_model(
        model,
        _cdb_config(exit_threshold=0.5, use_async_batching=True),
        dtype=torch.float32,
        model_adapter=adapter,
    )
    original_stage = engine.runner.stage_delayed_exit_signals

    def stage_with_order_assertion(
        items: list[DepthWorkItem],
        exit_signals: torch.Tensor,
    ) -> list[PendingGateResult]:
        pending = original_stage(items, exit_signals)
        for result in pending:
            result.ready_event = OverlapEvent(model, result.recurrent_step)
        return pending

    engine.runner.stage_delayed_exit_signals = stage_with_order_assertion  # type: ignore[method-assign]

    engine.generate_batch(input_ids=[[1]], max_new_tokens=2, eos_token_id=None, warmup=False)

    assert engine.last_stats.exit_depth_histogram == {1: 1}
    assert model.seen_recurrent_steps == [[0], [1]]


def test_delayed_gate_consumption_waits_only_on_gate_event() -> None:
    class FakeEvent:
        def __init__(self) -> None:
            self.sync_count = 0

        def synchronize(self) -> None:
            self.sync_count += 1

    class ForbiddenComputeStream:
        def synchronize(self) -> None:
            raise AssertionError("delayed gate consumption must not synchronize the compute stream")

    model = TokenIncrementLoopModel(_config())
    engine = ContinuousDepthBatchingEngine.from_model(
        model,
        _cdb_config(exit_threshold=0.5),
        dtype=torch.float32,
        model_adapter=TokenIncrementLoopAdapter(model),
    )
    engine.inputs_and_outputs.compute_stream = ForbiddenComputeStream()  # type: ignore[assignment]
    state = RequestState(request_id="r0", initial_tokens=[1])
    item = DepthWorkItem(state=state, token_id=1, token_position=0, recurrent_step=1)
    event = FakeEvent()
    item.pending_gate = PendingGateResult(
        host_logits=torch.tensor([10.0]),
        batch_index=0,
        recurrent_step=0,
        ready_event=event,
    )

    assert engine._consume_delayed_gate(item)
    assert event.sync_count == 1
    assert item.pending_gate is None


def test_exit_threshold_accumulates_before_min_recurrent_steps() -> None:
    model = TokenIncrementLoopModel(_config())
    engine = ContinuousDepthBatchingEngine.from_model(
        model,
        _cdb_config(exit_threshold=0.5, min_recurrent_steps=2),
        dtype=torch.float32,
        model_adapter=TokenIncrementLoopAdapter(model),
    )
    item = DepthWorkItem(
        state=RequestState(request_id="request-0", initial_tokens=[1]),
        token_id=1,
        token_position=0,
        recurrent_step=0,
    )
    exit_signals = torch.tensor([[[10.0]]])

    assert not engine._should_exit(item, recurrent_step=0, exit_signals=exit_signals, batch_index=0)
    assert item.policy_state.exit_cdf > 0.5
    assert engine._should_exit(item, recurrent_step=1, exit_signals=exit_signals, batch_index=0)


def test_cdb_cache_gathers_and_copies_token_kv_rows_across_slot_layers() -> None:
    cache = _cache(kv_policy="last_exited")  # 1 physical layer x 3 recurrent slots
    cache.allocate_blocks(2, "req", allocated_blocks=0)

    rows = cache.gather_token_kv_rows([("req", 0), ("req", 5)])
    assert rows.tolist() == [cache.cache_allocator.get_write_indices("req", pos, 1)[0] for pos in (0, 5)]

    for layer_idx in range(cache.num_layers):
        cache.key_cache[layer_idx].fill_(float(layer_idx))
        cache.value_cache[layer_idx].fill_(float(-layer_idx))
    cache.key_cache[0][rows] = 42.0
    cache.value_cache[0][rows] = -42.0
    cache.copy_kv_rows(0, [1, 2], rows)

    # The copied rows carry the source layer's values into every target; all other
    # rows of each target layer keep theirs.
    untouched = torch.ones(cache.cache_shape[0], dtype=torch.bool)
    untouched[rows] = False
    for target_layer_idx in (1, 2):
        assert torch.all(cache.key_cache[target_layer_idx][rows] == 42.0)
        assert torch.all(cache.value_cache[target_layer_idx][rows] == -42.0)
        assert torch.all(cache.key_cache[target_layer_idx][untouched] == float(target_layer_idx))


def _recorded_kv_copies(engine: ContinuousDepthBatchingEngine) -> list[tuple[int, int, list[int]]]:
    """Record every exit-KV copy the engine issues while still performing it."""

    copies: list[tuple[int, int, list[int]]] = []
    original = engine.cache.copy_kv_rows

    def recording(source_layer_idx: int, target_layer_idxs: list[int], rows: torch.Tensor) -> None:
        for target_layer_idx in target_layer_idxs:
            copies.append((source_layer_idx, target_layer_idx, rows.tolist()))
        original(source_layer_idx, target_layer_idxs, rows)

    engine.cache.copy_kv_rows = recording  # type: ignore[method-assign]
    return copies


@pytest.mark.parametrize("refill", [True, False])
@pytest.mark.parametrize("force_delayed", [False, True])
def test_last_exited_policy_copies_exit_kv_into_deeper_slots(refill: bool, force_delayed: bool) -> None:
    # Every early exit must propagate the exit step's KV rows into all deeper slots,
    # through both serving loops, and on the delayed path the copy source must follow
    # the one-step-late effective exit. With one physical layer, cache layer == slot.
    prompts = [[1, 2], [5]]
    exit_depths = [[0, 2], [1, 0]]

    model = TokenIncrementLoopModel(_config())
    engine = ContinuousDepthBatchingEngine.from_model(
        model,
        _cdb_config(synthetic_exit_replay=True, refill=refill, kv_policy="last_exited", kv_pressure_mode="none"),
        dtype=torch.float32,
        model_adapter=TokenIncrementLoopAdapter(model),
    )
    if force_delayed:
        engine._uses_delayed_synthetic_exit = lambda: True  # type: ignore[method-assign]
    copies = _recorded_kv_copies(engine)

    outputs = engine.generate_batch(
        input_ids=prompts,
        max_new_tokens=3,
        eos_token_id=None,
        warmup=False,
        exit_depths=exit_depths,
    )

    copied_rows_per_pair = Counter[tuple[int, int]]()
    for source_layer_idx, target_layer_idx, row_list in copies:
        copied_rows_per_pair[(source_layer_idx, target_layer_idx)] += len(row_list)
    if force_delayed:
        # Exits apply one step late: replayed depths [0, 2, 1, 0] execute as [1, 2, 2, 1].
        assert copied_rows_per_pair == {(1, 2): 2}
    else:
        # Depth-0 exits fill slots 1 and 2, the depth-1 exit fills slot 2, depth 2 fills nothing.
        assert copied_rows_per_pair == {(0, 1): 2, (0, 2): 2, (1, 2): 1}

    # KV routing must not change scheduling or sampling: the toy model reads no KV, so the
    # run matches the same replay under the default single-slot policy token for token.
    _, baseline_tokens = _run_replay(refill, prompts, exit_depths, max_new_tokens=3, force_delayed=force_delayed)
    assert [output.generated_tokens for output in outputs] == baseline_tokens
    assert engine.cache.get_num_free_blocks() == engine.cache.num_blocks


def test_cdb_cache_uses_one_shared_kv_slot_per_physical_layer() -> None:
    model_config = _config()
    cache = PagedAttentionCache(
        config=model_config,
        continuous_batching_config=_cdb_config(),
        device="cpu",
        dtype=torch.float32,
    )

    assert cache.kv_policy == "single"
    assert len(cache.key_cache) == model_config.num_hidden_layers
    assert len(cache.value_cache) == model_config.num_hidden_layers
    assert cache.num_layers == model_config.num_hidden_layers


def test_cdb_cache_uses_configured_kv_slot_budget() -> None:
    model_config = _config(num_hidden_layers=2)
    cache = PagedAttentionCache(
        config=model_config,
        continuous_batching_config=_cdb_config(
            kv_policy="first_then_shared",
            kv_slots_per_layer=2,
        ),
        device="cpu",
        dtype=torch.float32,
    )

    assert cache.kv_policy == "first_then_shared"
    assert cache.kv_slots_per_layer == 2
    assert len(cache.key_cache) == 4
    assert cache.num_layers == 4
    assert cache.kv_slot_for_recurrent_step(0) == 0
    assert cache.kv_slot_for_recurrent_step(3) == 1


def test_cdb_scheduler_keeps_multi_slot_recurrent_batches_slot_homogeneous() -> None:
    cache = PagedAttentionCache(
        config=_config(),
        continuous_batching_config=_cdb_config(
            kv_policy="first_then_shared",
            kv_slots_per_layer=2,
        ),
        device="cpu",
        dtype=torch.float32,
    )
    scheduler = CDBScheduler(cache=cache, max_recurrent_steps=4, max_num_seqs=4)
    state = RequestState(request_id="request-0", initial_tokens=[1])
    for recurrent_step in [0, 1, 2, 0]:
        scheduler.enqueue_depth(
            DepthWorkItem(
                state=state,
                token_id=1,
                token_position=0,
                recurrent_step=recurrent_step,
            )
        )

    batch = scheduler._schedule_recurrent_batch()

    assert [item.recurrent_step for item in batch] == [0, 0]
    assert {cache.kv_slot_for_recurrent_step(item.recurrent_step) for item in batch} == {0}
    assert [item.recurrent_step for item in scheduler.ready_queue] == [1, 2]


def test_cdb_cache_reuses_cached_request_block_table_after_growth() -> None:
    cache = PagedAttentionCache(
        config=_config(),
        continuous_batching_config=_cdb_config(num_blocks=4, block_size=2, max_model_len=8),
        device="cpu",
        dtype=torch.float32,
    )
    cache.allocate_blocks(1, "request-0", allocated_blocks=0)
    cached_row = cache._request_block_tables["request-0"]

    first_table = torch.full((4,), -1, dtype=torch.int32)
    cache.fill_block_table("request-0", past_length=0, query_length=1, block_table=first_table)

    cache.allocate_blocks(1, "request-0", allocated_blocks=1)
    second_table = torch.full((4,), -1, dtype=torch.int32)
    cache.fill_block_table("request-0", past_length=2, query_length=1, block_table=second_table)

    assert cache._request_block_tables["request-0"] is cached_row
    assert first_table.tolist() == [0, -1, -1, -1]
    assert second_table.tolist() == [0, 1, -1, -1]


# --- KV-cache-pressure modes (reserve admission + self-preemption) --------------------------------


def _pressure_cache(num_blocks: int = 8, block_size: int = 4, max_model_len: int = 16) -> PagedAttentionCache:
    return PagedAttentionCache(
        config=_config(),
        continuous_batching_config=_cdb_config(
            num_blocks=num_blocks, block_size=block_size, max_model_len=max_model_len
        ),
        device="cpu",
        dtype=torch.float32,
    )


def _reserve_scheduler() -> CDBScheduler:
    cache = _pressure_cache(num_blocks=8, block_size=4, max_model_len=16)
    return CDBScheduler(
        cache=cache,
        max_recurrent_steps=3,
        max_num_seqs=2,
        safety_margin=0.0,
        kv_pressure_mode="reserve",
        max_model_len=16,
    )


def test_cdb_reserve_admits_only_reservation_fitting_prefix() -> None:
    scheduler = _reserve_scheduler()
    # Peaks over max_model_len=16, block_size=4: a -> ceil(8/4)=2, b -> ceil(12/4)=3, c -> ceil(16/4)=4.
    a = RequestState(request_id="a", initial_tokens=[1, 2, 3, 4], max_new_tokens=4)
    b = RequestState(request_id="b", initial_tokens=[1, 2, 3, 4], max_new_tokens=8)
    c = RequestState(request_id="c", initial_tokens=list(range(8)), max_new_tokens=8)
    for state in (a, b, c):
        scheduler.add_waiting_request(state)

    # 2 + 3 = 5 blocks fit; c (+4 -> 9) overflows num_blocks=8 and stops admission at the head of line.
    assert [state.request_id for state in scheduler.get_prefill_candidates()] == ["a", "b"]


def test_cdb_reserve_capacity_check_rejects_cache_too_small_for_a_max_length_request() -> None:
    cache = _pressure_cache(num_blocks=3, block_size=4, max_model_len=64)  # needs ceil(64/4)=16 blocks
    with pytest.raises(ValueError, match="cannot serve a max-length request"):
        CDBScheduler(
            cache=cache,
            max_recurrent_steps=3,
            max_num_seqs=2,
            kv_pressure_mode="reserve",
            max_model_len=64,
        )


def test_cdb_block_new_requests_stops_admitting_waiting_prompts() -> None:
    scheduler = CDBScheduler(cache=_pressure_cache(), max_recurrent_steps=3, max_num_seqs=2, safety_margin=0.0)
    scheduler.add_waiting_request(RequestState(request_id="w", initial_tokens=[1, 2]))

    assert [state.request_id for state in scheduler.get_prefill_candidates()] == ["w"]

    scheduler.block_new_requests = True
    assert scheduler.get_prefill_candidates() == []


def test_cdb_admit_offloaded_restores_reallocates_and_excludes_them_from_prefill() -> None:
    scheduler = CDBScheduler(
        cache=_pressure_cache(), max_recurrent_steps=3, max_num_seqs=2, safety_margin=0.0, kv_pressure_mode="offload"
    )
    off = RequestState(request_id="off", initial_tokens=[1, 2, 3])
    off.is_cpu_offloaded = True
    off._status = RequestStatus.DECODING
    off.tokens_to_process = [9]
    off.position_offset = 3
    fresh = RequestState(request_id="fresh", initial_tokens=[1, 2])
    scheduler.add_waiting_request(off)
    scheduler.add_waiting_request(fresh)

    # Offloaded requests resume via the decode path, so they are not prefill candidates.
    assert [state.request_id for state in scheduler.get_prefill_candidates()] == ["fresh"]

    admitted = scheduler.admit_offloaded_restores()

    assert [state.request_id for state in admitted] == ["off"]
    assert scheduler.active_requests["off"].allocated_blocks > 0
    assert "off" not in scheduler.waiting_requests
    assert "fresh" in scheduler.waiting_requests


def test_cdb_admit_offloaded_restores_blocked_while_draining() -> None:
    scheduler = CDBScheduler(
        cache=_pressure_cache(), max_recurrent_steps=3, max_num_seqs=2, safety_margin=0.0, kv_pressure_mode="offload"
    )
    off = RequestState(request_id="off", initial_tokens=[1])
    off.is_cpu_offloaded = True
    off._status = RequestStatus.DECODING
    scheduler.add_waiting_request(off)
    scheduler.block_new_requests = True

    assert scheduler.admit_offloaded_restores() == []
    assert "off" in scheduler.waiting_requests


def test_cdb_preempts_the_least_computed_stalled_decoder() -> None:
    """CDB preempts through its stalled set, and picks by the same cost rule the full-depth engine uses.

    The engine never calls the inherited ``pop_request_to_evict``: only a stalled request is at a clean
    token boundary rather than mid-recurrence, so the victim is drawn from ``stalled_decoders``. The
    victim set differs from the full-depth engine's; the rule choosing within it does not.
    """

    model = TokenIncrementLoopModel(_config())
    engine = ContinuousDepthBatchingEngine.from_model(
        model,
        _cdb_config(kv_pressure_mode="recompute"),
        dtype=torch.float32,
        model_adapter=TokenIncrementLoopAdapter(model),
    )

    for req_id, computed in [("veteran", 20), ("rookie", 2), ("middling", 9)]:
        state = RequestState(request_id=req_id, initial_tokens=[1], max_new_tokens=64)
        state.status = RequestStatus.DECODING
        state.remaining_prefill_tokens = []
        state.tokens_to_process = [1]
        state.position_offset = computed
        engine.scheduler.active_requests[req_id] = state
        engine.stalled_decoders[req_id] = state

    engine._preempt_stalled_decoder()

    assert "rookie" not in engine.stalled_decoders
    assert engine.offloading_manager.num_preemptions == 1
    assert sorted(engine.stalled_decoders) == ["middling", "veteran"]


def _cdb_replay_tokens(
    prompts: list[list[int]],
    exit_depths: list[list[int]],
    max_new_tokens: int,
    force_delayed: bool = False,
    **config_overrides: Any,
) -> tuple[ContinuousDepthBatchingEngine, list[list[int]]]:
    model = TokenIncrementLoopModel(_config())
    engine = ContinuousDepthBatchingEngine.from_model(
        model,
        _cdb_config(synthetic_exit_replay=True, **config_overrides),
        dtype=torch.float32,
        model_adapter=TokenIncrementLoopAdapter(model),
    )
    if force_delayed:
        engine._uses_delayed_synthetic_exit = lambda: True  # type: ignore[method-assign]
    outputs = engine.generate_batch(
        input_ids=prompts,
        max_new_tokens=max_new_tokens,
        eos_token_id=None,
        warmup=False,
        exit_depths=exit_depths,
    )
    return engine, [output.generated_tokens for output in outputs]


def test_cdb_reserve_preserves_output_under_cache_pressure() -> None:
    # Reserve never preempts, so the token stream must be identical to an unconstrained run, even with
    # an early-exit schedule and a cache too small to admit all three requests at once.
    prompts = [[1, 2], [5], [3, 4]]
    exit_depths = [[0, 1], [1, 0], [2, 0]]
    _, reference = _cdb_replay_tokens(prompts, exit_depths, 3, num_blocks=64, block_size=4, max_model_len=16)

    # Peaks (block_size=4): [1,2] -> ceil(5/4)=2, [5] -> ceil(4/4)=1, [3,4] -> 2, so all three need 5
    # blocks. A 4-block cache admits at most two at once (pressure) yet fits a single max-length request.
    _, reserved = _cdb_replay_tokens(
        prompts, exit_depths, 3, num_blocks=4, block_size=4, max_model_len=16, kv_pressure_mode="reserve"
    )

    assert reserved == reference


def test_cdb_reserve_admits_a_max_length_request_at_block_size_two() -> None:
    # block_size=2 is the CDB CPU-fixture block size and the case where a rounding-up decode allocator
    # grows a request's blocks in steps of two and overshoots its reservation. The decode block is
    # allocated by the engine's _try_allocate_one_decode_token, a separate site from the scheduler's
    # prefill allocator, so it must use the same exact-ceil rule. A single request that grows to exactly
    # max_model_len tokens, on a cache sized to the reserve capacity check, must complete rather than
    # raise the "reservation accounting" over-commit error.
    # reserved_peak_blocks(prompt 1 + 7 generated = 8 tokens, block_size 2) = ceil(8/2) = 4 = num_blocks.
    _, outputs = _cdb_replay_tokens(
        [[1]],
        [[2, 2, 2, 2, 2, 2]],
        7,
        num_blocks=4,
        block_size=2,
        max_model_len=8,
        safety_margin=0.0,
        kv_pressure_mode="reserve",
    )

    assert len(outputs[0]) == 7


# Four requests that each grow to 5 blocks (prompt 2 + 8 generated = 10 tokens, block_size 2) share a
# 6-block cache: any two decoding at once over-subscribe it, while a single request still fits (so
# self-preemption can always make progress). safety_margin=0.0 admits every prompt immediately.
_OVERSUBSCRIBED = {"num_blocks": 6, "block_size": 2, "max_model_len": 12, "safety_margin": 0.0}
_OVERSUBSCRIBED_PROMPTS = [[1, 2], [3, 4], [5, 6], [7, 8]]
_OVERSUBSCRIBED_EXITS = [[2] * 7 for _ in range(4)]  # full depth (max_recurrent_steps - 1), 8 tokens each


def test_cdb_none_mode_raises_when_the_cache_is_oversubscribed() -> None:
    # With kv_pressure_mode='none' the engine does no preemption and must surface the overflow. This
    # also establishes that the config genuinely over-subscribes the cache, so the recompute test (same
    # config) must preempt to succeed.
    with pytest.raises(CacheFullError):
        _cdb_replay_tokens(
            _OVERSUBSCRIBED_PROMPTS, _OVERSUBSCRIBED_EXITS, 8, kv_pressure_mode="none", **_OVERSUBSCRIBED
        )


@pytest.mark.parametrize(
    ("refill", "force_delayed"),
    [(True, False), (False, False), (False, True)],
    ids=["refill", "no_refill_sync", "no_refill_delayed"],
)
def test_cdb_recompute_preemption_preserves_output_at_full_depth(refill: bool, force_delayed: bool) -> None:
    # The same over-subscribed cache (where 'none' raises) forces self-preemption. At full recurrent
    # depth the recomputed (full-depth) resume produces the same tokens as the unpreempted decode, so
    # recompute must reproduce the reference exactly - verifying the preemption/resume bookkeeping.
    # The delayed no-refill case also exercises the wave-boundary carry under KV pressure: eager
    # re-entries that cannot allocate fall back to consume-time staging and rejoin via the wave scan.
    _, reference = _cdb_replay_tokens(
        _OVERSUBSCRIBED_PROMPTS,
        _OVERSUBSCRIBED_EXITS,
        8,
        refill=refill,
        force_delayed=force_delayed,
        num_blocks=64,
        block_size=2,
        max_model_len=12,
    )

    engine, recomputed = _cdb_replay_tokens(
        _OVERSUBSCRIBED_PROMPTS,
        _OVERSUBSCRIBED_EXITS,
        8,
        refill=refill,
        force_delayed=force_delayed,
        kv_pressure_mode="recompute",
        **_OVERSUBSCRIBED,
    )

    assert recomputed == reference
    assert engine.cache.get_num_free_blocks() == engine.cache.num_blocks  # all blocks released at the end
    # Preemption genuinely fired (soft resets, since recompute has no CPU pool), and was counted.
    assert engine.offloading_manager.num_recompute_preemptions > 0
    assert engine.offloading_manager.num_offload_preemptions == 0


def test_recompute_resume_drops_the_pending_tokens_replayed_exit_depth() -> None:
    # At the preempt point the pending decode token is the last of ``generated_tokens`` and its recorded
    # exit depth (at the current cursor) has not been consumed. Recompute recomputes that token at full
    # depth in the prefill, so its depth is spent; the resumed decode must begin at the *next* recorded
    # depth. Slicing from the cursor rather than cursor + 1 would shift every remaining depth by one and
    # make recompute replay a different schedule than offload.
    state = RequestState(
        request_id="r", initial_tokens=[1, 2], max_new_tokens=5, synthetic_exit_depths=[10, 11, 12, 13, 14]
    )
    state._status = RequestStatus.DECODING
    # Two decode tokens have been staged (each consumed one recorded depth); a third is pending - staged
    # but not yet decoded - so its depth at cursor == 2 is still unconsumed.
    for token in (101, 102):
        state.next_synthetic_exit_depth()
        state.generated_tokens.append(token)
    state.generated_tokens.append(103)  # pending token, cursor still at 2

    fresh = state.create_equivalent_initial_request()

    assert fresh.initial_tokens == [1, 2, 101, 102, 103]  # generated (incl. pending) folded onto the prompt
    assert fresh.max_new_tokens == 2  # 5 - 3 already generated
    # depths[2] (the pending token, now recomputed at full depth) is dropped; resume starts at depths[3:].
    assert fresh.synthetic_exit_depths == [13, 14]


def test_recompute_resume_keeps_the_requests_timing_identity() -> None:
    # A soft-reset request is the same request resuming, not a new arrival: its creation and
    # first-schedule times must survive the rebuild, and re-scheduling must not restamp them.
    state = RequestState(request_id="r", initial_tokens=[1, 2], max_new_tokens=5)
    state.status = RequestStatus.PREFILLING  # leaves PENDING, stamping the first-schedule time
    first_scheduled = state.lifespan[0]
    state.status = RequestStatus.DECODING
    state.generated_tokens.append(101)

    fresh = state.create_equivalent_initial_request()

    assert fresh.created_time == state.created_time
    assert fresh.lifespan[0] == first_scheduled
    fresh.status = RequestStatus.PREFILLING  # resumed schedule must not restamp the start
    assert fresh.lifespan[0] == first_scheduled


def test_offload_requeue_keeps_the_pending_tokens_replayed_exit_depth() -> None:
    # Offload keeps the pending decode token and resumes it through the recurrent path, so unlike
    # recompute it must NOT drop the pending token's recorded depth: the cursor is left untouched, so the
    # restored decode replays depths[cursor] for that token. This is the counterpart that keeps offload
    # and recompute aligned on the tokens both actually decode recurrently.
    state = RequestState(
        request_id="r", initial_tokens=[1, 2], max_new_tokens=5, synthetic_exit_depths=[10, 11, 12, 13, 14]
    )
    state._status = RequestStatus.DECODING
    for token in (101, 102):
        state.next_synthetic_exit_depth()
        state.generated_tokens.append(token)
    state.generated_tokens.append(103)
    state.tokens_to_process = [103]

    state.prepare_for_offload_requeue()

    # The pending token (103) resumes at its own recorded depth 12, then 13, 14 - no shift, nothing dropped.
    assert state.next_synthetic_exit_depth() == 12
    assert state.next_synthetic_exit_depth() == 13


# A varied (non-full) exit schedule: recompute would recompute the folded tokens at full depth and so
# could not reproduce this, but offload restores exact KV and replays the recorded depths verbatim, so
# a one-off replay misalignment through preempt/restore would corrupt the output and fail the check.
_OVERSUBSCRIBED_EXITS_VARIED = [[0, 1, 2, 1, 0, 2, 1] for _ in range(4)]


@pytest.mark.parametrize("kv_policy", ["single", "last_exited"])
def test_cdb_offload_preemption_preserves_output(kv_policy: str) -> None:
    # Offload swaps the victim's KV to the (unpinned, CPU) pool and copies it back on restore, so it
    # reproduces the unpreempted reference even at partial exit depths, and the swap counters fire.
    # The pool mirrors every cache layer, so under last_exited the copy-on-exit routed rows must
    # survive the same round trip (the toy model reads no KV, so tokens match across policies).
    _, reference = _cdb_replay_tokens(
        _OVERSUBSCRIBED_PROMPTS, _OVERSUBSCRIBED_EXITS_VARIED, 8, num_blocks=64, block_size=2, max_model_len=12
    )

    engine, offloaded = _cdb_replay_tokens(
        _OVERSUBSCRIBED_PROMPTS,
        _OVERSUBSCRIBED_EXITS_VARIED,
        8,
        kv_policy=kv_policy,
        kv_pressure_mode="offload",
        cpu_offload_space=0.001,
        **_OVERSUBSCRIBED,
    )

    assert offloaded == reference
    assert engine.cache.get_num_free_blocks() == engine.cache.num_blocks  # all blocks released at the end
    assert engine.offloading_manager.num_offload_preemptions > 0
    assert engine.offloading_manager.num_restores > 0
    assert engine.offloading_manager.num_recompute_preemptions == 0  # pool is ample; no soft-reset fallback


def test_last_exited_full_offload_pool_raises_instead_of_recompute_fallback() -> None:
    # When the pool cannot hold a victim, offload normally falls back to a full-depth soft reset.
    # Under copy-on-exit routing that fallback would silently replace exit-step KV with full-depth
    # KV mid-run, so it must be a hard error instead.
    with pytest.raises(RuntimeError, match="soft-reset fallback is disabled"):
        _cdb_replay_tokens(
            _OVERSUBSCRIBED_PROMPTS,
            _OVERSUBSCRIBED_EXITS_VARIED,
            8,
            kv_policy="last_exited",
            kv_pressure_mode="offload",
            cpu_offload_space=1e-9,
            **_OVERSUBSCRIBED,
        )


def test_prefill_sampled_row_gather_matches_full_logits_outputs() -> None:
    # CDB's monolithic prefill restricts its LM head to the sampled rows when the model supports
    # logits_to_keep; tokens must match the full-logits path exactly (decode goes through the
    # adapter's coda and is unaffected).
    def run(model_cls: type) -> tuple[ContinuousDepthBatchingEngine, list[list[int]]]:
        model = model_cls(_config())
        engine = ContinuousDepthBatchingEngine.from_model(
            model,
            _cdb_config(num_blocks=32, max_num_batched_tokens=4, max_model_len=32),
            dtype=torch.float32,
            model_adapter=TokenIncrementLoopAdapter(model),
        )
        outputs = engine.generate_batch(
            input_ids=[[1, 2, 3, 4, 5, 6], [2, 5]], max_new_tokens=3, eos_token_id=None, warmup=False
        )
        return engine, [output.generated_tokens for output in outputs]

    full_engine, full_tokens = run(TokenIncrementLoopModel)
    gather_engine, gather_tokens = run(GatherAwareTokenIncrementLoopModel)

    assert not full_engine.runner.gather_sampled_rows
    assert gather_engine.runner.gather_sampled_rows
    assert gather_tokens == full_tokens


def test_cdb_prefill_never_carries_sampled_tokens_between_batches() -> None:
    """A soft-reset re-prefill shares its request id with the prefill batch that sampled for it.

    The depth engine's decode never flows through this pipeline, so a carried-over sampled
    token would overwrite part of the re-prefill's folded prompt. The CB pipeline carries
    over by design (its decode inputs hold placeholders); the CDB subclass must not.
    """

    from looped_cdb.continuous_batching.input_outputs import ContinuousBatchingIOs
    from looped_cdb.continuous_depth_batching.input_outputs import ContinuousDepthBatchingIOs
    from looped_cdb.continuous_depth_batching.requests import FutureRequestState

    def carry_row_after_reprefill(ios_cls: type) -> list[int]:
        cache = _cache(num_blocks=300, block_size=4, max_num_batched_tokens=16, max_model_len=64)
        ios = ios_cls(
            cache=cache,
            config=_config(),
            device=torch.device("cpu"),
            model_dtype=torch.float32,
        )
        cache.allocate_blocks(2, "req", allocated_blocks=0)
        # Original prefill, then the soft-reset re-prefill with the sampled token folded on.
        for tokens in ([10, 11, 12], [10, 11, 12, 13]):
            state = RequestState(request_id="req", initial_tokens=tokens)
            state.tokens_to_process = list(tokens)
            future = FutureRequestState(state=state, has_new_token=True, query_length=len(tokens))
            ios.prepare_batch_tensors([future], use_decode_fast_path=False, num_q_tokens=len(tokens), max_kv_read=0)
        return ios.host_buffers.carry_over_ids.tolist()

    assert any(row >= 0 for row in carry_row_after_reprefill(ContinuousBatchingIOs))
    assert all(row == -1 for row in carry_row_after_reprefill(ContinuousDepthBatchingIOs))


class PreloopExitAdapter(TokenIncrementLoopAdapter):
    """Fake adapter whose exit depth is fixed before the loop, as a pre-loop gate's is."""

    def __init__(self, model: TokenIncrementLoopModel, exit_step: int, num_depths: int) -> None:
        super().__init__(model)
        self.exit_step = exit_step
        self.num_depths = num_depths

    def decides_exit_before_loop(self) -> bool:
        return True

    def preloop_exit_pdf(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # All mass on one depth, so the cumulative distribution crosses any threshold there.
        rows = hidden_states.reshape(-1, hidden_states.size(-1)).size(0)
        pdf = torch.zeros(rows, self.num_depths, device=hidden_states.device, dtype=torch.float32)
        pdf[:, self.exit_step] = 1.0
        return pdf


def test_engine_exits_decode_tokens_where_the_preloop_gate_decided() -> None:
    # The gate fixes every token's depth at prelude time, so decode must stop at that step
    # rather than run the budget, and it must do so identically on both scheduling paths.
    def run(use_async_batching: bool) -> ContinuousDepthBatchingEngine:
        model = TokenIncrementLoopModel(_config())
        engine = ContinuousDepthBatchingEngine.from_model(
            model,
            _cdb_config(
                max_recurrent_steps=3,
                exit_threshold=0.5,
                use_async_batching=use_async_batching,
            ),
            dtype=torch.float32,
            model_adapter=PreloopExitAdapter(model, exit_step=1, num_depths=3),
        )
        engine.generate_batch(
            input_ids=[[1, 2], [5]],
            max_new_tokens=3,
            eos_token_id=None,
            warmup=False,
        )
        return engine

    sync_engine = run(False)
    async_engine = run(True)

    # Every decode token stopped at the decided step, none at the budget's last step.
    assert set(sync_engine.last_stats.exit_depth_histogram) == {1}
    assert sync_engine.last_stats.exit_depth_histogram == async_engine.last_stats.exit_depth_histogram
    decode_tokens = sum(sync_engine.last_stats.exit_depth_histogram.values())
    # Two recurrent steps per decode token instead of the budget's three.
    assert sync_engine.last_stats.recurrent_steps == 2 * decode_tokens
    assert sync_engine.cache.get_num_free_blocks() == sync_engine.cache.num_blocks


def test_preloop_exit_depth_tracks_the_gate_rather_than_the_budget() -> None:
    # A deeper decision costs a deeper loop, which is what makes the depth adaptive at all.
    histograms = []
    for exit_step in (0, 2):
        model = TokenIncrementLoopModel(_config())
        engine = ContinuousDepthBatchingEngine.from_model(
            model,
            _cdb_config(max_recurrent_steps=3, exit_threshold=0.5),
            dtype=torch.float32,
            model_adapter=PreloopExitAdapter(model, exit_step=exit_step, num_depths=3),
        )
        engine.generate_batch(input_ids=[[1, 2]], max_new_tokens=2, eos_token_id=None, warmup=False)
        histograms.append(engine.last_stats.exit_depth_histogram)

    assert set(histograms[0]) == {0}
    assert set(histograms[1]) == {2}
