"""GPU verification for the CDB refill / no-refill paths on a real Ouro model.

These run the staged continuous-depth engine end to end on an H100 with the real
paged-attention Ouro forward, CUDA graphs, and the async double-buffered IO that
the CPU tests can only fake. They pin the invariants the refill-vs-no-refill and
gate-readout work relies on: refill changes only scheduling (identical tokens),
the synthetic-replay gate readout is token-neutral, and no-refill never refills a
freed depth slot. Run on a GPU node via ``shells/pytest_cuda.sh`` with
``LOOPED_CDB_RUN_CUDA_TESTS=1``.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from looped_cdb.continuous_depth_batching import ContinuousDepthBatchingConfig, ContinuousDepthBatchingEngine
from looped_cdb.continuous_depth_batching.model_runner import PendingCodaResult
from looped_cdb.continuous_depth_batching.requests import TMP_TOKEN_ID, DepthWorkItem, RequestState
from looped_cdb.models.ouro import OuroConfig, OuroForCausalLM
from looped_cdb.models.ouro.kv_cache_policy import configure_kv_cache_policy

pytestmark = pytest.mark.cuda

MAX_RECURRENT_STEPS = 4
# Prompts (each starting with BOS) and a per-request 0-based exit-depth schedule with
# ``max_new_tokens - 1`` entries (the first output token is served full depth at prefill).
PROMPTS = [[1, 5, 7], [1, 9], [1, 3, 4, 6]]
EXIT_DEPTHS = [[0, 2, 1], [1, 0, 3], [2, 1, 0]]
MAX_NEW_TOKENS = 4
# A workload with more requests than the capped launch width, so no-refill waves mix carried
# successors with freshly scanned decoders in one prelude launch.
FUSED_PROMPTS = [[1, 5, 7], [1, 9], [1, 3, 4, 6], [1, 8, 2], [1, 6], [1, 4, 5]]
FUSED_EXIT_DEPTHS = [[0, 2, 1, 3], [1, 0, 3, 2], [2, 1, 0, 1], [3, 2, 1, 0], [0, 1, 2, 3], [1, 3, 0, 2]]


@pytest.fixture(scope="module")
def ouro_model() -> OuroForCausalLM:
    """A small real Ouro causal LM on CUDA with paged FlashAttention-3-style serving."""

    config = OuroConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=512,
        total_ut_steps=MAX_RECURRENT_STEPS,
        layer_types=["full_attention", "full_attention"],
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        exit_gate_type="early_exit",
    )
    torch.manual_seed(0)
    model = OuroForCausalLM(config).to("cuda", dtype=torch.bfloat16).eval()
    model.set_attn_implementation("paged|flash_attention_3")
    configure_kv_cache_policy(model, kv_policy="single")
    return model


def _cdb_engine(
    model: OuroForCausalLM,
    *,
    refill: bool,
    sync: bool,
    use_cuda_graph: bool,
    # The engine's own default, which for these three- to six-request workloads never binds; the
    # tests that need the cap to bind pass their own.
    max_num_seqs: int = 256,
    max_model_len: int = 8192,
    kv_policy: str = "single",
    kv_pressure_mode: str = "recompute",
    num_blocks: int = 256,
    use_cuda_graph_prefill: bool = True,
) -> ContinuousDepthBatchingEngine:
    config = ContinuousDepthBatchingConfig(
        num_blocks=num_blocks,
        max_num_batched_tokens=256,
        block_size=256,
        max_model_len=max_model_len,
        use_async_batching=not sync,
        use_cuda_graph=use_cuda_graph,
        use_cuda_graph_prefill=use_cuda_graph_prefill,
        max_num_seqs=max_num_seqs,
        max_recurrent_steps=MAX_RECURRENT_STEPS,
        min_recurrent_steps=1,
        delay_gate_consumption=True,
        kv_policy=kv_policy,
        kv_slots_per_layer=None,
        synthetic_exit_replay=True,
        refill=refill,
        kv_pressure_mode=kv_pressure_mode,
    )
    return ContinuousDepthBatchingEngine.from_model(model, cdb_config=config, dtype=torch.bfloat16)


def _run(
    engine: ContinuousDepthBatchingEngine,
    *,
    exit_depths: list[list[int]] | None = None,
    gate_on: bool = True,
    prompts: list[list[int]] | None = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> list[list[int]]:
    outputs = engine.generate_batch(
        input_ids=PROMPTS if prompts is None else prompts,
        max_new_tokens=max_new_tokens,
        eos_token_id=None,
        warmup=True,
        exit_depths=exit_depths,
        model_kwargs={"use_early_exit_gate": gate_on},
    )
    return [list(output.generated_tokens) for output in outputs]


def test_no_refill_matches_refill_on_gpu_with_real_graphs(ouro_model: OuroForCausalLM) -> None:
    # Refill changes only scheduling, not per-token compute, so the async delayed path
    # must produce identical tokens under real CUDA graphs while the schedulers genuinely
    # differ: refill mixes work through the depth queue, no-refill never does.
    refill_engine = _cdb_engine(ouro_model, refill=True, sync=False, use_cuda_graph=True)
    refill_tokens = _run(refill_engine, exit_depths=EXIT_DEPTHS)
    no_refill_engine = _cdb_engine(ouro_model, refill=False, sync=False, use_cuda_graph=True)
    no_refill_tokens = _run(no_refill_engine, exit_depths=EXIT_DEPTHS)

    assert refill_engine.last_stats.scheduler_refills > 0
    assert no_refill_engine.last_stats.scheduler_refills == 0
    assert no_refill_tokens == refill_tokens
    assert no_refill_engine.last_stats.exit_depth_histogram == refill_engine.last_stats.exit_depth_histogram
    assert no_refill_engine.last_stats.recurrent_steps == refill_engine.last_stats.recurrent_steps
    assert no_refill_engine.last_stats.prelude_gathered_tokens["wave_cohort"] > 0


def test_prefill_graphs_match_eager_prefill_on_both_depth_schedulers(ouro_model: OuroForCausalLM) -> None:
    # The depth engine shares the CB prefill path: with every token bucket captured at warm-up the
    # run only replays, and the tokens equal those of eager prefill under refill and no-refill alike.
    for refill in (True, False):
        graphed = _cdb_engine(ouro_model, refill=refill, sync=False, use_cuda_graph=True, use_cuda_graph_prefill=True)
        graphed_tokens = _run(graphed, exit_depths=EXIT_DEPTHS)
        eager = _cdb_engine(ouro_model, refill=refill, sync=False, use_cuda_graph=True, use_cuda_graph_prefill=False)
        eager_tokens = _run(eager, exit_depths=EXIT_DEPTHS)

        assert graphed.runner.prefill_graph_captures == len(graphed.runner.prefill_graph_buckets)
        assert graphed.runner.prefill_graph_hits >= graphed.last_stats.prefill_batches > 0
        assert eager.runner.prefill_graph_hits == 0
        assert graphed_tokens == eager_tokens


def test_fused_prelude_matches_a_host_only_launch_under_graphs(ouro_model: OuroForCausalLM) -> None:
    # A launch whose leading rows take their ids from a staged batch's device tokens must compute
    # exactly what a launch carrying the same ids from the host computes. Here that split write
    # lands in the captured graph's static input row, which no CPU run reaches (off CUDA the stage
    # never takes the bucket path), and which a wave only exercises when it happens to mix its two
    # token sources - so the primitive is driven directly rather than through the scheduler.
    engine = _cdb_engine(ouro_model, refill=False, sync=False, use_cuda_graph=True, max_num_seqs=8)
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
    sampled = torch.tensor([0, token_ids[0], 0, token_ids[1]], dtype=torch.long, device="cuda")
    staged = PendingCodaResult(items=[], host_tokens=sampled.cpu(), device_tokens=sampled)
    fused = prelude_rows(items_for([TMP_TOKEN_ID, TMP_TOKEN_ID, *token_ids[2:]]), device_groups=[(staged, [1, 3])])

    assert torch.equal(fused, host_only)
    # Read the runner's live counters: last_stats is only filled in at the end of a generation.
    assert runner.stage_graph_hits + runner.stage_graph_captures > 0


def test_no_refill_matches_refill_below_the_launch_width(ouro_model: OuroForCausalLM) -> None:
    # A launch width under the request count caps each wave's cohort, so requests wait for a later
    # wave and the boundary carry runs against a moving cohort. Tokens still may not change.
    refill_engine = _cdb_engine(ouro_model, refill=True, sync=False, use_cuda_graph=True, max_num_seqs=4)
    refill_tokens = _run(refill_engine, exit_depths=FUSED_EXIT_DEPTHS, prompts=FUSED_PROMPTS, max_new_tokens=5)
    no_refill_engine = _cdb_engine(ouro_model, refill=False, sync=False, use_cuda_graph=True, max_num_seqs=4)
    no_refill_tokens = _run(no_refill_engine, exit_depths=FUSED_EXIT_DEPTHS, prompts=FUSED_PROMPTS, max_new_tokens=5)

    assert no_refill_tokens == refill_tokens
    assert set(no_refill_engine.last_stats.prelude_batches) == {"wave_cohort"}
    assert no_refill_engine.last_stats.stage_graph_hits + no_refill_engine.last_stats.stage_graph_captures > 0


def test_no_refill_matches_refill_on_the_synchronous_path(ouro_model: OuroForCausalLM) -> None:
    # On the synchronous path the gate readout is a blocking GPU->CPU sync after every
    # recurrent step (now triggered under synthetic replay). Refill and no-refill must
    # still agree token-for-token there.
    refill_engine = _cdb_engine(ouro_model, refill=True, sync=True, use_cuda_graph=False)
    refill_tokens = _run(refill_engine, exit_depths=EXIT_DEPTHS)
    no_refill_engine = _cdb_engine(ouro_model, refill=False, sync=True, use_cuda_graph=False)
    no_refill_tokens = _run(no_refill_engine, exit_depths=EXIT_DEPTHS)

    assert no_refill_engine.last_stats.scheduler_refills == 0
    assert no_refill_tokens == refill_tokens


def test_gate_readout_is_token_neutral_on_gpu(ouro_model: OuroForCausalLM) -> None:
    # Running the live gate to time its readout must not change which tokens are produced:
    # the exit decision comes from the recorded schedule, and the gate value is discarded.
    gate_on = _run(
        _cdb_engine(ouro_model, refill=False, sync=False, use_cuda_graph=True), exit_depths=EXIT_DEPTHS, gate_on=True
    )
    gate_off = _run(
        _cdb_engine(ouro_model, refill=False, sync=False, use_cuda_graph=True), exit_depths=EXIT_DEPTHS, gate_on=False
    )

    assert gate_on == gate_off


def test_no_refill_full_depth_gate_off_matches_refill(ouro_model: OuroForCausalLM) -> None:
    # With the gate off no token exits early, so every token runs the full recurrent depth.
    # No-refill and refill must agree, and the no-refill run must reach the maximum depth.
    refill_tokens = _run(
        _cdb_engine(ouro_model, refill=True, sync=False, use_cuda_graph=True), exit_depths=None, gate_on=False
    )
    no_refill_engine = _cdb_engine(ouro_model, refill=False, sync=False, use_cuda_graph=True)
    no_refill_tokens = _run(no_refill_engine, exit_depths=None, gate_on=False)

    assert no_refill_tokens == refill_tokens
    assert no_refill_engine.last_stats.scheduler_refills == 0
    # Every decode token exits only at the forced final step.
    assert set(no_refill_engine.last_stats.exit_depth_histogram) == {MAX_RECURRENT_STEPS - 1}


def test_no_refill_keeps_recurrent_batches_homogeneous_and_shrinking(ouro_model: OuroForCausalLM) -> None:
    # The no-refill baseline must never touch the mixed-depth machinery and must let each
    # decode cohort shrink in lockstep: the count of tokens launched at each recurrent step
    # is non-increasing, and freed depth slots are left empty rather than refilled.
    engine = _cdb_engine(ouro_model, refill=False, sync=False, use_cuda_graph=True)
    _run(engine, exit_depths=EXIT_DEPTHS)

    stats = engine.last_stats
    assert stats.scheduler_refills == 0
    assert stats.mixed_recurrent_batches == 0
    assert not stats.mixed_recurrent_depth_histogram
    counts = [stats.recurrent_step_histogram[step] for step in range(MAX_RECURRENT_STEPS)]
    assert counts == sorted(counts, reverse=True), f"cohort did not shrink monotonically: {counts}"


def _assert_targets_match_source(
    engine: ContinuousDepthBatchingEngine,
    source_layer_idx: int,
    target_layer_idxs: list[int],
    rows: torch.Tensor,
) -> None:
    for target_layer_idx in target_layer_idxs:
        assert torch.equal(
            engine.cache.key_cache[target_layer_idx][rows],
            engine.cache.key_cache[source_layer_idx][rows],
        )
        assert torch.equal(
            engine.cache.value_cache[target_layer_idx][rows],
            engine.cache.value_cache[source_layer_idx][rows],
        )


def test_last_exited_copies_exit_kv_and_matches_across_schedulers(ouro_model: OuroForCausalLM) -> None:
    # last_exited keeps the depth-indexed layout and copies each exiting token's exit-step KV
    # into its deeper slots on the real paged cache. Refill must still change only scheduling
    # (identical tokens, exits, and copy volume); every issued copy must leave the target rows
    # bit-equal to the source rows; and the copies must still be intact when generation ends,
    # so no later launch clobbered a routed row.
    def run(refill: bool) -> tuple[list[list[int]], int]:
        engine = _cdb_engine(
            ouro_model,
            refill=refill,
            sync=False,
            use_cuda_graph=True,
            kv_policy="last_exited",
            kv_pressure_mode="none",
        )
        copied_rows = 0
        recorded: list[tuple[int, list[int], torch.Tensor]] = []
        original = engine.cache.copy_kv_rows

        def verifying(source_layer_idx: int, target_layer_idxs: list[int], rows: torch.Tensor) -> None:
            nonlocal copied_rows
            original(source_layer_idx, target_layer_idxs, rows)
            copied_rows += rows.numel() * len(target_layer_idxs)
            recorded.append((source_layer_idx, list(target_layer_idxs), rows))
            _assert_targets_match_source(engine, source_layer_idx, target_layer_idxs, rows)

        engine.cache.copy_kv_rows = verifying  # type: ignore[method-assign]

        original_free = engine._free_finished_or_active

        def checking_free() -> None:
            # Re-check every recorded copy while the rows are still allocated: a copy that
            # landed at exit time can still be clobbered by a later slot write.
            while recorded:
                _assert_targets_match_source(engine, *recorded.pop())
            original_free()

        engine._free_finished_or_active = checking_free  # type: ignore[method-assign]
        tokens = _run(engine, exit_depths=EXIT_DEPTHS)
        assert set(engine.last_stats.exit_depth_histogram) - {MAX_RECURRENT_STEPS - 1}, (
            "schedule produced no early exit, so the copy path was never exercised"
        )
        assert not recorded, "generation finished without the end-of-run copy check running"
        return tokens, copied_rows

    refill_tokens, refill_copied_rows = run(True)
    no_refill_tokens, no_refill_copied_rows = run(False)

    assert refill_copied_rows > 0
    assert no_refill_copied_rows == refill_copied_rows
    assert no_refill_tokens == refill_tokens


def test_last_exited_is_inert_when_nothing_exits_early(ouro_model: OuroForCausalLM) -> None:
    # With every replay depth at the final step there is nothing to route, so last_exited must
    # collapse to depth_indexed exactly: identical tokens and zero copies. This pins that the
    # two policies share their slot arithmetic, which the copy-time bit-equality check cannot.
    full_depth_schedule = [[MAX_RECURRENT_STEPS - 1] * len(depths) for depths in EXIT_DEPTHS]

    def run(kv_policy: str) -> list[list[int]]:
        engine = _cdb_engine(
            ouro_model,
            refill=True,
            sync=False,
            use_cuda_graph=True,
            kv_policy=kv_policy,
            kv_pressure_mode="none",
        )
        copies = 0
        original = engine.cache.copy_kv_rows

        def counting(source_layer_idx: int, target_layer_idxs: list[int], rows: torch.Tensor) -> None:
            nonlocal copies
            copies += 1
            original(source_layer_idx, target_layer_idxs, rows)

        engine.cache.copy_kv_rows = counting  # type: ignore[method-assign]
        tokens = _run(engine, exit_depths=full_depth_schedule)
        assert copies == 0
        return tokens

    assert run("last_exited") == run("depth_indexed")


def test_prefill_runs_full_depth_independent_of_depth_bucket(ouro_model: OuroForCausalLM) -> None:
    # Prefill is a monolithic variable-length forward bucketed to max_num_batched_tokens, not the
    # recurrent depth bucket. Even with a tiny max_num_seqs, prompts longer than it
    # prefill in one full-depth forward (not one recurrent wave per token), and refill/no-refill
    # still agree token-for-token.
    engine = _cdb_engine(ouro_model, refill=False, sync=False, use_cuda_graph=True, max_num_seqs=2)
    no_refill_tokens = _run(engine, exit_depths=EXIT_DEPTHS)
    refill_tokens = _run(
        _cdb_engine(ouro_model, refill=True, sync=False, use_cuda_graph=True, max_num_seqs=2),
        exit_depths=EXIT_DEPTHS,
    )

    total_prompt_tokens = sum(len(prompt) for prompt in PROMPTS)
    assert len(no_refill_tokens) == len(PROMPTS)
    assert all(len(tokens) == MAX_NEW_TOKENS for tokens in no_refill_tokens)
    assert no_refill_tokens == refill_tokens
    # Every prompt token ran the full recurrent depth, in far fewer forwards than one-per-token.
    assert engine.last_stats.prefill_recurrent_steps == total_prompt_tokens * MAX_RECURRENT_STEPS
    assert engine.last_stats.prefill_batches < total_prompt_tokens


def test_prefill_is_a_single_full_depth_forward(ouro_model: OuroForCausalLM) -> None:
    # With the default depth bucket every prompt fits one variable-length batch, so the whole
    # workload prefills in a single monolithic full-depth forward rather than token-by-token.
    engine = _cdb_engine(ouro_model, refill=True, sync=False, use_cuda_graph=True)
    _run(engine, exit_depths=EXIT_DEPTHS)

    total_prompt_tokens = sum(len(prompt) for prompt in PROMPTS)
    assert engine.last_stats.prefill_batches == 1
    assert engine.last_stats.prefill_recurrent_steps == total_prompt_tokens * MAX_RECURRENT_STEPS


def test_generate_batch_rejects_prompt_with_no_room(ouro_model: OuroForCausalLM) -> None:
    # The CDB engine wires the same max_model_len admission guard as CB: a prompt with no room to
    # generate (len == max_model_len) is rejected by raising before any staging or compute. The
    # clamp behavior is covered end-to-end on the CB engine, which calls the identical helper.
    engine = _cdb_engine(ouro_model, refill=True, sync=True, use_cuda_graph=False, max_model_len=4)

    with pytest.raises(ValueError, match="max_model_len"):
        engine.generate_batch(
            input_ids=[[1, 2, 3, 4]],
            max_new_tokens=4,
            eos_token_id=None,
            warmup=False,
        )


# The staged-CDB-vs-black-box-CB cross-check lives in the Step 5 cross-check benchmark, not
# here: on an untrained model the near-uniform logits make greedy argmax flip on any bf16
# difference between the monolithic CB forward and the staged CDB launches, so bit-exact token
# equality is not a meaningful assertion. That comparison needs a trained checkpoint (peaked
# logits) or a logit-tolerance metric, plus a verified matching KV policy.


def test_warmup_covers_every_stage_bucket_a_later_run_reaches(ouro_model: OuroForCausalLM) -> None:
    # Stage graphs are captured per batch bucket, so a run that reaches a bucket for the
    # first time pays the capture inside its own measured window. Warming every bucket up front is
    # only observable when the timed run is wider than the warmup was: the single warmup prompt
    # only ever launches a batch of one, so buckets 2 and 4 come from the six-prompt run below.
    engine = _cdb_engine(ouro_model, refill=True, sync=False, use_cuda_graph=True, max_num_seqs=4)
    _run(engine, exit_depths=EXIT_DEPTHS[:1], prompts=PROMPTS[:1])

    outputs = engine.generate_batch(
        input_ids=FUSED_PROMPTS,
        max_new_tokens=5,
        eos_token_id=None,
        warmup=False,
        exit_depths=FUSED_EXIT_DEPTHS,
        model_kwargs={"use_early_exit_gate": True},
    )

    assert all(output.generated_tokens for output in outputs)
    assert engine.last_stats.stage_graph_hits > 0
    assert engine.last_stats.stage_graph_captures == 0
    assert engine.last_stats.recurrent_graph_captures == 0


def test_warmup_falls_back_when_the_cache_cannot_hold_a_bucket(ouro_model: OuroForCausalLM) -> None:
    # Warming a bucket needs a block per fake request, so a cache smaller than the launch width
    # cannot hold the widest ones. That must cost coverage, not the run: those buckets capture
    # inside the run exactly as they did before any of them were warmed.
    engine = _cdb_engine(ouro_model, refill=True, sync=False, use_cuda_graph=True, max_num_seqs=4, num_blocks=3)
    outputs = engine.generate_batch(
        input_ids=FUSED_PROMPTS[:3],
        max_new_tokens=3,
        eos_token_id=None,
        warmup=True,
        exit_depths=[depths[:2] for depths in FUSED_EXIT_DEPTHS[:3]],
        model_kwargs={"use_early_exit_gate": True},
    )

    assert all(output.generated_tokens for output in outputs)
