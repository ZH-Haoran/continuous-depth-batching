"""Tests for the CPU-only helpers in looped_cdb.benchmarks.runner."""

import ast
import dataclasses
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from repo_paths import SCRIPTS_DIR
from torch import nn
from transformers import PreTrainedConfig

from looped_cdb.benchmarks import runner
from looped_cdb.benchmarks.metrics import BenchmarkConfig
from looped_cdb.benchmarks.runner import (
    WARMUP_OUTPUT_TOKENS,
    EngineArgs,
    PreparedRun,
    _delayed_synthetic_active,
    _length_summary,
    backend_stats,
    build_filler_prompts,
    materialized_to_replay,
    measure_once,
    requested_exit_distribution,
    resolve_cuda_graph,
    resolve_filler_token_id,
    set_recurrent_depth,
    warm_up_engine,
)
from looped_cdb.continuous_batching.config import ContinuousBatchingConfig
from looped_cdb.continuous_batching.continuous_api import ContinuousBatchingEngine
from looped_cdb.continuous_depth_batching import ContinuousDepthBatchingConfig, ContinuousDepthBatchingEngine
from looped_cdb.continuous_depth_batching.model_adapter import CDBModelAdapter

SHARED_BATCH_STAT_KEYS = frozenset(
    {"prefill_batches", "mean_prefill_batch_size", "decode_batches", "mean_decode_batch_size"}
)


def _call_sites(class_name: str) -> list[tuple[Path, ast.Call]]:
    """Every ``<class_name>(...)`` construction under scripts/, as (path, call node)."""

    sites = []
    for path in sorted(SCRIPTS_DIR.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == class_name:
                sites.append((path, node))
    return sites


class ReadOnlyDepthModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(total_ut_steps=4)

    @property
    def total_ut_steps(self) -> int:
        return self.config.total_ut_steps


def test_every_engine_args_call_site_passes_all_required_fields() -> None:
    """``EngineArgs`` gives its fields no defaults, so omitting one is a TypeError at runtime.

    The benchmark scripts build it deep inside GPU measurement loops that no CPU test can
    reach, so a missing field survives both the linters and the rest of the suite and only
    surfaces on the cluster. Adding a field to the dataclass is therefore an obligation on
    every construction site; this reads both sides off the source so neither a new field nor
    a new script needs the test updated.
    """

    required = {
        field.name
        for field in dataclasses.fields(EngineArgs)
        if field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING
    }
    sites = _call_sites("EngineArgs")
    assert sites, "no EngineArgs construction found under scripts/ - has it been renamed or aliased?"

    missing = {
        f"{path.name}:{call.lineno}": sorted(required - {kw.arg for kw in call.keywords})
        # A **kwargs spread carries arg=None and can supply anything, so it is not checkable here.
        for path, call in sites
        if not any(kw.arg is None for kw in call.keywords) and required - {kw.arg for kw in call.keywords}
    }
    assert not missing, f"EngineArgs call sites missing required fields: {missing}"


def test_benchmark_config_call_sites_pass_only_known_fields() -> None:
    """``BenchmarkConfig`` defaults every field, so an unknown keyword raises only at runtime -
    after the benchmark has already run, the most expensive moment to find out."""

    known = {field.name for field in dataclasses.fields(BenchmarkConfig)}
    sites = _call_sites("BenchmarkConfig")
    assert sites, "no BenchmarkConfig construction found under scripts/ - has it been renamed or aliased?"
    unknown = {
        f"{path.name}:{call.lineno}": sorted(passed - known)
        # A **kwargs spread carries arg=None and can supply anything, so it is not checkable here.
        for path, call in sites
        if (passed := {kw.arg for kw in call.keywords if kw.arg is not None}) - known
    }
    assert not unknown, f"BenchmarkConfig call sites pass unknown fields: {unknown}"


def test_set_recurrent_depth_updates_config_when_inner_property_is_read_only() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(total_ut_steps=4),
        model=ReadOnlyDepthModel(),
    )

    set_recurrent_depth(model, 8)

    assert model.config.total_ut_steps == 8
    assert model.model.total_ut_steps == 8


def test_set_recurrent_depth_dispatches_on_model_family() -> None:
    huginn = SimpleNamespace(config=SimpleNamespace(model_type="huginn_raven", total_recurrent_steps=32))

    set_recurrent_depth(huginn, 16)

    assert huginn.config.total_recurrent_steps == 16
    assert not hasattr(huginn.config, "total_ut_steps")


def test_build_filler_prompts_matches_requested_lengths() -> None:
    prompts = build_filler_prompts(np.array([3, 1, 5], dtype=np.int32), filler_token_id=7)
    assert [len(p) for p in prompts] == [3, 1, 5]
    assert all(token == 7 for prompt in prompts for token in prompt)


def test_build_filler_prompts_rejects_empty_prompt() -> None:
    with pytest.raises(ValueError, match="input_lens"):
        build_filler_prompts(np.array([2, 0], dtype=np.int32), filler_token_id=1)


def test_resolve_filler_token_id_prefers_valid_bos() -> None:
    assert resolve_filler_token_id(SimpleNamespace(bos_token_id=5)) == 5
    assert resolve_filler_token_id(SimpleNamespace(bos_token_id=None)) == 0
    assert resolve_filler_token_id(SimpleNamespace(bos_token_id=-1)) == 0


def test_materialized_to_replay_drops_first_token_and_offsets() -> None:
    materialized = [[1, 2, 3, 4], [4, 2]]
    # Immediate policy: drop the first token, subtract 1.
    assert materialized_to_replay(materialized, delayed=False) == [[1, 2, 3], [1]]
    # Delayed policy: subtract 2.
    assert materialized_to_replay([[2, 3, 4]], delayed=True) == [[1, 2]]


def test_materialized_to_replay_lengths_are_output_len_minus_one() -> None:
    materialized = [[2, 2, 2, 2, 2], [3, 3]]
    replay = materialized_to_replay(materialized, delayed=False)
    assert [len(row) for row in replay] == [4, 1]


def test_resolve_cuda_graph_maps_mode_to_decode_and_prefill_switches() -> None:
    assert resolve_cuda_graph("all") == (True, True)
    assert resolve_cuda_graph("decode") == (True, False)
    assert resolve_cuda_graph("none") == (False, False)


def test_resolve_cuda_graph_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="cuda_graph_mode"):
        resolve_cuda_graph("both")


def test_materialized_to_replay_rejects_depth_one_on_delayed_path() -> None:
    with pytest.raises(ValueError, match="delayed"):
        materialized_to_replay([[3, 1]], delayed=True)


def test_delayed_synthetic_active_truth_table() -> None:
    base = SimpleNamespace(
        backend="cdb",
        sync=False,
        no_delay_gate_consumption=False,
        refill=True,
    )
    assert _delayed_synthetic_active(base)
    assert not _delayed_synthetic_active(SimpleNamespace(**{**base.__dict__, "no_delay_gate_consumption": True}))
    assert not _delayed_synthetic_active(SimpleNamespace(**{**base.__dict__, "sync": True}))
    assert not _delayed_synthetic_active(SimpleNamespace(**{**base.__dict__, "backend": "cb"}))
    # Refill does not gate the delayed offset: the no-refill baseline also runs the async
    # delayed-gate path by default, so it replays with the same delayed offset.
    assert _delayed_synthetic_active(SimpleNamespace(**{**base.__dict__, "refill": False}))


def test_length_summary_reports_prefixed_stats() -> None:
    summary = _length_summary(np.array([2, 5, 8], dtype=np.int32), "output")
    assert summary == {
        "output_total_tokens": 15,
        "output_min_tokens": 2,
        "output_max_tokens": 8,
        "output_mean_tokens": 5.0,
    }


def test_requested_exit_distribution_counts_first_token_full_depth() -> None:
    # Each request's first output token is served full depth; only row[1:] contributes
    # recorded depths. This mirrors the effective summary so ideal speedup is not inflated.
    materialized = [[2, 3, 4], [1, 2]]
    summary = requested_exit_distribution(materialized, max_depth=4, generated_tokens=5, full_depth_prefix_count=2)
    # prefix: 2 tokens at depth 4; row[1:] -> [3, 4] and [2]
    assert summary.depth_histogram == {"2": 1, "3": 1, "4": 3}
    assert summary.total_depth_work == 4 * 3 + 3 + 2
    assert summary.mean_depth == pytest.approx(17 / 5)


def test_requested_exit_distribution_all_full_without_schedule() -> None:
    summary = requested_exit_distribution(None, max_depth=4, generated_tokens=3, full_depth_prefix_count=1)
    assert summary.depth_histogram == {"4": 3}


def test_measure_once_nvtx_range_wraps_generate_and_final_sync(monkeypatch) -> None:
    events: list[str] = []

    @contextmanager
    def fake_range(name: str, *, registered: bool = False):
        events.append(f"push:{name}:{registered}")
        try:
            yield
        finally:
            events.append("pop")

    monkeypatch.setattr(runner.nvtx, "range", fake_range)
    monkeypatch.setattr(runner, "generate_once", lambda *a, **k: events.append("generate") or ["out"])
    monkeypatch.setattr(runner, "maybe_sync_cuda", lambda: events.append("sync"))
    monkeypatch.setattr(runner, "peak_memory_stats", lambda: (None, None))

    prepared = PreparedRun(
        engine=object(),
        input_ids=[[1]],
        max_new_tokens=[1],
        exit_depths=None,
        materialized=None,
        model_kwargs=None,
        backend="cb",
        max_recurrent_depth=4,
    )
    measured = measure_once(prepared, nvtx_label="benchmark.generate")

    assert measured.outputs == ["out"]
    # The generate and the final sync both happen inside the nvtx range (before pop).
    assert events == ["sync", "push:benchmark.generate:True", "generate", "sync", "pop"]


def _cb_config() -> PreTrainedConfig:
    return PreTrainedConfig(
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        hidden_size=4,
        vocab_size=16,
        sliding_window=None,
        layer_types=None,
        _attn_implementation="paged|flash_attention_3",
    )


class _CBIncrementModel(nn.Module):
    def __init__(self, config: PreTrainedConfig) -> None:
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids: torch.Tensor, **_: Any) -> SimpleNamespace:
        logits = torch.full(
            (1, input_ids.size(1), self.config.vocab_size), -10.0, dtype=torch.float32, device=input_ids.device
        )
        next_tokens = (input_ids[0].to(dtype=torch.long) + 1) % self.config.vocab_size
        logits[0, torch.arange(input_ids.size(1), device=input_ids.device), next_tokens] = 10.0
        return SimpleNamespace(logits=logits)


def _cdb_config_obj() -> SimpleNamespace:
    return SimpleNamespace(
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        hidden_size=4,
        vocab_size=16,
        sliding_window=None,
        layer_types=["full_attention", "full_attention", "full_attention"],
        _attn_implementation="paged|flash_attention_3",
        total_ut_steps=3,
    )


class _CDBIncrementModel(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids: torch.Tensor, **_: Any) -> SimpleNamespace:
        depth = self.config.total_ut_steps
        next_tokens = (input_ids[0].to(dtype=torch.long) + depth) % self.config.vocab_size
        return SimpleNamespace(logits=self._logits_for_tokens(next_tokens))

    def _logits_for_tokens(self, next_tokens: torch.Tensor) -> torch.Tensor:
        logits = torch.full(
            (1, next_tokens.numel(), self.config.vocab_size), -10.0, dtype=torch.float32, device=next_tokens.device
        )
        logits[0, torch.arange(next_tokens.numel(), device=next_tokens.device), next_tokens] = 10.0
        return logits


class _CDBIncrementAdapter(CDBModelAdapter):
    def __init__(self, model: _CDBIncrementModel) -> None:
        self.model = model

    def configure_recurrent_steps(self, max_recurrent_steps: int) -> None:
        self.model.config.total_ut_steps = max_recurrent_steps

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
        **_: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del position_ids, cache_position
        hidden_states = hidden_states + 1.0
        gate_values = torch.where(recurrent_steps == 0, 10.0, -10.0).to(dtype=hidden_states.dtype)
        gate_logits = gate_values.view(1, -1, 1).to(device=hidden_states.device)
        return hidden_states, gate_logits

    def lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        next_tokens = hidden_states[0, :, 0].round().to(dtype=torch.long) % self.model.config.vocab_size
        return self.model._logits_for_tokens(next_tokens)


def test_cb_backend_stats_counts_prefill_and_decode_batches() -> None:
    cb_config = ContinuousBatchingConfig(
        num_blocks=8,
        block_size=4,
        max_num_batched_tokens=4,
        max_model_len=16,
        use_async_batching=False,
        use_cuda_graph=False,
    )
    engine = ContinuousBatchingEngine.from_model(_CBIncrementModel(_cb_config()), cb_config, dtype=torch.float32)

    engine.generate_batch(input_ids=[[1, 2], [5]], max_new_tokens=3, eos_token_id=None, warmup=False)

    stats = engine.last_stats
    # Two prompts prefill (each a >1-token batch), then the pair decodes together for the rest.
    assert stats.prefill_batches >= 1
    assert stats.decode_batches >= 1
    assert stats.prefill_requests >= stats.prefill_batches
    assert stats.mean_prefill_batch_size == pytest.approx(stats.prefill_requests / stats.prefill_batches)
    assert stats.mean_decode_batch_size == pytest.approx(stats.decode_tokens / stats.decode_batches)
    # Query tokens summed over prefill batches exceed request count when a prompt carries several tokens.
    assert stats.prefill_query_tokens >= stats.prefill_requests

    reported = backend_stats(engine)
    assert reported is not None
    assert reported.keys() >= SHARED_BATCH_STAT_KEYS
    assert reported["prefill_batches"] == stats.prefill_batches
    assert reported["decode_batches"] == stats.decode_batches
    assert reported["mean_decode_batch_size"] == pytest.approx(stats.mean_decode_batch_size)


def test_warm_up_engine_replays_a_bounded_slice() -> None:
    cb_config = ContinuousBatchingConfig(
        num_blocks=128,
        block_size=4,
        max_num_batched_tokens=4,
        max_model_len=64,
        use_async_batching=False,
        use_cuda_graph=False,
    )
    engine = ContinuousBatchingEngine.from_model(_CBIncrementModel(_cb_config()), cb_config, dtype=torch.float32)
    num_workload_requests = 20

    warm_up_engine(
        engine,
        backend="cb",
        input_ids=[[i % 8] for i in range(num_workload_requests)],
        max_new_tokens=[100] * num_workload_requests,
        exit_depths=None,
        model_kwargs=None,
    )

    # The pass is sized by the engine, not the workload: twice the resident cap of requests
    # (auto cap = 1.5 * the 4-token decode width = 6), each capped to WARMUP_OUTPUT_TOKENS.
    # Every scheduled appearance of a request lands in exactly one of the two counters (a mixed
    # tick counts as prefill), so their sum pins both the request count and the output cap.
    stats = engine.last_stats
    assert stats.decode_tokens + stats.prefill_requests == 2 * engine.scheduler.max_num_seqs * WARMUP_OUTPUT_TOKENS


def test_warm_up_engine_truncates_cdb_replay_schedules_with_the_slice() -> None:
    model = _CDBIncrementModel(_cdb_config_obj())
    cdb_config = ContinuousDepthBatchingConfig(
        num_blocks=256,
        block_size=4,
        max_num_batched_tokens=4,
        max_model_len=64,
        use_async_batching=False,
        use_cuda_graph=False,
        max_recurrent_steps=3,
        synthetic_exit_replay=True,
    )
    engine = ContinuousDepthBatchingEngine.from_model(
        model, cdb_config, dtype=torch.float32, model_adapter=_CDBIncrementAdapter(model)
    )
    num_workload_requests = 20
    output_len = 40

    # The schedules match the full output length; warm_up_engine must re-cut each one to the
    # capped slice or the engine's per-request length validation rejects the replay outright.
    warm_up_engine(
        engine,
        backend="cdb",
        input_ids=[[i % 8] for i in range(num_workload_requests)],
        max_new_tokens=[output_len] * num_workload_requests,
        exit_depths=[[1] * (output_len - 1)] * num_workload_requests,
        model_kwargs=None,
    )

    # Same bound as the CB slice, exercised through the staged decode path: every decoded token
    # exits through the coda, so the coda count pins both the request and output-token caps.
    stats = engine.last_stats
    assert stats.prefill_requests == 2 * engine.scheduler.max_num_seqs
    assert stats.coda_tokens == 2 * engine.scheduler.max_num_seqs * (WARMUP_OUTPUT_TOKENS - 1)


def test_cb_reset_clears_batch_counters_between_generations() -> None:
    cb_config = ContinuousBatchingConfig(
        num_blocks=8,
        block_size=4,
        max_num_batched_tokens=4,
        max_model_len=16,
        use_async_batching=False,
        use_cuda_graph=False,
    )
    engine = ContinuousBatchingEngine.from_model(_CBIncrementModel(_cb_config()), cb_config, dtype=torch.float32)

    engine.generate_batch(input_ids=[[1, 2]], max_new_tokens=4, eos_token_id=None, warmup=False)
    first = engine.last_stats.decode_batches
    engine.generate_batch(input_ids=[[1, 2]], max_new_tokens=4, eos_token_id=None, warmup=False)

    # Counters accumulate within one generation and reset for the next, so the repeat matches the first.
    assert first >= 1
    assert engine.last_stats.decode_batches == first


def test_cdb_backend_stats_reports_shared_and_native_counters() -> None:
    model = _CDBIncrementModel(_cdb_config_obj())
    cdb_config = ContinuousDepthBatchingConfig(
        num_blocks=16,
        block_size=4,
        max_num_batched_tokens=4,
        max_model_len=16,
        use_async_batching=False,
        use_cuda_graph=False,
        max_recurrent_steps=3,
    )
    engine = ContinuousDepthBatchingEngine.from_model(
        model, cdb_config, dtype=torch.float32, model_adapter=_CDBIncrementAdapter(model)
    )

    engine.generate_batch(input_ids=[[1, 2], [5]], max_new_tokens=3, eos_token_id=None, warmup=False)

    stats = engine.last_stats
    assert stats.prefill_batches >= 1
    assert stats.prefill_requests == 2
    assert stats.mean_prefill_batch_size == pytest.approx(stats.prefill_requests / stats.prefill_batches)
    assert stats.recurrent_batches >= 1
    # The recurrent-batch mean reuses the existing per-batch item total instead of a new accumulator.
    assert stats.mean_recurrent_batch_size == pytest.approx(stats.recurrent_steps / stats.recurrent_batches)
    assert stats.mean_coda_batch_size == pytest.approx(stats.coda_tokens / stats.coda_batches)

    # The shared decode key counts every decode-side launch, so it sums the recurrent and coda batches.
    # A CB decode launch runs the prelude, the recurrent stack, and the coda together, so comparing it
    # against recurrent batches alone would undercount CDB's launches by exactly the coda count.
    assert stats.decode_batches == stats.recurrent_batches + stats.coda_batches
    assert stats.mean_decode_batch_size == pytest.approx(
        (stats.recurrent_steps + stats.coda_tokens) / stats.decode_batches
    )

    reported = backend_stats(engine)
    assert reported is not None
    assert reported.keys() >= SHARED_BATCH_STAT_KEYS
    assert reported["decode_batches"] == stats.decode_batches
    assert reported["mean_decode_batch_size"] == pytest.approx(stats.mean_decode_batch_size)
    # Native CDB counters remain reported alongside the shared keys, so the sum stays decomposable.
    assert reported["recurrent_batches"] == stats.recurrent_batches
    assert reported["coda_batches"] == stats.coda_batches
    assert "exit_depth_histogram" in reported


def test_both_engines_expose_the_same_shared_batch_stat_keys() -> None:
    cb_config = ContinuousBatchingConfig(
        num_blocks=8,
        block_size=4,
        max_num_batched_tokens=4,
        max_model_len=16,
        use_async_batching=False,
        use_cuda_graph=False,
    )
    cb_engine = ContinuousBatchingEngine.from_model(_CBIncrementModel(_cb_config()), cb_config, dtype=torch.float32)
    cb_engine.generate_batch(input_ids=[[1, 2]], max_new_tokens=2, eos_token_id=None, warmup=False)

    model = _CDBIncrementModel(_cdb_config_obj())
    cdb_config = ContinuousDepthBatchingConfig(
        num_blocks=16,
        block_size=4,
        max_num_batched_tokens=4,
        max_model_len=16,
        use_async_batching=False,
        use_cuda_graph=False,
        max_recurrent_steps=3,
    )
    cdb_engine = ContinuousDepthBatchingEngine.from_model(
        model, cdb_config, dtype=torch.float32, model_adapter=_CDBIncrementAdapter(model)
    )
    cdb_engine.generate_batch(input_ids=[[1, 2]], max_new_tokens=2, eos_token_id=None, warmup=False)

    cb_reported = backend_stats(cb_engine)
    cdb_reported = backend_stats(cdb_engine)
    assert cb_reported is not None and cdb_reported is not None
    # A plotting script can read the shared keys off either row without checking the backend.
    assert cb_reported.keys() >= SHARED_BATCH_STAT_KEYS
    assert cdb_reported.keys() >= SHARED_BATCH_STAT_KEYS
