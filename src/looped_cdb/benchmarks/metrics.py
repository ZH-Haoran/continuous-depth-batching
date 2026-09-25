"""Shared metric schema for CB vs CDB benchmarks.

Backends replay a recorded workload: request lengths drive prefill/decode work
and, for the early-exit backends, a per-request recorded exit schedule drives the
recurrent depths. Each summary records both the requested exit distribution (the
recorded schedule) and the effective depth work the backend actually performed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from looped_cdb.benchmarks.flop_bound import StageFlops
from looped_cdb.utils import DEFAULT_MAX_NUM_SEQS

BackendName = Literal["cb", "cdb"]


def _latency_stats(values: list[float]) -> dict[str, float]:
    """Mean and tail percentiles for one per-request latency series.

    ``n`` is the population the percentiles were taken over; series that exclude some
    requests (``tpot_s`` drops single-token requests) carry a smaller ``n`` than the
    summary's request count.
    """

    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def request_latency_summary(outputs: list[Any]) -> dict[str, Any]:
    """Per-request serving-latency percentiles from finished generation outputs.

    Every request carries four stamps: ``created_time`` (arrival: the scheduled release
    instant on an open-loop run, request construction on a drain run), ``lifespan[0]``
    (first scheduled), ``first_token_time`` (first generated token consumed on the host,
    which on the async engines is when the token could actually be emitted), and
    ``lifespan[1]`` (finished). The summary reports, per request and in seconds:

    - ``queue_s``           arrival -> first scheduled
    - ``ttft_s``            arrival -> first token
    - ``e2e_s``             arrival -> finished
    - ``tpot_s``            (finished - first token) / (generated - 1), single-token requests excluded
    - ``norm_e2e_s_per_token``  e2e / generated, the ORCA/vLLM "normalized latency"

    Normalized latency is computed per request before averaging: on heavy-tailed output
    lengths ``mean(e2e / len)`` is not ``mean(e2e) / mean(len)``, so it cannot be
    recovered from the other series after the fact.
    """

    if not outputs:
        raise ValueError("request_latency_summary needs at least one output")
    queue: list[float] = []
    ttft: list[float] = []
    e2e: list[float] = []
    tpot: list[float] = []
    norm_e2e: list[float] = []
    for output in outputs:
        scheduled, finished = output.lifespan
        if output.created_time < 0 or scheduled < 0 or finished < 0 or output.first_token_time < 0:
            raise ValueError(f"request {output.request_id} is missing a latency stamp (did it finish?)")
        queue.append(scheduled - output.created_time)
        ttft.append(output.first_token_time - output.created_time)
        e2e.append(finished - output.created_time)
        generated = len(output.generated_tokens)
        norm_e2e.append((finished - output.created_time) / generated)
        if generated > 1:
            tpot.append((finished - output.first_token_time) / (generated - 1))
    return {
        "num_requests": len(outputs),
        "queue_s": _latency_stats(queue),
        "ttft_s": _latency_stats(ttft),
        "e2e_s": _latency_stats(e2e),
        "tpot_s": _latency_stats(tpot) if tpot else None,
        "norm_e2e_s_per_token": _latency_stats(norm_e2e),
    }


@dataclass(frozen=True)
class ExitDistributionSummary:
    """Summary of recurrent depths for generated tokens."""

    kind: str
    max_depth: int
    token_count: int
    total_depth_work: int
    mean_depth: float
    min_depth: int | None
    max_depth_observed: int | None
    depth_histogram: dict[str, int]

    @classmethod
    def from_depths(cls, kind: str, max_depth: int, depths: list[int]) -> ExitDistributionSummary:
        """Build a stable summary from per-token recurrent depths."""

        if max_depth <= 0:
            raise ValueError(f"max_depth must be positive, but got {max_depth}")
        for depth in depths:
            if depth < 1 or depth > max_depth:
                raise ValueError(f"depths must be in [1, {max_depth}], but got {depth}")

        token_count = len(depths)
        total_depth_work = sum(depths)
        mean_depth = total_depth_work / token_count if token_count else 0.0
        histogram: dict[str, int] = {}
        for depth in depths:
            key = str(depth)
            histogram[key] = histogram.get(key, 0) + 1

        return cls(
            kind=kind,
            max_depth=max_depth,
            token_count=token_count,
            total_depth_work=total_depth_work,
            mean_depth=mean_depth,
            min_depth=min(depths) if depths else None,
            max_depth_observed=max(depths) if depths else None,
            depth_histogram=dict(sorted(histogram.items(), key=lambda item: int(item[0]))),
        )

    @classmethod
    def from_histogram(
        cls,
        kind: str,
        max_depth: int,
        depth_histogram: dict[int, int],
    ) -> ExitDistributionSummary:
        """Build a stable summary from a depth histogram."""

        depths: list[int] = []
        for depth, count in sorted(depth_histogram.items()):
            if count < 0:
                raise ValueError(f"depth histogram counts must be non-negative, got {count} for depth {depth}")
            depths.extend([depth] * count)
        return cls.from_depths(kind=kind, max_depth=max_depth, depths=depths)

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-stable dictionary."""

        return asdict(self)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration fields needed to interpret one benchmark summary."""

    backend: BackendName
    model: str
    max_recurrent_depth: int
    workload_path: str
    workload_name: str
    num_requests: int
    measured_repeat: int
    seed: int
    attn_implementation: str
    use_async_batching: bool
    use_cuda_graph: bool
    cuda_graph_mode: str
    block_size: int
    num_blocks: int | None
    max_num_batched_tokens: int | None
    mem_fraction_static: float | None = None
    # Identity of the exact workload bundle behind ``workload_name`` (recording seeds, size, and
    # depth). Two bundles can share a dataset name; aggregation keys on this field so rows recorded
    # from different bundles never average together as repeats.
    workload_id: str | None = None
    # The decode batch the scheduler applied: the resident cap, which is also its widest launch.
    max_num_seqs: int = DEFAULT_MAX_NUM_SEQS
    # Prefill admission the scheduler applied: the resident slots that must stand open before a
    # prefill launch runs, as the scheduler resolved it. It bounds how far the decode batch falls
    # below its cap, so rows only compare when they agree.
    min_free_slots: int | None = None
    max_model_len: int | None = None
    cdb_kv_policy: str = "single"
    # Prelude-core-coda layer-split override (``None`` keeps the model's default split).
    layer_split: str | None = None
    # Admission and cache-pressure policy. These change how many requests run concurrently, so two
    # rows are only comparable when they agree; recording them keeps a policy change from reading
    # as a throughput change. ``safety_margin`` is the value the scheduler applied, which is 0.0
    # under ``kv_pressure_mode="reserve"`` regardless of what was requested. ``cpu_offload_space``
    # sizes the pinned CPU swap pool and so governs how often "offload" preempts.
    kv_pressure_mode: str | None = None
    safety_margin: float | None = None
    cpu_offload_space: float | None = None
    # Whether the row simulated sampled-EOS finishes: the replayed length finish discovered at
    # consume time, one lagged token per request computed and discarded.
    replay_eos_finishes: bool = False
    # Input (prompt) length statistics for the replayed workload.
    prompt_total_tokens: int = 0
    prompt_min_tokens: int | None = None
    prompt_max_tokens: int | None = None
    prompt_mean_tokens: float = 0.0
    # Output length statistics for the replayed workload.
    output_total_tokens: int = 0
    output_min_tokens: int | None = None
    output_max_tokens: int | None = None
    output_mean_tokens: float = 0.0
    # Offline PDF-to-depth derivation used for the early-exit backends.
    exit_threshold: float | None = None
    min_exit_step: int = 1
    exit_delay_steps: int = 0
    delay_gate_consumption: bool = True
    # cdb only: engine-side floor on how many recurrent steps a token runs before the gate may
    # exit it. Distinct from ``min_exit_step``, which floors the depths derived offline from the
    # recorded PDF: this one binds inside the engine regardless of what the schedule asked for.
    min_recurrent_steps: int = 1
    # cdb only: whether freed depth slots are refilled (CDB) or left empty (the
    # sequence-level no-refill baseline). Distinguishes the two cdb points that
    # otherwise share a backend, threshold, and workload.
    refill: bool = True
    # cdb only: pending-exit pool size the coda bucket waits for before launching. Changes how
    # stage reloads amortize, a different operating point, so aggregation keys on it.
    min_coda_batch_size: int = 1
    # Open-loop serving: offered Poisson arrival rate in requests/s and the seed of the
    # arrival trace (identical across backends at the same seed). ``None`` marks a drain
    # run, where every request is submitted upfront; rows at different rates measure
    # different offered loads and must never aggregate together.
    request_rate_rps: float | None = None
    arrival_seed: int | None = None

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-stable dictionary."""

        return asdict(self)


@dataclass(frozen=True)
class BenchmarkSummary:
    """Compact summary emitted for every measured benchmark run."""

    config: BenchmarkConfig
    run_id: str
    wall_time_s: float
    generated_tokens: int
    completed_requests: int
    # ``None`` for the full-depth cb baseline: it ignores the exit schedule, so a requested
    # distribution (and everything derived from it) would just echo whatever threshold the sweep
    # happened to launch the row under.
    requested_exit_distribution: ExitDistributionSummary | None
    effective_exit_distribution: ExitDistributionSummary
    # Per-stage FLOP weights of the served model, which set the speed-up bound adaptive depth
    # can reach (see :mod:`looped_cdb.benchmarks.flop_bound`).
    stage_flops: StageFlops
    peak_cuda_memory_allocated_bytes: int | None
    peak_cuda_memory_reserved_bytes: int | None
    kv_cache_num_blocks: int
    kv_cache_block_size: int
    kv_cache_max_num_batched_tokens: int
    kv_cache_num_pages: int
    kv_cache_peak_blocks_used: int | None = None
    kv_cache_final_blocks_used: int | None = None
    first_token_full_depth_count: int = 0
    backend_stats: dict[str, Any] | None = None
    request_latency: dict[str, Any] | None = None
    # Throughput inside the steady-state window, the run with its fill and drain cut away (see
    # ``BaseServingScheduler.steady_state_summary``). ``None`` for open-loop serving rows and runs too
    # short to hold a window.
    steady_state: dict[str, Any] | None = None
    # GPU used. ``None`` when the run did not stamp it.
    device_name: str | None = None

    @property
    def generated_tokens_per_second(self) -> float:
        """Generated-token throughput."""

        return self.generated_tokens / self.wall_time_s if self.wall_time_s > 0 else 0.0

    @property
    def completed_requests_per_second(self) -> float:
        """Completed-request throughput."""

        return self.completed_requests / self.wall_time_s if self.wall_time_s > 0 else 0.0

    @property
    def recurrent_steps_per_second(self) -> float:
        """Effective recurrent-step throughput paid by the backend."""

        return self.effective_exit_distribution.total_depth_work / self.wall_time_s if self.wall_time_s > 0 else 0.0

    @property
    def full_depth_work(self) -> int:
        """Recurrent work a full-depth CB backend would pay for these generated tokens."""

        return self.generated_tokens * self.config.max_recurrent_depth

    @property
    def skipped_depth_work_available(self) -> int | None:
        """Depth work CB pays that an ideal CDB implementation could skip."""

        if self.requested_exit_distribution is None:
            return None
        return self.full_depth_work - self.requested_exit_distribution.total_depth_work

    @property
    def effective_skipped_depth_work(self) -> int:
        """Depth work the measured backend skipped relative to full-depth CB."""

        return self.full_depth_work - self.effective_exit_distribution.total_depth_work

    @property
    def depth_work_over_requested(self) -> int | None:
        """Extra recurrent work paid beyond the requested exit distribution."""

        if self.requested_exit_distribution is None:
            return None
        return self.effective_exit_distribution.total_depth_work - self.requested_exit_distribution.total_depth_work

    @property
    def flop_bound_speedup(self) -> float | None:
        """Speed-up bound of the requested exit schedule on the served model.

        Prices in the prelude, coda and output head, whose FLOPs no early exit removes, so it
        is strictly below the depth ratio and far below it for models with heavy boundary
        stages. ``None`` for the full-depth baseline, which requests no schedule.
        """

        if self.requested_exit_distribution is None:
            return None
        return self.stage_flops.ideal_speedup(
            max_depth=self.config.max_recurrent_depth,
            mean_depth=self.requested_exit_distribution.mean_depth,
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-stable summary dictionary."""

        return {
            "run_id": self.run_id,
            "config": self.config.to_json_dict(),
            "wall_time_s": self.wall_time_s,
            "generated_tokens": self.generated_tokens,
            "completed_requests": self.completed_requests,
            "generated_tokens_per_second": self.generated_tokens_per_second,
            "completed_requests_per_second": self.completed_requests_per_second,
            "steady_state": self.steady_state,
            "recurrent_steps_per_second": self.recurrent_steps_per_second,
            "full_depth_work": self.full_depth_work,
            "requested_exit_distribution": (
                self.requested_exit_distribution.to_json_dict()
                if self.requested_exit_distribution is not None
                else None
            ),
            "effective_exit_distribution": self.effective_exit_distribution.to_json_dict(),
            "skipped_depth_work_available": self.skipped_depth_work_available,
            "effective_skipped_depth_work": self.effective_skipped_depth_work,
            "depth_work_over_requested": self.depth_work_over_requested,
            "stage_flops": self.stage_flops.to_json_dict(),
            "flop_bound_speedup": self.flop_bound_speedup,
            "first_token_full_depth_count": self.first_token_full_depth_count,
            "backend_stats": self.backend_stats,
            "request_latency": self.request_latency,
            "device_name": self.device_name,
            "peak_cuda_memory_allocated_bytes": self.peak_cuda_memory_allocated_bytes,
            "peak_cuda_memory_reserved_bytes": self.peak_cuda_memory_reserved_bytes,
            "kv_cache": {
                "num_blocks": self.kv_cache_num_blocks,
                "block_size": self.kv_cache_block_size,
                "max_num_batched_tokens": self.kv_cache_max_num_batched_tokens,
                "num_pages": self.kv_cache_num_pages,
                "peak_blocks_used": self.kv_cache_peak_blocks_used,
                "peak_blocks_used_fraction": (
                    self.kv_cache_peak_blocks_used / self.kv_cache_num_blocks
                    if self.kv_cache_peak_blocks_used is not None and self.kv_cache_num_blocks
                    else None
                ),
                "final_blocks_used": self.kv_cache_final_blocks_used,
            },
        }

    def to_json(self) -> str:
        """Serialize this summary as one compact JSON object."""

        return json.dumps(self.to_json_dict(), sort_keys=True)
