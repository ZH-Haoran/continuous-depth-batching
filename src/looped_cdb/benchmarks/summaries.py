"""Aggregate raw benchmark summary rows into one row per measured configuration.

The throughput sweep appends one JSON object per (backend, refill, exit threshold, repeat) to a
summary JSONL file. This module folds repeats together and exposes the result as
:class:`AggregateRow`, the single shape the paper's results exports consume.

Deliberately free of plotting dependencies: figures are drawn elsewhere from these rows.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AggregateRow:
    """One measured configuration, averaged over its repeats."""

    backend: str
    refill: bool
    depth: int
    # ``"full"`` for the full-depth baseline, otherwise ``"q<threshold>"``.
    exit_label: str
    workload_name: str
    # Offered Poisson rate of an open-loop serving row; ``None`` for a drain row. Rows at
    # different offered loads measure different things and never fold together.
    request_rate_rps: float | None
    # The decode batch: the resident cap, which is also the widest launch the scheduler can build.
    # Part of the key for the same reason.
    max_num_seqs: int
    # Minimum coda batch the scheduler waited for before launching codas; 1 launches immediately.
    min_coda_batch_size: int
    # Prelude-core-coda layer-split override (``None`` keeps the model's default split).
    layer_split: str | None
    repeats: int
    num_requests: int
    num_blocks: int
    generated_tokens: int
    gen_tps_mean: float
    # ``None`` for a single run: a singleton group has no variance estimate.
    gen_tps_std: float | None
    recurrent_tps_mean: float
    wall_s_mean: float
    # Speed-up bound of the requested exit schedule on the served model (eq. (1), prelude and
    # coda priced in). ``None`` for the full-depth baseline, whose rows record no requested exit
    # distribution, and for rows measured before the bound was recorded.
    flop_bound_speedup: float | None
    kv_peak_blocks_frac_mean: float
    peak_alloc_gib_mean: float
    recurrent_batches_mean: float
    mixed_recurrent_batches_mean: float
    # Shared prefill/decode batch counters, reported by both backends (see ``runner.backend_stats``).
    # ``decode_batches_mean`` / ``decode_batch_size_mean`` count CB full-depth decode batches for the
    # cb backend and CDB recurrent-step batches for the cdb backend: the same decode-side concept in
    # different units. Zero for a backend whose ``backend_stats`` omits the key.
    prefill_batches_mean: float
    prefill_batch_size_mean: float
    decode_batches_mean: float
    decode_batch_size_mean: float
    # cdb only: mean tokens per recurrent-core launch and per coda launch. ``decode_batch_size_mean``
    # blends the two, weighted by launch count, so it moves with the core/coda mix rather than with
    # how full either stage's launches are; these separate the two. ``None`` for cb, which runs
    # prelude, core and coda in one launch and so has no separate stage width to report.
    recurrent_batch_size_mean: float | None
    coda_batch_size_mean: float | None
    # Deepest the pending recurrent-work queue ever grew. Zero for engines that never refill
    # freed depth slots, and the size of the tail such an engine must drain at end of run.
    max_depth_queue_mean: float
    # Mean requests preempted under KV-cache pressure (offload swaps plus recompute soft resets).
    # Zero for kv_pressure_mode in {none, reserve}, which never preempt.
    preemptions_mean: float
    # Per-request latency aggregates (mean over repeats of each run's own statistic), from the
    # ``request_latency`` block. ``None`` for rows recorded before the block existed and for
    # ``tpot`` when every request emitted a single token. ``norm_lat`` is the ORCA/vLLM
    # normalized latency (per-request e2e divided by output length), the serving figure's y-axis.
    norm_lat_mean_s_per_tok: float | None = None
    norm_lat_p99_s_per_tok: float | None = None
    queue_mean_s: float | None = None
    ttft_mean_s: float | None = None
    ttft_p99_s: float | None = None
    e2e_mean_s: float | None = None
    e2e_p99_s: float | None = None
    tpot_mean_s: float | None = None
    # Mean size of the admitted set over the run's scheduler ticks, against the ``max_num_seqs`` that
    # capped it: residency, not an achieved stage batch (the stage widths above are those). It says
    # whether admission held the batch near the cap the point was swept at. ``None`` for rows recorded
    # before it was measured.
    resident_requests_mean: float | None = None
    # Steady-state throughput, the mean over repeats of each run's ``steady_state`` block. ``None`` for
    # open-loop rows and runs too short to hold a window; the std also for a single run.
    steady_gen_tps_mean: float | None = None
    steady_gen_tps_std: float | None = None
    steady_rps_mean: float | None = None
    steady_window_s_mean: float | None = None
    steady_requests_mean: float | None = None
    steady_resident_requests_mean: float | None = None

    @property
    def is_baseline(self) -> bool:
        return self.exit_label == "full"


def load_summaries(summary_root: Path) -> list[dict[str, Any]]:
    """Load benchmark summary JSON/JSONL rows from a file or a directory tree."""

    if summary_root.is_file():
        paths = [summary_root]
    else:
        paths = sorted(summary_root.rglob("*.jsonl")) + sorted(summary_root.rglob("*.json"))

    rows: list[dict[str, Any]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"no summary rows found under {summary_root}")
    return rows


def exit_label(config: dict[str, Any]) -> str:
    """Short label for a run's exit policy.

    Full-depth continuous batching decodes every token at the maximum recurrent depth and ignores
    the exit threshold, so it is the ``"full"`` baseline regardless of any threshold the sweep
    happens to record for it.
    """

    if config.get("backend") == "cb":
        return "full"
    threshold = config.get("exit_threshold")
    return "full" if threshold is None else f"q{float(threshold):g}"


def threshold_of(row: AggregateRow) -> float | None:
    """Numeric exit threshold, or ``None`` for the full-depth baseline."""

    return None if row.is_baseline else float(row.exit_label[1:])


def baseline_row(rows: list[AggregateRow]) -> AggregateRow:
    """The full-depth CB row that speedups in this workload are normalized against."""

    baselines = [row for row in rows if row.backend == "cb"]
    if len(baselines) != 1:
        found = sorted(f"{row.backend}/{row.exit_label}" for row in rows)
        raise ValueError(f"expected exactly one full-depth cb baseline, found {len(baselines)} in {found}")
    return baselines[0]


def _config_key(config: dict[str, Any]) -> tuple[Any, ...]:
    """Identity of one measured operating point.

    Missing keys use the defaults that applied before each field was recorded.
    Arrival seeds are excluded because traces at one rate are repeats.
    """

    return (
        str(config["backend"]),
        bool(config.get("refill", True)),
        int(config["max_recurrent_depth"]),
        exit_label(config),
        str(config.get("workload_name", "workload")),
        # Two bundles can share a workload name; rows recorded before the field existed
        # carry no id and group by name as before.
        str(config.get("workload_id") or ""),
        str(config.get("model") or ""),
        str(config.get("attn_implementation") or ""),
        int(config["num_blocks"]),
        int(config.get("block_size") or 16),
        int(config["max_num_batched_tokens"]),
        # 0 encodes "unset" for these: no real configuration admits it, since a zero resident cap
        # admits nothing and a zero-length context holds no prompt.
        int(config.get("max_num_seqs") or 0),
        int(config.get("max_model_len") or 0),
        float(config.get("request_rate_rps") or 0.0),
        # Admission and cache-pressure policy: these decide how many requests run concurrently,
        # so rows that disagree measure different loads rather than one load twice. A margin of
        # 0.0 is real (``reserve`` zeroes it), so unset is a negative sentinel instead.
        str(config.get("kv_pressure_mode") or ""),
        float(config["safety_margin"]) if config.get("safety_margin") is not None else -1.0,
        float(config.get("cpu_offload_space") or 0.0),
        # Prefill admission decides how close the resident set stays to its cap, so it decides the
        # decode batch a row ran. 0 encodes "unset".
        int(config.get("min_free_slots") or 0),
        # Recurrent KV layout, and the depth schedule the row actually served.
        str(config.get("cdb_kv_policy") or ""),
        int(config.get("min_exit_step") or 0),
        int(config.get("exit_delay_steps") or 0),
        int(config.get("min_recurrent_steps") or 1),
        bool(config.get("delay_gate_consumption", True)),
        # Coda batching changes how stage reloads amortize, a different operating point; rows
        # recorded before the field existed launched every coda immediately and group under
        # the default of 1.
        int(config.get("min_coda_batch_size") or 1),
        # Prelude-core-coda layer-split override (``None`` keeps the model's default split).
        str(config.get("layer_split") or ""),
        # Graph coverage changes what a row's launches cost; rows recorded before the field
        # existed group under its CLI default of "all".
        str(config.get("cuda_graph_mode") or "all"),
        # Replayed EOS-like finishes add one lagged token per request, a different schedule.
        bool(config.get("replay_eos_finishes", False)),
    )


def aggregate_summaries(rows: list[dict[str, Any]]) -> list[AggregateRow]:
    """Fold repeated summary rows into one :class:`AggregateRow` per configuration.

    Rows sharing a :func:`_config_key` are repeats of one measurement and average together;
    everything else stays separate. The aggregate's own fields are read back from the group's
    first row, which the shared key guarantees agrees with every other row in the group.
    """

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_config_key(row["config"]), []).append(row)

    aggregates = []
    for _, group in sorted(groups.items()):
        config = group[0]["config"]
        aggregates.append(
            AggregateRow(
                backend=str(config["backend"]),
                refill=bool(config.get("refill", True)),
                depth=int(config["max_recurrent_depth"]),
                exit_label=exit_label(config),
                workload_name=str(config.get("workload_name", "workload")),
                request_rate_rps=config.get("request_rate_rps") or None,
                max_num_seqs=int(config["max_num_seqs"]),
                min_coda_batch_size=int(config.get("min_coda_batch_size") or 1),
                layer_split=config.get("layer_split") or None,
                repeats=len(group),
                num_requests=_single_config_int(group, "num_requests"),
                num_blocks=int(config["num_blocks"]),
                generated_tokens=_single_int(group, "generated_tokens"),
                gen_tps_mean=_mean(group, "generated_tokens_per_second"),
                gen_tps_std=_std(group, "generated_tokens_per_second"),
                recurrent_tps_mean=_mean(group, "recurrent_steps_per_second"),
                wall_s_mean=_mean(group, "wall_time_s"),
                flop_bound_speedup=_mean_optional(group, "flop_bound_speedup"),
                kv_peak_blocks_frac_mean=_mean_kv(group, "peak_blocks_used_fraction"),
                peak_alloc_gib_mean=_mean(group, "peak_cuda_memory_allocated_bytes") / 1024**3,
                recurrent_batches_mean=_mean_backend_stat(group, "recurrent_batches"),
                mixed_recurrent_batches_mean=_mean_backend_stat(group, "mixed_recurrent_batches"),
                prefill_batches_mean=_mean_backend_stat(group, "prefill_batches"),
                prefill_batch_size_mean=_mean_backend_stat(group, "mean_prefill_batch_size"),
                decode_batches_mean=_mean_backend_stat(group, "decode_batches"),
                decode_batch_size_mean=_mean_backend_stat(group, "mean_decode_batch_size"),
                recurrent_batch_size_mean=_mean_optional_backend_stat(group, "mean_recurrent_batch_size"),
                coda_batch_size_mean=_mean_optional_backend_stat(group, "mean_coda_batch_size"),
                resident_requests_mean=_mean_optional_backend_stat(group, "mean_resident_requests"),
                max_depth_queue_mean=_mean_backend_stat(group, "max_depth_queue_size"),
                preemptions_mean=_mean_backend_stat(group, "preemptions"),
                norm_lat_mean_s_per_tok=_mean_latency(group, "norm_e2e_s_per_token", "mean"),
                norm_lat_p99_s_per_tok=_mean_latency(group, "norm_e2e_s_per_token", "p99"),
                queue_mean_s=_mean_latency(group, "queue_s", "mean"),
                ttft_mean_s=_mean_latency(group, "ttft_s", "mean"),
                ttft_p99_s=_mean_latency(group, "ttft_s", "p99"),
                e2e_mean_s=_mean_latency(group, "e2e_s", "mean"),
                e2e_p99_s=_mean_latency(group, "e2e_s", "p99"),
                tpot_mean_s=_mean_latency(group, "tpot_s", "mean"),
                steady_gen_tps_mean=_mean_steady(group, "generated_tokens_per_second"),
                steady_gen_tps_std=_std_steady(group, "generated_tokens_per_second"),
                steady_rps_mean=_mean_steady(group, "requests_per_second"),
                steady_window_s_mean=_mean_steady(group, "window_s"),
                steady_requests_mean=_mean_steady(group, "completed_requests"),
                steady_resident_requests_mean=_mean_steady(group, "mean_resident_requests"),
            )
        )
    return aggregates


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.mean(float(row[key]) for row in rows)


def _std(rows: list[dict[str, Any]], key: str) -> float | None:
    """Sample standard deviation across repeats, or ``None`` for a single run.

    A singleton group has no variance estimate; reporting 0.0 would present "not measured" as
    "measured to be zero".
    """

    return statistics.stdev(float(row[key]) for row in rows) if len(rows) > 1 else None


def _mean_optional(rows: list[dict[str, Any]], key: str) -> float | None:
    """Mean over rows where the value is present, or ``None`` when no row carries one."""

    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.mean(values) if values else None


def _single_int(rows: list[dict[str, Any]], key: str) -> int:
    values = {int(row[key]) for row in rows}
    if len(values) != 1:
        raise ValueError(f"expected one value for {key}, got {sorted(values)}")
    return values.pop()


def _single_config_int(rows: list[dict[str, Any]], key: str) -> int:
    values = {int(row["config"][key]) for row in rows}
    if len(values) != 1:
        raise ValueError(f"expected one config value for {key}, got {sorted(values)}")
    return values.pop()


def _mean_backend_stat(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.mean(float((row.get("backend_stats") or {}).get(key, 0.0)) for row in rows)


def _mean_optional_backend_stat(rows: list[dict[str, Any]], key: str) -> float | None:
    """Mean of one backend stat, or ``None`` when no row reports it.

    A backend that never emits the key does not measure the quantity at all, which must not read
    as having measured it to be zero.
    """

    values = [float((row.get("backend_stats") or {})[key]) for row in rows if key in (row.get("backend_stats") or {})]
    return statistics.mean(values) if values else None


def _steady_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(value) for row in rows if (value := (row.get("steady_state") or {}).get(key)) is not None]


def _mean_steady(rows: list[dict[str, Any]], key: str) -> float | None:
    """Mean of one ``steady_state`` statistic across repeats, or ``None`` when no repeat holds one."""

    values = _steady_values(rows, key)
    return statistics.mean(values) if values else None


def _std_steady(rows: list[dict[str, Any]], key: str) -> float | None:
    values = _steady_values(rows, key)
    return statistics.stdev(values) if len(values) > 1 else None


def _mean_latency(rows: list[dict[str, Any]], series: str, stat: str) -> float | None:
    """Mean of one ``request_latency`` statistic across repeats, or ``None`` when absent.

    Rows recorded before the latency block existed carry none, and ``tpot_s`` is null on a run
    where every request emitted a single token; both must read as unmeasured, not zero.
    """

    values = [
        float(value)
        for row in rows
        if (value := ((row.get("request_latency") or {}).get(series) or {}).get(stat)) is not None
    ]
    return statistics.mean(values) if values else None


def _mean_kv(rows: list[dict[str, Any]], key: str) -> float:
    """Mean of a ``kv_cache`` field, skipping rows that do not carry it.

    A missing measurement is not zero occupancy: counting it as zero would drag a fairness
    comparison toward under-utilization. Returns 0.0 only when no row carries the field.
    """

    values = [float(value) for row in rows if (value := (row.get("kv_cache") or {}).get(key)) is not None]
    return statistics.mean(values) if values else 0.0
