"""Benchmark summary rows and workload bundles the export tests replay.

The exporters read what a run wrote down, so their tests need rows shaped like real ones. These
builders keep that shape in one place, since the exit-sweep, launch-width and shared-plumbing
tests all publish from the same kind of row.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def summary_row(
    *,
    backend: str = "cb",
    refill: bool = True,
    exit_threshold: float | None = None,
    gen_tps: float = 100.0,
    bound: float = 1.0,
    stage_flops: dict[str, int] | None = None,
    workload_name: str = "alpaca",
    num_requests: int = 4,
    workload_path: str = "outputs/workloads/ouro_alpaca_recur4.json",
    num_blocks: int = 16,
    max_num_seqs: int = 8,
    min_coda_batch_size: int = 1,
    steady_gen_tps: float | None = None,
) -> dict[str, Any]:
    """One summary row as ``scripts/benchmark_throughput.py`` writes it.

    The default weights make the boundary stages weightless, so the FLOP bound reduces to the
    depth ratio and the expected values in the tests can be derived by hand from the bundle.
    ``steady_gen_tps`` adds the steady-state window block; ``None`` leaves the row without one, as a
    run too short to hold a window is written.
    """

    steady_state = (
        None
        if steady_gen_tps is None
        else {
            "warmup_requests": 2 * max_num_seqs,
            "window_s": 5.0,
            "ticks": 50,
            "completed_requests": num_requests // 2,
            "generated_tokens": int(steady_gen_tps * 5.0),
            "requests_per_second": num_requests / 2 / 5.0,
            "generated_tokens_per_second": steady_gen_tps,
            "mean_resident_requests": float(max_num_seqs),
        }
    )
    return {
        "steady_state": steady_state,
        "backend_stats": None,
        "completed_requests": num_requests,
        "config": {
            "backend": backend,
            "refill": refill,
            "exit_threshold": exit_threshold,
            "workload_name": workload_name,
            "workload_path": workload_path,
            "num_requests": num_requests,
            "output_total_tokens": 10,
            "max_num_batched_tokens": 32,
            "max_recurrent_depth": 4,
            "num_blocks": num_blocks,
            "model": "test-model",
            "attn_implementation": "paged|flash_attention_3",
            "block_size": 16,
            "max_model_len": 128,
            "cdb_kv_policy": "single",
            "min_exit_step": 2,
            "max_num_seqs": max_num_seqs,
            "min_coda_batch_size": min_coda_batch_size,
        },
        "generated_tokens": 10,
        "generated_tokens_per_second": gen_tps,
        "flop_bound_speedup": bound,
        "stage_flops": {"f0_params": 0, "fr_params": 100} if stage_flops is None else stage_flops,
        "peak_cuda_memory_allocated_bytes": 1024**3,
        "peak_cuda_memory_reserved_bytes": 1024**3,
        "kv_cache": {"peak_blocks_used_fraction": 0.5},
        "recurrent_steps_per_second": gen_tps * 4,
        "wall_time_s": 10 / gen_tps,
    }


def write_summaries(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def bundle(directory: Path, num_requests: int) -> Path:
    """A minimal on-disk workload bundle (definition + sibling exit-pdf file)."""

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "ouro_alpaca_recur4.exit_pdf.npz").write_bytes(b"pdf-bytes")
    definition = directory / "ouro_alpaca_recur4.json"
    definition.write_text(
        json.dumps(
            {
                "meta": {"dataset": "alpaca", "recur_steps": 4, "sampling_seed": 0, "shuffle_seed": 0},
                "exit_pdf_file": "ouro_alpaca_recur4.exit_pdf.npz",
                "requests": [{"id": str(i), "input_len": 1, "output_len": 1} for i in range(num_requests)],
            }
        ),
        encoding="utf-8",
    )
    return definition


def recorded_bundle(directory: Path) -> Path:
    """An on-disk recorded-PDF bundle whose depths are trivial to derive by hand.

    Every token's PDF is [0, 0.3, 0.3, 0.4], so at threshold 0.2 all tokens exit at depth 2
    and at 0.5 at depth 3; each request's first token counts at full depth 4. With two-token
    outputs the mean depths are 3 and 3.5, so with weightless boundary stages the bound is
    4/3 and 8/7. Prompts are 3 tokens long, so the mean sequence length is 5.
    """

    import numpy as np

    from looped_cdb.benchmarks.workload import Workload

    workload = Workload(
        ids=[str(i) for i in range(4)],
        input_lens=np.full(4, 3, dtype=np.int32),
        output_lens=np.full(4, 2, dtype=np.int32),
        offsets=np.arange(0, 9, 2, dtype=np.int64),
        exit_pdf=np.tile(np.array([0.0, 0.3, 0.3, 0.4], dtype=np.float16), (8, 1)),
        meta={"dataset": "alpaca"},
    )
    directory.mkdir(parents=True, exist_ok=True)
    json_path, _ = workload.save(directory / "ouro_alpaca_recur4.json")
    return json_path
