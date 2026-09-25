"""Promotion of open-loop serving summaries into the paper's committed serving JSON."""

import json
from pathlib import Path
from typing import Any

import pytest
from exporters.export_serving_results import (
    build_results,
    build_workload_entry,
    default_output_path,
    infer_model_key,
)


def _summary(
    *,
    backend: str = "cdb",
    refill: bool = True,
    exit_threshold: float | None = 0.5,
    rate_rps: float = 2.0,
    min_coda_batch_size: int = 1,
    max_num_seqs: int = 256,
    norm_lat_mean: float = 0.1,
    model: str = "test-model",
    workload_id: str = "sharegpt-d16-n10-s0-sh0",
    num_requests: int = 8,
    output_mean_tokens: float = 300.0,
) -> dict[str, Any]:
    """One open-loop serving summary row, trimmed to the fields the export reads."""

    return {
        "backend_stats": None,
        "completed_requests": 8,
        "config": {
            "backend": backend,
            "refill": refill,
            "exit_threshold": exit_threshold,
            "workload_name": "sharegpt",
            "workload_id": workload_id,
            "num_requests": num_requests,
            "output_mean_tokens": output_mean_tokens,
            "max_num_batched_tokens": 2048,
            "max_num_seqs": max_num_seqs,
            "max_recurrent_depth": 16,
            "min_coda_batch_size": min_coda_batch_size,
            "num_blocks": 2048,
            "request_rate_rps": rate_rps,
            "model": model,
            "attn_implementation": "paged|flash_attention_3",
            "block_size": 16,
            "max_model_len": 4096,
            "cdb_kv_policy": "single",
            "min_exit_step": 1,
        },
        "generated_tokens": 80,
        "generated_tokens_per_second": 400.0,
        "flop_bound_speedup": 2.0,
        "kv_cache": {"peak_blocks_used_fraction": 0.5},
        "peak_cuda_memory_allocated_bytes": 1024**3,
        "peak_cuda_memory_reserved_bytes": 1024**3,
        "recurrent_steps_per_second": 2400.0,
        "request_latency": {
            "num_requests": 8,
            "e2e_s": {"mean": 30.0, "p99": 40.0},
            "norm_e2e_s_per_token": {"mean": norm_lat_mean, "p99": norm_lat_mean * 1.4},
            "queue_s": {"mean": 0.01, "p99": 0.02},
            "tpot_s": {"mean": 0.09, "p99": 0.12},
            "ttft_s": {"mean": 0.2, "p99": 0.3},
        },
        "wall_time_s": 0.2,
    }


def test_min_coda_batch_separates_otherwise_identical_operating_points() -> None:
    rows = [
        _summary(min_coda_batch_size=1, norm_lat_mean=0.12),
        _summary(min_coda_batch_size=32, norm_lat_mean=0.09),
    ]

    entry = build_workload_entry("sharegpt", rows)

    by_min_coda = {point["min_coda_batch_size"]: point["norm_lat_mean_s_per_tok"] for point in entry["points"]}
    assert by_min_coda == {1: 0.12, 32: 0.09}


def test_mixed_decode_widths_are_rejected() -> None:
    rows = [
        _summary(max_num_seqs=8),
        _summary(max_num_seqs=64),
    ]

    with pytest.raises(ValueError, match=r"sharegpt panel mixes max_num_seqs values: 8, 64"):
        build_workload_entry("sharegpt", rows)


def _write_summaries(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("KristianS7/Ouro-1.4B", "ouro"),
        ("checkpoint-owner/Ouro-1.4B", "ouro"),
        ("tomg-group-umd/huginn-0125", "huginn"),
    ],
)
def test_model_family_is_inferred_from_supported_checkpoint_ids(model: str, expected: str) -> None:
    assert infer_model_key([_summary(model=model)]) == expected


def test_compatible_ouro_checkpoint_aliases_can_share_an_export() -> None:
    rows = [
        _summary(model="KristianS7/Ouro-1.4B"),
        _summary(rate_rps=3.0, model="checkpoint-owner/Ouro-1.4B"),
    ]

    assert infer_model_key(rows) == "ouro"


def test_owner_does_not_make_an_unsupported_model_valid() -> None:
    rows = [_summary(model="KristianS7/Ouro-2.6B")]
    with pytest.raises(ValueError, match="unknown serving model"):
        infer_model_key(rows)


def test_family_detection_preserves_recorded_model_provenance(tmp_path: Path) -> None:
    path = tmp_path / "summary.jsonl"
    model = "checkpoint-owner/Ouro-1.4B"
    _write_summaries(path, [_summary(model=model)])
    before = path.read_bytes()
    result = build_results([path])
    assert result["meta"]["model"] == model
    assert path.read_bytes() == before


def test_mixed_model_families_are_rejected(tmp_path: Path) -> None:
    summary = tmp_path / "mixed.jsonl"
    _write_summaries(
        summary,
        [
            _summary(model="KristianS7/Ouro-1.4B"),
            _summary(rate_rps=3.0, model="tomg-group-umd/huginn-0125"),
        ],
    )

    with pytest.raises(ValueError, match=r"serving export mixes model families: \['huginn', 'ouro'\]"):
        build_results([summary])


@pytest.mark.parametrize("model_key", ["ouro", "huginn"])
def test_export_records_model_key_and_uses_model_specific_default(tmp_path: Path, model_key: str) -> None:
    model = {
        "ouro": "KristianS7/Ouro-1.4B",
        "huginn": "tomg-group-umd/huginn-0125",
    }[model_key]
    summary = tmp_path / f"{model_key}.jsonl"
    _write_summaries(summary, [_summary(model=model)])

    results = build_results([summary])

    assert results["meta"]["model_key"] == model_key
    assert default_output_path(model_key) == Path(f"docs/paper/figs/serving/serving_{model_key}.json")


def test_mixed_workload_bundle_ids_are_rejected() -> None:
    rows = [
        _summary(workload_id="sharegpt-a"),
        _summary(rate_rps=3.0, workload_id="sharegpt-b"),
    ]

    with pytest.raises(ValueError, match="mixes workload bundle IDs"):
        build_workload_entry("sharegpt", rows)


def test_output_populations_are_preserved_by_request_count() -> None:
    rows = [
        _summary(num_requests=600, output_mean_tokens=280.0),
        _summary(rate_rps=3.0, num_requests=900, output_mean_tokens=300.0),
    ]

    entry = build_workload_entry("sharegpt", rows)

    assert entry["output_populations"] == [
        {"num_requests": 600, "output_mean_tokens": 280.0},
        {"num_requests": 900, "output_mean_tokens": 300.0},
    ]
