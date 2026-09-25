"""Deriving the paper's workload-distribution data from recorded bundles."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from exporters.export_workload_results import (
    MODEL_SETTINGS,
    WORKLOADS,
    build_exit_distributions,
    bundle_provenance,
    exit_counts,
    length_quantiles,
    main,
)

from looped_cdb.benchmarks.workload import Workload

OURO_DEPTH = MODEL_SETTINGS["ouro"][0]


def _workload(pdf_rows: list[list[float]], output_lens: list[int], meta: dict | None = None) -> Workload:
    offsets = np.zeros(len(output_lens) + 1, dtype=np.int64)
    np.cumsum(np.asarray(output_lens, dtype=np.int64), out=offsets[1:])
    return Workload(
        ids=[f"r{i}" for i in range(len(output_lens))],
        input_lens=np.full(len(output_lens), 4, dtype=np.int32),
        output_lens=np.asarray(output_lens, dtype=np.int32),
        offsets=offsets,
        exit_pdf=np.asarray(pdf_rows, dtype=np.float16),
        meta=meta or {},
    )


def test_exit_counts_never_place_tokens_below_the_minimum_exit_step() -> None:
    # A token whose gate would fire at depth 1 cannot exit before R_min=2, so depth 1 stays empty.
    workload = _workload([[1.0, 0.0, 0.0, 0.0]] * 3, output_lens=[3])

    counts = exit_counts(workload, threshold=0.5)

    assert counts[0] == 0
    assert sum(counts) == 3


def test_exit_counts_partition_every_output_token() -> None:
    workload = _workload(
        [[0.9, 0.1, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5], [0.1, 0.1, 0.1, 0.7]],
        output_lens=[3],
    )

    for threshold in (0.01, 0.5, 0.7):
        counts = exit_counts(workload, threshold=threshold)
        assert len(counts) == OURO_DEPTH
        assert sum(counts) == 3  # every token lands in exactly one depth


def test_a_higher_threshold_never_makes_tokens_exit_earlier() -> None:
    # Raising the gate threshold demands more cumulative probability, so mass moves deeper.
    rng = np.random.default_rng(0)
    pdf = rng.dirichlet(np.ones(OURO_DEPTH), size=64)
    workload = _workload(pdf.tolist(), output_lens=[64])

    def mean_depth(threshold: float) -> float:
        counts = np.asarray(exit_counts(workload, threshold), dtype=float)
        return float((counts * np.arange(1, OURO_DEPTH + 1)).sum() / counts.sum())

    depths = [mean_depth(t) for t in (0.01, 0.2, 0.5, 0.7)]
    assert depths == sorted(depths)


def test_bundle_provenance_reports_the_pool_a_subsample_was_drawn_from() -> None:
    subsampled = _workload([[0.25] * 4] * 2, output_lens=[2], meta={"sampled_from": 500})
    assert bundle_provenance(subsampled) == {"num_requests": 1, "sampled_from": 500}

    # A bundle recorded before sampled_from existed is its own pool, not an unknown one.
    whole = _workload([[0.25] * 4] * 2, output_lens=[2])
    assert bundle_provenance(whole) == {"num_requests": 1, "sampled_from": 1}


def test_length_quantiles_span_the_observed_range() -> None:
    values = length_quantiles(np.array([10, 20, 30, 40]), np.array([0.0, 0.5, 1.0]))
    assert values == [10.0, 25.0, 40.0]


def _bundles() -> dict[str, list[tuple[str, Workload]]]:
    bundles: dict[str, list[tuple[str, Workload]]] = {"ouro": [], "huginn": []}
    for label, dataset, _ in WORKLOADS:
        ouro_meta = {
            "dataset": dataset,
            "model_family": "ouro",
            "max_model_len": 16384,
            "depth_defaults": {"min_exit_step": 2, "exit_delay_steps": 0},
        }
        bundles["ouro"].append((label, _workload([[0.0, 0.5, 0.0, 0.5]], [1], ouro_meta)))
        huginn = Workload(
            ids=["r0"],
            input_lens=np.array([4]),
            output_lens=np.array([1]),
            offsets=np.array([0, 1]),
            exit_pdf=None,
            exit_values=np.array([[1.0, 0.2] + [0.0] * 14], dtype=np.float16),
            meta={
                "dataset": dataset,
                "model_family": "huginn",
                "max_model_len": 16384,
                "depth_defaults": {"min_exit_step": 2, "exit_delay_steps": 1},
            },
        )
        bundles["huginn"].append((label, huginn))
    return bundles


def test_current_heatmaps_include_both_models_and_arxiv() -> None:
    exported = build_exit_distributions(_bundles())
    assert set(exported["models"]) == {"ouro", "huginn"}
    for model in exported["models"].values():
        assert [p["label"] for p in model["workloads"]] == ["ShareGPT", "Alpaca", "ArXiv"]
        for panel in model["workloads"]:
            for point in panel["points"]:
                assert sum(point["exit_counts"]) == 1
                assert sum(point["exit_fractions"]) == 1
    huginn = exported["models"]["huginn"]["workloads"][0]["points"]
    assert huginn[0]["exit_counts"][2] == 1
    assert huginn[-1]["exit_counts"][-1] == 1


def test_missing_arxiv_is_rejected() -> None:
    bundles = _bundles()
    bundles["huginn"].pop()
    with pytest.raises(ValueError, match="requires ShareGPT, Alpaca, and ArXiv"):
        build_exit_distributions(bundles)


def test_wrong_model_trajectory_is_rejected() -> None:
    bundles = _bundles()
    bundles["ouro"] = bundles["huginn"]
    with pytest.raises(ValueError, match="expected model_family 'ouro'"):
        build_exit_distributions(bundles)


def test_cli_exports_all_workloads_without_accuracy_file(tmp_path: Path) -> None:
    directory = tmp_path / "bundles"
    directory.mkdir()
    bundles = _bundles()
    for model, (depth, _, _) in MODEL_SETTINGS.items():
        for (_, dataset, suffix), (_, workload) in zip(WORKLOADS, bundles[model], strict=True):
            workload.save(directory / f"{model}_{dataset}_recur{depth}{suffix}.json")
    target = tmp_path / "exports"
    main(["--workload-dir", str(directory), "--out-dir", str(target)])
    lengths = json.loads((target / "workload_length_distributions.json").read_text())
    exits = json.loads((target / "workload_exit_distributions.json").read_text())
    assert [p["label"] for p in lengths["workloads"]] == ["ShareGPT", "Alpaca", "ArXiv"]
    assert set(exits["models"]) == {"ouro", "huginn"}


def test_cli_missing_input_does_not_publish_partial_exports(tmp_path: Path) -> None:
    target = tmp_path / "exports"
    with pytest.raises(FileNotFoundError):
        main(["--workload-dir", str(tmp_path), "--out-dir", str(target)])
    assert not target.exists()


def test_paired_bundles_require_matching_request_order() -> None:
    bundles = _bundles()
    bundles["huginn"][0][1].ids = ["different"]

    with pytest.raises(ValueError, match="different request IDs or replay order"):
        build_exit_distributions(bundles)


def test_recorded_depth_policy_is_validated() -> None:
    bundles = _bundles()
    bundles["huginn"][0][1].meta["depth_defaults"] = {"min_exit_step": 2, "exit_delay_steps": 0}

    with pytest.raises(ValueError, match="depth_defaults"):
        build_exit_distributions(bundles)
