"""Shared export plumbing: what a published JSON records about where its numbers came from."""

import json
import os
from pathlib import Path

import pytest
from summary_rows import bundle, recorded_bundle, summary_row, write_summaries

from looped_cdb.benchmarks.exports import (
    bundle_fingerprint,
    export_meta,
    first_output_token_fraction,
    load_cb_stage_profiles,
    prefill_adjusted_speedup,
    require_cb_stage_profile,
)


def _write_cb_profile(
    root: Path,
    *,
    measured_label: str = "benchmark.generate",
    legacy_summary: bool = False,
) -> Path:
    profile_dir = root / "ouro_alpaca_w64"
    profile_dir.mkdir(parents=True)
    row = summary_row(max_num_seqs=64, num_requests=2000)
    write_summaries(
        profile_dir / ("summary.jsonl" if legacy_summary else "cb_full_summary.jsonl"),
        [row],
    )
    analysis = {
        "measured_label": measured_label,
        "stages": {
            "cb.compute.prefill": {"100us": {"gpu_launched_s": 20.0}},
            "cb.compute.decode": {"100us": {"gpu_launched_s": 80.0}},
        },
    }
    (profile_dir / "cb_full_analysis.json").write_text(json.dumps(analysis), encoding="utf-8")
    return profile_dir


def test_cb_stage_profiles_pair_full_window_gpu_times_with_the_benchmark_config(tmp_path: Path) -> None:
    profile_dir = _write_cb_profile(tmp_path)

    profiles = load_cb_stage_profiles(tmp_path)
    profile = profiles[("test-model", "alpaca", 64)]

    assert profile.source == (profile_dir / "cb_full_analysis.json").as_posix()
    assert profile.summary_source == (profile_dir / "cb_full_summary.jsonl").as_posix()
    assert profile.to_json_dict()["measured_label"] == "benchmark.generate"
    assert profile.num_requests == 2000
    assert profile.prefill_gpu_s == 20.0
    assert profile.decode_gpu_s == 80.0
    assert profile.prefill_fraction == 0.2
    assert prefill_adjusted_speedup(2.0, profile.prefill_fraction) == pytest.approx(5 / 3)


def test_cb_stage_profiles_accept_legacy_shared_summary(tmp_path: Path) -> None:
    profile_dir = _write_cb_profile(tmp_path, legacy_summary=True)

    profile = load_cb_stage_profiles(tmp_path)[("test-model", "alpaca", 64)]

    assert profile.summary_source == (profile_dir / "summary.jsonl").as_posix()


def test_cb_stage_profiles_prefer_the_matching_summary(tmp_path: Path) -> None:
    profile_dir = _write_cb_profile(tmp_path)
    write_summaries(
        profile_dir / "summary.jsonl",
        [summary_row(max_num_seqs=64, num_requests=10000)],
    )

    profile = load_cb_stage_profiles(tmp_path)[("test-model", "alpaca", 64)]

    assert profile.num_requests == 2000
    assert profile.summary_source == (profile_dir / "cb_full_summary.jsonl").as_posix()


def test_cb_stage_profiles_require_one_cb_row_in_the_matching_summary(tmp_path: Path) -> None:
    profile_dir = _write_cb_profile(tmp_path)
    row = summary_row(max_num_seqs=64, num_requests=2000)
    write_summaries(profile_dir / "cb_full_summary.jsonl", [row, row])

    with pytest.raises(ValueError, match="exactly one CB row, found 2"):
        load_cb_stage_profiles(tmp_path)


def test_cb_stage_profiles_require_the_full_generation_window(tmp_path: Path) -> None:
    _write_cb_profile(tmp_path, measured_label="benchmark.steady")

    with pytest.raises(ValueError, match=r"expected 'benchmark\.generate'"):
        load_cb_stage_profiles(tmp_path)


def test_required_cb_profile_must_match_the_throughput_request_count(tmp_path: Path) -> None:
    _write_cb_profile(tmp_path)
    profiles = load_cb_stage_profiles(tmp_path)

    with pytest.raises(ValueError, match="uses 2000 requests, expected 10000"):
        require_cb_stage_profile(
            profiles,
            model="test-model",
            workload="alpaca",
            width=64,
            num_requests=10000,
        )


def test_prefill_adjustment_counts_first_output_work_only_in_prefill() -> None:
    # One first token costs 5 units and one depth-2 decode token costs 3 units.
    # Full-depth output work is 10 units, giving an all-output ratio of 10/8.
    # With 20 seconds of prefill and 80 of decode, ideal runtime is 20 + 80 * 3/5.
    assert prefill_adjusted_speedup(10 / 8, 0.2, first_output_fraction=0.5) == pytest.approx(100 / 68)
    assert prefill_adjusted_speedup(1.0, 0.2, first_output_fraction=0.5) == 1.0
    assert prefill_adjusted_speedup(10 / 8, 1.0, first_output_fraction=0.5) == 1.0


@pytest.mark.parametrize("fraction", [-0.1, 1.0, float("nan")])
def test_prefill_adjustment_rejects_invalid_first_output_fractions(fraction: float) -> None:
    with pytest.raises(ValueError, match="first-output fraction"):
        prefill_adjusted_speedup(1.0, 0.2, first_output_fraction=fraction)


def test_prefill_adjustment_rejects_a_ratio_that_removes_all_decode_work() -> None:
    with pytest.raises(ValueError, match="non-positive decode work"):
        prefill_adjusted_speedup(2.0, 0.2, first_output_fraction=0.5)


def test_first_output_fraction_uses_recorded_tokens() -> None:
    row = summary_row(num_requests=4)
    row["first_token_full_depth_count"] = 3
    assert first_output_token_fraction([row, row]) == 0.3


def test_first_output_fraction_requires_recorded_counts() -> None:
    with pytest.raises(ValueError, match="requires recorded"):
        first_output_token_fraction([summary_row()])


def test_first_output_fraction_rejects_different_replay_populations() -> None:
    row = {**summary_row(), "first_token_full_depth_count": 4}
    with pytest.raises(ValueError, match="matching output-token counts"):
        first_output_token_fraction([row, {**row, "generated_tokens": 12}])


@pytest.mark.parametrize("first,total", [(0, 0), (4, 4), (5, 4), (-1, 4)])
def test_first_output_fraction_requires_a_valid_decode_population(first: int, total: int) -> None:
    row = {**summary_row(), "first_token_full_depth_count": first, "generated_tokens": total}
    with pytest.raises(ValueError, match="requires decode tokens"):
        first_output_token_fraction([row])


def test_export_meta_carries_engine_configuration_and_device(tmp_path: Path) -> None:
    row = summary_row()
    row["device_name"] = "Test GPU"
    summary = write_summaries(tmp_path / "alpaca.jsonl", [row])

    meta = export_meta([summary])

    assert meta["device_name"] == "Test GPU"
    assert meta["num_blocks"] == 16
    assert meta["max_recurrent_depth"] == 4
    assert not any(key.startswith("git_") for key in meta)
    assert "exported_utc" in meta


def test_a_row_without_a_recorded_gpu_publishes_none(tmp_path: Path, capsys) -> None:
    # Summaries written before run-time device stamping carry no GPU.
    summary = write_summaries(tmp_path / "alpaca.jsonl", [summary_row()])

    meta = export_meta([summary])

    assert meta["device_name"] is None
    assert "no row records the GPU" in capsys.readouterr().err


def test_rows_measured_on_several_gpus_record_all_of_them(tmp_path: Path, capsys) -> None:
    rows = [summary_row(), summary_row(backend="cdb", refill=False, exit_threshold=0.2)]
    rows[0]["device_name"] = "GPU A"
    rows[1]["device_name"] = "GPU B"

    meta = export_meta([write_summaries(tmp_path / "alpaca.jsonl", rows)])

    assert meta["device_name"] == ["GPU A", "GPU B"]
    assert "several GPUs" in capsys.readouterr().err


def test_disagreeing_engine_config_records_every_value_and_warns(tmp_path: Path, capsys) -> None:
    # kv_pressure_mode changes how many requests run concurrently, so it changes throughput without
    # any depth mechanism changing. Other disagreements are cosmetic (one checkpoint under two
    # names), so the export reports what differs and leaves the judgement to the reader.
    one = summary_row()
    one["config"]["kv_pressure_mode"] = "reserve"
    two = summary_row(workload_name="sharegpt")
    two["config"]["kv_pressure_mode"] = "recompute"
    paths = [write_summaries(tmp_path / "a.jsonl", [one]), write_summaries(tmp_path / "b.jsonl", [two])]

    meta = export_meta(paths)

    assert meta["kv_pressure_mode"] == ["reserve", "recompute"]
    assert "disagree on 'kv_pressure_mode'" in capsys.readouterr().err


def test_a_run_that_never_recorded_a_config_key_does_not_fabricate_a_disagreement(tmp_path: Path, capsys) -> None:
    # An older run defaulted to recompute but never wrote it down. Reading the absent key back as
    # None would report a config difference that never existed.
    old = summary_row()
    old["config"].pop("kv_pressure_mode", None)
    new = summary_row(workload_name="sharegpt")
    new["config"]["kv_pressure_mode"] = "recompute"
    paths = [write_summaries(tmp_path / "a.jsonl", [old]), write_summaries(tmp_path / "b.jsonl", [new])]

    meta = export_meta(paths)

    assert meta["kv_pressure_mode"] == "recompute"
    assert "do not record 'kv_pressure_mode'" in capsys.readouterr().err


def test_a_key_absent_from_every_run_is_published_as_unknown(tmp_path: Path) -> None:
    summary = write_summaries(tmp_path / "alpaca.jsonl", [summary_row()])

    assert export_meta([summary])["kv_pressure_mode"] is None


def test_fingerprint_is_withheld_when_the_bundle_is_newer_than_the_run(tmp_path: Path) -> None:
    # Bundles are regenerated in place, so a bundle written after the run cannot be the one that
    # was replayed. Publishing its hash would attach false provenance to the numbers.
    summary = write_summaries(tmp_path / "alpaca.jsonl", [summary_row()])
    definition = bundle(tmp_path / "outputs" / "workloads", num_requests=4)
    os.utime(summary, (1_000, 1_000))
    os.utime(definition, (2_000, 2_000))

    fingerprint = bundle_fingerprint(definition, summary)

    assert fingerprint["fingerprint_trusted"] is False
    assert fingerprint["workload_sha256"] is None
    assert fingerprint["workload_meta"] is None
    assert fingerprint["workload_sha256_at_export"]  # still reported, clearly labelled


def test_fingerprint_is_withheld_when_only_the_exit_schedule_is_newer(tmp_path: Path) -> None:
    # The hash spans the definition and its sibling exit-pdf file, so the trust check must too:
    # re-recording an exit schedule can touch only the npz and leave the definition looking old.
    summary = write_summaries(tmp_path / "alpaca.jsonl", [summary_row()])
    definition = bundle(tmp_path / "outputs" / "workloads", num_requests=4)
    os.utime(definition, (1_000, 1_000))
    os.utime(summary, (2_000, 2_000))
    os.utime(definition.parent / "ouro_alpaca_recur4.exit_pdf.npz", (3_000, 3_000))

    fingerprint = bundle_fingerprint(definition, summary)

    assert fingerprint["fingerprint_trusted"] is False
    assert fingerprint["workload_sha256"] is None


def test_fingerprint_is_published_when_the_bundle_predates_the_run(tmp_path: Path) -> None:
    summary = write_summaries(tmp_path / "alpaca.jsonl", [summary_row()])
    definition = bundle(tmp_path / "outputs" / "workloads", num_requests=4)
    # Both halves of the bundle must predate the run, since the hash covers both.
    os.utime(definition, (1_000, 1_000))
    os.utime(definition.parent / "ouro_alpaca_recur4.exit_pdf.npz", (1_000, 1_000))
    os.utime(summary, (2_000, 2_000))

    fingerprint = bundle_fingerprint(definition, summary)

    assert fingerprint["fingerprint_trusted"] is True
    assert len(fingerprint["workload_sha256"]) == 16
    assert fingerprint["workload_meta"]["shuffle_seed"] == 0
    assert "workload_sha256_at_export" not in fingerprint


def test_a_recorded_bundle_is_fingerprinted_whole(tmp_path: Path) -> None:
    # The recorded bundles the paper replays are a definition plus an npz of exit PDFs; both go
    # into the hash, so a re-recording is visible even when the definition is untouched.
    summary = write_summaries(tmp_path / "alpaca.jsonl", [summary_row()])
    definition = recorded_bundle(tmp_path / "outputs" / "workloads")
    for path in (definition, *definition.parent.glob("*.npz")):
        os.utime(path, (1_000, 1_000))
    os.utime(summary, (2_000, 2_000))

    fingerprint = bundle_fingerprint(definition, summary)

    assert fingerprint["fingerprint_trusted"] is True
    assert fingerprint["full_num_requests"] == 4
