"""Shared code for exporting raw benchmark summaries into the paper's committed JSON.

This module fingerprints replayed workloads and records shared engine settings and devices.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from looped_cdb.benchmarks.flop_bound import StageFlops, stage_flops_from_json
from looped_cdb.benchmarks.summaries import load_summaries
from looped_cdb.benchmarks.workload import Workload

# Sentinel for "this row never recorded the key", which is not the same as a recorded ``None``.
_MISSING = object()

type CbProfileKey = tuple[str, str, int]


@dataclass(frozen=True)
class CbStageProfile:
    """Whole-run CB GPU time used to adjust a decode-only speedup bound for prefill."""

    source: str
    summary_source: str
    model: str
    workload: str
    width: int
    num_requests: int
    prefill_gpu_s: float
    decode_gpu_s: float

    @property
    def prefill_fraction(self) -> float:
        return self.prefill_gpu_s / (self.prefill_gpu_s + self.decode_gpu_s)

    def to_json_dict(self) -> dict[str, Any]:
        """The timing and source files needed to reproduce one adjusted bound."""

        return {
            "source": self.source,
            "summary_source": self.summary_source,
            "measured_label": "benchmark.generate",
            "width": self.width,
            "num_requests": self.num_requests,
            "prefill_gpu_s": self.prefill_gpu_s,
            "decode_gpu_s": self.decode_gpu_s,
            "prefill_fraction": self.prefill_fraction,
        }


def _stage_gpu_seconds(analysis: dict[str, Any], analysis_path: Path, label: str) -> float:
    """GPU work attributed to one stage in an Nsight analysis."""

    try:
        value = float(analysis["stages"][label]["100us"]["gpu_launched_s"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{analysis_path} has no valid GPU-launched time for {label}") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{analysis_path} reports invalid GPU-launched time {value!r} for {label}")
    return value


def load_cb_stage_profiles(root: Path) -> dict[CbProfileKey, CbStageProfile]:
    """Load the focused CB profiles directly below ``root``.

    Each profile is paired with its sibling benchmark summary. The summary supplies the model,
    workload and launch width, while the Nsight analysis supplies GPU work attributed to prefill
    and decode inside the full generation window.
    """

    profiles: dict[CbProfileKey, CbStageProfile] = {}
    analysis_paths = sorted(root.glob("*/cb_full_analysis.json"))
    if not analysis_paths:
        raise ValueError(f"no CB analyses found directly below {root}")

    for analysis_path in analysis_paths:
        profile_dir = analysis_path.parent
        summary_path = profile_dir / "cb_full_summary.jsonl"
        if not summary_path.is_file():
            summary_path = profile_dir / "summary.jsonl"
        if not summary_path.is_file():
            raise ValueError(f"{analysis_path} has no matching cb_full_summary.jsonl or legacy summary.jsonl")

        cb_rows = [row for row in load_summaries(summary_path) if row["config"].get("backend") == "cb"]
        if len(cb_rows) != 1:
            raise ValueError(f"{summary_path} must contain exactly one CB row, found {len(cb_rows)}")
        row = cb_rows[0]
        model = str(row["config"]["model"])
        workload = str(row["config"]["workload_name"])
        width = int(row["config"]["max_num_seqs"])
        num_requests = int(row["config"]["num_requests"])

        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        if analysis.get("measured_label") != "benchmark.generate":
            raise ValueError(
                f"{analysis_path} measures {analysis.get('measured_label')!r}, expected 'benchmark.generate'"
            )

        source = analysis_path.as_posix()
        profile = CbStageProfile(
            source=source,
            summary_source=summary_path.as_posix(),
            model=model,
            workload=workload,
            width=width,
            num_requests=num_requests,
            prefill_gpu_s=_stage_gpu_seconds(analysis, analysis_path, "cb.compute.prefill"),
            decode_gpu_s=_stage_gpu_seconds(analysis, analysis_path, "cb.compute.decode"),
        )
        key = (model, workload, width)
        if previous := profiles.get(key):
            raise ValueError(f"duplicate CB profiles for {key}: {previous.source} and {source}")
        profiles[key] = profile

    return profiles


def require_cb_stage_profile(
    profiles: dict[CbProfileKey, CbStageProfile],
    *,
    model: str,
    workload: str,
    width: int,
    num_requests: int,
) -> CbStageProfile:
    """Return the profile for one plotted point, with a useful error when it is absent."""

    key = (model, workload, width)
    try:
        profile = profiles[key]
    except KeyError as error:
        raise ValueError(f"missing CB profile for model={model!r}, workload={workload!r}, width={width}") from error
    if profile.num_requests != num_requests:
        raise ValueError(
            f"CB profile for model={model!r}, workload={workload!r}, width={width} uses "
            f"{profile.num_requests} requests, expected {num_requests}"
        )
    return profile


def first_output_token_fraction(rows: list[dict[str, Any]]) -> float:
    """Fraction of output tokens produced by prefill in a shared replay workload."""

    counts = set()
    for row in rows:
        try:
            total = int(row["generated_tokens"])
            first = int(row["first_token_full_depth_count"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("prefill adjustment requires recorded generated and first-output token counts") from error
        if not 0 <= first < total:
            raise ValueError(f"prefill adjustment requires decode tokens, got {first=} and {total=}")
        counts.add((first, total))
    if len(counts) != 1:
        raise ValueError("prefill adjustment requires matching output-token counts across the replay rows")
    first, total = counts.pop()
    return first / total


def prefill_adjusted_speedup(
    decode_speedup: float, prefill_fraction: float, *, first_output_fraction: float = 0.0
) -> float:
    """Adjust a FLOP ratio for fixed prefill GPU time.

    When the ratio includes first output tokens, remove their full-depth work before scaling decode time.
    Their GPU cost is already included in prefill.
    A zero first-output fraction accepts a ratio that already describes decode tokens only.
    """

    if not math.isfinite(decode_speedup) or decode_speedup <= 0:
        raise ValueError(f"decode speedup must be positive and finite, got {decode_speedup!r}")
    if not math.isfinite(prefill_fraction) or not 0 <= prefill_fraction <= 1:
        raise ValueError(f"prefill fraction must lie in [0, 1], got {prefill_fraction!r}")
    if not math.isfinite(first_output_fraction) or not 0 <= first_output_fraction < 1:
        raise ValueError(f"first-output fraction must lie in [0, 1), got {first_output_fraction!r}")
    # In full-depth token-cost units, first tokens occupy this fraction of both workloads.
    decode_work_fraction = (1.0 / decode_speedup - first_output_fraction) / (1.0 - first_output_fraction)
    if decode_work_fraction <= 0:
        raise ValueError("FLOP ratio and first-output fraction imply non-positive decode work")
    return 1.0 / (prefill_fraction + (1.0 - prefill_fraction) * decode_work_fraction)


# Engine knobs copied verbatim from the runs into ``meta``. Panels drawn together should have been
# measured under one configuration, so a disagreement is reported.
SHARED_CONFIG_KEYS = (
    "model",
    "attn_implementation",
    "max_recurrent_depth",
    "num_blocks",
    "block_size",
    "max_num_batched_tokens",
    "max_model_len",
    "cdb_kv_policy",
    # Both halves of the exit policy: the earliest depth a request may leave at, and how many
    # steps its exit is held back past the gate's decision.
    "min_exit_step",
    "exit_delay_steps",
    # Admission policy changes concurrency, so it changes throughput without any depth mechanism
    # changing. The swap-pool size belongs here too: under "offload" it governs how often a
    # request is preempted.
    "kv_pressure_mode",
    "safety_margin",
    "cpu_offload_space",
)


def warn(message: str) -> None:
    """Report something the reader should check, without withholding the export."""

    print(f"warning: {message}", file=sys.stderr)


def shared_meta(summary_paths: list[Path]) -> dict[str, Any]:
    """Engine configuration of the exported runs, with any disagreement reported."""

    seen: dict[str, list[Any]] = {key: [] for key in SHARED_CONFIG_KEYS}
    omitted: dict[str, set[str]] = {}
    for path in summary_paths:
        for row in load_summaries(path):
            for key in SHARED_CONFIG_KEYS:
                value = row["config"].get(key, _MISSING)
                if value is _MISSING:
                    omitted.setdefault(key, set()).add(path.name)
                elif value not in seen[key]:
                    seen[key].append(value)

    meta: dict[str, Any] = {}
    for key, values in seen.items():
        if len(values) > 1:
            warn(f"runs disagree on {key!r}: {values}; check that the panels are still comparable")
        if key in omitted and values:
            warn(f"{sorted(omitted[key])} do not record {key!r}, while other runs report {values[0]!r}")
        meta[key] = values[0] if len(values) == 1 else (values or None)
    return meta


def measured_device(summary_paths: list[Path]) -> Any:
    """The GPU the runs recorded, or ``None`` when they recorded none."""

    devices: list[str] = []
    for path in summary_paths:
        for row in load_summaries(path):
            name = row.get("device_name")
            if name and name not in devices:
                devices.append(name)

    if len(devices) > 1:
        warn(f"rows were measured on several GPUs ({', '.join(devices)}); check that they are comparable")
    if not devices:
        warn("no row records the GPU it was measured on, so the export names no hardware")
        return None
    return devices[0] if len(devices) == 1 else devices


def export_meta(summary_paths: list[Path]) -> dict[str, Any]:
    """Engine configuration and environment shared by every exported panel."""

    meta = shared_meta(summary_paths)
    meta["exported_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    meta["device_name"] = measured_device(summary_paths)
    for module in ("torch", "transformers"):
        with contextlib.suppress(ImportError):
            meta[f"{module}_version"] = __import__(module).__version__
    return meta


def bundle_fingerprint(workload_path: Path, summary_path: Path) -> dict[str, Any]:
    """Content hash and recording metadata of a workload bundle.

    A bundle is a ``<name>.json`` definition plus a sibling ``.npz`` of exit PDFs. Both are
    hashed.
    """

    absent = {
        "workload_sha256": None,
        "workload_meta": None,
        "fingerprint_trusted": False,
        "full_num_requests": None,
    }
    if not workload_path.is_file():
        return absent

    definition = json.loads(workload_path.read_text())
    digest = hashlib.sha256(workload_path.read_bytes())
    pdf_path = workload_path.parent / definition.get("exit_pdf_file", "")
    bundle_mtime = workload_path.stat().st_mtime
    if pdf_path.is_file():
        digest.update(pdf_path.read_bytes())
        # The hash spans both files, so the trust check must too: a re-recorded exit schedule
        # can touch only the npz, leaving the definition's timestamp misleadingly old.
        bundle_mtime = max(bundle_mtime, pdf_path.stat().st_mtime)
    sha = digest.hexdigest()[:16]

    # A bundle written after the run's last summary row cannot be the bundle that run replayed.
    # The converse is not proof, so a passing check only means the hash is plausible.
    trusted = bundle_mtime <= summary_path.stat().st_mtime

    meta = definition.get("meta", {})
    keep = ("dataset", "recur_steps", "exit_gate_type", "sampling_seed", "shuffle_seed", "max_model_len")
    fingerprint: dict[str, Any] = {
        "workload_sha256": sha if trusted else None,
        "workload_meta": {key: meta[key] for key in keep if key in meta} if trusted else None,
        "fingerprint_trusted": trusted,
        "full_num_requests": len(definition.get("requests", [])) or None,
    }
    if not trusted:
        fingerprint["workload_sha256_at_export"] = sha
    return fingerprint


def agreed_config_value(raw_rows: list[dict[str, Any]], key: str, default: Any) -> Any:
    """The single value every row's config records for ``key`` (``default`` when absent).

    Unlike :func:`shared_meta`, a disagreement here is an error.
    """

    values = {row["config"].get(key, default) for row in raw_rows}
    if len(values) != 1:
        raise ValueError(f"runs disagree on {key!r}: {sorted(map(repr, values))}; they replayed different schedules")
    return values.pop()


def recorded_stage_flops(raw_rows: list[dict[str, Any]]) -> StageFlops:
    """The per-stage FLOP weights every row recorded."""

    payloads = {json.dumps(row["stage_flops"], sort_keys=True) for row in raw_rows}
    if len(payloads) != 1:
        raise ValueError(
            f"runs disagree on their per-stage FLOP weights {sorted(payloads)}. They were served with "
            "different layer splits, so their speed-up bounds differ and they cannot share a figure."
        )
    return stage_flops_from_json(json.loads(payloads.pop()))


def replayed_workload(raw_rows: list[dict[str, Any]], workload_path: Path) -> Workload:
    """The workload the runs replayed: the recorded bundle under their truncation and subset."""

    return (
        Workload.load(workload_path)
        .apply_max_model_len(agreed_config_value(raw_rows, "max_model_len", None))
        .take(agreed_config_value(raw_rows, "num_requests", None))
    )


def write_export(results: dict[str, Any], output_path: Path) -> None:
    """Write an export and report the recorded device."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    meta = results["meta"]
    print(f"Wrote {output_path}")
    print(f"  device: {meta.get('device_name') or 'unrecorded'}")
