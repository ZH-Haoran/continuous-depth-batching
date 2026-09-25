"""Promotion of raw GSM8k accuracy sweeps into the paper's committed accuracy JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from exporters.export_accuracy_results import build_model, build_results, kv_layouts


@pytest.fixture(autouse=True)
def _load_default_huginn_config(monkeypatch: pytest.MonkeyPatch) -> None:
    from looped_cdb.models.huginn import HuginnConfig

    monkeypatch.setattr(HuginnConfig, "from_pretrained", classmethod(lambda cls, _checkpoint: cls()))


def _row(**overrides: Any) -> dict[str, Any]:
    row = {
        "task": "gsm8k_cot",
        "model": "tomg-group-umd/huginn-0125",
        "backend": "cb",
        "recur_steps": 16,
        "kv_policy": "single",
        "num_fewshot": 8,
        "num_docs": 1319,
        "metrics": {"flexible-extract": 0.3, "strict-match": 0.29},
    }
    return {**row, **overrides}


def _gated(threshold: float, *, delayed: bool = True, **overrides: Any) -> dict[str, Any]:
    gated = {
        "backend": "cdb",
        "delay_gate_consumption": delayed,
        "refill": True,
        "min_recurrent_steps": None,
        "exit_gate_type": None,
        "exit_threshold": threshold,
        "exit_depth_counts": {"8": 10},
        "mean_exit_depth": 8.0,
    }
    return _row(**{**gated, **overrides})


def _sweep(tmp_path: Path, rows: list[dict[str, Any]], name: str = "sweep.jsonl") -> Path:
    path = tmp_path / name
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_every_measured_layout_becomes_its_own_fixed_curve() -> None:
    """The figure picks one curve and one depth range; the rest is measured backup."""

    rows = [
        _row(recur_steps=8),
        _row(recur_steps=16),
        # Beyond the budget the gated rows serve, so nothing but the export can report it.
        _row(recur_steps=32, metrics={"flexible-extract": 0.33, "strict-match": 0.32}),
        _row(recur_steps=16, kv_policy="depth_indexed", metrics={"flexible-extract": 0.31, "strict-match": 0.3}),
        _row(recur_steps=8, kv_policy="depth_indexed", metrics={"flexible-extract": 0.08, "strict-match": 0.07}),
        _gated(0.1),
    ]
    entry = build_model(rows, "huginn")

    fixed = {variant["name"]: variant for variant in entry["variants"] if variant["kind"] == "fixed"}
    assert sorted(fixed) == ["fixed_depth_depth_indexed", "fixed_depth_single"]
    assert [point["depth"] for point in fixed["fixed_depth_single"]["points"]] == [8, 16, 32]
    assert [point["depth"] for point in fixed["fixed_depth_depth_indexed"]["points"]] == [8, 16]
    assert fixed["fixed_depth_depth_indexed"]["kv_policy"] == "depth_indexed"


def test_gated_curves_split_by_kv_layout() -> None:
    """One policy measured on two layouts is two curves, each anchored on its own layout."""

    rows = [
        _row(recur_steps=16),
        _row(recur_steps=16, kv_policy="depth_indexed", metrics={"flexible-extract": 0.31, "strict-match": 0.3}),
        _gated(0.1),
        _gated(0.1, kv_policy="last_exited", metrics={"flexible-extract": 0.22, "strict-match": 0.21}),
    ]
    entry = build_model(rows, "huginn")

    gated = {variant["name"]: variant for variant in entry["variants"] if variant["kind"] == "gated"}
    assert sorted(gated) == ["delayed_last_exited", "delayed_single"]
    # last_exited never routes on cb, so its anchor is the depth_indexed run at the budget.
    anchors = {
        name: next(point["acc"] for point in variant["points"] if point["threshold"] == 0.0)
        for name, variant in gated.items()
    }
    assert anchors == {"delayed_single": 0.3, "delayed_last_exited": 0.31}


def test_ouro_anchors_its_curve_on_the_layout_cb_actually_serves() -> None:
    """Copy-on-exit never routes at fixed depth, so the fixed rows record depth_indexed."""

    rows = [
        _row(model="KristianS7/Ouro-1.4B", recur_steps=4, kv_policy="depth_indexed", num_fewshot=3),
        _gated(
            0.2,
            model="KristianS7/Ouro-1.4B",
            recur_steps=4,
            kv_policy="last_exited",
            num_fewshot=3,
            exit_gate_type="early_exit",
            exit_depth_counts={"3": 10},
            mean_exit_depth=3.0,
        ),
    ]
    entry = build_model(rows, "ouro")

    assert [variant["name"] for variant in entry["variants"]] == ["fixed_depth_depth_indexed", "gate_last_exited"]
    assert {variant["name"]: variant["kv_policy"] for variant in entry["variants"]} == {
        "fixed_depth_depth_indexed": "depth_indexed",
        "gate_last_exited": "last_exited",
    }


def test_gated_arms_split_by_consumption_timing_and_carry_a_no_exit_anchor() -> None:
    rows = [
        _row(recur_steps=16, metrics={"flexible-extract": 0.33, "strict-match": 0.32}),
        _gated(0.1, delayed=True),
        _gated(0.28, delayed=True),
        _gated(0.1, delayed=False),
    ]
    entry = build_model(rows, "huginn")
    arms = {variant["name"]: variant for variant in entry["variants"] if variant["kind"] == "gated"}

    assert sorted(arms) == ["delayed_single", "immediate_single"]
    # Huginn's criterion decays, so its no-exit anchor is 0.0 and sorts to the front.
    assert [point["threshold"] for point in arms["delayed_single"]["points"]] == [0.0, 0.1, 0.28]
    anchor = arms["delayed_single"]["points"][0]
    assert anchor["acc"] == 0.33
    assert anchor["mean_depth"] == 16.0
    # The anchor keeps the token total but places every token at the budget.
    assert anchor["exit_counts"] == [0] * 15 + [10]


def test_ouro_anchors_at_the_top_of_its_probability_scale() -> None:
    """Ouro's gate accumulates upward, so its no-exit end is 1.0, not 0.0."""

    rows = [
        _row(model="KristianS7/Ouro-1.4B", recur_steps=4, kv_policy="depth_indexed", num_fewshot=3),
        _gated(
            0.2,
            model="KristianS7/Ouro-1.4B",
            recur_steps=4,
            kv_policy="last_exited",
            num_fewshot=3,
            exit_gate_type="early_exit",
            exit_depth_counts={"3": 10},
            mean_exit_depth=3.0,
        ),
    ]
    gate = next(v for v in build_model(rows, "ouro")["variants"] if v["name"] == "gate_last_exited")

    assert [point["threshold"] for point in gate["points"]] == [0.2, 1.0]


def test_arms_of_one_model_may_run_different_exit_floors() -> None:
    """A delayed gate exits a step deeper than its knob, so Ouro's arms set different floors."""

    ouro = {"model": "KristianS7/Ouro-1.4B", "recur_steps": 4, "num_fewshot": 3}
    gated = {"exit_depth_counts": {"3": 10}, "mean_exit_depth": 3.0, "kv_policy": "last_exited"}
    rows = [
        _row(kv_policy="depth_indexed", **ouro),
        _gated(0.2, delayed=False, exit_gate_type="early_exit", min_recurrent_steps=2, **ouro, **gated),
        # The lookahead head is consumed a step late and needs no floor of its own.
        _gated(0.2, exit_gate_type="lookahead", min_recurrent_steps=None, **ouro, **gated),
    ]
    variants = {variant["name"]: variant for variant in build_model(rows, "ouro")["variants"]}

    assert variants["gate_last_exited"]["min_exit_step"] == 2
    assert variants["lookahead_last_exited"]["min_exit_step"] == 1


def test_a_rerun_row_replaces_the_one_it_corrects() -> None:
    """Sweeps append, so re-running a row under the same settings is a correction."""

    rows = [
        _row(recur_steps=16, metrics={"flexible-extract": 0.30, "strict-match": 0.29}),
        _row(recur_steps=16, metrics={"flexible-extract": 0.33, "strict-match": 0.32}),
        _gated(0.1, metrics={"flexible-extract": 0.20, "strict-match": 0.19}),
        _gated(0.1, metrics={"flexible-extract": 0.31, "strict-match": 0.30}),
    ]
    entry = build_model(rows, "huginn")
    variants = {variant["name"]: variant for variant in entry["variants"]}

    assert [point["acc"] for point in variants["fixed_depth_single"]["points"]] == [0.33]
    # The anchor takes the corrected fixed accuracy, and the curve the corrected gated one.
    assert [(point["threshold"], point["acc"]) for point in variants["delayed_single"]["points"]] == [
        (0.0, 0.33),
        (0.1, 0.31),
    ]


def test_consumption_timing_is_not_deduped_away_for_ouro() -> None:
    """Ouro's curve name carries the gate, not the timing, so the key has to carry it."""

    ouro = {"model": "KristianS7/Ouro-1.4B", "recur_steps": 4, "num_fewshot": 3}
    gated = {
        "exit_depth_counts": {"3": 10},
        "mean_exit_depth": 3.0,
        "kv_policy": "last_exited",
        "exit_gate_type": "early_exit",
        "min_recurrent_steps": 2,
    }
    rows = [
        _row(kv_policy="depth_indexed", **ouro),
        _gated(0.2, delayed=False, metrics={"flexible-extract": 0.40, "strict-match": 0.39}, **ouro, **gated),
        _gated(0.2, delayed=True, metrics={"flexible-extract": 0.55, "strict-match": 0.54}, **ouro, **gated),
    ]
    variants = {variant["name"]: variant for variant in build_model(rows, "ouro")["variants"]}
    immediate = variants["gate_last_exited_immediate_refill"]
    delayed = variants["gate_last_exited_delayed_refill"]

    assert [point["acc"] for point in immediate["points"]] == [0.40, 0.3]
    assert [point["acc"] for point in delayed["points"]] == [0.55, 0.3]
    assert immediate["delay_gate_consumption"] is False
    assert delayed["delay_gate_consumption"] is True
    assert immediate["refill"] is True


def test_a_mixed_attention_backend_is_reported_not_fatal(capsys) -> None:
    """The backend is what lets a sweep be topped up elsewhere; accuracy should not depend on it."""

    rows = [
        _row(recur_steps=16, attn_implementation="paged|flash_attention_3"),
        _gated(0.1, attn_implementation="paged|flash_attention_2"),
    ]
    entry = build_model(rows, "huginn")

    assert entry["attn_implementation"] == ["paged|flash_attention_2", "paged|flash_attention_3"]
    assert "mixes attn_implementation" in capsys.readouterr().err


def test_kv_layouts_carry_the_cache_sizes_the_checkpoints_allocate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sizes come from config, so they must reproduce the layouts the engine allocates."""

    from looped_cdb.models.huginn import HuginnConfig

    checkpoint = "tomg-group-umd/huginn-0125"
    loaded: list[str] = []

    def from_pretrained(cls: type[HuginnConfig], model_id: str) -> HuginnConfig:
        loaded.append(model_id)
        return cls()

    monkeypatch.setattr(HuginnConfig, "from_pretrained", classmethod(from_pretrained))
    rows = [
        _row(recur_steps=32, kv_policy=policy, metrics={"flexible-extract": acc, "strict-match": acc})
        for policy, acc in (("depth_indexed", 0.34), ("single", 0.34), ("first_then_shared", 0.35))
    ]
    layouts = {entry["kv_policy"]: entry for entry in kv_layouts(rows, "huginn", checkpoint, 32)}

    assert loaded == [checkpoint]

    # The values published in the accuracy appendix, in KiB per token.
    assert layouts["depth_indexed"]["kib_per_token"] == pytest.approx(2722.5)
    assert layouts["single"]["kib_per_token"] == pytest.approx(165.0)
    assert layouts["first_then_shared"]["kib_per_token"] == pytest.approx(247.5)
    assert layouts["depth_indexed"]["slots"] == 32


def test_a_missing_kv_layout_is_reported_and_skipped(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from looped_cdb.models.huginn import HuginnConfig

    monkeypatch.setattr(HuginnConfig, "from_pretrained", classmethod(lambda cls, _checkpoint: cls()))
    rows = [_row(recur_steps=16, kv_policy="single")]

    layouts = kv_layouts(rows, "huginn", "tomg-group-umd/huginn-0125", 16)

    assert [entry["kv_policy"] for entry in layouts] == ["single"]
    assert "first_then_shared" in capsys.readouterr().err


def test_a_missing_fixed_run_at_the_budget_is_an_error() -> None:
    with pytest.raises(SystemExit, match="no fixed-depth run at"):
        build_model([_row(recur_steps=8), _gated(0.1)], "huginn")


def test_rows_measured_under_different_settings_do_not_share_a_curve() -> None:
    with pytest.raises(SystemExit, match="rows disagree on 'recur_steps'"):
        build_model([_row(recur_steps=16), _gated(0.1), _gated(0.1, recur_steps=32)], "huginn")


def test_exit_depths_beyond_the_budget_are_rejected() -> None:
    rows = [_row(recur_steps=16), _gated(0.1, exit_depth_counts={"20": 10})]
    with pytest.raises(SystemExit, match="outside budget"):
        build_model(rows, "huginn")


def test_document_carries_both_models_and_their_evaluation_settings(tmp_path: Path) -> None:
    huginn = _sweep(tmp_path, [_row(recur_steps=16), _gated(0.1)], "huginn.jsonl")
    ouro = _sweep(
        tmp_path,
        [
            _row(model="KristianS7/Ouro-1.4B", recur_steps=4, kv_policy="depth_indexed", num_fewshot=3),
            _gated(
                0.2,
                model="KristianS7/Ouro-1.4B",
                recur_steps=4,
                kv_policy="last_exited",
                num_fewshot=3,
                exit_gate_type="early_exit",
                exit_depth_counts={"3": 10},
                mean_exit_depth=3.0,
            ),
        ],
        "ouro.jsonl",
    )

    document = build_results({"ouro": ouro, "huginn": huginn})

    assert set(document["models"]) == {"ouro", "huginn"}
    assert document["models"]["huginn"]["num_fewshot"] == 8
    assert document["models"]["ouro"]["num_fewshot"] == 3


def test_rows_with_different_kv_pressure_modes_are_rejected() -> None:
    rows = [
        _row(recur_steps=16, kv_pressure_mode="reserve"),
        _gated(0.1, kv_pressure_mode="recompute"),
    ]

    with pytest.raises(SystemExit, match="kv_pressure_mode"):
        build_model(rows, "huginn")


def test_document_task_is_derived_and_mixed_tasks_are_rejected(tmp_path: Path) -> None:
    huginn = _sweep(tmp_path, [_row(recur_steps=16), _gated(0.1)], "huginn-task.jsonl")
    ouro_rows = [
        _row(model="KristianS7/Ouro-1.4B", recur_steps=4, kv_policy="depth_indexed", num_fewshot=3),
        _gated(
            0.2,
            model="KristianS7/Ouro-1.4B",
            recur_steps=4,
            kv_policy="last_exited",
            num_fewshot=3,
            exit_gate_type="early_exit",
            exit_depth_counts={"3": 10},
            mean_exit_depth=3.0,
        ),
    ]
    ouro = _sweep(tmp_path, ouro_rows, "ouro-task.jsonl")

    assert build_results({"ouro": ouro, "huginn": huginn})["task"] == "gsm8k_cot"

    mixed = _sweep(tmp_path, [{**row, "task": "other"} for row in ouro_rows], "mixed-task.jsonl")
    with pytest.raises(SystemExit, match="disagree on task"):
        build_results({"ouro": mixed, "huginn": huginn})
