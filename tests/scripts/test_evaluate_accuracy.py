"""CLI argument-parsing tests (no GPU)."""

import pytest
from evaluate_accuracy import _result_payload, build_arg_parser, build_backend, reject_trained_gate_flags


def test_defaults():
    args = build_arg_parser().parse_args(["--num-blocks", "1024"])
    assert args.task == "gsm8k_cot"
    assert args.backend == "cb"
    assert args.num_fewshot == 8
    assert args.model == "KristianS7/Ouro-1.4B"
    # Both engines page the cache, so the default carries the prefix rather than adding it later.
    assert args.attn_implementation == "paged|flash_attention_3"
    assert args.kv_policy == "depth_indexed"
    assert args.num_blocks == 1024


def test_full_cdb_invocation():
    argv = [
        "--backend",
        "cdb",
        "--task",
        "gsm8k_cot",
        "--num-fewshot",
        "3",
        "--recur-steps",
        "4",
        "--exit-gate-type",
        "lookahead",
        "--exit-gate-path",
        "/gates/lookahead_exit_gate.safetensors",
        "--exit-threshold",
        "0.5",
        "--min-recurrent-steps",
        "2",
        "--limit",
        "20",
        "--summary-output",
        "outputs/ablations/accuracy/sweep.jsonl",
        "--num-blocks",
        "1024",
    ]
    args = build_arg_parser().parse_args(argv)
    assert args.backend == "cdb"
    assert args.num_fewshot == 3
    assert args.recur_steps == 4
    assert args.exit_gate_type == "lookahead"
    assert args.exit_threshold == 0.5
    assert args.min_recurrent_steps == 2
    assert args.limit == 20
    assert args.summary_output == "outputs/ablations/accuracy/sweep.jsonl"


def test_rejects_unknown_backend():
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(["--backend", "vllm"])


def test_removed_kv_policy_names_are_rejected():
    """A retired layout name must fail loudly rather than select a different one."""

    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(["--kv-policy", "ring_2"])


def test_cb_rejects_the_copy_on_exit_policy():
    """cb runs fixed depth, so a last_exited results label would claim routing that cannot fire."""

    args = build_arg_parser().parse_args(["--backend", "cb", "--kv-policy", "last_exited", "--num-blocks", "1024"])
    with pytest.raises(SystemExit, match="never routes on cb"):
        build_backend(args)


def test_parser_defaults_to_automatic_sizing():
    args = build_arg_parser().parse_args([])
    assert args.num_blocks is None
    assert args.mem_fraction_static is None


@pytest.mark.parametrize("flag, value", [("--exit-gate-type", "lookahead"), ("--exit-gate-path", "/g.safetensors")])
def test_huginn_rejects_trained_gate_flags(flag, value):
    """Huginn has no trained gate, so a gate flag would be a silent no-op."""

    args = build_arg_parser().parse_args(
        ["--backend", "cdb", "--recur-steps", "32", flag, value, "--num-blocks", "1024"]
    )
    with pytest.raises(SystemExit, match="trained gates"):
        reject_trained_gate_flags(args)


def test_huginn_accepts_the_shared_exit_threshold():
    """One threshold flag serves both families; Huginn reads it on the convergence scale."""

    args = build_arg_parser().parse_args(
        [
            "--backend",
            "cdb",
            "--recur-steps",
            "32",
            "--exit-threshold",
            "0.28",
            "--num-blocks",
            "1024",
        ]
    )
    reject_trained_gate_flags(args)
    assert args.exit_threshold == 0.28


def _payload_for(argv: list[str], *, served_gate: str | None, decides_before_loop: bool) -> dict:
    """Build a results payload for ``argv`` as if the engine had served ``served_gate``."""

    from types import SimpleNamespace

    args = build_arg_parser().parse_args([*argv, "--num-blocks", "1024"])
    result = SimpleNamespace(
        task="gsm8k_cot", num_fewshot=args.num_fewshot, num_docs=1, metrics={}, exit_depth_counts=None
    )
    backend = SimpleNamespace(
        engine=SimpleNamespace(cache=SimpleNamespace(num_blocks=1024)),
        exit_gate_type=served_gate,
        decides_exit_before_loop=decides_before_loop,
    )
    return _result_payload(result, args, backend)


def test_payload_records_the_served_gate_not_the_requested_one():
    # Omitting the flag serves the checkpoint's own gate, so recording the request would
    # write null for a run that did have a gate.
    payload = _payload_for(["--backend", "cdb", "--recur-steps", "4"], served_gate="preloop", decides_before_loop=True)
    assert payload["exit_gate_type"] == "preloop"
    # A pre-loop gate decides before the loop, so the engine cannot consume it a step late
    # however the flag was set.
    assert payload["delay_gate_consumption"] is False
    assert payload["num_blocks"] == 1024


def test_payload_keeps_delayed_consumption_for_hazard_gates():
    payload = _payload_for(
        ["--backend", "cdb", "--recur-steps", "4", "--exit-gate-type", "same_step"],
        served_gate="early_exit",
        decides_before_loop=False,
    )
    # The normalized name the engine resolved, not the alias the caller passed.
    assert payload["exit_gate_type"] == "early_exit"
    assert payload["delay_gate_consumption"] is True
