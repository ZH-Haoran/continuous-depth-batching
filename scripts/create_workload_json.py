r"""Create a recorded workload for looped LM serving benchmarks.

Loads ShareGPT, Alpaca, or ArXiv Summarization and records each output token's
threshold-free exit trajectory. Ouro stores gate PDFs; Huginn stores convergence
values. The bundle contains request lengths in JSON and trajectories in NPZ.
Exit depths can then be derived for any threshold.

Datasets:
    ShareGPT: https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
              (ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json)
    Alpaca:   https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json
    ArXiv:    https://huggingface.co/datasets/ccdv/arxiv-summarization ("document" config;
              one Parquet shard from the Hub's auto-converted export, e.g.
              https://huggingface.co/api/datasets/ccdv/arxiv-summarization/parquet/document/train/0.parquet)

``--limit`` fixes the bundle size: it is applied after every filter, so the bundle holds
exactly that many requests, drawn uniformly from those that survive (the raw records are
shuffled by ``--shuffle-seed`` before parsing). ``meta`` records both the limit and the
pool it was drawn from. ``--filter-models`` names the other models a dataset is recorded for:
a request is kept only if every listed tokenizer also accepts it (minimum lengths and context),
so the bundles of the same dataset hold the same requests in the same replay order and differ
only in their own token counts and exit trajectories.

Example:
    python scripts/create_workload_json.py \
        --dataset sharegpt \
        --data-path path/to/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json \
        --recur-steps 4 \
        --max-model-len 16384 \
        --limit 10000 \
        --output-path path/to/workloads/ouro_sharegpt_recur4_10k.json
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from looped_cdb.benchmarks.datasets import EncodeFn, SampledRequest, filter_by_context, load_dataset
from looped_cdb.benchmarks.exit_recording import record_exit_pdfs
from looped_cdb.eval.model_loading import (
    load_huginn_model,
    load_huginn_tokenizer,
    load_ouro_model,
    load_ouro_tokenizer,
    resolve_model_family,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["sharegpt", "alpaca", "arxiv"])
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Raw dataset: JSON list, JSON Lines, or Parquet file, or a directory of such shards.",
    )
    parser.add_argument(
        "--model",
        default="KristianS7/Ouro-1.4B",
        help="Hugging Face model ID or local looped-model checkpoint.",
    )
    parser.add_argument("--recur-steps", type=int, default=4, help="Recurrent depth budget.")
    parser.add_argument(
        "--model-family",
        default="auto",
        choices=["auto", "ouro", "huginn"],
        help="Looped model family; 'auto' reads model_type from the checkpoint config.",
    )
    parser.add_argument("--exit-gate-type", default=None, help="Ouro gate type; omit to use its built-in gate.")
    parser.add_argument("--exit-gate-path", default=None, help="Ouro trained-gate checkpoint path.")
    parser.add_argument(
        "--attn-impl", default="flash_attention_3", help="Dense attention impl (not the paged variant)."
    )
    parser.add_argument(
        "--filter-models",
        nargs="*",
        default=[],
        help="Other models this dataset is recorded for; a request is kept only if their tokenizers "
        "also pass the length and context filters, so the bundles hold the same requests.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-prompt-len", type=int, default=4)
    parser.add_argument("--min-output-len", type=int, default=4)
    parser.add_argument(
        "--max-model-len",
        type=int,
        required=True,
        help="Serving context length (vLLM-style max_model_len); drop requests whose prompt+output exceeds it.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Record exactly this many requests, sampled after every filter; omit to record all. "
        "Fails if fewer survive filtering.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=0,
        help="Seed for sampling under --limit (<0 keeps file order for sampling) and for the final "
        "replay-order shuffle, which is always applied so the bundle is never length-ordered.",
    )
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384, help="Padded-token budget per microbatch.")
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument(
        "--output-path", type=Path, required=True, help="Destination <name>.json (npz written alongside)."
    )
    args = parser.parse_args()
    # Reject here rather than inside select_requests, which only runs after the whole dataset has
    # been tokenized. Whether the limit *exceeds* the survivors is only knowable after filtering.
    if args.limit is not None and args.limit < 1:
        parser.error(f"--limit must be >= 1, got {args.limit}")
    return args


def encoder_for(model_id: str, family: str) -> EncodeFn:
    """The tokenizer of ``model_id`` as an encode function: prompts with special tokens, outputs bare."""

    tokenizer = load_huginn_tokenizer(model_id) if family == "huginn" else load_ouro_tokenizer(model_id)

    def encode(text: str, add_special_tokens: bool) -> list[int]:
        return tokenizer(text, add_special_tokens=add_special_tokens).input_ids

    return encode


def select_requests(
    dataset: str,
    data_path: Path,
    encode: EncodeFn,
    *,
    max_model_len: int,
    min_prompt_len: int,
    min_output_len: int,
    limit: int | None,
    shuffle_seed: int | None,
    filter_encoders: Sequence[EncodeFn] = (),
) -> tuple[list[SampledRequest], int, int]:
    """Tokenize, drop what cannot be served, then cut down to ``limit``.

    Returns the requests to record, how many the filters dropped after tokenization, and the
    size of the pool ``limit`` was drawn from.

    A request survives only if it passes the minimum lengths and the context under ``encode``
    and under every tokenizer in ``filter_encoders``, so bundles recorded for several models
    with the same filter set hold the same requests (each under its own token counts).

    ``limit`` is applied after every filter, so the bundle holds exactly ``limit`` requests.
    Limiting earlier (the tempting shape, since it lets tokenization stop early) hands back
    fewer requests than asked for, by an amount that varies with ``max_model_len``.

    ``load_dataset`` shuffles the raw records before parsing and every filter preserves order,
    so the surviving list is already in uniformly random order and a prefix of it is a uniform
    sample without replacement. With ``shuffle_seed=None`` the file's own order survives and a
    prefix is *not* a random sample.
    """

    def accepted_ids(encoder: EncodeFn) -> set[str]:
        # Same shuffle as the recording pass: a dataset without record ids is keyed by position.
        samples = load_dataset(
            dataset,
            data_path,
            encoder,
            min_prompt_len=min_prompt_len,
            min_output_len=min_output_len,
            shuffle_seed=shuffle_seed,
        )
        return {sample.id for sample in filter_by_context(samples, max_model_len)[0]}

    samples = load_dataset(
        dataset,
        data_path,
        encode,
        min_prompt_len=min_prompt_len,
        min_output_len=min_output_len,
        shuffle_seed=shuffle_seed,
    )
    tokenized = len(samples)
    samples, _ = filter_by_context(samples, max_model_len)
    for encoder in filter_encoders:
        keep = accepted_ids(encoder)
        samples = [sample for sample in samples if sample.id in keep]
    dropped = tokenized - len(samples)
    sampled_from = len(samples)
    if limit is not None:
        if limit < 1:
            raise ValueError(f"--limit must be >= 1, got {limit}")
        if limit > sampled_from:
            # A bundle quietly holding fewer requests than its name claims would misreport the
            # workload that every downstream throughput number is measured against.
            raise ValueError(f"asked for {limit} requests but only {sampled_from} survive filtering")
        samples = samples[:limit]
    return samples, dropped, sampled_from


def main() -> None:
    args = parse_args()

    family = resolve_model_family(args.model) if args.model_family == "auto" else args.model_family
    encode = encoder_for(args.model, family)
    filter_encoders = [encoder_for(model_id, resolve_model_family(model_id)) for model_id in args.filter_models]

    # ``sampling_seed`` shuffles which records survive ``--limit`` (``<0`` keeps file order for
    # sampling). ``order_seed`` shuffles the final replay order and is always applied, so the
    # saved bundle never inherits the source dataset's ordering even if the dataset is sorted or
    # ``--limit`` is not used. Both are recorded in ``meta`` for reproducibility.
    sampling_seed = None if args.shuffle_seed < 0 else args.shuffle_seed
    order_seed = args.shuffle_seed if args.shuffle_seed >= 0 else 0
    try:
        requests, dropped, sampled_from = select_requests(
            args.dataset,
            args.data_path,
            encode,
            max_model_len=args.max_model_len,
            min_prompt_len=args.min_prompt_len,
            min_output_len=args.min_output_len,
            limit=args.limit,
            shuffle_seed=sampling_seed,
            filter_encoders=filter_encoders,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if dropped:
        print(
            f"Dropped {dropped} tokenized requests (context of {args.max_model_len} tokens"
            + (f", filters of {args.filter_models}" if args.filter_models else "")
            + ")",
            flush=True,
        )
    if not requests:
        raise SystemExit("no requests survived filtering; check the dataset path, min-length and context settings")
    drawn = f" (sampled from {sampled_from})" if args.limit is not None else ""
    print(f"Recording {len(requests)} {args.dataset} requests{drawn} from {args.data_path}", flush=True)

    if family == "huginn":
        # Huginn has no gate, so the trajectory is recorded from a zeroed state to
        # keep the bundle reproducible; upstream draws the state per forward.
        model = load_huginn_model(
            args.model,
            recur_steps=args.recur_steps,
            state_init="zero",
            attn_impl=args.attn_impl,
            device=args.device,
        )
    else:
        model = load_ouro_model(
            args.model,
            recur_steps=args.recur_steps,
            exit_gate_type=args.exit_gate_type,
            exit_gate_path=args.exit_gate_path,
            attn_impl=args.attn_impl,
            device=args.device,
        )

    meta = {
        "dataset": args.dataset,
        "data_path": str(args.data_path),
        "model": args.model,
        "model_family": family,
        "recur_steps": args.recur_steps,
        "exit_gate_type": args.exit_gate_type,
        "exit_gate_path": args.exit_gate_path,
        "min_prompt_len": args.min_prompt_len,
        "min_output_len": args.min_output_len,
        "max_model_len": args.max_model_len,
        "filter_models": list(args.filter_models),
        "sampling_seed": sampling_seed,
        "shuffle_seed": order_seed,
        # The bundle holds ``limit`` requests drawn uniformly from the ``sampled_from`` that
        # survived filtering. ``limit`` is null when the whole pool was recorded, in which case
        # the bundle's size is ``sampled_from``. Both are needed to state what it represents.
        "limit": args.limit,
        "sampled_from": sampled_from,
        # Defaults used when deriving served depths from the recorded trajectory.
        # Ouro gate depths are floored at two and require no replay delay.
        # Huginn cannot satisfy its convergence criterion at step one; delayed consumption
        # therefore gives a minimum served depth of three.
        "depth_defaults": (
            {"min_exit_step": 2, "exit_delay_steps": 1}
            if family == "huginn"
            else {"min_exit_step": 2, "exit_delay_steps": 0}
        ),
    }
    if family == "huginn":
        from looped_cdb.benchmarks.huginn_exit_recording import record_workload
        from looped_cdb.models.huginn.exit_gates import latent_diff

        workload = record_workload(
            model,
            requests,
            criterion=latent_diff,
            max_depth=args.recur_steps,
            meta=meta,
        )
    else:
        workload = record_exit_pdfs(
            model,
            requests,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_batch_size=args.max_batch_size,
            device=args.device,
            meta=meta,
            progress=True,
        )
    # Always randomize the replay order so a sorted or grouped source dataset cannot produce a
    # length-ordered bundle (which starves scheduler concurrency and biases prefix sampling).
    workload = workload.shuffle(order_seed)

    total_output_tokens = int(workload.output_lens.sum())
    json_path, pdf_path = workload.save(args.output_path)
    print(
        f"Wrote {workload.num_requests} requests / {total_output_tokens} output tokens "
        f"(total_steps={workload.total_steps}) to:\n  {json_path}\n  {pdf_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
