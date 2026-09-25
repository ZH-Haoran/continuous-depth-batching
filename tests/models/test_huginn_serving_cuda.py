"""End-to-end GPU checks for Huginn on the paged serving engines.

``tests/test_huginn_parity_cuda.py`` pins a single forward pass against the
upstream release. These tests pin the serving loops around it, on the real
checkpoint: continuous batching must reproduce cache-free greedy decoding token
for token, and the staged continuous-depth engine must run the prelude, a
convergence-gated recurrent core, and the coda end to end with adaptive exit
live. Run on a GPU node via ``shells/pytest_cuda.sh``.
"""

from __future__ import annotations

import os
from collections import Counter

import pytest
import torch

pytestmark = pytest.mark.cuda

MODEL = os.environ.get("HUGINN_MODEL", "tomg-group-umd/huginn-0125")

PROMPTS = (
    "The capital of France is",
    "Q: What is 2 + 2?\nA:",
    "The three primary colors are",
)


def _generated_tokens(output, limit: int) -> list[int]:
    tokens = output.generated_tokens if hasattr(output, "generated_tokens") else output
    return list(tokens)[:limit]


@torch.no_grad()
def _reference_greedy(model, input_ids: list[int], max_new_tokens: int, steps: int) -> list[int]:
    """Greedy decode with no cache, recomputing the full prefix each step."""

    device = next(model.parameters()).device
    tokens = list(input_ids)
    generated: list[int] = []
    for _ in range(max_new_tokens):
        ids = torch.tensor([tokens], device=device)
        out = model(input_ids=ids, num_steps=steps, past_key_values=None)
        next_id = int(out.logits[0, -1].argmax())
        generated.append(next_id)
        tokens.append(next_id)
    return generated


def test_cb_reproduces_cache_free_greedy_decoding() -> None:
    """Paged continuous batching must be token-identical to cache-free decoding.

    The reference recomputes the whole prefix for every token, so it has no
    cache semantics to get wrong, and recomputing every recurrent step is by
    definition the depth-indexed policy. Paged KV addressing, virtual
    cache-layer indexing, position handling, and chunked prefill all have to be
    right for the token ids to match. The recurrent state is zeroed on both
    sides, and both run in bfloat16, which paged FlashAttention requires.
    """

    from looped_cdb.continuous_batching import ContinuousBatchingEngine
    from looped_cdb.continuous_batching.config import ContinuousBatchingConfig
    from looped_cdb.eval.model_loading import load_huginn_model, load_huginn_tokenizer

    recur_steps = 8
    max_new_tokens = 12
    tokenizer = load_huginn_tokenizer(MODEL)
    prompt_ids = [tokenizer(p)["input_ids"] for p in PROMPTS]

    ref_model = load_huginn_model(
        MODEL,
        recur_steps=recur_steps,
        state_init="zero",
        dtype=torch.bfloat16,
        attn_impl="sdpa",
    )
    reference = [_reference_greedy(ref_model, ids, max_new_tokens, recur_steps) for ids in prompt_ids]
    del ref_model
    torch.cuda.empty_cache()

    cb_model = load_huginn_model(
        MODEL,
        recur_steps=recur_steps,
        state_init="zero",
        dtype=torch.bfloat16,
        attn_impl="paged|flash_attention_3",
    )
    engine = ContinuousBatchingEngine.from_model(
        cb_model,
        cb_config=ContinuousBatchingConfig(num_blocks=512, use_cuda_graph=False),
    )
    outputs = engine.generate_batch(prompt_ids, max_new_tokens=max_new_tokens, eos_token_id=None)

    mismatches = []
    for prompt, ref, out in zip(PROMPTS, reference, outputs, strict=True):
        got = _generated_tokens(out, max_new_tokens)
        if ref != got:
            mismatches.append(f"{prompt!r}: reference {tokenizer.decode(ref)!r} != cb {tokenizer.decode(got)!r}")

    del engine, cb_model
    torch.cuda.empty_cache()
    assert not mismatches, "continuous batching diverged from cache-free greedy decoding:\n" + "\n".join(mismatches)


def test_cdb_runs_the_staged_path_with_adaptive_exit() -> None:
    """The staged engine decodes with the convergence exit live, graphed and eager alike.

    The prelude writes its own KV, the recurrent core exits per token on the
    state-convergence criterion, and the coda writes its KV before the LM head
    samples. Every prompt must produce tokens, and the exit-depth histogram
    must show depths inside the scheduled range with a mean below the maximum,
    which is what depth refill needs to have anything to fill with. The graphed
    engine (the default, which CUDA-graphs the prelude, recurrent, and coda
    stages) and the eager engine each pass the same behavioral checks; no token
    equality is asserted across the two, since their launches hit different GEMM
    shapes and bfloat16 argmax is not bit-stable across shapes.
    """

    from looped_cdb.continuous_depth_batching import ContinuousDepthBatchingConfig, ContinuousDepthBatchingEngine
    from looped_cdb.continuous_depth_batching.adapters import HuginnCDBAdapter
    from looped_cdb.eval.model_loading import load_huginn_model, load_huginn_tokenizer

    max_recurrent_steps = 32
    min_recurrent_steps = 2
    max_new_tokens = 16
    model = load_huginn_model(
        MODEL,
        recur_steps=max_recurrent_steps,
        kv_policy="single",
        # A zeroed recurrent state keeps the graphed and eager runs comparable; the
        # checkpoint's own default draws a random state per forward.
        state_init="zero",
        dtype=torch.bfloat16,
        attn_impl="paged|flash_attention_3",
    )
    tokenizer = load_huginn_tokenizer(MODEL)
    adapter = HuginnCDBAdapter(model)
    prompt_ids = [tokenizer(p)["input_ids"] for p in PROMPTS]

    for use_cuda_graph in (True, False):
        cdb_config = ContinuousDepthBatchingConfig(
            max_recurrent_steps=max_recurrent_steps,
            min_recurrent_steps=min_recurrent_steps,
            exit_threshold=0.03,
            num_blocks=512,
            use_cuda_graph=use_cuda_graph,
        )
        engine = ContinuousDepthBatchingEngine.from_model(model, cdb_config=cdb_config, model_adapter=adapter)
        outputs = engine.generate_batch(prompt_ids, max_new_tokens=max_new_tokens, eos_token_id=tokenizer.eos_token_id)

        for prompt, output in zip(PROMPTS, outputs, strict=True):
            assert len(_generated_tokens(output, max_new_tokens)) > 0, f"empty generation for {prompt!r}"

        histogram = getattr(getattr(engine, "last_stats", None), "exit_depth_histogram", None)
        assert histogram, "engine reported no exit-depth histogram"
        depths = Counter({int(step) + 1: int(count) for step, count in histogram.items()})
        assert all(min_recurrent_steps <= depth <= max_recurrent_steps for depth in depths)
        total = sum(depths.values())
        mean_depth = sum(depth * count for depth, count in depths.items()) / total
        assert mean_depth < max_recurrent_steps, f"no token exited early (mean depth {mean_depth:.2f})"

        if use_cuda_graph:
            assert engine.last_stats.stage_graph_hits + engine.last_stats.stage_graph_captures > 0, (
                "the graphed engine never launched a graphed prelude/coda stage"
            )
        del engine
        torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
