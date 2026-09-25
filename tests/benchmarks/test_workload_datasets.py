"""Tests for dataset loading, PDF-to-depth derivation, batching, and the bundle.

These exercise the CPU-only building blocks of the workload recorder: dataset
parsing/filtering, the output-token PDF slice, the offline threshold->depth
derivation (checked against the model's own torch implementation), microbatch
grouping, and round-tripping a Workload to disk.
"""

from __future__ import annotations

import numpy as np
import pytest

from looped_cdb.benchmarks.datasets import (
    ARXIV_PROMPT_PREFIX,
    ARXIV_PROMPT_SUFFIX,
    SampledRequest,
    filter_by_context,
    iter_alpaca,
    iter_arxiv,
    iter_sharegpt,
    load_dataset,
    read_records,
)
from looped_cdb.benchmarks.exit_recording import group_by_token_budget
from looped_cdb.benchmarks.workload import Workload, depths_from_pdf, slice_output_pdf


def whitespace_encode(text: str, add_special_tokens: bool) -> list[int]:
    """Trivial tokenizer: one id per word, with a leading BOS (id 1) when asked."""

    ids = [10 + (len(word) % 5) for word in text.split()]
    return [1, *ids] if add_special_tokens else ids


# --------------------------------------------------------------------------- datasets


def test_sharegpt_takes_first_human_then_assistant_turn() -> None:
    records = [
        {
            "id": "conv-a",
            "conversations": [
                {"from": "human", "value": "please explain gradient descent to me"},
                {"from": "gpt", "value": "gradient descent walks downhill on the loss"},
                {"from": "human", "value": "ignored follow up turn"},
            ],
        }
    ]
    (sample,) = list(iter_sharegpt(records, whitespace_encode, min_prompt_len=1, min_output_len=1))
    assert sample.id == "conv-a"
    assert sample.prompt_ids[0] == 1  # BOS on the prompt
    assert sample.input_len == 1 + 6  # BOS + six prompt words
    assert sample.output_len == 7  # seven output words, no special token


def test_sharegpt_skips_non_human_opening_and_short_turns() -> None:
    records = [
        {
            "id": "sys-open",
            "conversations": [
                {"from": "system", "value": "you are helpful"},
                {"from": "gpt", "value": "hi there friend"},
            ],
        },
        {"id": "too-short", "conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "yo"}]},
        {"id": "single-turn", "conversations": [{"from": "human", "value": "only one turn here"}]},
    ]
    kept = list(iter_sharegpt(records, whitespace_encode, min_prompt_len=3, min_output_len=3))
    assert kept == []


def test_alpaca_joins_instruction_and_input() -> None:
    records = [
        {
            "instruction": "translate the phrase below",
            "input": "good morning everyone",
            "output": "buenos dias a todos",
        },
        {"instruction": "say hello", "input": "", "output": "hello there general friend"},
    ]
    joined, no_input = list(iter_alpaca(records, whitespace_encode, min_prompt_len=1, min_output_len=1))
    # instruction (4 words) + input (3 words) + BOS = 8 prompt tokens.
    assert joined.input_len == 1 + 4 + 3
    assert no_input.input_len == 1 + 2  # BOS + two instruction words


def test_arxiv_wraps_article_in_a_summarization_request() -> None:
    framing_words = len(f"{ARXIV_PROMPT_PREFIX}{ARXIV_PROMPT_SUFFIX}".split())
    records = [
        {
            "article_id": "1234.5678",
            "article": "we study looped models at length here",
            "abstract": "looped models work",
        },
        {"article": "no abstract for this one", "abstract": ""},
        {"article": "no article id here", "article_id": None, "abstract": "abstract is long enough here"},
        {"article": None, "abstract": "abstract without an article"},
    ]
    kept = list(iter_arxiv(records, whitespace_encode, min_prompt_len=1, min_output_len=1))
    assert [s.id for s in kept] == ["1234.5678", "2"]  # a null id falls back to the position, like Alpaca
    assert kept[0].input_len == 1 + 7 + framing_words and kept[0].output_len == 3
    # The whole prompt, framing included, goes through the tokenizer.
    seen: list[str] = []

    def spy_encode(text: str, add_special_tokens: bool) -> list[int]:
        seen.append(text)
        return whitespace_encode(text, add_special_tokens)

    list(iter_arxiv(records[:1], spy_encode, min_prompt_len=1, min_output_len=1))
    assert seen[0] == f"{ARXIV_PROMPT_PREFIX}we study looped models at length here{ARXIV_PROMPT_SUFFIX}"


def test_read_records_accepts_json_jsonl_parquet_and_directories(tmp_path) -> None:
    import json

    pq = pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa

    records = [{"article": f"article {i}", "abstract": f"abstract {i}"} for i in range(6)]
    # Hub shards sort lexicographically (0, 1, 10, 2): that order is what both models' recordings
    # see, so it is pinned here rather than made numeric.
    (tmp_path / "0.json").write_text(json.dumps(records[:1]))
    (tmp_path / "1.jsonl").write_text("\n".join(json.dumps(r) for r in records[1:3]) + "\n")
    pq.write_table(pa.Table.from_pylist(records[3:5]), tmp_path / "10.parquet")
    pq.write_table(pa.Table.from_pylist(records[5:]), tmp_path / "2.parquet")
    (tmp_path / "notes.txt").write_text("ignored")

    assert read_records(tmp_path / "0.json") == records[:1]
    assert read_records(tmp_path / "1.jsonl") == records[1:3]
    assert read_records(tmp_path / "10.parquet") == records[3:5]
    assert read_records(tmp_path) == records
    loaded = load_dataset("arxiv", tmp_path, whitespace_encode, min_prompt_len=1, min_output_len=1)
    assert [s.id for s in loaded] == [str(i) for i in range(6)]

    (tmp_path / "bad.json").write_text('{"not": "a list"}')
    with pytest.raises(ValueError, match="JSON list"):
        read_records(tmp_path / "bad.json")
    (tmp_path / "shard.parquet.gz").write_bytes(b"\x1f\x8b")
    with pytest.raises(ValueError, match="unsupported dataset file"):
        read_records(tmp_path / "shard.parquet.gz")
    (tmp_path / "sub").mkdir()
    with pytest.raises(ValueError, match="holds no"):
        read_records(tmp_path / "sub")


def test_load_dataset_shuffle_is_deterministic(tmp_path) -> None:
    import json

    records = [
        {
            "id": f"c{i}",
            "conversations": [
                {"from": "human", "value": f"prompt number {i} here"},
                {"from": "gpt", "value": f"reply number {i} indeed"},
            ],
        }
        for i in range(20)
    ]
    path = tmp_path / "sharegpt.json"
    path.write_text(json.dumps(records))

    first = load_dataset("sharegpt", path, whitespace_encode, min_prompt_len=1, min_output_len=1, shuffle_seed=7)
    again = load_dataset("sharegpt", path, whitespace_encode, min_prompt_len=1, min_output_len=1, shuffle_seed=7)

    assert len(first) == 20
    assert [s.id for s in first] == [s.id for s in again]  # deterministic under a fixed seed
    # The seed must actually reorder, otherwise a prefix of the result is not a random sample.
    assert [s.id for s in first] != [f"c{i}" for i in range(20)]


def test_load_dataset_rejects_unknown_dataset(tmp_path) -> None:
    path = tmp_path / "x.json"
    path.write_text("[]")
    with pytest.raises(ValueError, match="unknown dataset"):
        load_dataset("nonsense", path, whitespace_encode)


def test_filter_by_context_drops_over_length_and_counts() -> None:
    samples = [
        SampledRequest(id="ok", prompt_ids=[1, 2, 3], output_ids=[4, 5]),  # seq 5
        SampledRequest(id="at-limit", prompt_ids=[1, 2, 3], output_ids=[4, 5, 6]),  # seq 6, kept (<=)
        SampledRequest(id="too-long", prompt_ids=list(range(5)), output_ids=list(range(3))),  # seq 8, dropped
    ]
    kept, dropped = filter_by_context(samples, max_seq_len=6)
    assert [s.id for s in kept] == ["ok", "at-limit"]
    assert dropped == 1


# --------------------------------------------------------------------------- slicing


def test_slice_output_pdf_uses_decode_off_by_one() -> None:
    # 5 positions, 3 steps; each row marked by its position for easy checking.
    pdf_row = np.arange(5 * 3, dtype=np.float32).reshape(5, 3)
    # prompt_len=2, output_len=3 -> output tokens produced at positions 1, 2, 3.
    sliced = slice_output_pdf(pdf_row, prompt_len=2, output_len=3)
    assert sliced.shape == (3, 3)
    np.testing.assert_array_equal(sliced, pdf_row[1:4])


def test_slice_output_pdf_rejects_out_of_range() -> None:
    pdf_row = np.zeros((4, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="too short"):
        slice_output_pdf(pdf_row, prompt_len=3, output_len=3)  # needs position 4, only 0..3 exist


# --------------------------------------------------------------------------- depths_from_pdf


def _reference_pdf(rng: np.random.Generator, n: int, steps: int) -> np.ndarray:
    raw = rng.random((n, steps)).astype(np.float32)
    return raw / raw.sum(axis=1, keepdims=True)


@pytest.mark.parametrize("threshold", [0.2, 0.5, 0.75, 0.9])
@pytest.mark.parametrize("min_exit_step", [1, 2, 3])
def test_depths_from_pdf_matches_model_qexit(threshold: float, min_exit_step: int) -> None:
    torch = pytest.importorskip("torch")
    from looped_cdb.models.ouro.modeling_ouro import qexit_steps_from_pdf

    rng = np.random.default_rng(0)
    pdf = _reference_pdf(rng, 64, 4)

    ours = depths_from_pdf(pdf, threshold=threshold, min_exit_step=min_exit_step)
    reference = qexit_steps_from_pdf(
        torch.from_numpy(pdf).unsqueeze(0),
        threshold=threshold,
        min_exit_step=min_exit_step,
    )
    # qexit returns 0-indexed steps; ours returns 1-indexed depths.
    np.testing.assert_array_equal(ours, reference.squeeze(0).numpy() + 1)


def test_depths_from_pdf_derives_float16_storage_in_float32() -> None:
    # PDFs are stored as float16; the derivation must accumulate in float32 (not
    # float16), so deriving from the stored float16 array matches deriving from a
    # float32 view of the same values -- a float16 cumsum could otherwise flip a
    # depth near a threshold boundary.
    rng = np.random.default_rng(3)
    pdf16 = _reference_pdf(rng, 512, 6).astype(np.float16)
    for threshold in (0.3, 0.5, 0.8):
        got = depths_from_pdf(pdf16, threshold=threshold, min_exit_step=2)
        expected = depths_from_pdf(pdf16.astype(np.float32), threshold=threshold, min_exit_step=2)
        np.testing.assert_array_equal(got, expected)


def test_depths_from_pdf_honors_delay_and_never_exceeded() -> None:
    # A degenerate PDF that never reaches a high threshold before the last step
    # must fall back to the last step (max depth).
    pdf = np.array([[0.1, 0.1, 0.1, 0.7]], dtype=np.float32)
    depths = depths_from_pdf(pdf, threshold=0.99, min_exit_step=1)
    assert depths.tolist() == [4]
    delayed = depths_from_pdf(pdf, threshold=0.05, min_exit_step=1, exit_delay_steps=1)
    assert delayed.tolist() == [2]  # crosses at step 0 -> depth 1, +1 delay -> depth 2


# --------------------------------------------------------------------------- batching


def _req(seq_len: int, request_id: str) -> SampledRequest:
    # Half prompt, half output; exact split does not matter for grouping.
    prompt = [7] * (seq_len // 2)
    output = [8] * (seq_len - seq_len // 2)
    return SampledRequest(id=request_id, prompt_ids=prompt, output_ids=output)


def test_group_by_token_budget_respects_padded_cost() -> None:
    requests = [_req(10, "a"), _req(10, "b"), _req(10, "c")]
    # Budget 20 padded tokens: two 10-length rows (2*10=20) fit, the third starts a new batch.
    batches = list(group_by_token_budget(requests, max_num_batched_tokens=20, max_batch_size=8))
    assert [[r.id for r in batch] for batch in batches] == [["a", "b"], ["c"]]


def test_group_by_token_budget_yields_oversized_request_alone() -> None:
    requests = [_req(4, "small"), _req(100, "huge"), _req(4, "tail")]
    batches = list(group_by_token_budget(requests, max_num_batched_tokens=16, max_batch_size=8))
    assert [r.id for r in batches[1]] == ["huge"]  # too big to share, but not dropped
    assert {r.id for batch in batches for r in batch} == {"small", "huge", "tail"}


def test_group_by_token_budget_respects_max_batch_size() -> None:
    requests = [_req(2, str(i)) for i in range(5)]
    batches = list(group_by_token_budget(requests, max_num_batched_tokens=10_000, max_batch_size=2))
    assert [len(batch) for batch in batches] == [2, 2, 1]


def test_record_exit_pdfs_returns_original_request_order() -> None:
    # The recorder sorts by length internally to pack microbatches, but the saved bundle must keep
    # the original request order so the length sort does not leak into the replay order.
    from types import SimpleNamespace

    import torch

    from looped_cdb.benchmarks.exit_recording import record_exit_pdfs

    class _FakeExitModel:
        config = SimpleNamespace(total_ut_steps=4)

        def __call__(self, *, input_ids, use_early_exit_gate, return_exit_pdf, **_kwargs):
            batch, length = input_ids.shape
            pdf = torch.zeros((batch, length, 4))
            pdf[..., 0] = 1.0  # a valid one-hot exit PDF (exit at the first step everywhere)
            return SimpleNamespace(ouro_exit_pdf=pdf)

    # Input order is deliberately not length-sorted (totals 20, 6, 12).
    requests = [_req(20, "a"), _req(6, "b"), _req(12, "c")]
    workload = record_exit_pdfs(
        _FakeExitModel(), requests, max_num_batched_tokens=64, max_batch_size=8, device="cpu", progress=False
    )

    assert workload.ids == ["a", "b", "c"]
    assert workload.input_lens.tolist() == [req.input_len for req in requests]
    assert workload.output_lens.tolist() == [req.output_len for req in requests]


def test_record_exit_pdfs_records_a_repeated_request_object_at_both_positions() -> None:
    # Requests are placed back by their position in the batching order, not by object identity.
    # Keying on identity would collapse the two slots of a repeated object, leaving a hole that
    # concatenating the schedules would then reject.
    from types import SimpleNamespace

    import torch

    from looped_cdb.benchmarks.exit_recording import record_exit_pdfs

    class _FakeExitModel:
        config = SimpleNamespace(total_ut_steps=4)

        def __call__(self, *, input_ids, use_early_exit_gate, return_exit_pdf, **_kwargs):
            batch, length = input_ids.shape
            pdf = torch.zeros((batch, length, 4))
            pdf[..., 0] = 1.0
            return SimpleNamespace(ouro_exit_pdf=pdf)

    shared = _req(10, "dup")
    requests = [shared, _req(6, "b"), shared]

    workload = record_exit_pdfs(
        _FakeExitModel(), requests, max_num_batched_tokens=64, max_batch_size=2, device="cpu", progress=False
    )

    assert workload.ids == ["dup", "b", "dup"]
    assert workload.output_lens.tolist() == [req.output_len for req in requests]
    assert workload.exit_pdf.shape[0] == sum(req.output_len for req in requests)


def test_workload_shuffle_permutes_requests_and_keeps_schedule_aligned() -> None:
    # Distinct per-request depth values let us verify each request's exit schedule follows it
    # through the permutation, and that the shuffle is deterministic given the seed.
    output_lens = np.array([2, 1, 3], dtype=np.int32)
    offsets = np.zeros(4, dtype=np.int64)
    np.cumsum(output_lens.astype(np.int64), out=offsets[1:])
    exit_depths = np.concatenate([np.full(int(n), i + 1, dtype=np.int32) for i, n in enumerate(output_lens)])
    workload = Workload(
        ids=["r0", "r1", "r2"],
        input_lens=np.array([5, 6, 7], dtype=np.int32),
        output_lens=output_lens,
        offsets=offsets,
        exit_depths=exit_depths,
        max_depth=4,
    )

    shuffled = workload.shuffle(seed=1)

    assert sorted(shuffled.ids) == ["r0", "r1", "r2"]
    for pos, rid in enumerate(shuffled.ids):
        orig = workload.ids.index(rid)
        assert int(shuffled.input_lens[pos]) == int(workload.input_lens[orig])
        assert int(shuffled.output_lens[pos]) == int(workload.output_lens[orig])
        segment = shuffled.exit_depths[shuffled.offsets[pos] : shuffled.offsets[pos + 1]]
        assert segment.tolist() == [orig + 1] * int(workload.output_lens[orig])
    assert workload.shuffle(seed=1).ids == shuffled.ids  # deterministic given the seed


def test_workload_shuffle_keeps_the_2d_exit_pdf_aligned() -> None:
    # Recorded bundles carry a 2D per-token exit pdf, not 1D exit depths, so this is the branch
    # the shuffle actually takes in production. Each request's rows are stamped with its own
    # value so a misaligned reorder cannot pass.
    output_lens = np.array([2, 1, 3], dtype=np.int32)
    offsets = np.zeros(4, dtype=np.int64)
    np.cumsum(output_lens.astype(np.int64), out=offsets[1:])
    exit_pdf = np.concatenate(
        [np.full((int(n), 4), i + 1, dtype=np.float16) for i, n in enumerate(output_lens)], axis=0
    )
    workload = Workload(
        ids=["r0", "r1", "r2"],
        input_lens=np.array([5, 6, 7], dtype=np.int32),
        output_lens=output_lens,
        offsets=offsets,
        exit_pdf=exit_pdf,
        max_depth=4,
    )

    shuffled = workload.shuffle(seed=3)

    assert shuffled.exit_depths is None
    assert shuffled.exit_pdf.shape == exit_pdf.shape
    for pos, rid in enumerate(shuffled.ids):
        orig = workload.ids.index(rid)
        rows = shuffled.exit_pdf[shuffled.offsets[pos] : shuffled.offsets[pos + 1]]
        assert rows.shape == (int(workload.output_lens[orig]), 4)
        assert np.all(rows == orig + 1)


# --------------------------------------------------------------------------- Workload round-trip


def _toy_workload() -> Workload:
    # Two requests with 2 and 3 output tokens, 4 steps each.
    pdf = _reference_pdf(np.random.default_rng(1), 5, 4).astype(np.float16)
    return Workload(
        ids=["r0", "r1"],
        input_lens=np.array([12, 30], dtype=np.int32),
        output_lens=np.array([2, 3], dtype=np.int32),
        exit_pdf=pdf,
        offsets=np.array([0, 2, 5], dtype=np.int64),
        meta={"dataset": "toy"},
    )


def test_workload_roundtrip_preserves_content(tmp_path) -> None:
    workload = _toy_workload()
    json_path, pdf_path = workload.save(tmp_path / "toy.json")
    assert json_path.exists() and pdf_path.exists()

    loaded = Workload.load(json_path)
    assert loaded.ids == workload.ids
    np.testing.assert_array_equal(loaded.input_lens, workload.input_lens)
    np.testing.assert_array_equal(loaded.output_lens, workload.output_lens)
    np.testing.assert_array_equal(loaded.offsets, workload.offsets)
    np.testing.assert_allclose(loaded.exit_pdf, workload.exit_pdf)
    assert loaded.meta["dataset"] == "toy"
    assert loaded.meta["num_requests"] == 2


def test_workload_materialize_depths_matches_per_request_derivation() -> None:
    workload = _toy_workload()
    per_request = workload.materialize_depths(threshold=0.5, min_exit_step=2)
    assert [len(d) for d in per_request] == [2, 3]
    # Materialized depths must equal deriving each request's PDF slice directly.
    for i, depths in enumerate(per_request):
        expected = depths_from_pdf(workload.request_pdf(i), threshold=0.5, min_exit_step=2)
        assert depths == expected.tolist()


def test_workload_rejects_misaligned_offsets() -> None:
    with pytest.raises(ValueError, match="offsets"):
        Workload(
            ids=["r0"],
            input_lens=np.array([4], dtype=np.int32),
            output_lens=np.array([2], dtype=np.int32),
            exit_pdf=np.zeros((2, 4), dtype=np.float16),
            offsets=np.array([0, 3], dtype=np.int64),  # claims 3 rows, array has 2
        )


def test_select_requests_keeps_only_requests_every_tokenizer_accepts(tmp_path) -> None:
    import json

    from create_workload_json import select_requests

    def wide_encode(text: str, add_special_tokens: bool) -> list[int]:
        return [1] * (2 * len(text.split()))

    records = [
        {
            "id": f"c{i}",
            "conversations": [{"from": "human", "value": f"ask {i} " * 2}, {"from": "gpt", "value": "a b c d e"}],
        }
        for i in range(6)
    ]
    records.append(
        {"id": "long", "conversations": [{"from": "human", "value": "w " * 20}, {"from": "gpt", "value": "a b c d e"}]}
    )
    records.append(
        {
            "id": "short",
            "conversations": [{"from": "human", "value": "hi there"}, {"from": "gpt", "value": "a b c d e"}],
        }
    )
    path = tmp_path / "sharegpt.json"
    path.write_text(json.dumps(records))
    common = {"max_model_len": 40, "min_prompt_len": 4, "min_output_len": 4, "limit": None, "shuffle_seed": None}

    alone, dropped, pool = select_requests("sharegpt", path, whitespace_encode, **common)
    assert [s.id for s in alone] == [f"c{i}" for i in range(6)] + ["long"] and dropped == 0 and pool == 7
    # The wide tokenizer doubles every length, so "long" no longer fits the context.
    both, dropped, pool = select_requests("sharegpt", path, whitespace_encode, filter_encoders=[wide_encode], **common)
    assert [s.id for s in both] == [f"c{i}" for i in range(6)] and dropped == 1 and pool == 6
    # The kept requests carry the recording tokenizer's own lengths.
    assert both[0].input_len == len(alone[0].prompt_ids)


def test_select_requests_filters_position_keyed_records_under_the_same_shuffle(tmp_path) -> None:
    import json

    from create_workload_json import select_requests

    def wide_encode(text: str, add_special_tokens: bool) -> list[int]:
        return [1] * (2 * len(text.split()))

    # Alpaca records carry no id, so requests are keyed by their position after the shuffle; the
    # long instruction is the only one the wide tokenizer rejects, wherever the shuffle puts it.
    records = [{"instruction": f"do thing {i} now please", "input": "", "output": "a b c d e"} for i in range(12)]
    records.insert(5, {"instruction": "w " * 20, "input": "", "output": "a b c d e"})
    path = tmp_path / "alpaca.json"
    path.write_text(json.dumps(records))
    common = {"max_model_len": 40, "min_prompt_len": 4, "min_output_len": 4, "limit": None, "shuffle_seed": 3}

    alone, _, _ = select_requests("alpaca", path, whitespace_encode, **common)
    both, dropped, pool = select_requests("alpaca", path, whitespace_encode, filter_encoders=[wide_encode], **common)
    assert dropped == 1 and pool == 12
    assert [s.id for s in both] == [s.id for s in alone if s.input_len < 20]
