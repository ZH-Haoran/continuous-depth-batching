"""Load ShareGPT, Alpaca, and ArXiv-Summarization datasets into tokenized request samples.

Each request becomes a :class:`SampledRequest` holding a tokenized prompt and
reference output; benchmarks use the resulting input/output token lengths to
build realistic serving workloads.

* **ShareGPT** -- multi-turn ChatGPT conversations. The first human turn is the
  prompt and the following assistant turn is the output (one request per
  conversation).
* **Alpaca** -- single-turn instructions. The instruction (plus its optional
  ``input`` field) is the prompt and ``output`` is the completion.
* **ArXiv-Summarization** -- scientific articles paired with their abstracts. The
  prompt wraps the full article in a short summarization request (an instruction
  before it and an ``Abstract:`` cue after it), so the recorded exit behaviour is that
  of a request rather than of an article the model was never asked to summarize; the
  abstract is the output. Long prompts, short outputs, so prefill dominates the served
  work. The Hub export carries no id column, so requests are keyed by position and the
  bundles of both models must be recorded from the same shards in the same order.

Records are read from a JSON list, JSON Lines, or Parquet file (or every such file
in a directory, in name order); the Hub exports ArXiv-Summarization as Parquet.

Tokenization is injected as an ``encode`` callable, so this module has no model
dependency.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

# encode(text, add_special_tokens) -> token ids. Matches a Hugging Face
# tokenizer call, but decoupled so tests can pass a trivial tokenizer.
EncodeFn = Callable[[str, bool], Sequence[int]]

# ShareGPT roles that count as the user side of the first turn. Assistant turns
# ("gpt") supply the completion.
_HUMAN_ROLES = frozenset({"human", "user"})


@dataclass(frozen=True)
class SampledRequest:
    """A single request reduced to its tokenized prompt and reference output."""

    id: str
    prompt_ids: list[int]
    output_ids: list[int]

    @property
    def input_len(self) -> int:
        return len(self.prompt_ids)

    @property
    def output_len(self) -> int:
        return len(self.output_ids)


def _tokenize_pair(
    request_id: str,
    prompt_text: str,
    output_text: str,
    encode: EncodeFn,
    *,
    min_prompt_len: int,
    min_output_len: int,
) -> SampledRequest | None:
    """Tokenize a (prompt, output) pair, or return ``None`` if either side is too short.

    The prompt is encoded with special tokens (e.g. BOS) so its length matches
    what the model attends to; the output is encoded as a bare continuation.
    """

    prompt_text = prompt_text.strip()
    output_text = output_text.strip()
    if not prompt_text or not output_text:
        return None

    prompt_ids = list(encode(prompt_text, True))
    output_ids = list(encode(output_text, False))
    if len(prompt_ids) < min_prompt_len or len(output_ids) < min_output_len:
        return None
    return SampledRequest(id=request_id, prompt_ids=prompt_ids, output_ids=output_ids)


def iter_sharegpt(
    records: Iterable[dict],
    encode: EncodeFn,
    *,
    min_prompt_len: int = 4,
    min_output_len: int = 4,
) -> Iterator[SampledRequest]:
    """Yield one request per conversation: first human turn -> next assistant turn."""

    for index, record in enumerate(records):
        turns = record.get("conversations") or []
        if len(turns) < 2:
            continue
        if str(turns[0].get("from", "")).lower() not in _HUMAN_ROLES:
            # Skip conversations that open with a system/assistant turn so the
            # prompt is genuinely the user's request.
            continue
        request_id = str(record.get("id", index))
        sample = _tokenize_pair(
            request_id,
            str(turns[0].get("value", "")),
            str(turns[1].get("value", "")),
            encode,
            min_prompt_len=min_prompt_len,
            min_output_len=min_output_len,
        )
        if sample is not None:
            yield sample


def iter_alpaca(
    records: Iterable[dict],
    encode: EncodeFn,
    *,
    min_prompt_len: int = 4,
    min_output_len: int = 4,
) -> Iterator[SampledRequest]:
    """Yield one request per instruction: instruction(+input) -> output."""

    for index, record in enumerate(records):
        instruction = str(record.get("instruction", "")).strip()
        extra_input = str(record.get("input", "")).strip()
        prompt_text = f"{instruction}\n{extra_input}" if extra_input else instruction
        sample = _tokenize_pair(
            str(record.get("id", index)),
            prompt_text,
            str(record.get("output", "")),
            encode,
            min_prompt_len=min_prompt_len,
            min_output_len=min_output_len,
        )
        if sample is not None:
            yield sample


ARXIV_PROMPT_PREFIX = "Summarize the following scientific article in an abstract.\n\n"
ARXIV_PROMPT_SUFFIX = "\n\nAbstract:"


def iter_arxiv(
    records: Iterable[dict],
    encode: EncodeFn,
    *,
    min_prompt_len: int = 4,
    min_output_len: int = 4,
) -> Iterator[SampledRequest]:
    """Yield one request per article: summarization request around the article -> abstract."""

    for index, record in enumerate(records):
        article = str(record.get("article") or "").strip()
        if not article:
            continue
        sample = _tokenize_pair(
            str(record.get("article_id") or record.get("id") or index),
            f"{ARXIV_PROMPT_PREFIX}{article}{ARXIV_PROMPT_SUFFIX}",
            str(record.get("abstract") or ""),
            encode,
            min_prompt_len=min_prompt_len,
            min_output_len=min_output_len,
        )
        if sample is not None:
            yield sample


_LOADERS: dict[str, Callable[..., Iterator[SampledRequest]]] = {
    "sharegpt": iter_sharegpt,
    "alpaca": iter_alpaca,
    "arxiv": iter_arxiv,
}

_RECORD_SUFFIXES = (".json", ".jsonl", ".parquet")


def read_records(path: Path) -> list[dict]:
    """Read raw dataset records from a JSON list, JSON Lines, or Parquet file.

    A directory reads every file with a supported suffix in name order and concatenates
    them, so a dataset published as a few shards needs no manual merge. Every record is
    materialized, so the input should be sized to what the bundle needs (a shard or two),
    not a whole multi-gigabyte export.
    """

    path = Path(path)
    if path.is_dir():
        shards = sorted(p for p in path.iterdir() if p.suffix in _RECORD_SUFFIXES)
        if not shards:
            raise ValueError(f"{path} holds no {'/'.join(_RECORD_SUFFIXES)} files")
        return [record for shard in shards for record in read_records(shard)]

    if path.suffix not in _RECORD_SUFFIXES:
        raise ValueError(f"unsupported dataset file {path}; expected one of {'/'.join(_RECORD_SUFFIXES)}")
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        # Streamed in row batches: converting the whole table at once holds the Arrow
        # table and its Python copy together, several times the file size.
        records: list[dict] = []
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
            records.extend(batch.to_pylist())
        return records
    if path.suffix == ".jsonl":
        with path.open() as handle:
            return [json.loads(line) for line in handle if line.strip()]
    records = json.loads(path.read_text())
    if not isinstance(records, list):
        raise ValueError(f"{path} does not contain a top-level JSON list of records")
    return records


def load_dataset(
    dataset: str,
    path: Path,
    encode: EncodeFn,
    *,
    min_prompt_len: int = 4,
    min_output_len: int = 4,
    shuffle_seed: int | None = None,
) -> list[SampledRequest]:
    """Read a dataset file and return every tokenized, minimum-length request.

    ``dataset`` selects the adapter (``"sharegpt"``, ``"alpaca"``, or ``"arxiv"``); ``path``
    is anything :func:`read_records` accepts. ``shuffle_seed`` shuffles the records before
    they are parsed, which is what makes a prefix of the result a uniform sample: the raw
    files are roughly ordered, and every filter downstream is order-preserving.

    There is deliberately no ``limit`` here. Capping the request count before the caller has
    applied its remaining filters (notably the context-length filter) silently returns fewer
    requests than were asked for, so subsampling belongs after the last filter.
    """

    key = dataset.lower()
    if key not in _LOADERS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {sorted(_LOADERS)}")

    records = read_records(path)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(records)

    return list(_LOADERS[key](records, encode, min_prompt_len=min_prompt_len, min_output_len=min_output_len))


def filter_by_context(samples: list[SampledRequest], max_seq_len: int) -> tuple[list[SampledRequest], int]:
    """Drop requests whose prompt+output exceeds ``max_seq_len``; return ``(kept, dropped)``.

    Sequences longer than the model's context window cannot be processed, so this
    enforces that hard limit (distinct from the dataset-quality minimum-length
    filter). Returns the surviving requests and how many were dropped.
    """

    kept = [s for s in samples if s.input_len + s.output_len <= max_seq_len]
    return kept, len(samples) - len(kept)
