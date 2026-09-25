from __future__ import annotations

import math
import re
from collections.abc import Callable

from datasets import Dataset, load_dataset

NEMOTRON_DATASET_NAME = "nvidia/Nemotron-Post-Training-Dataset-v2"
NEMOTRON_DATASET_REVISION = "5c89e01dd720ae0f4058445ed49c5fb68a03c76e"
NEMOTRON_CATEGORIES = ("math", "code", "stem", "chat")
VALIDATION_PACKED_SEQUENCES = 128
VALIDATION_SEED = 12_879
VALIDATION_ROW_SAMPLE_SIZE = 1024
VALIDATION_ROW_OVERSAMPLE = 1.5
MIN_VALIDATION_ROWS_PER_SOURCE = 512
TOKENIZE_NUM_PROC = 8
PACKING_MAP_BATCH_SIZE = 1000


def strip_assistant_thinking_traces(messages: list[dict[str, object]]) -> list[dict[str, str]]:
    """Remove assistant-only ``<think>...</think>`` traces from chat messages."""
    stripped_messages: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role", ""))
        content = str(message.get("content", ""))
        if role == "assistant":
            content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL).lstrip()
        stripped_messages.append({"role": role, "content": content})
    return stripped_messages


def get_stripped_messages(row: dict[str, object]) -> list[dict[str, str]] | None:
    """Return stripped chat messages when the row passes basic Nemotron filters."""
    if row.get("reasoning") not in {None, "off"}:
        return None
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None
    return strip_assistant_thinking_traces(messages)


def load_category_dataset(category: str) -> Dataset:
    """Load one hardcoded Nemotron parquet category."""
    return load_dataset(
        "parquet",
        data_files=f"hf://datasets/{NEMOTRON_DATASET_NAME}@{NEMOTRON_DATASET_REVISION}/data/{category}-*",
        split="train",
    )


def validation_row_count(
    dataset: Dataset,
    *,
    encoded_length: Callable[[dict[str, object]], int],
    sequence_length: int,
    seed: int,
) -> int:
    """Estimate enough validation rows to produce the fixed packed validation set."""
    if len(dataset) < 2:
        raise ValueError("Each Nemotron category must have at least two trainable rows")

    sample_size = min(VALIDATION_ROW_SAMPLE_SIZE, len(dataset))
    sample = dataset.shuffle(seed=seed).select(range(sample_size))
    lengths = [encoded_length(row) for row in sample]
    average_tokens = max(sum(lengths) / max(len(lengths), 1), 1.0)
    target_tokens_per_source = VALIDATION_PACKED_SEQUENCES * sequence_length / len(NEMOTRON_CATEGORIES)
    estimated_rows = math.ceil(target_tokens_per_source / average_tokens * VALIDATION_ROW_OVERSAMPLE)
    validation_rows = max(MIN_VALIDATION_ROWS_PER_SOURCE, estimated_rows)
    return min(validation_rows, len(dataset) - 1)
