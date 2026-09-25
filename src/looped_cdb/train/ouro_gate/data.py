from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

import torch
from datasets import Dataset, concatenate_datasets
from torch.utils.data import DataLoader, IterableDataset
from transformers import PreTrainedTokenizerBase

from looped_cdb.train.common_data import (
    NEMOTRON_CATEGORIES,
    PACKING_MAP_BATCH_SIZE,
    TOKENIZE_NUM_PROC,
    VALIDATION_PACKED_SEQUENCES,
    VALIDATION_SEED,
    get_stripped_messages,
    load_category_dataset,
    validation_row_count,
)

SOURCE_TO_ID = {source: source_id for source_id, source in enumerate(NEMOTRON_CATEGORIES)}
ID_TO_SOURCE = {source_id: source for source, source_id in SOURCE_TO_ID.items()}
TRAIN_DATALOADER_NUM_WORKERS = 1
TRAIN_DATALOADER_PREFETCH_FACTOR = 2
DataSplit = Literal["train", "validation"]


class PackedConversationDataset(IterableDataset[dict[str, list[int]]]):
    """Pack tokenized conversations into fixed-length training batches."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        sequence_length: int,
        max_sequences: int | None = None,
    ) -> None:
        """Create an iterable packer over a tokenized Hugging Face dataset."""
        super().__init__()
        self.dataset = dataset
        self.sequence_length = sequence_length
        self.max_sequences = max_sequences

    def __iter__(self) -> Iterator[dict[str, list[int]]]:
        """Yield packed sequences with shifted assistant-only loss masks."""
        emitted = 0

        for row in self.dataset:
            input_ids = [int(token_id) for token_id in row["input_ids"]]
            if len(input_ids) > self.sequence_length:
                raise ValueError("packed input length exceeds configured sequence_length")
            assistant_mask = [int(value) for value in row["assistant_mask"]]
            if len(input_ids) != len(assistant_mask):
                raise ValueError("input_ids and assistant_mask must have the same length")

            seq_lengths = [int(length) for length in row.get("seq_lengths", [len(input_ids)])]
            if sum(seq_lengths) != len(input_ids):
                raise ValueError("seq_lengths must sum to the packed input length")

            loss_mask = shifted_assistant_loss_mask(assistant_mask, seq_lengths=seq_lengths)
            if not any(loss_mask):
                continue

            source_ids = [int(source_id) for source_id in row["source_ids"]]
            if len(source_ids) != len(input_ids):
                raise ValueError("source_ids and input_ids must have the same length")

            yield {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "assistant_mask": assistant_mask,
                "loss_mask": loss_mask,
                "position_ids": position_ids_from_seq_lengths(seq_lengths),
                "seq_lengths": seq_lengths,
                "source_counts": _source_counts_from_ids(source_ids),
            }
            emitted += 1
            if self.max_sequences is not None and emitted >= self.max_sequences:
                return

        if self.max_sequences is not None and emitted < self.max_sequences:
            raise RuntimeError(
                f"Packed validation dataset produced {emitted} assistant-bearing sequences, "
                f"but {self.max_sequences} were requested. Increase the validation row holdout or inspect "
                "assistant-token masking before comparing runs."
            )


def shifted_assistant_loss_mask(assistant_mask: list[int], seq_lengths: list[int] | None = None) -> list[int]:
    """Return positions whose next token is assistant text."""
    if not assistant_mask:
        return []
    if seq_lengths is None:
        return [*assistant_mask[1:], 0]

    loss_mask = [0] * len(assistant_mask)
    offset = 0
    for seq_length in seq_lengths:
        if seq_length < 0:
            raise ValueError("seq_lengths must be non-negative")
        end = offset + seq_length
        if end > len(assistant_mask):
            raise ValueError("seq_lengths exceed assistant_mask length")
        for index in range(offset, max(end - 1, offset)):
            loss_mask[index] = assistant_mask[index + 1]
        offset = end
    if offset != len(assistant_mask):
        raise ValueError("seq_lengths must sum to assistant_mask length")
    return loss_mask


def position_ids_from_seq_lengths(seq_lengths: list[int]) -> list[int]:
    """Return packed-sequence position ids that reset at each sequence boundary."""
    position_ids: list[int] = []
    for seq_length in seq_lengths:
        if seq_length < 0:
            raise ValueError("seq_lengths must be non-negative")
        position_ids.extend(range(seq_length))
    return position_ids


def create_packed_dataloaders(
    *,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    eval_batch_size: int,
    sequence_length: int,
    seed: int,
) -> tuple[DataLoader[dict[str, torch.Tensor]], DataLoader[dict[str, torch.Tensor]]]:
    """Create train and fixed-validation dataloaders for packed conversations."""
    _validate_assistant_mask_support(tokenizer)
    train_dataset, validation_dataset = _load_tokenized_splits(
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        seed=seed,
    )
    train_packed = PackedConversationDataset(train_dataset, sequence_length=sequence_length)
    validation_packed = PackedConversationDataset(
        validation_dataset,
        sequence_length=sequence_length,
        max_sequences=VALIDATION_PACKED_SEQUENCES,
    )
    return (
        DataLoader(
            train_packed,
            batch_size=batch_size,
            collate_fn=collate_packed_features,
            pin_memory=torch.cuda.is_available(),
            num_workers=TRAIN_DATALOADER_NUM_WORKERS,
            persistent_workers=True,
            prefetch_factor=TRAIN_DATALOADER_PREFETCH_FACTOR,
        ),
        DataLoader(
            validation_packed,
            batch_size=eval_batch_size,
            collate_fn=collate_packed_features,
            pin_memory=torch.cuda.is_available(),
        ),
    )


def collate_packed_features(features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
    """Flatten packed sequence examples into one padding-free tensor row."""
    input_ids = _flatten_feature(features, "input_ids")
    attention_mask = _flatten_feature(features, "attention_mask")
    assistant_mask = _flatten_feature(features, "assistant_mask")
    loss_mask = _flatten_feature(features, "loss_mask")
    position_ids = _flatten_feature(features, "position_ids")
    seq_lengths = _flatten_feature(features, "seq_lengths")
    source_counts = [
        sum(int(feature["source_counts"][source_id]) for feature in features)
        for source_id in range(len(NEMOTRON_CATEGORIES))
    ]
    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "attention_mask": torch.tensor([attention_mask], dtype=torch.bool),
        "assistant_mask": torch.tensor([assistant_mask], dtype=torch.bool),
        "loss_mask": torch.tensor([loss_mask], dtype=torch.bool),
        "position_ids": torch.tensor([position_ids], dtype=torch.long),
        "seq_lengths": torch.tensor(seq_lengths, dtype=torch.long),
        "source_counts": torch.tensor([source_counts], dtype=torch.long),
    }


def _load_tokenized_splits(
    *,
    tokenizer: PreTrainedTokenizerBase,
    sequence_length: int,
    seed: int,
) -> tuple[Dataset, Dataset]:
    """Load, split, shuffle, and tokenize the hardcoded Nemotron categories."""
    train_parts: list[Dataset] = []
    validation_parts: list[Dataset] = []

    for source_id, category in enumerate(NEMOTRON_CATEGORIES):
        validation_seed = VALIDATION_SEED + source_id
        dataset = load_category_dataset(category).filter(
            _is_trainable_row,
            desc=f"Filter {category}",
        )
        validation_rows = validation_row_count(
            dataset,
            encoded_length=lambda row: _encoded_length(row, tokenizer=tokenizer),
            sequence_length=sequence_length,
            seed=validation_seed,
        )
        parts = dataset.train_test_split(
            test_size=validation_rows,
            seed=validation_seed,
            shuffle=True,
        )
        train_parts.append(_add_source_id(parts["train"], source_id))
        validation_parts.append(_add_source_id(parts["test"], source_id))

    train_dataset = concatenate_datasets(train_parts).shuffle(seed=seed)
    validation_dataset = concatenate_datasets(validation_parts).shuffle(seed=VALIDATION_SEED)
    remove_columns = train_dataset.column_names
    train_tokenized = train_dataset.map(
        _encode_row,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=remove_columns,
        num_proc=TOKENIZE_NUM_PROC,
        desc="Tokenize packed train split",
    )
    validation_tokenized = validation_dataset.map(
        _encode_row,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=remove_columns,
        num_proc=TOKENIZE_NUM_PROC,
        desc="Tokenize packed validation split",
    )
    return (
        _pack_tokenized_dataset(
            train_tokenized,
            sequence_length=sequence_length,
            seed=seed,
            desc="Pack train split with BFD",
        ),
        _pack_tokenized_dataset(
            validation_tokenized,
            sequence_length=sequence_length,
            seed=VALIDATION_SEED,
            desc="Pack validation split with BFD",
        ),
    )


def _pack_tokenized_dataset(
    dataset: Dataset,
    *,
    sequence_length: int,
    seed: int,
    desc: str,
) -> Dataset:
    """Pack tokenized rows using TRL's best-fit-decreasing sequence packer."""
    from trl.data_utils import pack_dataset

    packed = pack_dataset(
        dataset.select_columns(["input_ids", "assistant_mask", "source_ids"]),
        sequence_length,
        strategy="bfd",
        map_kwargs={
            "batch_size": PACKING_MAP_BATCH_SIZE,
            "num_proc": TOKENIZE_NUM_PROC,
            "desc": desc,
        },
    )
    return packed.shuffle(seed=seed)


def _flatten_feature(features: list[dict[str, list[int]]], key: str) -> list[int]:
    """Concatenate one list-valued feature from a batch of packed examples."""
    values: list[int] = []
    for feature in features:
        values.extend(int(value) for value in feature[key])
    return values


def _is_trainable_row(row: dict[str, object]) -> bool:
    """Return whether a row has assistant text usable for supervised training."""
    stripped_messages = get_stripped_messages(row)
    if stripped_messages is None:
        return False
    return any(message["role"] == "assistant" and message["content"].strip() for message in stripped_messages)


def _add_source_id(dataset: Dataset, source_id: int) -> Dataset:
    """Attach a compact integer source id to every row."""
    return dataset.map(
        lambda _row: {"source_id": source_id},
        desc=f"Add source id {source_id}",
    )


def _encoded_length(row: dict[str, object], *, tokenizer: PreTrainedTokenizerBase) -> int:
    """Return the encoded length of a trainable row including a boundary EOS."""
    encoded = _encode_messages(row, tokenizer=tokenizer)
    return len(encoded["input_ids"])


def _encode_row(row: dict[str, object], *, tokenizer: PreTrainedTokenizerBase) -> dict[str, list[int] | int]:
    """Render and tokenize one chat row, then attach its integer source id."""
    encoded = _encode_messages(row, tokenizer=tokenizer)
    return {
        "input_ids": encoded["input_ids"],
        "assistant_mask": encoded["assistant_mask"],
        "source_ids": [int(row["source_id"])] * len(encoded["input_ids"]),
    }


def _encode_messages(row: dict[str, object], *, tokenizer: PreTrainedTokenizerBase) -> dict[str, list[int]]:
    """Render and tokenize one chat row with assistant-token masks."""
    stripped_messages = get_stripped_messages(row)
    if stripped_messages is None:
        raise ValueError("Nemotron training rows must contain structured messages")
    tokenized = tokenizer.apply_chat_template(
        stripped_messages,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        add_generation_prompt=False,
    )
    input_ids = [int(token_id) for token_id in tokenized["input_ids"]]
    assistant_mask = [int(value) for value in tokenized["assistant_masks"]]
    if len(input_ids) != len(assistant_mask):
        raise ValueError("Tokenizer returned mismatched input_ids and assistant_masks lengths")

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None and (not input_ids or input_ids[-1] != eos_token_id):
        input_ids.append(int(eos_token_id))
        assistant_mask.append(0)

    return {
        "input_ids": input_ids,
        "assistant_mask": assistant_mask,
    }


def _validate_assistant_mask_support(tokenizer: PreTrainedTokenizerBase) -> None:
    """Fail clearly if the tokenizer chat template cannot mark assistant tokens."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    tokenized = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        add_generation_prompt=False,
    )
    if sum(int(value) for value in tokenized.get("assistant_masks", [])) == 0:
        raise ValueError(
            "Tokenizer chat_template must include {% generation %} / {% endgeneration %} around assistant content "
            "so return_assistant_tokens_mask=True can produce assistant-only loss masks."
        )


def _source_counts_from_ids(source_ids: list[int]) -> list[int]:
    """Return per-source token counts for one packed sequence."""
    counts = [0] * len(NEMOTRON_CATEGORIES)
    for source_id in source_ids:
        counts[source_id] += 1
    return counts
