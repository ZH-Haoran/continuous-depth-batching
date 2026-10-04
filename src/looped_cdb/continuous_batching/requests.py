# Copyright 2025 The HuggingFace Inc. team
# SPDX-License-Identifier: Apache-2.0
# Modified for looped-model serving and continuous depth batching.
# See THIRD_PARTY_NOTICES.md and LICENSES/Apache-2.0.txt.

"""Request lifecycle state for native continuous batching.

This module intentionally mirrors the shape of Hugging Face's continuous
batching request objects while keeping only the pieces needed for clean
implementation. A request tracks three token views:

- ``initial_tokens``: the prompt plus any generated tokens folded in by a soft reset.
- ``remaining_prefill_tokens``: prompt tokens not yet scheduled for prefill.
- ``tokens_to_process``: the prompt chunk or decode token scheduled in the
  current batch.

Chunked prefill is represented by keeping a request in ``PREFILLING`` while
``remaining_prefill_tokens`` is non-empty. Once the final prompt chunk has been
processed, the request moves to ``DECODING`` and each sampled token becomes the
next single-token decode input.

Reference:
https://github.com/huggingface/transformers/blob/main/src/transformers/generation/continuous_batching/requests.py
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Self

from looped_cdb.request_status import RequestStatus

__all__ = ["TMP_TOKEN_ID", "FutureRequestState", "GenerationOutput", "RequestState", "RequestStatus", "logger"]

# This is a temporary token ID used to represent a token that is not yet generated.
TMP_TOKEN_ID = -1


# We centralize the logger here to coordinate between logging and progress bars.
logger = logging.getLogger("ContinuousBatchingLogger")
if logger.propagate:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False


@dataclass
class GenerationOutput:
    """Tracks the output of a generation request.

    Attributes:
        request_id (str): The ID of the generation request.
        prompt_ids (list[int]): The IDs of the prompt tokens.
        generated_tokens (list[int]): The generated tokens.
        status (RequestStatus): The status of the request.
        created_time (float): The time the request was created.
        lifespan (tuple[float, float]): The time the request was no longer pending and the time the request finished.
        first_token_time (float): The time the host consumed the first generated token (-1 if none yet).
    """

    request_id: str
    prompt_ids: list[int] = field(default_factory=list)
    generated_tokens: list[int] = field(default_factory=list)
    status: RequestStatus = RequestStatus.PENDING
    created_time: float = field(default_factory=time.perf_counter)
    lifespan: tuple[float, float] = (-1, -1)  # (time request was no longer pending, time request finished)
    first_token_time: float = -1.0  # Time the first generated token reached the host (-1 if none yet)
    token_ready_times: list[float] = field(default_factory=list)

    def is_finished(self) -> bool:
        return self.status == RequestStatus.FINISHED


@dataclass
class RequestState:
    """Tracks the state of a generation request through its lifecycle.

    Attributes:
        request_id (str): The ID of the generation request.
        initial_tokens (list[int]): The prefill tokens, including folded generation after a soft reset.
        tokens_to_process (list[int]): The token IDs scheduled for the next model forward.
        remaining_prefill_tokens (list[int]): The prompt token IDs not yet scheduled for prefill.
        generated_tokens (list[int]): The generated tokens.
        allocated_blocks (int): The number of blocks allocated to the request.
        position_offset (int): The current position in the sequence for position_ids.
        status (RequestStatus): The current request lifecycle state.
        max_new_tokens (int | None): The maximum number of new tokens to generate.
        eos_token_id (None | int | list[int]): The ID(s) of the end-of-sequence tokens. Only used in post-init.
        _eos_token_ids (set[int]): The IDs of the end-of-sequence tokens, formatted as a set.
        created_time (float): The time the request was created.
    """

    # Required fields
    request_id: str
    initial_tokens: list[int]  # Tokens used to prefill this request.

    # Optional fields (generation parameters)
    max_new_tokens: int | None = 20  # Maximum number of new tokens to generate. None means no limit. Default to 20.
    eos_token_id: int | list[int] | None = None  # ID(s) of the end-of-sequence tokens. Only used in post-init.
    # Token-id sequences that end generation when one of them is a suffix of the generated tokens.
    # A stop string that does not tokenize to a single id cannot be an EOS id, and few-shot prompts
    # usually end on exactly such a string (the model starting the next exemplar), so without this
    # a request runs to its length cap and the surplus is discarded after decoding.
    stop_sequences: list[list[int]] = field(default_factory=list)

    # Internal fields (for scheduling)
    tokens_to_process: list[int] = field(default_factory=list)  # Tokens IDs currently being processed
    generated_tokens: list[int] = field(default_factory=list)  # Generated tokens
    position_offset: int = 0  # Current position in the sequence for position_ids
    allocated_blocks: int = 0  # Number of blocks allocated to the request

    _status: RequestStatus = RequestStatus.PENDING  # Status of the request, hidden behind a property
    _eos_token_ids: set[int] = field(default_factory=set)  # IDs of the end-of-sequence tokens, formatted as a set

    # Internal fields (for tracking)
    created_time: float = field(default_factory=time.perf_counter)  # Time the request was created
    lifespan: tuple[float, float] = (-1, -1)  # (time request was no longer pending, time request finished)
    first_token_time: float = -1.0  # Time the first generated token reached the host (-1 if none yet)
    record_token_times: bool = False
    token_ready_times: list[float] = field(default_factory=list)

    # True when the request's KV cache currently lives in the CPU swap pool (offloaded, awaiting restore).
    is_cpu_offloaded: bool = False
    # Number of this request's sampled tokens that exist only on the device: their batches have been
    # launched but their outputs have not been consumed by the host yet. Transiently 2 while the next
    # batch has been prepared and the previous one not yet consumed.
    tokens_in_flight: int = 0
    # Treat the length-limit finish like a sampled EOS: the scheduler stops predicting it, runs one
    # token past the limit, and discards that token at consume, as free generation does on an EOS.
    replay_eos_finishes: bool = False
    # Number of tokens in ``initial_tokens`` that are the true original prompt. Non-zero only after a soft
    # reset, where already-generated tokens are folded into ``initial_tokens``; it lets ``to_generation_output``
    # recover the real prompt/generation split.
    _true_initial_tokens: int = 0

    # Fields overwritten in __post_init__
    _new_tokens_limit: int = 2147483647  # An int to check the max number of new tokens w/out always comparing w/ None
    remaining_prefill_tokens: list[int] = field(default_factory=list)  # Initial tokens left to process

    def __post_init__(self) -> None:
        # If no max length is set, we set an absurdly high value which will never be reached
        self._new_tokens_limit = 2147483647 if self.max_new_tokens is None else self.max_new_tokens
        # Keep a copy of the initial tokens to process
        self.remaining_prefill_tokens = self.initial_tokens[:]
        # Format the EOS token ID(s) as a set of ints. If there is no EOS token ID, it's an empty set
        if self.eos_token_id is None:
            pass
        # If there is a single EOS token ID, add it to the set only if the ID is valid, ie. non-negative
        elif isinstance(self.eos_token_id, int):
            if self.eos_token_id >= 0:
                self._eos_token_ids.add(self.eos_token_id)
        # If there are multiple EOS token IDs, add them to the set only if they are valid, ie. non-negative
        else:
            for token_id in self.eos_token_id:
                if token_id >= 0:
                    self._eos_token_ids.add(token_id)

    @property
    def status(self) -> RequestStatus:
        return self._status

    @status.setter
    def status(self, value: RequestStatus) -> None:
        # A soft-reset request re-enters PENDING with its original first-schedule time carried
        # over; only stamp the start when the request was never scheduled before.
        if self._status == RequestStatus.PENDING and self.lifespan[0] < 0:
            self.lifespan = (time.perf_counter(), -1)
        elif value == RequestStatus.FINISHED:
            self.lifespan = (self.lifespan[0], time.perf_counter())
            self.log_end_of_request()
        self._status = value

    def log_end_of_request(self) -> None:
        prefill_len = len(self.initial_tokens)
        decode_len = self.generated_len()
        start_time = self.lifespan[0] - self.created_time
        end_time = self.lifespan[1] - self.created_time
        logger.debug(
            f"Request {self.request_id} finished: {prefill_len = } {decode_len = } {start_time = } {end_time = }"
        )

    def current_len(self) -> int:
        """Get the current length of the sequence (prompt + generated tokens)."""
        return self.position_offset

    def generated_len(self) -> int:
        """Get the number of tokens generated so far."""
        return len(self.generated_tokens)

    def total_generated_len(self) -> int:
        """Tokens generated so far, including generation a soft reset folded onto ``initial_tokens``."""
        true_prompt = self._true_initial_tokens or len(self.initial_tokens)
        return len(self.initial_tokens) - true_prompt + len(self.generated_tokens)

    def will_finish_on_pending_token(self) -> bool:
        """Whether the tokens already in flight are guaranteed to finish this request when consumed.

        The scheduler uses this to stop re-scheduling a request whose length limit is already met by
        its unconsumed tokens, so a length-capped request never computes a wasted extra token. An EOS
        finish cannot be predicted host-side, so EOS requests still compute one lagged token.
        ``replay_eos_finishes`` disables the prediction, making every finish behave like an EOS.
        """
        if self.replay_eos_finishes:
            return False
        return self.tokens_in_flight > 0 and self.generated_len() + self.tokens_in_flight >= self._new_tokens_limit

    def update_and_check_completion(self, token_id: int) -> bool:
        """Update the request with a newly generated token and check whether it is now complete."""
        if self.status != RequestStatus.DECODING:
            raise RuntimeError(f"Cannot update request {self.request_id} while status is {self.status.name}")
        self.tokens_in_flight = max(0, self.tokens_in_flight - 1)

        # Stop if we reached an EOS token
        is_eos = token_id in self._eos_token_ids
        current_len = self.generated_len()

        # Keep EOS in the output, but discard a token computed after the length limit.
        if is_eos or (current_len < self._new_tokens_limit):
            # First generated token: stamp TTFT once. A soft reset folds generated tokens into the
            # prompt (emptying ``generated_tokens``), so the guard is on the stamp, not the list.
            if self.first_token_time < 0 or self.record_token_times:
                ready_time = time.perf_counter()
                if self.first_token_time < 0:
                    self.first_token_time = ready_time
                if self.record_token_times:
                    self.token_ready_times.append(ready_time)
            self.generated_tokens.append(token_id)
            self.tokens_to_process = [token_id]
            current_len += 1
        else:
            logger.warning(f"Request {self.request_id} generated a useless token: {token_id}")

        if is_eos or current_len >= self._new_tokens_limit or self._matches_stop_sequence():
            self.status = RequestStatus.FINISHED
            return True
        return False  # We still need to process more tokens

    def _matches_stop_sequence(self) -> bool:
        """Whether the generated tokens now end with one of ``stop_sequences``.

        Checked against the tail only, so cost is bounded by the longest stop
        sequence rather than the generation length.
        """

        for sequence in self.stop_sequences:
            length = len(sequence)
            if length and len(self.generated_tokens) >= length and self.generated_tokens[-length:] == sequence:
                return True
        return False

    def __repr__(self) -> str:
        msg = [
            f"request_id={self.request_id}",
            f"status={self._status}",
            f"out_tokens={self.generated_len()}",
            f"query_length={len(self.tokens_to_process)}",
            f"remaining_tokens={len(self.remaining_prefill_tokens)}",
            f"kv_length={self.position_offset}",
            f"full_prompt_length={len(self.initial_tokens)}",
            f"allocated_blocks={self.allocated_blocks}",
            f"generated_tokens={self.generated_tokens}",
        ]
        return "RequestState(\n\t" + ",\n\t".join(msg) + "\n)"

    def get_request_config(self) -> dict:
        """Return the generation-parameter fields needed to recreate an equivalent request."""

        return {
            "max_new_tokens": self.max_new_tokens,
            "eos_token_id": self.eos_token_id,
            "stop_sequences": [list(sequence) for sequence in self.stop_sequences],
        }

    def prepare_for_offload_requeue(self) -> None:
        """Reset a just-offloaded request so it resumes from its restored KV without recomputing.

        The request keeps its status and is re-queued; only its block allocation is dropped, since the
        physical blocks (and their KV) come back on restore. A decoding victim keeps its pending token
        in ``tokens_to_process`` and resumes by running it as a decode step; a mid-prefill victim keeps
        its ``remaining_prefill_tokens`` and resumes its prefill. Both engines use this single
        decode-first resume, so the recompute-vs-offload comparison is not confounded by the resume
        path (no wake-up token is ever re-run through the full-depth prefill path).
        """

        self.allocated_blocks = 0
        if self._status == RequestStatus.DECODING:
            self.remaining_prefill_tokens = []

    def create_equivalent_initial_request(self) -> Self:
        """Rebuild this request as a fresh prefill for soft-reset preemption (recompute).

        The already-generated tokens are folded onto the prompt (``initial_tokens + generated_tokens``)
        and ``max_new_tokens`` is reduced by the number generated, so the recomputed request resumes at
        the same output position. The new request keeps the same ``request_id`` and records how many of
        its ``initial_tokens`` are the true original prompt via ``_true_initial_tokens``.
        """

        request_config = self.get_request_config()
        if self.max_new_tokens is not None:
            request_config["max_new_tokens"] = self.max_new_tokens - len(self.generated_tokens)
        new_state = type(self)(
            request_id=self.request_id,
            initial_tokens=self.initial_tokens + self.generated_tokens,
            **request_config,
        )
        # Preserve the original prompt boundary across repeated soft resets.
        new_state._true_initial_tokens = self._true_initial_tokens or len(self.initial_tokens)
        # Same request resuming, not a new arrival: latency keeps measuring from the original times.
        new_state.created_time = self.created_time
        new_state.lifespan = self.lifespan
        new_state.first_token_time = self.first_token_time
        new_state.record_token_times = self.record_token_times
        new_state.token_ready_times = self.token_ready_times.copy()
        return new_state

    def to_generation_output(self) -> GenerationOutput:
        """Convert the request state to a GenerationOutput object."""
        # After a soft reset, generated tokens were folded into initial_tokens; split them back out so
        # the output reports the true prompt and the full generation.
        if self._true_initial_tokens:
            self.generated_tokens = self.initial_tokens[self._true_initial_tokens :] + self.generated_tokens
            self.initial_tokens = self.initial_tokens[: self._true_initial_tokens]
            self._true_initial_tokens = 0
        return GenerationOutput(
            request_id=self.request_id,
            prompt_ids=self.initial_tokens,
            generated_tokens=self.generated_tokens,
            status=self.status,
            created_time=self.created_time,
            lifespan=self.lifespan,
            first_token_time=self.first_token_time,
            token_ready_times=self.token_ready_times,
        )


class FutureRequestState:
    """Tracks the current state of a request and the relevant information to update it."""

    # This makes instantiating this class faster
    __slots__ = ("has_new_token", "query_length", "state")

    def __init__(self, state: RequestState, has_new_token: bool, query_length: int) -> None:
        self.state = state
        self.has_new_token = has_new_token
        self.query_length = query_length
