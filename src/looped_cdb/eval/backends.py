"""Generation backends behind one interface: CB and CDB.

Each backend exposes ``generate(prompts, *, max_new_tokens, stop_strings) ->
list[str]`` and shares ``EngineBackend``. Both decode with
``skip_special_tokens=False`` and truncate at the first stop string via
:func:`decode_generations`.

The token<->string helpers at the top bridge the backends (which run at the token
level) to the string-level prompt assembly and scoring around them.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Sequence
from typing import Any, Protocol

logger = logging.getLogger(__name__)


def tokenize(tokenizer: Any, text: str) -> list[int]:
    """Tokenize a final prompt string into a flat list of token IDs."""

    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if input_ids and isinstance(input_ids[0], list):
        return list(input_ids[0])
    return list(input_ids)


def generated_tokens(output: Any) -> list[int]:
    """Extract generated token IDs from the repo-local engine output shape."""

    if hasattr(output, "generated_tokens"):
        return list(output.generated_tokens)
    if isinstance(output, dict) and "generated_tokens" in output:
        return list(output["generated_tokens"])
    if isinstance(output, Sequence):
        return list(output)
    raise TypeError(f"Unsupported generate_batch output type: {type(output)!r}")


def eos_token_ids(tokenizer: Any) -> list[int]:
    """Return tokenizer EOS IDs as a list, regardless of HF scalar/list shape."""

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        return []
    if isinstance(eos_token_id, int):
        return [eos_token_id]
    return list(eos_token_id)


def stop_string_eos_ids(tokenizer: Any, stop_strings: Sequence[str]) -> list[int]:
    """Combine tokenizer EOS IDs with any single-token stop strings."""

    token_ids = eos_token_ids(tokenizer)
    for stop in stop_strings:
        if not stop:
            continue
        stop_ids = tokenize(tokenizer, stop)
        if len(stop_ids) == 1:
            token_ids.append(stop_ids[0])
    return list(dict.fromkeys(token_ids))


def stop_string_sequences(tokenizer: Any, stop_strings: Sequence[str]) -> list[list[int]]:
    """Return the multi-token stop strings as token-id sequences.

    Single-token stops are already covered by :func:`stop_string_eos_ids`. The
    rest need suffix matching in the engine; without it a few-shot prompt runs
    to its length cap, since the string that actually ends the answer is usually
    the model opening the next exemplar (``Q:``) rather than an EOS token.
    """

    sequences = []
    for stop in stop_strings:
        if not stop:
            continue
        stop_ids = tokenize(tokenizer, stop)
        if len(stop_ids) > 1 and stop_ids not in sequences:
            sequences.append(stop_ids)
    return sequences


def truncate_at_stop(text: str, stop_strings: Sequence[str]) -> str:
    """Cut ``text`` at the earliest occurrence of any stop string."""

    cut = len(text)
    for stop in stop_strings:
        if not stop:
            continue
        index = text.find(stop)
        if index != -1:
            cut = min(cut, index)
    return text[:cut]


def decode_eos(tokenizer: Any) -> str | None:
    """Decode the first EOS token so it can be added to the stop strings."""

    ids = eos_token_ids(tokenizer)
    if not ids:
        return None
    return tokenizer.decode([ids[0]], skip_special_tokens=False)


def decode_generations(
    tokenizer: Any,
    token_id_lists: Sequence[Sequence[int]],
    stop_strings: Sequence[str],
) -> list[str]:
    """Decode generated token IDs and truncate each at the first stop string.

    The decoded EOS token is appended to the stop strings so an emitted EOS ends
    the answer even when it is not already listed in the task's stop strings.
    """

    stops = list(stop_strings)
    eos_text = decode_eos(tokenizer)
    if eos_text:
        stops.append(eos_text)
    decoded = []
    for token_ids in token_id_lists:
        text = tokenizer.decode(list(token_ids), skip_special_tokens=False)
        decoded.append(truncate_at_stop(text, stops))
    return decoded


def engine_accepts(engine: Any, parameter: str) -> bool:
    """Whether ``engine.generate_batch`` takes ``parameter``.

    Engines in this repo do not share a base class, so support is probed rather
    than assumed; an engine without it keeps its previous behavior.
    """

    generate_batch = getattr(engine, "generate_batch", None)
    if generate_batch is None:
        return False
    try:
        return parameter in inspect.signature(generate_batch).parameters
    except (TypeError, ValueError):
        return False


class Backend(Protocol):
    """A greedy text-generation backend."""

    def generate(self, prompts: list[str], *, max_new_tokens: int, stop_strings: list[str]) -> list[str]: ...


class EngineBackend:
    """Wrap a repo-local CB/CDB engine (anything exposing ``generate_batch``).

    ``model_kwargs`` are forwarded verbatim to the model's forward via
    ``generate_batch``. This passes early-exit settings (``use_early_exit_gate``,
    ``exit_threshold``, ``min_exit_step``, ``return_exit_steps``) reach the model
    without any engine-specific plumbing.
    """

    def __init__(
        self,
        engine: Any,
        tokenizer: Any,
        *,
        warmup: bool = False,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.engine = engine
        self.tokenizer = tokenizer
        self.warmup = warmup
        self.model_kwargs = model_kwargs

    def generate(self, prompts: list[str], *, max_new_tokens: int, stop_strings: list[str]) -> list[str]:
        input_ids = [tokenize(self.tokenizer, prompt) for prompt in prompts]
        eos_token_id = stop_string_eos_ids(self.tokenizer, stop_strings)
        extra = {"model_kwargs": self.model_kwargs} if self.model_kwargs else {}
        stop_sequences = stop_string_sequences(self.tokenizer, stop_strings)
        if engine_accepts(self.engine, "stop_sequences"):
            extra["stop_sequences"] = stop_sequences
        elif stop_sequences:
            # Without suffix matching these stops cannot fire, so generation runs to
            # its length cap. That is invisible in the scored text, which is truncated
            # anyway, and shows up only as inflated token counts.
            logger.warning(
                "%s does not accept stop_sequences; %d multi-token stop string(s) will not "
                "end generation and every request will run to its length cap.",
                type(self.engine).__name__,
                len(stop_sequences),
            )
        outputs = self.engine.generate_batch(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            warmup=self.warmup,
            **extra,
        )
        token_id_lists = [generated_tokens(output) for output in outputs]
        return decode_generations(self.tokenizer, token_id_lists, stop_strings)

    @property
    def exit_depth_counts(self) -> dict[int, int]:
        """1-indexed exit-depth -> decode-token count from the last generation.

        Populated by the CDB engine's ``last_stats`` (adaptive-depth serving); a
        fixed-depth engine leaves it empty because every token runs to full depth.
        """

        last_stats = getattr(self.engine, "last_stats", None)
        histogram = getattr(last_stats, "exit_depth_histogram", None)
        if not histogram:
            return {}
        return {int(exit_step) + 1: int(count) for exit_step, count in histogram.items()}


class CBBackend(EngineBackend):
    """Fixed-depth continuous batching backend.

    Every token is decoded at the model's full recurrent depth; the depth is
    configured at load time. Adaptive early exit lives in :class:`CDBBackend`.
    """

    @classmethod
    def from_model(cls, model: Any, tokenizer: Any, *, cb_config: Any) -> CBBackend:
        from looped_cdb.continuous_batching import ContinuousBatchingEngine

        engine = ContinuousBatchingEngine.from_model(model, cb_config=cb_config)
        return cls(engine, tokenizer)


class CDBBackend(EngineBackend):
    """Continuous-depth batching backend.

    The model's depth and gate are configured at load time; the exit threshold and
    minimum depth live in ``cdb_config`` because the CDB scheduler owns the exit
    decision.

    ``model_adapter`` optionally overrides the adapter inferred from the model type.
    For convergence-based exits, the adapter computes the convergence signal and declares its decision rule.
    The exit policy applies the threshold from ``cdb_config``.
    """

    @classmethod
    def from_model(cls, model: Any, tokenizer: Any, *, cdb_config: Any, model_adapter: Any = None) -> CDBBackend:
        from looped_cdb.continuous_depth_batching import ContinuousDepthBatchingEngine

        engine = ContinuousDepthBatchingEngine.from_model(model, cdb_config=cdb_config, model_adapter=model_adapter)
        return cls(engine, tokenizer)

    @property
    def exit_gate_type(self) -> str | None:
        """The gate the engine resolved, which may differ from the one requested.

        A run that omits the gate flag is served by whatever the checkpoint's config
        declares, so recording the request would misattribute it.
        """

        config = getattr(getattr(self.engine, "model", None), "config", None)
        gate_type = getattr(config, "exit_gate_type", None)
        return str(gate_type) if gate_type is not None else None

    @property
    def decides_exit_before_loop(self) -> bool:
        """Whether the served gate fixes exit depths before the recurrent loop runs."""

        adapter = getattr(self.engine, "model_adapter", None)
        return bool(adapter is not None and adapter.decides_exit_before_loop())
