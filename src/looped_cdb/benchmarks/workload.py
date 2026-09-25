"""Serving workload: per-request lengths plus a per-output-token exit schedule.

A workload has two interchangeable sources of exit depths:

* *Recorded* -- for every output token, the model's exit PDF over recurrent steps
  (recorded by ``create_workload_json.py``); depths for any threshold and minimum
  exit step are derived offline with :func:`depths_from_pdf`.
* *Synthetic* -- fixed (or uniformly-sampled) lengths with explicit per-token depths
  drawn from a named distribution (:meth:`Workload.synthetic`), analogous to a
  fixed/random serving benchmark.

Downstream code only calls :meth:`Workload.materialize_depths`, so the two are
interchangeable. On disk a workload is two sibling files:

* ``<name>.json`` -- the definition: metadata and one row per request
  (``id``, ``input_len``, ``output_len``).
* ``<name>.exit_pdf.npz`` -- ``offsets`` plus the exit schedule: a flat ``float16``
  ``exit_pdf`` (recorded) or an ``int32`` ``exit_depths`` (synthetic). JSON is a poor
  container for millions of numbers, so they live in this compact binary alongside it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_PDF_SUFFIX = ".exit_pdf.npz"


def depths_from_values(
    values: np.ndarray,
    *,
    threshold: float,
    min_exit_step: int = 1,
    exit_delay_steps: int = 0,
) -> np.ndarray:
    """Derive 1-indexed exit depths from recorded convergence-criterion values.

    ``values`` has shape ``(..., total_steps)`` and holds the exit criterion
    evaluated after each recurrent step, so a token exits at the first step whose
    value falls below ``threshold``. This is the convergence-exit analogue of
    :func:`depths_from_pdf`: both record a threshold-free trajectory once and
    derive depths offline, but a trained gate accumulates probability upward
    while a convergence criterion decays, so the crossing is inverted.

    Tokens that never converge take the last step, matching a decode that runs to
    the depth budget.
    """

    if values.shape[-1] < 1:
        raise ValueError("values must have at least one step in the last axis")
    total_steps = values.shape[-1]
    mask = np.asarray(values, dtype=np.float32) < float(threshold)
    min_depth = max(1, int(min_exit_step or 1))
    if min_depth > 1:
        mask[..., : min_depth - 1] = False
    exited = mask.any(axis=-1)
    depths = np.where(exited, mask.argmax(axis=-1) + 1, total_steps).astype(np.int32)
    delay = max(0, int(exit_delay_steps or 0))
    return np.minimum(np.maximum(depths, min_depth) + delay, total_steps).astype(np.int32)


def depths_from_pdf(
    pdf: np.ndarray,
    *,
    threshold: float,
    min_exit_step: int = 1,
    exit_delay_steps: int = 0,
) -> np.ndarray:
    """Derive 1-indexed exit depths from a step PDF at a given threshold.

    ``pdf`` has shape ``(..., total_steps)`` and sums to 1 over the last axis.
    This mirrors the model's :func:`qexit_steps_from_pdf` exactly (cumulative
    probability crossing ``threshold``, with a floor at ``min_exit_step`` and an
    optional delay), but runs offline in numpy and returns depth = step + 1.

    The cumulative sum is computed in float32 so the result matches the model
    regardless of the PDF's storage precision (PDFs are stored as float16, whose
    cumulative rounding could otherwise flip a depth near a threshold boundary).
    """

    if pdf.shape[-1] < 1:
        raise ValueError("pdf must have at least one step in the last axis")
    cumulative = np.cumsum(np.asarray(pdf, dtype=np.float32), axis=-1)
    mask = cumulative >= threshold
    min_depth = max(1, int(min_exit_step or 1))
    if min_depth > 1:
        # Forbid exiting before the minimum depth (steps are 0-indexed, so a
        # minimum depth of m zeroes the first m-1 steps).
        mask[..., : min(min_depth - 1, mask.shape[-1])] = False
    steps = np.argmax(mask, axis=-1)
    last_step = pdf.shape[-1] - 1
    never = ~mask.any(axis=-1)
    steps = np.where(never, last_step, steps)
    delay = max(0, int(exit_delay_steps or 0))
    if delay:
        steps = np.minimum(steps + delay, last_step)
    return steps.astype(np.int64) + 1


def slice_output_pdf(pdf_row: np.ndarray, prompt_len: int, output_len: int) -> np.ndarray:
    """Slice the ``(output_len, total_steps)`` PDF for a request's output tokens.

    The forward at position ``i`` predicts token ``i + 1``, so output token ``j``
    is produced at absolute position ``prompt_len - 1 + j``.
    """

    if prompt_len < 1:
        raise ValueError(f"prompt_len must be >= 1, got {prompt_len}")
    if output_len < 0:
        raise ValueError(f"output_len must be >= 0, got {output_len}")
    start = prompt_len - 1
    end = start + output_len
    if end > pdf_row.shape[0]:
        raise ValueError(f"pdf_row too short: need positions up to {end - 1}, have {pdf_row.shape[0]}")
    return pdf_row[start:end]


def _sample_lengths(
    rng: np.random.Generator,
    count: int,
    low: int,
    high: int | None,
    *,
    name: str,
    minimum: int,
) -> np.ndarray:
    """Return ``count`` int32 lengths: fixed at ``low`` (high None) or uniform ``[low, high]``."""

    if low < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {low}")
    if high is None:
        return np.full(count, int(low), dtype=np.int32)
    if high < low:
        raise ValueError(f"{name}_high must be >= {name}, got {high} < {low}")
    return rng.integers(low, high + 1, size=count).astype(np.int32)


@dataclass
class Workload:
    """A serving workload: aligned request lengths and a per-output-token exit schedule.

    The exit schedule is stored one of two ways, and exactly one must be set:

    * ``exit_pdf`` -- a recorded ``float16 [total_tokens, steps]`` PDF over recurrent
      steps (from ``create_workload_json.py``); depths are derived offline at a
      threshold via :func:`depths_from_pdf`.
    * ``exit_values`` -- recorded ``float16 [total_tokens, steps]`` convergence
      criterion values (from ``record_huginn_workload.py``); depths are derived
      offline at a threshold via :func:`depths_from_values`.
    * ``exit_depths`` -- explicit 1-indexed ``int32 [total_tokens]`` depths (from
      :meth:`synthetic`); used directly, independent of any threshold.

    ``offsets`` maps request ``i`` to its token span ``offsets[i]:offsets[i + 1]`` in
    whichever array is set. Downstream code only calls :meth:`materialize_depths`, so
    the two sources are interchangeable.
    """

    ids: list[str]
    input_lens: np.ndarray  # int32 [N]
    output_lens: np.ndarray  # int32 [N]
    offsets: np.ndarray  # int64 [N + 1], row i spans offsets[i]:offsets[i + 1]
    exit_pdf: np.ndarray | None = None  # float16 [total_tokens, steps] (recorded gate PDF)
    exit_values: np.ndarray | None = None  # float16 [total_tokens, steps] (recorded criterion values)
    exit_depths: np.ndarray | None = None  # int32 [total_tokens], 1-indexed (explicit/synthetic)
    max_depth: int | None = None  # required with exit_depths; inferred from exit_pdf otherwise
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.ids)
        if self.input_lens.shape[0] != n or self.output_lens.shape[0] != n:
            raise ValueError("ids, input_lens and output_lens must have the same length")
        if self.offsets.shape[0] != n + 1:
            raise ValueError("offsets must have length num_requests + 1")
        sources = [self.exit_pdf, self.exit_values, self.exit_depths]
        if sum(source is not None for source in sources) != 1:
            raise ValueError("exactly one of exit_pdf, exit_values or exit_depths must be set")
        rows = next(source for source in sources if source is not None).shape[0]
        if int(self.offsets[-1]) != rows:
            raise ValueError("offsets[-1] must equal the number of exit-schedule rows")
        if self.exit_pdf is not None:
            self.max_depth = int(self.exit_pdf.shape[1])
        elif self.exit_values is not None:
            self.max_depth = int(self.exit_values.shape[1])
        elif self.max_depth is None:
            raise ValueError("max_depth is required when exit_depths is set")

    @property
    def num_requests(self) -> int:
        return len(self.ids)

    @property
    def total_steps(self) -> int:
        return int(self.max_depth)

    def request_pdf(self, index: int) -> np.ndarray:
        """Return the ``(output_len, total_steps)`` recorded PDF for request ``index``."""

        if self.exit_pdf is None:
            raise ValueError("request_pdf is only available for recorded (PDF) workloads")
        return self.exit_pdf[self.offsets[index] : self.offsets[index + 1]]

    def materialize_depths(
        self,
        *,
        threshold: float | None = None,
        min_exit_step: int = 1,
        exit_delay_steps: int = 0,
    ) -> list[list[int]]:
        """Per-request 1-indexed exit-depth sequences.

        Recorded workloads derive depths from the PDF at ``threshold``; explicit
        (synthetic) workloads use their stored depths directly. Both then apply the
        same ``min_exit_step`` floor and ``exit_delay_steps`` shift, capped at
        ``total_steps``, so the two sources behave identically downstream.
        """

        if self.exit_depths is not None:
            min_depth = max(1, int(min_exit_step or 1))
            delay = max(0, int(exit_delay_steps or 0))
            adjusted = np.minimum(np.maximum(self.exit_depths, min_depth) + delay, self.total_steps)
            return [
                adjusted[self.offsets[i] : self.offsets[i + 1]].astype(int).tolist() for i in range(self.num_requests)
            ]

        if threshold is None:
            raise ValueError("threshold is required to materialize depths from a recorded trajectory")
        if self.exit_values is not None:
            depths = depths_from_values(
                self.exit_values,
                threshold=threshold,
                min_exit_step=min_exit_step,
                exit_delay_steps=exit_delay_steps,
            )
            return [depths[self.offsets[i] : self.offsets[i + 1]].tolist() for i in range(self.num_requests)]
        depths = depths_from_pdf(
            self.exit_pdf,
            threshold=threshold,
            min_exit_step=min_exit_step,
            exit_delay_steps=exit_delay_steps,
        )
        return [depths[self.offsets[i] : self.offsets[i + 1]].tolist() for i in range(self.num_requests)]

    def mean_requested_depth_at(
        self,
        *,
        threshold: float | None = None,
        min_exit_step: int = 1,
        exit_delay_steps: int = 0,
    ) -> float:
        """Mean recurrent depth the exit schedule requests per generated token.

        Matches the benchmark's requested exit distribution
        (:func:`looped_cdb.benchmarks.runner.requested_exit_distribution`): each request's
        first output token is produced by the full-depth prefill forward, so it counts at
        ``total_steps`` regardless of the schedule. Recorded workloads accept any
        ``threshold``, not just measured ones, which lets the FLOP bound
        (:meth:`looped_cdb.benchmarks.flop_bound.StageFlops.ideal_speedup`) be evaluated as
        a dense curve without an engine run.
        """

        if self.exit_depths is not None:
            min_depth = max(1, int(min_exit_step or 1))
            delay = max(0, int(exit_delay_steps or 0))
            depths = np.minimum(np.maximum(self.exit_depths, min_depth) + delay, self.total_steps).astype(np.int64)
        else:
            if threshold is None:
                raise ValueError("threshold is required to derive depths from a recorded trajectory")
            derive = depths_from_values if self.exit_values is not None else depths_from_pdf
            source = self.exit_values if self.exit_values is not None else self.exit_pdf
            depths = derive(
                source,
                threshold=threshold,
                min_exit_step=min_exit_step,
                exit_delay_steps=exit_delay_steps,
            ).astype(np.int64)
        if depths.shape[0] == 0:
            raise ValueError("cannot compute a mean depth for a workload with no output tokens")
        first_rows = self.offsets[:-1]
        depths[first_rows[first_rows < self.offsets[1:]]] = self.total_steps
        return float(depths.mean())

    @classmethod
    def synthetic(
        cls,
        *,
        num_requests: int,
        input_len: int,
        output_len: int,
        exit_dist: str,
        max_depth: int,
        seed: int = 0,
        input_len_high: int | None = None,
        output_len_high: int | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Workload:
        """Build a synthetic workload with fixed (or uniformly-sampled) lengths and a chosen exit distribution.

        Lengths are ``input_len``/``output_len`` when ``*_high`` is None, else sampled
        uniformly in ``[len, *_high]`` (inclusive). Per-token exit depths are drawn from
        the named distribution (see :mod:`looped_cdb.benchmarks.exit_distributions`).
        Deterministic given ``seed``.
        """

        from looped_cdb.benchmarks.exit_distributions import sample_token_depths

        if num_requests < 1:
            raise ValueError(f"num_requests must be >= 1, got {num_requests}")
        rng = np.random.default_rng(seed)
        input_lens = _sample_lengths(rng, num_requests, input_len, input_len_high, name="input_len", minimum=1)
        output_lens = _sample_lengths(rng, num_requests, output_len, output_len_high, name="output_len", minimum=1)
        offsets = np.zeros(num_requests + 1, dtype=np.int64)
        np.cumsum(output_lens.astype(np.int64), out=offsets[1:])
        exit_depths = sample_token_depths(exit_dist, int(offsets[-1]), max_depth, seed=seed)
        base_meta: dict[str, Any] = {
            "dataset": f"random_{exit_dist}",
            "synthetic": True,
            "exit_dist": exit_dist,
            "input_len": input_len,
            "output_len": output_len,
            "input_len_high": input_len_high,
            "output_len_high": output_len_high,
            "seed": seed,
        }
        if meta:
            base_meta.update(meta)
        return cls(
            ids=[f"synthetic-{i}" for i in range(num_requests)],
            input_lens=input_lens,
            output_lens=output_lens,
            offsets=offsets,
            exit_depths=exit_depths,
            max_depth=int(max_depth),
            meta=base_meta,
        )

    def take(self, num_requests: int | None) -> Workload:
        """Return a workload with only the first ``num_requests`` requests (self if None/all)."""

        if num_requests is None or num_requests >= self.num_requests:
            return self
        if num_requests < 1:
            raise ValueError(f"num_requests must be >= 1, got {num_requests}")
        end_row = int(self.offsets[num_requests])
        return Workload(
            ids=self.ids[:num_requests],
            input_lens=self.input_lens[:num_requests],
            output_lens=self.output_lens[:num_requests],
            offsets=self.offsets[: num_requests + 1].copy(),
            exit_pdf=self.exit_pdf[:end_row] if self.exit_pdf is not None else None,
            exit_values=self.exit_values[:end_row] if self.exit_values is not None else None,
            exit_depths=self.exit_depths[:end_row] if self.exit_depths is not None else None,
            max_depth=self.max_depth,
            meta=self.meta,
        )

    def shuffle(self, seed: int) -> Workload:
        """Return a copy with requests replayed in a seeded random order.

        Recorded bundles are stored sorted by length. Replaying in that order is unrealistic
        (real arrivals are not length-ordered) and starves scheduler concurrency once the long
        tail arrives all at once; it also makes a prefix (:meth:`take`) a length-biased sample.
        Shuffling gives a realistic arrival order and makes ``take`` an unbiased random subsample.
        The permutation is drawn over the ids in sorted order, so bundles holding the same
        requests replay in the same order under the same seed whatever their stored order or
        tokenizer. The per-token exit schedule is reordered to stay aligned with each request's
        output.
        """

        by_id = np.argsort(np.asarray(self.ids, dtype=object), kind="stable")
        return self._reorder(by_id[np.random.default_rng(seed).permutation(self.num_requests)])

    def _reorder(self, indices: np.ndarray) -> Workload:
        """The workload made of the requests at ``indices`` (a permutation or a subset), in that order."""

        schedule = self._schedule_array()
        new_output_lens = self.output_lens[indices]
        offsets = np.zeros(len(indices) + 1, dtype=self.offsets.dtype)
        np.cumsum(new_output_lens.astype(np.int64), out=offsets[1:])
        rows = [schedule[int(self.offsets[i]) : int(self.offsets[i + 1])] for i in indices]
        new_schedule = np.concatenate(rows, axis=0) if rows else schedule[:0]
        return Workload(
            ids=[self.ids[int(i)] for i in indices],
            input_lens=self.input_lens[indices],
            output_lens=new_output_lens,
            offsets=offsets,
            **self._schedule_kwargs(new_schedule),
            max_depth=self.max_depth,
            meta=self.meta,
        )

    def apply_max_model_len(self, max_model_len: int | None) -> Workload:
        """Drop unservable prompts and truncate outputs so ``input_len + output_len <= max_model_len``.

        Mirrors the engine's ``max_model_len`` admission (see
        :func:`looped_cdb.utils.cap_max_new_tokens_to_model_len`): a request whose prompt leaves no
        room for a generated token (``input_len >= max_model_len``) is removed, and every kept
        request's output is truncated to at most ``max_model_len - input_len`` tokens with its
        per-token exit schedule sliced to match, so the replay schedule stays aligned with the
        (shortened) output length. Returns ``self`` when ``max_model_len`` is ``None`` or when
        every request already fits, rather than rebuilding the exit schedule to reach the same
        answer; raises if no request survives.
        """

        if max_model_len is None:
            return self
        # A bundle recorded at this context length changes not at all, and its schedule can run to
        # tens of millions of rows. Callers apply this before slicing, so the common case is a no-op.
        allowed = max_model_len - self.input_lens.astype(np.int64)
        if bool((allowed > 0).all()) and bool((self.output_lens <= allowed).all()):
            return self
        schedule = self._schedule_array()
        keep_ids: list[str] = []
        keep_input: list[int] = []
        keep_output: list[int] = []
        keep_rows: list[np.ndarray] = []
        for i in range(self.num_requests):
            input_len = int(self.input_lens[i])
            allowed = max_model_len - input_len
            if allowed <= 0:
                continue
            new_output = min(int(self.output_lens[i]), allowed)
            start = int(self.offsets[i])
            keep_ids.append(self.ids[i])
            keep_input.append(input_len)
            keep_output.append(new_output)
            keep_rows.append(schedule[start : start + new_output])
        if not keep_ids:
            raise ValueError(f"no request fits max_model_len={max_model_len} (every prompt is too long)")
        new_output_lens = np.asarray(keep_output, dtype=self.output_lens.dtype)
        offsets = np.zeros(len(keep_ids) + 1, dtype=self.offsets.dtype)
        np.cumsum(new_output_lens.astype(np.int64), out=offsets[1:])
        new_schedule = np.concatenate(keep_rows, axis=0)
        return Workload(
            ids=keep_ids,
            input_lens=np.asarray(keep_input, dtype=self.input_lens.dtype),
            output_lens=new_output_lens,
            offsets=offsets,
            **self._schedule_kwargs(new_schedule),
            max_depth=self.max_depth,
            meta=self.meta,
        )

    def _schedule_array(self) -> np.ndarray:
        """The one populated exit-schedule array, whichever source is set."""

        for source in (self.exit_pdf, self.exit_values, self.exit_depths):
            if source is not None:
                return source
        raise AssertionError("no exit schedule set")

    def _schedule_kwargs(self, schedule: np.ndarray) -> dict[str, np.ndarray | None]:
        """Route a rebuilt schedule array back to the field it came from."""

        return {
            "exit_pdf": schedule if self.exit_pdf is not None else None,
            "exit_values": schedule if self.exit_values is not None else None,
            "exit_depths": schedule if self.exit_depths is not None else None,
        }

    def save(self, json_path: str | Path) -> tuple[Path, Path]:
        """Write ``<name>.json`` (definition) and ``<name>.exit_pdf.npz`` (exit schedule)."""

        json_path = Path(json_path)
        pdf_path = (
            json_path.with_suffix("").with_suffix(_PDF_SUFFIX)
            if json_path.suffix
            else Path(str(json_path) + _PDF_SUFFIX)
        )
        json_path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {"offsets": self.offsets.astype(np.int64)}
        if self.exit_pdf is not None:
            arrays["exit_pdf"] = self.exit_pdf.astype(np.float16)
        elif self.exit_values is not None:
            arrays["exit_values"] = self.exit_values.astype(np.float16)
        else:
            arrays["exit_depths"] = self.exit_depths.astype(np.int32)
        np.savez(pdf_path, **arrays)
        definition = {
            "meta": {**self.meta, "num_requests": self.num_requests, "total_steps": self.total_steps},
            "exit_pdf_file": pdf_path.name,
            "requests": [
                {"id": self.ids[i], "input_len": int(self.input_lens[i]), "output_len": int(self.output_lens[i])}
                for i in range(self.num_requests)
            ],
        }
        json_path.write_text(json.dumps(definition, separators=(",", ":")))
        return json_path, pdf_path

    @classmethod
    def load(cls, json_path: str | Path) -> Workload:
        """Load a workload from its JSON definition and sibling exit-schedule npz."""

        json_path = Path(json_path)
        definition = json.loads(json_path.read_text())
        requests = definition["requests"]
        ids = [str(r["id"]) for r in requests]
        input_lens = np.array([int(r["input_len"]) for r in requests], dtype=np.int32)
        output_lens = np.array([int(r["output_len"]) for r in requests], dtype=np.int32)
        meta = definition.get("meta", {})
        pdf_path = json_path.parent / definition["exit_pdf_file"]
        with np.load(pdf_path) as payload:
            keys = set(payload.files)
            offsets = payload["offsets"]
            exit_pdf = payload["exit_pdf"] if "exit_pdf" in keys else None
            exit_values = payload["exit_values"] if "exit_values" in keys else None
            exit_depths = payload["exit_depths"] if "exit_depths" in keys else None
        return cls(
            ids=ids,
            input_lens=input_lens,
            output_lens=output_lens,
            offsets=offsets,
            exit_pdf=exit_pdf,
            exit_values=exit_values,
            exit_depths=exit_depths,
            max_depth=int(meta["total_steps"]) if exit_depths is not None else None,
            meta=meta,
        )
