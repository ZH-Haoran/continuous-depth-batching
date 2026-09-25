"""Request lifecycle state for continuous depth batching.

Extends the continuous batching request state (``looped_cdb.continuous_batching.requests``)
with per-token exit-depth tracking, and adds the work items the depth scheduler queues: each
decode token runs as a chain of single recurrent steps, so the engine tracks exit decisions
per token and per step instead of per sequence.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from looped_cdb.benchmarks import nvtx
from looped_cdb.continuous_batching.requests import (
    TMP_TOKEN_ID,
    FutureRequestState,
    GenerationOutput,
    logger,
)
from looped_cdb.continuous_batching.requests import RequestState as CBRequestState
from looped_cdb.request_status import RequestStatus

from .exit_policy import TokenExitPolicyState

if TYPE_CHECKING:
    import torch

__all__ = [
    "TMP_TOKEN_ID",
    "DepthWorkItem",
    "FutureRequestState",
    "GenerationOutput",
    "PendingGateResult",
    "RequestState",
    "RequestStatus",
    "logger",
]


# repr=False keeps the base's compact __repr__; the generated one would dump full token lists.
@dataclass(repr=False)
class RequestState(CBRequestState):
    """CB request state plus per-token exit-depth tracking for depth scheduling."""

    current_token_exit_depth: int | None = None
    exit_depths: list[int] = field(default_factory=list)
    # Replayed exit schedule: one 0-based recurrent step per staged decode token,
    # consumed in order via ``next_synthetic_exit_depth`` when the engine drives
    # exits from a recorded workload instead of the live gate. Empty when unused.
    synthetic_exit_depths: list[int] = field(default_factory=list)
    _synthetic_exit_cursor: int = 0

    def reset_depth_state_for_next_token(self) -> None:
        """Reset request-visible depth output for the next decode token."""

        self.current_token_exit_depth = None

    def next_synthetic_exit_depth(self) -> int | None:
        """Pop this request's next replayed exit step, or ``None`` if none is set.

        Returns the 0-based recurrent step for the current decode token and advances the
        cursor. Raises if the schedule is exhausted before the request finishes (one step is
        expected per decode token). Under ``replay_eos_finishes`` the final token's doomed
        successor is staged one past the schedule; it repeats the last recorded depth, which only
        shapes the wasted work computed before its cancellation. Reading further still raises, so
        a replay desync does not hide behind the flag.
        """

        if not self.synthetic_exit_depths:
            return None
        if self._synthetic_exit_cursor >= len(self.synthetic_exit_depths):
            if self.replay_eos_finishes and self._synthetic_exit_cursor == len(self.synthetic_exit_depths):
                self._synthetic_exit_cursor += 1
                return self.synthetic_exit_depths[-1]
            raise RuntimeError(
                f"Request {self.request_id}: synthetic exit-depth schedule exhausted; "
                "expected one exit depth per decode token"
            )
        exit_depth = self.synthetic_exit_depths[self._synthetic_exit_cursor]
        self._synthetic_exit_cursor += 1
        return exit_depth

    def prepare_for_offload_requeue(self) -> None:
        """Reset a just-offloaded request so it resumes decoding from its restored KV.

        On top of the base reset, the pending token is re-processed through the recurrent decode
        pipeline from step 0 - never the full-depth prefill path - so the resume cost is a normal
        exit-depth decode, matching the CB engine's decode-first resume. The per-token depth
        accumulators are reset so recurrence restarts cleanly; the synthetic exit-schedule cursor
        is left untouched so the replay stays aligned.
        """

        super().prepare_for_offload_requeue()
        self.reset_depth_state_for_next_token()

    def prompt_len(self) -> int:
        """Length of the true prompt, before any soft reset folded generated tokens into it.

        ``create_equivalent_initial_request`` rebuilds a preempted request with its generation folded
        onto ``initial_tokens`` and an empty ``generated_tokens``, recording the original prompt length
        in ``_true_initial_tokens``. So ``len(initial_tokens)`` overstates the prompt of a request that
        has been preempted, and ``len(generated_tokens)`` understates its generation.
        """

        return self._true_initial_tokens or len(self.initial_tokens)

    def has_grown_past_prompt(self) -> bool:
        """Whether this request holds KV beyond its prompt, i.e. whether its KV is still growing."""

        return self.current_len() > self.prompt_len()

    def create_equivalent_initial_request(self) -> "RequestState":
        """Rebuild this request as a fresh prefill for soft-reset preemption (recompute).

        On top of the base rebuild, the exit schedule is sliced from ``cursor + 1``, not ``cursor``:
        the pending decode token at the preempt point is the last of the folded ``generated_tokens``,
        and its recorded exit depth (``synthetic_exit_depths[cursor]``) is the depth it *would* have
        decoded at. Recompute instead recomputes it at full depth in the prefill, so that step is
        spent and the first decode after resume is the *next* token, which must consume
        ``synthetic_exit_depths[cursor + 1]``. Slicing from ``cursor`` would shift every remaining
        depth by one and make recompute replay a different schedule than offload (which keeps the
        pending token as a recurrent decode).
        """

        new_state = super().create_equivalent_initial_request()
        new_state.synthetic_exit_depths = self.synthetic_exit_depths[self._synthetic_exit_cursor + 1 :]
        return new_state


@dataclass
class PendingGateResult:
    """Host-visible delayed scalar exit signal for one recurrent work item."""

    host_logits: "torch.Tensor"
    batch_index: int
    recurrent_step: int
    ready_event: Any | None = None
    keepalive: Any | None = None

    def signal(self) -> "torch.Tensor":
        """Wait only for this signal's copy event and return its scalar value."""

        if self.ready_event is not None:
            # Only a wait that actually blocks appears in the trace; the query itself is a CUDA
            # call per item, so the unprofiled hot path skips it entirely.
            if nvtx.is_enabled() and not self.ready_event.query():
                with nvtx.range("cdb.gate_wait"):
                    self.ready_event.synchronize()
            else:
                self.ready_event.synchronize()
        return self.host_logits[self.batch_index]


@dataclass
class DepthWorkItem:
    """One scheduled recurrent-step work item for continuous depth batching.

    ``recurrent_step`` and ``token_position`` are 0-based.  Sequence position is
    advanced once, after coda sampling, not after each recurrent step.
    """

    state: RequestState
    token_id: int
    token_position: int
    recurrent_step: int
    policy_state: TokenExitPolicyState = field(default_factory=TokenExitPolicyState)
    synthetic_exit_depth: int | None = None
    apply_exit_gate: bool = True
    hidden_slot: int | None = None
    pending_gate: PendingGateResult | None = None
    pending_synthetic_exit_step: int | None = None
    # 0-based exit step a pre-loop gate chose from this token's embedding, before any
    # recurrent step ran. Set at prelude time; None for the per-step hazard gates.
    preloop_exit_step: int | None = None
