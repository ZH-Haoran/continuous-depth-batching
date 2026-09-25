"""Batch-shaped inputs and outputs for continuous depth batching's prefill path.

The depth engine reuses the continuous batching pipeline
(``looped_cdb.continuous_batching.input_outputs``) for prefill; the recurrent and coda
stages own their static buffers separately (see ``model_runner.RecurrentStageBuffers``)
and use only the streams owned here.
"""

from looped_cdb.continuous_batching.input_outputs import ContinuousBatchingIOs
from looped_cdb.continuous_batching.requests import FutureRequestState

__all__ = ["ContinuousDepthBatchingIOs"]


class ContinuousDepthBatchingIOs(ContinuousBatchingIOs):
    """The CB batch pipeline without the sampled-token carry-over between batches."""

    def prepare_batch_tensors(
        self,
        requests_in_batch: list[FutureRequestState],
        use_decode_fast_path: bool,
        num_q_tokens: int,
        max_kv_read: int,
    ) -> None:
        # The depth engine routes decode tokens through the depth queue (prelude stage), never through
        # this prefill pipeline, so no input prepared here is ever a placeholder. Carrying over between
        # two prefill batches that share a request id -- a soft-reset re-prefill after its original
        # prefill -- would overwrite part of the folded prompt with a stale sampled token.
        self._prev_req_id_to_new_token_position = {}
        self.host_buffers.prepare_batch_tensors(requests_in_batch, use_decode_fast_path, num_q_tokens, max_kv_read)
        self._fill_carry_over_ids()
