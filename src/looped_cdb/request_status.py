# Copyright 2025 The HuggingFace Inc. team
# SPDX-License-Identifier: Apache-2.0
# Modified for looped-model serving and continuous depth batching.
# See THIRD_PARTY_NOTICES.md and LICENSES/Apache-2.0.txt.

"""The request lifecycle enum shared by the continuous-batching and continuous-depth-batching engines.

Both engines run the same request lifecycle (a prompt is admitted, optionally prefills in chunks, then
decodes one token at a time until it stops), so they schedule against one status enum. Keeping a single
definition lets :mod:`looped_cdb.scheduler_base` reason about request status without importing either
engine, and makes a status from one engine comparable with a status from the other.
"""

from enum import IntEnum


class RequestStatus(IntEnum):
    """Status of a generation request through its lifecycle."""

    PENDING = 0
    PREFILLING = 1
    DECODING = 2
    FINISHED = 3
