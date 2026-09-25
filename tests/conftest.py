"""Gate CUDA-marked tests behind an explicit opt-in.

Tests marked ``cuda`` load real checkpoints and need a GPU, so a plain
``pytest`` run stays CPU-only even on a machine with one. They run only when
``LOOPED_CDB_RUN_CUDA_TESTS=1`` and CUDA is available.
"""

import os

import pytest
import torch


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("LOOPED_CDB_RUN_CUDA_TESTS") == "1" and torch.cuda.is_available():
        return
    skip_cuda = pytest.mark.skip(reason="CUDA tests require CUDA and LOOPED_CDB_RUN_CUDA_TESTS=1")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip_cuda)
