"""Static validation of per-request generate_batch inputs for both engines."""

import pytest

from looped_cdb.continuous_batching.continuous_api import ContinuousBatchingEngine
from looped_cdb.continuous_depth_batching.continuous_api import ContinuousDepthBatchingEngine

ENGINES = [ContinuousBatchingEngine, ContinuousDepthBatchingEngine]


@pytest.mark.parametrize("engine_cls", ENGINES)
def test_validate_generate_inputs_accepts_scalar_and_list(engine_cls: type) -> None:
    engine_cls._validate_generate_inputs([[1, 2], [3]], 4)
    engine_cls._validate_generate_inputs([[1, 2], [3]], [4, 2])


@pytest.mark.parametrize("engine_cls", ENGINES)
def test_validate_generate_inputs_rejects_length_mismatch(engine_cls: type) -> None:
    with pytest.raises(ValueError, match="entries"):
        engine_cls._validate_generate_inputs([[1], [2], [3]], [4, 4])


@pytest.mark.parametrize("engine_cls", ENGINES)
def test_validate_generate_inputs_rejects_non_positive_elements(engine_cls: type) -> None:
    with pytest.raises(ValueError, match=">= 1"):
        engine_cls._validate_generate_inputs([[1], [2]], [4, 0])
    with pytest.raises(ValueError, match="positive"):
        engine_cls._validate_generate_inputs([[1]], 0)


@pytest.mark.parametrize("engine_cls", ENGINES)
def test_validate_generate_inputs_rejects_empty_prompt(engine_cls: type) -> None:
    with pytest.raises(ValueError, match="empty prompts"):
        engine_cls._validate_generate_inputs([[1], []], 4)


@pytest.mark.parametrize("engine_cls", ENGINES)
def test_validate_generate_inputs_rejects_no_prompts(engine_cls: type) -> None:
    with pytest.raises(ValueError, match="at least one prompt"):
        engine_cls._validate_generate_inputs([], 4)
