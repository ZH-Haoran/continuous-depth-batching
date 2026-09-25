"""Real CUDA smoke tests for automatic KV sizing and graph warmup."""

import gc

import pytest
import torch

from looped_cdb.continuous_batching import ContinuousBatchingEngine
from looped_cdb.continuous_batching.config import ContinuousBatchingConfig
from looped_cdb.continuous_depth_batching import ContinuousDepthBatchingConfig, ContinuousDepthBatchingEngine
from looped_cdb.models.ouro import OuroConfig, OuroForCausalLM
from looped_cdb.models.ouro.kv_cache_policy import configure_kv_cache_policy

pytestmark = pytest.mark.cuda


@pytest.fixture(scope="module")
def ouro_model() -> OuroForCausalLM:
    config = OuroConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=256,
        total_ut_steps=4,
        layer_types=["full_attention"] * 2,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        exit_gate_type="early_exit",
    )
    model = OuroForCausalLM(config).to("cuda", dtype=torch.bfloat16).eval()
    model.set_attn_implementation("paged|flash_attention_3")
    configure_kv_cache_policy(model, kv_policy="single")
    return model


def _fraction_for_small_cache() -> float:
    gc.collect()
    torch.cuda.empty_cache()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return (total_bytes - free_bytes + 512 * 1024**2) / total_bytes


def test_cb_auto_cache_survives_graph_warmup(ouro_model: OuroForCausalLM) -> None:
    engine = ContinuousBatchingEngine.from_model(
        ouro_model,
        ContinuousBatchingConfig(
            mem_fraction_static=_fraction_for_small_cache(),
            block_size=16,
            max_num_batched_tokens=16,
            max_num_seqs=4,
            max_model_len=64,
        ),
    )
    assert engine.cache.num_blocks > 0
    assert engine.cb_config.num_blocks == engine.cache.num_blocks
    outputs = engine.generate_batch([[1, 5, 7], [1, 9]], max_new_tokens=3, eos_token_id=None, warmup=True)
    assert [len(output.generated_tokens) for output in outputs] == [3, 3]


def test_cdb_multi_slot_auto_cache_survives_graph_warmup(ouro_model: OuroForCausalLM) -> None:
    engine = ContinuousDepthBatchingEngine.from_model(
        ouro_model,
        ContinuousDepthBatchingConfig(
            mem_fraction_static=_fraction_for_small_cache(),
            block_size=16,
            max_num_batched_tokens=16,
            max_num_seqs=4,
            max_model_len=64,
            max_recurrent_steps=4,
            kv_policy="first_then_shared",
            synthetic_exit_replay=True,
        ),
    )
    assert engine.cache.num_blocks > 0
    assert engine.cache.num_layers == 4
    outputs = engine.generate_batch(
        [[1, 5, 7], [1, 9]],
        max_new_tokens=3,
        eos_token_id=None,
        exit_depths=[[0, 2], [1, 0]],
        warmup=True,
    )
    assert [len(output.generated_tokens) for output in outputs] == [3, 3]
