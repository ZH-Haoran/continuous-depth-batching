# Third-party notices

The project license is in `LICENSE`.
The upstream portions identified below retain their original licenses and attribution.
The Apache-2.0 license text is included in `LICENSES/Apache-2.0.txt`.

## Hugging Face Transformers

The following files adapt the continuous batching implementation and generation configuration from [Hugging Face Transformers](https://github.com/huggingface/transformers):

- `src/looped_cdb/continuous_batching/cache.py`
- `src/looped_cdb/continuous_batching/cache_manager.py`
- `src/looped_cdb/continuous_batching/config.py`
- `src/looped_cdb/continuous_batching/continuous_api.py`
- `src/looped_cdb/continuous_batching/input_outputs.py`
- `src/looped_cdb/continuous_batching/model_runner.py`
- `src/looped_cdb/continuous_batching/requests.py`
- `src/looped_cdb/continuous_batching/scheduler.py`
- `src/looped_cdb/continuous_batching/utils.py`
- `src/looped_cdb/continuous_depth_batching/cache.py`
- `src/looped_cdb/continuous_depth_batching/config.py`
- `src/looped_cdb/offloading_manager.py`
- `src/looped_cdb/scheduler_base.py`
- `src/looped_cdb/request_status.py`

These upstream sources use Apache-2.0.
Their copyright notices credit the HuggingFace Inc. team (2022, 2024, 2025, and 2026); `continuous_api.py` also credits NVIDIA CORPORATION (2020).
The corresponding notices are retained in the adapted files.
Modifications support looped models, paged KV layouts, asynchronous scheduling, staged execution, and continuous depth batching.
The CDB engines also extend and reuse these adapted components.

## Ouro

`src/looped_cdb/models/ouro/configuration_ouro.py` and `modeling_ouro.py` adapt the Apache-2.0 [ByteDance/Ouro-1.4B release](https://huggingface.co/ByteDance/Ouro-1.4B), distributed for these experiments through [KristianS7/Ouro-1.4B](https://huggingface.co/KristianS7/Ouro-1.4B).
The upstream configuration credits the Qwen team, Alibaba Group, and the HuggingFace Inc. team (2024).
Modifications add paged attention, recurrent KV layouts, staged execution, exit gates, and compatibility with the installed Transformers API.

## Huginn

`src/looped_cdb/models/huginn/configuration_huginn.py`, `modeling_huginn.py`, and `exit_gates.py` implement the configuration, model computations, and convergence criterion from the Apache-2.0 [tomg-group-umd/huginn-0125 release](https://huggingface.co/tomg-group-umd/huginn-0125).
The local implementation uses paged attention, exposes prelude/core/coda stages, and evaluates exits per token.
`tests/fixtures/huginn_golden.pt` contains numerical reference outputs recorded from the upstream implementation.

## vLLM and SGLang references

The asynchronous scheduling in `src/looped_cdb/continuous_batching/input_outputs.py` follows the structure of [vLLM's model runner](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_model_runner.py), whose attribution is Copyright contributors to the vLLM project and whose license is Apache-2.0.
The breakable CUDA graphs in `src/looped_cdb/continuous_batching/prefill_graph.py` and the decode capture buckets in `utils.py` follow designs from [SGLang](https://github.com/sgl-project/sglang).

## EleutherAI Language Model Evaluation Harness

The eight GSM8K chain-of-thought examples in `src/looped_cdb/eval/gsm8k.py` were copied from the `gsm8k-cot` task in [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
The upstream project uses the MIT License, Copyright (c) 2020 EleutherAI.
Its full license is included in `LICENSES/lm-evaluation-harness-MIT.txt`.

## External dependencies and downloads

Python dependencies, model weights, and datasets obtained separately retain their respective licenses.
Their inclusion in experiment commands does not change those licenses.
