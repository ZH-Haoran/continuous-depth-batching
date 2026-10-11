"""Random-weight Huginn stage profiling at a fixed CDB resident cap.

No checkpoint, tokenizer, workload or exit gate is needed. CUDA timings cover
runner stage launches on the compute stream, including paged KV and graph
padding. Coda includes projection and argmax, but excludes D2H/host consumption.
"""

from __future__ import annotations

import math
import platform
import statistics
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any

from .feedback import CostTable


@dataclass(frozen=True)
class SyntheticStageConfig:
    resident_cap: int = 64
    context_length: int = 1024
    hidden_size: int = 3072
    num_heads: int = 24
    intermediate_size: int = 8192
    vocab_size: int = 32768
    prelude_layers: int = 2
    core_layers: int = 4
    coda_layers: int = 2
    dtype: str = "bfloat16"
    backend: str = "flash_attention_3"
    cuda_graph: bool = True
    warmup: int = 4
    repeats: int = 15
    seed: int = 0

    def __post_init__(self) -> None:
        positive = (
            self.resident_cap,
            self.hidden_size,
            self.num_heads,
            self.intermediate_size,
            self.vocab_size,
            self.core_layers,
            self.repeats,
        )
        if any(type(value) is not int or value < 1 for value in positive):
            raise ValueError("capacities, widths, core layers and repeats must be positive integers")
        if self.hidden_size % self.num_heads or (self.hidden_size // self.num_heads) % 2:
            raise ValueError("hidden_size must give an even integer head dimension")
        if any(
            type(value) is not int or value < 0
            for value in (self.context_length, self.prelude_layers, self.coda_layers)
        ):
            raise ValueError("context length and boundary layers must be nonnegative integers")
        if self.vocab_size < 3 or self.warmup < 4:
            raise ValueError("vocab_size must be >=3 and warmup >=4 (capture both rotating stage buffers)")
        if self.dtype not in ("bfloat16", "float16", "float32"):
            raise ValueError("unsupported dtype")
        if self.backend not in ("flash_attention_3", "flash_attention_2"):
            raise ValueError("choose the repo's paged FA3 or FA2 backend")


def build_random_model(settings: SyntheticStageConfig, *, device: str = "cpu", dtype: str = "float32") -> Any:
    """Build locally without downloading weights; CPU uses SDPA for wiring checks."""
    import torch

    from looped_cdb.models.huginn import HuginnConfig, HuginnForCausalLM

    # Huginn has no eager-attention fallback, so Transformers' source heuristic
    # rejects set_attn_implementation(). Select the backend at construction.
    attention = f"paged|{settings.backend}" if device.startswith("cuda") else "sdpa"
    config = HuginnConfig(
        n_embd=settings.hidden_size,
        n_heads=settings.num_heads,
        intermediate_size=settings.intermediate_size,
        vocab_size=settings.vocab_size,
        n_layers=settings.prelude_layers + settings.core_layers + settings.coda_layers,
        n_layers_in_prelude=settings.prelude_layers,
        n_layers_in_recurrent_block=settings.core_layers,
        n_layers_in_coda=settings.coda_layers,
        block_size=max(16, settings.context_length + 2),
        total_recurrent_steps=32,
        state_init="zero",
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        attn_implementation=attention,
        dtype=getattr(torch, dtype),
    )
    torch.manual_seed(settings.seed)
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(getattr(torch, dtype))
        with torch.device(device):
            model = HuginnForCausalLM(config).eval()
    finally:
        torch.set_default_dtype(previous_dtype)
    if model.config._attn_implementation != attention:
        raise RuntimeError(
            f"requested attention {attention!r}, but the model selected {model.config._attn_implementation!r}"
        )
    return model


def cpu_stage_smoke() -> dict[str, Any]:
    """Exercise real 2/4/2 adapter stages with tiny SDPA tensors, without timing."""
    import torch

    from looped_cdb.continuous_depth_batching.adapters.huginn import HuginnCDBAdapter

    settings = SyntheticStageConfig(
        hidden_size=32, num_heads=2, intermediate_size=64, vocab_size=32, context_length=0, resident_cap=2
    )
    model = build_random_model(settings)
    adapter = HuginnCDBAdapter(model)
    positions = torch.zeros((1, 2), dtype=torch.long)
    with torch.inference_mode():
        hidden = adapter.prelude(torch.tensor([[3, 4]]), position_ids=positions)
        hidden, gate = adapter.recurrent_step(hidden, positions, positions, kv_slot=0)
        logits = adapter.lm_head(hidden, position_ids=positions, kv_slot=0)
    if not torch.isfinite(hidden).all() or not torch.isfinite(logits).all():
        raise RuntimeError("nonfinite random-model stage outputs")
    return {
        "kind": "cpu_wiring_smoke",
        "hidden_shape": list(hidden.shape),
        "logits_shape": list(logits.shape),
        "gate_enabled": gate is not None,
    }


def profile_stages(settings: SyntheticStageConfig) -> dict[str, Any]:
    """Measure every effective width at one capacity; fail rather than fill gaps."""
    import torch

    from looped_cdb.continuous_depth_batching import ContinuousDepthBatchingConfig, ContinuousDepthBatchingEngine
    from looped_cdb.continuous_depth_batching.requests import DepthWorkItem, RequestState
    from looped_cdb.request_status import RequestStatus

    if not torch.cuda.is_available():
        raise RuntimeError("GPU profiling requires CUDA; use --cpu-smoke for a local wiring check")
    device = "cuda:0"
    model = build_random_model(settings, device=device, dtype=settings.dtype)
    block_size = 256 if settings.backend == "flash_attention_2" else 16
    blocks_per_request = math.ceil((settings.context_length + 1) / block_size)
    engine = ContinuousDepthBatchingEngine.from_model(
        model,
        ContinuousDepthBatchingConfig(
            block_size=block_size,
            num_blocks=settings.resident_cap * blocks_per_request + 8,
            max_model_len=blocks_per_request * block_size,
            max_num_batched_tokens=settings.resident_cap,
            max_num_seqs=settings.resident_cap,
            max_recurrent_steps=32,
            use_cuda_graph=settings.cuda_graph,
            use_cuda_graph_prefill=False,
            kv_policy="single",
            kv_pressure_mode="none",
            safety_margin=0,
            min_free_slots=1,
        ),
    )
    runner, cache = engine.runner, engine.cache
    if runner.engine_config.max_num_seqs != settings.resident_cap:
        raise RuntimeError("engine changed the resident cap")
    stream = runner.inputs_and_outputs.compute_stream
    assert stream is not None
    items = []
    token_ids = torch.randint(0, settings.vocab_size, (settings.resident_cap,), device="cpu").tolist()
    for index in range(settings.resident_cap):
        token = token_ids[index]
        state = RequestState(str(index), [token], tokens_to_process=[token], position_offset=settings.context_length)
        state.status = RequestStatus.DECODING
        allocated = cache.allocate_blocks(blocks_per_request, state.request_id, 0)
        if allocated is None:
            raise RuntimeError("insufficient KV capacity")
        state.allocated_blocks = allocated
        items.append(DepthWorkItem(state, token, settings.context_length, 0, apply_exit_gate=False))
    with torch.inference_mode():
        for tensor in (*cache.key_cache, *cache.value_cache):
            tensor.uniform_(-0.02, 0.02)
        torch.cuda.synchronize()
        runner.compute_prelude_batch(items)
        torch.cuda.synchronize()

        def measure(call: Any) -> tuple[float, list[float]]:
            for _ in range(settings.warmup):
                keepalive = call()
                stream.synchronize()
                del keepalive
            before = (runner.stage_graph_captures, runner.recurrent_graph_captures)
            samples = []
            for _ in range(settings.repeats):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record(stream)
                keepalive = call()
                end.record(stream)
                end.synchronize()
                del keepalive
                samples.append(start.elapsed_time(end))
            if before != (runner.stage_graph_captures, runner.recurrent_graph_captures):
                raise RuntimeError("graph capture occurred inside the timing window")
            return statistics.median(samples), samples

        curves: dict[str, list[float]] = {name: [0.0] for name in ("core_ms", "coda_ms", "prelude_ms")}
        diagnostics = []

        def prelude(selected: list[Any]) -> None:
            # Prelude assigns new hidden slots; recycle the old assignments
            # before each repeat, keeping the bank capacity and pointers fixed.
            runner.release_hidden_slots(selected)
            runner.compute_prelude_batch(selected)

        def coda(selected: list[Any]) -> Any:
            # No D2H reader: all coda buffers are consumed on the same stream.
            # Keep eager staging allocations alive until the timing end event.
            with torch.cuda.stream(stream):
                result = runner._run_coda_forward(selected)
                runner._record_coda_buffer_reuse(None)
            return result

        for batch in range(1, settings.resident_cap + 1):
            selected = items[:batch]
            calls = {
                "prelude_ms": partial(prelude, selected),
                "core_ms": partial(runner.compute_recurrent_batch, selected),
                "coda_ms": partial(coda, selected),
            }
            row: dict[str, Any] = {"batch_size": batch, "kernel_batch_size": runner._stage_bucket_size(batch) or batch}
            for name, call in calls.items():
                median, samples = measure(call)
                curves[name].append(median)
                row[name] = {"samples_ms": samples, "min_ms": min(samples), "max_ms": max(samples)}
            diagnostics.append(row)
            print(f"profiled effective batch {batch}/{settings.resident_cap}", flush=True)
    metadata = {
        "source": "measured_cuda_stages",
        "settings": asdict(settings),
        "timing_basis": "compute_stream_cuda_events",
        "initial_context_length": settings.context_length,
        "context_policy": "fixed_one_token_decode_position",
        "coda_scope": "coda_layers_lm_head_argmax_without_D2H",
        "boundary_scope": "coda_plus_successor_prelude_serial",
        "kv_policy": "single",
        "parameters": sum(p.numel() for p in model.parameters()),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
    }
    result = CostTable(
        settings.resident_cap, **{name: tuple(values) for name, values in curves.items()}, metadata=metadata
    ).to_dict()
    result["diagnostics"] = diagnostics
    return result
