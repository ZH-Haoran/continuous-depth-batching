# Paper experiments

## Overview

### Data preparation

| Script | Purpose |
| --- | --- |
| `workload.sh` | Creates ShareGPT, Alpaca, and ArXiv workload bundles containing request lengths and threshold-free exit trajectories; all models' bundles preserve the same requests and replay order |

### Main experiments

| Script | Measures | Paper |
| --- | --- | --- |
| `benchmark_decode_step_latency.sh` | Per-recurrent-step decode latency against batch size and context length | **Sec. 5, Fig. 4:** "Efficiency Limits of Adaptive Depth"; **Appendix B, Fig. 9:** "Recurrent-Step Latency Details" |
| `throughput_exit-sweep.sh` | Offline throughput against the exit threshold, at a fixed maximum decode batch size | **Sec. 6.2, Fig. 5 left:** "Offline Throughput" |
| `throughput_width-sweep.sh` | Offline throughput against the maximum decode batch size, at a fixed exit threshold | **Sec. 6.2, Fig. 5 right:** "Offline Throughput" |
| `serving_rate-sweep.sh` | Normalized serving latency against the offered Poisson request rate | **Sec. 6.3, Fig. 6:** "Online Serving"; **Appendix D, Figs. 12 and 13:** "Complete Online Serving Results" |

### Ablations

| Script | Measures | Paper |
| --- | --- | --- |
| `ablations/benchmark_scheduling_overhead.sh` | Profiles GPU idle time under synchronous scheduling, asynchronous scheduling, and lookahead | **Sec. 4.3, Fig. 3 right:** "Asynchronous Scheduling" |
| `ablations/profile_nsys.sh` | Profiles CB and CDB GPU activity, launch-site timing, and host waits | **Sec. 6.2, Fig. 5:** "Offline Throughput" (timing for prefill-adjusted bound); supporting diagnostics |
| `ablations/min_coda_batch.sh` | Ablates the minimum coda batch size for offline throughput | **Sec. 6.4, Fig. 7 left three panels:** "Ablations" |
| `ablations/layer_split_huginn.sh` | Ablates Huginn's prelude-core-coda layer split (0-4-0, 1-4-1) | **Sec. 6.4, Fig. 7 right:** "Ablations" |
| `ablations/schedule_trace.sh` | Traces the executed scheduler stages for both CDB modes | **Appendix A, Tabs. 2 and 3:** "Scheduling Examples" |
| `ablations/accuracy_gsm8k_ouro.sh` | GSM8K accuracy sweep for Ouro: fixed depths and exit thresholds | **Appendix E, Tab. 4, Fig. 14 top:** "Accuracy-Efficiency Trade-off Ablations" |
| `ablations/accuracy_gsm8k_huginn.sh` | GSM8K accuracy sweep for Huginn: fixed depths and exit thresholds | **Appendix E, Tabs. 4 and 5, Fig. 14 bottom:** "Accuracy-Efficiency Trade-off Ablations" |
| `ablations/train_ouro_gate.sh` | Trains the same-step, lookahead, and preloop Ouro exit gates | **Appendix E.2, Fig. 14 top:** "Early-exit Gating" |

## Details

### `benchmark_decode_step_latency.sh`

- Runs `scripts/benchmark_decode_step_latency.py` with Ouro 1.4B, Ouro 2.6B, or Huginn 3.5B.
- Writes a JSONL file with sweep results to `outputs/decode_step_latency/`.
    - Each batch-size and context-length point contains best-window latencies, a regression fit, and diagnostics.
- Follow-up: `scripts/exporters/export_decode_step_latency_results.py` writes a JSON file containing the data used to generate the paper figures.
    - It filters the data and computes the weight-reload cost, saturation batch `B*`, and per-sequence compute and KV-streaming slope.

### `throughput_exit-sweep.sh`

- Runs `scripts/benchmark_throughput.py` for CB, CDB no-refill, and CDB with refill at multiple exit thresholds.
- Writes one JSONL file per model and dataset to `outputs/exit-sweep/`.
- Each row records whole-run throughput and throughput inside a steady-state window.
- Follow-up: `scripts/exporters/export_exit_sweep_results.py` writes a JSON file containing the data used to generate the paper figure.
    - It exports throughput normalized to CB and the decode FLOP bound over the threshold range.
    - Passing `--cb-profile-root outputs/nsys` also exports the prefill-adjusted end-to-end bound (requires running `ablations/profile_nsys.sh` first).

### `throughput_width-sweep.sh`

- Runs `scripts/benchmark_throughput.py` for CB, CDB no-refill, and CDB with refill at multiple maximum decode batch sizes with a fixed exit threshold.
- Writes one JSONL file per model and dataset to `outputs/width-sweep/`.
- Follow-up: `scripts/exporters/export_width_sweep_results.py` writes a JSON file containing the data used to generate the paper figure.
    - It exports throughput normalized to CB, the decode FLOP bound, and the saturation batch `B*` (requires running `benchmark_decode_step_latency.sh` first).
    - Passing `--cb-profile-root outputs/nsys` also exports the width-dependent prefill-adjusted end-to-end bound (requires running `ablations/profile_nsys.sh` first).

### `serving_rate-sweep.sh`

- Runs an open-loop serving benchmark with Poisson arrivals for CB, CDB no-refill, and CDB with refill.
- Writes one JSONL file per model and dataset to `outputs/serving-rate/`.
- Follow-up: `scripts/exporters/export_serving_results.py` combines one model's dataset summaries and writes `serving_ouro.json` or `serving_huginn.json` for the paper figure.
    - It exports normalized latency, TTFT (including P99), TPOT, queueing, throughput, batch-size, and residency metrics.

### `ablations/benchmark_scheduling_overhead.sh`

- Runs `scripts/benchmark_throughput.py` under Nsight Systems for CDB with synchronous scheduling, asynchronous scheduling with immediate gate consumption, and asynchronous scheduling with lookahead.
- Writes Nsight `.nsys-rep` traces, SQLite exports, benchmark summaries, and analyzed GPU timing files to `outputs/ablations/scheduling-overhead/`.
- Follow-up: `scripts/exporters/export_scheduling_overhead_results.py` writes the validated idle fractions to a tex file.

### `ablations/profile_nsys.sh`

- Runs `scripts/benchmark_throughput.py` under Nsight Systems for CB, CDB without refill, and CDB with refill at different maximum decode batch sizes.
- Writes Nsight `.nsys-rep` traces, SQLite exports, benchmark summaries, and analyzed GPU timing files to `outputs/nsys/`.
- Profiles can provide the prefill-adjusted end-to-end bounds exported for Fig. 5.

### `ablations/min_coda_batch.sh`

- Runs the ShareGPT experiment from `throughput_exit-sweep.sh` with different minimum coda batch sizes.
- Runs only CDB with refill.
- Sweeps minimum coda batch sizes and maximum decode batch sizes.
    1. The Ouro experiment uses a maximum decode batch size of 64.
    2. The Huginn experiment sweeps multiple maximum decode batch sizes.
- Writes one JSONL file per model to `outputs/ablations/min-coda-batch/`.
- Follow-up: `scripts/exporters/export_coda_ablation_results.py` combines the minimum-coda and layer-split results.

### `ablations/layer_split_huginn.sh`

- Runs the ShareGPT experiment from `throughput_exit-sweep.sh` with Huginn's boundary stages removed (0-4-0) or halved (1-4-1).
- Uses the same recorded workloads for all layer splits.
- Writes one JSONL file to `outputs/ablations/layer-split/`.
- Follow-up: `scripts/exporters/export_coda_ablation_results.py` combines the minimum-coda and layer-split results.

### `ablations/schedule_trace.sh`

- Runs `scripts/benchmark_throughput.py` for CDB no-refill and refill at width 64, with minimum coda batch sizes 1 and 16 for refill and tracks the executed scheduler stages.
- Writes one JSONL trace per run to `outputs/ablations/schedule-trace/`.
- Follow-up: `scripts/export_schedule_trace.py` produces LaTeX tables visualizing the traces.

### `ablations/accuracy_gsm8k_ouro.sh` and `ablations/accuracy_gsm8k_huginn.sh`

- Run `scripts/evaluate_accuracy.py` for GSM8K.
- Fixed-depth rows use CB, and early-exit rows use CDB.
- Write one JSONL file per model to `outputs/ablations/accuracy/`.
- Follow-up: `scripts/exporters/export_accuracy_results.py` writes a JSON file containing the data used to generate the paper figure and table.
    - It exports accuracy-depth curves, exit distributions, and fixed-depth KV-layout accuracy and cache size.

### `ablations/train_ouro_gate.sh`

- Runs `scripts/train_ouro_gate.py` with Ouro 1.4B to distill lookahead and preloop exit gates from the frozen built-in gate, together with a same-step diagnostic control.
- Trains on packed math, code, STEM, and chat examples from the Nemotron post-training dataset.
- Writes the three trained gate weights and `gate_training_config.json` to `outputs/ablations/ouro_gate/`.
- Follow-up: run `ablations/accuracy_gsm8k_ouro.sh` to evaluate the lookahead and preloop gates.
