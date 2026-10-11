# 固定 K 的调度 headroom：最小 profiling 实验

目标：测同一 workload 的最佳固定 coda 阈值与 offline 调度之间的时间差，再决定是否研究在线算法。GPU 提供阶段成本，CPU 在同一条有限 token trace 上比较 best static K 与 exact offline OPT。

## 1. 模型与 batch 语义

`max_num_seqs=B` 是驻留请求上限，同时限制单次 launch 宽度。退出 core 后，请求仍占驻留槽位；完成整条 token 链才离开。各策略固定相同的 B、KV 配置和成本曲线。

模拟器中，`n` 为剩余驻留链数，`b` 为可执行 core 的 token 数，`q` 为待 coda 数。操作边界上 `b+q=n≤B`：

- **Core**：全部 b 个 token 各推进一次 loop，耗时 `core_ms[b]`；归零者进入 q。
- **Boundary**：一次处理全部 q 个待 coda token。只有 coda 完成并执行后继 prelude，下一 token 才能进入 core。若有 ℓ 个后继，耗时 `coda_ms[q]+prelude_ms[ℓ]`。
- **固定 K**：q≥K 或 b=0 时执行 Boundary，否则执行 Core。

从初始 prelude 计到最终 coda，所有策略完成相同的有限 token 链，不补入额外请求。先用固定 context 的串行 stage 成本观察这一权衡；实际服务中的异步 gate/readback、prefill 接纳和 context 增长另行测量。仓库异步实现会在旧 coda 读回前启动后继，不能直接把其 in-flight 计数相加来构造 q。

## 2. Offline OPT 怎么算

默认 oracle 是 `feedback_dp.py` 的 exact DP。Boundary flush 全部 pending token，因此 Boundary 后 q=0，所有链的 head 都为正。下一次 Boundary 前只能连续执行 Core；调度可写成 `Core^h₁ → Boundary → Core^h₂ → Boundary → …`。

DP 状态 `S=tuple(sorted(chains))` 保留每条链的当前剩余深度与完整后继链。设当前 head 为 dᵢ，枚举所有整数 `h∈[min dᵢ,max dᵢ]`。第 r 轮 Core 的 batch 是 `bᵣ=|{i:dᵢ>r}|`；经过 h 轮后，`qₕ=|{i:dᵢ≤h}|`，其中有后继的数量为 ℓₕ。

$$
V(\emptyset)=0,\qquad
V(S)=\min_h\left[
\sum_{r=0}^{h-1}C_{\mathrm{core}}(b_r)
+C_{\mathrm{coda}}(q_h)+C_{\mathrm{prelude}}(\ell_h)+V(S_h)
\right].
$$

`Sₕ` 的构造：未退出者 head 减 h；已退出者删除 head，释放 tail；terminal 链消失。完整 OPT 是 `prelude_ms[初始链数]+V(S₀)`。每个宏动作至少完成一个 token，因此状态图无环。实现用显式栈和 memo，长单链不受 Python recursion limit 限制；每组候选 h 复用预计算的 Core 成本前缀。可行的 wait-for-all 后缀给出上界，剩余阶段工作量与单链串行需求给出下界，用于精确剪枝。

DP 输出精确 runtime、宏动作序列、重放后的标准 schedule、`states_evaluated`、`pruned_segments` 和 solver wall time。独立逐步穷举与完成搜索的 A* 用于交叉验证；测试 fixture 的 OPT=30，best static=32。

CLI 默认在 DP 开始、每隔 5 秒和结束时打印进度到 stderr，并立即 flush：`solved_states` 是已精确求值的状态数，`feasible_cache` 是可行后缀缓存条目数，`depth` 是当前搜索栈深度，`segments`、`pruned`、`memo_hits` 分别是检查的宏动作数、剪枝数和精确 memo 命中数。另有 static 扫描和报告导出阶段的提示。可达状态总数未知，这些计数不代表完成百分比。

原 A* 保留，显式指定 `--oracle astar` 后，`--oracle-max-states` 与 `--oracle-time-limit` 才生效。它触及预算时仍报告 `L≤OPT≤U`；此时 gap 是 `[T/U,T/L]`。DP 不使用这两个预算，完成后报告 `L=U=OPT` 与精确 `T/OPT`。异质链仍可能产生较大的 DP 状态空间。

每个候选调度独立决定 successor 的释放时刻；OPT 不回放固定 K 产生的退出到达时刻。

## 3. 本地运行

无需 GPU，模拟与无图报告只使用 Python 标准库；在 repo 根目录运行（Python 3.12+）：

```bash
PYTHONPATH=src python3 scripts/simulate_feedback_batching.py \
  --costs tests/benchmarks/fixtures/feedback_costs.json \
  --trace tests/benchmarks/fixtures/feedback_trace.json \
  --all-k --no-plots --output-dir results/feedback/local_smoke
```

该 fixture 的 best static 为 32、OPT 为 30，验证导出和反馈逻辑；成本为测试设定值，输出保留其来源标签。检查真实随机模型的 2/4/2 阶段接线需安装仓库对应的 torch、transformers：

```bash
PYTHONPATH=src python3 scripts/profile_feedback_stages.py --cpu-smoke
```

定向验证：

```bash
python3 -m pytest -q tests/benchmarks/test_feedback.py \
  tests/benchmarks/test_feedback_oracle.py \
  tests/benchmarks/test_feedback_dp.py \
  tests/benchmarks/test_feedback_report.py \
  tests/benchmarks/test_synthetic_stages.py
```

## 4. GPU 服务器实验

使用仓库已安装的 uv 环境。首次实验测 GPU 成本，再运行 CPU DP：

```bash
uv run --no-sync python scripts/profile_feedback_stages.py \
  --backend flash_attention_3 \
  --output results/feedback/stage_costs.json

uv run --no-sync python scripts/simulate_feedback_batching.py \
  --costs results/feedback/stage_costs.json \
  --sequences 64 --tokens-per-sequence 4 \
  --distribution geometric --mean-depth 4 --max-depth 32 --seed 0 \
  --oracle dp --all-k \
  --output-dir results/feedback
```

GPU profiler 不变：随机权重 Huginn 2/4/2，约 1B 参数；B=64、context=1024、paged FA3、bfloat16、single KV、decode CUDA graph。固定容量逐一测有效 batch 1–64，记录 graph 补齐宽度及预热后样本。Coda 包括 LM head 和 argmax；CUDA event 在 compute stream 上计时。Boundary 展示曲线为 coda(k)+prelude(k)，模拟按实际后继数 ℓ 计费。

Synthetic depth 采用 `min(sampled_depth,max_depth)`，默认 32，与当前 runtime 的有限循环上限一致。`trace.json` 和 `report.json.trace_metadata` 记录分布、seed、截断前均值、max_depth 和 clipped_tokens；report 同时记录实际最大深度与样本均值。默认 geometric/mean=4/seed=0 的 64×4 trace 最大深度为 26，不受新增截断影响。

**已完成 GPU 测量的实验只需重算 CPU，沿用原 trace：**

```bash
uv run --no-sync python scripts/simulate_feedback_batching.py \
  --costs results/feedback/stage_costs.json \
  --trace results/feedback/trace.json \
  --oracle dp --all-k \
  --output-dir results/feedback/dp
```

导入旧 trace 不会重新采样或截断。每个输出目录包含 `report.json`、`summary.md`、`static.csv`、保存的 `costs.json` / `trace.json`、每个 K 与 oracle 的 schedule CSV，以及成本曲线与 K 对比图（PNG/SVG）。DP 的 `gap.lower=gap.upper=T_best_static/OPT`；检查 `all_thresholds_scanned=true`，确认分子是全部 K 中的最佳值。

### 链长对 static K 的诊断

沿用已有 GPU 成本与已保存的 4-token trace，运行：

```bash
uv run --no-sync python scripts/diagnose_feedback_chain_length.py \
  --costs results/feedback/stage_costs.json \
  --trace results/feedback/trace.json \
  --output-dir results/feedback/chain_length
```

入口比较原短链与 64-token 长链，后者保留每条链的原始前缀，追加 geometric(mean=4)、max_depth=32、seed=1 的后续 token。两组都扫描 K=1..B，跳过所有 oracle。`comparison.json` 记录每组最佳 K、时间、每 token 时间和 T(16)/T(best)；`short/` 与 `long/` 各自保存报告、trace、static CSV 和曲线。static 扫描每完成 8 个 K 打印进度。通用入口也支持 `--oracle none`，此时 report 的 `oracle` 和 `gap` 都为 null。

## 5. 保留的反馈理论雏形

饱和阶段 `n=B`、`q=B-b`。若成本可近似为 `C(b)=a_C+p_C b`、`G(k)=a_B+p_B k`，对固定 core visits R 和配对 Boundary token 数 H，有：

$$
T(\pi)=\left(p_C+\frac{a_C}{B}\right)R+p_BH
+\frac{a_C}{B}\sum_{t\in\mathrm{Core}}q_t+a_BN_B.
$$

调度影响的是空缺 slot 在各轮累积的成本与 Boundary 启动次数。可尝试 DebtFlush：每轮累计 `a_C q/B`，达到 `a_B` 后 flush 并清零；core 为空时立即 flush。有限链的 drain 阶段用总空缺 `B-b`，末次 coda 不执行后继 prelude，实验仍直接累加阶段曲线。

上式是给定假设下的代数分解；DebtFlush 的性能、最优性与竞争比尚未建立。反馈释放让 ALG 与 OPT 面对不同的到达时刻，外生到达的 dynamic batching 证明不能直接移用。

## 6. 参考

- [CDB，arXiv:2608.09444](https://arxiv.org/abs/2608.09444)：固定 K 与 core underfill / coda batching 权衡。
- [Dynamic Batching of Online Arrivals to Leverage Economies of Scale，arXiv:2309.16911](https://arxiv.org/abs/2309.16911)：外生到达下的 batching 理论。
- 可选：[CacheFlow，arXiv:2604.25080](https://arxiv.org/abs/2604.25080)：硬件成本驱动的 KV restoration 规划。
