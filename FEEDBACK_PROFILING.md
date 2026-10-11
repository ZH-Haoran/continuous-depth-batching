# 固定 K 的调度 headroom：最小 profiling 实验

目标：测同一 workload 的最佳固定 coda 阈值与 offline 调度之间的时间差，再决定是否研究在线算法。当前已实现并通过本地 CPU 验证；GPU 成本曲线与研究结论待服务器实验。

## 1. 模型与 batch 语义

`max_num_seqs=B` 是驻留请求上限，同时限制单次 launch 宽度。退出 core 后，请求仍占驻留槽位；完成整条 token 链才离开。各策略固定相同的 B、KV 配置和成本曲线。

模拟器中，`n` 为剩余驻留链数，`b` 为可执行 core 的 token 数，`q` 为待 coda 数。操作边界上 `b+q=n≤B`：

- **Core**：全部 b 个 token 各推进一次 loop，耗时 `core_ms[b]`；归零者进入 q。
- **Boundary**：一次处理全部 q 个待 coda token。只有 coda 完成并执行后继 prelude，下一 token 才能进入 core。若有 ℓ 个后继，耗时 `coda_ms[q]+prelude_ms[ℓ]`。
- **固定 K**：q≥K 或 b=0 时执行 Boundary，否则执行 Core。

从初始 prelude 计到最终 coda，所有策略完成相同的有限 token 链，不补入额外请求。先用固定 context 的串行 stage 成本观察这一权衡；实际服务中的异步 gate/readback、prefill 接纳和 context 增长另行测量。仓库异步实现会在旧 coda 读回前启动后继，不能直接把其 in-flight 计数相加来构造 q。

## 2. Offline OPT 怎么算

状态不是只有 `(b,q)`，而是每个 sequence 的**当前剩余循环数和完整后继链**。例如 `(0,3,2)` 表示当前 token 等待 coda，后面两个 token 分别需要 3、2 次 loop。Core 只减少正数头部；Boundary 删除零头部并释放后继。同样的 `(b,q)` 可能对应不同未来，不能合并。

设 `S_C`、`S_B` 是这两个动作后的状态，`V(∅)=0`，则：

$$
V(S)=\min\left\{
C_{\mathrm{core}}(b)+V(S_C)\quad(b>0),\quad
C_{\mathrm{coda}}(q)+C_{\mathrm{prelude}}(\ell)+V(S_B)\quad(q>0)
\right\}.
$$

完整 OPT 再加初始 prelude。实现用 A* 搜索这张状态图，以固定 K 的最佳已知调度作为可行上界。每种 stage 的最小 `cost(k)/k` 乘剩余必需工作量，构成下界；同成本、同 context 的 sequence 可交换标签，但完整后继链仍保留。

搜索结束时给出精确 OPT；触及状态数或时间预算时给出 `L≤OPT≤U`，U 对应已导出的完整可行调度。令固定 K 最佳时间为 T，则：

$$
\frac{T}{U}\le\mathrm{Gap}=\frac{T}{\mathrm{OPT}}\le\frac{T}{L}.
$$

若 `T/U` 已明显大于 1，即使搜索未结束，也已证明至少有这么多改进空间。若区间很宽且包含 1，增加搜索预算。**这里的 OPT 优化 Core / flush-all Boundary 两种动作；每个候选调度独立决定后继释放，不能回放某个固定 K 产生的退出到达时刻。** 可处理任意子集的 Boundary 是更大的动作空间，尚未实现。

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
  tests/benchmarks/test_feedback_report.py \
  tests/benchmarks/test_synthetic_stages.py
```

## 4. GPU 服务器实验

使用仓库已有 GPU 环境；新环境可先 `uv sync`。图输出额外需要 `uv pip install matplotlib`；以下直接使用该环境的 Python。

```bash
# 先用小模型验证 GPU / paged attention / graph 的整条测量路径。
.venv/bin/python scripts/profile_feedback_stages.py \
  --resident-cap 4 --context-length 32 \
  --hidden-size 128 --num-heads 1 --intermediate-size 256 --vocab-size 128 \
  --repeats 5 --output results/feedback/gpu_smoke_costs.json

# 正式成本：随机权重 2/4/2，约 1B 参数，固定 B=64、context=1024。
.venv/bin/python scripts/profile_feedback_stages.py \
  --output results/feedback/stage_costs.json

# 第一条 trace：64 条短链，几何分布，平均 loop 数 4。
.venv/bin/python scripts/simulate_feedback_batching.py \
  --costs results/feedback/stage_costs.json \
  --tokens-per-sequence 4 --mean-depth 4 --seed 0 \
  --output-dir results/feedback/geometric

# 补扫全部 K，才得到同一条 trace 的 best static K。
.venv/bin/python scripts/simulate_feedback_batching.py \
  --costs results/feedback/stage_costs.json \
  --trace results/feedback/geometric/trace.json --all-k \
  --oracle-max-states 500000 --oracle-time-limit 120 \
  --output-dir results/feedback/geometric_all_k
```

默认使用 paged FA3、bfloat16、single KV、decode CUDA graph；FA2 环境加 `--backend flash_attention_2`。profiler 固定容量，逐一测有效 batch 1–64，记录实际 graph 补齐宽度、预热后样本、中位数与范围。测量重复覆盖同一 decode position，context 不增长。coda 包括 LM head 和 argmax；CUDA event 计时在 compute stream 上，未包含 D2H 与 host 消费。Boundary 曲线展示 coda(k)+prelude(k)，模拟中按真实后继数 ℓ 计费。

原有 `benchmark_decode_step_latency.py` 改变每点容量，用 CB 深度回归得到斜率与截距；本入口直接调用固定容量的 CDB stage runner，以测到小 batch 和 graph padding 的成本。

可选第二条 trace：相同分布均值、更大方差，取深度 `{1,16}`：

```bash
.venv/bin/python scripts/simulate_feedback_batching.py \
  --costs results/feedback/stage_costs.json --all-k \
  --distribution two_point --mean-depth 4 --high-depth 16 --seed 0 \
  --output-dir results/feedback/two_point
```

两种分布的**生成均值**都为 4，有限样本的实际均值可能不同，可由 `core_visits/token_count` 检查。

每个输出目录包含 `report.json`、`summary.md`、`static.csv`、保存的 `costs.json` / `trace.json`、每个 K 和 oracle 的 schedule CSV，以及成本曲线与 K 对比图（PNG/SVG）。先看来源、`all_thresholds_scanned`、oracle 是否精确及 gap 区间，再判断 headroom。64 条异质链可能无法在预算内求出精确 OPT；上下界仍保留有效信息。

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
