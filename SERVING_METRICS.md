# 先导实验结果解读与 Serving 指标说明

本文对应 `scripts/benchmark_throughput.py` 的可选事件记录与 W&B 上传路径，涵盖 fork 后最早四个 serving 提交，以及之后新增的 11 类机制诊断指标。目标是先用请求可感知的吞吐量和延迟判断结果，再用队列、KV、抢占与 CDB 阶段事件解释结果。这里的首 token 时间是引擎把 token 交到主机的时间，不是 HTTP 服务的网络首字节时间。

最早四个提交的作用不同，不能全部算作新的物理测量：

| 提交 | 增加了什么 | 在本文件中的位置 |
| --- | --- | --- |
| `bfc78bc` | 可选的逐输出 token 时间戳；软重置后保留之前的时间戳 | ITL/TBT、每请求最大 ITL；也是后续抢占空档分析的基础。 |
| `8094ea4` | 各调度模式共用的等待队列采样 | 等待队列时间线和采样峰值。 |
| `1ed1e9d` | 原始事件 JSONL、按运行导出的比较 CSV；由事件重新计算 TTFT、ITL、E2E 等 | 下文“核心结果指标”的计算与本地导出。 |
| `2feecf6` | 在计时结束后将核心标量、队列表和完整事件上传到 W&B | 下文各 W&B 键；并未新增一种计时来源。 |

## 如何获得这些指标

在已有的 benchmark 命令后加上 `--latency-events-dir <目录> --wandb-project <项目名>`。前者开启请求/token 时间戳和诊断事件记录，并写出完整的 `*.events.jsonl`；后者在计时结束后把汇总值、图表及原始事件文件上传到 W&B，且必须与前者一起使用。只需本地事件文件时可以不传 `--wandb-project`。CDB 阶段图不要求另外开启 `--trace-output`。多次运行后，可用 `python scripts/exporters/export_latency_results.py <目录> --output <比较表.csv>` 把核心指标汇总成每运行一行的本地 CSV。

所有事件的 `time_s` 都以本次测量的开始时刻为零点。开环请求的 `arrival_s` 是**计划到达时刻**，不是调度器下一次发现它的时刻，因此发现和接纳的延迟不会被抹掉。W&B 图表最多展示约 10,000 个点；标量用完整事件计算，原始 JSONL 也保存完整事件。事件打点和 CUDA 计时会给测量带来一定开销，比较不同模式时应让它们使用相同的记录设置。

下文的 `gpu_to_cpu` 表示将 KV 从 GPU 移到 CPU；`cpu_to_gpu` 表示恢复。`N` 表示可分配的 KV block 总数，`T` 表示最后一个请求的计划到达时间（相对测量起点）。

## 核心结果指标：先判断是否快、是否稳定

下面的逐请求/逐 token 指标从 JSONL 的 `request` 事件重新计算，`scripts/exporters/export_latency_results.py` 将每个运行写成 CSV 的一行；W&B 使用同一套比较值。分位数 p50/p95/p99 在排序后的样本位置 `(n − 1) × 百分位` 上做线性插值。部分指标在 fork 前已出现在 benchmark 的 `request_latency` summary 中；这里明确指出新增的是时间戳、事件级分布或展示方式。

### 全程请求与输出 token 吞吐量

- **W&B：**`throughput/requests_per_s`、`throughput/output_tokens_per_s`；CSV 为 `requests_per_s`、`output_tokens_per_s`。
- **定义：**分别为 `完成请求数 / wall_time_s` 与 `生成的输出 token 总数 / wall_time_s`。`wall_time_s` 从测量开始到**所有请求完成且 GPU 同步**，包含首个请求到达前的等待和最后到达后的排空。原有 summary 已有这两个量；早期提交只是将它们送入 CSV/W&B。
- **解读：**离线 drain 实验可用来粗估处理能力；开环实验必须同时看“到达窗口内 token 速率”和“最后到达时未完成请求数”。如果队列在到达阶段持续增长，全程均值可能被排空阶段掩盖。
- **边界：**`request_rate_rps` 是设定的 Poisson 平均到达率，不保证短 trace 的实际到达速率恰好等于它。不同模式应复用同一到达 seed、请求顺序和输出长度。

### 首 token 延迟 TTFT

- **W&B：**`latency/ttft_p50_ms`、`latency/ttft_p95_ms`、`latency/ttft_p99_ms`；CSV 使用同名后缀。
- **定义：**每请求 `1000 × (first_token_s − arrival_s)`，再对**请求**取分位数。到达时刻是计划释放时刻，首 token 时刻是主机可消费时刻。原有 summary 已有 TTFT 的均值/p50/p90/p99；事件导出新增统一的 p95 口径和 W&B 展示。
- **解读：**反映新请求获得服务并产出首 token 的等待，既包含首次排队，也包含 prefill 与首 token 计算。先看队列是否已积压，再谈调度策略对 TTFT 的影响。
- **边界：**TTFT 不是仅有 prefill 的执行时间，也不含网络传输。不要把不同实际到达 trace 的尾部数值直接归因于 refill。

### 相邻输出 token 间隔 ITL/TBT

- **W&B：**`latency/itl_p50_ms`、`latency/itl_p95_ms`、`latency/itl_p99_ms`；`latency/tbt_p50_ms` 等是**同一数值的别名**。CSV 使用 `itl_pXX_ms`。
- **定义：**对每请求的相邻两个 `token_ready_s` 计算 `1000 × (后一个 − 前一个)`，把**所有请求的间隔汇成一组**后取 p50/p95/p99。逐 token 打点由 `bfc78bc` 引入，分布由 `1ed1e9d` 导出。
- **解读：**直接观察已经开始输出的请求是否出现长时间无 token，是研究 TBT 代价的核心指标。高负载下应和队列、KV 压力及阶段时间线一起看。
- **边界：**只有一个输出 token 的请求没有间隔；长输出请求贡献更多间隔，因此这是**按 token 间隔加权**的分布，不是“每请求平均后再取分位数”。它也不等同于 TPOT。

### 每请求最大 ITL

- **W&B/CSV：**`latency/request_max_itl_p50_ms`、`latency/request_max_itl_p95_ms`、`latency/request_max_itl_p99_ms`（CSV 不带 `latency/` 前缀）。
- **定义：**先在每个至少有两个输出 token 的请求内取最大的相邻 token 间隔，再对这些**每请求最大值**取分位数。
- **解读：**回答“典型请求最糟的一次卡顿有多长”。若总体 ITL p95 不高，但此指标很高，说明长停顿分散在许多请求上，或被大量正常 token 间隔稀释。
- **边界：**单 token 请求被排除；请求输出越长，遇到极端间隔的机会也越多。比较模式时须固定输出长度分布。没有任何多 token 请求时 W&B 不上传这些分位数。

### 请求端到端延迟 E2E

- **W&B：**`latency/e2e_p50_ms`、`latency/e2e_p95_ms`、`latency/e2e_p99_ms`；CSV 为 `e2e_pXX_ms`。
- **定义：**每请求 `1000 × (finish_s − arrival_s)`，再对请求取分位数。`finish_s` 是引擎确认请求完成的时刻，可以晚于最后一个 token 的 `token_ready_s`。
- **解读：**适合评价用户等到整个生成结束所花的时间。和 TTFT、ITL 一起看，区分首 token 慢与后续生成慢。
- **边界：**强烈受输出长度影响；相同 E2E 不代表相同 token 输出速度。原有 summary 已有 E2E 分布，事件导出补了统一的 p95 与 W&B 展示。

### 按请求平均的归一化延迟

- **W&B：**`latency/normalized_mean_ms_per_output_token`；CSV 为 `normalized_mean_ms_per_token`。
- **定义：**先对每个请求计算 `1000 × (finish_s − arrival_s) / 该请求输出 token 数`，再对请求取算术平均。**不是**全体 E2E 之和除以全体 token 数；原有 summary 中对应 `request_latency.norm_e2e_s_per_token`。
- **解读：**便于在固定 workload 上看每请求的总耗时相对于生成长度是否改善。
- **边界：**仍包含排队和首 token 等待，不是相邻 token 的平均间隔。短请求与长请求在最终平均中权重相同，不能用它代替 TBT 尾部。

### 等待队列、队列峰值与首次调度等待

- **W&B：**`queue/waiting_requests` 是可查看的采样 **Table**，`queue/peak_waiting_requests` 是标量；CSV 为 `queue_peak_waiting`。首次调度等待 `request_latency.queue_s` 只在 benchmark summary/事件中，当前没有独立的 W&B 标量。
- **定义：**每个调度 tick 记录 `len(waiting_requests)`，结束时再记录一次；峰值是这些**采样值的最大值**。首次调度等待按请求算 `scheduled_s − arrival_s`，原有 summary 提供其均值/p50/p90/p99/max。
- **解读：**队列长期增长比单个非零峰值更能提示积压。首次调度等待上升通常会推高 TTFT；结合“未完成请求数”区分待调度请求和正在服务的请求。
- **边界：**等待集合也可能包含被移出显存后等待恢复的请求，不全是新到达请求。采样峰值可能漏掉两个 tick 间的短峰值。`queue_s` 记录**第一次**被调度前的等待，soft reset 保留原始调度时间，后续被抢占后的等待应看本文件的恢复指标。

### TPOT：与实际 TBT 分布不同的旧 summary 指标

- **位置：**benchmark summary 的 `request_latency.tpot_s`，当前**没有**对应的 W&B 标量或比较 CSV 列。
- **定义：**仅对至少生成两个 token 的请求计算 `(finish_time − first_token_time) / (输出 token 数 − 1)`，然后按请求汇总均值/p50/p90/p99/max。
- **解读：**它是每请求从首 token 到完成的平均速度近似值，可作为已有结果的辅助参考。
- **边界：**它使用请求完成时刻而非最后一个 token 的就绪时刻，且先按请求平均；因此与逐 token 时间戳得出的 ITL/TBT 分布不是同一个量。研究长停顿时优先看 ITL 和每请求最大 ITL。

## 机制诊断指标：解释差异从何而来

### 1. KV 占用率

- **W&B：**`kv/occupancy_pct`（折线图）；原始事件 `kv.used_blocks`。
- **定义：**每次采样的已用 block 数为 `N - free_blocks`，占用率为 `100 × used_blocks / N`。采样通常发生在每次调度 tick 接纳新工作之前，结束时再采一次。
- **解读：**和活跃请求数、等待队列一起看。请求数相近而 KV 占用更高，说明每个请求持有的缓存更多；接近上限时，再看接纳暂停及抢占是否发生。
- **边界：**这是引擎可分配 KV block 的比例，不等于整张 GPU 的显存利用率，也不代表一次模型调用的 batch size。采样点之间的短暂峰值可能看不到。

### 2. 驻留请求数

- **W&B：**`batch/resident_requests`（折线图）；原始事件 `resident.requests`。
- **定义：**采样时调度器 `active_requests` 中的请求数量，同样通常在本 tick 接纳新请求之前记录。
- **解读：**将它与配置中的 `max_num_seqs` 对照，判断请求是否长期占满允许驻留的槽位。它与 KV 占用率同步升高，可以帮助定位容量压力。
- **边界：**驻留请求可能正在 prefill、recurrent、coda 或等待 KV block；此数**不是**当前 GPU kernel 实际处理的 batch size。CB 与 CDB 的 tick 所代表的工作也不同，因此不能把两条曲线的“每 tick 平均值”直接当作同一时间平均值。

### 3. KV 余量导致的新请求接纳暂停

- **W&B：**`kv/admission_paused`（0/1 时间线）、`kv/admission_paused_s`（累计秒数）；原始事件 `kv_admission_pause.paused`。
- **定义：**满足以下全部条件时记为 1：使用的不是 `reserve` 策略；不处于抢占后的主动排空阶段；有尚未从 CPU offload 的新请求在等待；活跃请求中有 decode 工作；可用驻留槽位至少达到 `min_free_slots`；且 `free_blocks <= safety_margin × N`。累计秒数把相邻采样时间差乘以前一个采样值后求和，相当于假设两次采样之间状态保持不变。
- **解读：**1 表示此刻 KV 安全余量门槛会阻止新 prompt 进入，同时驻留槽位门槛已满足。若它持续出现且 TTFT、等待队列上升，值得研究 KV 接纳策略。
- **边界：**这是按采样点推断的门槛状态，不是每个请求被拒绝的精确持续时间；coda 优先级、token budget 等其他因素仍可能同时影响调度。`reserve` 模式恒为 0，不能据此推断它没有内存压力。折线图在 0 与 1 之间的斜线只是绘图连接，不表示中间状态。

### 4. 抢占到下一 token 的等待

- **W&B：**`preemption/offload_to_next_token_ms`、`preemption/recompute_to_next_token_ms`（散点图）及相应的 `*_p95_ms`；原始事件 `preemption` 与同一请求的 `request.token_ready_s`。
- **定义：**抢占开始时记录请求 ID 和**实际执行**的策略；对每次抢占，找到该请求时间不早于抢占的第一个输出 token。纵轴为 `1000 × (下一 token 时间 - 抢占开始时间)`。如果该请求之后没有 token，这次事件不进入图和 p95。
- **解读：**这是请求可感知的抢占后输出空档，可直接与该请求的 TBT 异常联系起来。区分 offload 与重新计算，能看出哪条恢复路径更容易带来长空档。
- **边界：**它包含排队、重新接纳、复制或重新预填及正常生成时间，**不是**纯粹的复制耗时。一个请求若在下一 token 前反复被抢占，多个空档可能重叠，不能相加。p95 必须结合事件数看；少数事件的分位数不稳定。

### 5. KV 移出与恢复的搬运量

- **W&B：**`kv_transfer/gpu_to_cpu_gib`、`kv_transfer/cpu_to_gpu_gib`（全程总量）和对应的 `*_cumulative_gib`（累计折线图）；原始事件 `kv_transfer.bytes`。
- **定义：**一个 KV block 的字节数为 `2 × 层数 × block_size × KV heads × head_dim × dtype.itemsize`，其中 2 对应 key 与 value。每次移出或恢复的字节数等于实际复制的 block 数乘以该值，再除以 `1024³` 得 GiB；图上按时间累加。同一请求被多次搬运会重复计入。
- **解读：**搬运量快速增长说明系统在 KV 压力下频繁交换。两方向总量不完全相同可能意味着运行结束时仍有未恢复的 offload，需结合原始事件检查。
- **边界：**这是提交到复制路径的 KV 数据量，不是 PCIe/NVLink 吞吐率、复制耗时或总 GPU 内存流量。没有实际 CPU offload 时，这组图/标量缺席；它不表示重新计算的成本。

### 6. 请求到达窗口内的输出 token 速率

- **W&B：**`throughput/arrival_window_output_tokens_per_s`（全窗口标量）、`throughput/arrival_window_token_rate`（逐秒折线图）；从 `request.arrival_s` 和 `request.token_ready_s` 计算。
- **定义：**仅对使用 `--request-rate` 的开环实验计算。令 `T = max(arrival_s)`。标量为 `在 [0,T] 内送达的输出 token 数 / T`；图上每个整数秒箱的 token 数除以该箱的实际长度，最后不足 1 秒的箱用其实际长度。恰好在 `T` 送达的 token 计入最后一个箱。
- **解读：**它回答“请求还在进入系统时，实际送出了多少 token”。与现有全程 `throughput/output_tokens_per_s` 对照，可以发现一个模式是否主要靠到达停止后的**排空阶段**才得到看似不错的吞吐量。
- **边界：**分母从测量开始算起，包含第一个请求到达前的空档；全程吞吐量则包含排空尾段。不同到达率实验应尽量使用相近的到达窗口长度和相同随机种子；短窗口或最后一个很短的秒箱会让逐秒曲线波动较大。离线一次性提交的运行不生成此指标。

### 7. 最后到达时的未完成请求数

- **W&B：**`backlog/requests_at_last_arrival`（标量）、`backlog/unfinished_during_arrivals`（时间线）；从每个 `request` 的到达和完成时间计算。
- **定义：**时间 `t` 的未完成请求数 = 截至 `t` 已到达的请求数 − 截至 `t` 已完成的请求数。标量取 `T` 时的值，等价于 `finish_s > T` 的请求数。图从零开始，按到达加 1、按完成减 1，仅画到 `T`。
- **解读：**曲线持续上升且窗口结束时仍有大量未完成请求，提示该运行点可能在积压；结合更长窗口和重复种子再判断是否达到可持续容量上限。
- **边界：**包含正在服务的请求与等待中的请求，**不等于 waiting queue 长度**；最后到达时仍有一些正在处理的请求是正常现象。单个运行的非零末值不能单独证明过载。离线运行不生成此指标。

### 8. 重新计算需要重新预填的 token 数

- **W&B：**`preemption/tokens_to_reprefill`（总数）、`preemption/cumulative_tokens_to_reprefill`（累计折线图）；原始事件 `recompute.tokens_to_reprefill`。
- **定义：**每次 soft reset 后，计算新请求状态的 `remaining_prefill_tokens` 长度并累加。这个长度包括原 prompt 和在抢占前已生成、必须并入新 prompt 的 token。一个请求多次重新计算会重复计入。
- **解读：**与 KV 搬运量、抢占到下一 token 的等待共同看，判断在 KV 压力下选择重新计算需要付出多少额外预填工作。
- **边界：**这是**安排重新预填的 token 数**，不是实际完成的 GPU FLOPs、耗时或输出 token 数；取消/中断以及其他执行细节可能使实际工作不同。没有 soft reset 时图/总数缺席。

### 9. CDB 阶段启动与批次大小

- **W&B：**`stage/prefill_batch_size`、`stage/prelude_batch_size`、`stage/recurrent_batch_size`、`stage/coda_batch_size`（分别为散点图）；原始事件 `stage_launch`。
- **定义：**CDB 在每次相应阶段 launch 后记录主机时间和该阶段的工作项数量。纵轴是那次阶段 launch 的 batch size，横轴是本次运行已经过的秒数。
- **解读：**查看某个请求出现长 token 间隔的附近，是否密集发生 prefill、coda 或很小的 recurrent launch；同时比较 refill 与 no-refill 的阶段序列和 batch 大小。
- **边界：**时间戳在 launch 调用之后记录，不代表 GPU 完成时刻；点间距也不能直接当作某阶段的执行耗时。prefill 的 batch size 是请求数，不是预填 token 总数。图不含请求 ID，阶段活动与某请求延迟的同现只提供线索，不能单凭它认定因果。CB 不生成这些图。

### 10. GPU 上的 KV 复制耗时

- **W&B：**`kv_transfer/gpu_to_cpu_duration_ms`、`kv_transfer/cpu_to_gpu_duration_ms`（散点图）和相应的 `*_p95_ms`；原始事件 `kv_transfer_duration`。
- **定义：**仅在 CUDA 设备上、实际发生 CPU offload/restore 时，用放在计算流上的两个 CUDA event 包住复制操作。测量结束并同步 GPU 后读取两事件间的毫秒数。一次恢复调用可能合并多个请求，其耗时按**该次合并复制**记录，而非分摊到各请求。
- **解读：**判断复制本身是否显著；与搬运量一起观察不同数据量对应的耗时，再与请求的抢占后空档比较。
- **边界：**不包括等待重新调度、请求排队、重新计算或 CPU 侧准备时间；也不能简单从用户可感知空档中减去它，因复制可异步排队。CPU 设备运行以及未发生 offload 的运行不会有此图。开启 CUDA event 会增加少量测量开销。

### 11. 移出后到开始恢复的等待

- **W&B：**`preemption/offload_to_restore_ms`（散点图）、`preemption/offload_to_restore_p95_ms`；原始事件 `preemption` 与同一请求的 `restore`。
- **定义：**对一次实际 offload，匹配同一请求随后开始恢复的时间；纵轴为 `1000 × (开始恢复时间 - 抢占开始时间)`。恢复事件在 CPU→GPU 复制前记录。若没有匹配的恢复事件，该次 offload 不进入图和 p95。
- **解读：**它主要显示被移出请求等待重新获得服务机会的时间。若它很长、GPU 复制却很短，优先检查排空与重新调度，而不是优化复制带宽。
- **边界：**起点在 GPU→CPU 复制发起之前，因此还包含其主机侧提交过程；终点在恢复复制之前。它不是纯粹的队列等待，也不能与第 10 项的 CUDA 耗时机械相减来精确分解下一 token 的延迟。

## 分位数、缺失值与比较顺序

新增事件耗时的 p95 与现有延迟指标使用相同的线性插值：对升序的 `n` 个样本，取位置 `0.95 × (n − 1)`，在相邻样本间插值。只有一个样本时 p95 就是该样本；没有相关事件时通常**不上传该图和标量**，这与测得 0 不同。KV 接纳暂停的 0 则是实际采样得到的 0。W&B 图为阅读方便会稀疏取点，正式计算应使用标量或完整 JSONL。

比较前，固定模型、workload bundle 与输出长度、到达 seed、早退轨迹、`max_num_seqs`、`min_free_slots`、KV block 数、KV 压力策略和其他执行配置；核对 W&B run config 与事件文件的 `run.summary.config`。CB、CDB/no-refill、CDB/refill 的标签记录在导出 CSV 的 `mode` 列。对长尾指标至少重复多个 seed，并报告样本数；单个短 trace 的 p99 容易被个别请求支配。按以下顺序读结果：

1. 先看机制诊断第 6、7 项，确认比较点是否在到达期间持续积压。持续过载时，不要把尾延迟变差直接解释为 refill 固有的 TBT 代价。
2. 在两种 CDB 都能稳定服务的相同**绝对到达率**下，比较核心 TTFT、ITL/TBT 与每请求最大 ITL；再用机制诊断第 2、9 项检查是否存在阶段调度或批次变化。
3. 若差异只在较小 KV 容量出现，看机制诊断第 1、3 项，再看第 4、5、8、10、11 项区分接纳、抢占、搬运和重新计算。CPU offload 与 recompute 只有在确实发生时才有对应图。

| 观察到的组合 | 可以谨慎得出的结论 | 还需检查什么 |
| --- | --- | --- |
| 相同到达率下两种 CDB 都没有持续积压，refill 的 ITL/每请求最大 ITL 却更差 | 存在值得研究的生成过程中卡顿信号。 | 看阶段 launch、KV 占用和抢占；不能只由 ITL 曲线断言是哪一阶段造成。 |
| no-refill 的未完成请求持续增长，refill 没有，且 refill 的到达窗口 token 速率更高 | refill 在这一到达率附近显示了容量优势。 | 过载的 no-refill 尾延迟不能作为“refill 固有 TBT 代价”的同条件对照。 |
| 两者的未完成请求都持续增长 | 本次到达率可能超出两者的可持续处理能力。 | 降低到达率、延长到达窗口并重复种子；不要只用含排空期的全程吞吐量判断稳定性。 |
| 只有缩小 KV 后 TTFT/ITL 恶化，且 KV 余量暂停或抢占增多 | 问题更可能和 KV 接纳或恢复路径有关。 | 分别比较恢复等待、复制耗时、搬运量与重新预填 token 数；不能把这几段时间直接相减。 |
| KV 占用、暂停和抢占都没有明显变化，但 ITL 变差 | KV 压力暂时缺少支持性证据。 | 回看阶段启动、批次大小、实现配置与请求级 token 时间线。 |

最后整理结果时，至少同时给出：模式、设定到达率与 seed、请求/输出 token 数、到达窗口长度、窗口末未完成请求、全程及到达窗口吞吐量、TTFT/ITL/每请求最大 ITL 的尾部分位数，以及实际 KV 和 batch 配置。若教授没有预先给出 TTFT/TBT 的服务目标，不应事后挑一个阈值宣称违反 SLO。

本文件描述的是当前代码中**实际记录与计算的量**。实现入口为 [`scheduler_base.py`](src/looped_cdb/scheduler_base.py)、[`offloading_manager.py`](src/looped_cdb/offloading_manager.py)、[`latency_events.py`](src/looped_cdb/benchmarks/latency_events.py) 和 [`wandb_serving.py`](src/looped_cdb/benchmarks/wandb_serving.py)。
