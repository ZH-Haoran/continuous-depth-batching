# 先导实验 Serving 指标说明

本文对应 `scripts/benchmark_throughput.py` 的可选事件记录与 W&B 上传路径，解释本次新增的 11 类指标。它们用于回答两个问题：请求到达率升高时，refill 是否真正提高了可持续服务能力；当延迟恶化时，问题出在排队、KV 空间、抢占恢复，还是 CDB 的阶段调度。现有的 TTFT（首 token 延迟）、ITL/TBT（相邻输出 token 间隔）和端到端延迟仍是最终用户体验指标；下面的时间线主要帮助解释它们。

## 如何获得这些指标

在已有的 benchmark 命令后加上 `--latency-events-dir <目录> --wandb-project <项目名>`。前者开启请求/token 时间戳和诊断事件记录，并写出完整的 `*.events.jsonl`；后者在计时结束后把汇总值、图表及原始事件文件上传到 W&B，且必须与前者一起使用。只需本地事件文件时可以不传 `--wandb-project`。CDB 阶段图不要求另外开启 `--trace-output`。

所有事件的 `time_s` 都以本次测量的开始时刻为零点。开环请求的 `arrival_s` 是**计划到达时刻**，不是调度器下一次发现它的时刻，因此发现和接纳的延迟不会被抹掉。W&B 图表最多展示约 10,000 个点；标量用完整事件计算，原始 JSONL 也保存完整事件。事件打点和 CUDA 计时会给测量带来一定开销，比较不同模式时应让它们使用相同的记录设置。

下文的 `gpu_to_cpu` 表示将 KV 从 GPU 移到 CPU；`cpu_to_gpu` 表示恢复。`N` 表示可分配的 KV block 总数，`T` 表示最后一个请求的计划到达时间（相对测量起点）。

## 1. KV 占用率

- **W&B：**`kv/occupancy_pct`（折线图）；原始事件 `kv.used_blocks`。
- **定义：**每次采样的已用 block 数为 `N - free_blocks`，占用率为 `100 × used_blocks / N`。采样通常发生在每次调度 tick 接纳新工作之前，结束时再采一次。
- **解读：**和活跃请求数、等待队列一起看。请求数相近而 KV 占用更高，说明每个请求持有的缓存更多；接近上限时，再看接纳暂停及抢占是否发生。
- **边界：**这是引擎可分配 KV block 的比例，不等于整张 GPU 的显存利用率，也不代表一次模型调用的 batch size。采样点之间的短暂峰值可能看不到。

## 2. 驻留请求数

- **W&B：**`batch/resident_requests`（折线图）；原始事件 `resident.requests`。
- **定义：**采样时调度器 `active_requests` 中的请求数量，同样通常在本 tick 接纳新请求之前记录。
- **解读：**将它与配置中的 `max_num_seqs` 对照，判断请求是否长期占满允许驻留的槽位。它与 KV 占用率同步升高，可以帮助定位容量压力。
- **边界：**驻留请求可能正在 prefill、recurrent、coda 或等待 KV block；此数**不是**当前 GPU kernel 实际处理的 batch size。CB 与 CDB 的 tick 所代表的工作也不同，因此不能把两条曲线的“每 tick 平均值”直接当作同一时间平均值。

## 3. KV 余量导致的新请求接纳暂停

- **W&B：**`kv/admission_paused`（0/1 时间线）、`kv/admission_paused_s`（累计秒数）；原始事件 `kv_admission_pause.paused`。
- **定义：**满足以下全部条件时记为 1：使用的不是 `reserve` 策略；不处于抢占后的主动排空阶段；有尚未从 CPU offload 的新请求在等待；活跃请求中有 decode 工作；可用驻留槽位至少达到 `min_free_slots`；且 `free_blocks <= safety_margin × N`。累计秒数把相邻采样时间差乘以前一个采样值后求和，相当于假设两次采样之间状态保持不变。
- **解读：**1 表示此刻 KV 安全余量门槛会阻止新 prompt 进入，同时驻留槽位门槛已满足。若它持续出现且 TTFT、等待队列上升，值得研究 KV 接纳策略。
- **边界：**这是按采样点推断的门槛状态，不是每个请求被拒绝的精确持续时间；coda 优先级、token budget 等其他因素仍可能同时影响调度。`reserve` 模式恒为 0，不能据此推断它没有内存压力。折线图在 0 与 1 之间的斜线只是绘图连接，不表示中间状态。

## 4. 抢占到下一 token 的等待

- **W&B：**`preemption/offload_to_next_token_ms`、`preemption/recompute_to_next_token_ms`（散点图）及相应的 `*_p95_ms`；原始事件 `preemption` 与同一请求的 `request.token_ready_s`。
- **定义：**抢占开始时记录请求 ID 和**实际执行**的策略；对每次抢占，找到该请求时间不早于抢占的第一个输出 token。纵轴为 `1000 × (下一 token 时间 - 抢占开始时间)`。如果该请求之后没有 token，这次事件不进入图和 p95。
- **解读：**这是请求可感知的抢占后输出空档，可直接与该请求的 TBT 异常联系起来。区分 offload 与重新计算，能看出哪条恢复路径更容易带来长空档。
- **边界：**它包含排队、重新接纳、复制或重新预填及正常生成时间，**不是**纯粹的复制耗时。一个请求若在下一 token 前反复被抢占，多个空档可能重叠，不能相加。p95 必须结合事件数看；少数事件的分位数不稳定。

## 5. KV 移出与恢复的搬运量

- **W&B：**`kv_transfer/gpu_to_cpu_gib`、`kv_transfer/cpu_to_gpu_gib`（全程总量）和对应的 `*_cumulative_gib`（累计折线图）；原始事件 `kv_transfer.bytes`。
- **定义：**一个 KV block 的字节数为 `2 × 层数 × block_size × KV heads × head_dim × dtype.itemsize`，其中 2 对应 key 与 value。每次移出或恢复的字节数等于实际复制的 block 数乘以该值，再除以 `1024³` 得 GiB；图上按时间累加。同一请求被多次搬运会重复计入。
- **解读：**搬运量快速增长说明系统在 KV 压力下频繁交换。两方向总量不完全相同可能意味着运行结束时仍有未恢复的 offload，需结合原始事件检查。
- **边界：**这是提交到复制路径的 KV 数据量，不是 PCIe/NVLink 吞吐率、复制耗时或总 GPU 内存流量。没有实际 CPU offload 时，这组图/标量缺席；它不表示重新计算的成本。

## 6. 请求到达窗口内的输出 token 速率

- **W&B：**`throughput/arrival_window_output_tokens_per_s`（全窗口标量）、`throughput/arrival_window_token_rate`（逐秒折线图）；从 `request.arrival_s` 和 `request.token_ready_s` 计算。
- **定义：**仅对使用 `--request-rate` 的开环实验计算。令 `T = max(arrival_s)`。标量为 `在 [0,T] 内送达的输出 token 数 / T`；图上每个整数秒箱的 token 数除以该箱的实际长度，最后不足 1 秒的箱用其实际长度。恰好在 `T` 送达的 token 计入最后一个箱。
- **解读：**它回答“请求还在进入系统时，实际送出了多少 token”。与现有全程 `throughput/output_tokens_per_s` 对照，可以发现一个模式是否主要靠到达停止后的**排空阶段**才得到看似不错的吞吐量。
- **边界：**分母从测量开始算起，包含第一个请求到达前的空档；全程吞吐量则包含排空尾段。不同到达率实验应尽量使用相近的到达窗口长度和相同随机种子；短窗口或最后一个很短的秒箱会让逐秒曲线波动较大。离线一次性提交的运行不生成此指标。

## 7. 最后到达时的未完成请求数

- **W&B：**`backlog/requests_at_last_arrival`（标量）、`backlog/unfinished_during_arrivals`（时间线）；从每个 `request` 的到达和完成时间计算。
- **定义：**时间 `t` 的未完成请求数 = 截至 `t` 已到达的请求数 − 截至 `t` 已完成的请求数。标量取 `T` 时的值，等价于 `finish_s > T` 的请求数。图从零开始，按到达加 1、按完成减 1，仅画到 `T`。
- **解读：**曲线持续上升且窗口结束时仍有大量未完成请求，提示该运行点可能在积压；结合更长窗口和重复种子再判断是否达到可持续容量上限。
- **边界：**包含正在服务的请求与等待中的请求，**不等于 waiting queue 长度**；最后到达时仍有一些正在处理的请求是正常现象。单个运行的非零末值不能单独证明过载。离线运行不生成此指标。

## 8. 重新计算需要重新预填的 token 数

- **W&B：**`preemption/tokens_to_reprefill`（总数）、`preemption/cumulative_tokens_to_reprefill`（累计折线图）；原始事件 `recompute.tokens_to_reprefill`。
- **定义：**每次 soft reset 后，计算新请求状态的 `remaining_prefill_tokens` 长度并累加。这个长度包括原 prompt 和在抢占前已生成、必须并入新 prompt 的 token。一个请求多次重新计算会重复计入。
- **解读：**与 KV 搬运量、抢占到下一 token 的等待共同看，判断在 KV 压力下选择重新计算需要付出多少额外预填工作。
- **边界：**这是**安排重新预填的 token 数**，不是实际完成的 GPU FLOPs、耗时或输出 token 数；取消/中断以及其他执行细节可能使实际工作不同。没有 soft reset 时图/总数缺席。

## 9. CDB 阶段启动与批次大小

- **W&B：**`stage/prefill_batch_size`、`stage/prelude_batch_size`、`stage/recurrent_batch_size`、`stage/coda_batch_size`（分别为散点图）；原始事件 `stage_launch`。
- **定义：**CDB 在每次相应阶段 launch 后记录主机时间和该阶段的工作项数量。纵轴是那次阶段 launch 的 batch size，横轴是本次运行已经过的秒数。
- **解读：**查看某个请求出现长 token 间隔的附近，是否密集发生 prefill、coda 或很小的 recurrent launch；同时比较 refill 与 no-refill 的阶段序列和 batch 大小。
- **边界：**时间戳在 launch 调用之后记录，不代表 GPU 完成时刻；点间距也不能直接当作某阶段的执行耗时。prefill 的 batch size 是请求数，不是预填 token 总数。图不含请求 ID，阶段活动与某请求延迟的同现只提供线索，不能单凭它认定因果。CB 不生成这些图。

## 10. GPU 上的 KV 复制耗时

- **W&B：**`kv_transfer/gpu_to_cpu_duration_ms`、`kv_transfer/cpu_to_gpu_duration_ms`（散点图）和相应的 `*_p95_ms`；原始事件 `kv_transfer_duration`。
- **定义：**仅在 CUDA 设备上、实际发生 CPU offload/restore 时，用放在计算流上的两个 CUDA event 包住复制操作。测量结束并同步 GPU 后读取两事件间的毫秒数。一次恢复调用可能合并多个请求，其耗时按**该次合并复制**记录，而非分摊到各请求。
- **解读：**判断复制本身是否显著；与搬运量一起观察不同数据量对应的耗时，再与请求的抢占后空档比较。
- **边界：**不包括等待重新调度、请求排队、重新计算或 CPU 侧准备时间；也不能简单从用户可感知空档中减去它，因复制可异步排队。CPU 设备运行以及未发生 offload 的运行不会有此图。开启 CUDA event 会增加少量测量开销。

## 11. 移出后到开始恢复的等待

- **W&B：**`preemption/offload_to_restore_ms`（散点图）、`preemption/offload_to_restore_p95_ms`；原始事件 `preemption` 与同一请求的 `restore`。
- **定义：**对一次实际 offload，匹配同一请求随后开始恢复的时间；纵轴为 `1000 × (开始恢复时间 - 抢占开始时间)`。恢复事件在 CPU→GPU 复制前记录。若没有匹配的恢复事件，该次 offload 不进入图和 p95。
- **解读：**它主要显示被移出请求等待重新获得服务机会的时间。若它很长、GPU 复制却很短，优先检查排空与重新调度，而不是优化复制带宽。
- **边界：**起点在 GPU→CPU 复制发起之前，因此还包含其主机侧提交过程；终点在恢复复制之前。它不是纯粹的队列等待，也不能与第 10 项的 CUDA 耗时机械相减来精确分解下一 token 的延迟。

## 分位数、缺失值与比较顺序

新增事件耗时的 p95 与现有延迟指标使用相同的线性插值：对升序的 `n` 个样本，取位置 `0.95 × (n − 1)`，在相邻样本间插值。只有一个样本时 p95 就是该样本；没有相关事件时通常**不上传该图和标量**，这与测得 0 不同。KV 接纳暂停的 0 则是实际采样得到的 0。W&B 图为阅读方便会稀疏取点，正式计算应使用标量或完整 JSONL。

建议先固定模型、workload、到达 seed、`max_num_seqs`、`min_free_slots`、KV block 数和压力策略，再按以下顺序读结果：

1. 先看第 6、7 项，确认比较点是否在到达期间持续积压。持续过载时，不要把尾延迟变差直接解释为 refill 固有的 TBT 代价。
2. 在两种 CDB 都能稳定服务的相同**绝对到达率**下，比较已有 TTFT、ITL/TBT 与每请求最大 ITL；再用第 2、9 项检查是否存在阶段调度或批次变化。
3. 若差异只在较小 KV 容量出现，看第 1、3 项，再看第 4、5、8、10、11 项区分接纳、抢占、搬运和重新计算。CPU offload 与 recompute 只有在确实发生时才有对应图。

本文件描述的是当前代码中**实际记录与计算的量**。实现入口为 [`scheduler_base.py`](src/looped_cdb/scheduler_base.py)、[`offloading_manager.py`](src/looped_cdb/offloading_manager.py)、[`latency_events.py`](src/looped_cdb/benchmarks/latency_events.py) 和 [`wandb_serving.py`](src/looped_cdb/benchmarks/wandb_serving.py)。
