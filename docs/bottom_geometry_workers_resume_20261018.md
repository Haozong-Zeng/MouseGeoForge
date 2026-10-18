# 当前几何适配臂：四进程加载与完整状态续训

2026-10-18，用户明确要求将4个DataLoader workers接入**当前适配臂**，先保存checkpoint，保留优化器、EMA、RNG及采样进度，实测端到端加速；结束后不启动冻结对照。此记录是现有纯几何v2的运行时修订，不替换原实验协议，不启动新实验臂。

## 保存和恢复

原训练在第1,500次优化器更新结束后已原子保存raw模型、EMA、AdamW完整状态、loss weights、曝光计数、Python/NumPy/CPU/CUDA RNG。已把该checkpoint复制为 `output/pure_geometry_bottom_20261018_v2/resume_workers4/pre_stop_checkpoint.pt` 并登记SHA256；随后SIGTERM停止原进程。

停止时处于EMA验证。原训练器的临时EMA参数交换没有异常恢复保护，其 `interrupted.pt` 不可作为raw恢复来源；本次使用**EMA交换之前保存的checkpoint**。续训入口为EMA验证添加try/finally恢复raw参数，补做缺失的step1500验证。旧进程的`interrupted/failed`状态是此次主动停止的记录，不是新训练失败。

恢复逐张量校验可训练模型参数、EMA、优化器状态，并恢复和比对全部RNG。对照原固定schedule累计前1500更新的来源曝光计数，下一条训练数据明确从**第1501更新、第12001微批**开始。worker预取而未消费的数据不计入进度，随时可丢弃。原发布权重及14,912个source pins重新校验。

LR余弦进度、EMA更新计数、裁剪累计次数和8微批累积均延续，不重做梯度校准、不重新warmup。续训继续写原 `geometry_seed17/train.jsonl`；状态文件注明resume session和workers。checkpoint增加 `.resume.json` 伴随文件，记录下一更新/微批位置、schedule哈希、运行时间和运行时修订哈希。

## 加载与测量

CPU worker配置：4进程、spawn、每worker torch/OpenCV线程1、persistent workers、prefetch factor2、固定有序schedule、独立随机generator；主进程pinned-memory non-blocking CUDA传输。原串行加载器及预处理规则不改。

使用同一step1500快照及同一段16更新序列，按0/4/4/0 workers顺序进行四次可丢弃计时。每次前4更新作为启动及缓存warmup，后12更新计入：包括8微批前向、全部原目标、反向、梯度裁剪、AdamW与EMA；不包括checkpoint、验证、模型启动及profiler。所有计时更新丢弃，正式训练重新加载step1500完整快照。

最初设置了最终状态逐位相同的门槛；该门槛没有通过，原始`benchmark_summary.json`及失败记录保留。RNG和曝光计数完全一致，最终模型/优化器/EMA并不逐位一致，**相同串行配置重复跑同样不一致**。额外做32个实际续训微批的输入bitwise对照，以及同快照的一步串行/串行/四worker独立诊断：初始所有状态和首步损失完全相同，单步模型最大差为串行重复`6.47e-8`、四worker`6.71e-8`，relativeL2分别`3.63e-9`与`3.57e-9`。该CUDA反向的非确定性已有独立重复证据，不能要求恢复后的未来轨迹逐位复现。

运行时修订v2因此使用`continuation_gate.json`：初始状态、RNG、实际输入和采样严格一致；同一步数值偏差通过显式阈值（max_abs≤2e-6、relativeL2≤1e-6），并与原串行重复的数值波动对照。`numerical_resume_diagnostic.json`保留原始偏差、installed PyTorch deterministic backward检查及全部恢复校验；`runtime_amendment_v1_ab_timing.json`保留先前修订，不覆盖原bitwise失败证据。

结果在 `resume_workers4/benchmark_summary.json`，每次原始更新耗时和200ms GPU采样也保留。另对一次可丢弃微批做CPU/CUDA profiler，记录同步次数、kernel时间和内存。纯加载吞吐、训练更新吞吐、包含验证和存盘的总墙钟耗时分开解释。

四次A/B实测平均：原串行`2.2737秒/更新`，四worker`2.1146秒/更新`，训练更新吞吐为原来的`1.0753倍`（约+7.5%）；数据等待从`0.3597`降至`0.00552秒/更新`（约-98.5%）。每配置计入24更新。该结果不包含周期验证和存盘，不能称整次实验墙钟提速7.5%。两次四worker计时分别2.1134/2.1157秒，串行2.3094/2.2379秒，保留运行波动。

## GPU与显存

当前模型包含约11.44亿个几何state elements，其中可训练参数约2.86亿。每个物理微批为一个512尺度多视角样本；累积8次是串行，显存中不会同时保存8套激活。编码器大部分冻结，也减少激活与优化器内存。因此96GB显存有大量余量，需要结合数据等待、CPU/GPU同步和kernel并行度判断算力瓶颈。

本次保持物理微批、损失、优化器和数据顺序一致，确保加载A/B可解释。扩大物理微批须先将几何目标中显式`batch[0]`的计算改为逐样本等价归约，再独立验证；不能直接把DataLoader batch_size设成2，否则几何监督会遗漏样本。

实测峰值allocated约11.19GiB，nvidia-smi进程上下文及reserved计入后约14–15GiB。一次`[1,3,3,512,512]`微批trace包含83次cudaStreamSynchronize、39次标量读取和6,857次cudaLaunchKernel，显示减少数据等待后仍有同步和大量细粒度启动开销。profiler自身的device同步用于计时，不解释成训练逻辑新增同步。`gpu_profile.json`的CPU算子device time与kernel条目存在归因重叠，其合计不用于推算GPU利用率。

## 运行边界

有限启动器 `tools/continue_geometry_with_ab_timing_20261018.py` 仅执行上述可丢弃A/B和同一适配臂续训。续训由 `tools/resume_geometry_workers_20261018.py` 执行。完成或失败即退出，代码中无冻结对照或关键点启动路径。主状态为 `geometry_seed17/status.json`，本次协调状态为 `resume_workers4/supervisor_status.json`。
