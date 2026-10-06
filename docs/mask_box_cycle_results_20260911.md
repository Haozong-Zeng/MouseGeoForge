# 双鼠框定位：六小时周期最终结果

授权周期为2026-09-11 11:39:23–17:39:23 UTC，六小时窗口现已结束。
截止后已通过应用工具将既有mousegeoforge自动化设为PAUSED，
17:40:05.802 UTC完成最终只读核验：19个候选产物、基础权重、encoder及相关源码
文件均与既有审计哈希一致；GPU空闲、无训练锁。本轮收尾，不再自动启动实验。
训练与候选接口验证已完成；六小时包含半小时状态检查，不是GPU连续运行六小时。

## 已得到的结论

单纯把预测mask外接框替换回归框，能提高双鼠平均IoU，但极少量远端前景会撑大框，
并造成旧单鼠侧面退化。只拼接mask形状到普通MLP，800→1600步继续降低训练损失，
却没有带来实质开发定位收益，因此停止追加该方案。

显式将“原框到mask全域/主域参考框”的位移作为可学习残差后，两个种子均有收益：

| 参考修正开关 | seed17双鼠IoU | seed29双鼠IoU | 双鼠检出 /64 | 旧侧面检出 /32 |
|---|---|---|---|---|
| 同预算OFF | .6495 | .6411 | 53 / 51 | 28 / 28 |
| ON | .7794 | .7687 | 62 / 62 | 32 / 32 |

每个种子两臂同为800步、相同原训练输入和预算，OFF/ON均读取mask特征，唯一
区别是显式参考修正路径。两个ON的正常输入和固定四角噪点测试均通过原17门槛。
噪点条件下两种子双鼠检出均61/64，并非完全免疫噪点。

关键点、objectness、深度及基础模型权重没有更新。双鼠检测感知PCK从OFF的
.5183/.4959提升至ON的.5673/.5652，不能称为关键点坐标或端到端3D精度提升。
当前阶段正式训练总计6400个新增框分支优化步骤，基础模型0优化步骤；不是重新
训练了6400步VGGT。两个种子复用同一合成开发集，不是独立真实数据测试。

## 可用候选与验证证据

- 推理包：`output/mask_box_candidates_v1/seed17/`、`seed29/`，每个新增分支
  权重约148KB，manifest明确绑定对应基础温度checkpoint和encoder；不包含优化器。
- 推荐接口：`mousegeoforge.mask_box_candidate_safe.MaskBoxCandidate`，默认关闭。
  用法与限制见`docs/mask_box_candidate_usage.md`。
- 176次完整GPU前向验证：原预测和导出后框逐值相同，只有boxes字典项可变，
  其他输出对象不变；默认关闭无hook，退出后hook/原参数flags/modes保持不变。
- CPU安全加载守卫在成功/失败时都保留Torch随机数状态；继承同一已验证计算函数，
  没有将CPU守卫证明冒称为额外GPU重放。
- 全部211项测试通过，包括有限异常输入、错误配对、文件/源码哈希损坏、退出清理
  与再次使用。损坏测试用一次性夹具，真实候选未被修改；不是穷举模糊测试或安全认证。

关键证据索引：

| 内容 | 位置 |
|---|---|
| 全部mask框冻结对照 | `output/mask_box_decode_v1/acceptance.json` |
| 远端噪点、遮挡、错鼠诊断 | `output/mask_box_failures_v1_actual_train/analysis.json` |
| 普通MLP停止续训的依据 | `output/mask_box_refiner_seed17_v1_step1600/acceptance.json` |
| 显式参考框seed17 | `output/mask_reference_box_seed17_v1/acceptance.json` |
| 显式参考框seed29复验 | `output/mask_reference_box_seed29_v1/acceptance.json` |
| 导出与接口重放 | `output/mask_box_candidates_v1/verification.json` |
| RNG安全加载证明 | `output/mask_box_candidates_v1/loading_guard_verification.json` |
| 有限负向测试与耗时分组 | `output/mask_candidate_contracts_v1/report.json` |
| 起止时间、失败与决策台账 | `docs/mask_box_six_hour_cycle_20260911.md` |

## 性能与剩余问题

已有176条观测按V/H/W分组；两种子合并仅用于描述耗时，不构成独立精度样本量。
下表只计CPU后处理及传输，不包含基础前向、加载或上下文参数哈希验证：

| 视角数 | 每视角分辨率 | 观测数 | 中位ms | P95 ms |
|---|---|---|---|---|
| 1 | 768×768 | 48 | 42.18 | 49.51 |
| 2 | 768×432 | 32 | 46.35 | 52.39 |
| 2 | 768×768 | 48 | 85.39 | 94.43 |
| 3 | 768×768 | 32 | 129.82 | 145.41 |
| 4 | 768×768 | 16 | 173.80 | 194.38 |

当前CPU连通域原型不能作为25fps实时推理实现；上下文哈希验证应在成段推理入口
进行，不应每帧重新进入上下文。若以后优化速度，必须另行证明输出/匹配不变，
不能默默跳过低分query或改变mask阈值。

帧296顶部两个query仍重复关注黑鼠，框后处理无法创建缺失棕鼠实例；跨视角/时间
身份与实例分离需要独立处理。真实Cylinder底部视图不在本地，尚不能验证真实泛化。
当前候选是可显式调用的研究原型，不是生产升级：没有自动挂入推理、替换checkpoint、
修改数据、下载、重渲染或git push。
