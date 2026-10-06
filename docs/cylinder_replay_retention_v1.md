# 双鼠回放保持预检（2026-09-11）

## 结论

**不启动框＋Core23 教师保持的 800 步训练。** 原始模型不是在既有双鼠回放训练
姿态上全面优于混合模型的教师：身体定位较好，但框和尾巴较差。预设的三个质量
门槛有两个不通过，不能通过临时删去尾巴或框来宣称原方案通过。

本轮只做冻结权重的训练侧检查，无 optimizer 更新，未增加训练量、改生产权重或
修改数据。不是新的失败训练方法；连续无实质改善训练方法计数仍为 **1**。
结果是复用合成训练姿态上的机制判断，不是真实 Cylinder 或独立测试效果。

## 固定方案与结果

沿用混合方案已有 8 个双鼠训练姿态，各两个精确顺序视角；teacher 为原始生产
checkpoint，student 为已完成的 source_mix checkpoint。分别用原 Hungarian 匹配
连接同一 GT instance_id，不按 query 序号连接。仅保留 teacher score≥.5 且 GT
box IoU≥.5 的实例，共 **31/32 instance-view**。关键点仅使用原有有效标签。

拟议保持项为 normalized box 与 Core23 xy smooth-L1(beta=.01) 之和、权重1，
先按同一实例的可用视角平均，再按实例平均。仅计划作用于双鼠回放，原 GT
监督继续保留；不叠加极线、深度联合、温度或新采样变化。

| 同一合格训练子集 | 原始 teacher | 混合 student | teacher 质量门槛 |
|---|---:|---:|---|
| 框 IoU ↑ | 0.84840 | 0.89997 | 不通过 |
| 身体16 NME ↓ | 0.06256 | 0.06971 | 通过 |
| 尾巴7 NME ↓ | 0.08591 | 0.07897 | 不通过 |

聚合门槛在首次预检协议中已锁定；仍报告个体例外：31个 instance-view 中，
teacher 框/身体/尾巴分别有 **24/16/20** 个不如 student。身体平均较好不意味着
所有身体姿态都适合保持。NME 使用 GT 框像素对角线归一化。

梯度量级合格：混合 checkpoint 处辅助/原总任务梯度中位范数，shared adapter
为 **1.97%**，完整 instance head 为 **0.69%**，低于25%。问题不在梯度过大，
而在保持目标质量。所有8姿态初始化保持损失及梯度严格为0；关闭辅助项与原
loss逐值一致，学生每次loss只调用一次原匹配。167项回归测试通过，包含实例
视角数量不等时的等实例权重及 JSON 汇总恢复测试。

## 数值路径修复与审计记录

首次预检 `output/cylinder_retention_preflight_v1/` 在第一个姿态的严格自保持0
断言失败。独立 `output/cylinder_teacher_capture_mode_v1/report.json` 发现：
no_grad teacher 与梯度启用的相同模型 xy 最大差异1.1921e-7，保持loss1.6653e-14、
梯度范数2.2770e-7；相同前向模式后立即 detach 则三者全部0。该实验确认模式相关
数值差异，但未定位具体底层 kernel，不将其写成已证实的 attention 实现缺陷。

保留第一次 runner/protocol；独立 `tools/retry_cylinder_replay_retention.py` 使用
实际训练引擎的 head train 模式（冻结 encoder 仍 eval），teacher 同样启用
autograd 前向后立即 detach，随后不用其计算图。teacher 停止梯度，不更新权重。
重试完成8次 encoder/24次 head前向及逐记录核验，输出完整 quality/gradients。

重试在最终报告保存时因 numpy.int64 计数无法 JSON 序列化失败；没有重复 GPU
前向。独立 `tools/summarize_cylinder_replay_retention.py` 从完整保存行恢复，
显式转内置 int，复核代码/权重哈希和全部32个梯度记录，输出：

- `output/cylinder_retention_preflight_v1_summary/report.json`
- `output/cylinder_retention_preflight_v1_summary/teacher_quality.png`

两个失败目录均保留 status/protocol，完整 teacher 缓存留在 retry 目录；summary
记录来源文件及自身代码哈希。`passed=true` 表示恢复后的技术核验成功；真正的
训练启动决策是 **`training_supported=false`**，两者不可混用。已目视检查三项
比较图，方向和数值与报告一致；此图不是预测叠图，不能代替姿态视觉评价。

## 下一步（待执行，先不训练）

优先检查**双鼠回放姿态覆盖/泛化**，不马上叠加第二种小 loss：现在8个反复回放
训练姿态的框和尾巴已经优于原始模型，开发源却退化，至少不支持“普遍忘记双鼠”
这一解释。需要区分记住8个姿态和对同训练源未回放姿态泛化。

下一项只读审计从原manifest明确为train的双鼠记录中，排除既有8个回放姿态，
按源帧顺序等距选16个，先锁定清单再推理；不读取开发指标选姿态，不借用test。
比较原始/混合模型在原始两个视角的框、身体/尾巴、top/side检测门控及固定叠图，
同时报告与8个已回放姿态的距离及源动作分布。不得把它当作新独立测试集。
只有覆盖证据支持，才另行锁定“200个双鼠回放槽位、增加不同姿态但总800步不变”
的单因素候选；不从极线ON续训，不同时开教师保持。当前没有启动此审计或训练。
