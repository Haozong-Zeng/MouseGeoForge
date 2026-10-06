# 共享参数任务梯度审计（2026-09-11）

结论：**不启动梯度投影/保护训练**。完整Cylinder任务与双鼠身体任务的组均值
梯度没有在两个checkpoint达到预设的持续负余弦<−.1门槛。

只用原source_mix日程：8双鼠、ClimbWall/Sample10各4个训练姿态，Cylinder每个
取其日程内首个侧底双视输入；原始/混合权重，16encoder/32head前向及分项autograd，
38.3秒，零optimizer更新。head采用train模式、冻结encoder采用eval。

| 参数组、Cylinder分项→双鼠身体 | 原始均值余弦 | 混合均值余弦 |
|---|---:|---:|
| adapter：完整任务 | −.05507 | −.02058 |
| adapter：pose任务 | −.05527 | −.02129 |
| adapter：身体坐标 | −.02916 | −.08793 |
| adapter：深度 | +.00559 | +.04690 |
| instance head：完整任务 | −.00078 | −.00238 |
| instance head：身体坐标 | +.06136 | −.27567 |

完整任务64个Cylinder×双鼠姿态对负余弦比例：adapter两checkpoint均50%，
head均53.1%。混合head的身体分项均值确实有负相关，但原始状态不是负相关，
完整任务也近正交，不将一个分项的负余弦外推为持续优化冲突。
深度对instance head梯度严格为零（coupling off），其余弦未定义，报告null，
不是把零梯度人为当成零余弦。深度通过adapter的平均方向也不是主要负相关来源。

梯度为未裁剪、未经过Adam状态的一阶诊断，不代表实际更新轨迹或因果归因。
完整任务的head梯度范数远大于身体分项，不能仅看总范数就断言哪个loss有害。

一次原Hungarian匹配供分解使用；与原loss逐值一致，身体＋尾巴贡献与权重5
完整坐标项最大误差5.96e-8（浮点求和），全部标签及原监督保留。源、代码、
权重哈希及模型state不变，173项测试通过。已查看gradient_conflict.png。
输出 `output/cylinder_task_gradients_v1/`，helper `mousegeoforge/task_gradient_audit.py`，
runner `tools/audit_cylinder_task_gradients.py`。

## 下一项决策

不继续添加未经支持的梯度机制。回到已有局部热图高熵证据：统一将logits放大
能改善底视，却伤害双鼠，因此下一候选为**输入相关、可学习的局部logit系数**，
不是统一设T=8；先验证初始化逐值/梯度等价、有效梯度与有界系数，再训练。
粗窗口半径、encoder、采样、GT监督均不变，它仍不能直接修复GT窗外问题。
此只读审计不计失败训练方法，现计数仍1。
