# 逐层查询位置：下一训练对照的边界

状态更新（2026-09-11 22:01 UTC）：已按以下预声明计划完成阶段24，两臂各800步，
verification passed、各17/17门槛通过，但ON双鼠PCK较OFF下降0.001359，TP和重复
mask数量相同，未达到实质收益条件。不跑seed29、不追加训练；本十小时窗口已触发
连续两机制无实质收益的停止训练条件。结论见
`docs/instance_separation_cycle_results_20260911.md`。以下保留原训练前计划，不是待办。

训练前要求`output/query_position_preflight_v2/report.json`存在且passed；不得用
原v1 failed状态冒充通过。正式训练前runner及protocol需锁定。

## 单因素与共同设置

- 起点固定原seed17温度ON checkpoint，非多样性实验的续训结果。
- 两臂均使用原8双鼠姿态、原800-slot序列/768输入/1–4视角；不增加16姿态，
  不改变Hungarian或mask损失，不启用mask框后处理、NMS或时序。
- 两臂共同只更新instance_pose_head参数；encoder、shared_adapter、dense_head、
  depth_pose_bridge均不更新。与上一阶段的训练scope不同，因此必须新跑本轮OFF，
  不能拿上一阶段全头/adapter训练OFF当本轮控制。
- 保留原requires_grad flags以保留数值dispatch。用显式优化器参数名单限制更新，
  每步`model.zero_grad(set_to_none=True)`清掉全部梯度，只对选中参数clip。
  未入优化器参数即使反向产生梯度也不更新；不得把这说成“所有冻结参数无梯度”。
- 原实例head lr3e-6；原temperature lr3e-4；ON独有3072 query_position参数
  lr3e-4。fresh AdamW、weight_decay .01、clip_norm1，两臂原参数初始化相同。
- OFF安装零参数模块但不挂hook、不更新其参数；ON上下文仅在cross-attention Q
  注入每层查询向量。不改变K/V、全图可见范围、query数量或动物身份定义。
- 原pose损失配置保持；depth输出作为共同冻结分支，必须评价逐值验证其不变。
  head-only设置目的是隔离上一轮共同深度漂移，不是去掉深度验收门槛。

## 预算与判据

每臂固定新增800步；先seed17，当前不安排自动延长。需报告warm-start初始与
新同预算OFF/ON各88条，以及原历史17门槛，不能把gate_results默认15门槛当17。
第二种子条件沿用固定实质判据：ON全部17通过；双鼠检测感知PCK较OFF至少+.01；
双鼠box-matched NME<=1.02×OFF；TP不下降。此外FP与重复mask视角不能增加。
同时报告固定296是否真正找回棕鼠，不能由单帧结果取代全体验收。

若未达到，上述数据多样性与本机制将构成本周期连续两个未达收益训练机制，
停止新训练并总结，不因剩余十小时窗口尚长而堆叠第三个机制。
预检/数值诊断失败不计作已训练机制；但需要解决后才能启动正式训练。

## 数值复现说明

原v1预检的88条OFF/零ON前向全部逐值通过，但公共反向梯度不满足逐位相同。
独立不改后端的probe_v2测得：原OFF重复反向相对L2差6.74e-8，OFF与零ON差
5.73e-8；纯检测/分割损失也存在该波动，不能归咎于单独grid_sample算子。
新预检在运行前规定公共梯度相对L2差<=1e-6且有限、新层梯度非零；前向仍为0。
这是明确修订不适用于原CUDA反向的测试要求，非宣称原严格反向测试已通过。
另有CPU精确局部Jacobian测试，证明零加法对原Q的导数为恒等，新增参数梯度为
批次Q梯度之和。此数学/数值预检不是对检测收益的证明。
