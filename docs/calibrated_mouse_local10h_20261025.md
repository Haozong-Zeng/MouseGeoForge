# 本机已标定小鼠分割与毫米深度：10小时循环（2026-10-25）

用户授权：已标定多视角，优先毫米尺度深度与点云；10小时自动推进。
固定窗口：洛杉矶时间2026-10-25 02:29:14–12:29:14，11:59:14停止优化。
入口：`output/calibrated_mouse_10h_20261025/plan.json`、`queue.json`、`state.json`。
单卡串行、共享GPU锁；独立watchdog执行绝对截止；历史循环保持暂停。

## 数据与方法

V5只准入TRAIN576场景/1728视图、DEV144场景/432视图。分割另用既有真实人工mask TRAIN889、DEV447；不混旧合成mask，不读取封存模型评分。
深度是冻结VGGT末层图像特征＋已知K/E的射线代价头，96个毫米深度候选。固定capture AABB由TRAIN皮肤点范围外扩50mm得到。输入不含目标GT mask/depth/关键点；GT mask仅用于监督及明确标出的oracle-support评估。
D1/D2共用seed17、初值、场景与射线曝光，固定6000更新；D1 logdepth，D2另加0.2局部logdepth梯度。首个smoke在任何优化前发现参数组接口错误；原版本和失败保留，v2重新smoke后运行。
分割L/M：同head、同初值、同曝光，512px/global batch4，完整LR日程18000；每500更新成对暂停恢复；最少6000、最多18000。所有stratum的TRAIN loss/IoU/边界F1连续5次监测满足阈值，才视为共同收敛；DEV不选终点。
仅共同收敛且剩余时间足够，才从同一M EMA head分叉F/U各6000更新；U末1→2→4层解冻，末层LR1e-6、向前乘0.65、head3e-5。
完整链路固定使用M共同终点EMA与D1，预测mask生成bbox和深度支持域，输出逐视角NPZ及同一时刻毫米世界点云PLY。

## 当前证据

生成时间UTC：2026-10-25T09:45:23.855963+00:00。当前状态：running。

- depth_fit300_eval_v2：平均逐视图RMSE 5.395 mm，AbsRel 0.0178，可见表面点云F1@1/2/5mm = 0.432/0.724/0.970。此处使用GT mask评估支持域。
- depth_D1_6000_s17_v2_dev：平均逐视图RMSE 7.080 mm，AbsRel 0.0205，可见表面点云F1@1/2/5mm = 0.416/0.697/0.949。此处使用GT mask评估支持域。
- depth_D2_6000_s17_v2_dev：平均逐视图RMSE 7.341 mm，AbsRel 0.0213，可见表面点云F1@1/2/5mm = 0.417/0.701/0.949。此处使用GT mask评估支持域。

分割逐stratum、所有失败/跳过/部分任务、源代码/输入哈希和计算时间见相邻JSON及output内不可覆盖的job/log/checkpoint。

## 解释边界

- V5 six motion poses; rig/environment DEV not unseen-pose proof
- Real mask data not dense real depth truth; no real millimeter-accuracy claim
- Depth-only evaluation uses oracle support; full-pipeline evaluation uses predicted masks
- Hair/glass excluded from synthetic skin truth
- Single-mouse union bbox is not multi-mouse instance detection
- 这是首轮可学习性与对照实验；不自动晋升发布，不提交H100，不把短smoke耗时推算为完整训练预算。
