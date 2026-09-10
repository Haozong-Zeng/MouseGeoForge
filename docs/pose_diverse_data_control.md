# 多数据集 Core23 训练：400 个训练场景

2026-09-10。承接 [收敛对照](pose_convergence_control.md)：当前头可以拟合训练双鼠，
但留出来源泛化较差。本轮按用户要求扩充数据，固定 512 输入、架构和优化协议。

## 数据与隔离

| 训练数据集 | 场景数 | 目标数 | 输入 H×W |
|---|---:|---:|---:|
| Homecage BlackBrownFur CRIM13 Diverse400 | 200 | 2 | 288×512 |
| BrownFur ClimbWall Core23 | 32 | 1 | 512×512 |
| Homecage BlackFur MoSeq Sample10 Frames200–399 | 84 | 1 | 288×512 |
| BrownFur MoSeq Sample10 First200（圆柱场景） | 84 | 1 | 512×512 |

400 个不同场景，800 张双视角图像；原来为 64 场景、128 张图像。
每轮均匀打乱场景，双鼠/单鼠各 200，保留原训练的全部 64 场景。
双鼠采用训练 CRIM13 视频 `021709_A25_Block22_BCfe1_t` 对应的全部 200 个渲染姿态，
不是将留出视频混进来，也不是生成新视频或补插帧。新单鼠来源均匀选择 84/200 帧。
两个 Sample10 片段及其毛色/环境变体保守地归为同一个 `moseq_sample10` 训练组。
这一实验分组保存在 manifest 中，不修改原数据目录的时序认定或共享分组文件。

验证集**完全不变**：原 16 个 WashFace 单鼠场景，以及原 16 个留出 CRIM13 双鼠场景。
不训练 WashFace、WalkAround、留出 CRIM13 视频或任何既有 val/test 来源。
原训练/验证清单、源计划与分组哈希受到检查；同一源动作换毛色不能跨划分。
仍使用相机 ID [0,2]；张量顺序与实际机位须查看各数据集 batch.view_ids。
新数据保持原图比例，不把 Homecage 16:9 拉伸为正方形。

## 为什么不是直接加入所有目录

已检查原训练组中多个数据集的首/中/末场景：

- Water Reaching、旧 PseudoSynthetic 标注主要是手指/腕部的 29 点，不是 Core23；
  不把 wrist 冒充 forepaw center，不将缺失关节填成负样本。需独立部分标注适配后再混入。
- CranialWindow 为单视角，本轮固定双视角协议，暂不复制同一相机充当第二视角。
- SBeA 的 3D ground alignment 与原标定坐标缺少已知变换；DANNCE 含部分标签和伪深度，
  不能不加区分地等权当作当前完整 Core23/真实深度监督。
- 本轮新增 MoSeq 数据具备完整 Core23、相机和渲染深度，已通过现有多实例加载检查。

这是第一阶段的**合成多数据集**训练，并非已经完成真实数据混合。
双鼠的源视频数仍为 1 个训练来源；主要增加双鼠姿态采样和单鼠环境/毛色/动作覆盖，
不能声称已获得跨多个真实双鼠视频的训练多样性。

## 协议、审计与运行

- 配置：`configs/experiments/pose_diverse_core23_512_v1.json`。
- baseline：`output/pose_convergence_512_15000_v1/`，固定使用已有 15k 终点，不重训挑选。
- 同 seed17、同原始 Omega 权重、相同新头初始化、冻结 encoder、depth/pose 权重均 1、
  local heatmap loss=0、bridge off；不加增强、ROI、时序、尾曲线或新 3D 损失。
- 仍 15000 步，前 5000 步 LR=1e-4，后段余弦到 1e-5；AdamW 与梯度裁剪不变。
  400 场景约 37.5 遍，旧 64 场景约 234.4 遍，因此不是等 epoch 或等计算量对照。
- 数据规模与覆盖一起改变，不把效果单独归因于“多样性”。全部 400 场景用原图重新
  提取冻结 RGB 特征，不重用不同场景/尺寸的旧缓存。
- 432 个训练+验证场景已逐个检查深度表面 anchor 对齐（median AbsRel < .15）、
  相机/关节重投影（max < .01 输入像素）、完整 mask/框/实例数和 Core23 顺序。
  PFM 仅读取时 flip_y=true，不更改源深度或标注。世界单位经相机基线尺度统一。
- 已视觉检查包含四个训练数据集的 GT 拼图；标签投影与鼠体姿态整体对应。
  新棕鼠圆柱环境含较暗画面与反射，这是真实保留的渲染外观，不自动筛掉困难画面。
- 42 项回归测试通过，包括验证清单不可更改、来源隔离、原训练样本保留及配置对照。

预检：`output/pose_diverse_core23_preflight_v1/`，内含 manifest、data_audit、
geometry_audit 及 `training_sources.png`；原目录探测见 `output/diverse_pose_source_audit.json`。

```bash
.venv-vggt-omega-eval/bin/python tools/run_pose_diversity_queue.py --launch
```

队列状态：`output/pose_diversity_queue_v1/status.json`。
模型与结果：`output/pose_diverse_core23_512_v1/`。
串行执行训练、独立重载全部 32 验证场景、导出困难帧和配对报告；禁止重复启动或覆盖旧结果。
失败停止，不自动重启。初步预算约 1.5–2 小时，实际以进度及 GPU 负载为准。

## 验收要求

配对检查初始化/LR 哈希、原始 checkpoint、验证清单及初始验证指标完全相同。
训练采样序列本来就应改变，不要求与 64 场景 baseline 的顺序哈希相同。
使用 `tools/compare_pose_diversity.py` 汇总到
`docs/experiments/pose_diverse_core23_results.json`（完成后生成）。

同时评估新的 400 训练场景和原来的 64 训练场景；不能用不同来源组成的宏平均
直接宣称训练退步或改善。分开报告单/双鼠 NME、计漏检 PCK、检测、尾部与深度指标。
固定查看双鼠 40/98（训练）和 246/393（验证），与原 512/15k 对照；
验证只使用固定最终 checkpoint，不以验证选择最优步数。结果仍是单种子合成域实验。

当前状态：预检和源代码测试通过，已从提交 `16d4bef` 启动 GPU 训练队列；
正在重新提取 RGB 特征，尚无新增数据收益结论。自动验收任务已恢复。
验收后停止自动复查，不自动追加训练、架构更改或真实数据权重实验。
