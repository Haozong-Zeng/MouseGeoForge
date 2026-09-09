# VGGT-Omega 多鼠实例姿态长期架构

2026-09-09 更新：具体时序模块、实现边界与公平消融矩阵见
[固定机位时序架构与消融设计](temporal_architecture_ablation_design.md)。
该设计尚未实现；现有 Aggregator 的 inter-frame attention 在多视角训练中不等于视频时序学习。

状态：Phase 0 已落地；Phase 1 的原生 M=1/M>1 混合 dataloader 已落地（2026-09-01）；Phase 2 的按实例标定三角化、3D 与重投影损失已落地（2026-09-07）；Phase 3 的真实时序数据契约与固定相机设计已确定（2026-09-08）。旧单鼠 heatmap checkpoint 仅作历史 demo，不作为兼容边界。

## 目标

统一支持 1–N 只、同色或异色小鼠，在多视角视频中联合预测：

1. 每视角实例存在概率、框和 mask；
2. 每实例 Core23 二维关键点、可见性和不确定性；
3. 跨视角与跨时间的身份 embedding；
4. 后续基于相机内外参的三维融合与轨迹。

实例 query 是无序检测槽，不是永久身份。任何训练数据都不得用 query 下标表达 mouse ID。

## 已实现的数据流

```text
VGGT Aggregator cached layers 4/11/17/23
  -> Simple Feature Pyramid (1/32, 1/16, 1/8, 1/4)
  -> contextual patch memory (VGGT 已做 inter-frame attention)
  -> Q instance queries
       -> objectness / normalized cxcywh / 1:4 mask / ReID embedding
       -> K keypoint sub-queries
            -> coarse normalized xy / visibility / sigma
            -> image-conditioned local heatmap refinement
  -> Hungarian set matching
  -> box + mask + partial-keypoint + visibility + ReID losses
  -> matched instance_id across calibrated views
       -> differentiable triangulation
       -> per-instance 3D + reprojection losses
```

核心代码：

- `vggt_omega/models/heads/instance_pose_head.py`
- `training/instance_pose_losses.py`
- `training/finetune_multi_mouse_instance_pose.py`
- `training/test_instance_pose_head.py`

默认 `Q=6`，Core23 数据时 `K=23`。模型本身允许从数据 schema 获取其他 K，但正式多鼠实验应固定 Core23 并记录 schema 版本。

## 多鼠标签契约

新数据集应直接产生以下 padded tensor，其中 `M` 是 batch 内最大实例数：

| 字段 | 形状 | 定义 |
|---|---:|---|
| `instance_valid` | `[B,V,M]` | 该视角是否存在该标注实例 |
| `instance_ids` | `[B,V,M]` | 同一序列/场景内持久 ID；不等于 query index |
| `boxes_cxcywh` | `[B,V,M,4]` | 归一化中心、宽高 |
| `box_valid` | `[B,V,M]` | 是否有可用于 box loss 的强框监督 |
| `masks` | `[B,V,M,H,W]` | 每实例二值 mask |
| `mask_valid` | `[B,V,M]` | 是否有真实 mask 监督 |
| `keypoint_xy` | `[B,V,M,K,2]` | 归一化二维坐标 |
| `keypoint_valid` | `[B,V,M,K]` | 此数据源是否标注该点 |
| `keypoint_visible` | `[B,V,M,K]` | 标注点在该视角是否可见 |
| `keypoint_visibility_valid` | `[B,V,M,K]` | 数据源是否提供可靠可见性监督 |
| `keypoint_world_xyz` | `[B,M,K,3]` | 每实例世界坐标 |
| `world_valid` | `[B,M,K]` | 每个 3D 点是否有真实几何标注 |
| `intrinsics` | `[B,V,3,3]` | 零畸变针孔内参 |
| `extrinsics` | `[B,V,4,4]` | world-to-camera 外参 |
| `camera_available` | `[B,V]` | 相机几何是否有效 |

`keypoint_valid=False` 的点必须完全屏蔽梯度，以便混合 CRIM13 等部分标注数据。没有 mask 的真实数据只令 `mask_valid=False`，不得由少量关键点制造“精确 mask”强标签。

`BlendedMVSDataset` 现在在读取阶段统一产生 padded 实例张量。它先从所选场景发现最大实例数 `M`，然后把单鼠样本放入实例槽 0，其余槽置 `instance_valid=False`；双鼠数据保留两个真实实例。因此 M=1 与 M=2 样本可以由默认 PyTorch collate 放进同一个 batch。`single_mouse_batch_as_instances()` 只保留给尚未迁移到原生实例字段的外部调用者。

兼容输入包括：

- 旧单鼠 `keypoints.npz`：`[V,K,...]` 自动提升为 `[V,1,K,...]`；
- 新多鼠 `keypoints.npz`：原生读取 `[V,M,K,...]`、逐实例 mask 与 ID；
- PseudoSynthetic：8 个现有数据集已由对应 `AniposeSyntheticReady.blend` 补齐 816 个 29 点 `keypoints.npz`，现在提供真实 Blender 世界坐标、逐视角投影和可见性；对于未来仍缺 NPZ 的旧导出，继续回退读取 `annotations/*.json` 的 2D 点；
- 混合关键点 schema：dataloader 取名字并集并用 `keypoint_valid` 屏蔽未标注点。当前整个 BlenderPyMouse 根目录的并集为 Core23 加水到达手部 29 点，共 52 点。若实验要求固定 Core23，应用 `--scene-path-includes` 只选择 Core23 数据集。

## Phase 路线

### Phase 0：当前

- 冻结 VGGT，训练多层特征金字塔和实例姿态头；
- 用单鼠数据做数值和回归验证；
- 生成真正 2–4 鼠的 Blender 数据，必须包含逐实例 mask/ID/Core23；
- 指标：box AP、mask AP、实例 PCK/OKS、query permutation 测试。

### Phase 1：多实例二维

- 接入多鼠 padded 标签，不再经过单鼠适配器；
- 覆盖同色鼠、接触、交叉、上下遮挡；
- 增加接触/遮挡分层指标、左右点交换率、漏检和重复检出率；
- 比较冻结 backbone、adapter、解冻最后一个 within/inter-frame block。

### Phase 2：跨视角三维

- Hungarian 仍在视角内完成实例集合匹配；
- 已实现：用持久 `instance_ids` 汇合逐视角 Hungarian 结果，避免把不同实例的射线混合；
- 已实现：对每个实例、每个有效关键点进行带退化保护的可微标定三角化；
- 已实现：先在实例内平均、再跨实例平均的 3D 与重投影损失，使可见点多的实例不会压制其他实例；
- 待实现：推理阶段在没有 GT `instance_ids` 时，使用 ReID、极线、重投影和时间连续性联合关联；
- 待实现：把 head 输出的 `sigma` 纳入 uncertainty-aware triangulation；
- 加入 3D MPJPE、重投影误差和跨视角身份一致率。

训练开关：`--instance-keypoint-3d-weight`、`--instance-reprojection-weight`、
`--instance-world-scale`。默认自动/基线尺度归一化时 `world_scale=1`；若明确使用
原始毫米坐标，可把该值设置为约 `100`，防止 3D 数值量级压过二维损失。

### Phase 3：时序身份

MouseGeoForge 的目标实验通常是固定机位：同一 recording 内的内参、外参和静态背景
不随时间变化，真正变化的是小鼠实例。时序模型应显式利用这个结构，不能把
`T×V` 展平成 VGGT 的 frame/view 轴，也不应逐帧重新预测一套相机。

推荐数据与模型张量：

| 字段 | 形状 | 处理方式 |
|---|---:|---|
| `images` | `[B,T,V,3,H,W]` | 每个时刻只在 `V` 轴做多视角 VGGT 编码 |
| `instance_tokens` | `[B,T,M,D]` | 跨时间进入 tracklet memory / temporal adapter |
| `intrinsics` | `[B,V,3,3]` | 每个 clip 只存一次 |
| `extrinsics` | `[B,V,4,4]` | 每个 clip 只存一次，world-to-camera |
| `timestamps_seconds` | `[B,T]` | 统一不同帧率，时间编码使用真实 `Δt` |
| `temporal_transition_valid` | `[B,T,M,K]` | 屏蔽缺失点和原始跟踪跳点；首帧恒为 false |

模型分为两条不同时间尺度的路径：

1. 每个 `t` 独立使用同一组静态 rig token，对 `V` 个视角完成检测、实例匹配与三维融合；
2. 只把匹配后的实例 token、3D 关键点、置信度与时间间隔送入轻量 causal temporal
   adapter。adapter 输出单帧结果的残差修正、身份记忆和短期 future-pose 预测，不取代
   原有单帧头。

相机已知时冻结相机分支；未知时先从无遮挡背景帧估计一套 clip-level consensus rig，
再固定或 stop-gradient。动物 mask 区域不得参与相机/静态背景估计。若为了鲁棒性仍保留
逐帧相机残差，需加入 `SE(3)` rig-constancy loss，并把残差初始化为零。

ReID 只作为身份线索之一，推理关联还要联合极线/重投影误差、3D 运动门控和遮挡前后的
tracklet memory。指标加入 HOTA、IDF1、ID switches、遮挡后身份恢复率。

### Phase 3 数据配比与训练顺序

- 大量无时序 Blender 帧继续训练 backbone、多视角几何和实例姿态头，不伪装成视频；
- 少量高质量 Blender 动画提供精确 mask、depth、2D/3D、相机和身份连续性；
- 大量真实视频承担外观变化、接触遮挡和真实动力学。SBeA 当前可直接用于四视角图像、
  独立 3D 时序监督，其中 10 个序列另有 mask/box/身份；s-DANNCE Mouse 是 50 Hz
  pose/behavior-only 时序，不虚构 RGB 或相机；
- 初始 batch 采样建议 `50%` 静态多视角、`40%` 真实时序、`10%` 高质量 Blender 时序，
  再按验证集结果调整。时序阶段保留静态 replay，避免 temporal adapter 反向破坏单帧精度；
- 以物理时间窗口采样（例如 0.3–0.6 秒），不是把不同 fps 数据固定成相同帧数；
- 时序损失使用 future 3D、速度、骨长、低权重且 motion-gated 的加速度、ReID/tracklet
  对比学习。所有损失按实例归一化，并由 `temporal_transition_valid` 屏蔽异常跳点。

已落地的统一窗口 loader：`training/core23_temporal_sequence_dataset.py`。它保持 `T/V`
轴分离、对单鼠/双鼠做 padded instance、按 clip 返回一次静态相机，并可懒加载 SBeA AVI。

### 可借鉴的动态 VGGT 工作

- [StreamVGGT](https://github.com/wzzheng/StreamVGGT) 的 causal temporal attention、KV
  cache 和 VGGT teacher distillation 值得借鉴，但 cache 的基本单元应从全图 patch token
  改为少量实例/遮挡事件 token；固定机位不需要反复缓存静态背景。
- [CUT3R](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Continuous_3D_Perception_Model_with_Persistent_State_CVPR_2025_paper.pdf)
  的 state update/readout 适合改造成每个 mouse track 的 persistent state；训练时可做一次
  causal pass，再用最终 state 回看困难遮挡帧作为离线蒸馏目标。
- [DynamicVGGT](https://arxiv.org/abs/2603.08254) 在同一参考坐标中联合预测 current/future
  point maps、motion-aware temporal attention 和 velocity token。首版只采用 current/future
  Core23 与 motion token；在 mask/depth 伪标签成熟前，不上 dynamic Gaussian/稠密 scene-flow
  head。
- [VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long) 的 chunk overlap、SIM(3)/SE(3)
  对齐和 loop closure 主要解决移动相机的长距离漂移。固定实验机位没有这个主要矛盾，
  仅保留“分块训练、跨块 track memory”思想，不引入 chunk 相机对齐。

## Phase 0 启动方式

先做不写 checkpoint 的模型 dry run：

```bash
cd /home/haozong/projects/MouseGeoForge/vggt-omega
python training/finetune_multi_mouse_instance_pose.py \
  --blendedmvs-root /home/haozong/projects/MouseGeoForge/dataset/BlenderPyMouse/PyMouseStudio-WhiteFur-WaterReaching-FrameToFrame-FScamC-mapping \
  --dry-run model \
  --image-width 256 --image-height 256 \
  --num-views 2 --max-scenes 2 --val-scenes 0
```

正式多鼠数据接入前，不应把单鼠适配训练所得权重当作多鼠 checkpoint，也不应据此比较最终精度。
