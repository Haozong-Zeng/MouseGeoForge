# Free-Moving Mouse Core-23 关键点技术规范

日期：2026-08-04  
版本：`freemoving_core23_v1`

## 1. 目标与适用范围

本规范定义自由活动小鼠（free moving mouse）的统一关键点集合，用于：

- open field、home cage、top view、bottom view 和斜视角视频；
- 单目、RGB-D、同步多视角和 Blender synthetic data；
- 2D 姿态估计、多视角三角化、VGGT/VGGT-Omega 几何学习；
- Blender mesh/rig 配准和自由活动行为分析。

对于 free-moving 记录，本规范中的主关键点集合就是项目所需的默认输出。Water Reaching 的手掌、手腕和手指细节，以及 bottom-view 的趾部细节，不进入主集合，而由可选爪部细化头提供。

## 2. 总体设计

模型使用共享视觉/几何 backbone，并划分为四个输出层级：

1. `axial_tail_core14`：头部、躯干中轴和高密度尾巴曲线；
2. `freemoving_limb8`：四个脚掌中心和四个肢体中间关节；
3. `bottom_abdomen1`：bottom view 腹部中心点，采用视角和数据定义掩码；
4. `paw_refinement`：可选爪部细化头，不计入 Core-23。

主输出共有：

```text
14 个轴向/尾巴点
+ 4 个脚掌中心
+ 4 个肢体中间关节
+ 1 个腹部中心点
= 23 个关键点
```

## 3. FreeMoving Core-23

| ID | Canonical name | 分组 | 定义 |
| ---: | --- | --- | --- |
| 0 | `nose` | axial | 鼻尖表面中心 |
| 1 | `head_center` | axial | 头部中心；应采用稳定的头骨/mesh 锚点，而非可变轮廓质心 |
| 2 | `left_ear` | axial | 解剖学左耳中心或耳根中心 |
| 3 | `right_ear` | axial | 解剖学右耳中心或耳根中心 |
| 4 | `neck` | axial | 头颈与上躯干连接处中点 |
| 5 | `spine_center` | axial | 躯干脊柱中段锚点 |
| 6 | `lumbar_spine` | axial | 腰椎/后躯干中轴锚点 |
| 7 | `tail_base` | tail | 尾根中心 |
| 8 | `tail_q1` | tail | 尾根至尾尖中心线弧长的 1/6 位置 |
| 9 | `tail_q2` | tail | 尾巴中心线弧长的 2/6 位置 |
| 10 | `tail_q3` | tail | 尾巴中心线弧长的 3/6 位置 |
| 11 | `tail_q4` | tail | 尾巴中心线弧长的 4/6 位置 |
| 12 | `tail_q5` | tail | 尾巴中心线弧长的 5/6 位置 |
| 13 | `tail_tip` | tail | 尾尖 |
| 14 | `left_forepaw_center` | paw | 解剖学左前脚掌/掌垫中心，不使用趾尖均值 |
| 15 | `right_forepaw_center` | paw | 解剖学右前脚掌/掌垫中心 |
| 16 | `left_hindpaw_center` | paw | 解剖学左后脚掌/跖垫中心 |
| 17 | `right_hindpaw_center` | paw | 解剖学右后脚掌/跖垫中心 |
| 18 | `left_elbow` | limb_joint | 左前肢肘关节中心 |
| 19 | `right_elbow` | limb_joint | 右前肢肘关节中心 |
| 20 | `left_knee` | limb_joint | 左后肢膝关节中心 |
| 21 | `right_knee` | limb_joint | 右后肢膝关节中心 |
| 22 | `abdomen_center` | bottom | 与 `spine_center` 对应的腹侧表面锚点；主要由 bottom view 监督 |

“四肢肘部”在模型和数据格式中统一称为 `limb_joint` 组。前肢使用解剖学名称 `elbow`，后肢使用 `knee`，避免将后肢关节错误命名为 elbow。

## 4. 尾巴定义

尾巴保留 `tail_base + 5 个中间点 + tail_tip`，共 7 点。尾巴蕴含转向、平衡、加速、探索、梳理和应激等行为信息，也是头尾判别和完整身体姿态的重要依据，因此不应退化为仅预测尾根和尾尖。

尾巴中间点必须按尾巴中心线的归一化弧长定义：

```text
tail_qi = centerline(s = i / 6), i = 1...5
```

不得使用 `tail_base` 与 `tail_tip` 之间的二维直线插值。对于 Blender，应沿变形后的尾巴骨骼链或 mesh 中心线采样；对于真实标注，应沿可见尾巴中心线采样。

尾巴被截断、遮挡或运动模糊时，逐点记录可见性和标注置信度，不用虚构直线尾巴补齐缺失点。

训练时不要求 `tail_q1...tail_q5` 始终采用强逐点位置匹配。推荐将 7 个预测点按置信度拟合为从尾根到尾尖的连续曲线，以保序曲线距离为主损失，并对可见片段、尾长、曲率、多视角异常和时间连续性分别处理。`tail_base` 保持强位置锚定，`tail_tip` 在可见或多视角可信时锚定。完整设计见 [密集尾巴关键点的曲线监督设计](tail_curve_supervision_design_20260825.md)。

## 5. 脚掌与肢体关节

四个 `paw_center` 表示脚掌主体，不表示腕、踝或趾尖。其目的包括：

- 描述步态、支撑多边形和落足位置；
- 为地面接触、滑动和穿透判断提供稳定锚点；
- 在不预测全部脚趾时保留四肢末端姿态。

四个 `limb_joint` 表示前肢肘和后肢膝。它们与脚掌中心共同约束肢体弯曲方向，降低仅依赖脚掌时产生的反关节解和左右肢体交换。

肩、髋、腕和踝不属于 `freemoving_core23_v1` 的强制输出：

- 肩和髋经常被躯干与毛发遮挡，可作为 kinematic latent joint 或可选监督；
- 腕、踝与脚掌中心的语义距离较近，只有细粒度任务才需要独立输出；
- Blender rig 中应继续保留这些关节，用于 IK、骨长和运动学约束。

## 6. Bottom-View 腹部中心头

`abdomen_center` 是一个独立、视角条件化的点。它不是二维腹部 mask 的质心，而是与 `spine_center` 对应的腹侧表面位置。

监督规则：

- bottom view：有清晰腹侧区域时提供强监督；
- 斜下视角：可见时提供带置信度监督；
- top/side view：保留输出通道，但将 `defined_by_dataset` 或 `surface_visible` 设为 false；
- Blender：同时导出内部 `joint_world_xyz` 和腹侧 `surface_world_xyz`；
- 真实图像：标注腹部可见表面锚点，不把背部投影位置当作可见腹部深度。

该点用于 bottom-view 身体中心定位、腹侧朝向、身体高度估计和跨视角 torso 对齐。

## 7. 可选爪部细化头

`paw_refinement` 不改变 Core-23 的通道定义。它采用任务和数据集可配置的 query 集合，并通过缺失标签掩码训练。

### 7.1 Water Reaching

Water Reaching 主要启用左右前爪细化：

- `left_wrist`、`right_wrist`；
- `left_palm_center`、`right_palm_center`；
- 每根手指的根部、中间关节和指尖；
- 可选 `mouth`、`spout_tip` 和接触点。

当前 Blender rig 中名为 `Wrist_L/R` 的骨骼头实际对应左右肘。数据转换时必须映射为 `left_elbow/right_elbow`，不得将其当作真正的 wrist。真正的手腕点需要从相应骨骼尾部或独立 mesh 锚点导出。

### 7.2 Bottom-View Paw Refinement

Bottom view 可按需要启用：

- 前后脚掌接触中心；
- 可见趾尖或趾根；
- 腕/踝表面锚点；
- paw/contact mask 和地面接触概率。

Water Reaching 和 bottom view 可以共享细化头结构，但使用不同 query availability mask。Water Reaching 不需要后爪手指细节，普通 bottom view 也不必强制采用 Water Reaching 的四指三关节定义。

## 8. 坐标、左右侧与可见性

Canonical body frame：

- 原点：`lumbar_spine`，或由它定义的 pelvis/root；
- 正 X：从 `tail_base` 指向 `nose`；
- 正 Y：动物解剖学左侧；
- 正 Z：从腹侧指向背侧。

所有 left/right 均指动物自身左右侧，与图像左右无关。水平翻转增强必须同步交换左右标签。

每个点、每个视角独立保存：

```text
defined_by_dataset
in_frame
surface_visible
occluded_but_annotated
confidence
joint_world_xyz
surface_world_xyz
camera_xyz
uv
surface_depth
```

内部关节位置与可见毛发表面位置不得混为同一深度目标。肘、膝等内部关节使用 `joint_world_xyz` 约束运动学，深度可见性使用附近的 `surface_world_xyz`。

## 9. 跨数据集映射

### TopViewMouse-27

- `nose`、双耳、`neck`、`tail_base`：直接映射；
- `head_midpoint -> head_center`；
- `tail1...tail5, tail_end -> tail_q1...tail_q5, tail_tip`；
- `mid_back`、`mouse_center`、`mid_backend`、`mid_backend2`、`mid_backend3`：经视觉审计后映射或插值到 `spine_center/lumbar_spine`；
- shoulders、hips、eyes、ear tips 和 body sides：保留为数据集辅助监督；
- TopViewMouse 不提供脚掌、肘和膝，对应通道使用 `defined_by_dataset=false`。

### Tail-Focused Dataset

其 14 点定义可直接映射到 `axial_tail_core14`。需要确认额外 6 个尾巴点是否按中心线弧长等分；若不是，应先拟合尾巴曲线，再重采样到本规范位置。

### Water Reaching

- 真实 DLC/Anipose 当前没有显式肩、肘标签；
- `l_mid_A/r_mid_A` 可映射或辅助监督前掌中心；
- 手指、鼻子、嘴和出水口进入可选细化头；
- synthetic Blender 数据可以为 Core-23、肘关节和细化头提供完整监督；
- 真实数据缺失点采用 gradient mask，不使用错误伪标签填充。

### Bottom View

- 主要监督四脚掌中心、四肢中间关节和 `abdomen_center`；
- 中轴与尾巴仍保持 Core-14 语义；
- 不可见背侧点可以保留 2D/3D 隐式位置，但必须与表面可见性分开记录。

## 10. 模型输出与损失

推荐共享 backbone 后使用：

```text
core23_heatmap_head        [23, H, W]
core23_visibility_head     [23]
core23_uncertainty_head    [23, ...]
optional_paw_query_head    [K_task, ...]
body_mask_head
optional_surface_cse_head
```

建议损失包括：

- 逐点 2D heatmap/coordinate loss；
- visibility 和 uncertainty calibration；
- 多视角 Gaussian-ray triangulation；
- 固定 rig 重投影和时间一致性；
- 尾长、相邻尾段长度与曲率平滑；
- 四肢骨长和合理弯曲方向；
- 脚掌-地面接触与穿透惩罚；
- `abdomen_center` 的 bottom-view 条件监督。

尾巴 7 点应先在组内求平均，再与 axial、paw、limb 和 abdomen 组加权，避免仅因尾巴点数量多而支配总损失。

## 11. 评估

必须按语义组分别报告：

- axial/head PCK、OKS 和 3D error；
- paw-center 2D/3D error、落足和接触误差；
- elbow/knee error、左右交换率和反关节率；
- tail-base、tail-tip error，以及整条尾巴曲线距离；
- tail length/curvature temporal consistency；
- bottom-view abdomen-center error；
- 可见点、遮挡点和 source-undefined 点分组指标；
- raw metric 3D 与相似变换对齐后的 3D 指标。

整体平均误差不能替代分组报告，否则容易被躯干和头部的低误差掩盖尾巴、脚掌与关节问题。

## 12. 项目决策

项目的 free-moving 默认关键点规范确定为 `freemoving_core23_v1`：

```text
Core-14 轴向/尾巴
+ 四脚掌中心
+ 左右前肢肘关节
+ 左右后肢膝关节
+ bottom-view 腹部中心
```

Water Reaching 与 bottom-view 趾部分析使用可选 `paw_refinement`，不扩张或改变 Core-23。其他数据集提供的肩、髋、眼、耳尖、身体侧缘和额外轮廓点继续作为辅助监督，不作为 free-moving 主输出的硬性要求。
