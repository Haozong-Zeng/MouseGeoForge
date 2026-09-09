# 密集尾巴关键点的曲线监督设计

日期：2026-08-25  
状态：设计提案，等待消融实验  
适用范围：FreeMoving Core-23、SuperAnimal TopViewMouse、tail-focused 数据、Blender synthetic、多视角真实视频

## 1. 问题与项目决策

Core-23 使用 `tail_base + tail_q1...tail_q5 + tail_tip` 共 7 个尾巴点。中间点按尾巴中心线的归一化弧长定义，而不是 7 个彼此独立的解剖关节。

在 top/bottom view 中，尾巴中心线通常较完整；在 side/oblique view 中，尾巴可能被身体、cylinder、支架或运动模糊遮挡。同一个可见尾段也容易被不同标注者或不同视角分配给相邻的 `tail_qi`。若对 7 点逐点使用严格位置损失，会产生三个问题：

1. 相同或非常接近的几何曲线，仅因弧长采样位置略有偏移而被重罚；
2. 一个错误视角中的中间点可通过多视角三角化把整条 3D 尾巴拉偏；
3. 尾巴有 7 点，未经组内归一化时会在总损失中被重复计权。

项目采用以下方向：

- 保留 7 个输出通道，维持数据兼容性、可解释性和尾根到尾尖的方向；
- `tail_base` 保持强位置锚定，`tail_tip` 在可见或多视角可信时保持中等位置锚定；
- `tail_q1...tail_q5` 不再作为五个强独立位置目标，而作为拟合连续尾巴曲线的带置信度观测；
- 预测点和 GT 点分别拟合平滑曲线，主要比较曲线而不是控制点；
- 对遮挡视角只比较可见曲线片段，并对整视角异常进行鲁棒降权；
- 额外约束尾长、曲率、连接性和时间连续性，防止曲线损失产生折返、缩短或抖动。

这是一项项目设计推断，不是下列论文中某一个方法的直接复现。

## 2. 为什么不能只用一种距离

### 2.1 纯逐点损失

```text
L_point = mean_i Huber(pred_i - gt_i)
```

优点是语义明确，缺点是把弧长参数误差当成空间误差。它适合尾根、可见尾尖和训练初期，不适合长期强约束全部中间点。

### 2.2 纯 Chamfer 距离

Chamfer 对采样密度和参数位置较鲁棒，但把曲线当作无序点集。Curve-GCN 指出，无序匹配不利用多边形/样条的顺序，可能允许不合理拓扑；其工作因此采用保序点匹配。对尾巴而言，纯 Chamfer 还可能容忍从尾根折返、局部点堆积和错误的尾根/尾尖方向。

结论：Chamfer 可作为辅助形状项或评估指标，不能单独作为主损失。

### 2.3 对应参数位置的曲线采样损失

把预测与 GT 曲线各自密集采样后，在相同参数 `s` 比较，比直接比较样条控制点合理。曲线建模和矢量地图工作均显示，控制点偏差不等价于实际曲线偏差，恢复曲线后监督更直接。

但严格比较相同 `s` 仍会惩罚局部弧长重参数化，因此本项目优先使用保序的 soft-DTW divergence。

### 2.4 soft-DTW 与 soft-DTW divergence

soft-DTW 对两个有序序列进行可微单调对齐，允许局部拉伸和压缩，适合比较同一条尾巴上采样位置略有偏移的两组点。原始 soft-DTW 有熵正则偏差，甚至可能不是非负且不在两序列相同时取唯一最小值；因此实现时采用 2021 年提出的 soft-DTW divergence，而不是直接把原始 soft-DTW 值当作距离。

## 3. 数据表示

每个样本建议提供：

```text
tail_names                  [7]
tail_world_xyz              [7, 3]       # 有完整 3D GT 时
tail_uv                     [V, 7, 2]
tail_defined                [7]
tail_in_frame               [V, 7]
tail_surface_visible        [V, 7]
tail_label_confidence       [V, 7]
tail_centerline_polyline    [V, N, 2]     # 可选，允许 N 可变
tail_centerline_visible     [V, N]        # 可选
tail_part_mask              [V, H, W]     # 可选
tail_occluder_mask          [V, H, W]     # 可选
```

必须区分：

- `full_curve_gt`：Blender、可靠多视角 3D 或人工完整标注提供的整条曲线；
- `visible_fragment_gt`：单视角只标注真正可见的尾段；
- `undefined`：数据集没有定义，不产生梯度；
- `occluded_but_annotated`：有可信 3D 位置但该视角不应使用可见表面/像素位置强监督。

禁止为了凑齐 7 点，在不可见区域用尾根到尾尖直线补点。

## 4. 可微曲线拟合层

### 4.1 输入

关键点头继续输出 7 张 heatmap、visibility logit 和 pixel covariance。通过 soft-argmax 得到按 `tail_base -> tail_tip` 排列的点 `p_i`，使用如下权重：

```text
w_i = defined_i * label_conf_i * predicted_visibility_i / uncertainty_i
```

训练初期应裁剪 `w_i`，避免模型通过无限放大 uncertainty 逃避监督。

### 4.2 推荐样条

首版使用 clamped cubic B-spline。Catmull-Rom 可作为实现更简单的基线。B-spline 控制点 `c` 通过带平滑项的加权最小二乘得到：

```text
c* = argmin_c sum_i w_i ||B(s_i)c - p_i||^2
                 + lambda_fit_smooth ||D2 c||^2
```

其中 `s_i = i/6`，`D2` 是控制点二阶差分。线性方程求解可保留梯度。为减少单个错误点造成波浪，可再做 1 至 2 轮 Huber/Cauchy IRLS；鲁棒权重建议 `detach`，防止出现通过操纵权重降低 loss 的退化路径。

`tail_base` 使用 clamped endpoint 或高权重固定。`tail_tip` 仅在可见、多视角一致或时序可信时提高权重。

### 4.3 重采样

预测和 GT 曲线先高密度采样，再按累计弧长重采样为 `M=64` 个等弧长点：

```text
C_pred = {c_pred(s_j)}_(j=0...M-1)
C_gt   = {c_gt(s_j)}_(j=0...M-1)
```

等弧长重采样可降低不同样条参数速度造成的偏差。训练中可用固定密集采样加线性插值近似，避免昂贵的精确弧长求逆。

## 5. 推荐损失

总损失按尾巴组先求平均，再与 axial、paw、limb、abdomen 组组合：

```text
L_tail = lambda_base    * L_base
       + lambda_tip     * L_tip_visible
       + lambda_curve   * D_softDTW_div(C_pred, C_gt)
       + lambda_partial * L_visible_fragment
       + lambda_length  * L_length
       + lambda_curv    * L_curvature
       + lambda_temp    * L_temporal
       + lambda_mask    * L_tail_mask
```

### 5.1 锚点损失

- `L_base`：尾根强 Huber/L1 位置损失；
- `L_tip_visible`：尾尖仅在可见或 3D 可信时监督；
- 中间五点的严格位置损失只在 warm-up 使用低权重，之后降至 0 或主曲线项的 `0.05` 以下。

锚点用于确定方向，并阻止曲线匹配把尾根和尾尖交换。

### 5.2 完整曲线损失

对于 synthetic 或完整 3D GT，主项为保序 soft-DTW divergence。可额外加入低权重 symmetric Chamfer，改善局部最近表面距离，但不允许其替代保序项。

3D 曲线损失在统一的 metric world/canonical 坐标中计算；任何 median-depth 或 baseline normalization 必须同时作用于尾巴曲线和相机 translation。

### 5.3 可见片段损失

对于 side/oblique view 的可见片段，只计算 GT 可见片段到预测投影曲线的单向鲁棒距离：

```text
L_visible_fragment = mean_(g in GT_visible) softmin_(p in C_pred) rho(||g-p||)
```

不要反向要求整条预测曲线都靠近可见 GT 片段，否则被遮挡的真实尾段会被错误压缩到可见区域。若有 tail part mask，可对预测曲线的可见部分增加 mask 内部距离，但遮挡区域不产生背景惩罚。

### 5.4 几何正则

- `L_length`：预测与 GT 尾长的相对 Huber 误差；无完整 GT 时约束到个体/序列的稳定尾长；
- `L_curvature`：比较等弧长曲线的离散曲率，或惩罚超过生理上限的曲率；
- 二阶平滑只防止高频折线，不应把真实弯曲尾巴拉直；
- 对相邻采样点加入最小间距，防止点塌缩；
- 监测投影自交，但不把所有 2D 自交直接视为错误，因为 3D 尾巴在某些视角可发生合法投影重叠。

### 5.5 时间项

时间项在 metric/canonical 3D 曲线上计算，建议使用速度或加速度的 Huber loss，并由图像运动、标签置信度和遮挡状态门控。快速甩尾时不应使用过强的静态平滑。

## 6. 多视角与错误视角抑制

当前 `training/keypoint_head_losses.py` 的 `triangulate_gaussian_rays` 已使用 visibility、heatmap covariance 和迭代鲁棒权重，但它仍以固定 `tail_qi` 对应逐点融合。尾巴需要增加曲线级处理：

1. 每个视角先得到 2D 尾巴曲线和可见片段；
2. 使用固定 rig 或预测 rig 将共享 3D 尾巴曲线投影回各视角；
3. 在每个视角计算曲线残差 `r_v`，而不只看单点残差；
4. 使用 Huber/Cauchy/Tukey 或 median/MAD 得到视角权重；
5. 视角权重 `detach` 后再聚合，整视角异常不会拖动其余视角；
6. 少于两个可靠视角时，不生成强 3D pseudo label，只保留 2D 可见片段和时序先验。

视角权重应主要由实际可见弧长、heatmap uncertainty、重投影残差和 tail mask coverage 决定，不应简单写死为 top/bottom 永远可信。top/bottom 通常提供更完整尾巴，但反光、玻璃边缘、身体覆盖和运动模糊仍可能使其失效。

可增加 leave-one-view-out 检查：用其余视角拟合 3D 曲线，再评估被留出视角。若该视角的整曲线残差超过 median 加若干 MAD，则把它标为本帧 tail-view outlier。

## 7. 可选 tail mask/centerline 头

Blender 数据可以低成本导出 tail part mask。真实 top/bottom 数据也可用较稀疏的中心线或粗 mask 标注。尾巴是细长连通结构，仅用普通 Dice/IoU 容易忽略断裂。soft-clDice 通过预测 mask 与其 skeleton 的交叉覆盖强调连通性，适合作为辅助项：

```text
L_tail_mask = L_BCE_or_Dice + lambda_cldice * L_soft_clDice
```

mask/centerline 头的作用是提供“哪里像尾巴”的连续证据，曲线头负责输出有方向、可重采样、可三角化的几何表示。二者互补，不建议用 mask 质心替代尾巴曲线。

## 8. 首轮实验配置

建议从小规模消融开始，而不是一次开启所有项。

### A. Point baseline

- 保持当前 7 点逐点 heatmap/3D center loss；
- 尾巴组内平均；
- 作为旧方法基线。

### B. Curve supervision

- `M=64` 等弧长采样；
- cubic B-spline；
- 强 `tail_base`，可见 `tail_tip`；
- 中间点逐点权重降到原来的 `0.05`；
- soft-DTW divergence 为主曲线损失；
- 加尾长和轻量曲率项。

### C. Occlusion robust

- 加 visible-fragment 单向损失；
- 加整视角 Cauchy/Tukey 降权；
- 随机遮挡和随机丢弃一个视角；
- 记录被拒绝视角数量与原因。

### D. Mask assisted

- synthetic tail part mask；
- BCE/Dice + soft-clDice；
- 检验是否改善真实 side/oblique view，而非只改善 synthetic mask。

初始相对权重可从以下量级起步，最终必须按梯度范数和 held-out real validation 调整：

```text
lambda_base       = 1.0
lambda_tip        = 0.25
lambda_curve      = 1.0
lambda_partial    = 1.0
lambda_length     = 0.10
lambda_curv       = 0.05
lambda_temp       = 0.05
lambda_mask       = 0.25   # 仅有 mask 时
interior_point    = 0.05   # warm-up 后可降为 0
```

## 9. 评估指标

逐点 PCK/MPJPE 继续报告以保持兼容，但不作为尾巴的唯一指标。至少增加：

- `tail_base` 和可见 `tail_tip` 的 2D/3D error；
- 等弧长采样后的 symmetric Chamfer distance，单位 px/mm；
- ordered discrete Fréchet distance 或 soft-DTW divergence；
- HD95，避免单个像素决定最大 Hausdorff；
- 尾长相对误差；
- 曲率分布误差和不合理高曲率率；
- 时间速度、加速度与 jerk；
- 多视角重投影曲线误差；
- view outlier rejection precision/recall；
- top、bottom、side、oblique 分组；
- full-visible、partial-visible、fully-occluded 分组。

必须加入“错误视角注入”压力测试：人为把一个视角的若干 `tail_qi` 偏移或交换，比较逐点 baseline 与曲线鲁棒方案的 3D 曲线误差增幅。

## 10. 当前实现状态

2026-08-25 已完成第一阶段代码接入：

- `vggt-omega/training/tail_curve_losses.py`
  - 独立实现置信度加权 cubic B-spline 和等弧长重采样；
  - 支持 `ordered`、symmetric `chamfer` 和 soft-DTW divergence 三种距离；
  - 支持 2D 全曲线、可见片段单向距离、fixed-rig 3D 和 predicted-rig 3D；
  - base/tip、尾长、曲率和 view-level median/MAD 鲁棒权重均可独立消融；
  - 缺少完整七点定义或有效点不足时返回 masked zero，不影响混合数据训练。
- `vggt-omega/training/keypoint_head_losses.py`
  - 复用 heatmap 均值、visibility、pixel sigma 和两种 rig 三角化结果；
  - `tail_q1...tail_q5` 的旧逐点 heatmap/3D loss 可单独降权；
  - 旧配置默认仍为逐点权重 1、曲线总权重 0。
- `vggt-omega/training/finetune_blendedmvs.py`
  - synthetic 与 real keypoint head 共用同一套曲线配置；
  - CLI 已暴露各分项权重、距离、采样数、平滑度和鲁棒视角开关；
  - TensorBoard/训练日志包含 2D、partial、3D fixed/predicted 和有效样本数。
- `vggt-omega/training/test_tail_curve_losses.py`
  - 覆盖样条反传、异常中间点、非均匀参数化、方向反转、局部可见片段、错误视角降权及完整 2D/3D 路径。

当前尚未实现的是时间序列 loss、tail mask/clDice 和 3D 曲线回投后的 leave-one-view-out 检查；它们保留为第二阶段，不混入本轮静态关键点消融。

已覆盖的最少单元测试：

1. 完全相同曲线的 loss 接近 0；
2. 同一曲线采用不同非均匀采样时，curve loss 显著小于逐点 loss；
3. 反转尾根/尾尖时必须被强惩罚；
4. 单个中间点异常不会产生大幅波浪；
5. 一个错误视角不会显著拉偏三视角 3D 曲线；
6. 只有可见片段时不会把隐藏尾段压缩到片段中；
7. 无 tail 标签的数据仍可通过 missing-label mask 正常训练。

## 11. 可复现实验开关

保持旧 baseline：

```text
--keypoint-head-tail-curve-weight 0
--keypoint-head-tail-interior-point-weight 1
```

推荐首轮曲线实验：

```text
--keypoint-head-tail-curve-weight 1
--keypoint-head-tail-curve-distance soft_dtw
--keypoint-head-tail-curve-samples 32
--keypoint-head-tail-interior-point-weight 0.05
--keypoint-head-tail-curve-2d-weight 1
--keypoint-head-tail-curve-3d-weight 1
--keypoint-head-tail-partial-2d-weight 0.25
--keypoint-head-tail-robust-views
```

关键消融只需改一组开关：

- 仅 2D：`--keypoint-head-tail-curve-3d-weight 0`
- 仅 3D：`--keypoint-head-tail-curve-2d-weight 0`
- 无局部片段：`--keypoint-head-tail-partial-2d-weight 0`
- 无几何正则：`--keypoint-head-tail-length-weight 0 --keypoint-head-tail-curvature-weight 0`
- 无鲁棒视角：`--no-keypoint-head-tail-robust-views`
- 恢复严格内部点：`--keypoint-head-tail-interior-point-weight 1`
- 快速有序基线：`--keypoint-head-tail-curve-distance ordered`
- 无序形状对比：`--keypoint-head-tail-curve-distance chamfer`

## 12. 参考依据

- Cuturi and Blondel, [Soft-DTW: a Differentiable Loss Function for Time-Series](https://proceedings.mlr.press/v70/cuturi17a.html), ICML 2017。提供可微保序序列对齐。
- Blondel et al., [Differentiable Divergences Between Time Series](https://proceedings.mlr.press/v130/blondel21a.html), AISTATS 2021。修正原始 soft-DTW 的熵偏差，适合作为真正的 divergence。
- Ling et al., [Fast Interactive Object Annotation with Curve-GCN](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ling_Fast_Interactive_Object_Annotation_With_Curve-GCN_CVPR_2019_paper.pdf), CVPR 2019。说明曲线/多边形监督需要保留点顺序，无序 Chamfer 可能损害拓扑。
- Feng et al., [Rethinking Efficient Lane Detection via Curve Modeling](https://openaccess.thecvf.com/content/CVPR2022/html/Feng_Rethinking_Efficient_Lane_Detection_via_Curve_Modeling_CVPR_2022_paper.html), CVPR 2022。展示从标注点拟合 Bézier 曲线并在恢复曲线上采样监督，比直接监督控制点更贴近几何误差。
- Van Gansbeke et al., [End-to-end Lane Detection through Differentiable Least-Squares Fitting](https://openaccess.thecvf.com/content_ICCVW_2019/html/CVRSUAD/Van_Gansbeke_End-to-end_Lane_Detection_through_Differentiable_Least-Squares_Fitting_ICCVW_2019_paper.html), ICCV Workshop 2019。提供可微加权最小二乘曲线拟合先例。
- Shit et al., [clDice: A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation](https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html), CVPR 2021。为细长连通 tail mask/centerline 辅助监督提供依据。
- Kaltenmark et al., [A General Framework for Curve and Surface Comparison and Registration with Oriented Varifolds](https://openaccess.thecvf.com/content_cvpr_2017/html/Kaltenmark_A_General_Framework_CVPR_2017_paper.html), CVPR 2017。提供对参数化不敏感并利用局部切向的曲线距离，可作为后续高级消融。
- Simon et al., [Hand Keypoint Detection in Single Images Using Multiview Bootstrapping](https://openaccess.thecvf.com/content_cvpr_2017/html/Simon_Hand_Keypoint_Detection_CVPR_2017_paper.html), CVPR 2017。支持以多视角三角化和 outlier rejection 生成遮挡关键点监督。
- Härenstam-Nielsen et al., [Semidefinite Relaxations for Robust Multiview Triangulation](https://openaccess.thecvf.com/content/CVPR2023/html/Harenstam-Nielsen_Semidefinite_Relaxations_for_Robust_Multiview_Triangulation_CVPR_2023_paper.html), CVPR 2023。说明高噪声和 outlier 下需要显式鲁棒多视角估计。
- Bolaños et al., [3D mouse pose from single-view video and a new dataset](https://www.nature.com/articles/s41598-023-40738-w), Scientific Reports 2023。其结果也报告肢体关键点受遮挡影响更大，说明动物姿态评估应按部位和可见性分组，而非只看全局平均。
