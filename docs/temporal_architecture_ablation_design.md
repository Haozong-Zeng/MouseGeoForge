# 固定机位多鼠时序架构与消融设计

最新决策：主实验已改为 T（时序）× J（深度—关键点显式耦合）× R（尾巴曲线监督）8 组，
见 [联合调研与工作记录](depth_pose_joint_research_and_plan.md)。本文第 5 节的 2×2
时序模块/时序损失矩阵保留为二级诊断，不再是主实验矩阵。

日期：2026-09-09。状态：**设计规格，尚未实现 temporal adapter 或启动消融训练**。
目标不是把姿势变平滑，而是在遮挡、同色鼠接触、快速动作中利用历史视觉证据，
改善实例身份、2D/3D 姿态与几何一致性，同时保留可严格比较的无时序模型。

## 1. 当前实现边界

- `vggt-omega/vggt_omega/models/vggt_omega.py` 接受 `[B,V,3,H,W]`。
- Aggregator 的 `inter_frame_blocks` 在当前多视角输入中混合的是同一时刻的视角，
  名称中的 frame 不意味着训练已利用连续视频。
- `MultiInstanceMouseHead` 有 Q=6 个无序 query，D=256；解码 Core23、框、mask、ReID。
  内部已有 instance/keypoint token，但没有公开跨时间 token/state 接口。
- 按实例的标定三角化、3D 和重投影损失已有实现；推理跨视角身份关联仍需补齐，
  不能用训练时的 GT instance IDs 代替推理关联。
- Blender 可靠动画已有总目录/分段索引，逐帧 loader 可用；尚缺返回整段张量的 wrapper。
  DANNCE 有独立完整时序 loader，但主训练入口尚未统一接入。
- 现有空间训练入口的 `split_scenes` 不会自动采用 motion-source 分组划分。
- 混合数据历史 schema 含 Core23 与 forelimb 29 点；先明确按名称映射到 Core23，
  缺失点 valid=false，不把两个 schema 简单并成 52 点用于本轮核心实验。

## 2. 推荐主线：空间网络不变，追加可旁路的实例时序残差

```text
images [B,T,V,3,H,W]
  -> reshape [B*T,V,3,H,W]：T 进入 batch，不进入多视角 attention
  -> 同一个 VGGT-Omega + instance/keypoint spatial decoder
  -> 当前时刻跨视角几何关联 -> 当前检测及 coarse 3D（可不可用）
  -> 预测驱动 track association -> 历史 token/memory
  -> identity / frame-only MLP / temporal adapter（可切换）
  -> 原有框、mask、关键点预测层及 image-conditioned local refinement
  -> 同一空间损失 + 独立开关的时序损失
```

空间输出必须先产生，才能提供关联用的当前 ReID、位置和不确定性。第一版允许
先形成 coarse 预测、再用 adapter 修正 token 并重复轻量预测层，缓存像素特征，
不重复跑 backbone。基线也走同一 coarse/final API，便于控制计算差异。

建议 API（尚未实现）：

```python
forward_clip(images, cameras, timestamps, valid, state=None, temporal_mode="off")
# predictions: [B,T,V,Q,...]
# state: session-scoped predicted track memory, never a GT-ID lookup
```

`temporal_mode=off` 必须真实 bypass，既不读取也不更新影响姿态的历史状态；
不是仅把 loss 权重设为零。T=1 与旧空间网络在同输入/精度下应数值等价。

### Token 结构与融合

- 维持视角实例 token `[B,T,V,Q,D]`、关键点 token `[B,T,V,Q,K,D]`；
  可靠跨视角对应可汇聚为 track token `[B,T,M,D]`。
- 跨视角关联结合 ReID、极线约束、重投影误差、预测不确定性和可用 3D。
  使用训练参数固定的预测关联；GT 匹配只用于监督和单独标注的 oracle 诊断。
- 首选小型 2 层、D=256、8-head 的 **causal temporal cross-attention adapter**。
  当前实例/关节 token 查询最近 8/16/32 个时刻的关联特征；初始默认窗口 16。
- 采用 `z_out = z_spatial + gate * temporal_residual`，残差输出投影零初始化，
  gate 正常初始化，避免 gate 与投影同时为零导致梯度停滞。低关联置信度时旁路历史。
- 关节使用类型 embedding，身体/头部、四肢、尾巴可分别设置 gate；不把尾部强制刚性化。
  先共享 adapter，只增加轻量类型条件；分支化作为后续消融，避免一开始引入过多变量。
- 预测身份不确定、离场或重新出现时允许 new-track/no-match；长期遮挡不强行绑定旧鼠。
  第一版不在实例之间混合历史；接触关系建模作为后续 inter-instance attention 扩展。
- 每条 track 只保留固定预算历史 token 与时间，不缓存整段视频的 dense patch。
  推理每个时刻计算一次，不能每个窗口重复跑历史 backbone。训练短段内允许梯度，
  跨 clip state detach；第一版独立 clip 训练默认重置 state。

### 固定相机与静态背景

相机 K/E 按 recording 缓存一次，维度 `[B,V,...]`。有真实标定时所有消融组共享该标定，
称为 calibrated track；不能只给时序模型标定而让基线自行估计。
无标定实验单列：使用相同前缀估计一套 rig 后冻结，并固定坐标原点/尺度。
禁止在 causal 实验中用整段未来图像估计相机或背景。

静态 rig/background token 不等于静态场景假设：前景深度/关节随时间变化，
不能对运动小鼠施加“同一像素深度不变”。第一轮不新增背景记忆；已有标定作为公共输入。
后续单列 fixed-rig memory 消融，不能混进“有无 temporal adapter”的主比较。

## 3. 数据与时间契约

统一 window wrapper 接受 `TemporalWindowIndex` 的记录，区分 time 与 view，
为整个窗口固定 camera IDs/order；几何增强及其 K/E 变换必须逐帧一致。
归一化原点/尺度也按 clip/rig 固定，不允许每帧独立 canonicalization 消除真实运动。
Blender 单位从 metadata 显式转换，DANNCE 已为 mm；单位未知时屏蔽 metric loss。

必须区分三种概念：帧是否存在、实例是否有标注、实例是否真的不存在。
DANNCE 未人工标注帧不能当空目标；无标签的 objectness/pose 损失不能被错误打开。
mask 与 depth 的 pseudo 标志及置信度保留，不能提升为 GT。实例和点的 valid 独立。

时间字段：`source_frame_ids`、`timestamps_seconds`、`timestamp_valid`、
`transition_valid`、`segment_id`、`motion_source_id`、`fixed_camera`。
窗口不能跨真实断帧、动作拼接、session 或 split；纯存储 chunk 边界可以跨。

- DANNCE：25 fps，源帧 0/4/8/...，dt=0.04 秒，每鼠 4500 时刻、6 视角。
- Blender：大量数据为独立姿态；只有明确标记可靠的连续动画走时序支路。
  `frame_order_only` 不等于可靠秒时钟。未核实帧率的数据只用相对步序编码，
  屏蔽 mm/s 或 mm/s² 的速度/加速度监督；把 timing validity 输入模型。
- SBeA/s-DANNCE：区分 RGB+calibration、pose-only 和 ground-aligned/camera-frame 数据。
  未对齐标定坐标的 3D 不直接参加重投影；pose-only 不能算作图像时序训练样本。
- 独立姿态训练以 T=1 使用共享空间模块；禁止复制同一姿态伪造连续样本。
- 少量可靠 Blender + 较多真实 clip 的采样比例只用训练/验证集选择，并对各消融锁定。
  长 session 不应凭帧数淹没短动画，采用 dataset/session-balanced 抽样。

## 4. 时序损失不是简单“越平滑越好”

保留空间集合匹配、2D、mask、3D、重投影损失不变，分别开关：

1. 有可靠身份监督的跨时间 ReID 对比损失；没有可靠 ID 的帧不造伪身份强监督。
2. 有相邻 3D GT 和真实 dt 的速度误差 `Huber(v_pred - v_GT)`；
   验证后才加 GT 加速度误差，尾部单独报告。
3. 无 3D GT 的视频使用可选光流/重投影对应一致性，屏蔽遮挡、出界和低置信度；
   外部光流模型与参数各组固定。第一轮可不启用，以减少依赖和混杂。
4. 骨长/几何合理性作为独立公共正则，不冒充时序架构贡献。

损失按有效实例/有效点归一化，再按 clip/dataset 汇合，避免有更多点/视角的数据
主导优化。不采用全局速度趋零损失；不能通过压低快速前肢或尾部运动换取漂亮曲线。

## 5. 主消融：2×2 因子设计

所有组使用**完全相同的 clip 列表、空间预训练 checkpoint、采样种子、帧/视角预算、
增强、标签、伪深度版本、空间损失和 backbone 冻结策略**。离散姿态和视频比例固定。
无时序模型也接收同样 T 帧，只把它们作为 B*T 个独立空间样本，按同样帧数归一化损失。

| 实验 | 姿态预测读取历史 | 时序损失 | 用途 |
| --- | --- | --- | --- |
| A0 spatial | 否，identity bypass | 关 | 主基线 |
| A1 spatial+TL | 否，逐帧预测 | 开 | 仅时序训练正则的收益 |
| A2 temporal | 是，causal adapter | 关 | 仅时序架构的收益 |
| A3 temporal+TL | 是，causal adapter | 开 | 完整组合及交互作用 |
| C0 capacity | 否，同帧 residual MLP | 关 | 控制新增参数/非线性容量 |
| C1 smoother | 否，A0 输出后处理 | 无重新训练 | 控制普通因果滤波的收益 |

A1 可以在相邻独立预测之间计算 loss；不允许任何未来/历史特征混入预测网络。
A0/A1 做 ID 指标时使用与 A2/A3 同一配置的外部 tracker，但 tracker 历史不反馈姿态。
在四组之外另报不含 tracker 的逐帧检测/姿态指标；说明关联与姿态提升的来源。
C0 参数量与 adapter 尽量匹配，报告精确差异与 FLOPs；不能宣称完全等算力。
C1 仅在验证集调 Kalman/One-Euro 等参数，报告延迟；不拿离线双向滤波作在线公平基线。

建议每组至少 3 个种子，共享空间预训练起点；stage 1 冻结 backbone，只训练 head/adapter；
stage 2 的解冻/LoRA 策略对所有组保持一致，单独报告。配置规格见
`configs/experiments/temporal_ablation_v1.json`，目前不被训练入口消费。

### 二级消融（主矩阵确认后）

- history 长度 1/8/16/32；shuffle history、reset every frame、移除 dt 编码。
- 同权重同参数的 self-only history mask 控制；不要与真正 legacy bypass 等同。
- instance-only vs instance+joint token；关节分类 gate；移除几何关联或 ReID。
- fixed-rig 输入/记忆；有无伪深度；低标注比例；不同 synthetic/real 比例。
- offline bidirectional 单列上界，不能与 causal 结果不加标记地混报。
- GT-ID association 只能作为 oracle 上界，与无 GT 推理结果分表。

## 6. 防泄漏与评估

按原视频/动作/动物分组，不按帧随机切。黑白棕毛色的同源动作必须同组；
session 的人工稀疏帧、RGB、伪深度、pose-only 衍生物也绑定同一 split。
现有 DANNCE mouse1=train、mouse2=test；不能用 mouse2 调参。另选独立源动作/动物做 val。
两只单鼠不足以支持总体泛化或多鼠身份结论，必须报告样本局限并加独立多鼠测试。

现有 DANNCE 伪标签使用测试鼠的人工提示和离线前后传播：它们不能作为独立 GT，
也不能用作因果测试输入。人工 prompt 帧列表应进入评估排除清单；若重建使用了全体
人工 3D 来定 bounds，需记录这一 provenance。正式伪深度泛化实验要重生成不使用
保留评估标注的版本。主测试仅用 RGB/同等可用标定作输入，并用独立人工标签评分。

指标至少包括：

- 人工 GT 上实例 2D PCK/OKS、3D MPJPE、重投影误差；分头/躯干/四肢/尾部。
- 左右点交换率、遮挡/接触分层误差、身份 IDF1/HOTA/ID switches（需可靠 GT tracks）。
- 有连续 GT 时速度/加速度误差、快速动作幅度与时间滞后；不能只报轨迹 jerk 降低。
- mask AP/IoU 只在独立真实 mask GT 上；相对 SAM2 mask 的分数应称伪标签一致性。
- 每时刻延迟（包含 V 视角与关联）、显存峰值、吞吐、参数量、历史预算、冷启动开销。

按 session/动物/源动作重采样计算区间，不能把高度相关的视频帧当独立重复样本。
使用同一 held-out 测试帧清单；重叠 clip 预测去重，causal 推理一次顺序扫完 session。
跨 session/相机 rig/动作断点清空 state，禁止跨 split 缓存；打乱 dataloader 时也不得串 state。

## 7. 实现顺序与验收

1. 统一 Blender/DANNCE clip contract、Core23 映射、group split，接入显式 temporal trainer。
2. 暴露空间 token 与 coarse/final decode API；先实现 identity 模式，固定视角采样和 clip 尺度。
3. 补预测驱动多视角/时间关联；匹配失败可旁路，不让 GT IDs 进入推理。
4. 增加 bounded causal adapter、frame-only capacity control；实现四组共用的 loss 开关。
5. tiny clips 单元测试后训练主矩阵；不在同一实验中同时换 encoder、标签和 temporal module。

必须通过的测试：T=1/off 等价；未来帧扰动不影响过去预测；query permutation 不改变集合结果；
同窗口 view ordering 一致；gap/segment/split 隔离；所有缺标签 loss 为零但不标为空目标；
无效 padded instance 不入 memory；单位/尺度正确；history bounded；保存恢复不串 session；
每组读到相同训练/评估清单且 frame-weighted loss normalization 一致。

## 8. 参考工作及取舍

- [DynamicVGGT](https://arxiv.org/abs/2603.08254)：研究动态 point maps、motion-aware temporal
  attention 和 Gaussian velocity/scene-flow 监督。借鉴“动态表示需要显式时间建模”，
  不直接照搬自动驾驶的场景流/全场景 4DGS 头。我们的实例/关节 adapter 是针对当前
  数据条件的设计推断，不是论文已验证的鼠类方案。
- [StreamVGGT](https://arxiv.org/abs/2507.11539)：因果 attention 与历史 KV 缓存适合在线推理。
  借鉴 causal/state 契约；暂不改造整个 VGGT backbone，以保持空间基线可比。
- [OVGGT](https://arxiv.org/abs/2603.05959)：强调固定缓存预算与坐标锚点保护。借鉴有界记忆，
  但本方案缓存 track/joint token 而非直接采用其完整 dense KV 压缩算法。

全 backbone 时空 attention、dynamic point/4DGS head 作为后续路线，不能在当前小规模
可靠时序数据上默认会优于轻量 adapter。主张必须由上述消融验证，而不是以更平滑的视频代替。
