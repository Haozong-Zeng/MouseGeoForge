# 深度—关键点联合：调研、决策与工作记录

2026-09-09。目标：固定机位、多实例、RGB 推理；训练可用 Blender 真值、RGB-D、
人工关键点和有置信度的伪深度。不能把有传感器深度的测试与 RGB-only 测试混报。

## 核心结论

优先实现共享可训练特征和实例内局部 depth-feature 融合，再研究表面到关节的偏移。
不直接施加 `D(u_joint)=Z_joint`：Core23 包括可见表面锚点、内部关节和尾巴中心线；
深度图是首个可见表面的 camera-Z，重叠鼠甚至可能来自另一个实例。

旧代码已有语义像素加权、鼠/背景平衡、关键点邻域深度加权、按实例三角化与 3D loss。
但旧多实例入口强制关闭 depth head；独立冻结头不会因另一个头的深度 loss 获得收益。
旧 `multiview_keypoint_geometry_loss` 在给定 xy 直接采样深度的路径不应直接套用到
所有 Core23 内部关节。现有单鼠 tail-curve loss 也尚未迁移至新的多实例联合训练。

## 参考工作

| 工作 | 已核对的方法 | 对本项目的采用边界 |
| --- | --- | --- |
| [A2J, ICCV 2019](https://arxiv.org/abs/1908.09999) | 深度图上的多个 anchor 汇合为关节预测 | 借鉴局部多点证据，不宣称原方法等同于本项目 |
| [AWR, AAAI 2020](https://arxiv.org/html/2007.09590v1) | 自适应权重聚合、depth/3D offset 表示 | 后续实现表面—关节 offset；先做 feature bridge |
| [SimpleDepthPose, 2025](https://arxiv.org/html/2501.18478v2) | 可见关节周围深度采样，显式表面/关节偏移 | 可作简单 RGB-D baseline；人体固定偏移不移植到鼠 |
| [Kendall et al., CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html) | 共享多任务表示与 uncertainty weighting | 借鉴损失尺度管理；首轮固定验证集选定权重，不同时改权重算法 |
| [HEMlets, ICCV 2019](https://arxiv.org/abs/1910.12032) | 关节对的相对深度表示 | 是骨架 ordinal depth，不等同于 dense surface-depth 联合；后续路线 |

上述人体/手部结果不直接证明对毛发小鼠和 SAM2/visual-hull 伪深度有效。
PyMouseLifter 本地稿件的 Semantic Loss 强调前景/背景及小部位/大区域的权重平衡；
它提供的是监督分配思想，不是将关节深度等同于皮毛表面的依据。

## 模型与损失层次

1. RGB -> VGGT cached layers -> 共享 residual bottleneck adapter -> depth / instance head。
2. pose query 利用粗 xy 在深度 decoder feature 上取局部样本，以预测实例 mask 软加权，
   再残差更新 keypoint token。无额外 GT 输入；mask prior 仅为软选择，不声称完美实例隔离。
3. default detach depth feature，先防止姿态 loss 扭曲 depth decoder；之后单独比较打开梯度。
   共享 adapter 仍接收两任务各自 loss 的梯度。
4. 后续 surface-to-joint：相机表面点 `X(p)=D(p) K^-1 [u,v,1]`，
   预测 `J_k=sum_p a_kp*(X(p)+Delta_kp)`。同一实例、可靠观测才参与。
   表面点和内部关节保持独立 GT，不能用自由 offset 掩盖错误 depth。

必须始终保留独立 depth 和 keypoint supervision；surface 对应/offset 只在定义及标签可靠时用。
Blender 可提供强几何监督，Orbbec 提供传感器深度但未必有 3D joint；DANNCE 的 visual-hull
伪深度只作弱表面监督，不能当毫米级皮毛/骨骼间距真值。无标签帧不是空目标。
预测深度 confidence 的数值不直接视为校准概率；首轮不用它做可逃避的 loss 权重。

## 正式 8 组：三个独立开关

`T: temporal` × `J: explicit depth-pose coupling` × `R: tail-curve supervision`。

J=off 定义为**共享 encoder/adapter、两个独立 decoder 和相同基础任务损失**；
J=on 增加显式局部 bridge（将来若增加几何约束，须固定版本并明确声明）。
因此测试的是显式耦合，不是“有无任何多任务学习”。另列 pose-only 诊断，不混进 8 组。

所有组固定数据与标签、共享初始化、训练帧/视角预算、空间/深度语义权重、优化器与解冻策略。
时序辅助 loss 首轮全部关闭，T 只切换网络；如使用同一时序正则必须两组都可计算并保持一致。
R 首轮保留相同 7 点输出头，只改变 curve-vs-point supervision；曲线后处理单列。
独立的 full factorial 规格在 `configs/experiments/temporal_depth_tail_factorial_v1.json`。
旧 temporal-module × temporal-loss 矩阵保留为二级诊断，不再作为主矩阵。

预实验阶梯：P0 pose-only -> P1 shared independent heads -> P2 local feature bridge ->
P3 surface/joint offset + geometry。先验证 P1/P2 的梯度路径、off 数值等价和小规模收敛。

## 实施边界与验收

首轮位于 `mousegeoforge/`，复用 `vggt-omega/`，只新增两个向后兼容接口：
depth decoder 可选返回内部 feature；instance head 可选 token refiner。
不得修改旧默认训练行为，不启动全量训练，不把 tiny smoke loss 下降当泛化结果。

验收：off/零初始化等价；chunked depth features 对齐；CPU/GPU 有限梯度；
共享 adapter 同时接收两任务梯度；detach 开关正确；空/无效局部窗口旁路；
query permutation 等变；不同视角无误串接；实际 Blender 图像/标签 forward/backward；
checkpoint 参数覆盖显式核验。时序、3D offset、多实例曲线 loss 和完整 clip trainer 为后续阶段。

研究评价：独立 GT 上 pose 2D/3D、部位 depth、遮挡/接触、尾巴几何与运行开销；
不用参与生成伪深度的标签作独立几何验证，不把 SAM2 mask 一致性当真值分割准确率。

## 第一阶段工作记录：已完成

- 新增 `mousegeoforge.models.DepthPoseModel`，组合原 aggregator/depth/instance head。
- shared bottleneck adapter 同时给两头供特征；encoder 默认冻结。
- `coupling=off/local_depth_features`：后者在预测粗关键点附近采样 depth decoder feature，
  以当前预测实例 mask 作 detached soft prior，残差更新关键点 token。框/mask 分支仍是原空间输出。
- bridge 输出零初始化；off 真正不调用 bridge；`detach_depth_features=True` 为默认。
- depth decoder 新增可选 feature 返回，instance head 新增可选 callback；旧调用不变。
- 12 项新增 CPU 单元测试全部通过：off/初始等价、chunk feature、双任务梯度、detach、
  query permutation、view isolation、无效窗口、checkpoint roundtrip、8 组合完整性等。
- 11 项既有窗口/实例/几何回归测试通过，共 23 项。
- RTX PRO 6000 GPU 实际数据检查：BrownFur-ClimbWall 单鼠、Homecage DiversePose400 双鼠，
  各取 1 个已渲染场景、2 视角、128×128、2 个更新步。encoder/depth 的 1343 个参数张量
  均从本地原始 Omega checkpoint 覆盖，实例头/adapter 新初始化。
- 两个场景的 depth/pose loss 与四个可训练模块的梯度均有限且非零；encoder 无梯度。
  本次仅用 2D instance loss + depth loss，未将尚未验证的 surface/joint equality 加入。
- smoke 使用固定 rig median-baseline 尺度，不逐帧用 median depth 消除高度变化。
- 报告：`output/depth_pose_joint_smoke/report.json`（本地运行产物，不上传）。未保存/覆盖权重。
  两步 loss 下降仅说明训练链路接通，不是消融、收敛或泛化证据。

复现：

```bash
python tools/validate_repository_snapshot.py
python tools/smoke_depth_pose_joint.py \
  --checkpoint vggt-omega/vggt_omega_1b_512.pt \
  --scene dataset/BlenderPyMouse/PyMouseStudio-BrownFur-ClimbWall-Core23/frame_000001 \
  --scene dataset/BlenderPyMouse/Homecage-BlackBrownFur-LightningPose-CRIM13-DiversePose400-DepthHeightV1/frame_000001
```

下一步：统一完整 clip trainer 和 label-availability gating，建立按源动作划分的 P1/P2
小规模对照；再决定是否加入 offset。生产训练前需确认细粒度 semantic mask 覆盖率，
完善 per-instance/part depth balance、部分实例标注的负样本屏蔽和跨数据单位映射。
现有 smoke 明确要求完整人工实例标注，不可直接拿 DANNCE 无标注帧套用负样本 loss。

### 后续预审计更正（2026-09-09）

小规模对照的可见表面投影审计和图像检查发现：本轮所用 Homecage DiversePose400、
BrownFur ClimbWall/WashFace 的 PFM 应以 `pfm_flip_y=True` 读取。之前 smoke 的 False
设置虽能通过有限梯度测试，但深度标签上下颠倒，不能作为几何训练正确性的证据。
已更正 smoke 默认设置；正式对照在正确翻转后运行。没有修改/翻转数据目录中的原文件，
也没有把这个结论泛化到所有真实数据集。对照入口逐帧检查深度与可见表面锚点对齐。

### 配对小规模对照已完成

随后完成 300/1000 步、三种子、开/关共 12 次训练及全部 checkpoint 重载验证。
目前未发现稳定的联合收益，基础关键点定位仍不足；维持 off 默认，不启动完整 8 组。
详见 [对照结果与诊断](depth_pose_pilot_results.md) 和 [协议](depth_pose_pilot_protocol.md)。
