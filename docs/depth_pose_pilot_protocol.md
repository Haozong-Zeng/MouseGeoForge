# 深度—关键点联合开/关：小规模对照协议

日期：2026-09-09。用户授权自主推进至完成验证。目标是检查显式局部深度特征融合的
训练行为和有限验证收益，不运行时序或尾巴曲线的另外 6 组，也不修改原始数据。

## 固定条件

- 两模式：`off` / `local_depth_features`。都训练 shared adapter + depth + instance head；
  后者额外训练 bridge。encoder 冻结，bridge 输入默认 detach。
- 三个配对种子 17、29、43。每对头/adapter/bridge 初始化 SHA256 相同、训练顺序 SHA256
  相同、初始全部验证指标完全相同。每组固定最后一步权重，不按验证结果挑 checkpoint。
- 64 个训练场景、32 个验证场景，各 2 视角（相机 ID 0、2，保留实际加载顺序）。
- 双鼠：CRIM13 `021709_A25_Block22_BCfe1_t` 的 32 个姿势训练，
  `021309_A29_Block16_BCma1_t` 的 16 个姿势验证；源视频整体分开。
- 单鼠：BrownFur ClimbWall 的 32 个帧训练、WashFace 的 16 个帧验证，
  遵守现有 motion_source_groups；不用 WalkAround test。没有随机拆相邻动画帧。
- 双鼠保持 16:9，256×144；单鼠保持正方形，256×256。RGB-only 模型输入；
  相机标定用于统一 rig median-baseline 尺度，不用逐帧深度中位数归一化。
- AdamW 1e-4、weight decay 0.01、全局梯度裁剪 1；depth/pose 各系数 1；
  保持既有实例 loss 各项权重，3D/重投影项为 0，body-depth balance 0.5。
- 首轮 300 更新步。若训练集本身仍明显欠拟合，允许另建**统一更长预算**版本复核，
  所有模式/种子一起延长，保留原结果，不单独为胜负调某个模式的超参数。

## 数据与实现预审计

- 本次三个 Blender 数据集 PFM 必须 `pfm_flip_y=True`。96 场景逐个检查可见表面
  anchor 深度，且制作 RGB/未翻转/翻转对照图并目视检查；不使用内部关节 Z 等式。
- 未翻转的每场景表面相对误差中位数再按数据集取中位数：双鼠约 8.27%、爬墙
  121.05%、洗脸 199.83%；翻转后约 0.89%、1.35%、1.32%。该方向核对是格式审计，
  不是模型超参数选择。它更正此前 smoke 的错误 PFM 读取设置。
- 只纳入完整 synthetic instance/box/mask/keypoint/depth/camera 标注；仍实现显式
  `detection_annotation_complete[B,V]`，False 时未匹配 query 不作为负样本，
  已标记正实例仍监督。缺省保留旧接口行为；未声称完整真实混合标注训练器已完成。
- 冻结 RGB 特征只在 RAM 缓存一次，绑定场景/图像/视角顺序，双方使用同一份。
  GPU 验证 live forward 与 cached forward 的全部输出精确相等。
- CuDNN deterministic、TF32 关闭、固定 RNG；CUDA grid_sample backward
  仍不保证逐位确定性，因此不能把微小差异当作强证据。

## 指标与验证方式

- 匹配只使用 box + objectness，**不使用 GT 关键点选择最好的 query**。
- `box_matched_nme`：关键点像素距离 / GT 框对角线；是框匹配后的诊断指标。
- `detection_aware_pck_0.1`：NME≤0.1，同时 score≥0.5 且 box IoU≥0.5；漏检计失败。
- 另外报告可见/遮挡点、非尾部/尾部 NME、实例 mask IoU、检测 precision/recall。
- depth 报告 body / background / all AbsRel。BrownFur 的 metadata 仅写 Blender world
  units，因此不把该数据的 raw-unit MAE 冒称毫米。
- 每场景先汇总、每来源再汇总、最后两来源等权平均；同时保留逐来源/逐场景结果。
  三种子均值±样本标准差仅表示优化随机性，不是跨动物统计置信区间。
- Jon 另做推理时禁用 bridge 的诊断，不冒充重新训练的 Joff。
- 从磁盘重新加载每份最终权重，再复算独立验证集；保存预测/真值可视化并目视检查。

## 结论边界

即使 J on 改善，也只是这套小规模合成域、冻结 encoder、有限训练预算中的结果。
两侧仍共享相同 Blender 资产与机位，不等于跨真实动物/场景泛化。J on 新增可训练
参数，后续需容量匹配控制才能进一步区分深度信息与容量效应。若关键点基础头
尚未学会训练姿势，应报告欠拟合，不据此宣称深度融合无效或已经提高精度。

入口：`tools/run_depth_pose_ablation.py`、`tools/analyze_depth_pose_ablation.py`、
`tools/verify_depth_pose_ablation.py`。配置：`configs/experiments/depth_pose_pilot_v1.json`。
运行产物在 `output/depth_pose_ablation_pilot_v1/`，数据/特征/权重均不上传 GitHub。
