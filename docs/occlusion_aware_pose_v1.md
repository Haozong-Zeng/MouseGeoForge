# 遮挡感知姿态：实现、数据补齐与 768 诊断

2026-09-10。已实施第一阶段；没有替换生产 checkpoint、重渲染 RGB/depth、保存修改后的
Blender 项目，也没有启动全量训练。继续预测完整骨架，而不是删除所有不可见点。

## 方法依据与边界

- [Lightning Pose 3D](https://lightning-pose.readthedocs.io/en/latest/source/lightning_pose_3d.html)
  建议保留可靠的遮挡点投影标签，并用 ensemble 改善不确定性。其
  [2026 预印本对应实现](https://lightning-pose.readthedocs.io/en/stable/index.html)
  加入跨视角特征融合、patch masking、几何损失与几何异常的 variance inflation。
- [Learnable Triangulation, ICCV 2019](https://saic-violet.github.io/learnable-triangulation/)
  提供逐相机/逐关节加权三角化和体积特征融合两条路线。
- [RLE, ICCV 2021](https://jeffli.site/res-loglikelihood-regression/) 提供概率回归思路。
  本阶段是简单 Gaussian scale-only NLL，不是完整 RLE 或已校准的 Bayesian 模型。
- 单视角补全、时序先验与几何测量不能混报；两个视角没有识别任意错误观测的冗余。
  相同 VGGT 跨视角特征产生的预测可能相关，不能按独立证据盲目提高置信度。

## 已实现

### 标注与 dataloader

`keypoints.npz` 原文件保持不变，增加 `keypoint_validity.npz` sidecar：

| 字段 | 含义 |
| --- | --- |
| `xy_label_valid` | 有可靠二维坐标标签；不等于可见 |
| 原有 `in_frame` | 原导出器的画面内标记；不等于有坐标标注 |
| `visibility_valid` | 可见性标签已知；未知不能当作不可见 |
| `world_valid` | 有可靠三维标签 |
| `surface_valid` | 有可靠表面锚点；不能将内部关节等同于表面 |

原 NPZ 的显式字段优先于 sidecar；旧 NPZ 保持向后兼容的默认语义。
坐标监督有效性为 `xy_label_valid & in_frame & canonical_name_available`。
非有限坐标不参与损失，并在张量中安全置零，但零值本身不代表有标签。
可见性 BCE 独立使用已知可见性标签，不再强制依赖二维坐标标签是否存在。

旧 JSON 中只有 `xy_label_valid=true` 或 `occluded_but_annotated=true` 等明确依据
才能恢复不可见点的坐标监督；不把 `(0,0), visible=false` 的旧缺失占位符变成真值。
张量新增 `instance_keypoint_in_frame/label_valid/occlusion_cause` 与 `instance_surface_valid`。
缺失遮挡原因统一为 -1，不猜测“自遮挡”。

### 不确定性监督

沿用现有 scalar sigma 参数和 state_dict，不改变坐标头形状或前向数值。
`keypoint_uncertainty_weight` 默认 0，显式开启时计算 normalized-UV 的二维各向同性 NLL：

`0.5 * ||detach(xy_pred)-xy_gt||² / sigma² + 2 * log(sigma)`。

sigma 输入特征也 detach，仅更新误差尺度分支，不能靠移动 pose/shared features
来降低校准损失；原有全标注点坐标损失照常监督。像素尺度使用 `[W,H] * sigma`，
不能把同一个 normalized sigma 当作相同的 x/y 像素误差。
这不是自动获得校准概率；融合使用 sigma 必须明确通过独立可靠性检查。

### 多视角几何与实例关联

- `vggt_omega.utils.robust_pose_geometry.triangulate_robust`：加权线性初始化、
  至少三视角时的候选对一致性筛选、detached IRLS/Gauss-Newton、正深度和射线夹角检查。
- 单点输入为已经属于同一实例的多视角像素坐标。无 GT 输入，不自行猜实例身份。
  不足两视角、退化或冲突时返回 unsupported/NaN，不伪造零坐标或补全结果。
- sigma 未校准时禁止传入加权融合。两个视角的 supported 状态明确注明无异常值冗余。
- 训练损失可选 `geometry_mode=legacy/robust/robust_sigma`；保留按实例平均和 GT 关联的
  训练路径，不能冒充推理身份关联。几何权重全为 0 时真正跳过求解。
- 新增预测端两视角关联基线：objectness/NMS + 去尾巴的中位极线距离 + 可拒绝匹配。
  不使用 GT 框、关键点或实例 ID 建立跨视角配对，但不是完备的社交鼠跟踪器。

## BlenderPyMouse 数据补齐结果

24 个数据集、3,931 帧均有现有骨架对应的 `keypoints.npz`。补齐 3,931 个有效性 sidecar，
512,637 个有效二维标签中有 60,034 个已知遮挡点；**坐标有效集合与原 Blender 数据完全一致**。
不同数据集的 29 点手部骨架与 Core23 不强行视为同名通道，不凭空制造缺失解剖点。

双鼠 DiversePose400 额外从生产 `.blend` 的求值网格补充 `keypoint_occlusion.npz`：

- 最终交付 376 帧，4 个视角、2 个实例；与 NPZ 支持顶点逐点比较，允许误差 0.001 原单位。
- 可见 59,041；自遮挡 7,933；另一只鼠遮挡 1,868；混合遮挡 280；未解释 62。
- 原因针对无毛身体网格的支持点射线，不是完整 Cycles 毛发外观可见性。
  对不可见点要求所有支持射线均被身体网格解释，否则仍为 unknown。
- 保存 source blend SHA256 和原 NPZ SHA256；loader 拒绝已过期的原因 sidecar。
- 24 帧源求值与现有导出支持点未通过一致性检查，保留原标签、不补原因、不重渲染。
  背景重复求值的通过集合曾变化，尚不能断言是源文件版本变化还是依赖图/约束求值问题。
  例如 frame 5 的差异集中在头/耳/颈部支持点，最大约 0.207 mm，不是整体平移。
  200 帧旧源文件的该帧也未通过。**需单独解决确定性求值，再补剩余帧。**
- 14 个初次生成但最终未取得完整校验信息的原因 sidecar 已移到
  `output/occlusion_pose_v1/quarantine_unverified_causes/`，可恢复；未删除原始数据。

主报告：数据集根 `occlusion_backfill_report.json`；完整交付清单：
`output/occlusion_pose_v1/delivery_verification.json`。

## 768 小规模验证：结果而非预期

固定原 768/15k/seed17/off checkpoint；24 个训练场景只用于 sigma 分支的隔离拟合，
32 个原独立验证场景用于检查。只优化 257 个 scale 参数，300 步；没有更新
RGB encoder、adapter、坐标、检测、mask 或 depth。32 场景二维指标相对原结果的
最大差异为 `1.49e-8`，属于 CPU/GPU 指标计算舍入误差。

### 不确定性没有通过启用门槛

| 95% 预测区域实际覆盖率 | 初始 sigma | 拟合后 sigma |
| --- | ---: | ---: |
| 训练点 | 98.85% | 94.09% |
| 独立验证点 | 87.59% | 52.99% |

拟合后验证集 NLL 也恶化；训练残差明显小于跨源验证残差，scale 分支过度自信。
这不是增加 sigma 训练步数就能证明解决的问题。`sigma_only_pilot.pt` 只作失败实验产物，
不得替换生产权重或启用生产加权融合。验证集只用于一次门槛检查，未用来调温度救分。

### 双鼠三维诊断（16 场景、736 点，毫米）

| 方法 | 支持率 | 支持点平均误差 | 相同支持点的线性基线误差 |
| --- | ---: | ---: | ---: |
| 线性，GT 辅助关联 | 100% | 13.63 | 13.63 |
| 鲁棒几何，GT 辅助关联 | 65.08% | 10.32 | 10.30 |
| 加预测可见性过滤，GT 辅助关联 | 55.57% | 10.33 | 10.32 |
| 鲁棒几何，预测端关联 | 68.07% | 13.14 | 10.20 |

误差下降主要来自**拒绝部分点**，不是同一批点的重建精度显著提高。
预测端关联在全部 48 个验证实例中建立 46 条关联，44 条与 GT 辅助参照一致；
不是检测 AP，亦不能用 GT 辅助关联表格替代端到端成绩。
原 sigma 和新 sigma 都不能未经校准视为可信概率。

视觉检查 `hard_occlusion_views.png`：frame 287/296 的俯视视角中，可见点也有明显
姿态/局部定位偏移；侧视角空间压缩加重歧义。当前误差不只是遮挡点混入三角化。
按自遮挡/他鼠遮挡分层统计已输出，但验证中的他鼠遮挡样本很少，不能作强泛化结论。

## 验收与复现

- 源码回归：55 项 unittest（含 4 项时序索引）、7 项头/几何函数测试、5 项 loader fixture，
  共 67 项通过。覆盖 sigma 梯度隔离、部分标签、查询置换、尺度变化、缺视角、退化布局、
  四视角离群点、过期 source 标签拒绝和真实 geometry off。
- 24 个数据集代表场景均通过多实例 loader，单/双鼠混合 batch 通过。
- 全部 3,931 帧有效性 sidecar 数值核对通过；376 帧原因 sidecar 校验通过。
- 未执行全量训练、时序平滑、尾巴曲线监督或像素遮挡增强。

```bash
PYTHONPATH=.:vggt-omega .venv-vggt-omega-eval/bin/python tools/validate_repository_snapshot.py
.venv-vggt-omega-eval/bin/python tools/run_instance_loader_regression.py
.venv-vggt-omega-eval/bin/python tools/audit_pose_label_contract.py --apply
.venv-vggt-omega-eval/bin/python tools/verify_occlusion_delivery.py
PYTHONPATH=.:vggt-omega .venv-vggt-omega-eval/bin/python tools/diagnose_occlusion_pose.py
PYTHONPATH=.:vggt-omega .venv-vggt-omega-eval/bin/python tools/diagnose_occlusion_pose.py --report-only
PYTHONPATH=.:vggt-omega .venv-vggt-omega-eval/bin/python tools/visualize_occlusion_audit.py
```

Blender 原因补齐通过后台 Blender 执行 `BlenderScripts/backfill_dual_occlusion_causes.py`，
指定生产 blend 和 `--dataset`。默认严格遇到不一致即停止；`--skip-mismatched` 仅跳过并记录，
不放宽误差、不覆盖旧坐标。新渲染若改变 keypoints.npz，需要重新生成相应 sidecar。

## 后续顺序

1. 优先核对剩余 24 帧的确定性求值与 RGB/标注一致性；不要以当前源网格强制覆盖旧标签。
2. 建立独立于 pose 训练和最终测试的校准集，或使用 out-of-fold/多种子 ensemble 残差；
   不再仅以已拟合训练样本的残差标定泛化不确定性。
3. 加强实例局部定位/跨视角关联，再验证鲁棒融合的同点误差与覆盖率。
4. 像素级遮挡增强必须重新编码 RGB 或产生独立缓存版本；在冻结全局特征后遮挡 token
   不等价，不能直接混入旧缓存。三维强监督先使用通过审计的 Blender 几何。
5. `configs/experiments/occlusion_aware_pose_768_v1.json` 已准备但未启动：保留原 768 数据预算，
   增加隔离 NLL（权重 0.05）；sigma 融合关闭、3D/重投影权重仍为 0，权重尚非验证最优值。
6. 时序 × 显式深度联合 × 尾巴曲线的 8 组实验应共享最终通过验收的可靠性基础；
   本轮不额外打开这三个开关，也不把帧编号当作可靠时间连续性。
