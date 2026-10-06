# 遮挡感知第二阶段：独立帧块校准与 Blender 求值排查

2026-09-10。结论：**没有通过生产融合门槛，不增加全量训练、不替换 checkpoint。**
接续 [第一阶段](occlusion_aware_pose_v1.md)。本轮仅冻结模型推理、拟合误差尺度的一个
温度参数、比较几何结果，并对 Blender 源文件做只读求值诊断。

## 已落地的工具

- `mousegeoforge/uncertainty_calibration.py`：二维 normalized-UV 各向同性 Gaussian
  的单标量温度 MLE、覆盖率/NLL、按不确定性筛选后的误差曲线。输入必须有限且尺度为正。
  常数尺度没有可靠性排序，分位误差明确留空，不能靠任意同值排序制造收益。
- `tools/calibrate_pose_heldout.py`：先锁定校准/测试协议，再冻结 768 模型推理；
  校准集拟合温度，测试集仅评估。检查 checkpoint、原始标签哈希以及缓存记录对应关系。
- `BlenderScripts/diagnose_evaluated_geometry.py`：记录骨骼约束、网格修改器，比较求值后
  的骨骼/表面支持点与原始 NPZ，可测试重复切帧、导出器上下文、REST/POSE 重置。
  不渲染、不保存 `.blend`、不覆盖已有坐标。
- `tools/visualize_heldout_calibration.py`：生成覆盖率、可靠性排序、几何覆盖对照图。
- 新增 10 项校准单元测试；仓库 65 项 unittest、7 项头/几何测试、5 项 loader
  fixture 测试，共 **77 项通过**。`git diff --check` 通过。

## 预先锁定的数据划分

固定模型为 `output/pose_diverse_core23_768_v1/seed17_off/final_heads.pt`；本轮不更新
encoder、adapter、坐标头、深度头、检测头或已有 sigma 权重。分辨率 768×432。

从双鼠 DiversePose400 的第二段源视频 `021309_A29_Block16_BCma1_t` 选取：

- 排除原姿态训练/验证的所有场景；排除旧验证帧前后各 100 个**源视频帧**。
- 只用第一阶段通过源几何严格核验的场景，共 145 个候选。
- 按源视频帧号排序，前 48 个作为校准、后 48 个作为新测试；块间源帧号间隔 6,479。
- 每个块有 4,416 个二维标签、2,208 个可评估三维点；使用 001/003 两个视角。
- 两块均未参加姿态训练，但来自**同一段未参与姿态训练的源视频**，不是校准/测试
  之间的独立动物或独立 session。姿态仍经过合成重定向，不能外推真实动物泛化。
- 用 GT 框辅助建立预测 query 与实例的对应；不是端到端检测/跨视角身份跟踪测试。

主方案预先指定为 `temperature_original_sigma`，没有看过测试结果再更换主方案。
测试门槛：总体 95% 覆盖率 90%–99%、可见/遮挡子组 85%–99.5%、NLL 比未校准改善；
融合后同一交集点上的平均误差不超过基线 1.01 倍、P95 不超过 1.05 倍，
有效三维点覆盖率最多下降 3 个百分点。

温度仅在校准块求解：`T² = mean(||residual / sigma||²) / 2`。
原始 sigma 的 T=1.045782；第一阶段小样本训练 sigma 的 T=2.554753。
常数尺度对照也只用校准块拟合，sigma=0.029139 normalized UV。
温度不改变坐标，也不改变可靠性排名。

## 新测试结果

NLL 省略各方案共同的 Gaussian 常数，越小越好。覆盖率不是关键点正确率。

| 方案 | 95% 区间实际覆盖率 | NLL | 说明 |
| --- | ---: | ---: | --- |
| 原始 sigma | 96.20% | -5.7592 | 尺度分支未专门训练 |
| 原始 sigma × T | 96.63% | -5.7250 | **主方案：NLL 恶化，未通过** |
| 第一阶段学得 sigma | 72.85% | -4.3958 | 明显过度自信 |
| 学得 sigma × T | 98.08% | -6.2063 | 次要分析：缓解过度自信，仍偏保守 |
| 常数尺度 | 95.81% | -6.2762 | NLL 仍好于校准后的学得 sigma |

学得 sigma 的最低不确定性 25% 点，平均二维误差为 0.01746，而全部点为 0.02698，
约低 **35.3%**；原始 sigma 对应为 0.02637，区分作用很弱。
这是筛选后的误差差异，不是全体预测精度提高，且温度校准前后排名相同。
学得 sigma × T 的名义 68% 区间实际覆盖 90.81%，说明单温度 Gaussian 还不能
同时校准整个误差分布。不能仅凭 95% 覆盖率宣称已解决不确定性估计。

### 主方案三维融合

| 方案 | 有效点 / 2,208 | 覆盖率 | 同一交集 1,563 点平均误差 | 交集 P95 |
| --- | ---: | ---: | ---: | ---: |
| 均匀权重 robust | 1,711 | 77.49% | 9.62594 mm | 21.57353 mm |
| 原始 sigma × T 权重 | 1,581 | 71.60% | 9.62242 mm | 21.62754 mm |

覆盖率下降 5.89 个百分点，超过预设门槛；交集平均误差仅变化约 0.0035 mm。
`fusion_gate_passed=false`，生产融合保持关闭。没有将次要方案改名为主方案，
也没有在同一测试集反复调参直至通过。

代码检查还发现一个需要独立控制的因素：现有 IRLS 使用 `residual/sigma` 计算
Huber 权重，转折点为 3 个标准差，而最终重投影门槛固定为 8 像素。
因此 sigma 既改变测量权重，也改变鲁棒核的实际像素尺度；本次结果不能单独归因于
“不确定性不好”。本轮保留原算法以免事后改变主试验，下一轮应将这两个因素分开。

## Blender 残余求值差异

源文件：`BlenderPyMouseProjects/Homecage/` 下对应的 DiversePose400 `.blend`。
检查到两个身体网格均只有 Armature 修改器，没有 shape key；头颈骨链含旋转/尺度
限制，HeadEnd_M 含启用的 Spline IK。**这些是结构事实，还不是已证明的根因。**

- 第 5 帧黑鼠：支持点最大偏差约 0.20721 mm，骨骼点最大偏差约 0.05085 mm，
  集中在头、耳、颈；棕鼠偏差在 0.001 mm 核验阈值内。
- 同帧重复求值、前后跳帧、Blender 单线程、重建导出器隐藏/材质上下文，
  都没有消除第 5 帧差异；REST/POSE 重置也未解决。
- 第 393 帧黑鼠在重置测试中仍有支持点约 1.595 mm、骨骼点约 1.555 mm 偏差。
- 第 28 帧等在某些独立读取中能精确匹配，但先前顺序扫描曾不匹配；
  这提示求值状态/顺序值得继续追查，不能据一次通过就声称已确定修复方法。

未放宽 0.001 mm 阈值，未将新求值坐标写回旧标签。仍为 **376/400 帧**有通过验证
的遮挡原因 sidecar，剩余 24 帧保持未知。它们的原始坐标标签仍可按原有效性训练；
“未补遮挡原因”不等于删去这些帧的全部监督。
再次核验 3,931 帧：有效性 sidecar 完整、512,637 个二维标签的有效集合不变。

## 下一步顺序

1. 先做可切换的固定像素 Huber 核，将鲁棒阈值与 sigma 测量精度权重解耦；
   合成几何单元测试应覆盖全局 sigma 缩放、异常视角和共同支持点误差。
2. 次要结果支持继续研究学得 sigma 的可靠性排序，但新比较应预先锁定方案，
   使用新的源 session/动物或至少尚未检查的帧块。当前 48 测试帧从此视为已使用。
3. 如单尺度分布仍不合适，再比较分轴尺度或 ensemble；不要只延长训练或放宽融合门槛。
4. 源姿态差异需要可重复的求值/导出缓存方案。若必须修正历史姿态，应将 RGB、depth、
   mask、坐标、可见性一起重导出为新版本，不能只修改标签来“对齐”当前 Blender 求值。

## 复现与产物

```bash
.venv-vggt-omega-eval/bin/python tools/calibrate_pose_heldout.py --prepare-only
.venv-vggt-omega-eval/bin/python tools/calibrate_pose_heldout.py
# 已有冻结预测缓存时，不重新运行 GPU 推理：
.venv-vggt-omega-eval/bin/python tools/calibrate_pose_heldout.py --report-only
.venv-vggt-omega-eval/bin/python tools/visualize_heldout_calibration.py
.venv-vggt-omega-eval/bin/python tools/validate_repository_snapshot.py
.venv-vggt-omega-eval/bin/python tools/run_instance_loader_regression.py
```

本地输出（不纳入 Git）：`output/occlusion_pose_v2/calibration/` 下的 `protocol.json`、
`calibration_report.json`、`geometry_test.json`、`calibration_summary.png` 和 96 个小型预测缓存。
协议在候选集或源标签发生变化时拒绝混用缓存，需要新输出目录；不会默默重新抽测试集。
Blender 探针报告在 `output/occlusion_pose_v2/geometry_probe*.json`。

checkpoint SHA256：`0ff983559339a8c53f9bf64a31d8b3e6b061f98ebf2d886d0675ccf4c03acb27`。
本轮未提交或推送 Git，未修改源 Blender、RGB、depth、mask 或 keypoints.npz。
