# 数据集时序标记与 DANNCE 完整时序扩展建议

日期：2026-09-08。本文件保留最初的只读审计与设计依据。后续已按用户要求拆分 CranialWindow、发布分段/源动作索引，并开始 DANNCE 完整 RGB 时序导出与连续短段伪深度验证；实施说明见 `temporal_dataset_delivery.md`。未启动全量 DANNCE 深度生成。

## 推荐命名

名称尾部只放粗粒度时序类型，完整事实写入 metadata/sequence manifest。不要用一个 Reliable/Unreliable 标签合并图像连续性、时间戳可靠性、关键点质量和深度质量。

| 建议后缀 | 含义 | 例子 |
|---|---|---|
| Temporal | 同一记录或动作段内的连续帧，具体缺帧/时间基准另有标记 | 三个独立的 91 帧动作数据集 |
| SegmentedTemporal | 包含多个时序段，仅允许段内构造窗口 | CranialWindow 273 帧 |
| SparseFrames | 来自原视频的稀疏采样帧，保留实际源帧号 | 当前两套 DANNCE 81/91 帧导出 |
| PoseSet | 为姿态覆盖而选取/重排的独立姿势 | CRIM13 DiversePose400 |
| TemporalUnknown | 尚未核实源帧关系 | 待审计数据 |

例如显示名可用 `Openfield-BlackFur-CranialWindow-SegmentedTemporal`、`PyMouseStudio-BrownFur-WalkAround-Core23-Temporal`。DANNCE 当前版本是 SparseFrames；完整同步导出版本可标 Temporal。Orbbec 完整序列可标 Temporal，同时 `timing=frame_order_only`，在实际时间戳核实前不计算真实速度。

建议保留稳定 dataset_id，先增加 display_name/tags 和 catalog。若后续正式改目录名称，应批量更新脚本/config/manifest 中的路径并保留 old_name 映射；dataloader 不应通过解析目录名称决定训练规则。

## 应当分别记录的事实

- temporal.mode：continuous / segmented / sparse / pose_set / unknown。
- segment_id、source_frame_id、transition_valid：硬切、缺帧、姿态重置等边界。
- timing：timestamps / nominal_video_fps / animation_fps / frame_order_only；fps、fps_base、时间单位及证据来源。
- view_count、views_synchronized、camera_motion：视角数量、同步与 fixed/moving/unknown。相机固定还应分别检查 K 是否变化，不能只看外参。
- annotation_source/availability：逐帧、逐点记录 manual/synthetic/predicted/absent；未标注帧不是空目标帧。
- depth_source：rendered / sensor_quantized / visual_hull_pseudo / gaussian_pseudo / absent；附单位、量化步长、有效性和置信度。
- calibration_status、image_space：verified/prior/unknown，raw_distorted/rectified/registered；内参适用分辨率与裁剪变换。
- PFM 行方向：项目不同历史脚本存在不同 flip_y 约定，混合加载时应显式读取，不从扩展名推断。
- motion_source_id / subject_group_id：用于跨数据集分组划分 train/val/test，防止同一动画或同一只鼠泄漏。

示例，采用闭区间帧范围：

```json
{
  "temporal": {
    "mode": "segmented",
    "timing": "animation_fps_pending_inspection",
    "segments": [
      {"id": "climb_wall", "start": 1, "end": 91, "motion_source_id": "mouse_climb_wall_91"},
      {"id": "wash_face", "start": 92, "end": 182, "motion_source_id": "mouse_wash_face_91"},
      {"id": "walk_around", "start": 183, "end": 273, "motion_source_id": "mouse_walk_around_91"}
    ],
    "invalid_transitions": [[91, 92], [182, 183]],
    "allow_cross_segment_windows": false
  }
}
```

fps 未审计前不要填 30。连续性可支持相邻帧表征、跟踪和无量纲差分，但以 mm/s 为单位的运动监督还要求可靠 dt。即使两个段拼接处视觉恰好相似，源动作不同仍应禁止跨界窗口。

## Blender 数据实测

读取四套数据全部 keypoints.npz，并比较相邻帧位移及各段的相对骨架形态。

| 数据 | 帧数 | 视角数 | 结论 |
|---|---:|---:|---|
| Openfield-BlackFur-CranialWindow | 273 | 1 | 三个 91 帧段拼接 |
| PyMouseStudio-BrownFur-ClimbWall-Core23 | 91 | 8 | 独立动作段 |
| PyMouseStudio-BrownFur-WashFace-Core23 | 91 | 8 | 独立动作段 |
| PyMouseStudio-BrownFur-WalkAround-Core23 | 91 | 8 | 独立动作段 |

CranialWindow 在 91→92、182→183 的平均关节点位移分别约 37.18、68.81 Blender 世界单位，普通帧间位移的中位数约 1.07。不能仅凭连续的目录编号认为这两处是自然动作过渡。

将每帧骨架减去 spine_center，再对整段拟合固定旋转和尺度，得到：

- 1–91 对应 ClimbWall，坐标 RMS 约 3.26e-6。
- 92–182 对应 WashFace，坐标 RMS 约 4.38e-6。
- 183–273 对应 WalkAround，坐标 RMS 约 7.92e-6。

三段的尺度均接近 1。这表明它们共享几乎完全相同的局部骨架动画，尽管毛色、场景、视角及全局位置可能不同。应共享 motion_source_id；不能把棕鼠版本放训练集、CranialWindow 对应段放测试集来声称对未见动作泛化。它们提供的是外观/视角增强，并非额外三条独立动作来源。

本轮未重新评估全部动画的接触、穿模或生物学合理性；Temporal 表示采样结构适合时序使用，不等于每帧骨骼都是真实生物学 GT。

## DANNCE 现有文件实际覆盖

检查了两组共 72 个 MP4 的流头、帧数及编号，以及 label3d_dannce.mat 的标签与同步表。

| 项目 | Mouse 1 | Mouse 2 |
|---|---:|---:|
| 同步相机 | 6 | 6 |
| 每相机视频分片 | 6×3000 帧 | 6×3000 帧 |
| 每相机总帧数 | 18,000 | 18,000 |
| 视频标称帧率 | 100 fps | 100 fps |
| 标称时长 | 180 s | 180 s |
| 图像分辨率 | 1152×1024 | 1152×1024 |
| 手工标注时刻 | 81 | 91 |
| 手工源帧号范围 | 27–17858 | 307–17707 |
| 相邻手工标注帧差中位数 | 129 | 124.5 |

视频分片 0/3000/6000/9000/12000/15000 的编号及数量衔接。每个相机的 sync.data_frame 均为 0..17999；data_sampleID 为 1..179991、步长 10，六相机帧映射一致。不能把 sampleID 直接当秒或当视频帧号；时间单位应按源定义说明。此检查未逐帧人工验证视频内容或硬件采集是否发生隐含丢帧。

关键发现：sync.data_2d [18000,44] 和 sync.data_3d [18000,66] **全部为零占位**，它们不是 18,000 帧关键点标签。真正手工标签仍仅在 labelData 的 81/91 个时刻。

两组均有完整 com.com3d [18000,3] 和 sampleID，且坐标随时间变化。COM 可辅助定位、初始化分割和限制视觉外壳计算区域，但不能代替全身 22 点人工真值。其来源/质量应另外记录。

现有 DANNCE 输出的 frame_000001 等是手工标签的导出序号，真正的视频帧号保存在 scene_meta/keypoints.npz 中。将这 81/91 帧当连续动作训练会混淆 dt，甚至把相隔十余秒的样本强行连接。

## 能否导出完整时序和深度

完整 RGB 和标定时间索引可以导出；沿用现有转换器的畸变矫正、零 skew 针孔相机和毫米单位，并按 sync 映射帧号。保留全部 100 fps 原视频和标注真值时刻，训练端可设置 frame_step=1/2/4 等读取不同时间尺度。

完整深度可以生成，但 DANNCE 原始视频没有测量深度。当前已采用的深度来源是 SAM2 mask + 标定视觉外壳，属于 foreground pseudo-depth。建议沿用已检查的这一方法扩展，不改称 sensor/GT depth，也不将 Gaussian 的渲染深度混写为同一种来源。

建议处理顺序：

1. 导出独立本地视频/图像副本、完整 sync/COM、静态标定、稀疏关键点索引；源数据依赖只用于溯源，符合用户要求的独立归档。
2. 从少量已知标注时刻或 COM 投影初始化 SAM2，在每个视角分段传播 mask。使用重叠窗口、正反向传播检查和定期重置，限制三分钟传播中的漂移。
3. 使用多视角 mask 一致性筛选，逐同步时刻生成视觉外壳深度、有效 mask、深度置信度。现有工具依赖每帧 3D 关键点给 SAM2 提示和设置 3D bounds，因此不能原样循环 18,000 帧；需改用 COM/固定场地范围和传播后的 mask。
4. 先做两组各一段数百帧的连续验证，再扩展全程；审计细尾/爪、遮挡、玻璃反光、轮廓漂移、跨视角一致性和伪深度闪烁。手工标签只能在其真实时刻用于验证，不能插值成 GT。
5. 深度初版可以保留较低的计算频率（例如每 4 帧一张，即标称 25 Hz），完整 RGB 仍为 100 fps，并明确各帧 depth_available。后续生成剩余帧时保持相同索引；不通过插值填成“测量深度”。如要求全量 100 fps 深度，则直接逐帧计算。

存储估算：两组完整数据为 36,000 同步时刻、216,000 张相机图像。1152×1024 的 float32 PFM 若覆盖每个时刻、每个视角，仅深度约 **1.019 TB（十进制）**；25 Hz 深度约 255 GB。独立复制完整 RGB 不等于必须立刻展开所有 PFM，可以在完整序列中同时保留原视频、帧索引和逐步填充的深度缓存。

PFM 深度需与 RGB 分辨率一致、单位毫米，背景/不可重建区域设无效。全量 COM 或预测骨架可当弱监督，但 `joint_world_valid` 仅对明确人工/合成 GT 开启，预测结果应有单独来源与权重。

## 下一步优先级

先建立跨数据集 catalog/temporal segments/motion-source split 规则，使当前已经完成的三条动画及 Orbbec 序列马上可以被正确采样；再扩展 DANNCE 的完整时序目录和连续伪深度。改名是可读性改进，真正防止错误训练的是这些机器可读规则。

证据：`output/temporal_dataset_audit/coverage.json`；可重跑 `tools/audit_temporal_dataset_coverage.py`。原有 DANNCE 深度方法定义位于 `tools/add_dannce_real_multiview_pseudo_depth.py`，完整 RGB 矫正逻辑位于 `BlenderScripts/convert_dannce_mouse_to_blenderpymouse.py`。
