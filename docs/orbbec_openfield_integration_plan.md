# Orbbec-Openfield 整合检查与方案

检查日期：2026-09-08。原文为首次审计方案；后续已实现九组完整 session、标注帧 BlenderPyMouse 导出、单帧/时序读取器，以及显式 YOLO 框加载。用户随后要求独立文件副本，存储策略已改为真实复制。当前使用说明以 `dataset/Orbbec-Openfield/README.md` 为准；两组 bag 和其余 29 组源导出仍属于后续阶段。

## 已核实的数据

源目录：`/home/haozong/projects/Orbbec OpenField/Orbbec OpenField`。
目标统一入口：`/home/haozong/projects/MouseGeoForge/dataset/Orbbec-Openfield`。

- 删除外层 `projects/Orbbec OpenField` 范围内 1,526,018 个 Zone.Identifier 文件，逻辑内容大小 158,705,872 bytes；复查剩余 0。未删除 RGB、深度、TXT 标注或 JSON 标注。删除未经过回收站；压缩日志保留删除路径和字节数，不包含被删标记文件的正文。
- 直接位于源目录下的 38 个 RGB-D `*-crop` 导出目录各自 JPG/PNG 数量相等，合计各 746,356 个文件。这里只对这 38 组做目录计数，并未逐张验证所有图像。
- 9 个 `*-labeled` 目录对应的完整数据有 223,725 对 RGB-D；这 9 组已逐帧号验证 RGB 与深度集合完全一致。
- 2,511 个标签帧全部找到同帧 RGB/深度和源 RGB。2,510 个非空 TXT 每行都是 25 个数、class=0、每图一个实例，即 bbox + 10 个二维点，没有 visibility 列；符合 YOLO pose `kpt_shape: [10, 2]`。
- 每组抽查 RGB 和深度均为 720×720。样本深度是 uint8 三通道 PNG，三个通道完全相同；每组抽查的 labeled RGB 与 full RGB 解码后完全相同。
- 5 个带原始 JSON 的目录有 1,234 个 JSON（1,233 个对应 TXT，另有一个额外 JSON）。通过坐标匹配核实关键点顺序。23 个文件各有一个 TXT 点与 JSON POINT 坐标不能在 0.01 px 内匹配，部分 JSON 有重复/缺失名称或冲突 visibility 属性，需要转换前逐项检查。
- 空 TXT：`Orbbec-TTA-Gcamp8s-7mon-CW-20241209-crop-labeled/rgb_000000021294.txt`。视觉检查对应图像明确有鼠，不能当成负样本；暂归入未标注数据。

| 序列简写 | 完整 RGB-D 对数 | 标签文件数 |
|---|---:|---:|
| B6J M1 | 26,476 | 301 |
| B6J M5 | 29,287 | 86 |
| FVB negative | 32,164 | 357 |
| FVB positive | 26,233 | 383 |
| Pva L2 | 20,233 | 284 |
| Pva R2 | 26,197 | 226 |
| Pva R3 | 20,146 | 336 |
| TTA CW | 21,955 | 317（含一个空 TXT） |
| TTA noCW | 21,034 | 221 |

FVB negative 的原始帧号范围为 0..33460，缺失 1,297 个帧号，其余八组帧号连续。禁止重新编号后把间断当作连续时间。TTA CW 的 `output.mp4` 编码为 35 fps、21,955 帧；这只能证明该视频的播放帧率，不能作为所有序列的采集时间戳。优先恢复采集时间戳；否则保留 frame_id，将 fps 标记为已知播放值或待核实，不默认全部 30 fps。

现有 `dataset/Orbbec-Openfield` 的两个 bag 已解包 21,362 和 19,324 对 RGB-D，共 40,686 对，有设备时间戳和原始 Y16 深度，深度单位因子为 1 mm。其设备元数据为 Gemini 335L，序列号 CP3F542000CE。

## 配准、内参与坐标系

**用户已明确确认旧导出的深度与 RGB 配准完成。** 按同一 RGB 像素网格读取旧 JPG/PNG；禁止再应用 depth-to-color 刚体变换。诊断中用 bag 标定再次注册确实使鼠体轮廓错位，和这个结论一致。

旧导出深度使用 RGB 网格，所以反投影采用 RGB 相机的有效内参；不能因为数据来自深度传感器就采用 native depth K。现有 bag 的 native 深度尚需按其记录的标定注册到 RGB，随后两条处理路径汇合。还需核实旧注册输出保存的是 RGB camera Z 还是原 depth-camera Z；在确认前记录该语义，不能将二者混称精确 GT。

同型号不保证逐台设备内参、畸变或安装外参完全一致。参考标定可以作为候选，先登记 `calibration_source=reference_device_prior`、`intrinsics_verified=false`。如能确认同一设备、相同采集 profile 与无额外 resize，可提高可信度。

若旧图像来自 1280×720 RGB 的水平居中裁剪，x0=(1280-720)/2=280，y0=0，且未缩放，则：

```text
K_crop = [[fx, 0, cx-280], [0, fy, cy], [0, 0, 1]]
```

从参考 bag 的 1280×720 RGB 参数得到候选：

```text
K_crop = [[612.3773193, 0, 362.09021],
          [0, 612.6641235, 350.6878052],
          [0, 0, 1]]
```

这一数值是条件推算，不是旧序列重新标定的结果。裁剪前若还有 resize，应完整保存像素仿射 A，以 `K_out=A K_original` 计算。不能把 fx/fy 乘以 720/1280 来模拟纯裁剪。

“深度配准到 RGB”不意味着“RGB 已去畸变”。参考 RGB 有非零畸变，native depth 的记录畸变为零；旧导出是否已 rectified 仍待核实。若需要输出针孔相机格式，应在确认后对已配准 RGB/深度/2D 点使用同一去畸变映射，并记录新 K；深度采用有效值感知或最近邻重采样，防止背景和鼠体深度混合。不能直接将参考畸变重复应用到已矫正图像。

单相机可直接选每个序列的 RGB camera frame 为世界坐标，写 `extrinsic=I`，x 右、y 下、z 前；这是坐标系选择，不是测得各次相机安装姿态相同。若需要统一的地面向上坐标，用每个序列多帧地板点拟合固定平面，地板法向与偏移确定高度和倾斜，再用场地边缘确定水平朝向与原点，并同步变换外参。仅平面无法确定平面内原点和 yaw。

## 深度单位、精度与有效性

按用户提供的编码：`q = integer(depth_mm / 2.3529)`。转换候选为：

```text
depth_mm_est = float32(q) * 2.3529
depth_valid = (q != 0) AND sensor/registration quality checks
```

不要再次取整，也不要对深度做逐图 min-max 归一化。若原编码向下取整，真实深度位于 `[q*s, (q+1)*s)`，估计偏低不到 2.3529 mm；若四舍五入，量化误差约 ±1.17645 mm。具体取整策略待源导出代码确认；量化精度不能通过写 float32 恢复。9 个样本地板中位值约 q=214..221，即 503.5..520.0 mm，量级合理。

uint8 最高码值 255 对应约 600 mm。样本未出现 255，但全量转换应检查饱和、截断或异常码；零值无效，不能作为距离为零监督。样本零值比例约 16.7%..20.8%，主要集中于外围，必须提供独立 depth_valid mask。

建议输出 PFM float32、相机 Z 深度，明确 `depth_units=millimeter`、`world_coordinate_unit=millimeter`；外参平移、3D 点与深度必须同单位。采用当前 DANNCE 实数数据的毫米约定；BlenderPyMouse 的部分旧 metadata 仅标记 “Blender world units”，因此训练混合时仍要读取/核实各源尺度，不能宣称 PFM 扩展名本身规定毫米。

保留场地墙壁的真实测量深度；不要套用合成数据的前墙透明规则。该规则是渲染遮挡处理，不应改变传感器观测。

## Core23 映射与监督强度

以下顺序由带 JSON 的五个目录通过坐标匹配确认；其余四个仅 TXT 目录结构一致且视觉抽查吻合，转换时仍逐组检查。

| YOLO 点序号 | 原名称 | Core23 |
|---:|---|---|
| 0 | 鼻子 | nose |
| 1 | 头部 | head_center |
| 2 | 左耳 | left_ear |
| 3 | 右耳 | right_ear |
| 4 | 颈椎 | neck |
| 5 | 胸椎 | spine_center |
| 6 | 腰椎 | lumbar_spine |
| 7 | 尾根 | tail_base |
| 8 | 尾巴中端 | tail_q3（语义近似，需保留映射置信度） |
| 9 | 尾巴末端 | tail_tip |

不存在四足、肘、膝标注；其余 13 个 Core23 点 valid=false，不插值成 GT。头部/脊柱点是人工图像标志点，和合成骨骼 bone head 的定义也有偏差，应记录 keypoint_definition。

YOLO TXT 无显式 visibility。对 TXT-only 点可以保留 2D 坐标有效性，但 `visibility_valid=false`；有可匹配且无冲突的 JSON 时恢复可见/遮挡状态。坐标为缺失哨兵或超范围时单独处理；不能把所有点默认为可见。

已配准深度可以为可靠的可见点提供 `surface_world_xyz`：在点附近同一前景区域取稳健深度，用正确像素射线反投影；拒绝零深度、遮挡点、薄尾边缘混合和背景泄漏。采样得到体表点，不是体内关节中心，因此初版 `joint_world_xyz` 的 `world_valid=false`。可另加 surface_valid/confidence 与量化误差权重，训练低权重体表约束；不把体表点冒充高精度骨骼 3D GT。

原始 TXT 没有分割轮廓。可通过多帧地板平面、相对高度、RGB 和框生成实例伪 mask，另存 mask_source/confidence。对尾巴和伏地部位不能仅靠高度阈值。未确认质量时 `mask_valid=false`，更不能用整个 bbox 充当分割 GT。

## 建议统一目录与 dataloader 接入

保留当前 bag/unpacked 文件；在同一 `dataset/Orbbec-Openfield` 根目录建立统一 manifest 和新的 sessions 子目录，避免立即搬动旧路径。逻辑上包含 legacy quantized exports 与原 bag 两种 source_type，可按相同 schema 读取。

```text
dataset/Orbbec-Openfield/
  metadata.json
  sequences.jsonl
  splits.json
  sessions/<session_id>/
    calibration.json
    frames.jsonl                  # source_frame_id、时间戳/可信度、各监督 availability
    frame_<source_frame_id>/
      blended_images/00000000.jpg
      rendered_depth_maps/00000000.pfm
      depth_valid/00000000.png
      cams/00000000_cam.txt
      keypoints.npz
      annotations/00000000.json
      scene_meta.json
      masks/00000000.png          # 仅有通过验收的伪 mask 时
```

每个时间点只有 view 0；实例 N=1，但沿用多实例 schema：uv=[V,N,23,2]，world=[N,23,3]，instance_ids=[0]。跨序列使用 session_id 限定实例 ID，避免将所有鼠当成同一身份。缺失监督由明确的 valid mask 表示。相机 TXT 的 identity 外参仅在 camera-world 模式下成立。

当前 `vggt-omega/training/finetune_blendedmvs.py` 会过滤视角数小于 `args.num_views` 的 scene；单视角数据必须单独使用 num_views=1 的 source sampler，或扩展混合 batching 以 view_valid padding。不能重复同一帧充作多个相机，也不能把不同时间的运动鼠视为同步静态多视角。时序应另用 T 维组织，形式 [T,V=1,N,23,...]，为缺帧、未知 dt 和缺标注设置 validity。

现有 keypoints loader 支持 world_valid 和 visibility_valid，可接入部分 2D 标注；要训练体表深度约束，还需要显式 surface_valid/confidence，而不是复用 joint world_valid。该扩展为后续实施事项，此轮未修改训练代码。

训练/验证按动物与记录分组，同一鼠跨会话、相邻帧及标签副本不能跨 split。不能简单随机拆每个 frame scene。

推荐分阶段物化：先完成全部 2,511 个标注帧的标签 QA 和 RGB-D/相机验收，导出其中可靠的监督；将 9 个完整序列的 223,725 帧用 manifest 纳入可按需加载的深度/时序池，再接入两组原 bag，最后扩展其余 29 组。避免立刻把 746,356 张深度全部膨胀为约 1.55 TB 未压缩 PFM（720×720×4 bytes，十进制），可对训练采样帧建立 PFM 缓存，完整源深度保持无损原 PNG。

## 审计产物与参考

- `output/orbbec_openfield_audit/summary.json`：覆盖范围与帧号间断。
- `output/orbbec_openfield_audit/labeled_pairs.json`：每组配对、JSON 名称、样本深度统计。
- `output/orbbec_openfield_audit/label_summary.json`：点顺序、空标签、JSON 冲突明细。
- `output/orbbec_openfield_audit/inventory.json.gz`：目录统计及 Zone.Identifier 删除日志（压缩保存）。
- `output/orbbec_openfield_audit/labeled_rgb_depth.jpg`：RGB 点标注与深度预览。
- `output/orbbec_openfield_audit/registration_preview.jpg`：对照诊断；左侧原导出已对齐，右侧再次套用参考标定造成偏移，不作为正式输出。
- `output/orbbec_openfield_audit/empty_annotation_rgb.jpg`：空标注图实际存在鼠。
- [Ultralytics pose 数据格式](https://docs.ultralytics.com/datasets/pose/)：支持 [K,2] 和 [K,3]。
- [OpenCV 相机标定](https://docs.opencv.org/4.10.0/dc/dbb/tutorial_py_calibration.html)：内参和畸变的含义。
