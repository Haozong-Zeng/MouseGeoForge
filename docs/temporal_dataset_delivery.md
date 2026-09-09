# 分段时序与 DANNCE 完整序列

2026-09-09 更新：按用户要求，DANNCE 完整时序现已降采样到 **25 fps、每只 4,500 帧**，保留源帧号 0/4/.../17996。下方 100 fps、216,000 张 RGB 和 200 帧 pilot 的统计是历史审计结果，不是当前存储规模。当前布局、全量 mask／伪深度生成和读取方式见 [DANNCE 25 fps 说明](dannce_25fps_full_pseudo_depth.md)。原始视频和全部稀疏人工标注仍保留。

实施与审计已完成（2026-09-08）：CranialWindow 三目录、13 套 Blender 时序索引、源动作分组、两套 DANNCE 完整 RGB 序列及每只 200 帧伪深度验证段均已落盘。216,000 张 JPEG 全部通过尺寸/格式/文件尾与独立存储检查；172 个原人工标注时刻的新旧 RGB 解码像素完全相同，回投误差最大约 0.0000653 px。原 Blender 读取器检查了 39 个代表场景；新 DANNCE 读取器通过实际时间/视角/实例维度、无标注帧和可选深度有效性检查。

## CranialWindow 的实际拆分

原 `dataset/BlenderPyMouse/Openfield-BlackFur-CranialWindow` 已拆为三个独立目录：

| 新目录尾部 | 原始帧号（闭区间） | 帧数 | 源动作分组 |
|---|---|---:|---|
| CranialWindow-ClimbWall | 1–91 | 91 | mouse_climb_wall_91 |
| CranialWindow-WashFace | 92–182 | 91 | mouse_wash_face_91 |
| CranialWindow-WalkAround | 183–273 | 91 | mouse_walk_around_91 |

每帧的 RGB、相机、关键点、mask、深度整体移动，没有重渲染或改姿态。原帧号保留，索引 `local_index` 从 0 开始。仅原目录级元数据保存在 `output/dataset_reorganization/`，路径迁移见该目录的 `path_migrations.json`。每段与相同动作的 BrownFur 数据共享分组，不能跨 train/val/test。

## 分段与源动作分组规则

- 总目录索引：`dataset/temporal_catalog.json`。
- 源动作分组：`dataset/motion_source_groups.json`。其中 split 是推荐划分；现有 Orbbec/SBeA/s-DANNCE 划分保留。
- 13 个确认可靠的 Blender 序列各有 `temporal_index.jsonl`：三段 CranialWindow、三个 BrownFur 动作、七个 WaterReaching FrameToFrame。
- WhiteFur FGcamB 为原始帧号 1,4,7,…,391，共 131 帧，`source_frame_step=3`。其他确认的 Blender 段步长为 1。
- 未核实的 Blender 数据为 `unknown`；DiversePose400 为 `pose_set`，不参与连续窗口采样。不能因为帧号连续就视为自然时序。
- Blender 与 Orbbec 暂不臆造 fps；`frame_order_only` 可用于顺序/跟踪，不可直接算 mm/s。
- SBeA 与 s-DANNCE 保留各自 NPZ/原视频读取方式；RGB 是否可用、3D 是否已对齐标定坐标是独立标记。s-DANNCE 的 pose-only 数据不因此获得 RGB 或相机参数。

`vggt-omega/training/temporal_window_index.py` 是可执行的采样索引：检查源帧差、segment_id 和每一条 intervening transition；即使采样跳帧，也不允许跨越缺帧或动作切换。源动作分组冲突直接报错。旧训练入口的尾部切验证集逻辑**不会自动变成分组划分**；应使用此索引给出的 split/场景列表，不能继续随机或按末尾帧切分后声称动作独立。

示例（在项目根，`PYTHONPATH=vggt-omega`）：

```python
from training.temporal_window_index import TemporalWindowIndex
windows = TemporalWindowIndex('dataset', split='train',
    window_size=16, frame_step=2, window_stride=16, require_rgb=True)
window = windows[0]
# window['rows'] 为场景/RGB 引用；NPZ 序列使用 source_positions。
# frame_delta 保存真实源帧差；T 和 V 不能拼成同一维。
```

## DANNCE 的完整导出布局

两只鼠的完整序列分别在原数据集的
`sessions/markerless_mouse_1/`、`sessions/markerless_mouse_2/`，不放入 BlenderPyMouse。

```text
DANNCE-MarkerlessMouse1-Core23-RealMultiview/
  frame_000001/...                  # 原先的稀疏人工标注场景，不改变
  sessions/markerless_mouse_1/
    metadata.json                  # export_complete 必须为 true 才可完整读取
    frames.jsonl                   # 真实源帧号、同步 sample ID、时间、RGB/人工标签引用
    sync_com.npz                   # 全时序 COM，非关节点 GT
    label3d_dannce.mat              # 独立复制的原始标定与标注
    cams/00000000_cam.txt ...       # 六个固定 K/E，另有校正有效区域
    rgb/00000000/000000.jpg ...     # 每视角 18,000 个独立 JPEG，无软/硬链接
    export_chunks/...              # 断点续传完成标记，不是训练标注
    pseudo_depth_pilot/<短段>/...   # 仅本轮验证片段的 mask/PFM/confidence
```

每只鼠 18,000 个时刻 × 6 视角，共 108,000 张；两只共 216,000 张。原分辨率 1152×1024，视频标称 100 fps、180 秒；时间由帧号/fps 计算，并非独立硬件时间戳。所有 RGB 经过与原稀疏转换一致的 Brown–Conrady（含原 K 的 skew）去畸变，使用 Lanczos4，JPEG quality 95；校正后的 K/E 与像素对应。

全部 36,000 个时刻中，只有 81+91=172 个有原始人工骨架标注（mouse22 最多映射 Core23 的 16 点，具体缺失点仍保留 valid=false）。其余标记为未标注，**不是空目标**。`sync.data_2d/data_3d` 的零数组是占位，完全不当成 GT。完整 COM 是另一种定位信息，不替代关节。

新的 `DANNCECompleteTemporalDataset` 输出 `[T,V,3,H,W]`，静态 K/E 为 `[V,...]`，实例维可填充为 2；人工 2D/3D、可见性和深度分别有有效性掩码。坐标保持原标定坐标，毫米；不做未经验证的 ground alignment。

```python
from training.dannce_temporal_dataset import DANNCECompleteTemporalDataset
ds = DANNCECompleteTemporalDataset('dataset', split='train',
    window_size=16, frame_step=2, image_size=(576,512), max_instances=2,
    load_pilot_pseudo_depth=False)
sample = ds[0]
```

Mouse 1 推荐 train，Mouse 2 推荐 test；本次不人为从相邻帧切出独立验证集。新时序 loader 可直接使用，但没有启动训练或改动旧模型的时序网络。旧 `BlendedMVSDataset` 仍读原来的 sparse frame 场景；不能把稀疏场景序号误作实际相邻帧。

时序训练接入时，必须用 `pose_supervision_valid` 筛选参与实例/关键点监督的帧，再送入旧空间损失；不能直接把整个 T 维展平并将未标注帧当成负样本。此轮交付的是时序数据读取与有效性约定，不是已接入旧训练入口的完整时序训练器。

## 连续短段伪深度验证

- Mouse 1：源帧 0–199；只用帧 27 初始化分割，72/168 保留检查。
- Mouse 2：源帧 800–999；只用帧 833 初始化，893/955 保留检查。
- SAM2.1 视频传播，前后传播覆盖完整短段；不是逐帧都使用人工标签。
- Mouse 2 初始框 18 px 留边会截掉蹲伏时背毛；对比 18/64/96/128 px 后采用 96 px。64 px 在 view 3 仍少一小块背毛，96 px 已完整覆盖，未使用额外人工标注。
- 对比 4/5/6 视角约束后改用 **6/6 silhouette visual hull**，固定世界格点，2 mm voxel。4/6 条件会产生较膨胀表面，虽覆盖率高，回投一致性却差。
- 保存原分辨率 float32 PFM，相机 Z、毫米、0 无效；`pfm_flip_y=true` 与历史 DANNCE PFM 相同，loader 显式翻转。
- confidence 为回投后落在六幅前景分割中的票数/6，并非标定概率。默认训练仅用至少 5/6 视角支持且深度有效的像素。
- 仅前景伪深度，无背景/墙壁深度；伪深度默认不启用。`load_pilot_pseudo_depth=True` 才加载，短段外仍为 unavailable。
- 检查包括留出的人工点轮廓包含率、光流补偿后的相邻 mask IoU、光流补偿深度变化、格点边界触碰和多视角支持。它们不能证明绝对表面深度等于真值，也不能把骨骼内部点深度当表面 GT。
- SAM2 的可选 CUDA 连通域扩展在本环境未编译，使用其允许的跳过路径及脚本 OpenCV 闭运算；这一点记录为运行限制。

预览放在 `output/dannce_temporal_depth_pilot_v1/`。视频按 10 fps 慢放（源数据仍为 100 fps）；采用每个视角固定的深度色标，避免逐帧归一化掩盖波动。最终统计以每个 mouse 的 `summary.json` 为准。

## 可重复执行与审计

```bash
.venv-vggt-omega-eval/bin/python tools/organize_temporal_datasets.py
.venv-vggt-omega-eval/bin/python tools/export_dannce_complete_temporal.py --workers 12
OPENBLAS_NUM_THREADS=1 TQDM_DISABLE=1 .venv-vggt-omega-eval/bin/python tools/validate_dannce_temporal_pseudo_depth.py --mouse 1
OPENBLAS_NUM_THREADS=1 TQDM_DISABLE=1 .venv-vggt-omega-eval/bin/python tools/validate_dannce_temporal_pseudo_depth.py --mouse 2 --box-padding 96
.venv-vggt-omega-eval/bin/python tools/organize_temporal_datasets.py
.venv-vggt-omega-eval/bin/python tools/finalize_dannce_temporal_pilots.py
.venv-vggt-omega-eval/bin/python tools/audit_temporal_delivery.py
```

完整 RGB 导出未完成时，不允许时序 loader 把整个 session 当成已就绪。最终验证报告为 `output/temporal_dataset_audit/delivery_validation.json`：检查所有 JPEG 的尺寸/格式/文件尾、独立存储、人工帧对应、分片边界解码、投影误差及真实 loader 有效性掩码。伪深度只验证短段，不表示已导出两只鼠的全部深度。

`finalize_dannce_temporal_pilots.py` 校验全部 2,400 组 PFM/mask/confidence，并将对应路径合入完整 `frames.jsonl`；只清除本轮生成的 SAM2 数字文件名临时 RGB 副本，原始视频和正式导出的 RGB 不受影响。
