# DANNCE：25 fps 全时序 RGB、mask 和伪深度

用户要求从原始 100 fps 改为每 4 帧取 1 帧。两只小鼠各保留 **4,500 个同步时刻 × 6 视角**，合计 54,000 张 RGB；不是补帧或插值。原始 100 fps 视频、标定 MAT、全部 81+91 个稀疏人工标注场景均保留。

## 存储与时间坐标

数据仍在 `dataset/DANNCE-MarkerlessMouse{1,2}-Core23-RealMultiview/sessions/markerless_mouse_{1,2}/`。

- `fps=25`，`source_fps=100`，`source_frame_step=4`。
- `local_index=0..4499`，源帧号和文件名仍为 `000000, 000004, ... 017996`。
- `time_seconds=source_frame_id/100=local_index/25`。保留原时长，不把降采样后的帧序号除以 100。
- `sync_com.npz` 同步缩减到 4,500 条；原标定 MAT 保留所有原始 COM 和人工标注。
- 未落在 25 fps 网格上的人工标注帧只保留在原来的稀疏目录，不伪造邻帧关键点标签。
- 图像仍为 1152×1024、已去畸变、独立 JPEG，未再次编码保留帧。

RGB 现共约 **12.39 GB**，约为原来的四分之一。float32 全时序 PFM 预计约 **254.81 GB**，而非原 100 fps 的 1.02 TB；PNG mask/confidence 额外占少量空间。由于原始视频和稀疏人工标注完整保留，所有文件加总不会严格等于旧目录的四分之一。

本轮已删除约 **45.8 GB** 可重新生成的转换数据，包括未选中的 RGB、旧 pilot 未选中的深度/mask/confidence，以及中断的 100 fps 新任务中间文件。精确清单、校验和原索引在 `output/dannce_25fps_migration/`。原视频没有删除，可用于重新导出。

## 全量 mask / 深度流程

生成结果位于各 session 的：

```text
pseudo_depth_full_v1/
  config.json
  metadata.json
  chunks/f000000-000396/
    masks/00000000/000000.png ...
    rendered_depth_maps/00000000/000000.pfm ...
    confidence/00000000/000000.png ...
    frames.jsonl
    frame_metrics/000000.json ...
    chunk_complete.json
```

每段跨越 400 个源帧，实际处理 100 个 25 fps 时刻。两只鼠交替处理，按分段断点续跑。

SAM2.1 small 在 CUDA 上跟踪，使用段内真实人工标注作提示，并继承前一段末帧的 mask。若提示帧不在 25 fps 采样网格，从保留的稀疏人工场景读入 RGB，**仅作为临时分割提示**；不把它变成额外输出帧或伪造姿态。临时图像副本在每个视角完成后清理。

深度使用此前验证的 **6/6 多视角 silhouette visual hull、2 mm 世界格点、相同栅格化与前景平滑规则**。COM 和人工点只确定足够大的搜索范围，不驱动或插值身体姿态。投影索引缓存，六视角投票在 GPU 计算；已通过真实 pilot 与原 NumPy 算法逐点完全一致的测试。

输出仍是相机 Z、毫米、float32 PFM；`pfm_flip_y=true`，与当前 DANNCE/BlenderPyMouse 的 PFM 读取约定一致。mask 是 0/255 PNG。confidence 表示回投 silhouette 票数（0..255），不是校准后的概率。

**“全量”指覆盖全部选定时刻，不是有真值的整幅场景深度。** 仅生成鼠前景伪深度，背景、无有效重建处为 0。六视角重建不能恢复真实凹陷或被遮挡表面，不能称作深度 GT。默认不启用伪标签训练，建议至少 5/6 confidence 门限并检查质量标记。

每一帧写完后检查形状、非负有限数、背景为 0、confidence 对齐、PNG 二值性；完成整段后才写 `chunk_complete.json` 并将引用并入主 `frames.jsonl`。异常或边界截断的帧单独记录、关闭训练有效性；一段严重异常超过 5%（最少容忍 2 帧）则停止，不继续发布失效标签。每段导出首/中/末帧及异常帧的六视角可视化到 `output/dannce_full_pseudo_depth/markerless_mouse_*/review/`。

## Dataloader

```python
from training.dannce_temporal_dataset import DANNCECompleteTemporalDataset

ds = DANNCECompleteTemporalDataset(
    'dataset', split='train', window_size=16, frame_step=1,
    image_size=(576, 512), load_pseudo_depth=True,
    confidence_min_votes=5,
)
```

`frame_step=1` 表示相邻存储帧，源帧差为 4、时间差 0.04 秒。保留 `[T,V,...]` 维度，不把时间与视角合并。

新增 `body_masks`、`mask_available`、`mask_is_pseudo_label`、`mask_supervision_valid`；原来的 `depth_available/depth_masks/depth_confidence` 继续有效。旧参数 `load_pilot_pseudo_depth=True` 兼容。未标注帧的 `pose_supervision_valid` 和关键点有效性仍为 false；不能因为获得 mask 就将伪姿态或零数组当成 GT。

## 启动与状态

WSL 项目根目录：

```bash
.venv-vggt-omega-eval/bin/python tools/launch_dannce_full_pseudo_depth.py
.venv-vggt-omega-eval/bin/python tools/dannce_full_pseudo_depth_progress.py
```

后台日志为 `output/dannce_full_pseudo_depth/queue.log`，状态为 `status.json`。全部完成后自动执行 `validate_dannce_25fps.py --require-full-depth`，只有验证通过后全局状态才设为 complete。电脑需要保持唤醒。

`export_dannce_complete_temporal.py` 默认及允许的导出步长已改为 4，避免以后重复运行时意外重新生成已清理的 100 fps RGB。
