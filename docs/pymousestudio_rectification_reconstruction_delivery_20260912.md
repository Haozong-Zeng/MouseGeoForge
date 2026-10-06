# PyMouseStudio 矫正、视频重建检查与全视角标注包

2026-09-12。按用户明确指定的“视频尚未矫正”假设执行；未改原始数据。

## 已完成

新目录 `dataset/PyMouseStudio/`，不是 `BlenderPyMouse`：

- 12 段的全部 78 路 RGB 视频矫正，原尺寸1080×1200、52fps、1560帧完整保留。
- 66 路原有侧视 mask 同步矫正，二值FFV1；66份DLC source31关键点同步变换，仍标记预测。
- 每相机使用自己的K/D，输出同K、D=0、同R/t；保存源标定、有效像素图及逐相机完成哈希。
- RGB视频为H264 CRF16有损编码；待标PNG、重建输入PNG从原始视频解码后直接remap，
  不经过这次视频重编码，逐张检查保存前后像素一致。不能把RGB视频称作无损。
- 原始输入是否已矫正不是本次证实的事实；保留明确假设和可重新生成的来源。

## 3D试验与视觉结论

复用 DANNCE 最终的标定多视角前景 visual-hull 路线，**本轮没有训练新的4D-GS**。
仅在开发序列 AZL3 的780–811帧（32帧，约0.615秒）做连续试验，保留7视角。
三角化DLC只用于搜索范围，不生成训练姿势标签。初次预检因高置信有效点不足停止；
改为0.8阈值+至少3视角几何共识，仅确定有40mm余量的包围范围，逐帧不强行补全31点。

两版结果：

1. `sessions/AZL3/reconstruction_pilot_v2/`：6/6侧视mask严格交集。
2. `sessions/AZL3/reconstruction_bottom_refined_v1/`：底视SAM2约束，并容许6路侧视中1路不一致。
   另保留首/中/末帧的严格7/7对照。

近距离相机采用1mm格点；输出每版224张深度，共448张主深度PFM（另外有对照PFM）。
均为1080×1200、float32相机Z、毫米、PFM上下翻转规则与当前DANNCE相同。
仅鼠前景有效，背景/失败处为0，不能当完整场景GT。mask是伪标签，不是人工GT。

|诊断量（32帧均值）|侧视严格交集|加入底视并放宽一路侧视|
|---|---:|---:|
|底视轮廓与SAM2 mask一致性IoU|0.686|0.861|
|侧视轮廓与原mask一致性IoU|0.870|0.816|

这些是**参与重建的伪mask一致性，不是独立精度**。底视SAM2的提示又来自粗几何，
不能把提升解释成真实深度误差下降。

人工视觉检查了780、796、811及占据体积跳变最大的808、809帧的全部视角，
以及第780帧的3D体素点云旋转视图：

- 底视加入约束后腹部外扩减少；放宽侧视投票使部分尾巴更连续。
- 部分侧视体积偏厚；末帧尾端、部分足掌仍缺失，底视SAM2自己也漏爪。
- 点云有体素台阶和尾端零碎片段，不是高精度表面网格。
- 808→809帧占据体积变化约15.1%，有不应忽略的时序不稳定。不能称作已经稳定的重建。
- 搜索体积边界命中率0，当前问题不是简单的包围盒截断。

因此只批准几何检查，**未生成12段全量深度、未开启伪深度训练**。
下一步优先人工复核底部/足部mask或关键点，再试连续视频mask跟踪和跨视角置信约束；
不要只平滑深度或增加4D-GS训练步数来掩盖输入几何不一致。

可检查的文件位于第二版 `review/`：

- `multiview_depth_review_h264.mp4`：7路RGB、重投轮廓和深度，8fps慢放，源视频52fps。
- `bottom_comparison_000780.png`、`000796.png`、`000811.png`：底视多策略对照。
- `orbit_voxel_cloud_frame000780.mp4`：实际1mm体素云表面旋转，不增加或生成解剖细节。
- `temporal_diagnostics.png`：短段一致性与占据体积曲线。
- 首/中/末帧 `hull_voxels_mm.ply`；坐标仍为原标定毫米坐标，不擅自ground-align。

## 给标注人员的正式新包

`dataset/PyMouseStudio/PyMouseStudio-Rectified-AllViews-300Frames-Annotation-v2.zip`

- 1,293,474,148 bytes，约1.29GB；ZIP CRC通过。
- SHA256：`8faf76eb2e7ac733fbb67bb800eec9b98bce5f57af78a5e2c9b793be306d6880`。
- **300个时间组 × 每组全部可用视角，共1950张图片**；底视300张，侧视1650张。
- 按原split保留：开发225组/1475张，holdout75组/475张；均不自动进入训练。
- 6个开发试标组共42张。保留缺失相机清单，不把缺视频的标定路伪造成RGB。
- 7组实际 Blender PyMouseStudio 合成GT定义参考，中文规则、31→23差异表、
  全部编号与空模板、相机参数、有效像素图、来源/帧号/时间与哈希。
- 不把任何DLC/SAM2/重建预测叠到待标图片上；包中不包含伪深度或mask预标注。
- 旧Cylinder的肘腕命名、腹点重复、耳尖/骨起点和后爪/踝定义冲突仍保留在说明中，
  不强迫标注员猜测23个点。先试标再核定有歧义的语义。

新标注使用矫正后像素，**不要与旧288张原图包混用坐标**。旧包未删除。

## 验证入口

- `delivery_verification.json`：最终原始/结果哈希、关键点变换、图像及包检查。
- `annotation_delivery.json`：标注包收据。
- `sessions/AZL3/reconstruction_review.json`：主深度校验、量化一致性与使用限制。
- 8个针对性unittest通过：镜头点变换、鲁棒三角化、毫米PFM方向、原始数据安全读取等。
  本轮不改变训练环境依赖，不运行训练。

实现脚本在 `tools/`：`export_pymousestudio_rectified.py`、
`reconstruct_pymousestudio_pilot.py`、`refine_pymousestudio_bottom_pilot.py`、
`finalize_pymousestudio_reconstruction_review.py`、`preview_pymousestudio_hull_orbit.py`、
`package_pymousestudio_rectified_annotations.py`、`verify_pymousestudio_rectified_delivery.py`。
