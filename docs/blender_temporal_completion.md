# Blender 固定步长采样补齐

本次将原始 Blender 动画中被渲染步长跳过的帧补齐。不生成新的插值动画，不改动骨骼、毛色、相机、源 `.blend`，不把 Sample10/DiversePose 的选姿态集合自动改成连续视频。

**2026-09-09 更新：用户确认 PseudoSynthetic 不是可靠时序。四套基础版本及四套 Transfered 版本已从活动队列移除，取消补帧和后续风格迁移；已有数据不删除。当前仅补齐 WhiteFur FGcamB。**

## 范围和编号

| 数据集 | 原帧数 / 步长 | 完成后的编号 | 新增时刻 |
| --- | --- | --- | --- |
| WhiteFur WaterReaching FrameToFrame FGcamB | 131 / 3 | `frame_000001`–`frame_000392` | 261 |
| E / FB / FJ / FS 的 4 套 PseudoSynthetic | 各 102 / 4 | 保留原有稀疏编号，已取消补帧 | 0 |
| 上述 4 套 PseudoSyntheticTransfered | 各 102 / 4 | 保留原有稀疏编号，已取消补帧 | 0 |

活动计划仅新增 261 个 WhiteFur 时刻 / 1,305 个视角，完成后 WhiteFur 为 392 帧。取消的 8 套数据仍各 102 帧。旧计划保存在 `output/temporal_completion/plan_before_pseudosynthetic_exclusion.json`，仅用于审计，不得据此恢复 PseudoSynthetic 补帧。

WhiteFur 工程 UI 的结束帧是 250，但实际骨骼 Action 为 1–392，且旧数据的 1/196/391 帧骨架与该工程完全匹配。因此包含旧 stride-3 导出漏掉的最后一帧 392。

只调整上述补渲染数据的连续序号；原来已渲染的编号不改，其值就是 Blender 源帧号。CranialWindow 三个独立动作文件夹本轮暂保留原始源编号。索引同时记录 `local_index` 和 `source_frame_id`，不可把存储位置当作源视频帧号。

## 兼容性和质量约定

- 保持各自 1200×800、Cycles 64 samples、GPU OptiX、JPEG quality 95。
- 复用原项目及原相机顺序；旧深度忽略规则、单位和 PFM 方向不变。
- 保持原生 29 个手部/腕部/头部关键点，不伪造缺失的 Core23 解剖标记；由已有多实例 loader 做合法映射。
- 每个新增帧有 RGB、深度、标定、逐视角 JSON、`keypoints.npz/json` 和二值 `body_masks`。原来没有 body mask 的旧帧也只新增 mask，不覆盖原来的 RGB/深度/标注。
- `_masked.jpg` 沿用旧版的 RGB 占位副本约定，真实前景监督来自 `body_masks`。
- 原风格迁移配方的参考测试保留作审计；当前队列不再执行任何 PseudoSynthetic 风格迁移。
- 原帧文件 SHA256 事先存入 `output/temporal_completion/original/`，完成后全部复核。
- 新增帧先在 `output/temporal_completion/staging/` 渲染并验证，再逐文件夹原子迁移。半帧不进入正式数据集；每批通过多实例 loader 的代表帧读取检查。
- 队列运行时 `temporal_completion.export_complete=false`；新的连续索引只在完成后开放。完成标志以 `status.json` 和验证报告为准，不能把存在脚本/启动进程视作渲染已经完成。

已做同帧参考测试：WhiteFur 五视角深度有效区域 IoU 全部 > 0.99996，RGB 平均绝对误差为 0.64–1.08 / 255；原风格迁移单图复现误差约 0.25 / 255。均做了视觉对照。这不是新动画每帧的视觉验收结果。

首个新增帧 `WhiteFur/.../frame_000002` 已通过五视角文件/尺寸/深度/mask 校验和真实多实例 loader 读取；4,454 个旧帧文件 SHA256 均保持不变。PseudoSynthetic E trial 的三个同帧参考视角也通过比较：RGB MAE 0.36–0.73 / 255，深度有效区域 IoU 均为 1.0。测试报告位于 `output/temporal_completion/`。

## 时序和分组

WhiteFur 完成后步长改为 1，索引保留源帧差、分段和固定相机检查，仍归入 `water_reaching_fg_cam_b`。PseudoSynthetic 及 Transfered 永久按 `temporal_mode=pose_set`、`temporal_eligible=false` 处理：即使 Blender 内有逐帧动画、或未来帧号连续，也不能据此判定为可靠时序。两种外观仍共用 `anipose_trial/<trial>`，避免独立姿态训练时的风格副本泄漏；此分组不代表时序可信。

原动画时钟为 WhiteFur 24 fps、PseudoSynthetic 30 fps，仅记录为 `source_animation_fps`；不据此宣称已经验证原视频采集时钟。

## 运行与查看进度（WSL）

从 MouseGeoForge 根目录执行：

```bash
# 后台启动；存在活动队列时拒绝重复启动。
.venv-vggt-omega-eval/bin/python tools/launch_temporal_completion.py

# 只读进度：已发布帧、目标帧数、当前阶段及最新日志。
.venv-vggt-omega-eval/bin/python tools/temporal_completion_progress.py

# 若失败，修复原因后重复启动；完成的原始/新增帧不会重渲染。
# 前台执行同一队列也可以：
.venv-vggt-omega-eval/bin/python tools/complete_subsampled_blender_datasets.py --chunk-size 8
```

队列和逐批日志在 `output/temporal_completion/queue.log`、`logs/`。`plan.json` 是冻结的补帧清单，不应在续跑时删除或重新生成。

最后自动重建时序索引、检查连续窗口并进行多实例 dataloader 审计；不执行已取消的风格迁移。电脑需保持唤醒，WSL 和后台 Blender 进程不能关闭；可切换编辑其他项目，但不宜并行启动占满同一 GPU 的训练/渲染，也不要在队列运行中改名/移动 WhiteFur 源项目或数据目录。
