# BlenderPyMouse：06 / 07 / 23 体表锚点迁移

用户已批准 synthetic demo 的体表定位，本次将定义落实到合成训练标签。版本为 `pymousestudio_manual_surface_core23_v3`。是否完成写回以 `output/surface_keypoint_migration_v3/manifest.json` 的 `status=committed` 和 `validation_live.json` 为准。

## 范围与定义

更新 9 个 Core23 数据集，共 1,346 帧、1,746 个实例帧。另 15 个手指/前肢 29 点数据集不含这些部位，未改动。未修改真实数据、源 Blender 动画、RGB、深度、mask、相机、帧号或时序索引，未启动训练。

| 显示编号 / 数组索引 | 保留的模型通道名 | 新的 `semantic_names` 定义 |
| --- | --- | --- |
| 06 / 5 | spine_center | dorsal_thorax_surface：背中体表 |
| 07 / 6 | lumbar_spine | dorsal_lumbar_surface：腰背体表 |
| 23 / 22 | abdomen_center | ventral_abdomen_surface：腹面体表 |

`names` 保持现有通道顺序和字符串，以免已有模型名称匹配丢失监督；这只是槽位兼容，**不代表旧骨骼坐标与新体表坐标语义等价**。权威解剖语义见新增 `semantic_names` 和 `definition_version`。和真实数据混训前，必须确认这三个点也是体表定义，否则应屏蔽旧定义点或完成迁移，不能仅凭同名混用。

三个点使用已批准 REST 网格三角形的固定重心权重，经动画变形后求坐标。不同项目逐一验证全部 9,984 顶点的拓扑及静止坐标仿射对应，不按每帧最高/最低点选点，不使用相机相关选点。三角形和权重保存在各数据集根目录 `surface_keypoint_definition.json`。其余 20 个点的 NPZ 数值和 JSON 点记录逐项校验不变。

历史字段 `joint_world_xyz` 在这三个槽位现在也存体表锚点，等于对应 `surface_world_xyz`，不再是骨骼头。`bones` 留作参考骨骼；`endpoint=mesh_surface`、`coordinate_kind` 明示新定义。16 个 support 顶点用于局部可见性检查，**不能对它们求平均来还原精确锚点**。

## 同步内容和可见性

每帧同步 `keypoints.npz`、`keypoints.json`、`scene_meta.json`、全部 `annotations/*.json`、`keypoint_validity.npz` 和 `keypoint_occlusion.npz`。遮挡 sidecar 的 annotation SHA256 已重算；其余 20 点的遮挡原因保持原值。旧 support 遮挡比例不适用于新中心点，更新槽位的旧比例字段置为 NaN，不作为训练标签。

可见性用裸鼠网格到中心锚点的视线检测，包含另一只鼠的遮挡，并检查现有 raster mask 的一致性。它不是毛发逐丝可见性，也不是要求人工标注者隔着遮挡猜坐标。无法验证时 `visibility_valid=false`、`occlusion_cause=-1`，而非强行判成自遮挡。旧支持区域多数投票不再定义这三个点的中心可见性。

总计 528 个新点的视角可见性标签保守置为未知：双鼠 408、黑鼠 Homecage 48、棕鼠 MoSeq 72。主要来自源回放在其他局部存在偏差，因此不能断言整只鼠的遮挡关系。它们不等于 528 个世界坐标无效。

## 源姿态不一致的例外

双鼠 DiversePose400 的 4 个实例帧未通过新锚点局部源几何复核：

| 帧 | 受影响实例 |
| --- | --- |
| 62 | 1：棕鼠 |
| 66 | 0：黑鼠 |
| 160 | 1：棕鼠 |
| 348 | 1：棕鼠 |

这些实例的 06 / 07 / 23 共 12 个点只保留有限值候选以便排查；`world_valid`、`surface_valid`、`xy_label_valid` 均为 false，JSON `candidate_only=true`。**不能绕过 validity 掩码把候选坐标当真值。**其余实例和其余 20 点不受此次屏蔽影响。整个迁移有 5,238 个待更新实例点，其中 5,226 个通过局部几何检查。

双鼠原始导出在实例循环外和每个实例内部都调用 `frame_set`；复核需要匹配这一求值顺序。修正求值顺序解决了一部分假差异，余下 4 个实例未放宽阈值硬通过。全网格回放不一致帧的可见性按未知处理。

## 验证与文件

- 全量 1,346 帧；验证 72,513 个 JSON 新点的重复表示与 NPZ 一致，未改的 20 点逐项相等。
- 从 NPZ 世界坐标及原 K/E 独立重投影，最大差约 `0.000078` 像素。
- 06 与 23 的三维距离最小约 17.04 原生世界单位，已不再重合；没有未经校验把单位改称毫米。
- 9 个数据集以及全部 4 个受影响帧，共 33 组真实 dataloader 检查，覆盖 1 / 2 / 全部视角，读取尺寸 768×432，验证坐标有限、world/xy/visibility 掩码一致。
- 当前环境无 pytest，直接以独立临时目录调用现有多实例 loader 测试模块的 5 个测试函数，全部通过。
- `output/surface_keypoint_migration_v3/surface_keypoint_qa.jpg`：走路/洗脸/攀墙的侧视、底视、俯视以及双鼠多视角叠加。实心代表已判可见，空心仅表示未判可见；具体“遮挡/未知”以标签状态为准。

可恢复备份：`backup/keypoint_migrations/BlenderPyMouse-before-surface-core23-v3.zip`。包含全部被替换文件的原件、原/新 SHA256、原来不存在的新增路径清单。已校验 ZIP CRC。恢复时应按清单核对当前版本，恢复原有文件并仅移除清单里原先不存在的新增文件，不能笼统删除数据目录。

各数据集原 `*support*.pkl` / `*support*.audit.json` 是历史 v1 导出资料，保留不改；对新三个槽位不再是权威定义。不要直接用旧 exporter 覆盖新标签。重新导出时先生成隔离的 v1 标签，再用以下迁移流程和新的 `surface_keypoint_definition.json` 生成 v3；后续可把固定体表定义合并进主 exporter。

## 复现脚本

1. `tools/inventory_surface_keypoint_migration.py`：库存与源项目核对。
2. `BlenderScripts/stage_surface_keypoint_migration.py`：Blender 后台只读源项目，先 WalkAround 建立 REST 参考，再处理其余项目。输出 staged 几何，不保存 Blender。
3. `tools/migrate_blenderpymouse_surface_keypoints.py prepare`：生成完整候选标签，校验原文件哈希和未改槽位。
4. `tools/validate_surface_keypoint_migration.py`：候选全量检查和实际 loader 检查；`--visualize` 输出叠加图。
5. `tools/migrate_blenderpymouse_surface_keypoints.py commit`：先校验、备份，再原子逐文件写回；拒绝覆盖已有备份。
6. `tools/validate_surface_keypoint_migration.py --live`：使用备份作为旧标签基准，检查实际落盘数据。

已迁移的数据会被再次 prepare/stage 拒绝；不要删除版本标记后重复迁移。
