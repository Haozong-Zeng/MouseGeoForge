# Homecage + Lightning Pose Core23 数据集

## PyMouseLifter 原则

这里沿用 PyMouseLifter 的核心分工：真实数据负责目标域、稀疏姿势和双鼠空间关系；合成数据负责解剖先验、完整 Core23、逐实例 mask、深度和相机几何。真实伪标签不被冒充为精确三维真值。

## 已生成数据

### LightningPose-CRIM13-Core23-Partial2D-v1

- 5,260 张真实顶视图；InD 3,986，OOD 1,274；
- 每张两只鼠，每只 7 个点映射到 Core23；
- InD 按 session 整体拆为 train 3,233 / val 753，OOD 1,274 全部作为 test；
- 7 点外接框只作为匹配辅助信息，`box_valid=false`；
- 不监督 mask、visibility、camera、depth 或 3D。

### Homecage-BlackFur-Core23-Full3D-v1

- 200 个姿势，每帧 4 个校准视角；
- 完整 Core23、可见性、mask 强框、毫米深度、内外参和毫米世界坐标；
- 连续 80/10/10 拆分为 train 160 / val 20 / test 20；
- 单一序列的 val/test 只用于回归检查，不代表跨动物泛化。

### Homecage-LightningPose-Core23-Hybrid-v1

- 合并上述 5,460 个 sample，图像均原地引用，不重复约 12 GB 源数据；
- 完整 train/val/test 以及 96/36/40 sample 的 pilot 清单；
- 提供 CRIM13 到 Core23 映射、推荐 source 权重和损失开关；
- 必须按 source 组成同质 batch，因为真实为 `V=1,M=2`，合成为 `V=4,M=1`。

### Homecage-CRIM13-PoseConditioned-Core23-MultiMouse-Pilot-v1

- 40 张实际新生成的 1280x720 双黑鼠 Homecage 图像；
- CRIM13 的双鼠七点布局分别检索最接近的 Blender Core23 姿势；
- RGB、mask 和 23 点使用同一个二维相似变换，并按合成遮挡顺序更新可见 mask 和关键点 visibility；
- train/val/test 为 24/8/8；当前平均七点条件残差约 19.4 px；
- 14 张存在实例 mask 接触/重叠，11 张产生关键点遮挡，共屏蔽 32 个被遮挡点；另有独立 `*_contact.jsonl`；
- 可监督 objectness、强框、逐实例 mask、完整 2D Core23 和 visibility；
- 不能监督 camera、depth、3D 或 metric geometry，因为二维合成不是 Blender 物理重渲染。

### Homecage-BlackBrownFur-LightningPose-CRIM13-DualMouse-Blender-Pilot

- 实际 Blender 双鼠工程，57 帧、8 fps、1280×720；
- 黑 C57 与棕色两套独立 mesh、armature、毛发和柔性尾；
- 使用会话 `021709_A25_Block22_BCfe1_t` 的八个稀疏双鼠标签约束位置、朝向、接触关系和尾根；
- 内部身体姿势暂用经过侧视筛选的稳定帧 15/98，不声称逐关节重建 CRIM13；
- 机器审计中两鼠代理中心最小距离约 39.96 mm，最大条件点误差约 39.28 mm，两条尾巴 57/57 帧通过；
- 详细设计、限制与入口见 `docs/lightningpose_two_color_dual_mouse_blender_pilot.md`。

## 代码入口

```bash
python scripts/build_homecage_lightningpose_core23_datasets.py
python scripts/build_crim13_conditioned_homecage_composites.py
python scripts/validate_core23_instance_manifests.py
```

训练读取器位于 `vggt-omega/training/core23_instance_manifest_dataset.py`。当前建议先用 `batch_size=1`，或实现按 `(views, instances, source)` 分桶的 batch sampler。

## 下一版生产数据

二维合成适合启动检测和实例装配，但不能代替真实多鼠三维数据。下一版应在 Blender 中放置两个独立 armature，以 CRIM13 七点布局约束头、躯干、尾根和后膝，再由碰撞、地面和关节限位求解完整姿态，最后重新渲染 RGB、每实例 mask、depth、Core23 2D/3D 和相机参数。该版本才可加入多鼠三维损失。

尾巴采用 `docs/flexible_tail_generation_collision_v1.md` 的解耦策略：真实图像只约束 `tail_base`，尾根方向为弱先验，远端尾形由时序平滑曲率采样并通过地面、身体、另一只鼠和笼壁穿模矫正。不要从 CRIM13 七点推断不存在的远端尾真值。
