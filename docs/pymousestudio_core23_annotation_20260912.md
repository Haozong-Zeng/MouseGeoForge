# PyMouseStudioRaw 人工标注包与 Core23 语义核对

2026-09-12。目的：补齐真实 Cylinder 底视人工标签，并提供相同时间索引的侧视图。
本轮不训练、不运行模型预测、不修改旧 synthetic 标签或原视频。

## 先说明：31 点不是另一种顺序的 Core23

依据实际数据和代码，而非仅按名称猜测：

- `dataset/PyMouseStudioRaw/FR2/config.yaml` 的 `dlc_parts_name` 与 H5 列按位置对应。
- `BlenderScripts/convert_cylinder_renders_to_blenderpymouse_core23.py` 的 `CORE23_SPECS`
  和 `evaluated_core23()`；实际 `keypoints.json` 的 `bone/endpoint/joint_world_xyz/uv`。
- 现有侧视 H5 是 DLC 预测，不是人工真值；12 段 camA 都没有 H5 或 mask。
- 没有找到原 DLC 的人工逐点操作手册。因此下面区分“代码事实”和“待解剖语义确认”，
  不把骨骼名自动认定为真实动物的可见皮肤标志。

|部位|侧视 source31 / 随包模型映射|当前 Cylinder Core23|结论|
|---|---|---|---|
|鼻|nose → Nose_M|nose → HeadEnd_M.head|候选同部位；真实图标鼻尖可见中心|
|耳|ear_end_l/r → EarEnd_L/R|left/right_ear → Ear01_L/R.head|耳尖与骨骼起点不是可直接互换的定义；不可直接复制|
|眼|eye_l/r|无独立眼点|保留额外观察点，不用眼中点冒充 head_center|
|前爪|fore_paw_digit_2b 为 NotUsed；fore_paw → ForePaw|Fingers1_L/R.head|掌/指根部中心候选；不是指尖，也不保证等于原 fore_paw 坐标|
|后爪|hind_paw_digit_2b → HindPaw；hind_paw → Ankle|Toes1_L/R.head|先前“hind_paw 直接映射 paw_center”的候选应撤回；踝不能当足掌|
|前肢关节|ulna → Ulna；humerus → NotUsed|名为 elbow，但引用 Wrist_L/R.head|现有实现命名有歧义，暂不硬映射，不让标注员在肘/腕之间猜|
|后肢关节|tibia → NotUsed；另有 Ankle|Knee_L/R.head|膝、胫、踝不同，不能凭名称相似合并|
|躯干|Spine0–4|Head1/Neck3/Spine3/Spine1.head|节段位置和内外表面定义不同，无已证实的一对一映射|
|腹部|无|abdomen_center 与 spine_center 都取 Spine3_M.head|在本批 Cylinder 数据坐标完全重合；ventral_patch 只改变表面支持/可见性，不改变 uv。不能要求人工另标腹部皮肤中心后声称与旧标签等价|
|尾|tail_root, tail_0–4, tail_end|Tail0,2,4,6,8,10,11.head|端点是候选；中间点是指定骨关节，不是已证明的等弧长分位点，不能从 q 名称猜比例|

此结论针对本包采用的 WalkAround / WashFace / ClimbWall Cylinder 导出器，
不能未经核对推广到所有其他 BlenderPyMouse 导出器。

## 标注员操作说明

先阅读 `synthetic_examples/` 的原图与编号投影示例。**这些是实际合成骨骼 GT 投影示例，
不是人工可见点范本**；不代表隐藏的骨骼中心在真实图中能够精确观察。
空心标记表示导出器的表面支持不可见，并非人工可见性判定。

### 文件与坐标

- `real_images/real_development/`：216 张；`real_images/real_locked_holdout/`：72 张。
  合计 288 张 = 12 段 × 8 个固定均匀时间索引 × A/D/E 三路；底视 96 张，侧视 192 张。
  是 96 个多视图时间组，不是 288 个独立姿态。没有按模型表现挑图，不保证动作最大多样性。
- PNG 是视频解码像素的无损保存，原尺寸 **宽 1080 × 高 1200**，没有缩放、裁剪、翻转、
  旋转或新做畸变矫正。PNG 无损不意味着源 MP4 无压缩损失。
- 原视频是否已矫正尚未确认；标注当前图片像素即可。将来若确认需要矫正，图像和点一起转换。
- 原点左上，x 向右、y 向下，0 基连续像素坐标。不要标到周围玻璃中的倒影上。
- 同一个 session 与 frame 的 A/D/E 可相互参考，但尚未验证硬件同步，不把别的视图投影点
  当这张图的真值。文件名 frame 为视频 **0 基解码序号**，时间 = frame / 52 秒。
- 标签索引 `id` 为 0–22；图中展示编号是 **1–23**。左右永远以小鼠自身为准，
  底视不能照搬屏幕左右；如果无法区分，记入备注，不强行交换。

### 推荐分两步，避免全包返工

1. 先标 `pilot_ids.json` 的 18 张开发图（6 个三视图组），交给负责人确认语义、左右和可见性。
   不用 holdout 来改规则。剩余图片也已全部交付，可在规则确认后批量完成。
2. 首批必做：每只真实鼠的实例记录、body-only 框、下面 7 个观察点与状态。
   `nose`、四个 `*_paw_center`、`tail_base`、`tail_tip`。爪点取掌/跖部向趾根过渡区域中心，
   不取最长指尖或腕踝；不清楚就留空。它们是与 Core23 同部位的候选，仍需首批对照
   确认表面观察点与内部骨关节投影的偏差，**不能直接宣称人工图得到精确骨骼 GT**。
3. 有预算时加标 `observed_ear_tip_l/r`、`observed_ear_attachment_l/r`、
   `observed_eye_l/r`、`observed_wrist_l/r`、`observed_elbow_l/r`、
   `observed_ankle_l/r`、`observed_knee_l/r`，记录在 `extra_observations`。
   都是相应可辨认解剖标志的中心，不以剪影最突出点替代关节。额外点与旧 Core23 分开存，
   用来解决上述语义缺口；目前不自动映射。
4. 其余 Core23 槽位默认 `deferred_definition`，不是让标注员凑够 23 个点：
   耳/躯干内部中心/旧 elbow/knee/重复腹点，在定义核定前不要求点击。
   `tail_q1–q5` 先不手工硬猜；可另外沿可见尾巴中线从根到尖画折线，
   中间看不见就拆成多段，禁止跨遮挡直线连接。将来确定骨长比例后再做带 derived 标志的转换，
   2D 曲线弧长采样也不能当成透视下的精确 3D 骨长采样。

### 遮挡与实例

- `visible`：位置在当前视图直接可见；提供 x/y。
- `occluded_localizable`：不可见但能合理定位；提供 x/y，并明确是人工推测。
  如果只靠另一视图猜位置，必须备注，不可标 visible。
- `unlabelable`：图内可能存在但无法定位；x/y 必须 null。
- `out_of_frame`：确定在画外；x/y 必须 null。
- `deferred_definition`：定义待确认；x/y 必须 null，不能被当作负样本或缺失器官。
- 模板的 `pending` 表示尚未人工处理，不能作为完成标签导入训练。
- 每张图枚举所有真实鼠，玻璃反射不算第二只；`instances=[]` 的待标模板**不表示无鼠真值**。
  session 内能确定身份时用固定 `instance_id`；无法确定则置 `identity_verified=false`。
- body-only 框包含头、耳、躯干与四肢，排除尾巴、影子、倒影；严重遮挡时只框可见实体，
  并设 `bbox_extent=visible`。可另存含尾巴的 full-animal 框，勿混用指标。
- 不强制 mask 标注，不显示 DLC 或模型预测作为预标注。不要把 DLC likelihood 当可见性。

### 回交格式

可用标注团队熟悉的点标注软件；本包不绑定软件版本。
保留原文件名和尺寸，回交每图每实例的关键点、状态、框和标注员/审核记录。
`annotation_template.jsonl` 是可机读的空任务清单，`instance_template.json` 为实例模板，
`schema.json` 为字段约定，`core23_definitions.csv` 为完整编号字典。
它们是项目中间格式，**不是宣称可以直接导入 CVAT/Labelme/COCO 的原生格式**。
团队如果使用现成平台，可导出其原生 JSON 加原图；按本文语义保留可见性和额外点，之后统一转换。

## 复核与评估边界

288 张均按事先的来源划分保留 train_allowed=false。holdout 导出供独立人工标注，
不展示模型预测，也不因为导出而解除训练/模型选择的隔离。评估前应双人复核至少首批、
左右不确定、遮挡和所有额外点；冻结映射规则后再审核 holdout 并评估。
未确认的内部点不计算精确人工 PCK，不强行三角化伪造 3D 真值。

建议后续另立版本修正 Cylinder 的命名/腹点语义，并同时修正合成导出与数据适配；
当前没有悄悄改动已有数据或模型输入。
