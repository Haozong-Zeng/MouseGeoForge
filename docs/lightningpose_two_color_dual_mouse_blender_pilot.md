# LightningPose 双毛色双鼠 Blender Pilot

## 产物

- Blender 项目：`dataset/Homecage/Homecage-BlackBrownFur-LightningPose-CRIM13-DualMouse-Pilot-f5306-5600.blend`
- 构建计划：`dataset/Homecage/LightningPose-TwoColor-DualMouse-Pilot.plan.json`
- 机器审计：`dataset/Homecage/Homecage-BlackBrownFur-LightningPose-CRIM13-DualMouse-Pilot-f5306-5600.audit.json`
- 源片段检查图：`dataset/Homecage/review/lightningpose_dual_mouse_source_5306_5600.png`
- 构建脚本：`blender/build_lightningpose_two_color_dual_mouse_pilot.py`
- 最终顶部预览：`dataset/Homecage/review/lightningpose_two_color_dual_mouse_pilot_v2/HomecageCam_Top_frame029.png`
- 最终交叉侧视：`dataset/Homecage/review/lightningpose_two_color_dual_mouse_pilot_v2/HomecageCam_SideNegX_frame029.png`

项目为 1280×720、8 fps、57 帧。两只鼠分别具有独立 mesh、armature、毛发对象与 14 点柔性尾曲线；黑鼠使用 C57 LocalColor V2 材质，白色源实例映射为棕色外观。原始单鼠模板集合只作克隆来源，在最终项目中同时隐藏视口和渲染。

双鼠的18个 `CURVES` 毛发对象默认关闭视口显示，但保持 `hide_render=false`。这能显著降低GUI时间线播放时的依赖图开销，同时不改变F12或后台数据集渲染结果。场景属性 `MGF_fur_viewport_default=hidden` 和 `MGF_fur_render_enabled=true` 记录该约定。

## LightningPose 条件

来源为 CRIM13/LightningPose 会话 `021709_A25_Block22_BCfe1_t`，采用标签帧 5306、5348、5390、5432、5474、5516、5558、5600。八个真实标签映射到 Blender 帧 1、9、17、25、33、41、49、57。

真实数据提供两实例的顶视位置、朝向、接触关系与尾根；场景坐标以 0.55 mm/px 映射并居中到 Homecage。中间帧是 Blender 插值，不是真实逐帧标签。源数据没有多相机外参与可靠深度，不能把该二维条件解释成真实三维真值。

## 姿势与碰撞策略

当前 pilot 使用通过侧视检查的稳定内部姿势：黑鼠帧 15，棕鼠帧 98。棕鼠最初使用的帧 14 是后肢站立姿势，虽然脚没有悬空，但该高度无法由顶视 CRIM13 标签确认，因此已替换为低头四足姿势。

LightningPose 在本版驱动整鼠平移、水平朝向、双鼠接触布局与尾根；没有把每个稀疏七点逐关节烘焙到完整 rig。姿势库匹配帧仍保留在计划与审计中作为实验诊断，但直接稀疏重定向会造成约束链数值不稳定，所以未作为最终内部骨骼动画。

两鼠身体用定向代理保持最小接触距离，允许遮挡但阻止躯干中心合并。尾巴只依赖尾根与弱方向先验，远端曲线由时序平滑采样和地面、身体、另一只鼠、笼壁矫正产生；真实数据不为远端尾点提供真值。

## 当前审计

- 两鼠代理中心最小距离：39.9585 mm；
- 最大 LightningPose 条件点误差：39.2816 mm；
- 黑、棕尾巴均为 57/57 帧通过；
- 黑尾最大残余穿透：0.02674 mm；棕尾：约 4.40e-10 mm；
- 尾段最大长度误差约 1e-14 mm；
- 原始单鼠模板集合 `MGF_First200_Mouse` 的 `hide_render` 与 `hide_viewport` 均为真。

最终毛发预览确认只出现两个实例；顶部视角中两鼠轮廓保持分离并发生局部社交接触；交叉侧视中两鼠躯干高度合理、后爪承重，没有整鼠悬空。棕鼠尾巴在侧视中有遮挡，但顶部视角可见尾根和曲线连续；黑尾末端轻微上翘，未穿地或穿墙。前爪是否接地由下述骨骼审计单独判断。

## 爪标注与逐帧站姿审计

CRIM13/LightningPose 每只鼠只有鼻、左右耳、颈顶、左右后膝和尾根七点，不包含前爪、后爪、腕或踝。Core23 虽然预留 `left/right_forepaw_center` 和 `left/right_hindpaw_center`，但真实 CRIM13 样本中这四点必须保持 `NaN` 且 `valid=false`；只有 Blender 合成样本可从 `Fingers1_L/R`、`Toes1_L/R` 及表面支持区域导出爪点。

逐帧机器审计位于 `dataset/Homecage/Homecage-BlackBrownFur-LightningPose-CRIM13-DualMouse-Pilot-f5306-5600.paw-audit.json`：

- 黑鼠髋部最大左右高差 0.0122 mm、滚转代理 0.073°；棕鼠为 0.0153 mm、0.089°，均无系统性身体侧倾；
- 黑鼠左右后爪中心分别约在地面上方 2.90/2.36 mm，棕鼠两侧均约 2.09 mm；
- 黑鼠前爪末端约在地面上方 18.0–20.1 mm，左右基本对称；棕鼠约 11.0–17.9 mm，左前肢通常更低，最大爪中心高差约 4.50 mm；
- 当前动作因此更接近后足承重、前爪抬起的接触姿势，不是严格四足同时踩地；CRIM13 没有爪标签，不能声称该抬爪来自真实标注；
- canonical 趾链在插值帧32出现轻微穿垫料：棕鼠左后趾最低约 -3.77 mm、右后趾约 -1.65 mm。`Toes1` 爪中心仍在地面上方约 2.09 mm，问题来自远端趾节插值，不是腿或身体整体歪斜。

如果把该 pilot 升级为生产数据，应在稀疏关键帧插值后增加足底/趾端地面矫正：对承重爪施加约 0.5–1.0 mm 的地面间隙，并只修正远端趾链或相应 IK 目标，避免改变由 CRIM13 约束的整鼠位置和朝向。

## 训练用途与下一步

该项目适合验证双实例渲染、两种外观、遮挡、逐实例 mask/depth/Core23 导出和尾巴穿模矫正管线。现阶段不要把它当成 CRIM13 动作的逐关节三维重建。

生产版建议新增一个干净的 FK/解剖代理层，用 SE(3) 或四元数优化在关节限位、足底接地和身体碰撞下拟合七点，再把结果稳定传给高细节 rig。通过这层后，才应让八个稀疏关键帧改变内部姿势，并渲染完整多视角 RGB、逐实例 mask、depth、Core23 2D/3D 与相机参数。
