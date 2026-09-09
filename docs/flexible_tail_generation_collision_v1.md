# Homecage 灵活尾巴生成与穿模矫正 v1

## 决策

CRIM13/Lightning Pose 只提供 `base_of_tail`。生产级 Blender 数据不再让完整尾巴追随真实图像中不可可靠恢复的尾尖或中间点：

- `tail_base` 是硬位置约束；
- 身体后轴到尾根的切向是弱方向先验；
- 前两节尾骨降低随机扰动，保证尾根连续；
- 其余尾节由时空相关的平滑曲率采样，不依赖原图远端尾姿；
- 每帧独立检查地面、笼壁、自己的身体代理和其他小鼠身体代理；
- 穿模推出后重新恢复逐节骨长和最大弯曲角，无法收敛的帧拒绝进入数据集。

这与训练监督保持一致：真实 CRIM13 仅监督 `tail_base`；Blender 生成的 `tail_q1...tail_q5/tail_tip` 是有效合成几何标签，但内部点继续以曲线级损失为主，不把它们解释成真实图像逐点对应。

## 实现

- `blender/flexible_tail_solver.py`
  - 不依赖 `bpy` 的 NumPy 几何求解器；
  - 保持尾根和 13 节现有尾骨长度；
  - 远端噪声随弧长增加，帧间使用 AR(1) 低频随机过程；
  - 支持地面、AABB 内壁、任意朝向椭球身体代理；
  - 输出每帧是否接受、修正次数、残余穿模和骨长误差。
- `blender/apply_flexible_tail_animation.py`
  - 缓存原动作的尾根和身体朝向；
  - 只重建 `Tail_NURBSCurve.001` 的关键帧，通过现有 13-bone Spline IK 驱动尾巴；
  - 不写 Root、脊柱、四肢或任何非尾骨；
  - 支持 `--other-armature`，双鼠时把另一只鼠加入碰撞代理；
  - 保存前比较非尾骨姿势摘要，不一致则报错并拒绝保存。
- `configs/flexible_tail_homecage_v1.json`
  - 固化采样幅度、时序相关、弯曲限制、碰撞容差和拒绝规则。
- `blender/test_flexible_tail_solver.py`
  - 覆盖可复现性、骨长、地面/身体/笼壁、远端多样性和时序连续性。

## 当前 Homecage 派生项目

输入：`dataset/Homecage/Homecage-BlackFur-11_01_22_12wk_m0-Core23-Sample10-Frames200-399.blend`

输出：`dataset/Homecage/Homecage-BlackFur-11_01_22_12wk_m0-Core23-Sample10-Frames200-399-FlexibleTailV1.blend`

200 帧审计结果：

- 接受 200，拒绝 0；
- 尾链总长度约 65.76 mm，逐节长度不变；
- 最大残余穿模约 `5.02e-10 mm`；
- 最大逐节长度误差约 `1.42e-14 mm`；
- 地面修正 558 次，笼壁和身体修正本段均为 0；
- 非尾骨姿势摘要前后一致，姿势未变。

视觉抽检位于 `dataset/Homecage/flexible_tail_review_v1/`。首、中、末帧顶部图中尾根连续且曲线平滑；第 100 帧侧视图中尾巴落在垫料面，没有悬空或穿入地面。第 100 帧尾尖小弯钩接近当前随机幅度上限，暂不再提高 `distal_noise_gain`。

## 双鼠生产渲染的接入顺序

1. 先根据 CRIM13 七点放置两套独立 armature，只把 `tail_base` 用作尾巴条件。
2. 固定身体与四肢动作后，对两只鼠分别缓存尾根、身体坐标系和身体代理。
3. 同时生成两条尾链；每条尾链都加入自己身体、另一只鼠身体、地面和笼壁约束。
4. 若两条尾巴之间也需要互斥，增加逐节 capsule-capsule 检查；v1 先处理视觉上更严重的尾-身体穿模。
5. 写入两条 Spline IK 曲线后，做最终 evaluated mesh 审计；未通过帧重新采样 seed，不能仅在 RGB 后处理。
6. 渲染 RGB、实例 mask、depth、Core23 2D/3D，并把 seed、修正计数和拒绝原因写入 manifest。

## 运行示例

```bash
blender -b input.blend --python blender/apply_flexible_tail_animation.py -- \
  --output output-FlexibleTailV1.blend \
  --report output-FlexibleTailV1.tail-audit.json \
  --seed 5706
```

双鼠时给每只鼠单独运行一次，并把另一套 rig 作为碰撞代理；正式生成器中应改成同一轮联合求解，避免顺序偏差：

```text
--armature MouseA --other-armature MouseB
--armature MouseB --other-armature MouseA
```

## 后续限制

- v1 使用中心线加尾半径做解析碰撞；保存前仍需 evaluated mesh 复核毛发/皮肤的真实包络。
- AABB 适用于 Homecage 内壁近似；复杂装置应换成 signed-distance field 或 Blender BVH 最近点。
- 当前单鼠片段没有触发身体或墙壁推出，相关能力由单元测试覆盖；必须在双鼠接触/追逐子集上再统计实际触发率。
- 远端随机过程应按行为类别设不同先验，例如探索、追逐、直立和理毛不能共用完全相同的曲率分布。
