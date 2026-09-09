# SBeA ground-aligned 3D 与标定坐标审计

日期：2026-09-08

## 结论

SBeA 的 `id3d.mat/coords3d` 与 `caliParas.mat/cam_mat_all` 不是同一个三维坐标系。
二者单位都保持为毫米，差异主要是每个 recording 固定的一次地面校正刚体变换，
不是尺度变化，也不是逐帧相机运动。

令标定投影矩阵接受的三维点为 `X_calib`，发布的地面对齐点为 `X_ground`，则应存在
每序列固定的变换：

```text
X_calib = R_gc X_ground + t_gc
x_view  ~ P_calib_view [X_calib; 1]
```

也可以把变换吸收到投影矩阵中：

```text
P_ground_view = P_calib_view [R_gc | t_gc]
```

但发布文件没有保存最终地面校正使用的 `R_gc/t_gc`，因此目前不能直接把
`id3d.coords3d` 与 `cam_mat_all` 组合成严格重投影标签。

## 代码证据

SBeA 的公开 MATLAB 代码先用 `cam_mat_all` 和多视角二维点三角化，得到标定重建坐标；
随后从四个足部点拟合平面，以 SVD 得到法向，再通过 Rodrigues 旋转把该法向对齐到
`[0,0,1]`。公开的编译模块 `level_correction_3D.pyd` 还保留了
`normal_vector_plane`、`normal_vector_z`、`origin` 和 `Rot_2_ground` 等符号，说明实际
处理可能还围绕一个 ground origin 旋转/平移。

发布的 `id3d.mat` 只包含 `coords3d` 和 `name3d`；发布的 `caliParas.mat` 只包含
`cam_mat_all`、棋盘格 `rotation` 和 `translation`。其中没有保存足平面法向、ground
origin 或最终 `Rot_2_ground`。

`caliParams.rotation/translation` 来自主相机的棋盘格外参。它描述棋盘格坐标与主相机
坐标的关系，但不等价于随后由动物足点估计的地面校正变换。把它直接作用到 `id3d`
后，在不同 recording/subset 上不能稳定恢复四视角投影。

## 实证检查

- `id3d` 的第三轴集中在地面附近，而前两轴覆盖 arena，符合 ground-aligned 表示；
- 直接用 `cam_mat_all` 投影 `id3d`，投影点大多不落入对应实例 mask/box；
- 直接套用 `caliParams.rotation/translation` 只在部分 fig2 recording 上得到粗略合理结果，
  在 fig3 recording 上明显失配；
- 用四视角 box center 对每序列拟合一个自由 SE(3) 后，绝大多数点可以回到图像范围，
  与“主要缺口是固定刚体变换”的代码证据一致。但 box center 不是关键点真值，拟合后的点落入实例
  box 的比例仍只有约 12%–63%，不足以作为精确重投影监督。

## 恢复顺序

1. 最可靠：取得 SBeA 预处理时保存的 `Rot_2_ground`、ground origin，或校正前
   `raw3d/new3d`；可无损计算逆变换。
2. 次优：在每序列少量帧上获得 16 点二维关键点，以所有相机、帧和实例共同优化一个
   固定 `SE(3)`，再用未参与拟合的人工帧验证重投影误差。
3. 现有 mask/box 只能给 SE(3) 初始化和可见范围约束，不能单独产生精确关键点投影。

在第 1 或第 2 条完成前，保持 `pose_to_camera_alignment_valid=false` 是必要的。
