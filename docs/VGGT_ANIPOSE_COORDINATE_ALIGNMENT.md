# VGGT-Omega 与 Anipose 坐标对齐

## 结论

训练时不使用逐帧 Sim(3)。所有监督先转到首输入相机（Camera A）坐标系，再使用一个
trial 全程固定的 scene-unit 尺度。最终视频评分也只能使用一个整段视频共享的变换。

## Anipose 坐标恢复

部分 `pose-3d/pawfilt.csv` 保存的是 Anipose 调整后的坐标，并包含 `M_00...M_22`
和 `center_0...center_2`。恢复 calibration 世界坐标：

```text
X_calibration = inv(M) @ (X_anipose_adjusted + center)
```

22 个 trial 中，`center` 的跨度约为 `[52.6, 62.4, 18.4] mm`，`M` 的最大相对
旋转差约为 `21.4 deg`，因此不能跨视频直接比较原始 CSV 坐标。

当前真实几何 loader 不直接使用该 CSV 世界坐标，而是通过固定 calibration 和 DLC
2D 重新三角化，从源头生成 calibration 世界坐标。

## 首相机规范

设 calibration 外参是 OpenCV camera-from-world 矩阵 `E_i=[R_i|t_i]`。以 Camera A
作为 VGGT 世界系：

```text
E_i_canonical = E_i @ inv(E_A)
X_A = R_A @ X_calibration + t_A
```

诊断中 Camera A 规范化后的最大旋转单位阵误差约 `6e-8`，最大平移误差约 `1e-10`。
原始 VGGT-Omega 的首相机预测误差也很小（旋转约 `0.0017`，平移约 `0.0002`），
因此训练不需要再拟合自由旋转或平移。

## 固定 Scene Unit

官方 synthetic 规范是：

1. 转到首相机坐标系。
2. 反投影有效深度，计算所有 3D 点到原点的平均距离 `s_scene`。
3. 深度和外参平移同时除以 `s_scene`。

真实视频没有可靠的完整 dense GT，因此使用冻结的原始 VGGT-Omega，在每个 trial 的
多个帧上预测 ABC rig baseline。去除 MAD outlier 后取中位数 `b_pred`：

```text
s_trial = b_physical / b_pred
```

`s_trial` 对整个 trial 固定，不能逐帧更新。该尺度表由原始 checkpoint 生成，不能用
待评估 checkpoint 自适应拟合，否则会泄漏并掩盖模型漂移。

## 评分

优先报告两组结果：

- `canonical`: 使用 Camera A 规范和预先固定的 `s_trial`，不再拟合。
- `trial_fixed_sim3`: 只为诊断残余 gauge，在整段视频的校准帧上拟合一次 Sim(3)，
  然后原样应用到所有帧。

禁止对每帧独立拟合尺度、旋转或平移。相机相对位姿、深度和 3D 关键点必须共享同一套
trial 级变换。

## 诊断文件

- `output/coordinate_alignment/real_trial_gauge_diagnostic.json`
- `output/coordinate_alignment/real_trial_gauge_metadata.json`
- `output/coordinate_alignment/vggt_original_trial_rig_scales_624x416.json`
