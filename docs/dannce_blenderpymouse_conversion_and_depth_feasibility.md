# DANNCE 小鼠数据转换与稠密几何伪标注评估

日期：2026-09-07

## 已落地数据

官方数据、标定与完整手工标签保存在：

- `dataset/DANNCE/raw/markerless_mouse_1`
- `dataset/DANNCE/raw/markerless_mouse_2`
- `dataset/DANNCE/download_manifest.json`

即用 BlenderPyMouse 风格数据保存在：

- `dataset/DANNCE-MarkerlessMouse1-Core23-RealMultiview`
- `dataset/DANNCE-MarkerlessMouse2-Core23-RealMultiview`

它们是真实相机数据，只复用 BlenderPyMouse 的目录与标注契约，不属于
`dataset/BlenderPyMouse` 合成数据集合。

两套数据分别包含 81 与 91 个手工标注时刻，每个时刻 6 个同步视角，图像为
1152×1024。使用的是完整 `label3d_dannce.mat`，不是每套只有 10 帧的
`label3d_demo.mat`。

转换脚本：`BlenderScripts/convert_dannce_mouse_to_blenderpymouse.py`。

```bash
/home/haozong/projects/MouseGeoForge/.venv-vggt-omega-eval/bin/python \
  /home/haozong/projects/MouseGeoForge/BlenderScripts/convert_dannce_mouse_to_blenderpymouse.py
```

## 相机与畸变处理

DANNCE MATLAB 文件使用行向量约定，转换后采用项目统一的 OpenCV 列向量约定：

- `K_cv = K_dannce.T`
- `R_cv = r_dannce.T`
- `t_cv = t_dannce`
- 外参含义为 world-to-camera

图像矫正使用三项径向畸变与两项切向畸变，并保留 DANNCE 原内参中的 skew
参与反向采样。输出内参为零畸变、零 skew 的针孔模型。转换后的 2D 点不对原
2D 标签做近似逆变换，而是从同一帧 3D 标签经过新针孔相机重新投影，因此图像、
2D、3D 与相机严格处于同一坐标系。

全数据审计结果：

| 会话 | 场景 | 图像 | 相机文件 | 有效 Core23 点 | 可读失败 |
|---|---:|---:|---:|---:|---:|
| mouse 1 | 81 | 486 | 486 | 1273 / 1296 | 0 |
| mouse 2 | 91 | 546 | 546 | 1439 / 1456 | 0 |

dataloader 实测可将两套数据混合成 `[B,6,3,H,W]`，相机与 2D/3D 的最大归一化
重投影误差为 `1.33e-7`。DANNCE mouse22 到 Core23 有 16 个直接语义对应；其余
7 个点以及源标签中的 NaN 使用 `world_valid=False`，不会获得伪造的零坐标监督。
DANNCE 不提供可靠逐点遮挡可见性，因此 `visibility_valid=False`。

## 动态 3D Gaussian 是否能生成深度

结论：**技术上能渲染深度，但不能把它当作真实深度。**

DANNCE 的六台固定、同步、已标定相机非常适合多视角动态重建。Dynamic 3D
Gaussians 能维护随时间移动的持久 Gaussian，并直接展示了深度合成与轨迹输出；
4D-GS 和 Deformable 3DGS 也能表达时变场景。参考：

- [DANNCE 官方数据格式与相机参数](https://github.com/spoonsso/dannce)
- [Dynamic 3D Gaussians 官方项目](https://dynamic3dgaussians.github.io/)
- [4D Gaussian Splatting, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_4D_Gaussian_Splatting_for_Real-Time_Dynamic_Scene_Rendering_CVPR_2024_paper.html)
- [Deformable 3D Gaussians, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_Deformable_3D_Gaussians_for_High-Fidelity_Monocular_Dynamic_Scene_Reconstruction_CVPR_2024_paper.pdf)

主要风险不是相机，而是观测条件：只有 6 个稀疏视角；黑色毛发弱纹理；爪和尾巴
细而快速；玻璃、反光和固定背景会主导光度优化。Gaussian 的 alpha 合成深度还依赖
表示和渲染定义，不天然等于第一真实表面。因而直接用 dynamic 3DGS 导出的 depth
训练，会把浮点、表面厚度、遮挡错误和背景泄漏固化进模型。

## 推荐实施顺序

1. **先做逐视角前景 mask。** 用固定相机的时间中值背景、分割模型和时序传播组合，
   再用六视角轮廓重投影一致性过滤玻璃反光与错误区域。
2. **先建立视觉外壳基线。** 六个已标定 silhouette 的反投影锥交集可得到保守 3D
   occupancy；视觉外壳是物体形状的上界，适合输出前表面深度和沿射线的厚度/不确定性，
   对黑毛弱纹理通常比直接 MVS 更稳。参考
   [CMU Visual Hull Construction](https://publications.ri.cmu.edu/visual-hull-construction-alignment-and-refinement-across-time)。
3. **再训练动态 Gaussian。** 静态笼体/背景与动态小鼠分层建模；用视觉外壳初始化或
   约束前景 Gaussian，使用已知相机而不是重新做 SfM，并屏蔽玻璃/反光区域。
4. **把 Gaussian 深度当软伪标签。** 只保留同时满足多视角轮廓、关键点重投影、时间
   连续性和前后向渲染一致性的像素；保存来源、置信度与深度区间。Shape of Motion 也
   明确把单目深度与长程 2D track 作为噪声先验融合，而不是地面真值，见
   [官方项目](https://shape-of-motion.github.io/)。
5. **与 Blender 精确深度分权训练。** Blender depth 权重最高；视觉外壳深度用于
   conservative occupancy/区间损失；dynamic-GS/VGGT/单目深度只用于低权重一致性或
   teacher-student 伪监督。

## 可生成的附加 annotation

| Annotation | 建议可信级别 | 生成方式 |
|---|---|---|
| 2D 前景 mask / box / COM | 中高 | 背景建模 + 分割传播 + 六视角一致性 |
| 粗 3D occupancy / visual hull | 中高 | 标定轮廓锥交集 |
| 前景深度下界、上界、厚度 | 中 | 视觉外壳首/末交点 |
| Dynamic-GS alpha / depth / normal | 中低，软标签 | 分层动态 Gaussian 渲染 |
| 稠密 2D track / optical flow | 中 | 长程点跟踪 + 前后向一致性 |
| 3D point trajectories / scene flow | 中低 | persistent Gaussians + 多视角过滤 |
| 全 18k 帧 mouse22/Core23 | 中，软标签 | DANNCE/sDANNCE 推理 + 骨长/重投影过滤 |
| 遮挡/可见性 | 中低 | 深度排序 + 多视角 silhouette，不替代人工真值 |
| surface correspondence / CSE | 实验性 | 动态 Gaussian track 或模板鼠配准 |

下一项建议实验不是直接跑完整 4D-GS，而是先从每个会话抽取约 200 个连续同步时刻，
做 mask + visual hull 小规模基线，量化六视角 silhouette IoU、关键点到外壳距离和跨视角
深度一致性；基线通过后再比较 Dynamic 3D Gaussians 与 4D-GS。
