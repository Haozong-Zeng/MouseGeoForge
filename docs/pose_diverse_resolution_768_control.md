# Diverse400：512 → 768 分辨率对照

2026-09-10。按用户要求检查：扩大至 400 场景后，提高分辨率是否进一步改善双鼠。
基线为已验收的 `output/pose_diverse_core23_512_v1`，不是旧 64 场景模型。

## 固定协议

- 完全相同的 400 训练、32 验证场景及其顺序，原 64 训练子集仍单独复算。
- 长边 768：Homecage 输入 W×H=768×432，方形场景 768×768。
  从源 RGB/深度直接读取缩放，不放大 512 缓存；像素数为 512 版的 2.25 倍。
- 相同 seed17、新头初始化和原始 Omega 权重；不是从训练好的 512 checkpoint 续训。
- 冻结 encoder，重新提取 RGB 特征；相同相机 ID [0,2] 和实际视角顺序。
- 相同 15000 步、5000 步 LR=1e-4 后余弦至 1e-5、采样顺序、优化器及损失。
- bridge off；不加入 ROI、增强、时序、尾曲线、3D 损失、新数据源或真实图像。
- 等更新数、等 epoch，不是等计算量；初始化权重应一致，但初始预测/指标可因
  输入和 token 网格改变而不同，不要求跨分辨率初始指标相等。

## 验证与资源

`tools/pose_diverse_resolution.py audit` 检查全部 432 对场景：除 image_hw 外清单
完全相同，世界坐标/尺度、归一化 UV、实例、相机顺序不变，内参乘 1.5；
逐点重投影误差 <0.01 输入像素、深度 anchor median AbsRel <0.15。
还检查当前代码能原样重建已完成的 512 manifest，防止扩展影响旧协议。

训练结束后独立重载全部 32 验证场景；比较初始化、LR 和训练顺序哈希，
输出单/双鼠分来源 NME、计漏检 PCK、检测、尾部、鼠体/全图深度误差。
固定终点验收，不用验证挑 checkpoint。视觉复核双鼠 40/98（训练）和
246/393（验证），与 512 Diverse400 对比；相机 2 为侧视，0 为顶视。

启动前 GPU 约 1.4/95.6 GiB 已用，WSL 约 122 GiB 内存可用、磁盘约 2.9 TiB 可用。
仍需根据运行时峰值确认资源占用，不将 2.25 倍像素数直接当成准确耗时预测。
初步预算约 2–4 小时，待首批训练计时校正。若 OOM 或其他异常则停止并检查，
不擅自降低分辨率、改变样本数或重启覆盖结果。

## 文件与执行

- 配置：`configs/experiments/pose_diverse_core23_768_v1.json`
- 队列：`output/pose_diverse_resolution_768_queue_v1/`（audit/train/verify/compare 日志）
- 几何审计：`output/pose_diverse_core23_768_preflight_v1/`
- 训练输出：`output/pose_diverse_core23_768_v1/`
- 配对报告：`docs/experiments/pose_diverse_resolution_768_results.json`（完成后生成）

```bash
.venv-vggt-omega-eval/bin/python tools/run_pose_diverse_resolution_queue.py --launch
```

队列依次审计、训练、重载和比较；独占锁和独占日志防止重复启动，不覆盖 512
结果。37 项根目录回归、7 项几何损失、4 项时序索引共 48 项测试已通过，
其中新增 6 项分辨率配对/不变性检查。
本轮结果仍是单种子合成域实验，不足以证明真实动物泛化。

当前状态：全部 432 对几何审计通过，512 manifest 重建不变；从提交 `8d7a2f6`
启动的 GPU 队列正在重新提取 768 RGB 特征，尚无 768 收益结论。
已恢复本任务自动验收，完成后重载与视觉复核；有异常则报告，不自动改变实验。
