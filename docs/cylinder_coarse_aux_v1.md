# 粗坐标辅助监督 V1

2026-09-10 自主循环首项。动机见 `cylinder_localization_audit_v1.md`：相当多 GT
已在局部搜索窗外，直接锐化热图会损伤双鼠。此轮只增加身体粗坐标辅助监督。

## 锁定协议

- 完全复用 `cylinder_combinations_v2` 的 800 步源姿态、相机顺序、尺寸和回放。
  关闭组为已完成的 `combination_coverage`，开启组从同一原始 checkpoint 初始化。
- seed17，lr3e-6，768 输入，冻结 encoder；共享 adapter、depth、pose 头均按原方式训练。
- 原最终坐标 smooth-L1 权重5；新粗身体坐标 smooth-L1 权重2.5，beta=.01，
  所有匹配实例的有效身体点一起取均值；Core23 tail7:14不加辅助监督。
- 包含有效的遮挡 xy 标签，不用 visibility 去删除坐标监督。
- 最终坐标 Hungarian 匹配只调用一次，原 loss 与辅助项共享匹配结果。
  通过复制 Python 函数查找环境注入匹配，不编辑历史锁定的 upstream loss 文件，
  也不全局替换其 matcher。窗口、温度、显式局部热图 loss 和其他监督不变。
- 新增辅助目标自然会影响总梯度及其裁剪；不声称辅助项不影响原有分支。
- 1,800 图像、3,540,384 patch tokens；固定最终步，88 开发记录与17门槛不变。

## 运行与验证

```bash
PYTHONPATH=vggt-omega:. .venv-vggt-omega-eval/bin/python tools/train_cylinder_coarse_aux.py --prepare-only
PYTHONPATH=vggt-omega:. .venv-vggt-omega-eval/bin/python tools/train_cylinder_coarse_aux.py --launch
PYTHONPATH=vggt-omega:. .venv-vggt-omega-eval/bin/python tools/verify_cylinder_coarse_aux.py
PYTHONPATH=vggt-omega:. .venv-vggt-omega-eval/bin/python tools/analyze_cylinder_coarse_aux.py
PYTHONPATH=vggt-omega:. .venv-vggt-omega-eval/bin/python tools/audit_cylinder_coarse_candidate.py
```

GPU 运行中请先读 `output/cylinder_coarse_aux_v1/status.json`、`run.log` 与运行锁，
不要重复启动或修改已锁定代码。`coarse_loss_history.json` 单独记录原 pose loss、
新辅助项及监督点数，可逐步核对总损失。

启动前 130 项单元测试通过；实际双鼠样本验证关闭辅助项逐值复现原 loss，
仅一次匹配、64 个身体点参与。这里“实际双鼠样本”指已渲染合成样本，不是真实视频。

## 完成结果：淘汰此候选

训练及初末评估共346.2秒，协议/源数据/代码/权重/匹配/重载验证通过，初始化
指标相对关闭组最大差异0；仅通过 **9/17** 性能门槛，关闭组为10/17。没有推广。

| 底视输入上下文 | OFF身体NME | ON身体NME | 相对变化 | 改善姿态数 |
|---|---:|---:|---:|---:|
| 仅底视 | .18127 | .17886 | -1.3% | 5/8 |
| 侧＋底 | .18864 | .20641 | +9.4% | 0/8 |
| 两侧＋底 | .18011 | .20219 | +12.3% | 0/8 |
| 两侧＋底＋顶 | .17250 | .19276 | +11.7% | 0/8 |

旧侧视对检测感知 PCK 从 .47690 降到 .46603，跌破 .46728 保留门槛；双鼠从
.49049 降到 .48777，仍低于 .50601 门槛。双鼠 Core23 NME .12927、深度
AbsRel .02476 也未通过。各底视深度仍通过相对原始模型的保留门槛，不能据此
忽略关键点退化。

另外重跑88条候选定位分解，保存于 `localization_points.json` 和
`localization_report.json`；预测与训练评估保存值一致，未更新权重。

| 上下文 | OFF粗NME → ON粗NME | OFF入窗率 → ON入窗率 |
|---|---|---|
| 仅底视 | .22422 → .22133 | 45.3% → 43.8% |
| 侧＋底 | .22456 → .24395 | 51.6% → 48.4% |
| 两侧＋底 | .21367 → .23778 | 56.3% → 48.4% |
| 四视图 | .20426 → .22723 | 59.4% → 53.1% |

不是只在最后的热图修正阶段退化：多视开发集的粗定位和窗口覆盖也变差。
辅助训练 loss 前/后100步均值 .05684/.03569，说明训练目标下降不保证开发集泛化。
不能仅凭这一轮判定所有粗坐标监督无效；该结果针对当前权重、步数、源姿态池。

已实际查看 comparison、training_curves、hardest_bottom_pose、localization_comparison
四图。固定帧55的单底视略好，但双/三视图头部点继续向右上方偏移，局部肢体
偏差增加；没有用单底视的有利图掩盖多视图失败。

最终 checkpoint SHA256：
`08ce02e8a598c20a8c4a5b9b33c4ef1edc96014cb6d938e1ca216e0b52a45714`。

## 自主下一阶段

先做局部网格梯度路由的只读审计，不接着这个失败权重训练、不加大辅助权重。
比较当前关闭辅助组在相同训练样本上的坐标梯度：完整路径 vs 仅 detach 局部
采样网格；分辨窗口采样的间接梯度与显式 refined=coarse+delta 路径，检查是否
稳定出现与 GT 方向冲突的间接梯度，再决定是否做独立800步控制。

历史 `pose_overfit_diagnostic.md` 已做过“保留热图loss＋detach”的小样本实验，
不能将其当作当前“热图loss=0＋detach”问题的答案，也不应重复原实验。
这是新的待检验假设，不宣称已找到唯一根因。后续由自动台账继续记录。
