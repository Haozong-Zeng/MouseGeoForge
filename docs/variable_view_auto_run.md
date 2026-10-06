# 1–N 视角自动推进：当前有界轮次

> 最新状态：有界自动阶段链已完成，V4 新种子复验未稳健通过，不再追加训练。
> 输出 `output/variable_view_seed_v4/`；V1–V4 全部已完成执行与验收。
> 参见 [最终总结](variable_view_final_summary.md)；heartbeat 在最终交付时暂停。
> 下文为已完成 V1 的历史记录，
> 其“暂停/不追加预算”描述不再代表当前接续状态。

2026-09-10，用户授权自动循环推进。复用当前任务的 heartbeat，每 10 分钟检查。
依据 [OpenAI Docs 调度说明](https://learn.chatgpt.com/docs/automations?surface=app)
使用当前任务跟进；无可行动变化不反复通知。

本轮目标：实现并运行 `tools/train_variable_view_adaptation.py`，完成固定双视角、
随机双相机对、随机 1–4 视角三组适应对照，每组 400 步，共 1,200 步。
配置：`configs/experiments/variable_view_adaptation_v1.json`。不是全量训练。

使用原双鼠训练 session 的 32 个姿态、原验证 session 的 8 个姿态作为开发评估，
不将已查看开发数据回流训练，不宣称独立测试或真实动物泛化。
三组从同一 768 checkpoint 初始化，使用同一场景抽样日程、固定场景尺度、损失与优化器。
固定训练更新数，不是等图像/等算力；必须报告图像数、encoder token 数、耗时与显存。
单/双/三/四视角分别评估，原固定双相机对独立做回退检查。

自动执行边界：

- 正常训练期间只检查，不重复启动或抢占 GPU。
- 不改变原数据、生产 checkpoint、分辨率、训练步数、预设门槛或划分。
- 输出独立写在 `output/variable_view_adaptation_v1/`；运行锁防并发，日志/状态持久化。
- 最多 40 GiB 编码特征缓存，8 小时墙钟预算；到限停止并报告，不擅自降配。
- 确定性实现错误可修复，但必须保留失败产物，确认旧进程结束；不能循环碰运气重试。
- 对照完成后重载验证、检查曲线和可视化、更新报告/测试，再暂停当前自动任务。
  后续若要加大数据/预算，先给出证据和建议，不自动无上限训练。

主要状态文件：`status.json`、`run.log`、`protocol.json`、各 arm 的 `history.json`、
`final_heads.pt`、`final_evaluation.json`。工具开发/启动完成后在本文补记。

## 当前执行状态与一次实现修正

首个目录 `output/variable_view_adaptation_v1/` 在缓存阶段触及 40 GiB 限额，
于 100.6 秒自动失败停止，**尚未开始任何参数更新**。原因是完整精度上下文特征
及 batch 比估算更大。保留该目录的日志、协议和已生成缓存，未删除任何数据。

已确认原进程退出，修正缓存执行方式，使用新目录：

**`output/variable_view_adaptation_v1_retry1/` 是当前有效运行目录。**

新执行采用最多 32 GiB 的 RAM LRU 缓存，不再生成新的编码特征缓存文件；
旧目录已完成的缓存仅在编码器、loader、原数据版本、子集和顺序完全匹配时复用，
缺失项按需重新编码。特征不降精度，不从四视角缓存截取成单/双视角上下文。
仍为同一三组、32/8 场景、每组 400 步，原优化预算、划分、尺度及门槛未变。
旧的约 40 GiB 磁盘产物作为失败审计保留；本次 RAM 上限与历史磁盘占用分别报告。

此为一次确定性实现问题修正，不是训练效果不佳后的调参/重跑。后续若再次失败，
先分析并报告，不应自动无限重试。新进程由 `--launch` 在 WSL 后台启动，具有独立运行锁。

训练完成后运行：

```bash
.venv-vggt-omega-eval/bin/python tools/analyze_variable_view_adaptation.py
```

该工具默认读取当前 retry1 目录，生成 `acceptance.json`、`comparison.png`、
`training_curves.png`、`best_v4_comparison.png`、`worst_v4_comparison.png`。
评估文件实际为各 arm 的 `evaluation.json`；自动验收必须查看图像并写结论，
不能仅看 gate 或只展示最佳样例。

已验证 retry1 进程进入固定双视角组参数更新（50/400 步时损失有限，GPU 正常使用），
并已通过缓存前向与实时前向逐字段精确一致检查。运行时禁止修改锁定源代码；
仅新增的验收脚本和文档不影响正在运行的训练流程。

## 最终验收

三组已完成共 1,200 步，有效运行 287.7 秒，进程退出且运行锁已释放。
指标、保存张量、原始文件哈希、视觉样例与 105 项回归检查已验收。
统一 1–4 视角的适应有明确二维收益，但固定对 PCK/深度保留门槛失败，未推广生产。
完整数据与下一步建议见 [第一轮结果](variable_view_adaptation_v1_results.md)。
本轮工作已完成，按约定暂停 heartbeat，不追加训练预算。
