# 衍生研究仓库如何组织

核查日期：2026-09-09。使用作者仓库 README、文件树，以及 GitHub REST API 的 `fork`/`parent`。
`fork=false` 仅表示当前不属于 GitHub fork network，不能证明代码无继承或历史从未 fork。

| 项目 | API 状态 | 结构和来源记录 |
| --- | --- | --- |
| [DynamicVGGT](https://github.com/NickHezhuolin/DynamicVGGT) | fork=false, parent=null | `dynamicvggt/` 与 `vggt/` 并存，独立 `training/`，README 明确 Built on VGGT，保留 LICENSE.txt |
| [StreamVGGT](https://github.com/wzzheng/StreamVGGT) | fork=false, parent=null | `src/streamvggt/`、`src/vggt/`、`src/dust3r/`、训练/评估代码；README 致谢多个上游 |
| [D-NeRF](https://github.com/albertpumarola/D-NeRF) | fork=false, parent=null | 自己的 `run_dnerf.py`/helpers/configs，README 明确扩展 NeRF 且大量基于 NeRF-pytorch |
| [Nerfies](https://github.com/google/nerfies) | fork=false, parent=null | 独立 `nerfies/` 包，train/eval/configs/third_party，自己的研究项目入口 |
| [4DGaussians](https://github.com/hustvl/4DGaussians) | fork=false, parent=null | 延续 GS 风格目录、使用 submodules；README 指向 3D-GS 等上游 |

DynamicVGGT README 还明确声明当前版本因合规原因重新实现、可能有 bug，且没有发布预训练权重。
因此可借鉴其项目分层，不直接把现有公开代码视为论文性能的可复现保证。

## 决策：保持 MouseGeoForge 独立仓库，新增自有 package

这不是“改动超过某个百分比就必须新建仓库”的规则。若主要为了给上游提交补丁，fork 有利于
同步与 PR；若研究目标、数据、训练入口和输出契约均已独立，独立项目 + 明确上游来源更清楚。
我们属于后者：多实例动物姿态、真实/合成数据、固定 rig 时序、tail curve 和联合几何。

```text
MouseGeoForge/
  mousegeoforge/       # 新的组合模型、adapter、任务损失/训练接口
  vggt-omega/          # 已纳入版本控制的上游来源快照及既有本地扩展
  tests/              # 新模块测试
  configs/experiments/ # 消融定义与执行配置严格区分
  docs/               # 调研、决策、工作记录
  BlenderScripts/     # 合成数据生产工具
```

不为本轮另建第二个 GitHub 仓库，也不把数据/backup 纳入。暂不整体重命名/移动旧模型目录，
避免破坏现有 editable installation、checkpoint key、脚本导入和本地 nested Git 历史。
新代码优先组合而非复制模型；不可避免的上游接口修改保持默认行为兼容并单独测试。

## 来源、许可证、权重三个层面都要记录

README/论文应分别说明：模型架构来源、实现代码来源、初始化 checkpoint 来源与版本。
独立仓库和致谢不能代替保留原版权/许可证。VGGT 与 VGGT-Omega 的许可证也不能因名称相似
就混同：本地 Omega 的 `LICENSE` 为 FAIR Noncommercial Research License；继续原样保留。
参见 [Omega 上游 LICENSE](https://github.com/facebookresearch/vggt-omega/blob/main/LICENSE)。
本说明是工程发布策略，不是新的许可证授权，也不把整个混合仓库重新标成 MIT/Apache。

沿用 `THIRD_PARTY_NOTICES.md` 中的上游 commit；每次同步单独提交。上游 checkpoint 只加载
可核对的 keys，新增 adapter 的缺失 keys 属预期，但原 encoder/depth keys 缺失不能静默忽略。
首个已推送提交 `828e862` 保留为本轮新模块之前的源码基准。
