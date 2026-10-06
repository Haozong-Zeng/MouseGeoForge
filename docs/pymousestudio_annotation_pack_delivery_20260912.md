# PyMouseStudioRaw 标注包交付记录

2026-09-12，承接底视无标签的人工标注准备请求。

- 成品：`output/annotation_packages/PyMouseStudioRaw-Core23-AnnotationPack-v1.zip`
  （197,412,321 bytes，约 197.4 MB / 188.3 MiB）。同名目录为已解压版本。
- SHA256：`60c6d96756806ae106aea9cc26b9f11e5cbc34d3aded727c559ee9360c4c40b9`。
- 288 张真实 PNG，底视 96、侧视 192；来自 12 段、每段 8 个固定索引、A/D/E 三视图。
  216 张开发集、72 张独立标注 holdout；来源分组仍然隔离，未开启训练许可。
- 1080×1200 原尺寸；逐视频从零顺序解码提取，逐张 PNG 解码与源解码像素比较一致。
  288 个图像哈希唯一，全部可解码；ZIP CRC 检查通过。
- 7 组 synthetic GT 示例，取 Cylinder WalkAround/WashFace/ClimbWall 第 31 帧，
  覆盖侧视、底视与一个顶视。每组包含干净原图、1–23 号投影参考图、原坐标 JSON。
- 人工查看了全部 7 张编号图与 AZL3 两个时刻的 A/D/E 六图预览：标签没有裁切，
  足掌/尾部示例可辨认；真实侧视可见装置遮挡与玻璃反射，应明确标注不确定状态。
  未查看 holdout 像素或模型预测以作模型选择。
- 附中文说明、完整 Core23 编号表、source31 对照表、空任务与实例模板、18 张开发试标清单、
  离线 HTML 索引、来源路径/帧号/哈希、校验报告。不绑定或伪称某标注平台原生格式。
- 实际核对 inventory 的 31 个 H5 名称与全部 11 份可用 config 的顺序映射，全部一致。

核心语义发现与操作规则见
[Core23 人工标注说明](pymousestudio_core23_annotation_20260912.md)。
其中后爪/踝、耳尖/骨起点、elbow/Wrist 命名、腹点/脊柱重合与尾骨采样问题，
意味着不能要求人工不加区别地填满旧 23 个点。先按清单做 18 张试标，首批优先
7 个候选观察点；其余歧义槽位暂缓，真实解剖额外点单独保留，不静默混入旧语义。

生成器：`tools/build_pymousestudio_annotation_pack.py`。
未更改任何原始视频、既有 synthetic GT、模型或训练配置；不恢复自动训练。
