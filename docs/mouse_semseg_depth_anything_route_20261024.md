# 独立分割支线：借鉴 Depth Anything 的 encoder 到语义分割迁移

2026-10-24。用户新增要求：参考 Depth Anything 做语义分割的思路，在远端单独尝试分割老鼠，可与 keypoints 分开。本支线与[关键点路线](head_first_progressive_encoder_route_20261024.md)并列，不等待本机新渲染，也不要求先训练好关键点。机器规格为 [mouse_semseg_depth_anything_20261024.json](../configs/experiments/mouse_semseg_depth_anything_20261024.json)，仍是 design_only；本次更新没有启动训练或 Slurm。

## 1. 具体借鉴什么

核查的 Depth Anything 官方版本：`1d03336771fe09c5398ffdd211441e33941a97dc`。[官方语义分割说明](https://github.com/LiheYoung/Depth-Anything/blob/1d03336771fe09c5398ffdd211441e33941a97dc/semseg/README.md)采用其预训练 ViT-L encoder 作下游语义分割初始化；[backbone实现](https://github.com/LiheYoung/Depth-Anything/blob/1d03336771fe09c5398ffdd211441e33941a97dc/semseg/dinov2.py)提取最后四层 token；[ADE20K配置](https://github.com/LiheYoung/Depth-Anything/blob/1d03336771fe09c5398ffdd211441e33941a97dc/semseg/config/depth_anything/depth_anything_large_mask2former_16xb1_160k_ade20k_896x896.py)将其经 Feature2Pyramid 的4/2/1/0.5缩放，输入 Mask2Former，并给 backbone 更小的 LR。

所以本次借鉴的是**预训练图像特征 + 中间层特征融合 + 专用语义分割 decoder**。Depth Anything 的深度分支另用 DPT，不能把 DPT 说成其官方 semantic head，也不将深度阈值当鼠体真值。官方城市/室内语义 checkpoint 不提供本项目的鼠体专用标签，需要真实鼠体 mask 监督。

首轮复用现有 VGGT 内的图像编码器 `aggregator.patch_embed`，不经过跨视图聚合、相机头或深度头；没有加载 Depth Anything 权重。因此这一主实验的准确名称是“Depth-Anything-inspired VGGT segmentation”，不能报告成实际 Depth Anything encoder 的效果。实际 Depth Anything 初始化作为后文单列对照。

## 2. 任务、监督与数据角色

输出全图的 `background / visible_mouse` 二类语义 mask，包含可见身体、耳、爪与尾；不补全被遮挡部分，不做实例身份或关键点预测。首轮面向已准入的单鼠图像；多鼠图只可输出鼠类并集，未经实例标注不宣称实例分割。

使用全图 RGB、保持纵横比的512 letterbox，无真值鼠框、无关键点提示、无推理时标定输入。每张图记录缩放、有效ROI、padding以及回原图变换。新模型对padding显式提供背景监督；推理在512空间先裁有效ROI再映回原图。原始鼠体外图像内容不因相机角色更换。

沿既有分割清单：真实标签为 PyMouseStudio 人工确认的可见鼠体mask，合成为原v3/准入回放的body真值。历史数量为TRAIN真实134底视+755其他，合成579底视+4,304其他；DEV真实71底视+376其他，合成72底视+144其他。它与关键点清单的441张真实DEV不同，不用关键点清单计数替代。所有source/label/policy/role必须重新pin并核对；不因存在文件就扩大准入。

合成body真值是排除玻璃/毛发的primary-ray皮肤表面代理，真实人工mask边界可能包含毛发。两种标签政策须逐图记录、分域报告，不能把边界差异全部解释成模型误差。新渲染若提供不同mask政策，单独冻结版本，不中途换标签。

PanoMouse 当前只有用于诊断的 SAM2关键点提示/视觉外壳参考，没有独立人工鼠体mask清单：首轮只作预先固定DEV上的跨装置定性和**明确标记的proxy**评估，不将其SAM2/外壳mask当监督GT或晋升依据。若要纳入Pano监督，先在既定TRAIN session建立人工审计mask清单并另登记数据对照。S04/S05、8sealed rigs、WaterReaching、DANNCE2、GR2/UR2/UR3和保护TEST继续排除；keypoint标签不是分割mask。

TRAIN sampler按真实/合成 × 底视/其他四层各2张组成有效batch8，层内按rig/session/cohort平衡；两对照用完全相同的样本次序与曝光。DEV不用于调threshold、后处理或重抽样。

## 3. 已有基线与新增比较

已有S1是冻结图像特征+浅层分割decoder。2026-10-20修复版冻结/LoRA配对的真实底视IoU约94.15%，native1px边界F1约0.416；该固定LoRA配方无明显收益。后续记录还暴露padding碎块与tail/paw误删问题。新实验应先在一致的ROI变换下评估S1，主结果不使用2%连通域面积删除或DEV调出来的清理规则。

| 名称 | 特征 | head / 作用 |
| --- | --- | --- |
| SEG-R | 发布encoder最后层 + 现有S1权重 | 历史参考，修正ROI裁剪后重评；不冒充新decoder的等容量对照 |
| SEG-L | 发布encoder block23的同一特征复制到四输入 | 新的同一分割decoder，冻结encoder训练；检验decoder能力 |
| SEG-M | 发布encoder最后四层blocks20/21/22/23 | 与SEG-L相同decoder/初始化/预算；检验增加层间特征的作用 |

按官方语义分割路径，最后四层不是广泛间隔的层[5,11,17,23]；后者若将来尝试，要列为不同配置。VGGT本机API应显式调用 `aggregator.patch_embed.get_intermediate_layers(x, n=[20,21,22,23], norm=True, reshape=True)`，注意本实现的`n`是keyword-only，自动去掉CLS/storage tokens。所有图像归一化、norm与位置编码沿发布encoder，不硬编码DINOv2的14px patch到VGGT的16px patch。

冻结encoder时可独立缓存这四层特征，记录层号/shape/dtype/权重/预处理哈希与空间开销。历史S1 cache只有最终层，不能从中重建另外三层；只允许SEG-L显式复制最后层，SEG-M必须实际提取四层。最后层应与发布encoder的`x_norm_patchtokens`在线输出对拍。

新decoder采用 **Mask2Former轻量变体**：四路1024维特征先投影到256维，经4/2/1/0.5特征金字塔；pixel decoder 3层、transformer decoder 3层、8个attention heads、16queries，二类语义+no-object。该尺寸是本轮资源受限候选，非官方1024维/200queries/9层decoder配置的复现。若远端框架不支持这个形状，应先报实现差异和冻结新配置，不静默换成U-Net/DPT后仍使用原实验名。

Mask2Former采用class CE / mask sigmoid BCE / Dice权重2/5/5，按相同框架的匹配与语义输出定义实现；SEG-L/M损失完全相同。padding标签为已知背景且参与监督，非padding的未知标签使用ignore；不把未知像素当背景。以二类semantic argmax得到mask，不用DEV挑阈值。若未来增加boundary loss、RGB skip支路或深度输入，另做独立消融。

## 4. 训练顺序：head先收敛，再考虑encoder

先做seed17的SEG-L/M head训练：encoder严格冻结，AdamW head peakLR1e-4、WD0.01、norm/bias不衰减、warmup500、cosine预设总长18,000、floor0.05、EMA0.995、clip10；最少6,000、最多18,000更新。两head从同seed相同decoder参数出发，计算路径不同但数据/曝光、前处理、损失和优化器相同。

每500步用固定TRAIN probes评估EMA完整mask loss、鼠体IoU与归一化边界F1。plateau定义：最近5检查点中 `max(0, first_loss-min(loss))/max(first_loss,1e-12)<1%`，IoU与边界F1跨度均<=0.2pp，且step>=6,000。**共同终点**是两臂都满足plateau的首次共同检查步，否则到18,000；不能一臂早停后拿不同曝光当纯特征比较。未收敛臂明确标注；两臂均未拟合时先诊断实现/标签，不立即微调整个encoder。

DEV只在6,000、12,000、18,000及共同终点（若不同）作固定评价，不驱动提前终止或选择最佳checkpoint。seed17完成后，候选若未触发退化且有预登记信号，再完成seed29/43同样配对；不只保留最优seed。SEG-R单列参考，不以它与新head的差值单独归因于中间层特征。

若SEG-M完成head收敛仍需要检验encoder适配，另做SEG-F/U：同一SEG-M EMA终点，两臂各6,000步；F继续冻结，U按最后1→2→4blocks、head3e-5、末层1e-6、每向前一层乘0.65、每次新启用层warmup200的日程，复用[关键点计划的数值与resume约定](head_first_progressive_encoder_route_20261024.md#4-fu用相同-head-终点做等预算配对)。只是复用协议，分割与keypoints的权重、优化器、sampler、run目录和结果完全独立，首轮不做联合多任务学习。U在线重算活动特征，旧冻结cache不能用于训练encoder；冻结部分和默认geometry模型不修改。

## 5. 真正使用Depth Anything encoder的附加对照

远端若已具备或能准备官方Depth Anything ViT-L预训练encoder，记录确切版本/权重SHA256，可增加`SEG-DA`。使用与VGGT候选相同的二类decoder，重建共同560px full-image letterbox配对（560同时整除14和16），相同图像/标签、损失和head预算；VGGT也必须在560重新训练，不能拿512结果直接声称纯encoder权重比较。最后四层特征提取各自遵循其模型API并记录token/grid/归一化。

这个比较区分不同预训练encoder方案，但VGGT图像encoder与DA编码器的patch/预训练/结构仍不同，不能完全归因于depth pretraining。Depth Anything的深度decoder无需训练，也不加载城市标签head后直接宣布会分割老鼠。权重或环境缺失不阻塞SEG-L/M和keypoints；本提交不下载新权重或安装mmseg。

## 6. 评价与通过条件

分割主评价独立于几何求解器：foreground IoU、Dice、native1px boundaryF1（保留历史口径），以及原图长边统一512后的1px/2px boundaryF1。不要用占比很大的背景正确率代替鼠体分割精度，也不把原生1px和归一化2px混作同一指标。按真实/合成、底视/其他、session/rig报告分布、空mask/全前景、padding原始前景像素和阈值固定结果。

另外报告尾巴/四肢错误、凸包偏差与离主体碎块，只有独立可用的part标注才量化part召回，否则保存固定TRAIN/DEV面板作定性审计。不给GT点作推理提示，不用面积删除掩盖细部漏检。

预登记筛选信号：SEG-M对SEG-L，真实底视归一化512空间2px边界F1均值提高>=2pp；四个域/层分层IoU均值均不得下降>0.5pp，边界F1均值均不得下降>0.5pp，每层空mask比例<=2%。逐seed配对报告，三seed均值达到收益门槛、至少2/3改善且没有任何seed超过退化门槛，才认为多层特征候选有益。没有收益也完整交付，不因此自动搜层或加大encoder LR。SEG-F/U沿用同一分割指标，不使用关键点PCK门槛。

Pano的proxy IoU、定性图、下游几何只作额外诊断，不能越过缺独立mask真值这一限制。几何测试必须使用另一个固定发布geometry模型，按新mask输入影响作报告，求解器不收敛不能直接判定segmentation差，IoU通过也不等于相机姿态已通过。所有reused DEV仍是开发证据，不开启sealed。

## 7. 远端交付

独立source/config/输入回执与四层feature shape记录；SEG-R统一ROI参考；SEG-L/M完整训练和固定终点；stage/参数组/cache/梯度/恢复测试；raw/EMA mask概率图与二值mask、每图指标、边界及细部面板、失败分母、compute/曝光与3seed配对总结。实际DA权重对照若缺输入，单列missing而不是阻塞主线。

可与关键点开发并行；GPU作业按远端配额串行或合理分配，二者不互相覆盖、不争用旧run锁、不自动恢复暂停automation。新渲染到达后仍先pin新的mask定义与split，不能改变运行中的训练清单。
