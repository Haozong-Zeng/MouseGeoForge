# 渲染完成前的主线：真实关键点 head 收敛，再渐进解冻图像编码器

日期：2026-10-24（America/Los_Angeles）。这是用户要求发布给远端的新路线；本次本机操作只更新源代码、证据与计划，没有提交 Slurm、启动训练或渲染。机器可读规格为 [head_first_progressive_encoder_20261024.json](../configs/experiments/head_first_progressive_encoder_20261024.json)，远端交接入口为 [执行清单](remote_head_first_handoff_20261024.md)。规格是 `design_only`，不是现有训练器可直接读取的运行配置。

同日新增用户要求：[借鉴Depth Anything的独立鼠体分割支线](mouse_semseg_depth_anything_route_20261024.md)。分割与keypoints分别训练、存盘、评价，不以联合多任务或共享可写encoder作为首轮前提；二者均可在工作站新渲染到达前推进。

本路线将真实 2D 关键点学习前置，不再要求先通过未知相机几何实验才能训练独立 2D head。旧几何 checkpoint 的失败结论、几何晋升门槛、封存划分保持有效；这些几何门槛不作为本次独立关键点支线的启动条件。已有 H100 文档中的“先完成 V5 / L0 / P0/P1，再开始所有关键点工作”顺序在这条支线上由本文替代。旧文中的几何 P0/P1 与本文的 F/U 对照是不同实验。

## 1. 决策依据与 V5 的位置

PanoMouse 已完成的不只是零训练检验。历史关键点实验 A/B/C 各有 seed17/29/43 三个最终模型，每个 6,000 更新；本次从 `eval_rows.json` 重算底视爪指标，并核对汇总、终点与划分。[紧凑证据及文件哈希](experiments/panomouse_head_first_evidence_20261024.json)随 Git 发布，原始图片、权重和大输出不进 Git。

| 三种子均值 | A：只用 PyMouseStudio | B：加入 PanoMouse 五点 | C：另加 24 辅助点 |
| --- | ---: | ---: | ---: |
| 本域 DEV 底视爪 PCK@0.05 | 97.99% | 98.94% | 99.05% |
| PanoMouse DEV 底视爪 PCK@0.05 | 30.07% | 92.65% | 92.19% |
| PanoMouse DEV 底视爪左右混淆代理率 | 34.47% | 5.39% | 6.10% |

PCK 阈值为标注框最长边的 5%；左右代理率指预测点更接近对侧标签，不是整对互换事件率。三种子均值不是置信区间。B/C 每更新额外使用 8 张 PanoMouse 图，本域仍为 8 张，所以历史 A/B 不是等算力比较。后续 F/U 都沿用 B 的同一数据、头、批量与损失，C 不自动扩展。

历史结果使用真值/标注框 top-down 裁剪，只说明给定鼠框后的定位能力；未证明自动检测、遮挡恢复或物理 3D 精度。历史 head 的 6,000 更新也不能直接当成“已收敛”回执。

PanoMouse 原图的下层相对旋转误差中位 80.61°，S1 自动掩码去背景后 108.23°；其背景已是不对称实验室。该结果说明未知底视相机仍有困难，但不足以认定非对称环境就是充分解法；掩码误差、鼠体占比、跨视角重叠和透视方向仍是待区分因素。PanoMouse 只有一套物理 rig，DEV S03/S11 与部分 TRAIN 共用标定时期，不能写成未见物理装置认证。

因此，**本路线不等待本机新渲染，也不将 V5 全量数据作为前置依赖**。本机正在进行或将来的渲染由工作站管理；本文不变更其进程、STOP、时钟或计划。V5 仍可作为未知相机姿态/环境因果研究的后续独立数据版本。用户提供的[原V5草案](bottom_geometry_v5_data_revision_20261024.md)原样保留作来源，新路线与其不同之处以本文为准。新渲染到达后先审计、冻结清单，不能中途混入正在进行的 F/U。V5 草案中“发布模型必须继续失败”不再是准入条件；零训练成功也应如实报告，不反向挑环境维持失败。

## 2. 先明确解冻的是哪段网络

历史关键点特征流是：

`RGB crop -> aggregator.patch_embed (24-block DINO image encoder) -> x_norm_patchtokens -> heatmap head (+ RGB branch)`。

这里 `aggregator.patch_embed` 是完整 DINO 图像编码器，内部的 `patch_embed` 才是最初的图像 patch 投影。关键点 head **没有经过** `aggregator.frame_blocks` 或 `aggregator.inter_frame_blocks`；只解冻这两组聚合层，对当前关键点损失不会有梯度。

本次精确解冻目标为 `aggregator.patch_embed.blocks.23`，随后 `.22`，再 `.21/.20`；最后归一化 `aggregator.patch_embed.norm` 从第一阶段一起微调。内部 patch 投影、特殊 token、其余 blocks、RoPE 及所有跨视图聚合层/相机头/深度头均冻结。实现前必须打印参数路径、数量和真实反传梯度验证这一拓扑，不能仅凭模块名猜测。

这是 Lightning Pose 思路的项目化扩展。核查的官方版本为 `0b5ebf447bf551e0a2922ad6a0f17fddbe6ea711`：其 [UnfreezeBackbone](https://github.com/paninski-lab/lightning-pose/blob/0b5ebf447bf551e0a2922ad6a0f17fddbe6ea711/lightning_pose/callbacks.py#L79)先令 backbone LR 为零，到指定 epoch/step 后以 head LR 的一部分开始预热；[默认配置](https://github.com/paninski-lab/lightning-pose/blob/0b5ebf447bf551e0a2922ad6a0f17fddbe6ea711/scripts/configs/config_default.yaml#L58)在 epoch20 开始启用。它不是本计划的逐层协议，也不证明这些学习率已对 VGGT 有效。我们保留 head-first 和预热，另加按深度递减的极低 LR，并始终将 encoder LR 限制在 head 以下，不使用其最终追齐 head LR 的行为。

## 3. 数据、任务与收敛检查

沿用 B 的五点定义：nose、L/Rhand、`midpoint(Lheel,Lfoot)` 两个派生后爪中心。后爪中点与本项目足底中心只是近似语义，需在 TRAIN 细粒度点/图上复核并记录；不要改名当作精确同义点。沿用 `flag==2` 和 stage0 TRAIN 重投影 P99=81.485667 px 过滤；此过滤使用发布 3D，只是标签一致性，不是独立 3D 验收。

- 本域 TRAIN：AZL3/AZR2/CR2/EL2/EL2L2/ER3L2；DEV：AZR2L2/FR2/FR3。
- Pano TRAIN：S01/S02/S06/S07/S08/S09/S10；DEV：S03/S11；S04/S05 整个 E3 封存。
- 本域有效 TRAIN 889 图、DEV 441 图；Pano TRAIN 5,087 图、DEV 1,150 图。这是历史 items 的数量，远端须核对清单哈希，不按图片总数推断独立样本数。
- 每更新本域 8 + Pano 8 个 view，两个域各自已知点平均 CE 后相加，与 B 一致；记录分层曝光、丢标签与底视/非底视统计。不得悄悄改变域权重、加 teacher/几何损失、换 C 或加入伪标签。
- 512 方形裁剪（margin1.3）、128 热图、sigma2cells；RGB 支路、解码和像素中心变换保持历史 B。首轮不引入新的增强或 flip；以后单独对照。所有配方/质量阈值仅用 TRAIN 确定。
- `stage0/admission.json` 仍写 evaluation-only / train_allowed=false，而 stage2 已有实际训练产物。远端需建立本路线新的任务专用准入回执，引用本次用户要求、精确 split、语义审计和非商业标记；不改写历史 receipt，也不能把它描述为全部任务均已准入。视频同步未核实不阻断单图 2D，但会阻断视频多视角/时序约束的进一步使用。

### H0：先让同一 head 充分训练

从固定发布图像编码器和随机初始化 B 类五点 head 开始，每 seed 最少 6,000、最多 18,000 更新，AdamW peak LR3e-4、WD0.01、warmup200、EMA0.995；初始总长度固定18,000的 cosine，最低倍率0.05，不在看到 DEV 后重设曲线。每500更新用固定 TRAIN probe 原样无增强评估 EMA：记录 CE、底视/其他视角爪 PCK/混淆代理率。外层 DEV 不参与收敛判定。

预先固定以下训练收敛判据：step>=6,000，最近5个检查点（跨度2,000更新）里固定 TRAIN CE 的 best 相对下降<1%，底视爪 PCK 的跨度<=0.2pp，混淆代理率的跨度<=0.2pp；CE相对下降精确定义为 `max(0, first_CE-min(window_CE))/max(first_CE,1e-12)`，跨度为窗口max-min。首次满足即保存 H0 endpoint 和原始序列。该判据只表示优化平台，不等于泛化合格；另报告本域/Pano各自拟合质量与显著未拟合点。到18,000仍未满足则标为 head_not_converged，不自动解冻，不靠换 seed 或追选 DEV 最佳点补救。

现有 B_seed17/29/43 是重现参考。只有完整训练状态、清单/语义/实现完全匹配且能重建收敛序列时才能续用；历史 `final.pt` 只有 head/EMA，没有 AdamW/RNG/精确采样进度，不可伪装成精确续训。可作带明确标记的初始化工程试验；正式 H0 从固定发布权重和随机 head 重建。

## 4. F/U：用相同 head 终点做等预算配对

同一 seed 的两个分支均从相同 H0 EMA head + 发布 encoder 开始，分支时均重建 AdamW/EMA；不能一边承接旧 momentum、另一边重置。两边各6,000更新、相同数据顺序/批量/损失/head LR、crop、数值模式和固定终点评价。

| 分支/步数 | 可训练图像 encoder blocks | LR 峰值上限 |
| --- | --- | --- |
| F，全程 | 无；只继续训练同一 head | head3e-5 |
| U，1–2,000 | block23 + final norm | head3e-5；block23/norm1e-6 |
| U，2,001–4,000 | block22–23 + norm | block22 6.5e-7；block23/norm1e-6 |
| U，4,001–6,000 | block20–23 + norm | block21 4.225e-7；block20 2.74625e-7 |

对已开放层 l，`lr_peak(l)=1e-6*0.65**(23-l)`。越靠近输出，LR越高；更早的20层保持冻结。两分支 head 用200更新 warmup+6,000总长 cosine 到0.1倍；encoder 每层启用后独立200更新线性预热，再乘同一全局 cosine 包络，禁止后期追齐 head LR。Norm/bias 不做 weight decay，其余 trainable 参数WD0.01。LR/WD是本轮固定候选，不是已验证最优值。

采用**固定的逐阶段解冻日程**，不因 DEV 好坏或单 seed 分数临时多解冻几层。seed17先走完整 pair；若无数值/完整性问题，则seed29/43完成相同pair。收益结论以三个配对seed为依据，不只报胜出者。若seed17已触发预登记退化门槛，记录失败并停止扩大，而不是调LR重看同一DEV。

### 实现中必须检查的事项

1. H0/F 可复用冻结最终 token cache。U 必须对当前图像在线重算 trainable encoder；不得继续读取旧最终 token cache 后宣布 encoder 已微调。若优化冻结前缀，只能缓存严格冻结边界的 activation，并与在线路径对拍。
2. 不得把整个图像编码器包在 `no_grad/inference_mode` 下。冻结早层可 no_grad 后 detach，活动末层须建图；autocast 可用，`eval()` 不等于不建图。F/U采用相同eval forward数值模式，避免把dropout变化当解冻收益。
3. 全部可能启用的参数组可预先登记，尚未解冻的参数必须 requires_grad=false、grad=None；只置LR0不够。每阶段保存全部 trainable 名称、参数量、LR、梯度范数、实际更新范数、相对发布权重漂移；检查未开放参数逐字节不变、无重复/漏组。
4. checkpoint 保存 raw/EMA、AdamW、scheduler、AMP scaler、所有RNG、sampler/已提交曝光、global_step、stage、requires_grad与named参数组映射。跨2,000/4,000边界的中断恢复需对拍下一更新；不能只恢复optimizer组而把参数重新冻结。
5. 同一分支的gradient accumulation与物理batch对拍；梯度clipping保持10并报告实际裁剪率。数值非有限即停机并保存失败，不自动提高clip或LR。
6. 在64个固定TRAIN图像（两域、两层、各16）上做短工程反传，证明活动blocks与head有非零梯度，冻结模块没有梯度；工程更新丢弃，正式pair重新从同一H0 endpoint开始。全发布模型不作为可写的共享对象。

## 5. 评价、退化门槛与自动定位

两分支保存2,000/4,000/6,000阶段终点；**主比较为固定6,000终点EMA，raw并报**。外层DEV只在这些预定时点评价，不驱动解冻/早停/最佳模型挑选。分别报告两域、每session、底视/其他视角、前/后爪；统计单位按同步组/session汇总，随机seed只表示优化重复，不能把同一rig的多视图当作独立装置。用相同组的配对差异和分组bootstrap；仅两Pano DEV session的区间不应解释为装置级泛化区间。

固定暂定筛选门槛（在看到本轮结果之前登记）：U相对F，Pano底视爪PCK@0.05提高>=1pp，或左右混淆代理率相对下降>=15%且PCK不降；同时两域底视/非底视爪PCK均不得下降>0.5pp、左右代理率不得增加>0.5pp。逐seed报告，三seed均值达收益门槛且至少2/3方向一致才可称候选有益；任何单seed触发>0.5pp退化则不晋升，不能以平均值掩盖。若无额外收益，F/H0就是优先结果，不继续解冻更深层搜分。

2D分支有独立encoder副本；即使2D通过，也不替换已发布VGGT的geometry encoder。若未来要复用同一adapted encoder作相机/深度推理，需另跑冻结v4 geometry retention并通过原门槛；相机头/深度头参数冻结不保证其输出不变。本路线不重启旧几何P0/P1、不做8sealed rig或S04/S05模型评分。

在固定模型比较后，再做自动框的部署核验：GT框top-down保留为可比上界；自动检测/分割框只在TRAIN校准并固定，定位失败保留在分母，不能丢难例。先报告框覆盖/漏检、再报告端到端2D、最后在已知K/E下做三角化及留一视角重投影。Pano发布3D与同批2D相关，不宣称独立毫米精度；未知相机训练与视频同步/时序损失留作不同协议。

## 6. 远端在等待渲染期间的工作顺序

1. 接收本次Git提交，生成输入可用性/哈希、数据准入与模块拓扑回执；检查数据盘output软链接、封存排除和缓存。缺数据就列精确文件/哈希，先做CPU实现与测试，不扩大读取范围。
2. 实现独立的新head-first训练器，复现历史B的确定性特征/热图/像素约定；完成冻结/解冻、batch、checkpoint/resume的CPU和短GPU工程检查，工程更新丢弃。
3. H0 seed17达到训练收敛或上限；随后同一H0端点的F/U seed17完整配对。基于实测更新/存盘/评价耗时制定有界Slurm资源与walltime，在每个作业开始前写入run protocol；不编造吞吐/预算。
4. 没有失败/退化停止条件时完成seed29/43的同样H0与配对；发布完整终点、缺失/失败、曝光和paired metrics。
5. 本地新渲染到达后只做新数据QA/pins，冻结中的F/U不改样本池。若仍需要未知相机训练，再独立登记真实Pano标定监督或小规模V5环境对照；不自动排队全量V5几何训练。

历史`v3-12`/`pymousestudio`暂停自动化不恢复。本机这次只发布计划和可审查源文件；训练结果只能由远端实际运行后的回执证明。
