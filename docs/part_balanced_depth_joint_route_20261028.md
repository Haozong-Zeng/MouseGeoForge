# 2026-10-28：部位平衡深度监督与共享编码器双头训练

用户澄清的 **keypoint semantic loss 是空间/部位平衡深度监督**：鼻尖、爪、尾等
小区域不应被大量躯干像素淹没。它不是把内部关节Z当表面深度，也不等同于多加一个
关键点头。本路线把这个问题与“同一encoder同时保留depth和keypoint能力”分开检验。

状态：源码组件、CPU测试、设计检查入口已实现；**没有启动训练、Slurm、sealed评分或
旧自动化**。GitHub `h100-branch` 的 `b7721cd` 尚不含云端未提交的
`tools/depth_adapt_20261026.py`、`tools/own_rig_hull_20261026.py` 及其新计划文件。
因此不能声称已经修改/验证那个云端训练器；对接在云端完成，保留其现有采样和loss口径。
具体交接见 [cloud handoff](remote_part_balanced_joint_handoff_20261028.md)。

## 现有证据和本次问题

用户提供的2026-10-28云端汇总报告：关键点flat1e-4全encoder适配在三个种子上达到
约98%底视爪PCK；own-rig hull深度监督有效，anchor0.3+plane0.3轻配方已有三种子。
这些是报告中的DEV结果，不是本地重新评分。重配方最终reseeds与保存权重状态必须
从云端实际DONE/endpoint核实，不把报告中的预计完成时间当成已完成。

提高整个mask/hull权重不等于按部位平衡。当前鼻/尾比大部分躯干困难，整鼠平均AbsRel
无法单独判断小部位有没有学好。旧J-off/J-on局部feature bridge实验也不是这个消融。
动态GS教师单列未来教师质量实验，不与本轮同时更换。

## A：小区域的空间loss预算

以可见TRAIN二维点周围的截断Gaussian软ROI为保守proxy，与目标depth-valid、已登记
可信度相交。只有独立定义的synthetic part mask才能称部位分割。ROI半径固定为同
坐标系bbox长边的0.06；首轮nose、LF/RF/LH/RH预算为1/.5/.5/.5/.5。
耳/尾只有在可靠可见TRAIN语义及映射登记后加入，不凭Pano五点虚构标签。

```
r[k,p] = ROI[k,p] * valid[p] * confidence[p]
q[p] = sum_k(normalized_part_budget[k] * r[k,p] / sum_p(r[k,p]))
boost[p] = min(1 + strength * sum(confidence_valid) * q[p] / confidence[p], 4)
```

每个ROI先按其有效软面积归一化。小于8有效像素或软质量小于2不激活；未知、遮挡、
越界点不参与。缺部位不删除整张图，逐来源/视角/部位报告覆盖分母。
ROI、confidence和点坐标均detach；不能用学生当前mask逃避监督。有效误差非有限值
应报错，未知padding不进入loss。不要对每个部位独立拟合尺度。

### 云端SILog的精确对接

**保留原全局SILog、metric anchor、mask BCE、plane及source权重。** 全局SILog可能含
均值/方差/平方根，不能草率改为区域L1并称“只改权重”。首轮云端模式固定为
`bounded_focus_auxiliary`：在同一个原全支持gauge下构造非负log-depth残差SmoothL1
（beta1），加入返回的 `focus_loss`：

```
focus = mean_over_valid_views(sum(error * confidence_valid * (boost-1))
                              / sum(confidence_valid))
L_depth_task = L_original_cloud_depth_mask_anchor_plane + focus
```

strength0.5已包含在boost内，不再乘一次。OFF strength0精确返回0。cap约束新增项
相对于error-map均值的像素系数（最多3倍新增），**不是整个SILog/plane/网络参数梯度
的上限**。保留原global clipping，记录实际梯度与cap饱和率。无ROI时focus为0，原
depth样本正常训练。模块另返回归一化加权均值 `loss`，仅用于原来就是可分解pixel
均值的调用方，不能误当云端SILog整体替代。

```python
import torch
import torch.nn.functional as F
from mousegeoforge.part_balanced_depth import keypoint_regions, part_balanced_depth

# All tensors/points/boxes share the registered depth-map crop coordinates.
# Apply any parent scale alignment ONCE on its original full valid support.
safe_target = torch.where(valid, target_z, torch.ones_like(target_z))
safe_pred = torch.where(valid, pred_z, torch.ones_like(pred_z))
error = F.smooth_l1_loss(safe_pred.clamp_min(1e-6).log(),
                       safe_target.clamp_min(1e-6).log(), reduction='none', beta=1.)
regions = keypoint_regions(train_xy, train_visible, bbox, pred_z.shape[-2:],
                          radius_fraction=.06)
balanced = part_balanced_depth(error, valid, regions, confidence=target_confidence,
    part_weights=[1,.5,.5,.5,.5], strength=.5 if arm_part_on else 0,
    max_pixel_boost=4, min_region_pixels=8, min_region_mass=2)
loss = original_cloud_task_loss + balanced['focus_loss']
# Preserve parent per-domain averaging; apply source weight exactly once.
```

无效target必须在取log前替换，不能先算NaN再mask。关节点只划定图像邻域，不生成
joint-Z target。底视反光和错误外壳深度不能通过提高权重变成真值。评估用205个own
标注时刻不可招募为训练ROI/教师拟合数据；只join独立准入TRAIN。

## B：共享image encoder + depth/mask、keypoint两头

从release `aggregator.patch_embed` 初始化同一个24-block encoder，depth取5/11/17/23，
keypoint取最终norm patch map。不训练无关的frame/inter-frame blocks或camera head。
两头各用其在同一release encoder上收敛的H/H0 EMA初始化；只有UX适配后head时，先
在release冻结条件下做共同head兼容拟合，保存固定初始化，不混用不匹配head。

`SharedPatchTwoHeads` 每个RGB batch只遍历一次encoder，`task='both'`输出两头。
depth adapter接口 `(four_feature_maps, rgb)`，KP接口 `(last_map, rgb)`。
**四层norm、RGB normalization和decoder参数键须与云端原实现逐项核对**；默认
normalized taps不是已经证明与云端DPT等价，先做独立parent输出parity，再优化。

联合每个optimizer更新含一个depth和一个KP microbatch，累积后只step一次。两batch
可以来自不同已准入TRAIN源；在线编码，无最终token cache。encoder保持eval forward
但autograd在线，flat1e-4及原protected ramp保留。head LR、部分点CE、domain平均、
augmentation、EMA沿原协议。共享参数只在optimizer中出现一次。
先用depth1/KP1，在固定TRAIN预检测任务encoder梯度；若需一个固定任务系数，须在
J0/J1任一正式结果之前同时登记，不按DEV调参。本轮不同时加入PCGrad、feature bridge
或surface-to-joint offset。

`joint_train_step` 接收原loss callbacks，保留depth的mask/anchor/plane及KP缺失标签
口径。断点包含raw+EMA encoder/两头、Adam、Python/NumPy/Torch/CUDA RNG、两个
sampler状态/offset及code/data/release/heads/recipe SHA256。只在完整成功更新后写
原子checkpoint；optimizer突发错误的部分更新不能保存为可续训终点。

## 四臂与评价

| Arm | Depth部位focus | KP共同训练 | 隔离的问题 |
| --- | --- | --- | --- |
| D0 | OFF | OFF | 当前depth配方重放 |
| D1 | ON | OFF | 小部位loss预算是否有效 |
| J0 | OFF | ON | 共享encoder能否兼顾两任务 |
| J1 | ON | ON | 联合条件下部位预算是否仍有效 |

每臂12000个depth更新；J臂另含12000个KP microbatch，不减少/翻倍depth曝光。
同seed共享depth初始化/顺序，J0/J1另共享KP初始化/顺序。样本曝光、更新数、GPU时间
分别报告，等更新不等于等算力。先固定seed17完整对照，再按预登记规则确认29/43，
不用best step或单独早停一臂。

主评价：nose/每只paw/有标签tail/躯干的metric median、P90、AbsRel和完整支持数。
whole-body raw/aligned、synthetic unseen rigs、DANNCE、own bottom/sides、mask、
cross-view front violation和固定PLY视觉检查保留。用于训练ROI的点不能兼作独立评价。
DEV沿云端冻结划分，12 synthetic DEV rigs只评估，sealed/protected TEST不评分。

联合原目标：Pano bottom-paw PCK>=97.5，同时own DEV bottom aligned AbsRel<=.016，
保留原domain/view keypoint retention及synthetic20% guard。raw metric scale单列，
不能用oracle对齐替代部署精度。~.001差异看配对seed/session，不夸大单seed结果。
跨seed不是跨rig证据，无dense真实surface GT的限制仍在。

## 实现边界

已实现软ROI/有限区域预算/positive focus/覆盖诊断、共享patch两头、累积优化、完整
checkpoint/EMA/RNG恢复及显式cloud factory的TRAIN parity/梯度/一步丢弃预检。
CPU fixtures使用真实DINO transformer小模型验证梯度和精确resume。

**尚未实现/验证**云端未提交trainer的实际adapter、真实H/H0加载parity、H100预检、
12k收敛、三seed收益、完整推理入口和sealed确认。组件不是完整云端trainer的替代品。
先完成云端对接和预检，再按用户/资源执行规则运行；发布不会提交Slurm作业。

本地验证（2026-10-28）：11项loss测试、7项共享encoder/优化/完整resume/CLI绑定测试通过；
设计检查通过，source-only快照检查无缺失、大文件或启发式secret命中。仅CPU小模型/夹具，
没有真实数据训练或H100验证；原工作区未提交改动保持原样。
