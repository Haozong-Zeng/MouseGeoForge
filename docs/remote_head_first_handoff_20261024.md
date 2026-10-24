# 远端交接：渲染到达前先推进 head 收敛与分层解冻

先读 [完整路线](head_first_progressive_encoder_route_20261024.md)、[机器可读规格](../configs/experiments/head_first_progressive_encoder_20261024.json) 和 [PanoMouse历史证据](experiments/panomouse_head_first_evidence_20261024.json)。本页替代旧交接中“所有关键点工作必须等待V5几何成功”的顺序；不变更旧几何checkpoint失败、TEST/sealed和数据隔离规则。

## 给远端执行代理的任务

本地新渲染不应阻塞本路线。基于现有已审核PyMouseStudio/PanoMouse静态关键点训练图，先实现并充分训练B类五点heatmap head。训练收敛后，从同一H0 EMA终点做两条各6,000更新的matched continuation：F只继续训练head；U从DINO图像编码器末端逐步解冻1→2→4个block，head LR3e-5、末层1e-6、每向前一层乘0.65，其他层冻结。严格按完整文档的收敛判据、固定日程、seed和停止门槛执行。不要直接执行历史脚本当作新协议。

`tools/panomouse_stage2_kp_20261023.py`是历史B/C实现参考，保持原字节发布；它依赖本机历史模块/路径，不是本次可直接运行的新训练器。新的入口应独立命名，禁止覆盖历史产物。机器可读JSON是设计规格，不可直接传入旧v4训练器或历史stage2入口。

最重要的拓扑：head吃的是`aggregator.patch_embed(...)["x_norm_patchtokens"]`。因此要解冻`aggregator.patch_embed.blocks.{20,21,22,23}`与final norm；当前head不经过`aggregator.frame_blocks/inter_frame_blocks`，解冻后两者不能训练这个head。U需要在线当前encoder特征，不得沿用冻结最终token缓存。

## 输入准备回执

以下是文件角色，不是上传到Git的数据：

| 输入 | 需核对内容 |
| --- | --- |
| `vggt-omega/vggt_omega_1b_512.pt` | 发布权重SHA256 `c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934` |
| `dataset/PanoMouse/raw/annotations/{annotations_2d.json,annotations_3d.json,calib_data.json}` | 与证据内stage0哈希一致；只materialize本路线TRAIN/DEV items，不读取S04/S05图像进行推理 |
| `dataset/PanoMouse/unpacked/images/` | 按冻结items解析存在性/大小；精确文件清单由远端产生并pin，视频不需要 |
| `dataset/PyMouseStudio/annotation_versions/20261003_effective_v1/manifest.json` | 有效五点、已确认掩码和源RGB；沿历史OWN_TRAIN/OWN_DEV角色，TEST排除 |
| 历史 `output/panomouse_stage2_kp_20261023/{items,protocol,summary,eval_rows}.json` | 有则对拍哈希；没有可先实现CPU逻辑，由证据核对历史数值。没有原items/原数据时不能宣称复现完成 |
| 新任务准入回执 | 修复“stage0仍evaluation-only但stage2已有训练”的登记缺口；记录用户本次新路线要求、语义复核、分割、非商业标识，保留旧receipt不修改 |
| 数据盘 `output/` | 确认真实resolved路径/可用空间、独立run目录；不得回填历史输出或把大文件写到Git |

发布的原始v3最终渲染器及其导入依赖，仅供后续工作站数据代码衔接；本路线不依赖`.blend`或新v5数据。`replay_bottom_geometry_v3_render_design_20261024.py`需要原v3配置与回执，交接小包提供这些；缺少该小包不阻断关键点源代码实现。

## 按顺序交付的东西

1. `input_receipt.json`：exact source/weight/data pins、角色与排除、语义、计数；`parameter_topology.json`：特征流、block名字和参数量。
2. 新训练器及测试：H0/F/U配置解析；LR层级与阶段边界；缺失点/归约；冻结参数不变；真正活动层反传；U禁用旧最终特征cache；含阶段边界的完整resume。短GPU预检的更新丢弃。
3. H0收敛序列和endpoint，或head_not_converged回执。不能把历史6,000步文件直接叫收敛证明。
4. F/U同seed完整终点与等曝光记录、每阶段raw/EMA、固定DEV分层结果。先seed17，按已登记停止条件决定是否扩展完整seed29/43；不按分数单独挑seed。
5. 总结：head alone是否充分、渐进解冻是否提供额外收益、哪里退化、是否值得自动框/已标定3D闭环。不要把2D分支通过描述成未知rig相机或geometry模型通过。

若输入不全，返回“缺哪个文件、预期哈希/角色、哪个阶段受阻”，其余CPU开发继续；不要求上传源`.blend`、纹理、受保护TEST或sealed图像。若本地渲染先完成，先单独审计新清单，本轮F/U数据版本仍固定。

这次本机发布没有提交Slurm或直接向远端shell/聊天发消息。远端收到本Git提交后按本任务范围和其本地资源规则推进；原暂停自动化保持暂停。
