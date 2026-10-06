# 阶段36：候选关系中的显式相对几何开/关

05:08 UTC结果：完整性passed、ON初始1.788e-7，几何参数L2更新.402824，
但双鼠43→42TP、全FP155→198、重复33→40，未通过，两臂四帧已视觉查看。
阶段37上限诊断发现原query与中心候选互补，下一pool对照见
`docs/roi_candidate_pool_plan_20260912.md`，不再加几何评分模块。

2026-09-12 04:34 UTC预注册，代码/5项新测试完成，全套262通过。

04:36 UTC GPU空闲后launch PID285453，几何OFF→ON顺序，先冻结特征/输出
核验和梯度预检才优化；新目录`output/roi_geometric_objectness_pair_v1`。

## 阶段35结果

独立/关系两臂各3200，136.399/131.536秒，完整性passed、原模型/框不变，
共同初始state相同；初始OFF88评分0差、ON最大变化.00016442。
双鼠37TP6FP→42TP10FP，全开发75TP34FP→113TP78FP，框重复4→15；相对
旧maxIoU质量头的157TP253FP，一对一监督减少过检，但关系增加召回又增FP，
未通过预设保留标准。训练单鼠842TP90FP→900TP58FP/964GT，双鼠均63/64，
ON训练objectness MAE.03089而开发.17721，存在拟合/泛化差距。
两臂固定四帧已视觉检查，296均只检出黑鼠：OFF棕鼠score.17334、ON .00208。
不能用降低阈值恢复它后宣称成功；不把提高召回而增加重复当合格结果。

每800步loss OFF.06706/.04776/.04376/.04124，ON.04468/.03050/.02768/.02613。
训练已经拟合大多数，当前不优先继续同一结构加步数。

## 研究与假设

[Relation Networks for Object Detection](https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Relation_Networks_for_CVPR_2018_paper.html)
和[官方项目](https://github.com/msracver/Relation-Networks-for-Object-Detection)
同时建模外观和相对几何。现有候选self-attention仅把绝对框坐标混在ROI特征
MLP里，未显式提供两框重叠/相对尺寸。假设一个小的几何注意力偏置能帮助
区分重复候选与两个不同目标；不假设它一定降低FP，也不复现论文全部实现。

## 固定方案

- 两臂从同一个阶段35关系ON3200权重开始，原VGGT、16姿态中心头、top6框、
  ROI描述子、364train/88dev、768与1–4视角、二分固定一对一binary BCE不变。
  两臂都有原关系self-attention，开关只控制几何偏置。
- 每候选对(i,j)特征：box IoU、(cxj-cxi)/wi、(cyj-cyi)/hi、log(wj/wi)、
  log(hj/hi)、原center score差。位移/logratio裁[-4,4]，IoU裁[0,1]防浮点
  舍入超界。全部仅来自预测框/分数，不用GT。单视角内交互，不跨视角/时间。
- 6→4无bias线性层，24个参数、全零初始化；2*tanh限制4个attention head
  偏置在[-2,2]，不产生硬屏蔽/NMS，不移动框、不假设动物互斥或强制数量。
  无候选索引/ID，保持置换等变；相同候选仍对称，不保证唯一性。
- OFF跳过几何层，优化原49985参数；ON优化50009参数。注册参数/初始state
  相同，OFF新24参数冻结为0，ON须更新。OFF初始全部88评分逐值复现阶段35
  关系ON；ON零偏置引入float attention mask可能改变数值路径，训练前固定
  允许绝对1e-6初始评分差（rtol0），记录真实差，不训练后放宽标准。
- 两臂新AdamW lr.001/wd.01、clip1，各3200步（同800输入4遍），不增加数据
  或恢复不匹配的新参数optimizer。固定终点、无best-dev选择、阈值.5不变。
- 先冻结模型/center全部88输出0差、warmstart OFF0差/ON<=1e-6，再V1/2/4
  优化参数梯度有限非零才训练；保存几何参数delta，旧模型/框/深度不变。
- 新`tools/probe_roi_geometric_objectness_pair.py`输出
  `output/roi_geometric_objectness_pair_v1/{off,on}`，单GPU OFF→ON。每臂最多
  40分钟，原06:31截止不延长。保留所有旧产物；不下载/渲染/推送/生产覆盖。

## 判定

几何ON相对同预算OFF：双鼠及非双鼠TP均不降，全部FP/重复不增，且双鼠TP
至少+4或FP至少减少25%（要求OFF FP>0，避免零误检时把0→0当改进）。
完整接入门槛仍是双鼠TP>53、非双鼠TP≥124、全FP≤47、重复≤3、296两鼠。
不能以“去掉重复”冒充找回第二鼠，也不能用新的frame296成功代替完整开发。
这是固定框评分诊断，不产生新关键点/mask/尾巴坐标，无端到端17门槛收益。
若失败不继续机械叠加；审视冻结局部候选的质量/泛化限制，整理保留路线与
未达标原因。真实Cylinder缺失、重复开发集选择偏差继续明确。
