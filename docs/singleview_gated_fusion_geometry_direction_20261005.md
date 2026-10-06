# 单视角基线后的融合实验：语义残差首轮与几何路线

## 已完成的对照

固定上一轮 seed17 的 mixed800 ROI 检测头与 full800 姿态头，实际独立 V1 编码，不修改 encoder。只修正鼻尖及四足 5 点，其他 18 个 Core23 点与检测框/置信度保持原样。self 与 multiview 两组参数数目、初始化、800 步、场景抽样一致；self 保留同样的支持可用性，但支持内容替换成自身语义特征。

已知单实例场景是本次实验队列，不是输入模型的 GT 实例数量。训练 225 场景、1181 图；双鼠没有参与融合训练，保留双鼠基线输出不计为融合成果。模块根据预测置信度，只有恰好一个可信查询的视角才提供或接收修正；这个规则不能证明跨视角身份，只是保守诊断。

| 模式 | 真实可靠点条件误差 | 底视足点条件误差 | 真实 gated PCK32 |
| --- | --- | --- | --- |
| 固定单视角基线 | 33.246 px | 48.461 px | 0.566 |
| self 残差，800步 | 33.405 px | 50.571 px | 0.554 |
| multiview 残差，800步 | 33.382 px | 50.114 px | 0.554 |

误差为模型输入像素；GT 框辅助关联只用于训练损失/评价，不进入 forward。不是端到端 pose AP，不是外部测试或三种子融合结论。初始覆盖真实有效点约 94.3%、底视足点 75%；没有证据表明只是门控不开造成失败。TRAIN 真实误差 self/multiview 约30.10/30.16px，改进也很小。不能据此证明任意融合架构无效，但不应延长这个近似相同的残差试验。

单元测试12项通过；训练后关闭、单视角、无支持均精确回退。相机反转和重复视角检查通过；CPU重放与训练GPU输出最大差约5.96e-8。原图可见后足仍落在躯干，self/multiview都没有修正这个问题。尚未做新的实际 RGB 融合集成，因为缓存结果已没有采用理由。

## 本次调研与可借鉴部分

- [Cross View Fusion for 3D Human Pose Estimation：论文](https://arxiv.org/abs/1909.01203)与[作者代码](https://github.com/microsoft/multiview-human-pose-estimation-pytorch)：提供跨视角融合、2D训练及3D估计的参考实现。可借鉴的是把相机几何与空间特征联系起来，而不是直接平均不同相机的2D坐标。后续落地前还需具体阅读采样实现，不能声称本项目已经复现。
- [Learnable Triangulation of Human Pose：论文](https://openaccess.thecvf.com/content_ICCV_2019/html/Iskakov_Learnable_Triangulation_of_Human_Pose_ICCV_2019_paper.html)与[作者代码](https://github.com/karfly/learnable-triangulation-pytorch)：把多个2D观测组合为3D估计，包括可学习的三角化路线。可作为置信度加权几何基线与后续可训练模块的参照，不意味着当前鼠数据已有精确3D真值。
- [MVGFormer 作者代码](https://github.com/XunshanMan/MVGFormer)：显式几何模块与 Transformer 结合，利用3D到2D投影和特征更新。启示是单纯语义特征汇总可能缺少有效空间约束；这是针对本轮失败的工作假设，尚未在本项目验证。

没有下载外部权重、复制整套项目或上传私有数据。这些是人体任务的参考，不代表底部折射、鼠足可见性或多鼠身份问题已被解决。

## 下一项有界实验

先审计现有相机及数据坐标，再考虑添加新训练：

1. 使用同一次 `load_item` 加载完整同步场景，取 loader 实际使用的 intrinsics/extrinsics。不要把独立V1样本各自 canonicalize 的 extrinsics 直接拼到一起；原始像素缩放、平移、场景单位都必须一致。
2. 对合成可验证几何做数值重投影/极线检查；对真实已接受关键点，只把极线距离当相机或标注一致性的诊断，不能用模型结果覆盖原标签。
3. 先以预测自身选择的查询、方差和置信度做几何一致性分析；GT只在预测后评价。不能以 GT 足点、GT 可见性或 GT 跨视角身份筛选参与融合的观测。
4. 若标定/图像变换一致，再做不训练的加权/稳健三角化和留一相机重投影诊断，或沿极线采样概率图的门控残差。比较完整留一预测与原单视角结果，不以自重投影误差变小冒充真实精度提高。
5. 单视角、无有效几何支持、视角重复、低基线或异常标定时必须回退；底部玻璃折射和倒影区域需单独报告。真实的针孔标定近似不能自动视为这些像素的精确光路。

如果几何审计不通过，应先解释坐标/标定/点语义差异，不以增加训练步数掩盖问题。所有新增路径仍默认关闭；本轮训练截止2026-10-05 18:11:41UTC，硬截止18:41:41UTC。

## 07:42 UTC 更新：相机审计与预测驱动三角化

审计覆盖249个场景（训练225、开发验证24）。独立单视角缓存与完整同步场景重新加载的有效2D坐标完全相同；K与源文件按实际图像尺寸缩放、E平移按场景尺度归一化的结果也完全一致。已经确认当前 `load_item` 使用 `canonicalize_camera=False`，不是每个相机分别重置坐标系。相关9项数值单测通过。首个审计脚本把3×4与4×4外参直接相减而报错，已修正并保留失败记录；成功结果在 `camera_geometry_audit_v2`。

合成关键点极线距离平均约0.00005px；2D坐标与 `instance_keypoint_world_xyz` 的投影误差约0.00005px，而与另一个 `surface_world_xyz_instances` 字段约2.2–2.4px。这里仅说明字段对应关系，不据字段名称断言它是内部骨骼位置或可靠的真实皮肤点。

真实已接受标注的可靠点极线距离，TRAIN均值7.56/中位4.54px，开发验证均值6.63/中位4.52px；背部为12.90/11.33px均值。camA–camF训练足点的中位距离17.75px，比多数相机对大，需要将折射、标定近似、标注语义和离群点作为待区分的原因，不直接认定某一个原因。原标注不改。

随后完成固定规则、无训练的留一相机实验：目标物理相机及其重复视角全部不参与三角化；支持由预测置信度及热图方差确定，不用GT可见性或身份。稳健版要求至少3个支持内点，16px重投影阈值、2°最小射线夹角；保守版额外要求修正位移不超过48px并只混合50%。这些值在结果计算前固定，没有搜索开发验证阈值。7项单测通过，包括目标坐标NaN不影响结果、离群支持、重复相机、顺序变化及退化回退。

| 方案 | 真实可靠点条件误差 | 底视足点条件误差 | 真实gated PCK32 |
| --- | --- | --- | --- |
| 单视角基线 | 33.246 | 48.461 | 0.566 |
| 预测方差加权留一三角化 | 45.753 | 72.454 | 0.394 |
| 稳健留一三角化 | 36.985 | 55.781 | 0.520 |
| 有限位移、50%混合 | 33.169 | 47.440 | 0.571 |

最后一组仅改变66/175个真实可靠点和4/32个底视足点，改善太小，不推广、不继续调验证阈值。训练集有限混合30.508→30.203px，也没有解决定位根源。视觉检查camA和camF发现稳健三角化仍会把足点移向躯干/头部；几何一致不代表语义正确。只有两相机的合成开发验证样本没有足够独立支持，保持基线，不能算作融合成功。背部和尾巴完全不变。

下一步先重提取固定V1姿态头的空间概率图，检查鼻尖/四足的多峰、峰值与期望位置偏差、GT附近质量（仅事后诊断）。再考虑预测驱动、沿极线聚合的候选重评分，保留目标图像本身的局部证据与单视角回退。训练前先在TRAIN确定方案；不再对失败的坐标硬修正重复增加步数。多鼠身份仍未解决，不把这条单实例试验路径接入正式多鼠推理。

## 08:00 UTC 更新：空间多峰诊断与固定规则候选试验

重放1181张训练、80张开发验证图的独立V1输出，使用原生空间记忆、固定seed17姿态头和原预测框，保留全部6个查询、只存鼻尖和四足5点的48×48概率图。概率期望与既有缓存最大归一化坐标误差4.77e-7，没有重训encoder或换输入。压缩概率图总计约335MB，不重复存图像和token。

训练真实底视181个原误差>32px的足点，预测前五个NMS候选中最接近标注者的误差均值16.50px，95.0%在32px内；原均值预测为71.18px。**这个best-of-five使用GT事后选择，只是可恢复上限，不是推理成绩。**直接取最大峰也不稳定：全体真实可靠点TRAIN30.51→32.27px、开发验证33.25→35.28px。肉眼检查首个训练/验证底视图，确实存在两个同类足部候选导致期望位置落在中间的问题，但并非所有失败都源于此。

随后做无训练、参数预先固定的候选试验：预测图上5个NMS候选（抑制半径2格），每个位置取3×3邻域概率加权局部均值及质量。目标候选用自身质量加其他相机候选的极线一致性评分；sigma16px、几何权重1，选择最强的两个支持相机，要求至少两个不同物理相机支持。`unary_local`对照保留完全相同的支持条件但几何权重为0。只有每视角恰好一个可信预测查询时参与，不输入GT身份、数量、可见性或裁剪。原背部/尾巴/检测输出不变；只有两个视角时回退基线。

| 模式 | TRAIN可靠点 | TRAIN底视足点 | 开发验证可靠点 | 开发验证底视足点 | 开发验证可靠点gated PCK32 |
| --- | --- | --- | --- | --- | --- |
| 基线均值 | 30.508 | 32.920 | 33.246 | 48.461 | 0.566 |
| unary局部候选 | 30.241 | 34.207 | 33.572 | 39.572 | 0.611 |
| 极线局部候选 | 29.616 | 34.193 | 30.557 | 35.798 | 0.646 |

前向没有使用GT；表格依然是GT框辅助关联后的条件误差，不是端到端AP。开发验证24/32个底视足点、165/175个可靠点满足候选更新条件。camA固定示例的前足16号从另一前足附近回到对应足部，18号后足不再落在躯干中间；camF仍有头部附近足点混淆。部分侧视退步：camB40.39→43.17、camC46.25→48.18、camE31.30→32.69px。TRAIN底视均值也略退步，因此这不是已验证可推广的方法，不能自动采用。两视角合成验证全部回退，不能据此声称合成集没有代价。

候选模块8项测试通过：默认关闭与V1精确回退、缺失/重复视角、排列、非法支持、局部均值和几何选择。下一步检查真实缓存的多种视角数量/删除某个支持相机与重复、倒序的稳定性；之后选择固定同步样本从RGB重跑独立V1、预测框和概率图，验证最终候选输出与缓存一致。此后才考虑多种子复验或训练式候选选择。参数未在开发验证上搜索，不因这一次正向结果继续试探验证阈值。

## 08:12 UTC 更新：稳定性、原图重放与三种子复验

真实缓存上单视角和双视角精确回退，完整视角倒序/重复的差异均为0。固定按物理ID选择目标+两支持视角时，开发验证整体误差27.47px、底视42.59px；完整支持为30.56/35.80px。删除支持C/D时底视分别退到47.64/46.02px。支持越多不保证越好，不能据这个小开发集选择相机子集作为新默认。

从原图重新编码固定首个真实同步场景的5个合格视角及首个合成场景2视角，共7张；原生空间记忆、实例token与缓存完全相同，最终候选归一化坐标最大差5.36e-7。模型加载及整项检查约15.34秒。仍是逐图独立V1编码，不能当作与一次联合多视角前向计算量相同的高效架构。报告：`epipolar_candidate_rgb_audit_v1/results.json`。

随后使用既有seed29/43检测头和姿态头重新解码全部候选队列，不进行额外训练，几何参数与seed17完全相同。原检测和姿态golden检查通过；这是同一encoder、同一数据划分下的头部随机种子复验，并非独立外部数据。

| 种子 | 开发验证可靠点 基线→极线候选 | 开发验证底视足点 基线→极线候选 | TRAIN底视足点 基线→极线候选 |
| --- | --- | --- | --- |
| 17 | 33.25→30.56 | 48.46→35.80 | 32.92→34.19 |
| 29 | 33.31→29.75 | 51.29→47.99 | 34.80→36.35 |
| 43 | 34.19→29.98 | 47.68→40.97 | 35.79→37.07 |

三个种子开发验证方向一致，但幅度不同；训练底视均值都退步，不能宣布足点已解决。固定camA三种子图中后足18从躯干回到足附近较一致，前足16在seed29仍有混淆。缺失支持和检测重复导致生效点数不同，也是需要控制的因素。原始标注/生产默认仍不改。

合成TRAIN视角清单已核对：69场景中22个有4视角、44个有8视角、3个新渲染静态场景有7视角。目前前66个只取了2视角，另有308张**既有TRAIN场景的图像**可以编码，无须新渲染或改变源动作划分。下一步先补齐这些实际多视角上下文，避免候选监督只在真实数据及三个静态合成姿势上生效。之后再做局部候选选择/可靠性学习对照，GT只用于损失；继续保留原均值输出作为回退候选，不以固定几何一致性强制替代图像证据。现有合成开发验证仍只有两视角，只能说明回退正确，不是融合泛化检验。

## 08:28 UTC 更新：合成TRAIN补齐及候选学习输入

69个既有合成TRAIN场景已完成全视角缓存，共461张（153张复用、308张新编码）。未渲染、未改场景/动作划分、未读锁定真实数据。完整同步场景维持原归一化尺度，原视角的2D标注、valid与框逐项完全一致；全视角关键点投影最大误差0.000243px。新图像独立V1编码，先完整解码138个查询，再存鼻尖/四足5点。原图重放抽查通过。

固定候选规则在该完整合成TRAIN集合的结果：

| 集合 | 点数 | 原均值误差 | unary局部候选 | 固定极线候选 |
| --- | --- | --- | --- | --- |
| 所有461图 | 2305 | 26.36 | 28.29 | 27.16 |
| 原153图（比较分母不变） | 765 | 15.83 | 18.68 | 18.40 |
| 新增308图 | 1540 | 31.59 | 33.07 | 31.51 |

这是TRAIN诊断，不是泛化评估。更多视角暴露了此前两视角回退掩盖的代价：不可自动将局部峰替换原期望。棕鼠侧视示例中，候选选择会把原先分开的足点聚到相邻足部附近。

局部输入已准备：每点6个候选，第0个始终是原坐标，其余5个是只根据预测概率图得到的局部峰；每个候选按预测位置采样256维空间特征。真实TRAIN与既有开发验证共1108图另存紧凑候选特征，不重复图像或概率图。真实标注与预测字段分开，没有伪造可见性或3D。

新增实验头 `LocalCandidateSelector`，仅原型、**尚未训练**：语义token和候选局部特征各投影到32维，再结合候选质量及可选几何特征评分。unary/geometry两组参数数量相同，geometry关闭时把对应输入置零；末层零初始化，第0候选优先，所以初始输出及默认关闭均精确回退原坐标。缺失或非有限支持不能改点；梯度只进入新选择头。候选采样3项、选择头7项测试通过。

下一项是固定seed17基础模型上的匹配候选监督对照，先800步、相同初始化与图像调度，不训练encoder、检测头或原姿态头。候选GT距离只用于训练损失/事后评价，不能用它筛选推理候选、裁剪或支持相机。保持原均值作为可学习回退；真实可靠点与合成域分别报告，检测失败和未激活点不能从评价分母中剔除。不因开发验证分数调整阈值。

## 08:43 UTC 更新：匹配候选选择训练首轮

预测特征构建器已完成7项测试：单/双视角回退、支持缺失、重复、排列、多可信查询，以及同物理相机内容冲突时弃用。模型不使用相机ID作为学习特征；几何支持仍要求两个其他物理相机。unary与geometry同样适用这个门槛，以免本轮对照混入覆盖率差异。

两组各800步，22657参数，完全相同初始权重与抽样（每批4张真实底视、4张其他真实视角、8张合成图）。AdamW学习率0.001余弦下降至0.0001，weight decay0.001，梯度裁剪1。仅训练新选择头。每个有效点的目标为六个候选到已有GT距离产生的softmax软标签，sigma16px；交叉熵按图像平均，缺失标注不作不可见或背景。基线坐标作为第0候选保留。未搜索开发验证阈值或挑选checkpoint。

| 模式 | TRAIN真实 | TRAIN底视 | TRAIN合成 | 开发验证真实 | 开发验证底视 |
| --- | --- | --- | --- | --- | --- |
| 原均值 | 30.51 | 32.92 | 26.36 | 33.25 | 48.46 |
| unary选择 | 26.59 | 30.05 | 19.15 | 29.51 | 36.12 |
| geometry选择 | 24.72 | 30.89 | 18.34 | 29.59 | 61.16 |

几何组不能采用。底视PCK32虽然从0.375升到0.531，但原本约13px误差的后足被跳到前足位置、误差约280px。32个底视评价点里7个新增误差超过16px，最大新增267px；unary为2个、最大新增48px。固定首帧图肉眼确认17号后足被几何组错误移动到前足区域。不能让整体PCK或较低交叉熵掩盖这个错误尾部。

全部TRAIN图像上交叉熵unary0.893、geometry0.852，软标签熵下限约0.586。抽样训练最后100步均值0.773/0.743，前一个100步0.790/0.753，仍在下降但已变缓；抽样loss与全量loss的图像权重不同，不可直接相减推断欠拟合。没有以这次失败为由盲目追加相同训练。默认关闭、初始输出和训练后的无支持回退均检查通过。

检测完全未改变：真实开发验证48个GT实例、50个预测框、40个匹配TP；合成开发验证32/32/32。上述姿态数值仍为GT框辅助关联的条件指标，不是AP；合成开发验证因两视角门槛全部回退，不能证明选择头的合成泛化。

下一步先检查unary能否去掉跨视角支持门槛、独立用于V1，保留纯预测的多查询保护，并单独测试双鼠、不借GT实例数量。随后考虑有界的概率先验残差评分，防止小选择头压倒原热图证据而选取极弱远端候选；任何候选过滤或代价函数设置先在TRAIN确定，不能针对这些开发验证失败点调参。原始标注、尾巴/腹部决定及正式模型仍不变。


## 2026-10-05 independent unary eligibility audit

Frozen 800-step unary weights, no extra training: remove the requirement for other cameras and enable only the unique predicted query with score >= 0.5. No GT count, correspondence or visibility in forward. Geometry is explicitly zero. All labels are consulted only after prediction.

| Input-pixel mean error | Baseline | Old matched-support gate | Independent gate |
|---|---:|---:|---:|
| TRAIN real reliable |30.508|26.588|26.588|
| TRAIN real bottom feet |32.920|30.045|30.045|
| TRAIN synthetic reliable |26.359|19.154|19.175|
| VAL real reliable |33.246|29.511|29.511|
| VAL real bottom feet |48.461|36.122|36.122|
| VAL synthetic reliable |19.225|19.225|17.451|

The previously fallback-only synthetic validation is now activated. 117/160 points changed: 14 improve by >16px, 9 worsen by >16px, worst added error 41.50px. Real bottom 11 improve and 2 worsen by >16px. Full CPU versus previous GPU output replay, singleton/batch, permutation, duplicates and ignored NaN geometry all have exactly zero coordinate differences. Default-off remains exact.

Visual inspection: row060 view2 foot17 moves incorrectly from the hindfoot vicinity up onto the torso (22.06 -> 63.57px); row063 view0 front feet move from the lower body into the face-level washing posture, near the existing annotations. These are diagnostic extrema, not randomly representative cases. Synthetic view0 is not a physical bottom-view claim.

Decision: independent V1 unary remains promising, but not promoted. Next evaluate existing 32 dual-instance development images with frozen weights and prediction-only gates before new training. This reused development set is not an external generalization test. No GT-count/query matching in forward; report conditional pose metrics and unchanged detector counts separately. Avoid extending failed unconstrained geometry solely because CE is descending. Consider a TRAIN-predeclared bounded native-probability prior afterwards. Original labels, tail/back, locked sessions and production defaults unchanged.

Artifacts: `output/singleview_fusion_cycle_20261005/independent_unary_eligibility_audit_v1`; Windows report in same-name subfolder of `C:/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005`.


## 2026-10-05 dual-instance diagnostic and bounded probability-prior intervention

Frozen single-mouse-trained unary fails on 32 existing synthetic dual development images. All six queries are retained; GT counts select the offline evaluation cohort only, never enter forward. Each image has two confident queries, so the unique-query gate safely abstains everywhere. Enabling each predicted query worsens conditional nose/four-foot mean error 23.299 -> 26.063px, P95 56.86 -> 74.06, gated PCK32 .594 -> .553. 213/320 points change, with 20 improvements and 33 degradations exceeding 16px. Detector unchanged: 64GT,64 predictions,45 IoU>=.5 matches. Visual row13/view0 confirms a nose moving from the brown mouse to the black mouse and feet crossing instances. Other-instance-nearer count is only a diagnostic, not a reliable identity-swap label. Do not deploy this extension.

Next bounded experiment changes only the candidate-score parameterization, with identical original step800 initialization, image schedule, optimizer, eligibility and labels. New `PriorCandidateSelector` retains original mean as option0, adds native log-relative peak mass to other candidates and bounds learned relative correction to ±2 log-units (predeclared; no VAL search). A native relative mass below exp(-2) cannot overcome baseline. Seven unit tests pass: off/initial exactness, unavailable/NaN fallback, weak-candidate bound, head-only gradient, unary ignoring geometry, ordering and bound validation. No production factory modification.

| Bounded-prior experiment | TRAIN real | TRAIN bottom feet | TRAIN synthetic | VAL real | VAL bottom feet |
|---|---:|---:|---:|---:|---:|
| unary |27.137|29.504|21.772|30.438|41.197|
| geometry |26.425|29.765|21.248|26.324|39.704|

Unbounded controls: unary VAL29.511/36.122; geometry29.588/61.161. Bounded geometry now beats its matched unary control but does not outperform old unary on bottom feet. Do not choose a different model by VAL camera. Independent bounded unary improves synthetic VAL19.225 ->18.063, but dual per-query still worsens23.299 ->25.657. This constraint is not sufficient for multi-instance identity. Last100 sampled CE .8718/.8516 vs preceding100 .8872/.8661: slowing, not proof of convergence.

Visual inspection: bounded geometry preserves original point17 in row0, avoiding the previous ~267px added error; row15 still collapses points15/16 onto one visible paw, point15 worsening64.41 ->131.71px. Failures remain, no promotion. Next: recompute geometry under genuine view subsets/missing/duplicate/order stress, then controlled additional selector-training seeds29/43 if warranted. Frozen baseline remains seed17; these are not full-backbone seed repeats. Consider actual dual TRAIN contexts before any multi-instance training claim.

Artifacts and Windows reports: `dual_unary_diagnostic_v1`, `bounded_prior_candidate_pilot_v1` under the current cycle and `C:/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005`. Original annotations and locked sources unchanged.


## 2026-10-05 actual-view stress, selector-seed repeats and dual TRAIN coverage

Recomputed geometry on actual subsets, not sliced full-context tensors. Reversing and exact camera duplicates preserve predictions exactly. V1, V2 and all-invalid support fall back exactly. Fixed target-centric V3 gives real28.33/bottom41.20 versus full26.32/39.70 and baseline33.25/48.46. All seven leave-one-camera-out tests retain real improvement versus the same observed-target baseline. Coverage and missing target views are explicit; no selecting optimal cameras from these numbers.

Additional seeds29/43 change only selector initialization and sample schedule, with frozen original baseline17. Both arms share each seed's initialization/schedule, 800 steps. Geometry improves real mean in all three, but worsens bottom versus unary in two of three:

| Selector seed | Unary real | Geometry real | Unary bottom | Geometry bottom |
|---|---:|---:|---:|---:|
|17|30.438|26.324|41.197|39.704|
|29|29.292|27.852|39.886|42.215|
|43|31.141|28.006|38.414|42.122|

Mean ± sample SD: real unary30.290±.933, geometry27.394±.930; bottom unary39.833±1.392, geometry41.347±1.424. Not an external video/split/backbone-seed validation. Visual row15 shows all three geometry heads collapse points15/16 on one paw. Dual unary errors25.657/24.682/24.779 all worsen original23.299. No promotion or per-camera model picking.

Recovered the existing 42 TRAIN dual images (21 scenes) into `dual_train_candidate_audit_v1/train_inputs.pt`, no additional frames/renders or labels. Prediction-only probability/candidate construction first, then GT-only candidate-coverage diagnostic. TRAIN84GT84pred67TP; VAL64GT64pred45TP. Train/validation scenes disjoint, but not independent source videos.

| Dual reliable point error | Original mean | Optimistic best-of-six oracle | Prior-admissible oracle |
|---|---:|---:|---:|
|TRAIN420 points|22.235|11.645|15.033|
|VAL320 points|23.299|12.119|15.445|

Oracles use GT post hoc and are not achieved inference or annotations. Correct candidates often exist, motivating one controlled TRAIN-data inclusion experiment before new architectures: fixed unary prior-bound2 head,800steps, independent predicted-query score>=.5 gate in both arms. Compare original 4real-bottom+4other-real+8single-synthetic schedule with same first14 samples plus2 existing dual images (12.5% image budget), using preregistered independent dual RNG17042. Equal-image loss, recompute supervision masks (old masks include multiview eligibility), no GT count or identities in forward. The small dual set is heavily reused; do not assume more steps solves generalization. Evaluate single/dual development subsets and raw instance errors, no VAL-selected checkpoint.

Windows report: `C:/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005/bounded_prior_repeat_review_v1/index.html`. Original labels/locked sources/default model unchanged.


## 2026-10-05 dual TRAIN inclusion result and decoder-topology hypothesis

Data-only control: identical unary prior-bound2 head, initialization,800steps and per-query predicted-score>=.5 gate. Mixed arm replaces2/16 synthetic-single images with existing dual TRAIN42; first14 samples remain identical. Supervision recomputed independently of old multi-view support gate.

| Dataset (mean px) | Baseline | Single-only | Dual-mixed |
|---|---:|---:|---:|
|TRAIN real reliable|30.508|27.546|27.884|
|TRAIN bottom feet|32.920|28.930|28.479|
|TRAIN synthetic single|26.359|21.135|21.431|
|TRAIN synthetic dual|22.235|25.611|21.674|
|VAL real reliable|33.246|30.862|30.667|
|VAL bottom feet|48.461|39.823|41.778|
|VAL synthetic single|19.225|17.922|19.883|
|VAL synthetic dual|23.299|26.356|25.557|

Dual data improves TRAIN more than VAL; dual remains worse than original and synthetic-single regresses. No promotion or blind increase in steps. Last100 sampled losses are slightly higher than preceding100; raw sampled losses are not smooth full-dataset convergence curves. Dual other-instance-nearer count24/29/29; mixed adds6 such points, all6 within their own expanded predicted box.23/25 points harmed by >16px also remain in that box;21/320 GT points lie outside it. Hard box clipping is neither sufficient nor uniformly safe. Visual row13 nose improves relative to earlier errors, but cross-instance feet remain. Current original spatial readout already applies soft predicted-box suppression.

Primary-source research: [GroupPose](https://github.com/Atten4Vis/GroupPose) and its [official decoder](https://raw.githubusercontent.com/Atten4Vis/GroupPose/main/models/grouppose/transformer_deformable.py) separate within-instance and same-semantic across-instance attention. [ED-Pose](https://arxiv.org/abs/2302.01593) and [official repository](https://github.com/IDEA-Research/ED-Pose) motivate explicit instance-box/keypoint context. These human-pose results do not establish mouse transfer. Current local decoder flattens all6×23 queries into unrestricted attention; constrained interaction is a testable hypothesis, not a proven root cause. No external weights downloaded or paper reproduction claimed.

Prototype `TopologyPoseAdapter` now exists, **not trained**. Both arms copy the same decoder and add an identically zero-initialized token-residual projection over a frozen baseline. `full` preserves unrestricted new-branch attention; `structured_union` permits same-instance OR same-semantic interactions via one static mask. This is not GroupPose's sequential two-attention implementation. No GT identity or count enters forward; all138 queries remain. Only nose/fourfoot outputs can change. Back/tail/abdomen remain frozen. Default-off and initial-enabled exactness, frozen baseline, nonzero adapter gradient, query-permutation equivariance, mask rule and equal parameter tests pass (7 tests).

Next bounded trial: frozen genuine V1 native-memory pool1223TRAIN/112VAL (not previous extra308 synthetic candidate-only views), balanced4real-bottom+4other-real+6single-synthetic+2dual-synthetic, matched800steps. Coordinate-only `matched_image_point_loss`,5 reliable points, frozen box Hungarian only in loss/eval. Suggested predeclared AdamW decoder1e-5→1e-6, residual1e-4→1e-5 cosine,wd.001,clip1. Measure additional decoder forward compute/memory, check native golden replay and default-off before/after training. No checkpoint chosen on reused development VAL.

Windows report: `C:/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005/dual_train_inclusion_control_v1/index.html`. This prototype is isolated; production and labels unchanged.


## 2026-10-05 native topology control: no clear advantage

`native_topology_control_v1` stopped before training because21 synthetic frames have kind=`curve`, so literal kind=`synthetic` missed them. Failure preserved; v2 correctly includes132 synthetic+21 curve-kind synthetic single images, still supervising only nose/fourfeet. Native pool1223TRAIN/112VAL, frozen original baseline17, same800step schedule and coordinate loss,1119744 trainable adapter parameters in both arms. No encoder training. `full` versus `structured_union` changes only new-branch self-attention topology.

| Metric px | Baseline VAL | Full VAL | Structured VAL |
|---|---:|---:|---:|
|Real reliable|33.246|33.267|33.374|
|Bottom feet|48.461|46.296|46.177|
|Synthetic single|19.225|18.376|18.616|
|Synthetic dual|23.299|22.420|22.398|

TRAIN full/structured real28.575/28.764,bottom27.782/27.977,synsingle15.136/15.201,syndual21.073/21.116. The single synthetic TRAIN pool is153images here, not461 from prior candidate experiments; cross-experiment numbers are not controlled comparisons. Both architecture arms nearly identical; no extension/promotion. This does not disprove complete GroupPose architectures.

All frozen-path tests pass: initial residual exact0, default-off and native-source golden<2e-5, after-training untargeted18points exactunchanged. GPU cached training about24seconds/arm with~3.8GB peak allocated including caches; this excludes original1B RGB encoding. All100-step checkpoints retained. Visual row34 still has duplicate query feet on abdomen; dualrow61 feet remain clustered. Sampled last100 loss slightly above previous100, not evidence for blind step extension.

Reviewed prior heatmap failure before considering new losses: `output/core23_spatial_cycle_20261004/review_reliable_heatmap_v1/decision.json` rejected matched1600-step continuation with.002GaussianKL/sigma12px on reliable points in all3seeds (both real and dual degraded). Do not repackage the same sigma/weight sweep as a new hypothesis.

Next: pause new model training and verify actual-RGB integration of already-promising bounded candidate heads. Fixed seed17 unary gives independent local fallback; geometry overrides only eligible supported points, otherwise retains unary when local enabled. Both options default off, allqueries/boxes/back/tail unchanged. Establish cache parity, 1/2/allviews, ordering/duplicates/missing support and RGB runtime cost. Unique confident query is only a restricted single-subject association heuristic, not proven multi-mouse identity. No GTcount in forward, no automatic deployment. Reused development data cannot prove external generalization.

Windows report: `C:/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005/native_topology_control_v2/index.html`.


## 2026-10-05 composite readout: actual RGB parity, remaining semantic failures

Implemented isolated default-off `BoundedCandidateFusion`: local bounded unary selection works independently; supported geometry overrides it, including choosing the original candidate. Unsupported geometry retains local output. Only nose/fourfeet can change. Seven unit tests cover exact fallback, ordering/duplicates, invalid support, initial equality and frozen inference. No production integration or deployment.

On the reused 80-image development cohort, baseline/local/composite mean errors are real33.246/30.438/26.324px, bottom48.461/41.197/39.704px, synthetic-single19.225/18.063/18.063px. These are model-input pixels and conditional pose metrics, not end-to-end AP or external-test evidence. Mean gains do not eliminate outliers: composite bottom P95 is121.14 versus107.45 baseline;11 real points worsen by>16px, worst86.05px.

Fresh RGB replay used14 predetermined unique images and15 independent encoder calls. Maximum pose/composite cache difference5.364e-7 normalized; logits2.861e-6. View-order/duplicate and insufficient-view fallback checks pass. Subset tests reuse these fresh independent outputs, not joint-context re-encoding. Summed14 forward timings1.616s; total13.56s includes initialization/loading. No fairness claim against a joint multiview encoder. The two dual views each predict two queries and abstain; this does not establish robust multi-mouse association.

Visual inspection of four panels: first real bottom frame recovers feet from torso toward palms; second bottom frame still puts frontfoot15/16 near one paw; sideview5 retains a badly placed box and clustered feet; dual top is unchanged and imperfect. Cache/RGB parity is engineering evidence, not anatomical correctness.

Next bounded audit: all existing physical views of the first two dual validation scenes and first synthetic-single scene, fixed before seeing predictions. Verify camera gauge and RGB parity; examine whether missed mice create misleading unique-query cross-camera fusion. Do not infer instance identity from query index or unique detection. No new training or production promotion yet.

Report: `C:/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005/composite_candidate_rgb_audit_v1/index.html`.


## 2026-10-05 all-camera audit: dual fallback is not association success

Expanded from two dual scenes to all16 existing synthetic dual development scenes, four physical cameras each,64 independent RGB encoder calls. Common camera gauge checked against synthetic projections (max0.000084px); existing-view cache parity max1.91e-6. All view-order/duplicate/invalid and1/2-view local fallback checks pass. No training, rendering, annotation changes or deployment.

All64 images predict two confident queries, causing the current restricted local/fusion module to abstain everywhere. All modes therefore equal baseline:128 predictions/128 targets,97 true positives at IoU>=0.5; conditional640 reliable points mean21.826px, P9554.775px, detection-gated PCK32=0.6328. Correct count does not establish correct boxes or points. This all64-view cohort differs from the prior32-view score23.299px. No natural single-query association failure occurred, so this cannot validate safety under missed detections.

Visual group07/view1 retains misplaced boxes and paw positions on torso/tail; group10/view3 has overlapping boxes and clustered points. On the separately fixed single synthetic scene, eight views include one duplicate-detection view6 that abstains; mean25.863->22.652px with supported fusion. Same scene plus more cameras is not external generalization.

Next: a separately labeled fault-injection audit will suppress copied detection logits using predeclared prediction-only selection, leaving actual RGB and all original artifacts unchanged. Its purpose is to test whether missed detections can create mixed-instance support, not estimate natural error rates. Do not use GT identity/count for query selection and do not claim dropout tests are real-image occlusion tests. No further training until this assumption is understood.

Reports: `C:/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005/dual_allscene_association_audit_v1/index.html` and `composite_allview_association_audit_v1/index.html` in the same parent.


## 2026-10-05 logit fault injection: unique-query association is unsafe

This is an output fault test, NOT natural missed-detection frequency, RGB occlusion, or new generalization accuracy. Frozen64-image dual predictions retain all6queries and original RGB/boxes/point probabilities. A copied logit tensor preserves one original confident query by a fixed prediction-only rule; all other logits become-20. No GT identity/count chooses the retained query. Evaluation follows the entire forward pass.

| Fixed intervention | Mixed reliable identities, scenes/16 | Different-identity/reliable camera pairs | Fusion-eligible points | Points changed versus local |
|---|---:|---:|---:|---:|
|Highest confidence|11|32/67|320|116|
|Leftmost predicted box|14|39/67|320|127|
|Alternating confidence rank by physical-camera parity|8|16/52|320|140|

Identity audit requires original predicted-box assignment IoU>=0.5; other views are unknown. Original fixed and perturbed-logit operational assignments agree in all192 intervened images. Thus assignment switching does not explain this result. Default-off,1/2-view local fallback, order/duplicates, all-invalid support and unchanged18untargetedpoints pass; input file hash is unchanged. CPU5.16s, zero encoder calls/training.

Concrete visual counterexample: first scene highest-confidence keeps query4 in all four views, but posthoc animal IDs are1,0,1,0. In top view0 the retained box belongs to the brown mouse; composite point17 jumps onto the black mouse. That scene's20 retained reliable points worsen23.322->36.126px versus local, with worst added error95.814px. Across all retained IoU-qualified points the mean21.878->21.881px nearly cancels improvements and harms, so the average obscures the failure. No automatic model promotion.

The feature builder currently gates on the number of finite available camera supports, not an identity test or an absolute geometric acceptance threshold. Strong-looking epipolar evidence and identical query indices do not establish one animal. The single-subject development improvement remains scoped to that assumption; this prototype must not be advertised as general multi-mouse fusion.

### Primary-source research and proposed next step

- [Dong et al., MVPose project](https://zju3dv.github.io/mvpose/) separates cross-camera association from pose recovery, combining appearance and geometry with multi-way consistency and allowing missing/false detections without knowing the subject count. This is the closest short-term design reference.
- Inspected the official [geometry affinity code](https://raw.githubusercontent.com/zju3dv/mvpose/master/src/m_utils/geometry.py), [SVT matcher](https://raw.githubusercontent.com/zju3dv/mvpose/master/src/models/matchSVT.py), and [partial-assignment projection](https://raw.githubusercontent.com/zju3dv/mvpose/master/src/models/match_solver.py): same-camera blocks are constrained and matching is not simply equal-index linking. Do not copy its17-human-joint assumptions or scene-wide distance normalization blindly; use our verified camera convention and explicit invalid-camera behavior.
- [MvP](https://arxiv.org/abs/2111.04076) uses hierarchical pose queries and camera-guided projective attention for direct3D prediction. It is a longer-term learned3D-instance route, not a small change to this selector.
- [MVGFormer](https://github.com/XunshanMan/MVGFormer) iterates coarse3D queries,2D projection/refinement and triangulation, with evaluations across camera arrangements/counts. This is relevant to variable-view design but does not provide a single-view triangulation solution; retain a genuine image-only fallback.

Our next step is evidence first, not another pose-head sweep: recover the existing21dualTRAIN scenes' full4view prediction/calibration context, then audit cross-query affinity from corresponding-point epipolar distances and independent appearance features. Report whether frozen semantic tokens merely preserve query-index bias. Same/different-instance labels are permitted only for TRAIN calibration and posthoc held-out development evaluation, never inference routing. Do not force all views into matches; eventual association needs explicit rejection, at-most-one detection per physical camera per group, and multi-view consistency. Appearance evidence on black/brown synthetic mice alone does not prove same-color or real-mouse identity performance. No production integration until association is tested.

Windows visual report: `C:/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005/association_logit_fault_audit_v1/index.html`.


## 2026-10-05 cross-query evidence: current token similarity is not re-ID

Recovered all21 existing dual TRAIN scenes' four-camera context:84 fresh independent RGB passes,27.35s total including load, max synthetic projection0.000082px. Existing42view outputs replay within tolerance; TRAIN and development scenes are disjoint. No new rendering/actions/training. `dual_train_allview_affinity_context_v1/groups.pt` preserves predictions and camera geometry separately from evaluation targets.

New `mousegeoforge/association_evidence.py` computes four independent prediction-only channels: corresponding5point epipolar mean/median in input pixels, corresponding semantic-token cosine, and original-coordinate local-memory cosine. It does not match or fuse and accepts no labels/counts. Seven tests pass (analytic geometry, camera/query permutation, duplicate-camera handling, invalid camera, empty/single view and input validation).

| Evidence | TRAIN identity AUC | Development identity AUC | Development same-query AUC | Mutual known correct / known | Unknown / all proposed |
|---|---:|---:|---:|---:|---:|
|Epipolar mean|0.820|0.881|0.383|93/98|64/162|
|Epipolar median|0.822|0.873|0.402|92/99|63/162|
|Semantic cosine|0.433|0.420|0.945|43/106|79/185|
|Local cosine|0.458|0.439|0.844|38/88|58/146|

Identity labels are posthoc box matches with IoU>=0.5. TRAIN135/168 and development97/128 nodes qualify; unknown nodes remain candidates and are included in coverage counts. Both AUCs are diagnostic, with no fitted weights, thresholds or score combinations. The dual action-video sources are disjoint (verified in the provenance correction below), but the render scene/assets are shared and this is not independent real RGB validation.

The semantic mutual matches select identical query indices on185/185 development edges, despite only43/106 known edges linking the same animal. This is strong query-index bias, not a validated appearance descriptor. The first-scene visual matrices show high semantic similarities on different-animal equal-query pairs. Local point memory also fails, plausibly affected by inaccurate selected positions and view-dependent encoding; that mechanism is an inference, not proven causation. Reject direct cosine as a re-ID gate; do not simply invert the score based on development AUC.

Geometry is useful but not sufficient: mutual nearest still includes5 known wrong edges and64 unknowns. A singleton wrong candidate is trivially mutual. Next test a conservative partial matcher with an absolute cutoff calibrated on TRAIN only, camera uniqueness and clique consistency. Predeclare a1% empirical TRAIN different-identity-pair false-acceptance budget for mean epipolar distance (not a real-world guarantee), freeze before evaluating development/fault injection. Require at least3 distinct cameras before any future fusion. Report rejection and coverage even if it yields no usable groups; no threshold loosening to recover a preferred score.

No production changes, learned association model, or multi-mouse fusion deployment. A future dedicated predicted-ROI appearance descriptor or supervised association branch is a separate hypothesis, requiring same-color/real validation, not merely black/brown discrimination.

Windows report: `C:/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005/cross_query_affinity_diagnostic_v1/index.html`.


## 2026-10-05 strict partial matching has zero coverage; pre-query ROI cosine also fails

`partial_camera_association.py` implements a separate default-off partial grouping utility (7tests): mutual nearest across physical cameras plus frozen absolute geometry cutoff, maximal complete groups of>=3cameras, at most one detection/camera, reject ties and overlapping ambiguous groups. No animal-count input, no pose edits. TRAIN-only1% negative-edge calibration froze2.36447px before development evaluation:1/164 negative and only2/162 positive TRAIN pairs pass. Natural TRAIN21scenes, development16scenes and all three logit-fault modes produce ZERO qualifying groups. This is zero coverage, not safe-fusion success. Cutoff was not relaxed on development results.

A distinct no-training probe extracted features before instance query embeddings: current frozen Omega image backbone patch tokens1024D versus projected image memory256D, each mean-pooled on5x5samples within fresh predicted boxes. No GT boxes/masks or new downloads.148independent RGB passes,24.59s total, exact original pose/box replay;5ROI tests pass. This is feature-map pooling after full-image encoding, NOT crop-image re-encoding.

| Descriptor | TRAIN identity AUC | Development identity AUC | Development known mutual correct | Unknown / all mutual |
|---|---:|---:|---:|---:|
|Pre-query image backbone ROI|0.496|0.469|40/84|46/130|
|Projected memory ROI|0.416|0.411|41/102|56/158|

Neither raw cosine is usable as re-ID. Visual inspection of first-scene predicted crops confirms that some side-view boxes include both mice; boxed-region purity and full-image context are plausible confounds, not experimentally isolated causes. Do not claim a different feature-extraction layer solved identity association.

Next bounded hypothesis: predicted-box RGB crop BEFORE a separate pass through the existing frozen image backbone, compared with current full-image ROI pooling. Preserve aspect ratio,224square auxiliary crop input, unchanged768-based main detector/pose input; record extra encoder cost. Use only predicted-confident queries, no GT crop/count. Predeclare crop CLS and mean-patch readouts, RGB and grayscale controls, no training or VAL threshold optimization. Grayscale removes chroma but not coat brightness, so it cannot establish same-color generalization. If frozen crop descriptors still fail, stop raw-cosine variants and consider a TRAIN-supervised association descriptor with independent-source/same-color validation rather than more pose-head steps.

Windows report: `C:/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005/partial_and_prequery_review_v1/index.html`.


## 2026-10-05 crop re-encoding: reduced slot bias, insufficient raw identity separation

Encoded296 predicted-confident boxes from148images as RGB/gray pairs:592 main crop inputs plus5 repeat-check inputs,150 batch calls,11.42s total. Auxiliary224square aspect-preserving crops; frozen existing Omega image-backbone only, no new weights/download/training. Main768based detector/pose unchanged. MeanRGB padding is fixed across RGB/gray.5crop preprocessing tests pass. Reverse batch is exact; singleton versus batch bf16 maxdifference0.00805, minimum descriptor cosine0.999985, explicitly not claimed bitwise identical.

| Crop readout | TRAIN identity AUC | Development identity AUC | Development mutual known correct | Unknown / all mutual |
|---|---:|---:|---:|---:|
|RGB CLS|0.553|0.559|55/79|41/120|
|RGB patch mean|0.517|0.502|39/71|42/113|
|Gray CLS|0.519|0.524|42/74|35/109|
|Gray patch mean|0.504|0.491|34/73|39/112|

The crop encoder's same-query AUC is near0.5 rather than the former semantic0.945, but raw identity separation remains weak. RGB CLS improves the limited mutual-ranked subset, not enough to validate unknown-identity fusion. Actual input contact sheets still show mixed-animal crops. Gray removes chroma only, not coat brightness, and is not a same-color-animal test. Stop replacing raw cosine feature variants.

### Provenance correction

The previous wording suggesting TRAIN and development share the same source video was inaccurate. Checked all37frame records against `dataset/Homecage/LightningPose-CRIM13-Diverse400.plan.json` source_image paths: all21TRAIN scenes use `crim13/021709_A25_Block22_BCfe1_t`; all16development scenes use `crim13/021309_A29_Block16_BCma1_t`. The action sources are disjoint, while arena, meshes, materials and camera setup are shared. This is source-action-separated synthetic development, not independent real RGB or same-color validation. Historical numerical outputs remain unchanged; `dual_source_provenance_correction.json` supersedes inaccurate prose caveats.

Next bounded experiment: a small shared linear association projection trained only on TRAIN cropCLS features, with RGB/gray matched initialization/schedule and explicit unknown-label exclusion in the loss. Reserve5of21TRAIN scenes for association-stage internal calibration (same action source; do not call it source-independent), keeping the outer source split intact. Evaluate final fixed-budget models on the existing development source without choosing steps there. No pose-head training or fusion deployment; learned black/brown appearance remains a possible shortcut.

Windows report: `C:/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005/predicted_crop_affinity_diagnostic_v1/index.html`.


## 2026-10-05 supervised association projection and conservative partial groups

Completed matched RGB/gray1024->64 bias-free projections (65,536parameters), seeds17/29/43,800steps each. Main encoder/detector/pose frozen, cached predicted-box cropCLS only.16TRAIN fitting scenes plus5internal calibration scenes; outer16development scenes have a disjoint action source but shared render assets. Internal calibration is same source and main pose model already saw it. Unknown nodes enter forward but not loss positives/negatives.7projection tests and7partial-group tests pass. Six fits total20.5seconds (cached small head only).

The near-identical final RGB/gray fitting losses were checked: feature arrays do not alias, max difference0.7492. Multi-positive contrastive fit lower bound0.72492577 is reached by800steps; fit identityAUC reaches1 around100steps while calibration loss worsens later. Do NOT extend training. Retained fixed800endpoints rather than selecting steps on development.

Development RGB identityAUC0.930/0.924/0.927, gray0.735/0.746/0.756. RGB mutual known precision87/91,86/92,85/91, but63/154,62/154,64/155mutual pairs unknown. Gray removes chroma, not brightness; this is not same-color validation.

Calibrated partial grouping on5reservedTRAIN scenes:43knownnegative and43positive cross-camera edges. Predeclared1% empirical negative budget floors to0; strict distance<minimum negative. Mutual nearest, >=3distinct-camera complete cliques, reject overlaps. Unknown nodes retained. All6models plus rawRGB/gray evaluated after freezing thresholds. RGB17/29/43 natural development accepted5/9/10groups:5/7/7fully-known pure,0knownmixed,0/2/3unknown;15/28/32of128nodes assigned. Gray1/2/0groups mostlyunknown. Rawcosine and priorstrictgeometry accepted0groups. Small correlated calibration sample does NOT establish a1% safety bound. Fault tests observed0knownmixed accepted groups but unknowns and very low coverage remain. No pose changed.

Visual review: seed17scene4 brown mouse associated views0/2/3 at q3/q3/q4, overlapped sideview1 rejected. Unknown seed43scene11 topview boxes truncate body portions; visually plausible identity does not override unknown evaluation. Firstscene highest-confidence fault kept different mice across cameras; all four rejected. Curves confirm saturation/calibration deterioration. Illustrations are post-hoc categories, not random samples.

Next: default-off association-gated offline candidate fusion, fixed historicalseed17/800 and frozen threshold. No new training or threshold adjustment. Compare original/current local/unsafe unique-query composite/grouped local/grouped geometry/grouped composite. Keep unassigned nodes and18non-target points unchanged in grouped mode. Evaluate all original GT assignments only after forward, plus assigned/IoU-qualified subsets and unknown coverage. Quantify foot localization, P95/worst harms, not only association accuracy. Negative results stay negative; no production promotion.

Windows report: C:/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005/association_projection_review_v1/index.html


## 2026-10-05 grouped fusion negative result; observation quality is a separate problem

Implemented experimental default-off GroupedCandidateFusion,7tests. Predicted physical-camera/query groups only; no labels/count/correspondenceGT. Missing/bad calibration, fewer than3views, duplicateviews, overlappinggroups, or low-confidence query abstain. Each accepted group provides a masked copy of logits to frozen candidate selector; only its5reliable points copied back. Unassigned queries, other18points, boxes and source tensors unchanged. Actual-context reversed camera/group ordering and V1/V2 fallback pass; CPU recomputed seed17association groups match earlier inference groups. Prior training PID3773670 has exited. All new jobs complete.

Fixed RGBassociation seed17/800 and its frozen cutoff, all21TRAIN plus16development natural scenes, three development logit faults. NO training. Natural640points original21.826px, groupedlocal22.135px, groupedgeometry22.091px. Accepted75points original18.864/P9540.651px, groupedlocal21.502/66.887px, geometry21.123/62.634px. Geometry caused4harms>16px, worst+57.171px. Fault highest-confidence retained320points: baseline21.761px, oldunsafecomposite23.423px, grouped21.900px. Reduced unsafe jumps is not a localization improvement over no correction. Reject deployment; do not retune thresholds on development.

Visual scene5point15: baseline5.347px becomes62.518px, moving from correct front-body region toward rear/ground. Same-animal group retained, so instance association is not the only failure. Scene12point17 has candidates offbody and shifts farther outside. Precise visibility cannot be inferred from these overlays; GT is only diagnostic.

All5accepted groups x5points: exhaustive6^3 candidate combinations with positive camera depth and lowest joint reprojectionRMS selection gives only2.928px mean self-reprojection but29.510px actual error, worse than original18.864. Optimistic GT-per-view best-candidate bound10.630px; explicitly NOT inference performance. Correct-enough candidates exist for some observations, but independent epipolar support and pure geometric consistency fail to rank them reliably.

### Research checked against primary sources

[Learnable Triangulation, ICCV2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Iskakov_Learnable_Triangulation_of_Human_Pose_ICCV_2019_paper.html) distinguishes image-derived per-keypoint confidence-weighted algebraic triangulation and learned volumetric feature aggregation. [Official implementation](https://raw.githubusercontent.com/karfly/learnable-triangulation-pytorch/master/mvn/models/triangulation.py) normalizes algebraic confidence across views; volumetric code optionally uses GT pelvis, a switch we must not carry into inference. We did not download weights or claim reproduction.

[Generalizable Human Pose Triangulation, CVPR2022](https://arxiv.org/html/2110.00280) scores sampled3D pose hypotheses using learned pose plausibility rather than only reprojection error. It assumes corresponding human joints and thus does not solve our cross-mouse identity problem. A future mouse hypothesis prior needs accurate synthetic3D semantics and source-diverse training, not arbitrary human bone constraints.

Our inference: separate instance correspondence, per-keypoint localization reliability, and3D hypothesis plausibility. Current semantic residual already used uncalibrated entropy/variance and failed; old ROIquality probe predicts boxIoU, not point accuracy. A distinct next diagnostic is to predict calibrated localization quality without moving coordinates, compare scalar-only versus semantic+scalar quality, and test risk/coverage before using weights in fusion. Missing real labels remain unknown, not invisible/negative. Stage calibration must keep scenes/sources grouped and be documented as seen by base pose but unseen by new quality head. No blind step extension.

Report: C:/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005/association_gated_fusion_review_v1/index.html


## 2026-10-05 point-localization quality: monotonic scalar reference beats added heads

Prepared1244existingTRAIN images:1028real,132single-synthetic,84dual. Replaced42dual2view images with84full-view images;74oldTRAIN/VAL dual views replay within1e-5. Omitted21curve-kind images lacking explicit source_group, originals intact. New-stage fit1024/calibration220, outer development144. Calibration: whole realFR3, whole single-syntheticclimb action, dual5frames fromsameTRAINaction. Base detector/pose previously sawTRAIN; this is stage-specific holdout, not globally unseen. Metadata, hashes and masks persisted.

PointLocalizationQuality7tests passed. Target P(2Derror<=32inputpx), NOT visibility. Features7predictedscalars+5pointIDs, optional256semantic->32features. All6queries x5points forward; onlyconfidentoriginalprediction/IoU>=.5/point_valid supervise loss. Unknown/missing labels excluded, never negatives. Matched800steps seeds17/29/43 scalar versus semantic; sameinitialstate/schedule,8real/4single/4dual images. Nominal9697params but only1473active scalar versus9697semantic, explicitly unequal effective capacity. No coordinates, production weights, labels, cameras or boxes changed.

Development787knownpoints,263unknown among1050confidentpointpredictions. Objectness AUC.555/Brier.194; scalar AUC.622-.627/Brier.154-.155; semantic AUC.676-.682/Brier.153-.156. On24knownrealbottomfeet, learned AUC only.521-.556, whereas inverse heatmapstd ranking.875. This tiny conditional subset is not a deployment guarantee; earlier32bottomfeet did not impose currentIoUqualification.

Added deterministic TRAIN-fit references under the same balanced image/point objective: constant probability.80512394 and monotonic sigmoid(7.71045363 -11.91406722*log1p(pixel_std)/log769). Two-parameter weightedBCE L-BFGS-B converged15iterations, no calibration/development labels fitted. ConstantECE.002 withAUC.5 demonstrates lowECE alone is not useful ranking. Monotonic overallAUC.730/Brier.140; real.823/.154; single.710/.158; dual.667/.131; bottom.875/.164. Same monotonic rank as rawstd, no invented probability for uncalibrated rank baseline. These references were added after pilot inspection, not falsely preregistered.

Fit/calibration curves: semantic improves fitting and much of FR3, but worsens heldclimb calibration; do not blindly extend iterations. Risk/coverage plots keep unknowns in selection and report conditionalknownerror. Visual highest-monotonic-confidence wrong foot per cohort: bottom16error56px/P32.649, realcamF16error97.5px/P32.960 (nearface/reflectioncontext), dual15error38.3px/P32.943, single16error40.6px/P32.966. Quality still can be confidently wrong; no confidence-based fusion or pseudo-label promotion.

Next stop stacking comparable small heads. Audit whether current frozen-pixel feature pipeline is the bottleneck and whether a genuinely pixel-feature adaptation experiment is feasible, with strict no-change and gradient checks first. Existing SharedTokenAdapter operates on cached aggregator outputs and must not be confused with adapting image-backbone attention. Current DecoderSpatialPoseReadout detaches native_memory and instance_tokens; Core23SpatialReadout detaches memory and coarse. Simply marking encoder trainable cannot work through this path. Any future gradient bridge must be an explicit experimental opt-in with exact default behavior, no GT crop/count/identity, all138queries, fixed source splits, budget/compute controls and full RGB replay. Inspect previous adapter experiments before repeating them.

Report: C:/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005/point_quality_review_v1/index.html

## 2026-10-05 actual-RGB image-backbone adaptation preflight

The previous cached-feature SharedTokenAdapter experiments did not update the image encoder. An experimental opt-in gradient bridge now permits native memory and instance-token gradients through DecoderSpatialPoseReadout and Core23SpatialReadout; defaults retain detachment. The image backbone's final DINO attention qkv receives a rank-8, alpha-8 low-rank residual, preserving the original masked-bias Linear forward. Only 32,768 new A/B parameters were optimized in the preflight. This is the low-rank residual idea of [LoRA](https://github.com/microsoft/LoRA), not a pretrained external adapter.

Three prespecified TRAIN images (real bottom, synthetic single, synthetic dual top), two updates each with reset: all finite nonzero B gradients, then nonzero A gradients at update 2; all 951,509,921 frozen parameters retain SHA-256 370c87e7efaf493003668ac114e5c7fbdf648974f2b95d136b01cb28a8fdbb2e. Native-path, zero-adapter and disabled-after-update outputs agree exactly under no-grad; original cached parity max 4.768e-7 normalized XY. Peak allocated GPU memory 8.860 GB; measured forward/backward/update 0.075–0.460 seconds per image, excluding RGB loading. This establishes feasibility, NOT accuracy. All trial adapter updates discarded. Five bridge, four decoder and eight spatial-readout tests pass. Minor CPU attention differences across gradient modes are tested within float32 tolerance, not falsely called bitwise equality.

Next preregistered pilot: head-only versus head + image-LoRA, seed17, same 400 updates and balanced 2 real / 1 single / 1 dual image schedule, actual original floating RGB at 768, fixed source-group fit/calibration/development split. Pose decoder + spatial readout trainable; keypoint embeddings/coarse head and original backbone frozen. Fixed original predicted boxes/logits are not GT crops; this is conditional localization, not detector learning. Loss only nose and four feet, after prediction and GT-box evaluation matching. All 6 instance / 138 keypoint queries retained. No new labels/data or back/tail pseudo-GT. LoRA arm has extra parameters/backward cost, so image-update budget is matched, not capacity or wall time. Both final400 checkpoints precede development evaluation; calibration/fit curves every100. No conclusion that400 is converged. No production promotion.

Artifacts: output/singleview_fusion_cycle_20261005/pixel_adapter_preflight_v1 and pixel_adapter_matched_pilot_v1. Training cutoff18:11:41UTC and hard cutoff18:41:41UTC unchanged.

### Matched400 outcome and visual review

Both runs completed at12:42UTC. Fixed known-point development means (baseline / head-only / head+LoRA): all22.267/22.048/21.924px; real31.985/34.461/34.252; single19.225/21.191/20.971; dual20.425/18.697/18.629; bottomfeet43.465/45.853/45.896 (24known). The incremental LoRA change is small (0.124px overall), while both adapted heads harm real/single mean relative to the original. Do not promote or blindly extend this configuration. Reused development, one seed; no significance claim.

Internal calibration has71bottomfeet, mean31.130 original to42.287 head-only /41.723 combined; final100–400 partly recovers its initial jump but not its original level. Training-monitor fit improves while real/single calibration worsens, not evidence that everything just needs more epochs. Match uses only qualified known points, so counts differ from earlier broader conditional metrics.

Actual allocated GPU memory4.051GB head-only versus9.279GB combined; synchronous training compute113.28s versus217.00s, about1.92x. Overall arm time including initial/intermediate calibration218.46/321.56s; shared initialRGB loading and finalVAL excluded. Frozen950,392,993parameter SHA d339a5ee49a80dc86d8d16fa49c763b7d937ef31c5a432489f54fc2193c9303a unchanged. Trainablehead1,116,928 plusadapter32,768. Effective low-rank qkv update norm0.16313, relative originalqkv norm0.001721; actual update is nonzero, not fake frozen-feature training.

Inspected curve chart and seven best/worst cohort RGB comparisons: bottom predictions remain on ventral fur instead of visible feet; side failures remain near ground/opposite foot region. A 'best' incremental LoRA case can still be a large regression from original (bottom64.41→98.59→96.68px; side89.26→180.82→175.59px). Report explicitly uses best/worst LoRA-minus-head, not success showcase. Unsupervised tail/abdomen predictions can drift under retraining; this is coordinate-change diagnosis, NOT accuracy and does not alter source annotations. All-query drift includes background, with separate confident-query statistics in weight_update_audit.json.

Next exploratory arm (after seeing the above, not retrospectively preregistered): freeze the entire original pose head and train only the same rank8 image adapter, identical initialA/B, exact saved400step sampling schedule, loss/LR/weightdecay and source split. This completes the head-adapted/frozen × image-adapter-on/off decomposition; do not mix in new data or a different LR. Intermediate fit/calibration only, final400 development evaluation. No checkpoint promotion. Script tools/train_pixel_adapter_only_20261005.py; output pixel_adapter_only_pilot_v1. Deadline unchanged.

Windows report: C:/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005/pixel_adapter_review_v1/index.html

### Frozen-head LoRA400 and bounded convergence follow-up

Completed same-schedule LoRA-only400: development mean all22.27175 vs original22.26712; real32.12389 vs31.98456; single19.30742 vs19.22536; dual20.36513 vs20.42549; bottomfeet42.33170 vs43.46466 (24known,PCK32still.5). Thus no overall gain. Internal calibration all24.83142 vs24.78982, real31.03574 vs31.02059, single11.82603 vs11.55031, dual20.20648 vs20.34482, bottom30.35428 vs31.12989(71known). Keeping the head frozen avoided the large bottom regression of updated-head arms, but did not solve localization.

Visual review of4arm calibration chart/bottom best+worst/side worst: the best LoRA-only bottom improvement102.68→86.71px remains on ventral fur rather than foot; head-only happened to correct that individual foot to13.05px while worsening other feet. Worst bottom110.52→116.50px and side89.26→125.73px show remaining regressions. Do not choose an architecture based on isolated improvements. Original confident-query unsupervised tailmean drift2.48px, abdomen2.26px; drift is not accuracy, and non-target behavior still changes. Frozen951,509,921weight digest identical; training221.69s,peak9.267GB. Four-arm report at C:/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005/pixel_adapter_four_arm_review_v1/index.html.

Convergence probe, not deployment: resume exact LoRA-only400 checkpoint and AdamW state, keep original head/backbone frozen, new1200updates maximum(total1600), continued same RNG/data pools, constant1e-5 terminal LR floor without restart. All source-file hashes must match previous run; resume RGB-prediction parity checked beforeupdates. Repeated developer validation is already seen; intermediate decisions use only internal calibration and fixedfit72. Every200steps check mean ceilings derived from originalcal: real+2px,single+1px,dual+2px,bottom+3px; also confidentcal unsupervisedtail mean coordinate drift<=8px. Two consecutive exceeded checks stop atcurrentendpoint, no best-VAL selection. Finalendpointdevelopment only. This added budget cannot be claimed a matched400 architecture gain. Script tools/continue_pixel_adapter_only_20261005.py, OUT pixel_adapter_only_continuation_v1. All12hour bounds unchanged.

### 1600 endpoint: stop this optimization branch; audit view coverage

Continuation completed at13:07:09UTC, total1600 cap, no stop-gate violations. Incremental forward/backward/update644.15s, stage includingcal/final evaluation830.27s, peak9.267GB. Original frozenweight digest unchanged. Final development mean all22.1730, real32.2325, single19.3290, dual20.1660, bottomfeet42.7072/PCK32.5. Original means22.2671/31.9846/19.2254/20.4255/43.4647; thus tinyoverall change mainlydual, not stable real improvement. Internalcal all24.6748 andbottom30.2743; comparedto40024.8314/30.3543 onlysmallchanges, fixedfitmonitor21.2092→20.9806. Stop further automatic extension. This does not prove every encoder adaptation is ineffective, only this single rank8,lastimageblock,lowLR schedule. Reviewed curve and fixed400-selected bottom/side RGBs: bottomfoot86.71→86.79 remains onventralfur; side125.73→126.73 remains misplaced. No promotion.

Report: C:/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005/pixel_adapter_continuation_review_v1/index.html.

Physicalcamera audit (BlenderZ-up extrinsics, not camera-number guessing) shows currentfit singlemouse86images =21Cylinderposes ×(bottom+side) plus22Homecageposes ×(top+side). Thus there ARE21syntheticbottomimages, not zero. Currentdualfit64views =16top+48side. Realfit has120camA bottom images but differing label availability; realnonbottom left unclassified geometrically. ExistingmoseqTRAINdatasets eachhave200renderedframes; Cylinder1600images(200bottom+1200side+200top),Homecage800(top200+side600). These are same source/action, not independentvideo additions.

Source audit keeps RealCaptureSurfaceCurve21 excluded from this stage: final31/111/216 originate from the oldClimbWallrig/action; frame31 overlaps metadata mouse_climb_wall_91 frames1–91. Frames111/216 underlyingmotionsegments remain unresolved—do not labelallof them 'climb' solelyfromfilename. Buildscripts explicitly use FR2/AZR2L2 visual/pose references. Preserve originalfiles/checkpoints, disclose reference dependence; not safe independentfit additions.

Prepared all256actualRGBV1views of EXACT same43TRAINposes (21×8+22×4), old86prediction/labelparity passed. Qualifiedknown point counts: Cylinderbottom105/21images,side590/126(8zero-known),top25/21(16zero-known); Homecagetop110/22,side320/66(2zero-known). Topqualification failures require separate detectionaudit, not GTboxcropping toforcepose training.

Next data-only control: original head-only architecture/initialweights/AdamW/400sample schedule; preserve exact real2,dual1 andsinglepose slots. Only draw a different rendered SIDE camera within the same scene+physicalorientation. Preserve all86bottom and111Homecagetop draws,203side draws;148/400singlecamera draws change,9newviews contributezero pose loss because noqualifiedknownpoint, not resampled. Fixed denominator4 andforward allqueries remain. CylinderTOP is not newlymixed because it would confoundorientationexposure. No simultaneousLoRA/newpose/newsource changes. This is side-azimuth coverage, not increased bottom count or learned multiviewfusion. Compare to earlier head-only400, sameevalsets, no productionpromotion. Script tools/train_single_side_azimuth_control_20261005.py, OUTsingle_side_azimuth_control_v1.

### Side-azimuth outcome and original training-recipe audit

Completed13:20:43UTC. Development mean original/head-only/side-azimuth: all22.267/22.048/21.665, real31.985/34.461/33.227, single19.225/21.191/21.094, dual20.425/18.697/18.468, realbottom43.465/45.853/46.644px. Calibration bottom31.130/42.287/44.218. Thus useful local/dual changes do not establish a real or bottom improvement. Frozen950,425,761parameter SHA f7adf454393bc1082a35abef4d105946d5a83177a4766b2eebac3289fa08b2a6 unchanged. Training121.24s, total includingevaluation251.17s, peak4.051GB. No promotion.

Visually reviewed side-best, bottom-worst and dual-worst examples relative to head-only. A side hindfoot genuinely improved180.82→19.02px (original89.26), while a bottom hindfoot9.23→37.74 moved back onto ventral fur (original59.54). A dual foot10.80→23.06 regressed (original37.63). Extreme examples are not occurrence estimates. Report: C:/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005/single_side_azimuth_review_v1/index.html.

Separate TRAIN-only top audit: all21CylinderTOP images have one valid mask-backed instance and five valid nose/feet coordinates. Sixteen fail the original predicted-box IoU>=.5 qualification, not label availability. Ten have two confidentqueries despiteoneGT. Top RGB is markedly darker than bottom/side; detection/domain problems cannot be attributed purely to pose head. These topviews were not added to the side-azimuth experiment. Audit and actualRGB examples included in the report.

Source inspection of tools/train_view_context_control_20261004.py and mousegeoforge/matched_point_loss.py confirms important recipe differences. Original full-context800 uses SmoothL1 beta.01 on normalizedXY*(input_pixels/768), coordinate mean, valid-point mean perinstance then perimage; broader synthetic point targets with dualtail disabled; synthetic+.25real; eight images percohort with bottom-balanced real sampling. Group LRs are spatial5e-5→1e-5 and decoder1e-5→1e-6, AdamWwd.001. Recent400RGB pilots instead use five-point normalizedXY squared Euclidean error, allknownpoints-perimage normalization,2real1single1dual sampling, bothheadLR1e-4→1e-5 andwd.01. These pilots are internally matched, but their regression versus original does not isolate architecture or insufficient steps. This is an audit of distinct recipes, not a claim that MSE is inherently incorrect or that any one difference explains failure.

Next bounded diagnostic: 2x2 loss-recipe × LR-recipe, keep recentfive-point targets, original400schedule/views/initialstate/cohortweights/wd.01/normalization and frozenencoder/boxes fixed. Reuse immutable MSE/highLR400 control; train MSE/lowLR, pixelSmoothL1/highLR and pixelSmoothL1/lowLR400, one seed. PixelSmoothL1 includes aspect scaling, coordinate averaging and robustness; do not market it as isolated robustness. LowLR uses originalgroup rates; other originalrecipe changes remain outofscope. No new data, no full23-point or tail/back targets. Allarmscomplete before finaldevelopment evaluation;fit72/cal220every100. Four unit tests verify exactMSE reference, missingNaN exclusion forHuber, aspect/gradient and unknownrecipe rejection. Localpytest absent, tests use unittest with no installation. Script tools/train_pose_recipe_factorial_20261005.py; OUTpose_recipe_factorial_v1; session70621. No automaticextension orpromotion. Harddeadline18:41:41UTC unchanged.

### Recipe factorial completed13:44:43UTC: learning rate matters, bottom remains unsolved

Order baseline/MSE-high/MSE-low/Huber-high/Huber-low. Development means: real31.985/34.461/31.614/31.563/31.789; single19.225/21.191/20.333/20.388/18.414; dual20.425/18.697/19.693/18.540/19.430; bottom43.465/45.853/41.494/45.027/43.465. Calibration bottom31.130/42.287/34.170/39.874/32.803. Lower learning rate reduces bottom regression within bothrecipes but neitherbeatsoriginalcalbottom. Huber-low bottomdevelopmentPCK32 falls.500→.458 despiteunchangedmean; only24knownpoints. Huber-low realdevelopmentPCK32alsofalls.697→.683. Do not declare solved basedonmean alone. Huber-high improves overallmean partlythroughdual, notsafeallcohorts.

Three newarms training124.07/108.92/106.55s, peak4.051GB, all950425761frozenweightdigestf7adf454393bc1082a35abef4d105946d5a83177a4766b2eebac3289fa08b2a6unchanged. Visuallyinspectedcalchart andHuber-low versusoriginalextremes: bestbottom48.21→32.78 stillofffoot;worst75.19→92.88 staysventralfur;side46.42→82.90 shiftsfurtherfromcorrectfoot. No promotion orblindextension. ReportC:/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005/pose_recipe_review_v1/index.html.

New diagnostic measures accuracy against existing synthetic non-targetGT, not only coordinate drift. All point-validnosefeet masks matchtheexistingqualification; dualtail remainsdisabled; no realambiguousbackGTused. Singlecalother18mean10.205original→12.918MSE-high/14.244Huber-high/11.228Huber-low; dualcal9.983→11.196/11.976/10.655. Singledevelopmentother18mean19.929→24.213/24.947/20.469;dualdevelopment10.451→11.528/11.788/10.822. Individual groups differ: Huber-low singletail21.040→20.249 improves whileabdomen21.586→22.908 andsurfaceback19.894→21.671 worsen. Thus do not claim everypointdegrades or everydriftiswrong. This supports investigating lost multilandmark supervision, not relabeling.

One boundedretention control launched: tools/train_synthetic_retention_20261005.py OUTsynthetic_supervision_retention_v1 session58675. Sameoriginal400schedule/views/initialstate andlowLRHuberprimaryobjective; add.25meanauxiliaryloss on existing qualifiedsyntheticnon-targetcoordinates without changingprimarydenominator. Singleother18, dualother11(tailoff); realauxalwayszero. No pseudo-GT, no annotationedits, no addeddata orarchitectureunfreezing. Fourretentionunit tests passed: realprimaryexactvalue+gradient, auxdoesnotdiluteprimary, forbiddenprimary/real/dualtailtargets rejected, missingNaNexcluded. Fit/calcurvesevery100 andfixed400developmentendpoint. Compare bothprimaryaccuracy andnon-targetretention; ifnegative, stopnearlyidenticalheadprobes. Cutoffs unchanged.

### Retention negative for feet; change the diagnostic target

Completed13:59:40UTC. Huber-low/control vsauxiliary: development all21.453/21.484, real31.789/31.834,single18.414/18.448,dual19.430/19.455,bottom43.465/43.448px,PCK32both.458. Calibrationbottom32.803/32.911, notoriginal31.130. OtherpointGTaccuracy modestlyrecovers: singledevelopment20.469→20.338(original19.929),dualdevelopment10.822→10.588(original10.451);singlecal11.228→11.185(original10.205),dualcal10.655→10.307(original9.983). Notfullrecovery andnotfootfix. Training120.01s,peak4.051GB,frozenweightsunchanged. Reviewedfixedbottomworst:92.88→92.54 remainsventralfur. Stopnearidenticalheadmicro-tuning, notextend400blindly. ReportC:/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005/synthetic_retention_review_v1/index.html.

### Same-current-RGB specialized localizer: strong single-mouse evidence

Audited existing SuperAnimal-Quadruped HRNetW32 seed17, trained150epochs andolddevelopmentbest110. This is the requested quadruped backbone, newlytrainedsevenchannelhead, notTopViewMouse. Existingimmutablecheckpoint SHA8088e7ecc53f4da5951f07a69b3a2fd93556e11246bfb60326cbb6542614733f. No oldcyclemainentrypoint rerun, no downloads, oldSTOP/statepreserved.

Two checks prevented an invalid cachedcomparison. First, originalimageSHAandletterboxaffineagree but teacherderivative usesPILLanczos andfoundationusesPILBicubic (firstuint8meanabsdifference.118,max8), soexactpixelguardfailed. Second, sevenofcurrent48developmentimages areabsentfromold281teachercache becauseeligibilitychanged. These failedpreparations arepreservedinteacher_current_cohort_audit_v1/failure.json andteacher_current_cohort_cache_audit_v2/failure.json. Correctcacheintersection41images/134knownpoints isonlyalimitedreference, notfullcurrentcohort.

Fresh frozenreplay onall48currentBicubicimages plus41oldLanczosimages (tools/replay_quadruped_current_rgb_20261005.py, DLCenvDeepLabCut/.venv-mousegeoforge) usesnativeAlbumentationsImageNetnormalization,FP32GPU,singleimage,eval,noGTcrop/detector/update. Allold41coordinatesreproducecachewithin5.68e-14inputpx. AllmodelstateSHAeaf152768ab8a0917e0a75b4f9ffe93bb689c7df17a7393f8538e35313cc53a0unchanged. Measured89forward/decodeGPUcalls3.808s,excludingloading/preprocessing; notlatencycomparisonwith1Bencoder. Someunknown/unevaluatedteacherchannelsmoveup189pxwhenchanginginterpolation, so do not claim blanketrobustness orautomaticpseudoGTeligibility.

SameCURRENTlabels, sameoriginalfoundationqualifiedquerysubset, same768×704inputcanvas, no teacherTRAINoverlap:142knownnosefeet mean31.9846foundation versus4.5086teacher,PCK32.6972/.9930. Bottom24feet43.4647/4.0039,PCK.5/1.0;side83feet35.2337/4.2944. CurrentlabelsvsoldteacherGTon134sharedpoints meanchange.048px; currentlabelsusedforbotherrors, notoldlabels. This is a system-level diagnostic with different pretraining,trainingduration,real-only/mixeddata,head andoldtraininglabels; NOT architecture-only orcompute-matched evidence. Checkpointselectedonreuseddevelopment, notexternaltest. Singlemouse7pointteacher cannotreplace multiinstance/Core23/depthfoundationmodel directly.

Visuallyinspectedallfourselectedextremes. FR2f0camA17 foundation110.52px onventralfur versus1.03pxatcorrecthindfoot; FR2f1559camF18 156.77versus2.57atcorrectpaw. Worstteacherbottom9.24 andside12.83 stillimperfect; individualfoundation6.48canbeatteacher9.24. Inputimages containlocatablefootinformation; currentfoundationreadout/trainingpathfails toextractit. ReportC:/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005/teacher_current_cohort_fresh_review_v1/index.html.

Next distinct no-trainingdiagnostic: all6originalPREDICTEDROIs conditionfrozenhigh-resolutionHRNetheatmaps onexactcurrentreal+syntheticRGB. Evaluate144fixeddevelopmentviews including64dualviews; noGTcrop/count/identity, no silentdataset-kindgate, no pseudolabelrelease. Mustunit-testpadding/coordinatecenters/defaultoff/emptyROIfallback/querypermutation/non-targetpreservation, reportextraencodercompute andmultimousecollapse. A positive singlemouse result alone cannotauthorize multianimalintegration. Prepare exactloaderfloatinputs ratherthan guessresize. Ifpooronsynthetic/dual, retainasevidenceforfutureinstance-conditionedhighresolutionappearancebranch orTRAIN-onlyteachertransfer, notpromoteaheuristic. Training18:11:41UTC/hard18:41:41UTCunchanged.

### Frozen predicted-ROI specialist completed: reject direct integration

Completed14:25:09UTC; review through14:35UTC. Exported144 exact loader RGB tensors as PNG (roundtrip max absolute error zero), with inference manifest containing only images, shapes, original predicted boxes/logits and original predictions. Evaluation GT is separate. Frozen Quadruped HRNet heatmaps decoded for all6 predicted queries; five nose/foot channels only. Inputs padded right/bottom to32 without resizing; padded heatmap cells excluded. Seven unit tests passed; all144 default-off and other18point outputs exact, frozen model hash unchanged, real48 native outputs exactly match preceding replay. Additional144 HRNet forward/readout calls8.153s, peak allocated253MB; not a matched compute comparison with the foundation model.

Same-qualified-point development means original→ROI: real31.985→13.653 (142points), synthetic single19.225→41.391 (160), dual20.425→85.046 (485), realbottomfeet43.465→5.018 (24). The unconstrained teacher real4.509 was substantially better than hard-ROI13.653. Fourteen real GT points fall outside the original predicted box despite acceptable instance IoU; fifteen known predictions change under ROI restriction, including nose errors exceeding400px. Real points whose GT lies inside the box average4.512px. Thus a detection box is not a safe hard anatomical boundary.

Synthetic failure is not explained by box coverage alone: single11/160 and dual50/485 GT points lie outside boxes, but errors for inside-box GT remain42.663/85.112px. Dual477/485 ROI predictions lie within4px of a box edge,477 global channel peaks lie outside the associated predicted box,192 selected raw responses are nonpositive and median response is1.488e-6 (real median.979). This supports a weak/out-of-domain heatmap diagnosis; raw heatmap values are not calibrated probabilities. Appearance, scale, padding/context and single-animal training remain confounded. Do not assert a causal architecture failure or that simple expansion solves it.

Posthoc dual identity diagnostic: among304 feet with another animal's same-name GT available, closer to the other animal by>5px increases12→55. Confident interquery foot pairs within4px decrease6/256→0/256: absence of duplicate coordinates is not correct identity. Visually reviewed all6 report examples. Real FR2f584camA improves55.775→5.722; a nose cut by the box offsets better feet on AZR2L2f1559camA. Synthetic Washface best17.073→10.769, worst18.895→97.718; even the best dual change19.593→36.147 is a regression, not a gain. Worst dual points move to box corners. No dataset-kind gate, pseudo-GT release or production promotion.

Report: C:/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005/specialist_roi_review_v1/index.html. Code tools/prepare_specialist_roi_inputs_20261005.py, tools/run_specialist_roi_20261005.py, tools/review_specialist_roi_20261005.py, mousegeoforge/models/specialist_roi_readout.py. Preserve all inputs/results.

Next bounded no-training diagnostic: fixed predicted-box crop/context and scale normalization before frozen HRNet, distinct from masking a full-frame heatmap. Prespecify one1.25 box expansion and centered letterbox to704x768 (teacher native canvas); same rule for every image and all6 queries, no GT box/count, dataset-kind routing, confidence-selected fallback or validation-tuned margin. Save source-to-crop affine, test roundtrip, out-of-image padding and invalid-box fallback, and readout excludes letterbox padding. Original boxes/counts/other18 points remain fixed. Six forwards/image changes compute and context; explicitly not architecture or scale-only causal isolation. Evaluate same144 development images once, plus raw dual and bottom RGBs. Do not train/promote if frozen crop fails; then specify a learnable high-resolution instance-conditioned branch using qualified TRAIN labels, keeping geometry separate and original defaults intact. No more near-identical400-step Omega head probes. Original12hour deadline unchanged.

### Predicted crop completed14:39:35UTC: scale/context heuristic also rejected

Session8735/PID3789491 completedexit0. Fixed1.25 expansion, integer expanded bounds with out-of-source black pixels, Bicubic aspect-preserving centered letterbox704x768. Recorded864 source-to-canvas affines in specialist_crop_frozen_v1/geometry.json. Decoder excludes both out-of-source and letterbox padding, backprojects without coordinate clipping. Eight unit tests passed;864/864 valid crops, extraHRNet forward/decode36.689s, peak242MB, modelhash/defaultoff/non-target18parity unchanged. All6queries processed perimage, including lowconfidence; inference read no GT, kind or instance count.

Same144 development: original/hardROI/crop means real31.985/13.653/8.173, single19.225/41.391/68.765, dual20.425/85.046/95.859, bottomfeet43.465/5.018/4.246. Unconstrained fullframe realteacher4.509 still better than crop8.173. Dual closer-other diagnostic12/55/84 of304 knownfeet. Crop did not solve synthetic generalization; do not tune margins repeatedly or select outputs by dataset type. Context and scale changed together; this does not prove resolution never matters or HRNet cannot be trained on synthetic mice.

All6 extrema visually reviewed. FR2f1559bottom44.793→3.311 and AZR2L2f0bottom30.789→5.659 correct foot/nose regions, including formerly cut nose. Washface best25.362→21.528 remains imperfect; worst22.425→166.398 sends foot points above body/at expanded crop boundary. Dual best30.558→46.855 is still regression; worst31.715→193.548 sends points to empty space above overlapping mice. Original predicted boxes unchanged; diagnostics are conditional on original qualified GT association, not full detection AP. No promotion.

Report: C:/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005/specialist_crop_review_v1/index.html. Tools run_specialist_crop_20261005.py, review_specialist_crop_20261005.py; geometry module mousegeoforge/models/specialist_crop.py. Frozen heuristic branch closed.

Primary-source architecture review: [ED-Pose official repository](https://github.com/IDEA-Research/ED-Pose) unifies box-level and keypoint-level contextual modeling for end-to-end multi-person pose; [HigherHRNet official repository](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation) uses high-resolution feature pyramids with multi-resolution supervision/aggregation for scale variation. These human-pose approaches motivate explicit instance conditioning and spatial features, not an expectation that existing mouse weights work zero-shot. No external models downloaded.

Next is a feasibility preflight, not another deployment heuristic: one shared high-resolution appearance feature map per image, conditioned by each original predicted instance query/box through a learned readout; use boxes as soft context, never a hard boundary or GT crop. Keep Omega geometry and original default path unchanged. Local Quadruped config exposes32 backbone output channels before its7-channel1x1 head; investigate that existing feature hook first, without assuming tensor semantics. Cache only a small source-grouped TRAIN subset initially, with known five-point supervision and dualtailoff, plus separate internal calibration; do not use validation to choose hyperparameters. Teacher backbone has seen some TRAIN-derived calibration images and old real development was used to select teacher checkpoint, so disclose this provenance—such a comparison is diagnostic, not independent generalization or a pure architecture ablation. No teacher-predicted coordinate is automatically GT. Before broad feature caching/training, verify RGB/feature/coordinate parity, allquery permutation equivariance, finite gradients, padding/unknown targets, default-off exactness and isolation of original weights. Compare original readout with a matched learnable high-resolution branch; require small TRAIN overfit sanity and source-separated calibration evidence before allocating a bounded pilot. If feature information cannot fit even qualified TRAIN or introduces identity ambiguity, inspect that failure rather than accumulating tiny head variants. Training18:11:41UTC/hard18:41:41UTC unchanged.

### High-resolution branch preflight completed14:57:40UTC: feature bottleneck identified

Prepared42 exact loader RGB images:24fit (8perkind, halfrealbottom) and18internalcal (6perkind, halfrealbottom), sampled seed17 from existing fixed source-stage groups with known points, not selected by prediction errors. No development validation or locked images read. Inference tokens/boxes/baselineXY are stored separately from loss/evaluation GT. HRNet PoseModel.output_features exposes the native stride4 branch before the7-channel1x1 head; local config interpolate_branches=False returns final y_list[0], not concatenated32+64+128+256. Head replay exact0 and FP32 cache roundtrip exact;42images155,385,672bytes,1.374s feature/head forward, frozen model digest unchanged. Source and cache hashes recorded.

New experimental mousegeoforge/models/highres_instance_readout.py: shared32channel normalized visual feature map plus2position channels, learned spatial keys and per-instance/per-point query conditioned on original256token,4predictedbox values,2baselineXY. Dot-product residual to64px Gaussian prior, no hard ROI; output originalXY plus probability-mean change relative to same prior. Zero final query layer initializes enabled output exactly to baseline. Default-off returns input directly. Non-target18points unchanged. Independent loss consumes qualified current five-point GT: pixel/768 SmoothL1beta.01 +.01GaussianCEsigma8px, unknown targets excluded; padded cells excluded with safe0*-inf handling. Nine unit tests pass for zero/defaultoff, querypermutation, non-target, finitegrad, missingNaN, no-knownzero, softbox and zeroappearancecontrol.

Fixed300-step, seed17,3images/update oneperkind, identical schedule/initialstate, AdamW1e-3cosine1e-4 wd1e-4,clip1. Appearance versus zeroappearance (32visual channels zero, but position/tokens/boxes/XY retained): same nominal structure, unequal effective information/capacity, not independent pretrained architecture comparison. Training6.26s/4.62s,peak~243MB. Checkpoints50/100/200/300, no best-validation selection. Baseline/appearance/zeroappearance fit mean real27.093/4.601/21.967,single15.581/14.003/11.164,dual18.583/17.892/16.470. Internalcal real26.206/6.827/32.749,single13.467/32.089/15.887,dual19.639/20.178/20.082. Bottomcal11known27.792/5.252/33.751. Only real passes prespecified fit<=12px and<=half original; synthetic groups fail. No fullcache expansion orautomaticbudgetextension. Original teacher/base have seen parts of underlying TRAIN/cal; these are new-head-heldout internalcal, not external generalization.

Diagnostic of original cached features explains why more readout updates are not the next step. All32channels at every spatial cell inside qualified predicted boxes are exactly zero for8/14synthetic single images (all8fit),13/14dual (7/8fit),0/14real. Median qualified-box spatialzero fraction real.862, single1,dual1; median meanmaxabs feature response real.504,single0,dual0. This is evaluated using original predicted boxes, GTqualification only for grouping statistics. No label edits. It establishes a local feature information loss, not the root cause (BN/domain specialization remains hypothesis). The cached spatial map is a specialized final feature rather than a generic texture representation.

Visually checked fit/cal curves, feature energy panel with fixed logarithmic scale for firstFIT ofeachkind, plus realbottom-best/single-worst/dual-worst calibration overlays. Real FR3f1494 bottom35.780→5.364 moves feet from ventralfur to paws. ClimbWallf10side6.547→51.933 and dualf116view1 21.928→24.855 regress; all6queryoutputs shown, detection unchanged. Remaining three extrema available but not claimed visually inspected. Report C:/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005/highres_preflight_review_v1/index.html.

Next bounded no-update diagnostic: on the24FIT images only, record stage-wise activation statistics for native stem act2,layer1,stage2[0],stage3[0],stage4[0], with direct finalfeature parity to cache. Current local timm source confirms all these are stride4 outputs; verify shapes instead of assuming. Compare originalRGB against a single fixed PIL-L grayscale-replicated input to test color dependence, never select variant perimage/GT. Keep model.eval, all weights and BN buffers frozen; no batch-stat recalibration oncal/development. Save all6predicted-box statistics and a few feature-energy panels, perform GTqualified grouping only after forward. Locate where zeros appear before deciding between an earlier feature tap or a TRAIN-only backbone adaptation. Avoid attributing collapse to BN without evidence, don't extrapolate count rates to all datasets. No automatic wider training or repeated margin/LR variants. Original deadlines unchanged.

### Layer audit and matched pre-ReLU control completed15:13UTC

tools/audit_highres_layers_20261005.py records native act2(64channels),layer1(256),stage2/stage3 and each ofthree stage4units(32), plus32channel input to final fusion ReLU. All are stride4, verified.24FITimages × RGB/PIL-L-replicated gray =48forwards; full weights and BNbuffers unchanged, originalRGB finalfeature exact previouscache. All6predictedbox statistics are recorded without GT; qualification only in later report. Original v1 reported one all-zero GPU boolean reduction as.9999999404 and incorrectly tested float==1. V2 stores integer zero/nonpositive counts and uses counts==cells. Both rawruns preserved; v2 is authoritative, originalfeatures did not change. Do not usev1 table counts. V2run session78636/PID3791074,48forward3.853s.

Before last stage4 unit, none ofthe24FIT images has an all-zero qualified box. At the final unit's preReLU, allchannels are nonpositive throughout all qualified boxes for8/8syntheticsingle and7/8dual RGB; ReLU makes those boxes exactlyzero. Gray does not restore:8/8single and8/8dual zero; real0/8both. OriginalRGB preReLU median spatialstd real2.187,single1.392,dual1.073, so signed features retain variation beforeclipping. This localizes the immediate information loss to final fusion/ReLU and rules out simple grayscale conversion as a sufficient repair; it does not identify the training or BN cause. Visually reviewed both layer progression and gray panels. Report C:/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005/highres_layer_review_v2/index.html.

One causal feature-tap control followed, not a margin/LR search: same42inputs, native32channels before finalReLU instead ofafter, same29792parameter readout/initialstate/schedule300/optimizer/objective. Forall42, ReLU(cachedpre)==originalpost feature exact, headreplay exact, new155MB FP32cache and originalcheckpoint hash unchanged. Same oldbaseline/postReLU/zeroappearance controls reused, not rerun. tools/cache_highres_preflight_20261005.py --pre-relu and train_highres_preflight_20261005.py --pre-relu; session54135/PID3791230complete15:13:14.

Post/preReLU fit means real4.601/4.581,single14.003/11.935,dual17.892/17.862. PreReLU stillfails syntheticfit gates (single must alsohalveoriginal15.581;dual<=12andhalf18.583). Internalcal post/pre real6.827/6.270,single32.089/41.383,dual20.178/27.465,bottom5.252/5.029. Thus signedvariation alone does not give a useful frozen localization representation. No wider cache, deployment or step extension. Reviewed curves andall3calworst images: realFR3f1234camF6.932baseline→5.076pre remainsgood; ClimbWallf10side6.547→54.482 movesnose towardfrontfootregion; dualf95top17.063→36.615 points gatherbetween mice. Nonzero feature maps are not reliable instance localization.

Report C:/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005/highres_pre_relu_review_v1/index.html. Next feasibility question: can controlled TRAIN-only adaptation of the final HRNet fusion unit recover useful local features? First implement isolated copy with native parity, using cached input to that finalunit (all4branches, clone because timm mutates input lists). Keep all earlierHRNet/Omega/originalweights frozen andBNrunningbuffers fixed in eval; no grayscale switching. PreReLU tap must retain gradients (priorcache hookdetached bydesign); test clone/inplace handling and finite nonzero gradients into copiedunit, default-off and baseline parity before any fit. If preflight passes, one bounded300-step control againstexisting frozenpreReLU300, same42images/initialreadout/schedule/loss/readoutLR; only copiedunitweightsunfrozen withseparate1e-5LRcosfloor. Small24FIT/18cal only, no development andno automaticbroadertraining. Report additional trainable parameters/compute, unchangedsourcehashes/BNbuffers andfailuregates. A failure must not trigger repeatedtinyhyperparameter trials. Training18:11:41UTC/hard18:41:41UTC unchanged.

### Isolated final-unit adaptation completed15:27UTC: fitting improves, calibration worsens

tools/cache_highres_unit_inputs_20261005.py exported42 original inputs to lastfusionunit, all4branches32/64/128/256,291,316,950bytes. Inputlist/tensor clones protect against native timm mutation. mousegeoforge/models/highres_fusion_adapter.py deepcopies only lastunit, uses out-of-place captureReLU and differentiable preReLU output. All42 nativepreReLU parity exactly0; originalfullmodel digest unchanged. Copy smoke3steps uses deterministic fake offset targets, notcal labels; unitgradnorm0/6.41e-7/1.28e-6 demonstrates zero-initialized readout delays encodergradient until subsequentsteps. Smokeupdates discarded, pristineunitstate exported. Defaultoff/zeroexact, querypermutation, non-target18, unchangedcache and BNbuffers passed. Extraunit7,059,072parameters; readout29,792.

Runtime audit found foundationenv torch2.11.0+cu128 withouttimm versusDLCenv torch2.7.1+cu128. Therefore reran BOTH frozenunit andadaptunit300 inDLCenv, sameinitial/schedule/objective/42inputs, instead ofconfounding newtraining withruntime. UnitLR1e-5cos1e-6, readoutLR1e-3cos1e-4,wd1e-4,clip1; BNalwayseval, affineparameters allowed onlyadaptarm. Syntaxerror atfirstscriptparse was corrected beforeanytraining/output; py_compilepassed. Session85555/PID3791791completeexit0. Frozenruntime replay vsoldfoundation300maxnormalizedXY.03861 acrossallqueries—notbitwisereproducible trainingtrajectory; currentfrozenarmis authoritative matchedcontrol.

Frozen/adapt fit mean real4.579/4.110,single12.519/7.267,dual17.844/12.776. Original means27.093/15.581/18.583. Adaptpassesrealandsinglepredefinedfitgates, notdual. Internalcal frozen/adapt real6.294/7.392,single41.193/44.306,dual27.345/36.035,bottom4.960/5.307; originalcal26.206/13.467/19.639/bottom27.792. Thus adapting7Mparameters improves memorization butworsens synthetic generalization inthis24fit setting. Close this narrowfinalunit adaptation branch, notextendepochs orpromote. No developmentvalidation used.

Frozen/adapt training8.496/25.789s,peak414/678MB. Frozenunit SHA350a7b397199075a3c8becb88db58bd3f6fc4f816eb11dee893676003413d1da unchanged; adaptedcopyhashchangesonlythatarm, BNbufferSHA3abaf407ba32047d3877407b74712fc0b7fc173bc33a75f9a2682f9138513987 unchangedboth. Originalcheckpoint/cachefileshashunchanged. Visuallyreviewedallcurvesand3calworst: realFR3f1234view5 original6.932→adapt5.503 stillgood; ClimbWallf10side6.547→53.658 nose/feetconfused; dualf95top17.063→43.354 footpredictionsclusterbetweenanimals. Report C:/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005/highres_unit_review_v1/index.html.

Next distinct representation-provenance audit, no additional downloads: existing official SuperAnimal-Quadruped pretrained file is output/bottom_anchor_cycle_20261002/superanimal_quadruped_preparation_v1/superanimal_quadruped_hrnet_w32.pt (118,094,850bytes,SHAa42e58d56bb32f8b4f4fe17ea9ed9511cebc7b7949ac56b97fa6f3a49587c31e, verified15:31UTC). Old preparation report pins HuggingFace mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped revision1ad30fb80cd666f1e5c91578d1cf63bccfa84f80; old preparation script includesdownloads/oldcyclebudget, DO NOT rerun it. Existing teacher best110 used thisbackbone thenreal-only7point fine-tuning. Audit ORIGINALofficialbackbone1830tensors loadedstrictly from checkpoint model keys backbone.* against currentfinetunedteacher, on same smallTRAIN cohort. No decoder semantic mapping or automaticpseudoGT; pretrainedheadnames differfromCore23 so comparefeaturesonly. Keep checkpointsreadonly and normalization/canvas fixed. First measure final32channel/stagewise response andnorms, verify sourcehash/backboneparity/frozenbuffers, then decidewhether matchedsmallreadoutcontrol is justified. Nonzeroactivations alone are not accuracy. Distinguish pretraining provenance from architectural changes and don't label genericpretrain asalreadyvalidated BottomView. Original deadlines unchanged.
### Official pretrained backbone and visibility audit completed 15:52 UTC

No downloads: strictly loaded all 1,830 backbone tensors from the existing official SuperAnimal-Quadruped HRNetW32 checkpoint (SHA a42e58d56bb32f8b4f4fe17ea9ed9511cebc7b7949ac56b97fa6f3a49587c31e). No pretrained decoder-to-Core23 mapping assumed. tools/audit_official_quadruped_features_20261005.py exported the same 42 FP32 post-ReLU features (155,385,672 bytes), 2.325 seconds; original backbone digest da27531a0f817952050b3bcce1a0ed79a9b87ba8314e6e0a481fc8e0359dc453 unchanged. First v1 failed before inference because DLC BaseBackbone.eval() returns None; fixed by separate cuda/eval/requires_grad calls, v2 succeeded. Failed outputs preserved.

Qualified-box final-feature all-zero FIT counts finetuned/official: real 0/0 of8, single 8/0 of8, dual 7/0 of8; calibration dual 6/0 of6. Original pretrained features retain synthetic spatial variation, but nonzero activation is not localization accuracy. Visually reviewed real/single/dual energy comparison. Report official_quadruped_feature_review_v2.

One matched readout300 control used the same42 inputs, same29,792 parameters, initialization, schedule, objective and torch2.11 environment as original post-ReLU readout. Only frozen feature source changed. No new development validation. Official FIT mean real12.688/single10.053/dual12.758; all3 fail predefined <=12 AND <=half-original gates. Official CAL real37.143/single21.765/dual22.866/bottom40.706, versus original26.206/13.467/19.639/27.792 and finetuned-feature6.827/32.089/20.178/5.252. Training-fit improves in synthetic domains but generalization fails; no extension, broad cache, production promotion, or claimed architecture win. Original pretraining differs from real-only fine-tuning, not an architecture ablation. tools/review_highres_control_20261005.py --official exports official_quadruped_readout_review_v1. Curves and all3 CAL-worst RGB examples visually inspected: real1130 original6.932→33.254, single248 6.547→19.798, dual1319 24.622→34.193. Nosetip/foot drift remains visible.

tools/audit_highres_visibility_20261005.py then reloaded the same42 examples, reran exact box-only evaluation Hungarian associations, and asserted coordinate-valid/GT parity with cached targets. All original source hashes match; additional keypoint_occlusion.npz hashes now recorded (original export did not fingerprint those sidecars). Explicit visibility_valid & visible used; missing visibility stays unknown. Real53 known coordinates all visibility-unknown, NOT marked occluded. Synthetic FIT single37visible/3occluded, dual69/11; CAL single23/7, dual45/10. No label edits or inference.

Visible-only CAL error original/finetuned/official: single13.122/32.183/20.696; dual19.172/19.872/20.915. Occluded CAL original/official single14.603/25.279, dual21.742/31.646. Thus regression is not solely from occluded targets; most FIT points are explicitly visible. All4 FIT/CAL single/dual visible-worst overlays visually inspected. Report highres_visibility_audit_v1. These are small conditional subsets, not dataset-wide rates or AP; adapt_unit metrics in this descriptive audit use a different runtime than other frozen-feature controls.

Next distinct diagnostic: separate local semantic evidence from instance-query conditioning. Before further model/epoch expansion, test a small query-independent, 5-channel full-frame point heatmap probe on the existing frozen feature caches. Identical32channel+XY shallow convolutional head, same initialization/schedule/300steps for finetuned versus official features; fixedTRAIN subset, no new development inputs. TRAIN heatmap targets may use explicitly visible synthetic points and coordinate-valid real points with visibility recorded unknown; do not redefine visibility labels or remove occluded coordinates from main model supervision. No GT boxes/count/type routing in forward. The diagnostic can use class-wise Gaussian mixtures and fixed top6 peak extraction; evaluation association ONLY after forward. Report target heatmap mass/loss, nearest-fixed-top6 candidate distance explicitly as optimistic candidate coverage, NOT final instance pose/AP, plus single-instance top1. This asks whether local appearance supports point-type evidence independently of the instance token; success would justify an instance-conditioning redesign, failure would justify mixed-domain representation training rather than more same-head steps. Preserve all defaults/checkpoints and deadline 18:11:41 training /18:41:41 hard UTC.
### Query-independent evidence probe and mixed-domain preflight, 16:10 UTC

Implemented mousegeoforge/models/local_point_evidence_probe.py, a 19,973-parameter 32-feature-channel+XY / 64-channel3x3 / five-output1x1 local classifier. No tokens, boxes, instance counts or data-domain labels enter forward. Gaussian-mixture CE sigma8px uses explicitly visible synthetic coordinates and coordinate-valid real points (visibility remains unknown). Fixed6 peaks per point type with8px suppression, padding excluded. Eleven tests passed: noGT API, permutation/duplicate invariance, unknown/NaN masking, normalized mixtures, finite gradients, padding and peak separation. This is a diagnostic, not a pose model or production default.

tools/train_local_evidence_probe_20261005.py completed both300-step arms in session27824. Training finetuned/official6.434/4.539s, peak351MB. A final read-only hash check had ROOT[path] instead of ROOT/path, after all checkpoints, predictions and results had been saved. Original failure retained. Corrected helper --verify-completed independently verified84feature hashes, both300 endpoints/curves and all42 probability maps perarm; no training rerun or result substitution. Status complete_verified_after_finalization_error.

Finetuned/official eligible-point CAL nearestTop6 means: real2.088/190.749, single68.600/116.642, dual249.270/21.245. Single-instance top1 real2.441/322.343, single75.804/204.669. FIT nearestTop6 finetuned/official real2.369/148.387, single296.734/52.712, dual212.277/69.880. Real has visibility unknown, synthetic onlyexplicitvisible included. Top6 nearest is optimistic candidate coverage, not instance localization; one-to-one posthoc candidate matching is also reported. Some nearby candidates merely occupy the wrong foot/type/instance; no AP claim.

Visually reviewed both curve panels and all6 fixed first-of-kind FIT/CAL examples, not selected by error. Real bottom1055 fine-tuned response tightly covers nose/4feet; official response includes glass rim/background. Synthetic single316 finetuned map nearlyuniform with peaks atimageboundary; official map respondsaroundmouse. Dual1262 official candidates coverbothanimals but still overlap/typesambiguous; finetuned peaks lie nearframebottom. Domain-dependent evidence exists without instance queries, so query assignment alone is insufficient explanation. Stop frozen shallow feature probes. Report C:/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005/local_evidence_review_v1/index.html.

Prepared a genuinely broader mixed-domain visual training control rather than extending24-image fitting. tools/prepare_mixed_local_training_20261005.py exported362exact-roundtrip PNGs,106,908,439bytes: FIT128real (halfphysicalbottom),86single,59dual; internalCAL32real,38single,19dual. Original42 retained for parity, other real rows seeded17, no error-based selection. All original eligible synthetic FIT/cal rows included; existing source-stage groups preserved, no developmentVAL/locked sources read. Training uses ALL eligible instances in an image, not merely detector-qualified objects, to avoid discarding a secondmouse from dense supervision. Original qualified GT/valid parity checked; missing staysunknown. FITeligiblepoints521real/364single/454dual, CAL131/144/150.

mousegeoforge/models/mixed_local_evidence.py wraps an isolated officialHRNet backbone plus the same19,973head; full29,305,536backbone parameters trainable onlyinadaptarm, BN runningstatistics always eval. Five more tests passed (16total withprobe). Actual-RGB preflight session68002 exited0: all42 oldofficial nativefeature differences exactly0, source/hash restore exact; discarded3-step smoke backbonegrad norms0/.09894/.26817, BNbuffersunchanged, peak2.101GB, perstep1.194/.428/.400s. Smoke checkpoint NOTsaved; pristine readout initialization only. No pretrained decoder/Core23 semantic reuse.

Started tools/train_mixed_local_backbone_20261005.py in DLC torch2.7.1 environment: session13618, PID3794112, OUT mixed_local_backbone800_v1, exclusiveGPUlock. Both arms same actualRGB, pristineofficialbackbone, freshsamehead,800steps/seed17/schedule3images(step: real/single/dual), GaussianCE; readoutLR1e-3/backboneLR1e-5 cosinefloor.1,wd1e-4,clip1. Frozen backbone control also rerun onactualRGB in SAMEruntime; no confounding witholdtorch2.11/24fit results. Originalweights never overwritten. Fixed800 endpoint first; evaluate all362 at0/200/400/800, savecurves/compute/sourcehashes. Checkpoints onlynewresearchdirectory; fullmaps retainedonlyoriginal42 forvisualizations. At16:10:07 frozenarm200 completed and evaluating. Do not start duplicate job. Inspect matched curves and real-bottom/single/dual RGB before any extension or integration; local candidate coverage remains distinct from final multiinstance accuracy. Trainingstop18:11:41/hard18:41:41UTC unchanged.
### Mixed800 completed16:16:55UTC; diversity-versus-steps continuation started16:23

Session13618/PID3794112 completedexit0, nofailure. Frozen/adapted backbone training83.000/295.940s, peak2.369/4.514GB; sameinputs362, sameDLCtorch2.7.1, identicalinitialofficialbackbone/readout and800image-update schedule. Frozenbackbone digest da27531... unchanged; adaptedcopy18594fc7cd3a5587ce595ceac8c00c34f99c5572cffa6bcc5fec2a05ca753692. BNbufferdigest1f3841e8573241f20e09ad53362b278e2bb1689906f3e9c7b81efa777017acfc unchangedboth; originalweights/PNGhashesunchanged.

Eligible-point FIT nearestTop6 frozen/adapt real139.673/13.237, single62.432/7.138, dual62.293/9.811. CAL131real/144single/150dual points: nearestTop6 real184.376/52.456, single58.919/9.761, dual74.508/15.643. One-to-one dual15.650, versusnearest15.643. CALTop1 real247.052/92.704, single81.237/20.646; these remain much poorer than desirable pixelaccurate realpose. Candidate coverage is optimistic and NOTfinalinstanceassignment. ExplicitoccludeddualCAL33.030nearestTop6 versusvisible15.643; mainmodel occludedcoordinate supervision unchanged.

Adapted400→800 FITrealnearest24.552→13.237, CALreal54.668→52.456, CALrealTop1101.441→92.704, loss6.609→6.570. CALsingle11.439→9.761, dual17.330→15.643. Backbone learning helps all3domains versusmatchedfrozencontrol, but realgeneralization gap remains. Do notcallfullyconverged orintegrateasfinishedfoundationhead. Reviewed bothcurvepanels andall6fixedFIT/CALRGB+heatmap examples. Realbottom1055: responseincreasinglyfallsonthepaws but pointtypes/left-right stillconfused andsomeclasspeaks remainonglass. Synthetic1266 seesbothmice afteradaptation, unlikefrozenone-mouse response; cross-instanceassignment stillunresolved. Report mixed_local_backbone800_review_v1 created by review_local_evidence_probe_20261005.py --mixed.

Primary-reference check: [SuperAnimal paper](https://www.nature.com/articles/s41467-024-48792-2) and [official fine-tuning tutorial](https://deeplabcut.github.io/DeepLabCut/examples/COLAB/COLAB_YOURDATA_SuperAnimal.html) distinguish pretrained-encoder+freshdecoder transfer from decoder-preserving fine-tuning, and describe confidence-filtered memory replay for unannotated landmark types. Our current method is encoder transfer, NOTtheir full-decoder memory replay. Existing localpreflight explicitly declined paw-center versusendpoint equivalence and dorsal6/7mapping, so do notblindlyreusepretrainedchannels orconvertconfidentpredictionsintoGT. Here verifiedreal/syntheticlabel rehearsal is available and is preferable to inventing missing labels from noisybottom-view predictions. Domain specialization observed locally is consistent with a retention concern but notproof ofa single forgettingmechanism.

Decision: separate extra optimization from real-sample coverage ratherthan extendingonlyonearm oradding multiplechanges. tools/prepare_mixed_local_training_20261005.py --full-real completed session45117exit0; newmixed_local_fulltrain_rgb_inputs_v1 containsoriginal362unchanged plus693additional existingFITrealimages, total1055PNGs334,061,227bytes. FIT821real+86single+59dual, CALunchanged32/38/19. OriginalqualifiedGT/validparity andRGBroundtripexact; no newvalidation/lockedlabels orsyntheticchanges.

tools/train_mixed_real_pool_control_20261005.py started session50636 at16:23UTC, OUTmixed_real_pool_control_v1. Botharms start from SAME adaptedmixed800checkpoint INCLUDINGAdamWstate, trainallbackbone/head, BNstatsfixed; keep final800-step LR constant (no restart). Append800updates (absolute800→1600). Controls same_pool128real versusexpanded_real821availableFIT images; BOTH haveexactlysame800syntheticdrawpairs and400bottom/400nonbottomreal draws. Availablepool size is NOTnumberofuniqueimagesactuallysampled; protocolrecordsunique counts. Evaluateonlycommonoriginal362 at800/1000/1200/1600, fixedend1600; no newcal selection. Initialall362peakcoordinates andold42probmaps must replay800exactlybeforetraining. NewremainingrealFITimages are notexternalvalidation. Imagesloadonlyscheduledorcommoneval, nohugefeaturecache. OriginalcutoffsTRAIN18:11:41/hard18:41:41UTCunchanged. Inspectstatus/PID beforedispatchinganythingelse.


## 16:39 UTC — real-pool control completed; rotation hypothesis isolated

Paired same_pool/expanded_real continuations both reached1600, session50636 exit0. Both exact800 peak/probability replay, sourcecheckpoint/PNG hashes and frozenBN guards passed. Available real pools128/821; actual unique sampled128/406; same800 syntheticdraws and400/400 bottom/otherdraws. Real CALnearestTop6 51.553/47.728px; Top1 88.576/88.896px, compared baseline800 52.456/92.704. Bottom Top1 baseline118.182, same111.579, expanded113.848; other56.375/55.774/53.315. Thus expandedcoverage modestlyhelps candidatecoverage andCE but does NOT establish better finalpointselection. RealCALCE6.594/6.283; syntheticCALnearest single9.363/9.031, dual17.138/15.093. No more same-pool lowLR continuation justified by these curves alone.

New posthoc audit tools/audit_mixed_real_pool_20261005.py exports perpoint/perview Top1, nearestTop6, PCK8/16/32, median/p90 and a conservative opposite-foot-proximity diagnostic. 32realCALimages131knownpoints across19sourceframegroups; actualvisibility unknown, not relabeled. Bottom opposite-closer flags baseline29/62, same30/62, expanded29/62. Require bothfootGTknown, separation>10px, prediction<=32pxfromopposite and5pxcloserthanown; this is not proof ofanatomicalswap. Descriptive paired scene bootstrap expanded-minus-same Top1CI[-6.983,7.731]px, reusedinternalCAL not independent significance. Report mixed_real_pool_breakdown_v1. All4 deliberatelyselected mostimproved/regressed RGBs inspected: bottom1041 noseimproves but bothfootclassescollapse; bottom1153 fore/hind-LR stillconfused; side1100 expandedrecoversleftrear whileunknownnose canfireonbackground; side185 noseonrearfoot and pairedfoottypescollapse. Do notconverttheseunknownpredictions tolabels.

All6fixedFIT/CAL RGB+heatmaps andbothcurves alsoinspected in mixed_real_pool_review_v1. FIT675 feetreasonable; CAL1055 forefeetlabelsreversed andhindcandidateonglass; synthetic189/1266/246/1262 localresponsesmostlyaroundmicebutinstanceassignmentunresolved. Reports copiedtoWindowsDownloads andlinkedfromindex. No modelpromotion or annotation edits.

Next bounded hypothesis: orientation dependence. Added mousegeoforge/models/quarter_turn_evidence.py and7unit tests (identity, exactpixelcorrespondence, roundtrip, normalizedblackpadding, missingNaN, noidentitypermutation, invalidturn). tools/preflight_quarter_turn_20261005.py passed3864FIT coordinate transforms (966images*4) and threefirstFITdomains visualoverlays checked at0/90/180/270degrees. Pixel-index/W,H convention, torch.rot90 no interpolation/cropping/mirror; anatomicalIDsunchanged. No camera/3D losses are used inthisisolateddiagnostic.

Started tools/train_mixed_rotation_control_20261005.py session66604 at16:39UTC, OUTmixed_rotation_control_v1. Botharms SAMEoriginalmixed800checkpoint+AdamW, SAMEexpanded821real schedule (406uniqueseen), same syntheticdraws, sameLR/BN/additional800updates. native_rgb versus quarter_turn (fixedseed43 uniform0..3 pertrainingimage). Evaluation remains unrotatedoriginal362; no newCAL/VAL. Nativearm reproduces previous expanded_real control; verify endpredictions/digest match ifpossible. Trainstop18:11:41/hard18:41:41UTCunchanged; heartbeatpymousestudio verifiedACTIVE15min withsameUNTIL. Read actualstatus/PID and do NOT duplicate this job.


## 16:53 UTC — rotation1600 mixed; longer matched trajectory started

mixed_rotation_control_v1 completed session66604 exit0, native_rgb/quarter_turn both1600. Allinitial800parity, frozenBN, sourcecheckpoint andPNGhash guards passed. Native repetition is not bitwise training replay: tools/verify_rotation_native_replay_20261005.py found maxweightdelta backbone8.97e-5/readout8.63e-4 versusprevious expanded_real; CALmeanTop1predictiondrift real0.025/single0.126/dual0px. Large rank-slotTop6maxdrift reflects weakpeakranking changes, NOTGTerror/candidate-setshift. Sameinitialcheckpoint andexactdrawschedule verified; contemporarynative isprimarycontrol, notclaimdeterministicalgorithms.

CALnative/rotation realTop188.892/88.027, nearestTop647.660approximately/45.396; bottomTop1113.842/107.498, other53.315/60.263. BottomPCK16 31.17/35.06%, opposite-closer29/62 vs28/62. SingleCALnearest9.09/12.439, Top1rotation27.989;dual15.09/14.998. Rotationactualaugmentedtrainingloss recent25 drops6.167early to5.613late, but unrotatedFITrealCE worsens5.064→5.204 (do notconflateaugmentedobjective withnativeFITmetric). RealCALCE6.570→6.132 versusnative6.283; singleCALCE worsens5.149→5.279, dualimproves5.780→5.618. Effectmixed, notadopted. Bothcurves+all6fixedFIT/CAL panels visuallyinspected. FIT675 rotationcreatesrearclasscollision; CAL1055 LR/glassfaultsremain; single246forepawtypes switch;dual1266blacknosecandidateappearsbutgroupingnotimplemented. Fourselected extremesalsoinspected: bottom172recoversrightrearfromglassbutnose/hindleftstillwrong;1104nosemovesonglassandforepawIDswaps;side1159nosepeakcorrectbutfootLRstillwrong;side1100 losespreviousleftrear, collapsestypes. Reports mixed_rotation_review_v1 and mixed_rotation_breakdown_v1 Windowsmirrored.

Additionaldiagnostic tools/audit_mixed_foot_identity_20261005.py (--rotation optional) comparesidenticalfourTop1footpositions withGT-oracleone-to-onefreefootlabelassignment. Previous expandedrealbottom63knownfeet typed127.88px→oracle84.13px; untyped24candidateunion27.02px/PCK16 82.54%. Bothsemanticconfusion andpeakselection/missinglocalization remain. Oracle usesGT AFTERinference, cannotdeploy orchangeannotations; any24alsochangescandidatecountandpermitsreuse. Preserveallcaveats.

Important provenanceaudit: tools/audit_teacher_calibration_overlap_20261005.py foundcurrent32realCALimages/19sourceframes allcorrespondtosourceframesinoldspecialistTRAIN; oldteacherhas154FR3trainentries, explicitFR3f194/f324/f1494camAtrainconfirmed. Thus earlierfine-tunedteacher realCAL2–4px is NOTunseen-FR3generalization baseline; labelsversionsalso differ. Do NOTinitialize currentFR3heldoutmodel fromthisspecialist, and do NOT infercurrentmixed88pxcanbefixedbydirectcheckpointreuse. OriginalofficialQuadruped initialization usedinallmixedexperimentsdoesnotusetheseprojectlabels. Artifact teacher_calibration_overlap_v1/results.json.

Decision: noarchitecture orLRswitch. Fixedpairedextension1600→4000 adds2400updateseach toallowrotationadaptationbeforeclosinghypothesis. tools/continue_mixed_rotation_20261005.py started16:52UTC, session81906 PID3797039, OUTmixed_rotation4000_v1. Eacharmcontinuesitsown1600model+AdamWtrajectory; their COMMONANCESTOR isoriginalmixed800, notidentical1600models. Same821realpool/identical2400drawschedule,1200bottom1200other, actual681uniquerealimages, syntheticdrawsexactlypaired; seedsreal53/syn61/rotation71, augmentationotherwiseunchanged. Learningratesheldsame, frozenBN, fullbackbone+localhead; evalunrotatedcommon362 at1600/2400/3200/4000, fixedfinalnotbest-stepselection. Initial1600 replaymustexact beforeupdates. SourceSHA nativea9787530573bc056d27e1fc3a392ec8af6f2cf3a41ae7968f0f27abc283314c5 / rotation1049bd1e9608dd98f9e8d4efe7e30bc5d1f83d125fcfe0b75551923406bc972d. SameGPUlock, perstepcutoff18:11:41UTC andhard18:41:41; noextra12h. Nativeformatreportsnowhave--rotation for1600; addseparate--rotation4000 withdistinctoutputnamesandcorrect4000protocoltext ratherthanoverwriting1600. Confirmactualstatusbeforeanythingelse. Current23local/mixed/rotationunittests pass (5+11+7); no defaults/data overwritten.

### 17:01 UTC monitor and reporting preparation

Same session81906/PID3797039 active; native2500at16:58 and2750at16:59:49, nofailure, initial1600paritypassed. Interimnative2400CALrealTop183.95/CE6.20, syntheticnearestsingle7.87/dual16.47; notfinal, noexperimentdecisionyet. Prepared --rotation4000 options in review_local_evidence_probe_20261005.py, audit_mixed_real_pool_20261005.py and audit_mixed_foot_identity_20261005.py. Distinct4000outputfolders preserve1600reports. Fullcurvesjoincommonancestor800through4000 withduplicate1600removed; metadataexplicitly saysdifferent1600trajectories share800ancestor. Compilechecks passed; rerun1600breakdownregressionpassed withnative1600/rotated1600keys. Sortedbootstrapclusterkeys nowmake resampling stable acrossPythonhashseeds; descriptiveCIonly, no pointmetricchange. No trainingconfig, deadline, originaldata orproductionchange.

### 17:20 UTC — fixed final development verification preregistered

Training stillsamePID3797039/session81906, nofailure. Native4000completed withCALrealTop189.26/CE6.19, singleTop69.08/dual16.37; initialimprovementat2400 didnotpersist. Quarterturn3000at17:16; its2400realTop178.74/CE5.93 isINTERIM, notfinal. GlobalGPUmemory10699/97887MiB, utilization55%; WSLperprocessmemoryN/A. No changes totraining.

Preregistered final-only fixed800/native4000/rotation4000 comparison at17:15:40UTC beforeboth4000results. tools/prepare_mixed_local_development_20261005.py session56352exit0 generatedALL144originalvalidationrows without errorselection orbaselinedetectorqualification:48real(FR2:28,AZR2L2:20),32singleWashFace,64dualactionB. ExactRGBPNGroundtrip37,911,209bytes; known/eligiblepoints175real136single496dual, oneempty-real-labelimage retained. Allsource-scenesdisjointfromcurrentmixedFIT/CAL, but theseareREUSEDdevelopment sources frompreviouswork, NOTpristineexternaltest. LockedGR2/UR2/UR3notread. SeparateRGBmanifest/evaluation_only.pt, no GTROI orcount/pointinput.

Prepared tools/evaluate_mixed_local_development_20261005.py (compiled), runONLYafterboth4000completeandGPUlockfree. Threepredeclaredfixedcheckpoints, nooptimizer/noadaptation, sameDLCFP32nativepipeline, model/BN/source/PNGhashguards; deadlinehardminus5min. Reportsfullpointandfeet-only, source/bottom/otherandvisibilitygroupmetrics; empty-labelimageexcludedfromCEaverageonly, unknownnotinvisible. Originalmodel/imageinputsunchanged. AfterthisdevelopmentcheckNOpost-developmenttraininginthiscycle; onlyread-onlydiagnostics,visualreview,research,andfinaldelivery. Thisprevents choosingfutureweightupdatesfromnewdevelopmentfeedback. Neednewthree-armvisualreportbecauseoldreportassumesfit/calstages. Existing4000CALreportoptionsreadyanddistinctoutputpaths. SameTRAIN18:11:41/hard18:41:41UTC.

## 17:38 UTC — fixed 4000-step endpoints completed; no further training this cycle

Both continuations completed normally at 17:22:31 UTC (session81906 exit0). Native additional2400 steps took884.94s; rotation905.25s. Initial1600 exact prediction replay and final source/BN/PNG guards passed. All six fixed FIT/CAL RGB+heatmap panels, two curves and four deliberately selected CAL extremes were visually reviewed. Native FIT continues improving while real CAL plateaus; rotation improves real CAL but has no consistent synthetic benefit. Final CAL native/rotation: real Top1 89.262/80.037px and nearestTop6 51.642/33.800px; single Top1 17.269/30.505px; dual nearestTop6 16.369/17.171px. Unrotated FIT CE and augmented training CE are different objectives.

Final development evaluation (session86581 exit0) used all144 preregistered images and only the three fixed endpoints. All model/BN/source/PNG guards passed, no optimizer or adaptation. It is reused development data, not independent external validation; no locked GR2/UR2/UR3 images read. Units below are input pixels at768 long side, not source pixels or millimetres.

| Development metric | Original800 | Native4000 | Rotation4000 |
|---|---:|---:|---:|
| Real known175 points, Top1 mean |52.76|35.79|32.16|
| Real bottom feet32 points in8 images, Top1 mean |100.87|84.42|75.07|
| Real bottom feet PCK16 |37.50%|46.88%|56.25%|
| Synthetic single visible136 points, Top1 mean |48.84|42.37|48.16|
| Synthetic dual visible496 points, nearestTop6 oracle |18.39|20.74|16.56|

NearestTop6 and one-to-one Top6 are GT-posthoc candidate-coverage diagnostics, NOT deployed pose accuracy, instance association or AP. Five heatmap classes have no instance assignment. The eight bottom images are few correlated frames; no claim of statistical significance or broad generalization. Source-specific results also differ: rotation improves FR2 but worsens AZR2L2 relative to native. No production checkpoint promotion.

All13 development panels reviewed: all8 real-bottom images plus fixed real side1161, single1208/1209 and dual1324/1325. Bottom1175 recovers a hind foot from glass and is plausible;1165/1170 improve forefoot typing but hind channels collapse;1187 remains severe;1194/1201 trade one recovered foot for another channel collapsing on it. Side1161 still concentrates multiple classes at one foot. Synthetic single1208 roughly localizes the animal but1209 has typing/background candidates; dual1324 rotation recovers black-mouse nose evidence while1325 still lacks reliable second-nose coverage. Do not interpret the largest global peak as each animal's output. CAL1090 improves by removing hind-foot peaks from glass but retains class collapse;1111 regresses onto glass;1159 corrects hind-foot typing;1100 loses the left hind-foot Top1. These four CAL examples were selected by error difference and are not representative sampling.

The oracle foot-identity audit remains diagnostic only: on63 CAL bottom foot points, native typedTop1 127.87px -> free-label one-to-one oracle87.60px -> any24 candidate union26.64px; rotation116.20 ->81.91 ->17.16px. This supports both semantic assignment and candidate-selection/localization failures, not a deployable relabeling procedure. Any24 also increases the candidate count and permits reuse.

Decision: no further training after this development check. Remaining authorized time is for read-only audits, evidence packaging and next-experiment design; deadline remains18:41:41UTC. Do not extend or automatically promote. Preserve original annotations, R12 old mask, tails/abdomen, pending/excluded groups, old cycles, and model defaults.

Artifacts: mixed_rotation4000_review_v1, mixed_rotation4000_breakdown_v1, mixed_rotation4000_identity_audit_v1; mixed_local_development_inputs_v1, mixed_local_development_eval_v1, mixed_local_development_review_v1. Windows Downloads mirror has clickable image reports; metadata explicitly distinguishes known real coordinates from visible synthetic GT.

### Next-round design, not executed after development evaluation

The following is an experimental proposal inferred from current failures, not an established improvement.

1. Freeze this round's three endpoints and splits. Before new training, audit candidate extraction on FIT/internal CAL only: strict local maxima versus current radius8 greedy suppression, fixed candidate budget, duplicate-mode count, per-foot recall and one-to-one coverage. Current Top6 can spend multiple slots near one mode. Do not replace old predictions/metrics; do not select radius/threshold on the144 development results.
2. Separate localization, anatomical type and instance membership. The preferred bounded architecture test is the existing trainable high-resolution local branch plus explicit predicted-center/instance-conditioned grouping, compared against identical local branch without grouping. Do not initialize from the real-only specialist that trained on FR3. Preserve RGB-only inference and no GT boxes/count/association. Include unmatched/null outputs so a missing foot is not forced onto the other foot or glass.
3. Use reliable existing instance supervision and only available endpoint coordinates for grouping losses. Unknown endpoints produce no negative edges; a second animal is not background. Do not introduce uncertain real dorsal6/7 as mandatory anchors. Anchor definition must be explicit and stable under missing labels; a changing subset-centroid is not automatically the same target. Real approved masks may support an instance center, but missing/mismatched masks must be audited and cannot be silently replaced by rejected R12 repair.
4. Compare a center/keypoint association head with a modest structured within-instance decoder, not another frozen-feature gate on unchanged inaccurate points. Report detector recall, duplicate instances, keypoint type errors, candidate recall, grouping accuracy conditional on available candidates, and full per-instance pose error. Use identical train images/updates and report extra parameters, forward passes, runtime and memory. No GT permutation oracle as deployed score.
5. Only after single-view/local grouping passes, test optional cross-view residual with strict default-off parity and no-support fallback. A calibrated camera cannot resolve a consistently mislabeled paw by itself. Temporal modules remain separate and use only reliable continuous sources; do not treat diverse-pose Blender samples as consecutive motion.

Research anchors, checked2026-10-05:
- [SLEAP, Nature Methods2022](https://www.nature.com/articles/s41592-022-01426-1) separates localization and grouping, implements both confidence maps plus part-affinity fields and centered-instance top-down models. Its bottom-up/top-down terms describe algorithms, not camera orientations. Its comparative results do not establish one universal winner, particularly for homecage scenes.
- [CenterGroup, ICCV2021](https://arxiv.org/abs/2110.05132) groups identity-agnostic keypoints around predicted person centers using attention. This is a relevant mechanism to test, not evidence that a human-trained model resolves mouse paw anatomy.
- [Group Pose, ICCV2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Group_Pose_A_Simple_Baseline_for_End-to-End_Multi-Person_Pose_Estimation_ICCV_2023_paper.pdf) separates within-instance keypoint interaction from same-type cross-instance interaction. Current local experiment has neither full instance queries nor equivalent grouping; the earlier failed topology readout must not be renamed as this design.
- [DeepLabCut architecture documentation](https://deeplabcut.github.io/DeepLabCut/docs/pytorch/architectures.html) describes HRNet plus heatmap/PAF multi-animal variants, DEKR center offsets, and BUCTD pose-conditioned refinement. BUCTD is an alternative only once useful instance-specific proposals exist; the current ungrouped Top6 union is not already a valid pose proposal. No packages downloaded or architecture switched here.

Priority is therefore candidate integrity and explicit grouping with trainable visual evidence, not unlimited step increases or enabling geometry on unreliable semantics. Longer training helped some real outputs but did not produce an acceptable mixed-domain model. New experiments require a new declared protocol; no more training in the remainder of this cycle.

## 18:28 UTC — round closed

All bounded experiments, fixed endpoint evaluation, visual inspection and next-round research are complete. Final check: own training PID3797039 absent; GPU compute-process list empty; both4000 checkpoints, predictions, protocols, curves and development results present; all four Windows report entrypoints exist. Existing automation pymousestudio was officially updated to PAUSED and confirmed by the tool. Closed about13minutes before the12-hour hard deadline, rather than inventing further training after development evaluation. No jobs were killed, no dataset files deleted, and no default checkpoint promoted. Further experimental training awaits new user authorization.
