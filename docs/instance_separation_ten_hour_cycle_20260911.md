# 双鼠实例分离：十小时自主推进台账

## 授权与时间

用户要求从现在开始10小时，每半小时检查实际进度并推进下一步。
起点2026-09-11 20:31:03 UTC；截止2026-09-12 06:31:03 UTC。
北京时间9月12日04:31:03–14:31:03；洛杉矶9月11日13:31:03–23:31:03。
已使用OpenAI Docs核对计划任务，并通过应用工具恢复既有mousegeoforge heartbeat，
ACTIVE、间隔30分钟、截止上述时间；目标任务不变，无重复任务。
截止后安全收尾并暂停自动化；若最后一轮略早于截止，可短暂等待后暂停。

## 研究边界

上一周期阶段13–19显式mask参考框候选两种子均62/64，但frame296俯视两个query
重复关注黑鼠。框精修不能创造缺失棕鼠实例。本周期优先区分候选缺失、匹配/评分
失败和遮挡，不能用NMS删除一个query或强制框分离冒充找回第二只鼠。

先用现有缓存做零训练、零模型前向审计：所有88开发记录的6个query框/关键点，
以及既有48条诊断记录（8开发误例+40训练姿态）的全部query mask统计。
GT只用于诊断/监督，不改变推理、评价分配或历史17门槛，不按GT挑选输出。
观察训练Hungarian（objectness+box+keypoint）与评价（objectness+box）的差异，
是否有低分正确候选、两个高分重复实例，以及GT辅助最优覆盖上限。
mask统计仅覆盖指定48条，不冒充完整88条mask审计或全训练集。

据证据决定首个最小单因素机制，mask-aware Hungarian为候选而非先验结论。
训练前独立锁定协议：相同初始权重/种子、相同训练输入与步数、768分辨率、
OFF逐值一致、无GT推理泄漏、梯度合理；OFF/ON不得同时改变多个损失或框精修。
先seed17；有实质收益且保留通过才seed29。终点/最大步数提前记录，不能只看
train loss延长，不按开发集挑最好checkpoint。原88开发/17门槛保留，同时报告
坐标NME、检测感知PCK、IoU/召回/误检/重复实例与困难样本视觉。
绑定旧底模的mask框候选不能套用新底模；若组合必须新导出并独立证明。

一个GPU任务，检查运行锁/日志/status后再启动，锁定实验代码不得编辑。
不覆盖生产、下载、渲染、删除、付费或git push；数据规模超过既定2倍需询问。
真实Cylinder不在本地，不能声称真实泛化。连续两个新机制无实质收益/保留退化，
停止新增训练；剩余只做有明确问题的复盘或状态检查，不为凑时间重复实验。
每轮简报真实状态并更新此台账；十小时为自主推进窗口，不是GPU连续训练承诺。

## 启动检查

20:31:03 UTC取得起点。GPU无计算进程，无训练锁，旧成果已完成且未改动。
代码确认：训练Hungarian未用mask分配成本，但匹配后的loss已含mask BCE/Dice；
不能将“加入mask匹配”说成首次mask监督。评价使用框和objectness，不用GT关键点。
当前开始CPU缓存审计，不启动训练；先判断修改匹配是否确实能改变错误分配。

## 阶段20：20:33 UTC缓存审计完成

新增且已锁定`tools/audit_instance_query_coverage.py`；输出
`output/instance_query_coverage_v1/{protocol,report,status,box_rows,mask_rows}.json`。
两种子各88条已存预测、各184视角，全部6query；mask统计各48条/96视角，
其中40训练姿态/80视角/96实例，8开发误例/16视角/24实例。
输出输入哈希及源码哈希，零模型前向、零训练步骤，不改既有评价分配。

- 训练式和评价式Hungarian在每种子184视角仅1处不同：frame296 top。
- 原框双鼠评价检出53/51，GT辅助最优一对一框覆盖上限54/52；即便允许所有
  低分query，覆盖上限也不变。不能把这个GT上限当新推理结果。
- 既有训练代表视角mask已覆盖96/96，GT辅助mask最优匹配没有再增益；8开发
  误例为23/24，允许低分query也不能超过23。此证据不足以支持盲目增加匹配成本。
- 第296帧top棕鼠最大mask IoU（全部6query）仅.01273/.01060；q3/q4都指向
  黑鼠且score约.99998，棕鼠并非被置信度过滤，也不是遮挡不可见。
- 查得另一类问题：训练ClimbWall frame50 side为单鼠，但q3/q4都以>.95置信度
  覆盖同鼠（mask IoU约.953）。它属于多余实例假阳性，需与双鼠缺失候选分开报告。
- 单鼠bottom_only每种子8视角出现16个高分query，而只检出8个GT目标；此处
  当前仅框缓存，不将其一概解释为mask重复，下一步需查实例监督完整性与objectness。

已视觉检查`output/instance_query_coverage_v1/frame296_all_queries_seed29.png`：
RGB中黑棕鼠均清楚可见且分离；q3/q4的紫色mask都覆盖黑鼠，其他4query主要为
背景并已被低分过滤，没有隐藏的棕鼠候选。图中GT只用于展示，不参与推理。

### 下一轮的明确问题与分支

先核对原训练实际序列中的`detection_annotation_complete`、实例valid/可见性、
objectness负样本权重和loss日志，确认单鼠重复是否因为缺失负监督或稳定分配不足。
如需新增模型前向，固定种子17现有底模，对原356个唯一训练输入做一次完整成本/
分配快照，计算原分配与预先锁定mask BCE/Dice成本分配的差异、成本边际与梯度；
不是重跑旧48条诊断，目的是检验训练分配可改变性。仍不进行GT驱动推理。
如果完整训练分配基本不变、监督完整，则否定“只改匹配即可修复”的首选路线，
不启动近似等价OFF/ON训练；转向一个有依据的空间实例分离/训练多样性方案，
先研究/单因素预检，不能在相同训练输入上靠延长步数宣称真实泛化。
后续应保留两种错误计数：多余查询假阳性 vs 第二只鼠完全缺失；不能只靠减少
预测数量提高精度而隐藏召回下降。旧mask精修候选仍保持默认关闭和不可混配。

当前GPU空闲，只有CPU审计完成，没有后台训练。下一计划检查约21:01 UTC。

## 阶段21：21:01轮完整训练分配快照

21:01:20 UTC恢复检查，GPU空闲、无训练锁。新增诊断模块
`mousegeoforge/instance_assignment_audit.py`、5项单元测试及独立runner
`tools/audit_training_instance_assignment.py`。已完成后锁定，后续不得修改。
输出`output/training_instance_assignment_v1`，PID265204，131.28秒完成。
356唯一训练输入、40姿态、980视角观测，零优化步骤；所有356输入的原框、
实例token及旧匹配与既有缓存逐值一致，基础权重/encoder/源码/数据/flags未变。
所有标注complete、mask有效，所有未匹配query负权重为正（no_object=.1）。
216项完整单元测试通过（原211+新增5）；解析梯度与autograd核对一致。

加2maskBCE+2maskDice后980视角分配变化0，触发预先声明的<=1%停止匹配训练
判据。因此本周期不启动matching-only OFF/ON，不据此否定从头训练的mask匹配价值。
964个单鼠视角有11个高分未匹配查询，均为native-grid mask重复；16双鼠视角
32实例无高分未匹配。重复查询token cosine约.546–.647，不能说token完全相同。
该快照不是训练期间的匹配动态或完整网络梯度归因。

一手结构参考及暂缓原因记录在`docs/instance_query_spatial_routes_20260911.md`：
DAB-DETR空间查询、Mask2Former逐层query位置/mask注意力。当前不堆叠结构改动。

## 阶段22：同预算双鼠姿态多样性对照（已准备，准备启动）

原800步中600步单鼠、200步双鼠；双鼠仅8姿态各访问25次。实际训练manifest
已包含同一训练视频的200个双鼠姿态，无需下载/渲染。选择数据多样性作为本周期
第一个新训练机制，比在当前零改配上硬加mask匹配更有直接依据。

已锁定`tools/train_dual_pose_diversity.py`及
`output/dual_pose_diversity_seed17_v1/protocol.json`，preparation通过。
独立输出，不编辑任何阶段1–21代码。方案：

- 从已验证seed17温度ON最终权重warm-start，两臂相同初始化，fresh AdamW；
  原参数lr3e-6、温度lr3e-4，原encoder冻结、原loss/匹配/深度设置完全相同。
- OFF原8双鼠姿态；ON保留原8并增加8个train-only姿态。原帧
  1/29/58/86/115/143/172/200；新增8/15/43/72/100/129/157/186。
  新帧按原训练姿态序号最大间隙填补，未按开发误例/预测好坏挑选。
- 原双鼠每帧第偶数次访问换为对应新增帧，共替换96/200双鼠slot；原600个
  单鼠slot逐项不变。ON原帧各13次、新帧各12次。总训练姿态40→48（1.2倍），
  双鼠8→16（2倍）；训练源不变，与held-out视频/场景隔离。
- 两臂各800步/1800图像/3540384patch tokens，768分辨率与视角顺序不变。
  固定final800，不按开发集选checkpoint；这是新warm-start对照，不重跑旧800步。
- 训练前88预测严格重放；原门槛17、原88评价全部保留。额外报告多余查询/
  native-grid mask重复，保存246/296/352/383可用固定困难帧mask供视觉复查。
- 后续seed29条件：ON通过全部17门槛、双鼠检测感知PCK相对OFF至少+.01、
  双鼠box-matched NME不超过1.02×OFF、TP不低于OFF；FP/重复另行报告。
  未达到实质收益时停该机制，不凭单帧成功或train loss下降延长。
  若确有值得继续的泛化趋势，需另锁定等预算续训协议，本周期累计新增训练
  每臂上限1600步。当前首终点800，未授权程序自动延长。

未混用旧mask框精修候选，未覆盖生产；暂缓空间查询结构，避免与数据干预混因果。

### 阶段22运行交接

已launch，PID265880，输出目录`output/dual_pose_diversity_seed17_v1`，单GPU任务。
21:11:42 UTC检查status为initial_evaluation，正在完成训练前88记录严格重放；
此时尚不能宣称800步已开始/已完成。查看同目录run.log、status.json、
pretrain_parity.json；训练中会更新diversity_off/on的history.json和loss_components.json。
若失败，保留锁定协议/源码/日志；定位后使用独立修正版目录，不原地改运行代码。

下一计划轮约21:31 UTC：先检查该PID/锁，不能重复launch。完成后必须存在
verification.json且passed，再分析results.json、initial/diversity_off/diversity_on
的88预测与*_instance_quality.json。历史17门槛调用analyze_cylinder_view_balance.
gate_results时，用cylinder_source_mix_v1/results.json的baseline和
cylinder_view_balance_v1/results.json的side_control，不改成当前warm-start门槛。
每个候选映射为balanced_views以保留包含control对比的全部17条，勿仅算15条。
用matched_views计算双鼠框IoU/TP，对照必须是本轮同预算OFF而非旧未续训底模。
固定困难帧的mask_visual_*.pt只保存246/296/352/383中实际存在的评价帧，渲染
RGB+所有高分query的mask/框进行视觉检查，不能用GT挑一个好query。
检查ON是否满足预声明PCK/NME/TP与全部17门槛，之后才决定seed29或其他路线。

本轮结束前复查：pretrain_parity已通过，88条初始预测逐值一致；PID265880已进入
diversity_off训练，最近日志50/800步，loss/梯度有限，elapsed72.84秒，缓存22.67GiB。
训练正常后台继续，下一轮接续检查，不重复启动。

## 21:31轮：阶段22完成，未达到推进判据

21:31:20 UTC检查PID265880已completed，646.60秒，两臂各800步；verification
passed，264评价（初始+OFF+ON各88），GPU空闲、无训练锁。新增分析脚本
`tools/analyze_dual_pose_diversity.py`，产出acceptance.json、comparison.png、
fixed_dual_masks.png，完成后锁定。

| 结果 | 初始 | OFF八姿态 | ON十六姿态 |
|---|---|---|---|
| 双鼠框IoU | .646863 | .648157 | .658467 |
| 双鼠检出/64 | 53 | 54 | 54 |
| 双鼠检测感知PCK | .515625 | .527174 | .529212 |
| 双鼠box-matched NME | .126110 | .120100 | .116763 |
| 原门槛通过 | 17/17 | 16/17 | 16/17 |

ON相对同预算OFF的PCK增益只有.002038，未达预声明.01；NME略改善但不能用它
替换预声明判据。两臂共同失败的是old_dual_pair body_depth_absrel：
OFF .027078、ON .026769，限值.023732。不能将未续训初始作为唯一对照夸大收益。
按预声明停止该机制的seed29复验/追加训练，记为本周期第1个未达实质收益机制。

已视觉检查四帧top、初始/OFF/ON所有高分query：352/246/383均仍能分开两鼠；
296三组都将q3/q4的mask集中到黑鼠，棕鼠缺失没有修复。全184视角native-grid
mask重复从初始9降至两臂均6，双鼠重复均1，不能归因于姿态多样性。
原mask框候选未混用，生产未覆盖。

## 阶段23：独立逐层查询位置预检（已启动）

基于已记录Mask2Former逐层query-position思想，新建
`mousegeoforge/query_position.py`：给每层cross-attention的Q加入独立的6×256
learned向量，2层共3072参数、全零初始化；OFF不安装hook、ON上下文退出移除。
这是查询位置/身份的向量信号，不是框坐标，不是永久动物ID；不是DAB-DETR或
Mask2Former完整复刻。保持全图memory、不限制mask区域、不强迫两框远离。
GT从未传入该模块。每层单独信号旨在避免仅依赖最初query内容，实际收益待对照。

5项新增单元测试通过。独立runner`tools/preflight_query_position.py`已launch，
PID266541，输出`output/query_position_preflight_v1`；与阶段22不并发GPU任务。
原seed17温度ON起点（非失败的多样性续训checkpoint），88开发输入OFF/零ON
逐值比较所有输出，并严格重放旧预测；另用原训练双鼠输入一次OFF/ON
forward/backward，验证输出和公共梯度一致、新3072参数梯度有限非零，无优化器。
只有全部预检通过才考虑训练，不能把零初始化等价当有效性证据。

下一训练应单独锁定：同原8姿态及800-slot序列、同起点与预算，唯一ON干预为
逐层query-position；不要同时启用新mask matcher或16姿态。考虑两臂共同冻结
共享adapter/深度权重，只训练相同实例解码路径以避免上一轮共同深度漂移。
共同冻结是新实验的共同设置，不可把不同设置的阶段22 OFF冒充本轮控制组。
必须保留原数值路径/严格OFF预检，训练后验证冻结参数不变；若调整requires_grad
影响dispatch，保留原flags并用显式优化器参数选择/清梯度，不能放宽前向零容差。
具体scope/学习率/预算/判据尚需在训练协议中锁定，当前只做预检，未启动该机制训练。

### 阶段23数值预检问题与后续处理

v1 PID266541在43.74秒失败：88开发OFF/零ON所有前向均已逐值通过，实际train
反向在公共参数逐值比较失败，首报max abs1.49e-8。保留原源码/协议/日志，未训练。
独立`tools/probe_query_position_gradients.py`做OFF/OFF、ON/ON及OFF/ON比较；
其末尾开启全局deterministic模式的诊断改变了前向dispatch，max输出差7.63e-5，
按零容差正确失败，原目录保留；不能把该变化误认成query-position本身的变化。

另建`tools/probe_query_position_gradients_v2.py`，取消后端切换，保持其余实验输入：
1次encoder/8次head前向、零优化，所有前向和标量loss逐值一致、原权重不变。
输出`output/query_position_gradient_probe_v2/report.json`。全pose公共梯度：
OFF/OFF max1.371e-6、相对L2 6.740e-8；OFF/ON max9.537e-7、相对L2 5.732e-8。
纯检测/分割亦存在：OFF/OFF相对5.414e-8、OFF/ON4.021e-8，不能单独归因于
keypoint grid_sample。新参数两层梯度范数约.025891/.043961，有限非零。
这是当前输入的重复反向数值证据，不是穷举确定性证明。

新增CPU精确局部Jacobian测试`tests/test_query_position_derivatives.py`通过；
全套222测试通过。未编辑锁定query_position模块/既有5项测试。

据实修订数值预检要求，独立`tools/preflight_query_position_v2.py`在运行前锁定
公共梯度相对L2<=1e-6（基于已测原模型自身约6.7e-8的波动），仍要求所有前向
严格0差、新参数有限非零梯度、原参数/flags/encoder不变。原v1保持failed，
新report显式记录原严格反向检查没有通过，不把容差检查包装成逐值通过。
PID267404已launch，输出`output/query_position_preflight_v2`，零优化。

待训练计划具体写入`docs/query_position_pilot_plan_20260911.md`；下一轮先读
v2 status/run.log/report，通过后再实施并锁定新的head-only同预算800步OFF/ON。
阶段22未达收益是本周期第1个新训练机制失败；本结构目前仅预检，未计作训练失败。

### 阶段23复核通过与阶段24训练准备

21:45:30 UTC确认v2 completed（49.13秒）：88开发和实际train前向最大差0；
公共梯度相对L2差3.909e-8，满足新预声明1e-6限制，但不是逐位反向一致。
两层新梯度范数.025891/.043961；原权重、encoder、flags、CPU/CUDA RNG不变，
hook清理通过。222项测试通过。v1失败及probe_v1后端切换失败原样保留。

已实施并锁定`tools/train_query_position_pilot.py`和
`output/query_position_seed17_v1/protocol.json`；准备成功后launch。
新两臂query_position_off/on：同原8双鼠姿态/800 slots，每臂800步。
两臂均只优化instance_pose_head；shared adapter/dense/bridge不更新，保留
原requires_grad flags，每步model.zero_grad清所有梯度，仅clip优化器选中参数。
原head lr3e-6，temperature lr3e-4；ON新增3072位置参数lr3e-4。
无mask matcher、16姿态、mask框精修或时序混入；新OFF控制不能用阶段22替代。
运行中每组88评价校验depth/depth_conf哈希完全等于初始，非head参数哈希不变。

阶段24使用上一原温度checkpoint，不用已退化的多样性结果；生产不覆盖。
当前是本周期第2个新训练机制：ON须满足原17门槛、双鼠PCK较同预算OFF+.01、
NME<=1.02×OFF、TP不降、FP与重复mask不增，才做seed29。否则停止本周期新训练，
不再尝试第三机制或以train loss下降为理由续训；继续必要复盘/半小时检查至截止。

下一轮约22:01 UTC先读`output/query_position_seed17_v1/status.json`、run.log、
锁与GPU；若完成还需verification.json passed。结果格式与阶段22相同，分析
可复用旧分析逻辑但不能原地改`analyze_dual_pose_diversity.py`锁定文件；新脚本
必须使用query_position_off/on与新OUT，增加FP/重复判据，原17门槛定义不变。
随后视觉检查固定四帧，诚实区分框改善、坐标改善和是否找回第296帧棕鼠。

阶段24 PID267837，21:49:26 UTC status=initial_evaluation、无异常日志；
训练前初始重放进行中。已新增`tools/analyze_query_position_pilot.py`，编译检查
通过；只在正式verification passed之后运行，原门槛及新增FP/重复条件均已写定。
本轮截至此处未宣称结构已训练成功；下一轮继续检查该后台进程及产物。

离开本轮前再次检查：已完成初始评价并进入query_position_off，25/800步，
运行65.52秒，loss/梯度有限，GPU后台训练继续。下一轮不要重复launch。

### 22:01 UTC检查：阶段24完成，停止本窗口新增训练

22:01:23 UTC检查status/log/锁/GPU：PID267837任务completed，639.649秒，
OFF/ON各800步、264次评价，verification passed。深度/depth_conf输出严格0差，
non-head状态不变，encoder无梯度，hook移除。GPU已无计算任务，无训练锁。
已运行锁定`tools/analyze_query_position_pilot.py`并视觉查看固定四帧mask图。

OFF/ON均17/17通过；双鼠TP均54/64，PCK分别.529211957/.527853261，ON-OFF
为-.001358696，未达到+.01。框IoU分别.647444473/.646823201，NME分别
.121115734/.121239637。全184视角两臂均180TP/42FP、7个强重复mask视角；
双鼠重复均1。frame296仍为q3/q4双黑鼠、无棕鼠，352/246/383未见实例分离改善。
相对初始的共同改善不能归功于ON机制，整体NME改善主要来自尾巴，身体NME稍退。

为排除“分支没有训练”，CPU读取两臂final_heads.pt及optimizer_scope：两臂均
steps800；OFF offsets全零且不在optimizer，ON offsets3072个全非零，L2
1.016572833、max abs.0620368235，确在lr3e-4的独立优化组。公共head
8,321,353参数lr3e-6、temperature257参数lr3e-4两臂相同。证据与文件哈希
记录在`docs/instance_separation_cycle_results_20260911.md`。

阶段22数据多样性和阶段24查询位置两项均未达预声明收益，停止新增训练：不seed29、
不追加步数、不启动第三机制、不把新head混接旧绑定mask框候选。生产与既有实验
不覆盖。222项测试最近已通过，本轮仅分析/文档，不重复GPU运行。

自动化保持ACTIVE至原06:31:03 UTC截止；窗口尚未结束，不宣称连续训练10小时。
后续半小时检查只读状态/新反馈，有明确待解问题才复盘；不要重复已完成诊断。
当前无待启动实验。截止时通过应用工具暂停既有mousegeoforge并做最终状态交接。

### 22:31 UTC例行检查

22:31:18 UTC：无用户新增反馈。阶段22/24 status仍completed，阶段24日志以
800步和COMPLETED结束；GPU无计算进程，系统锁仅cron/snapd/apt-get，无训练锁。
未发现需要追加诊断的新问题，不重跑已完成分析，不启动新训练。既有结论及
停止条件不变；窗口未结束，继续按半小时检查至06:31:03 UTC。此次仅状态核验
和台账更新，不计新增实验或训练进度。

### 23:01 UTC例行检查

23:01:21 UTC开始核验：无新增用户反馈；阶段22/24均completed，两日志末尾均
COMPLETED，GPU无计算进程，锁仅cron/snapd/apt-get。无异常或新的待解问题，
未新增训练/诊断实验，保持两机制停止条件及原截止时间。检查完成后仅更新此台账。

## 用户新授权与阶段25：预算敏感性

用户随后明确：遇到此类负结果，可自主调研判断增加训练步数或调整架构。
因此不再将两次失败作为整个窗口的硬停止；旧800步验收仍保留为负结果，不能
事后改门槛。已使用OpenAI Docs技能核对并更新既有heartbeat提示，原半小时节奏
和06:31:03 UTC截止不变，无新自动化。主状态及后续计划已更新。

CPU同输入复访曲线审计（`output/query_position_budget_audit_v1/report.json`）：
8个双鼠输入，首两次/末两次pose loss OFF下降6.739%、ON下降6.595%；后半段
200-slot双鼠均值亦在下降，但ON无独有优势。16个单鼠输入反而上升4.245%/
3.717%；Cylinder13个重复输入下降13%左右。只有初始/最终开发评价，不足证明
800步充分收敛，训练损失下降也不足证明泛化。因此做一次有上限的预算敏感性验证。

新独立runner`tools/train_query_position_budget1600.py`已编译/prepare通过并锁定。
保留原8姿态及800-slot顺序，完整重复两次，OFF/ON各1600步、3600图像、
7,080,768 patch tokens；同学习率/head-only/原温度起点，不叠加新数据或结构。
旧checkpoint没有优化器状态，因此从原起点重新跑连续1600步，不能声称无缝接续
旧800优化器。新runner保存800/1600模型、优化器和RNG以便后续真实续训/评价。
不改历史文件、不挑最佳开发checkpoint，保持88/17门槛及逐值深度/非head冻结检查。

23:23:33 UTC：确认GPU空闲且无训练锁后已launch PID269051，输出
`output/query_position_seed17_budget1600_v1`；当前preparing，预计约20分钟量级，
以实际日志为准。首轮完成前不做另一个GPU任务。上限1小时且受原截止保护。
完成后需verification passed，新增适配1600曲线的分析脚本（旧分析写死800且
决策文字是历史停止语义，不能直接运行或修改它）；固定终点比较及四帧视觉。
必要时使用新800中点状态评价实际轨迹，不把旧独立800结果冒充同次运行中点。
若1600仍无收益/保留退化，不仅据train loss再加倍，转单独空间候选架构计划。

离开本轮前：初始88条严格重放通过，PID269051进入OFF 25/1600步，损失/梯度
有限。再次全套222单元测试通过（原test_occlusion_pose tensor转float警告不变）。
未启动第二GPU任务；下一半小时检查继续此进程，不重启。

### 23:31 UTC检查与分析准备

23:31:21 UTC：PID269051为唯一GPU计算任务，持有全局/本实验两把训练锁；
OFF1325/1600步，日志无错误。未重复launch或改锁定训练源码。
新增`tools/analyze_query_position_budget1600.py`并编译通过，原17门槛与预声明
PCK/NME/TP/FP/重复条件保持不变；生成固定四帧所有高分query图及1600曲线。
预算图明确标注旧800是独立运行，不冒充本次800中点验证。只在最终verification
passed后运行，现阶段不提前计算ON收益或下结论。

23:33:13 UTC复核：OFF1600及其88条评价完成，ON已进入1/1600；累计176条
评价的depth/depth_conf严格不变，non-head状态不变。总体约574秒，无报错。
下一轮先核验completed和verification，再运行新分析并实际视觉检查。
若固定1600仍失败，按新授权继续空间候选研究，不恢复旧的整个窗口停止规则。

## 00:01 UTC检查：阶段25完成，转空间注意力预检

00:01:23 UTC检查：PID269051 completed，1088.270秒，两臂各1600步，共3200。
verification passed，264评价，深度输出0差、non-head状态不变，800/1600真实
训练状态已保存，GPU空闲、无训练锁。运行新锁定分析并实际查看四帧mask图。

两臂均17/17。双鼠PCK均.534646739、TP均54/64、重复mask均1；框IoU
OFF .647633007、ON .646907431；NME .120306054/.120566068。相对旧独立800
终点，两臂都有小幅共同进步，但没有ON优势；全视角均182TP/36FP、3个重复视角。
身体NME .113713085/.113917621，尾巴.135375697/.135762519。不能把训练预算的
共同进步归因于查询位置。视觉：296仍双黑鼠无棕鼠；352/246/383未见实例分离
改善。该机制不做seed29、不据train loss再加倍；按用户新授权转新机制而非停工。

检索并核对SMCA论文/官方项目，软Gaussian空间注意力更适合做当前最小差异；
不将其称为完整DAB-DETR。新模块`mousegeoforge/spatial_query_prior.py`：
两层各6个可移动参考中心、可学习尺度和强度，加上来自当前query的条件偏移，
总2108参数；仅实例cross-attention logits加有限Gaussian bias。无硬分区、
无排斥/NMS/GT、中心可以重合；OFF不挂任何hook。初始sigma.5、strength.1，
ON有意非零，不能假装零ON等价；只要求OFF原前向0差、ON深度0差和梯度有限。
新7项测试通过，全套229通过。当前未改既有生产head或锁定模块。

`tools/preflight_spatial_query_prior.py`编译后launch，00:06:10 UTC PID270163，
输出`output/spatial_query_prior_preflight_v1`，88 OFF/ON冻结评价+3个train-only
V1/2/4梯度探针，零优化；最长20分钟、全局训练锁。当前预检运行中，不启动
另一GPU任务。只有report passed且初始行为可接受才锁定训练；具体方向见
`docs/spatial_query_prior_plan_20260912.md`。

### 阶段26预检修复与正式训练启动

v1 PID270163在14.71秒失败：state_hash收到GPU上的新模块state，尚未开始完整
开发评价。独立v2将哈希张量detach/cpu，PID270448在15.49秒失败：指标函数
收到GPU预测/CPU batch混用。两失败源码、协议与日志保持，均零优化；不是性能
失败。独立v3仅统一评价张量到CPU，模型/输入/验收不变，PID270713在48.45秒
completed/PASSED。不得宣称v1/v2通过，也不能将这些脚本错误当架构证据。

v3的88 OFF旧预测0差、ON depth/depth_conf0差，原权重/encoder/flags/RNG安装
不变，hooks清理；新2108参数四种张量在V1/2/4探针的梯度均有限非零。ON初始
框归一化坐标最大变动.006424159。双鼠OFF/未训练ON PCK均.515625、recall均
.828125，NME .126109529/.126067062；bottom_only及旧侧视检出率未变。初始
ON指标只是扰动评估，不是学习收益，CPU指标末位浮点与旧GPU聚合可略有差别。

已新增并编译、prepare锁定`tools/train_spatial_query_prior.py`，独立输出
`output/spatial_query_prior_seed17_v1`，PID271128于00:11 UTC launch。按原计划
两臂各1600、同8双鼠姿态及两遍800序列、3600图像、7,080,768 patch tokens；
原温度checkpoint，不是已续训query-position底模，未装query-position模块。
原head lr3e-6/temperature3e-4；ON2108空间参数3e-4；两臂共同head-only，
原flags保留、只clip/更新选中参数。保存800/1600模型/optimizer/RNG。
原17及PCK+.01等门槛未变；初始ON预检单列。确认GPU空闲/无训练锁才launch。

下一轮先status/log/GPU和verification；若完成，新建空间版分析（旧budget脚本
写死query_position臂和旧800比较，不要直接改锁定文件或误套）。需要全88结果、
原17、双鼠IoU/TP/PCK/NME、身体/尾巴、全视角FP/重复，以及296/352/246/383
所有高分query的视觉检查。可对比未训练ON预检但不要把前向初始变化说成学习。
若失败，先检查实际参考中心/尺度/强度及条件权重是否更新，再决定更有效的候选
初始化路线；不是自动训练加倍，也不是自动停止整个窗口。

## 00:31 UTC检查：阶段26完成、阶段27特征容量诊断

00:31:21 UTC：空间先验PID271128 completed，1079.942秒，OFF/ON各1600、
264评价，verification passed，深度/非head逐值不变，GPU空闲且无训练锁。
新增并运行锁定`tools/analyze_spatial_query_prior.py`，固定门槛不变，区别冻结
未训练ON与训练结果；实际查看四帧mask图，296仍双黑鼠，352/246/383基本不变。

空间OFF/ON均17/17；双鼠PCK .534646739/.536005435，增益.001358696<.01；
TP均54/64、重复mask均1；IoU .646909696/.646467350。全视角均182TP/36FP、
3个重复视角。身体NME .113836904/.113838943，尾巴.135344209/.135333811。
不进入seed29，不将门槛通过当机制有效，不凭train loss再次加倍。

CPU`tools/audit_spatial_prior_parameters.py`写入该目录parameter_audit.json：
OFF新参数全部保持初始化；ON center/sigma/strength/condition四类delta L2为
.224773/.243317/.196078/1.375434，确实被优化。基础sigma约.474–.515，
strength约.095–.113；基础坐标不含图像条件偏移，不能据此断言实际注意力轨迹。
该审计零模型前向/零优化，没有修改冻结旧模型。

下一问题：是不是局部特征已经足以识别棕鼠，而无序query解码未产生候选？
采用独立容量探针，不立即再堆decoder机制。新增
`mousegeoforge/foreground_feature_probe.py`、4项测试；全套233通过。
`tools/probe_frozen_foreground_features.py`编译后，确认无GPU任务/训练锁，
launch PID271906，输出`output/frozen_foreground_probe_v1`。

固定原温度底模，全程eval/no_grad，不改原flags；截取实例decoder前memory减去
确切sine位置，L2归一化，用257参数linear1x1区分前景/背景。356原train输入
和88开发输入，双鼠source严格隔离；原800序列两遍，1600探针步，lr.01，
balanced soft-mask BCE。只用train mask训练，开发阈值预先固定.5，不调阈值，
不挑checkpoint。前景union只是可分性诊断，不能冒充两实例检测。
缓存只放RAM上限8GiB，不落2GB特征数据到dataset；原模型及88旧预测逐值校验。

00:37附近已提取192/444输入，缓存1.05GiB，日志无错误；同一GPU任务运行中。
预声明：双鼠开发前景IoU>=.5、precision>=.7，296top两实例各自原生网格recall
>=.7时，支持下一步图像特征候选初始化。若失败不能断言encoder没有信息，需区分
线性容量/尺度/数据限制。完整计划见`docs/frozen_foreground_probe_plan_20260912.md`。

### 阶段27当轮完成与视觉复核

PID271906 completed，129.797秒；1600探针步，444特征输入、RAM2.496GiB；
原模型/encoder/flags均不变、无原模型梯度，88原预测0差，代码/输入哈希通过。
双鼠开发native-grid前景719TP/841FP/0FN，IoU=precision .460897436，recall1。
296top为24TP/34FP/0FN，两个GT鼠各12个native-grid像素、覆盖率都1。
预声明IoU>=.5/precision>=.7未达到，proposal_route_supported=false，未改阈值。

实际查看`output/frozen_foreground_probe_v1/fixed_foreground.png`：352/296/246/383
均出现两鼠相关亮区；296明显能激活此前缺失棕鼠，但轮廓粗、外扩，部分鼠间区域
粘连，不能据前景覆盖冒充实例分离。结果支持“有可读出的局部棕鼠信号”的有限
判断，不证明现有encoder足够解决所有困难，也不满足直接用该mask生成候选的条件。

不微调开发阈值、不用GT拆开预测前景。下一步改为独立中心/尺寸监督的候选容量
诊断，先检查局部图像特征能否产生两个正确中心及框，再谈接入query。已查阅
Objects as Points及官方CenterNet decode/loss实现；局部3x3峰筛选只用于新热图
候选，不删除旧query来冒充恢复。新任务仍不是关键点完整OFF/ON，需明确限制。
下一计划见`docs/dense_center_candidate_probe_plan_20260912.md`；当前无GPU任务，
待下一轮实施新模块、测试、数据中心碰撞审计及锁定协议，旧阶段不重跑。

## 01:01 UTC检查：阶段28中心/框候选探针实施

01:01:22 UTC核验阶段27completed、GPU空闲、无训练锁；最新计划仍为独立中心
候选容量诊断，不重跑前景探针。新增`mousegeoforge/dense_center_probe.py`与
6项测试（标签-解码回环、中心碰撞、近邻、V1/2/4梯度、空目标、重复/score门槛），
全套239通过。训练前实现固定参数，不据开发结果调参。

模块16773参数：256→64ReLU→5的1×1MLP，输入冻结特征去sine后L2归一化，
输出heat/offset_xy/wh。seed17；输出权重×.01，heat bias-2.19、offset0、
size bias logit(.2)。heat GT取中心floor格点、sigma=min(wh_grid)/6裁至[.5,2]；
focal alpha2/beta4 + offsetL1 +5×sizeL1，偏移/尺寸只在GT中心监督。
推理固定3×3热图局部峰top6，score>=.5与boxIoU>=.5；box/objectness Hungarian
沿用原公式5L1-score。不是对旧query删框或GT挑选中心。

新runner`tools/probe_dense_center_candidates.py`编译通过，确认无GPU/训练锁后，
01:05:43 UTC launch PID272774至`output/dense_center_probe_v1`。顺序：先444
输入冻结特征提取（356原train+88dev，RAM<=8GiB），全部标签中心碰撞审计；
同格碰撞明确报错，不静默丢标；报告Chebyshev距离<=1近邻（3×3筛峰潜在风险）。
再3个实际train输入V1/2/4做loss/各参数梯度有限非零检查，原模型无梯度；
全部通过才AdamW lr.001/wd.01/clip1训练候选头1600步，原800序列两遍。

固定终点、阈值及诊断条件：双鼠TP>原53/64，全视角FP不高于原底模、非双鼠TP
不降、重复视角不增，并单列296是否两个框均检出。全部top6框匹配覆盖另列，
不能把忽略score的覆盖冒充实际召回。原底模88预测0差和参数/encoder/flags不变
必须验证；不是完整关键点OFF/ON，不计算虚假的PCK收益或用它替代17门槛。
最终保存88预测/原框、train与dev检测统计、固定四帧对比图；最大40分钟且受
原06:31截止约束。当前运行中，下一轮不重复launch。

## 01:31 UTC检查：阶段28完成，启动阶段29有界续训

01:31:28 UTC确认PID272774 completed，137.065秒；3实际输入梯度通过，原模型
和88输出不变，GPU空闲。标签无同格中心碰撞，train无近邻，dev有1个3×3
近邻对。定位为开发index10/frame218/side相机2，中心网格(25,16)/(24,16)；
局部峰筛选可能合并，不能通过GT例外绕过。盒IoU重复统计不同于此前mask重复，
报告中的原baseline duplicate_views=3而非旧mask统计9，不能混用。

阶段28固定score.5实际双鼠仅8TP/2FP/56FN（原53TP/11FP）；非双鼠均0TP。
全部top6使用相同box/objectness分配但忽略score时覆盖55/64；训练集31TP/
27FP/996GT，而top6覆盖770。第296帧top两候选框IoU .52685/.59791，分数
.37374/.47369，均未达.5；原严格框检测此视角0/2。视觉检查热图能出现两鼠
响应但分数弱，图中只有高分框显示，不把绿色GT框误认成模型输出。
既定诊断criteria false，不直接接入关键点模型、不改阈值。

按400步区间平均：总loss2.589→1.952→1.802→1.658，heat1.850→1.321→
1.245→1.135；sizeL1 .0904→.0707→.0556→.0497，offset约.275尚未明显改善。
训练自身分数低、部分几何候选已有覆盖且训练趋势仍改善，支持做预算敏感性而
非立即抛弃机制；不是仅凭开发单帧就认定“没收敛”。

独立`tools/probe_dense_center_budget6400.py`恢复原probe.pt与AdamW实际moments/
step=1600，再跑同800序列六遍（新增4800、总6400）。lr.001、loss、架构、
数据、score.5和成功条件不变；固定3200/6400存模型、optimizer及train/dev指标，
总6400为终点，不选开发最好checkpoint。原probe无RNG保存，但MLP无dropout、
无数据随机采样/增强；不声称恢复不存在的随机状态。新CPU 3+4步AdamW恢复
逐值等于连续7步测试通过，全套240通过。

01:34:10 UTC确认GPU无任务/训练锁后launch PID273695，输出
`output/dense_center_probe_budget6400_v1`，当前重提取冻结特征，继承不落磁盘缓存
及40分钟上限/用户原截止。新协议绑定原probe/协议/report哈希，旧文件不覆盖。
后续需对比固定1600/3200/6400，分开读训练和开发以及score-gated/alltop6覆盖。
若训练也平台低分，考虑点监督与stride16/1×1容量冲突或细网格局部上下文；
若训练提升但开发退化，考虑源多样性/过拟合，不再据train loss盲目加倍。

## 02:01 UTC检查：阶段29完成，阶段30上下文对照预注册

阶段29 completed，158.090秒，原模型/88输出不变。3200→6400双鼠TP16→18、
FP4→5、top6覆盖59→58；训练TP99→216/996、top6覆盖812→895。6400全部
43TP12FP、4框重复视角，原177TP47FP、3框重复视角；原mask重复不能混入。
第296帧top棕鼠score.5803/IoU.6433检出，黑鼠score.2197/IoU.7434未检出。
已视觉检查固定四帧，不能把候选几何覆盖当实际召回，不能替换原模型。

训练loss继续下降，但训练低检出和开发边际改善减缓，先做结构效率诊断。
阶段30两臂同初值/相同17413注册参数，OFF屏蔽512邻域权重、ON使用3×3
逐通道邻域；保持stride16/768、数据、loss/阈值/优化器相同，各3200固定终点。
参考CenterNet官方head局部卷积，非完整复现、不同时加上采样。预注册已写入
`docs/dense_center_context_pair_plan_20260912.md`；4新模块测试通过，待全套及
实际V1/2/4梯度/初始等价预检。新路径不覆盖阶段28/29锁定代码与权重。

02:09:10 UTC：全套244测试通过、编译通过，GPU空闲/无训练锁后launch
PID274982，输出`output/dense_center_context_pair_v1`，OFF→ON顺序执行。
当前提取冻结特征；下一轮检查两臂status/report及顶层comparison，不重复启动。

## 02:31 UTC检查：阶段30完成，阶段31预算对照

阶段30 OFF/ON各3200，146.185/133.118秒，完整性passed，原模型不变。
训练141TP65FP→233TP32FP（GT996），top6覆盖827→915；开发双鼠19TP4FP→
14TP1FP（GT64）、top6覆盖60→56，全开发24TP8FP→29TP8FP、框重复2→0。
ON有效邻域权重L2=8.76495，OFF0；不是开关未生效。视觉查看两个固定四帧图，
296 OFF仅棕鼠高分、ON仅黑鼠高分，两组另一鼠均低分，整体验收失败。

loss按800步均值OFF2.257/1.707/1.482/1.329，ON2.019/1.465/1.219/1.048，
训练拟合仍低且在改善，尚不能认定充分训练，也不能忽略开发双鼠退步。
预注册恢复两臂actual3200模型/AdamW，额外9600到12800，固定6400/12800。
新runner`tools/probe_dense_center_context_budget12800.py`，新目录
`output/dense_center_context_budget12800_v1`，不改源码/数据/loss/阈值，246测试
通过（新增上下文OFF/ON恢复与连续训练逐值一致测试）。详细判据与后续分支
见`docs/dense_center_context_budget_plan_20260912.md`，当前待launch。

02:34 UTC确认GPU空闲后已launch PID277004，当前OFF提取冻结特征。
后续顺序ON，不重复启动；固定6400/12800报告和最终四帧图用于下一轮判断。

## 03:01 UTC检查：阶段31完成，转阶段32姿态覆盖对照

阶段31完整性passed，OFF183.659秒、ON172.207秒。6400双鼠TP18/14，终点
12800均19/64、各4FP；训练终点376TP53FP→561TP30FP（996GT），top6覆盖
958→984。固定四帧两臂已视觉查看，296 OFF只棕鼠、ON只黑鼠高分检出，
另一鼠低分几何框不算TP；没有端到端关键点改善，不具备采用资格。

03:03 UTC持训练锁做只读GPU重算ON的8原训练双鼠输入（无优化）：32TP、
0FP、0FN/32GT。已拟合这个子集但开发仅19/64，支持先改姿态覆盖而不加步数。
阶段32两臂都contextON、同初始seed17各12800，数据8→16：复用阶段22锁定
source-group隔离序列，每800仅96双鼠槽替换，600非双鼠/200双鼠预算不变。
原356→364输入、40→48总姿态、双鼠恰2倍；非新增视频源、不是新数据下载。
数据OFF需逐值复现阶段31ON终点88候选；明确报告shared356和各自训练分母。
预注册`docs/dense_center_context_diversity_plan_20260912.md`，runner独立新路径，
246测试与编译通过，待启动。此处不会复跑阶段22旧800步全头训练。

03:06 UTC确认GPU空闲后launch PID279174，
`output/dense_center_context_diversity_v1`数据OFF8→数据ON16；下轮先检查
status/log/comparison，注意这是数据开关而非上下文开关，两臂context都ON。

## 03:31 UTC检查：阶段32完成，阶段33定位误差，阶段34区域评分

阶段32两臂201.968/192.013秒，控制逐值复现阶段31ON终点88候选。双鼠开发
19TP4FP→30TP3FP（GT64），全开发50TP22FP→61TP32FP，框重复3→2；共同
训练356输入561TP30FP→759TP62FP/996GT，各自双鼠32TP0FP/32GT→62TP1FP/
64GT。已视觉检查296两鼠被新候选检出，score.6319/.5131，IoU.7570/.5687；
单帧成功不能代替整体验收，仍不接入原关键点、不seed29。

阶段33新CPU只读审计`output/center_supervision_geometry_audit_v1`完成，0forward/
0优化：开发双鼠30found、24score_only、3geometry_only、7两者皆错，GT中心
在背景仅1/64，不能当主因。单鼠side96GT全漏，90只是低分。原生中心热图与
GTmask并排图已视觉看过；相邻最大值仅用于诊断，没改推理筛峰/评分。

阶段34以IoU-Net区域质量预测为研究动机，冻结16姿态中心框，7×7 ROI特征
mean/max+框/原分数→33217参数质量头，soft maxGTIoU BCE。只训新头3200，
不NMS、不改框、GT不参与推理；OFF原分数严格原样，ON预测质量替换分数。
新模块5测试/全套251通过，预注册`docs/roi_candidate_quality_plan_20260912.md`，
runner`tools/probe_roi_candidate_quality.py`、新输出`output/roi_candidate_quality_v1`，
当前待launch；不是完整关键点OFF/ON，不报告虚假的mask/尾巴/PCK收益。

03:42 UTC GPU空闲后已launch PID281326，`output/roi_candidate_quality_v1`。
下轮先日志/预检/报告及`fixed_quality.png`，注意质量标签按每框IoU，不等于
唯一实例归属，重复框须计入FP而不能靠NMS隐藏。

## 04:01 UTC检查：阶段34完成，阶段35一对一ROI集合评分

阶段34 137.894秒完整性passed，原模型/中心框不变。双鼠30TP3FP→50TP12FP，
全部61TP32FP→157TP253FP，框重复视角2→70；训练单鼠FP62→728，说明
maxGTIoU的逐框质量目标不编码唯一性。质量MAE train.05589/dev.10451，四段
loss.37079/.32931/.32532/.32282。不再增加相同目标步数，未通过、不接入。
已视觉检查固定四帧，296保留两鼠不代表全视角质量（大部分重复在单鼠侧视）。

阶段35参考DETR一对一集合监督，固定ROI/框但新标签为原固定匹配且IoU>=.5
者1、其他0。两个相同warmstart质量MLP各3200新optimizer，一个独立，一个
加同视角候选self-attention0.1残差/LayerNorm；无ID嵌入、无NMS/强制实例数量。
新增6测试、全套257通过，OFF初始必须复现旧质量88评分，ON初始非零差记录，
两臂同初值参数但优化容量不同。预注册`docs/roi_set_objectness_plan_20260912.md`，
新runner/目录独立，待launch；不将监督和架构变化混说成纯架构收益。

04:07 UTC GPU空闲后launch PID283396，
`output/roi_set_objectness_pair_v1`独立OFF→关系ON，每臂3200固定终点。
下轮先状态/预检/顶层对照与图，不重复launch、不把inner和outer开关混用。

## 04:31 UTC检查：阶段35完成，阶段36相对几何

阶段35独立/关系各3200完成，136.399/131.536秒，初始OFF88评分0差、ON
最大变化.00016442，原模型/框不变。双鼠37TP6FP→42TP10FP，全部75TP34FP→
113TP78FP，重复4→15。监督相对旧质量头显著减少过检，但关系独有优势不合格。
训练900/964单鼠TP、63/64双鼠TP，不优先再加步；ON MAE train.03089/dev.17721。
两臂固定四帧视觉确认296漏棕鼠（OFF .17334、ON .00208），不能降阈值救验收。

阶段36依据Relation Networks的外观+几何关系思路，固定同一个阶段35关系
warmstart，在attention增加6→4零初值24参数相对框几何soft bias（2tanh界），
输入IoU、相对中心/尺寸及旧center分数差，无硬排除/NMS。原数据/监督/框不变，
两臂各3200、新同优化器；OFF初始严格0差，ON允许预注册1e-6数值路径差。
模块IoU浮点边界已在训练前测试时修正clamp[0,1]，新5测试/全套262通过。
预注册`docs/roi_geometric_objectness_plan_20260912.md`，runner及输出独立，待launch。

04:36 UTC GPU空闲后launch PID285453，
`output/roi_geometric_objectness_pair_v1`几何OFF→ON各3200，不重复启动。
下一轮先初始数值预检、实际梯度、全部视角FP/重复和固定图，不只看296。

## 05:01 UTC检查：阶段36失败，阶段37上限审计，阶段38候选池

阶段36 OFF138.263秒/ON134.446秒，完整性通过；ON初始评分差1.788e-7，
几何参数delta L2=.402824，OFF0。双鼠43TP12FP→42TP15FP，全154TP155FP→
153TP198FP、重复33→40，未通过。已视觉看两臂固定图，296仍只黑鼠，停止
沿固定6中心框继续叠加评分机制，不靠加步数或改阈值解释负结果。

阶段37新CPU零forward最大二分匹配审计：原query几何上限178/216、双鼠54/64；
中心候选198/216、双鼠54/64；两者12框并集212/216、双鼠63/64，在8个双鼠
视角补充9个原query未覆盖目标。这是GT辅助几何容量，绝不是实际检出率。
报告`output/frozen_center_ceiling_audit_v1`及`output/candidate_pool_union_audit_v1`，
已视觉查看实际TP/FP路线图；新3上限测试和3pool测试，全套268通过。

阶段38预注册相同关系评分模型，center6对union12候选池，无NMS/框移动/GT
推理，无新数据；每臂3200相同步数/同warmstart/同one-to-one BCE。来源score
分布与负样本比例可能变化，明确不是纯参数开关。union训练前需两来源都有
正标签，保存source_label_audit。计划`docs/roi_candidate_pool_plan_20260912.md`，
新runner及输出独立；待launch，仍06:31截止、不覆盖原模型或混入旧mask框refiner。

05:11 UTC GPU空闲后已launch PID289122，`output/roi_candidate_pool_pair_v1`，
center6→union12顺序各3200。下轮优先两来源正标签审计与全开发/固定图；
若标签来源退化或预检失败不跳过，不将GT辅助上限冒充模型能力。

## 05:38 UTC检查：阶段38相对正收益，阶段39固定6400续训

阶段38完成，union单臂132.470秒。来源正标签original467/center464，预检通过。
center6→union12：双鼠43TP12FP→48TP4FP；全部154TP155FP33框重复视角→
165TP48FP8重复。相对池策略判据全部通过，绝对接入失败：原模型177TP47FP
3重复且双鼠53TP。已视觉看两臂固定图，union改善246棕鼠但296仍漏棕鼠。
训练单鼠OFF923/964TP、ON761/964TP；ON每800步平均BCE .11538/.05926/
.04299/.03752，训练/开发MAE .04380/.05613。不是仅凭loss就宣布欠拟合，
结合池策略相对正收益，安排一次固定预算续训检验。

阶段39预注册两臂真实恢复各自stage38 head与AdamW，3200再加3200到6400，
复现88初始分数0差；原model/center/数据与评分阈值不变，不挑best-dev。
新runner`tools/probe_roi_candidate_pool_budget6400.py`；计划
`docs/roi_candidate_pool_budget_plan_20260912.md`；两个6/12候选恢复测试通过，
全套270通过。05:38确认PID291122已启动OFF preparing，GPU独占检查启动前
空闲。输出`output/roi_candidate_pool_budget6400_v1`，旧脚本/结果保留。
下轮检查resume_audit、比较终点与3200，不把candidate-only结论扩展为姿态
头收益；仍06:31截止，不因授权自主就无限续训。

## 06:08 UTC：阶段39预算结论、阶段40监督审计、阶段41单变量对照

阶段39两臂6400完成，各140.217/137.724秒，真实head/AdamW恢复88分数0差。
center6/union12全部160TP181FP40框重复/164TP41FP6重复，双鼠44/44TP。
union相比自身3200，48TP→44TP、全FP48→41；不是续训正收益，不再沿
原loss加步数。训练单鼠761→612TP且平均BCE/MAE降低，不能只看总loss。
已视觉检查两臂固定图，296棕鼠仍漏，union分数8.89e-7。

阶段40新冻结审计452输入、零优化、126.306秒完成，输出
`output/roi_pool_supervision_audit_v1`。train单鼠964GT，旧正标签867、几何
覆盖959，92个可覆盖却无正标签；train双鼠64/64，无缺口。单鼠正标签
平均score .7342→.6475，低于.5的正标签158→281；负分数.02692→.01115。
开发两个step评分全复现。研究DINO matcher、RT-DETR VFL及Stable-DINO，
来源与边界记录`docs/roi_pool_supervision_followup_20260912.md`。

阶段41预注册只改变训练标签分配，固定union12/关系头/meanBCE/3200；ON
最大一对一合格IoU覆盖，OFF旧5L1-sourceScore，并要求OFF终点复现stage38
union88分数0差；不是推理GT匹配。新增4测试，全套274通过。06:07:56 UTC
GPU空闲后launch PID293488，输出`output/roi_coverage_supervision_pair_v1`，
新计划`docs/roi_coverage_supervision_plan_20260912.md`。原06:31截止，此窗口
最后一个训练对照；下一轮检查/视觉/报告收尾，暂停原自动化，不覆盖生产。

## 06:32 UTC截止收尾

阶段41两臂完成133.505/133.144秒，OFF88复现stage38 union0差。ON训练正
标签931→1023，来源467/464→206/817，训练单鼠TP761→896但重复37→100。
开发双鼠48TP4FP→45TP11FP，全部165TP48FP8重复→160TP56FP12重复，五项
相对判据全失败。两臂四帧已视觉检查，296仍漏棕鼠，352新增棕鼠漏检。
未达接入门槛，不seed29，不覆盖生产。收尾274测试通过。

06:31:03 UTC窗口到期后未启动任何新训练；GPU空闲、训练锁释放。
按OpenAI Docs技能读官方定时任务文档，通过应用工具暂停原mousegeoforge
heartbeat，回读PAUSED，原提示/周期/截止/目标任务保持。没有另建自动化。
总结`docs/instance_separation_ten_hour_summary_20260912.md`记录完整证据与
候选框/关键点联合解码的下一轮建议；这些建议尚未启动。
