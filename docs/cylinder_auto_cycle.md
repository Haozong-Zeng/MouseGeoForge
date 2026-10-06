# Cylinder 1–N 视角自主迭代台账

2026-09-12后续用户任务：真实圆筒数据现已从Windows下载目录导入
`dataset/PyMouseStudioRaw`，12序列/78RGB/66mask，A底视12段均无随包标注。
新来源验证manifest为`_validation_v2/sequences.jsonl`；9开发/3锁定holdout，
不加入训练。镜头参数有，但是否已矫正仍待确认；没有新真实精度结果。
详见`docs/pymousestudio_raw_validation_20260912.md`及
`docs/multi_object_detection_design_review_20260912.md`。历史“真实数据不在本地”
已被本次导入更新；历史实验结果仍全部来自原数据，不追溯声称真实验证通过。
旧自动化保持PAUSED，本次未训练/推理新模型，未覆盖生产。

窗口已于06:31:03 UTC结束，06:32收尾：阶段41两臂完成，OFF88复现0差，
ON双鼠48→45TP、全165TP48FP8框重复→160TP56FP12重复，全部相对判据失败。
两臂固定图已视觉看，296仍漏棕鼠，352新增棕鼠漏检。274测试再次通过。
GPU空闲/训练锁释放；已有mousegeoforge自动化已通过应用工具PAUSED并回读。
不再自动launch，生产不替换，不继续冻结框评分路线加预算。
最终总结与下一轮未启动建议：`docs/instance_separation_ten_hour_summary_20260912.md`。
若以后用户重新授权，从总结恢复，不依据下面历史“已launch”文字重复运行。

最新06:08 UTC：阶段39实际6400完成，两臂恢复88分数0差。center6/union12
全部160TP181FP40框重复/164TP41FP6重复，双鼠44/44TP。union自身3200的48TP
降为44，停止原loss加预算；两臂固定图已查看，296均漏棕鼠。
阶段40监督审计完成：train单鼠几何覆盖959但旧正标签867，缺92；train双鼠
64/64没有缺口，不称全部漏鼠原因。train单鼠正标签score下降、负标签压低，
平均MAE掩盖召回下降。审计目录`output/roi_pool_supervision_audit_v1`，零优化。
阶段41仅改变一对一训练标签为覆盖优先，其他union12/头/meanBCE/3200均相同，
OFF须完整复现stage38 union88评分0差。274测试通过，已launch PID293488，
输出`output/roi_coverage_supervision_pair_v1`，计划
`docs/roi_coverage_supervision_plan_20260912.md`。下轮先status/log/source_label_audit/
preflight/comparison及两臂固定图，不重复launch。内层off/on是输入分数/新评分，
外层是标签分配OFF/ON；MAE目标不同不可直接作loss对照。原截止06:31 UTC，
到时收尾暂停既有mousegeoforge automation，不再新开实验。未覆盖生产。

最新05:38 UTC：阶段38候选池相对通过，center6/union12双鼠43TP12FP→
48TP4FP，全部154TP155FP33框重复→165TP48FP8重复，但仍未达原模型
177TP47FP3重复/双鼠53TP；296仍漏棕鼠，两臂固定四帧已视觉检查。
ON每800步平均BCE .11538→.05926→.04299→.03752且训练单鼠761/964，
支持一次受控预算对照。阶段39两臂真实恢复3200 head+AdamW再到固定6400，
不改数据/阈值/候选架构。270测试通过，已launch PID291122，当前OFF preparing，
输出`output/roi_candidate_pool_budget6400_v1`；新计划
`docs/roi_candidate_pool_budget_plan_20260912.md`。下轮先resume_audit两臂88分数0差、
status/log/comparison和固定图，比较3200/6400及原模型，不重复launch。
原截止06:31 UTC不变；开发未跟进就停止此预算路线，不机械加步数。

最新05:08 UTC：阶段36几何偏置未通过（双鼠43→42TP、全FP155→198），两臂
固定图已查看。阶段37几何上限审计：中心6与原query6双鼠上限各54/64，并集
12为63/64、全部212/216（GT辅助上限非实际检出）。因此阶段38不再叠加评分
模块，固定相同关系头对照center6/union12，各3200。268测试通过、已launch
PID289122，OFF→ON顺序。下轮先source_label_audit/预检/status/log/comparison/
固定图，不把63/64上限当实际检出，不重复launch。
计划`docs/roi_candidate_pool_plan_20260912.md`，输出`output/roi_candidate_pool_pair_v1`。

最新04:34 UTC：阶段35完成，独立/关系双鼠37/42TP、全开发75TP34FP/
113TP78FP、框重复4/15，关系仍未合格；296均漏棕鼠，已视觉看两臂四帧。
阶段36固定同一个关系warmstart，对照24参数相对框几何attention bias开/关，
各3200、不改框/数据/监督，无NMS。262测试通过、已launch PID285453，计划
`docs/roi_geometric_objectness_plan_20260912.md`，输出`output/roi_geometric_objectness_pair_v1`。
下轮先两臂status/log/preflight/comparison与固定图，检查ON初始<=1e-6，不
训练后放宽，勿重复launch；内层off/on仍是center旧分数/新评分。

最新04:05 UTC：阶段34完成，双鼠50TP12FP（中心30TP3FP），但全部157TP/
253FP、70框重复视角，未通过；maxIoU监督非唯一性，训练也重复。已视觉看
固定四帧。阶段35改相同一对一objectness监督，对照独立MLP/候选self-attention，
各3200、原框/模型不变、不NMS；257测试通过，已launch PID283396。计划
`docs/roi_set_objectness_plan_20260912.md`，新输出`output/roi_set_objectness_pair_v1`。
下轮先顶层及两臂status/log/gradient_preflight、comparison/report、固定四帧；
顶层off/on=独立/关系，内层off/on=center旧分数/新分数，避免混淆，勿重复launch。

最新03:40 UTC：阶段32数据16双鼠开发30TP3FP（控制19TP4FP），296两鼠检出，
但全开发61TP32FP，仍不够接入。阶段33零forward审计：34双鼠漏检中24仅
低分，只有1GT中心在背景。已视觉看四帧和中心/GTmask图。阶段34独立ROI
质量评分头已实现/251测试通过，计划`docs/roi_candidate_quality_plan_20260912.md`，
候选框/原模型冻结，无NMS/阈值改变，3200固定预算，已launch PID281326。
下一轮优先`output/roi_candidate_quality_v1`的status/log/preflight/report和
`fixed_quality.png`，比较旧候选及原模型两层标准；勿重复launch。

最新03:05 UTC：阶段31 OFF/ON12800均开发双鼠19/64，训练376→561/996TP。
只读重算ON训练双鼠32/32TP、0FP，说明此子集已拟合但开发未跟进；两臂四帧
已视觉检查，296仍各只检出一鼠。不再加步数，阶段32新数据8→16姿态对照，
两臂均contextON各12800，原800中仅96双鼠槽替换。246测试通过，已launch
PID279174，数据OFF8→数据ON16顺序执行；下轮先status/log/comparison和
各自report的shared_training_summary/dual_training_summary，及固定四帧图。
计划`docs/dense_center_context_diversity_plan_20260912.md`；新输出
`output/dense_center_context_diversity_v1`，不要重复阶段31。

最新02:34 UTC：阶段30 OFF/ON3200完成，原模型不变；ON训练233TP/32FP
优于OFF141TP/65FP（GT996），开发双鼠ON14/64却低于OFF19/64。已视觉检查。
两臂训练loss仍下降，预注册阶段31恢复各自3200模型+AdamW至12800，固定
6400/12800验证，246测试通过。计划`docs/dense_center_context_budget_plan_20260912.md`，
新runner/输出`dense_center_context_budget12800_v1`已launch PID277004，OFF→ON
顺序单GPU。下一轮先两臂status/log/fixed_checkpoint_metrics及顶层comparison，
再固定四帧视觉，勿重复launch或将弱探针比较称为端到端关键点收益。

最新结果（2026-09-12 02:09 UTC）：阶段29总6400完成，双鼠18TP/5FP，训练
216/996TP；296棕鼠找回但黑鼠低分，整体验收未通过，原模型不变。已视觉检查。
下一步阶段30局部上下文OFF/ON同初值各3200，计划
`docs/dense_center_context_pair_plan_20260912.md`，全套244测试通过，已launch
PID274982。输出新`output/dense_center_context_pair_v1`，OFF→ON顺序单GPU。
下一轮先顶层及off/on status/log、comparison/report和固定四帧；勿重复launch。

最新运行（2026-09-12 01:34 UTC）：阶段28候选探针1600完成，但高分双鼠仅8/64，
未达验收；top6几何覆盖55/64，296两正确框低于.5分。训练集检出亦低且loss还降，
已启动阶段29预算续训，PID273695，`output/dense_center_probe_budget6400_v1`。
恢复实际1600探针+AdamW到总6400（新增4800），数据/架构/loss/score.5不变；
保存3200/6400固定验证，不挑最佳点。240测试通过。下一轮先status/log/report/
fixed_checkpoint_metrics与fixed_candidates.png，勿重复launch。

最新运行（2026-09-12 01:05 UTC）：阶段28独立中心/框候选探针已launch，PID272774，
`output/dense_center_probe_v1`。239项测试通过；runner先提取444冻结输入、审计
中心碰撞、做V1/2/4梯度检查，全部通过才1600步新候选头训练。原模型不优化。
下一轮优先读该status/log、center_label_audit/gradient_preflight/report并查看
fixed_candidates.png；不重复launch，不将候选容量结果说成关键点OFF/ON收益。

最新结果（2026-09-12 00:40 UTC）：阶段27前景探针已完成129.80秒，原模型及
88输出不变。296两鼠均有前景信号/原生网格recall1，但双鼠precision/IoU仅.461，
未通过预声明候选质量条件，不能直接接入实例模型。已视觉检查边界膨胀/两鼠间
粘连。下一步实现独立中心热图+框候选容量诊断，先验证预测中心覆盖与FP，不改
原关键点头；计划`docs/dense_center_candidate_probe_plan_20260912.md`。当前无GPU
任务。不要重跑阶段27、调其阈值或把前景并集说成实例分离已成功。

最新阶段（2026-09-12 00:38 UTC）：阶段26空间先验1600两臂已completed/verified，
PCK仅+.001359、TP仍54/64、296漏棕鼠，参数确实更新；未达收益，不seed29。
阶段27只训练257参数的冻结特征前景诊断，PID271906，
`output/frozen_foreground_probe_v1`，当前extracting；不改原模型/生产，不重复launch。
下一轮先检查此status/log/report并视觉查看fixed_foreground.png，判断是否支持
图像驱动候选路线。方案见`docs/frozen_foreground_probe_plan_20260912.md`。

最新运行（2026-09-12 00:11 UTC）：空间先验预检v3已passed（v1/v2检查脚本设备
错误原样保留）。阶段26训练`output/spatial_query_prior_seed17_v1`，PID271128
已launch，OFF/ON各1600，原数据/head-only，ON2108空间参数。下一轮读此status/
日志/verification，勿重复启动；原query-position长训已完成、无独有收益。
尚需完成后新增空间版分析（不改锁定旧脚本），17门槛、身体/尾巴、FP/重复、
固定四帧视觉及与未训练ON区别；目前不是预检等待状态，也未完成新训练。

当前阶段（2026-09-12 00:06 UTC）：阶段25双臂1600步completed/verification passed，
ON无独有收益，困难296未修复，不仅凭train loss再加倍。阶段26空间先验预检
`output/spatial_query_prior_preflight_v1`，PID270163，零优化；下一轮先检查其
status/run.log/report，勿重复launch。新模块/7测试、全套229通过。
新计划`docs/spatial_query_prior_plan_20260912.md`；预检初始ON有意非零，OFF严格
0差，不能要求ON=OFF或宣称零初始化等价。当前尚未启动空间先验训练。

最新用户授权（2026-09-11 23:20 UTC附近）：负结果后可自主调研、增加训练步数
和调整架构，覆盖下方“两机制后停止整个窗口”的旧状态。原06:31:03 UTC截止
与每半小时检查不变。已更新既有自动化，按证据逐阶段推进，不机械空等。
阶段25：`output/query_position_seed17_budget1600_v1`，PID269051于23:23 UTC
启动同数据/同head-only设置的OFF/ON各1600步预算敏感性对照；不要重复launch。
计划及研究依据见`docs/instance_separation_adaptive_followup_20260911.md`。
下一轮先读status/log/verification；完成后新建分析，不改锁定800步文件。

当前状态（2026-09-11 22:01 UTC检查，22:08整理）：阶段24已完成，两种新训练
机制均未达到预声明实质收益条件，已触发本窗口停止新增训练。不跑seed29、不延长、
不启动第三机制。自动化仍ACTIVE至06:31:03 UTC，剩余仅明确问题的只读复盘和
半小时状态检查；截止再暂停自动化。勿从下方历史“下一步”重新launch。
阶段结论见`docs/instance_separation_cycle_results_20260911.md`，未覆盖生产。

最新授权（2026-09-11 20:31:03 UTC）：从现在起恢复10小时，每30分钟检查并推进，
截止2026-09-12 06:31:03 UTC（北京时间9月12日14:31:03）。既有自动化已恢复ACTIVE。
新工作台账为`docs/instance_separation_ten_hour_cycle_20260911.md`，优先双鼠实例重复/
匹配问题；以下六小时窗口停止记录仅为历史，不再阻止新授权内工作。
阶段1–19成果与锁定代码保持不变，新实验必须独立目录、单因素对照，生产不覆盖。

最新授权（2026-09-11 11:39:23 UTC）：恢复推进mask辅助框，接下来6小时每30分钟
检查并决定下一步，截止17:39:23 UTC。此授权替代本文旧的阶段12复验后暂停状态。
当前工作、边界与精确状态见 `docs/mask_box_six_hour_cycle_20260911.md`；阶段1–12
仍不得重跑，生产不推广。当前阶段13–19已完成：两种子显式参考框收益复验、默认
关闭候选导出及边界审计通过；结果见`docs/mask_box_cycle_results_20260911.md`。
六小时窗口已于2026-09-11 17:39:23 UTC结束；截止后通过应用工具将既有
mousegeoforge自动化设为PAUSED，17:40:05.802 UTC完成最终19文件哈希核验。
无GPU任务或训练锁；本轮已收尾，不再自动训练/检查，后续等待用户新指示。

用户在 2026-09-10 要求“自主循环推进下一步”，2026-09-11 改为“以5小时为单位
进行自动推进并根据反馈思考下一步”。已更新本任务已有 heartbeat（mousegeoforge），
从每10分钟改为每5小时启动一个自主推进/复盘周期，不另建重复任务。
每轮可串行完成多个有证据支持的步骤，结合数值、视觉及用户新反馈再决定下一步；
单轮最多5小时，不为凑时长空等或无依据训练。运行时/资源限制导致提前结束时，
保存精确状态供下轮接续，不宣称已连续工作5小时。
用户随后明确要求现在开始连续推进5小时，并授权按收敛情况自主增加单实验预算。
本轮起点2026-09-11 09:03:34 UTC，计划终点14:03:34 UTC；记录见
`docs/cylinder_five_hour_cycle_20260911.md`。新授权覆盖旧的固定800步/训练步数2倍
限制，但不扩大数据获取权限，不取消保留性能、单因素比较及失败停止条件。
每轮记录起止时间、完成项、否定假设和下一步理由；有实质结果时给出简短复盘，
无变化时保持安静，失败或需用户行动及时说明。原停止条件继续有效。
使用产品内自动跟进，不建立系统 cron；OpenAI Docs技能用于更新现有调度及
明确保留上下文的停止/交接规则。本地调度需电脑开机、应用运行且项目可访问。

## 当前边界

- 固定 768 输入，真实 Cylinder 不在本地；以合成开发集为诊断，不声称真实验证。
- 不覆盖生产模型、数据、历史锁定协议，不下载、重渲染、删除、付费或 git push。
- 每个候选以800步为起点，可依据收敛与泛化证据分阶段增加，扩展前锁定计划，
  对照与候选采用相同新增预算，不把更久训练当成新机制效果。至多一项GPU任务。
- 一次只改变一个方法变量；保留 17 项性能门槛和源数据/权重/采样预算核验。
- 连续两项方法无实质改善，暂停并总结；全部门槛通过则优先独立种子复验，
  复验结束暂停交付，不自动推广生产。数据规模超过既定规模两倍仍需先询问；
  训练步数增加已由用户授权，但须记录依据和明确终止上限。
- “实质改善”至少包括底视多视图平均身体 NME 降低 5%，且不能隐瞒保留性能退化；
  是否达到可用标准仍由完整门槛决定，不以指标平均掩盖失败。

## 阶段

1. 已完成：组合采样 V2。相对旧采样，多视底视 NME 下降 8–10%，10/17 门槛。
2. 已完成：粗定位/局部热图只读审计。窗外点多，硬改温度损伤双鼠。见
   `docs/cylinder_localization_audit_v1.md`。
3. 已完成并淘汰：粗身体坐标辅助监督 V1，目录 `output/cylinder_coarse_aux_v1/`，
   运行器 `tools/train_cylinder_coarse_aux.py`。对照是已完成的
   `output/cylinder_combinations_v2/combination_coverage`，不重复训练关闭组。
   新辅助权重 2.5，原最终 xy 权重 5；仅 Core23 身体16点，尾巴7:14排除，
   有效遮挡标签参与。复用原最终坐标 Hungarian 匹配，不重新按粗坐标匹配。
   采样/初始化/温度/半径/encoder/其余监督不变。
   执行验证通过，130项测试通过；9/17门槛，多视底视NME退化9–12%，入窗率也下降。
   连续无实质改善的方法计数目前 **1**，不得当作成功或继续加大辅助权重。
4. 已完成：**只读局部网格梯度路由审计**。阅读
   `docs/cylinder_coarse_aux_v1.md` 及 `docs/pose_overfit_diagnostic.md`，避免重复
   过去“保留热图loss＋detach”的不同实验。当前条件是热图loss=0、粗辅助关闭。
   从V2已有训练池选固定训练样本（不按开发结果选），比较完整路径与仅detach
   local sampling grid 的粗xy梯度方向、范数、GT方向对齐，确认前向逐值一致，
   不用GT给模型输入。可先固定少量单/双鼠、多视组合；区分分项梯度与最终总梯度。
   只有有机制证据时才实施第二个单因素800步训练；其基线应为V2新采样关闭组，
   不能从失败的粗辅助模型续训或同时改变温度/局部热图loss。若证据不支持，
   记录否定结果后依据源姿态覆盖/训练开发差距另定小对照，不能机械启动。
   结果见 `docs/cylinder_grid_gradient_audit_v1.md`，输出
   `output/cylinder_grid_gradient_audit_v1/`。24训练输入、两checkpoint、96次head前向，
   全部前向逐值一致，134项测试通过；底视身体点有害梯度方向逆转率为0，
   shared adapter总梯度完整/直接余弦约.9994。不满足预先的detach训练启动条件，
   **跳过detach-only训练**。这不是新的失败训练方法，连续失败方法计数仍为1。
5. 已完成：**相同相机组合的训练/开发泛化差距审计**（只读）。按V2协议使用
   全部16个Cylinder训练姿态和已有8个双鼠回放训练姿态，不扩大训练集或训练。
   用固定原始侧视对、底视、侧底、两侧底、四视图；让训练/开发相机组合和顺序一致，
   对比原始、V2新采样、粗辅助ON的保存权重。复用已有开发输出（如需补字段则只读
   重跑），核验标签/尺度/匹配。报告训练和开发的粗/最终body16、头部/四肢NME、
   GT入窗率、双鼠原视角保留，分离数据拟合与动作泛化。不要仅依据train loss下降
   选择延长训练。若源姿态覆盖不足，下一训练方案仍要预先锁定、保持有界预算。

   结果见 `docs/cylinder_pose_gap_v1.md`、`output/cylinder_pose_gap_v1/`。
   88个训练姿态输入、264次head前向；同Cylinder K/E差异0，投影误差<.001px。
   粗辅助让训练姿态多视底视NME下降约14%，开发动作却退化9–12%；
   同时旧侧视和双鼠保留受损。不是仅凭loss猜测，亦不能概括为全部训练已拟合。
6. 已完成训练源混合预检和**单个800步对照及完整验证**。
   见 `docs/cylinder_source_mix_v1.md`、`output/cylinder_source_mix_preflight_v1/`。
   用原manifest明确train的8个Cylinder Sample10姿态替换8个ClimbWall特殊姿态，
   不改变总40姿态/800步/400回放步/每槽V与HW及角色预算。WalkAround是test，禁止挪作训练。
   已验证8相机物理角色映射、V1–4读取、mask/深度/投影/场景固定尺度及源哈希。
   新wrapper准备及JSON协议重建/预算通过，144项测试通过；进程242999已完成。
   目录 `output/cylinder_source_mix_v1/`；禁止重复启动。
   训练前88条原始预测逐值相同，verify/analyze_cylinder_source_mix.py均完成。
   以V2作为关闭组、原始权重作为初始化，不启用失败粗辅助。
   多视底视身体NME平均降低22.4%，三个组合各7/8姿态改善，门槛14/17。
   双鼠Core23 NME/PCK和侧底改善幅度仍未通过；底视深度较V2退化20–53%，
   尽管仍通过相对原始的深度门槛。生产未推广，连续无实质改善方法计数重置为0。
7. 已完成：**只读保留/几何分解**，不加训练量。复用已保存的原始/V2/混合输出：
   同16个双鼠开发记录分解身体16/尾巴7、top/side、框匹配坐标误差与检测门控，
   防止将尾巴退化误归为身体定位。用户允许多鼠关闭尾巴，但不得改变历史17门槛来报成功。
   同8个Cylinder开发姿态检查侧底三角化共同支持点/重投影/objectness门控；
   混合组侧底2D改善，但3D支持率从20.3%降至10.9%，不能只报告条件3D误差。
   两侧底及四视图3D条件误差较好，也要核对支持点集。先定位原因，再锁定单机制
   800步保留约束或一致性实验；勿立刻增加步数、改尾巴＋新loss＋新采样一起上。
   结果 `docs/cylinder_retention_geometry_v1.md`、`output/cylinder_retention_geometry_v1/`。
   120份保存预测，CPU6秒，无encoder/optimizer；重算指标误差<5.35e-8，148项测试通过。
   双鼠全部漏检由框IoU<.5造成，不是score低；顶部尾巴误差增大。
   侧底取消objectness后支持点数不变，GT支持128/128；极线残差V2 36.35px→混合38.25px。
   V2与混合共同支持仅7点，失去19/新增7；三/四视图共同支持82/116点的3D误差确实改善。
8. **身体极线一致性预检、800步候选及完整验收已完成**。
   具体规范见阶段7文档末尾。先实现独立实验helper，不改锁定模型/旧loss；
   复用一次原Hungarian匹配和持久instance_id，身体16、有效校准视角对；V1返回可微0。
   预设图像对角线归一化Sampson残差smooth-L1 beta=.01、权重1，按实例平衡。
   先检验零权重原loss逐值一致、GT残差、置换/缺失/单视图/梯度有限，
   固定训练样本的辅助梯度中位范数不得超过原任务25%；失败则先审计，不扫权重。
   通过后以原始权重初始化、混合训练日程800步不变；不开粗辅助/改温度/删尾巴/
   加框蒸馏等第二机制。关闭组复用已完成混合训练；完整17门槛及深度/双鼠代价继续报告。
   见 `docs/cylinder_epipolar_v1.md`。预检20训练输入、两checkpoint、无更新，158测试通过；
   辅助梯度中位比例adapter约2%、instance head不足1%，零权重原loss差异0、GT残差近0。
   `output/cylinder_epipolar_v1/` 的epipolar_body已完成，禁止重跑/编辑锁定helper/runner。
   verify/analyze_cylinder_epipolar.py完成：100mono辅助0、700multi受监督、预算/初始化/哈希通过。
   多视底部NME仅下降1.78%（未达5%），仍14/17，双鼠PCK .49864→.48573。
   几何改善但不全面：侧底极线38.25→34.38px、支持14→16/128；两侧底101→109。
   初次localization只读汇总缺字段失败，保留原协议；独立retry_cylinder_epipolar_localization.py
   修正仅默认温度的汇总，在output/cylinder_epipolar_localization_v1_retry完成，预测差异0。
   侧底入窗63.3%→62.5%，未解决越窗；159项测试通过。连续无实质改善方法计数为1。
9. 已完成：**双鼠回放输出保留的训练侧可行性检查**，未启动训练。
   见阶段8文档末尾具体条件。只用既有8个双鼠回放训练姿态，原始teacher vs混合student；
   teacher高置信且GT框IoU≥.5，用GT实例身份连接，不用query序号直接配对。
   检验teacher框/身体/尾巴质量不劣于student、匹配/零权重/初始化零保持项/梯度量级；
   拟议normalized box+Core23 xy smooth-L1 beta=.01、权重1，仅双鼠回放，保留原GT监督。
   混合checkpoint处梯度中位比例≤25%；不过则记录原因，不能开发集调参后偷换条件。
   若通过再实施一个混合800步日程、原始初始化的保留候选，极线项关闭，不能从其ON续训。
   下一训练若也未达实质改善，连续2项后暂停总结。

   结果见 `docs/cylinder_replay_retention_v1.md`：31个teacher合格instance-view，
   teacher/student框IoU .84840/.89997，身体NME .06256/.06971，尾巴 .08591/.07897。
   框和尾巴质量不通过，因此跳过原拟议框＋Core23保持训练；不临时删项宣称通过。
   train-mode教师同路径前向后detach使初始化loss/梯度严格0，OFF差异0；
   adapter/head辅助梯度中位比例1.97%/.69%，167测试通过。
   两个预检技术失败（模式微差、numpy计数序列化）均保留；CPU恢复报告在
   output/cylinder_retention_preflight_v1_summary，training_supported=false。
10. 已完成：**双鼠回放覆盖只读审计**。从原manifest明确train、未属于既有8个
    回放姿态的记录，按源帧等距锁定16个；原始/混合模型、原始双视角，检查框、
    身体/尾巴和top/side检测，固定叠图。不借开发/test选姿态，不更改800步预算。
    详见阶段9文档末尾。证据支持再锁定增加回放姿态覆盖的单变量方案，
    暂不重新定义教师保持门槛，不叠加loss，不自动增加训练步数。

    见 `docs/cylinder_replay_coverage_v1.md`，24训练记录、48head前向、27.2秒。
    未回放/已回放Core23误差比1.0129，未达到1.15覆盖机制门槛，跳过增加姿态训练。
    两组混合模型检测均100%；身体NME均约.070，比原始高约10%，不局限未回放。
    未回放16姿态身体只有1个改善；已回放框和尾巴有收益，仍需保留分项差异。
    投影误差<.000131px，源/代码/权重通过，170测试通过，固定frame115/108已目视检查。
11. 已完成：**共享参数任务梯度只读审计**。具体锁定条件见阶段10文档末尾：
    8双鼠回放＋8Cylinder既有train姿态，原始/混合状态，分离身体定位和depth。
    不重跑旧local-grid梯度审计：本次看Cylinder任务对双鼠任务的参数梯度作用。
    单次原匹配、分adapter/head、无更新；检查组均值及姿态对分布，不以单个负余弦
    推导因果。证据不满足则不机械启动投影/加loss，不变相延长训练或改变既定门槛。

    见docs/cylinder_task_gradients_v1.md，16encoder/32head、38.3秒、173测试通过。
    完整任务vs双鼠身体平均余弦adapter原始/混合−.055/−.021，head−.0008/−.0024，
    不满足同参数组两checkpoint均<−.1门槛。深度对head梯度0，跳过梯度保护训练。
12. 首轮及独立训练顺序复验已完成：输入相关局部logit系数，方案见
    docs/cylinder_adaptive_temperature_v1.md。零初始化scale1、范围.5–8，只从token预测，
    不改窗口/GT监督，不统一T=8。先证明初始化前向/原参数梯度一致和新层可学习，
    再决定800步候选；可按用户新授权结合收敛证据做等预算OFF/ON分阶段扩展。
    首次固定梯度atol预检失败，原目录保留；重复性诊断证明OFF自身也有GPU浮点
    噪声，独立repeatable预检加入同路径重复对照及相对L2条件后通过，前向仍严格0。
    首轮800步、完整核验完成，177测试通过；进程250735已结束，禁止重跑。
    多视底部身体NME下降12.3%，17/17首次全过；双鼠深度仅比门槛低0.068%，
    不据此宣布稳定。只读定位显示主要是精修收益，粗定位/入窗率基本未解决。
    3D侧底支持14→21/128，两侧底101→108，四视124→127；共同支持四视3D
    误差17.10→17.81略退化，不能将全部17门槛通过解释成所有指标改善。
    因全过，优先独立seed29 OFF/ON各800步配对复验，而不是立刻加步数。
    09:25 UTC进程251994已启动output/cylinder_temperature_seed29_v1，禁止重复启动。
    seed29只打乱相同800输入槽次序，原初始化/输入多重集/88开发记录/17门槛不变。
    延长预算工具已准备但未运行，不将其描述为已执行。
    seed29两臂已完成，独立verify/analyze及视觉通过；ON底部多视NME下降11.41%，
    16/17，唯一失败是双鼠检测感知PCK .495924 < .506005。深度通过。
    两个近IoU=.5边界的框使ON检测53/64→51/64；坐标收益重复但完整验收不稳定。
    原PCK重算差异0，困难帧246棕鼠框偏移仍在，不能删尾巴/降低IoU追认成功。
    已触发复验后暂停交付条件；进程251994结束，禁止重跑，不延长预算，不推广。

阶段1–12及seed29配对复验均已完成，不需要重跑。按已有边界暂停自动推进并交付。
失败训练方法计数为0（两种子都有实质底部收益，但保留标准未稳定通过）。
后续恢复建议先核验GT框/实例mask/尾巴范围与推理框定义，再锁定单因素框保留方案；
不是立刻增加步数、重新启用被否定的teacher蒸馏或堆叠3D/窗口修改。
不得将两个相同开发集上的训练顺序种子当作独立真实数据验证。
每次补充此台账及阶段文档，持续保留生产模型。
