# 阶段26：软空间查询先验

当前状态00:11 UTC：预检v3 passed，48.45秒；训练已按下面候选协议正式锁定并
launch，PID271128，`output/spatial_query_prior_seed17_v1`。原v1/v2因检查脚本
CPU/CUDA混用失败，保持原样；没有改模块或放宽验收。下文“待锁定”是历史计划。
OFF/ON每臂1600步，初始88旧预测重放和最终264评价冻结验证由runner执行。

2026-09-12 00:06 UTC。阶段25固定1600步无查询位置ON优势；预算敏感性已有一次
配对验证，暂不继续加倍。用户授权后按证据转空间机制，原06:31 UTC截止保持。

## 依据与区别

第296帧缺少棕鼠候选，继续抽象query位置向量在800/1600均没有解决。SMCA将
查询预测的中心/尺度转换为空间Gaussian权重，使注意力更有定位偏好；这是本次
小模块最直接的研究参考。[SMCA论文](https://openaccess.thecvf.com/content/ICCV2021/html/Gao_Fast_Convergence_of_DETR_With_Spatially_Modulated_Co-Attention_ICCV_2021_paper.html)、
[官方项目](https://github.com/gaopengcuhk/SMCA-DETR/)。
另一条参考DAB-DETR用框坐标逐层更新查询，含宽高调制；本次不实现其完整框细化
体系，也不把简单Gaussian模块称为完整复现。[DAB-DETR论文](https://arxiv.org/abs/2201.12329)。

## 已实现模块

`mousegeoforge/spatial_query_prior.py`独立插件，默认关闭，生产head源码不改。
2层、6个query、dim256，共2108参数：中心logits24、尺度logits24、强度12、
条件映射权重2048。初始中心为归一化3×2规则格点；这只是初始化，无固定分区。
条件映射从每层cross-attention Q的L2归一化向量产生xy和sigma偏移，权重初始0。
第一层Q尚未吸收图像，初始中心非图像驱动；第二层可随图像内容变化。不是独立
image-region proposal网络，也没有直接锚框监督或跨层框残差细化。

center=sigmoid(base+delta_xy)；sigma=.08+.92*sigmoid(base+delta_sigma)，
sigma初始.5；strength=softplus(logit)初始.1、上界2。给cross-attention logits
加`-.5*strength*sum(((grid-center)/sigma)^2)`，下界-20；grid按实际非方形
patch中心、归一化xy构造。各attention head共享本次先验，memory K/V原样。
所有位置保留有限权重；不施加query间距离损失，两个参考点允许完全重合。
多鼠靠近/遮挡时不强制把输出分开。没有GT、相机角色或目标数量作为推理输入。

OFF不安装hook，保持旧路径及数值；ON初始就是弱非零干预，不宣称等价。已有
attention mask/causal路径明确拒绝，不静默覆盖。context清理和非重入检查齐备。
7项CPU测试：OFF/RNG逐值、有限软权重与梯度、中心重合/移动、条件性、非方形xy、
V1/2/4前后向、异常清理；全套229测试通过。

## 当前预检（零优化）

`tools/preflight_spatial_query_prior.py`，输出`output/spatial_query_prior_preflight_v1`。
原seed17温度ONcheckpoint，88开发输入OFF/ON，OFF严格重放旧预测，ON全部输出
有限且depth/depth_conf严格不变；分别记录未训练ON与OFF指标及固定4帧预测/mask。
3个训练输入V1、2(双鼠)、4检查每个新参数张量梯度有限非零。安装不改变CPU/CUDA
RNG，原权重/encoder/flags不变，hook清理。没有优化器，没有梯度等价要求，因为
ON有意改变注意力。新参数梯度通过不代表实例分离已改善。

只有预检通过并检查初始行为后才能正式训练。若失败保留协议/日志，定位问题；
不把ON初始不等OFF当代码bug，亦不能放宽OFF0差或深度0差。

## 下一训练的候选协议（预检后正式锁定）

原seed17温度起点、原8双鼠姿态与800-slot序列，不叠加16姿态/新matcher/尾巴
开关/时序/旧mask框精修。两臂共同head-only，原head lr3e-6、temperature3e-4，
ON空间参数lr3e-4；OFF安装但不执行/不优化新模块。优先各1600步配对（原序列
两遍），不要因800试点结果快就误判收敛；保存实际800和1600优化状态。
初始ON需要单独报告，防止将初始化影响包装成学习收益。原88/17门槛不改，
PCK ON-OFF>=.01、NME<=1.02×OFF、TP不降、全视角FP及重复不增；固定296只是
诊断，不能取代全体验收。身体/尾巴拆分、深度/非head逐值保持必须报告。

这是待锁定训练方向，不是已启动训练。预检结果若暴露严重扰动，先审视偏置尺度
及能力边界，不用开发集反复调初始强度择优；需要调整时新版本单独记录。
若同预算收益达标，再seed29；不达标需区分参考点是否学动、注意力是否真正改变、
是否需要图像驱动候选/去噪监督，不凭累计失败次数停整个窗口。
