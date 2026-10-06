# 实例重复：匹配成本与空间查询是两种不同机制

## 当前结构与证据

本地`MultiInstanceMouseHead`以6个learned query作为两层TransformerDecoder初始
内容，图像memory加入二维位置编码，但没有独立的逐层query位置输入、空间参考框
更新或mask注意力约束。最终token分别生成框、mask和关键点种子。
因此“框输出更准”和“query找对另一只鼠”不能混为一个问题。

已存frame296 top所有6query都缺少棕鼠mask，不是低分过滤。完整训练快照正在
检验mask成本能否改变当前分配；这只能证明当前checkpoint的可改变性，不能
推出从头训练时匹配从不重要。

## 可借鉴的一手实现

1. [DAB-DETR官方仓库](https://github.com/IDEA-Research/DAB-DETR)：把框坐标作为
   空间查询，并逐层更新，以中心和宽高调节注意力。可借鉴“内容与空间参考分离”；
   不是给两只鼠固定左右身份，也不是推理时强行拉开两框。
2. [Mask2Former官方解码器](https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/transformer_decoder/mask2former_transformer_decoder.py)：
   query内容和query位置分别保存，各层用预测mask限制cross-attention，全部被
   屏蔽的行回退为不屏蔽。基于本项目的推断：直接用已经指向黑鼠的mask硬裁注意力，
   可能进一步遗漏棕鼠，不能未经预检直接加上。
3. [Mask2Former官方匹配器](https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/matcher.py)：
   匹配中包含分类、mask BCE与Dice，并通过共享采样点节省计算。本项目的固定
   小规模诊断使用原mask网格所有像素以排除采样随机性，不能称为逐项复刻官方模型。

## 实验选择约束

先完成当前356训练输入的匹配/负监督快照，成本固定为原成本加2BCE+2Dice。
若改配比例<=1%且负监督完整，不优先安排matching-only训练；这不是证明未来
所有mask-aware训练无效，只是当前预算下证据较弱。

后续候选一：独立query身份/空间信息逐层注入，先做严格默认OFF/零初始化重放和
梯度预检。应保留全图搜索路径，不将所有queries都绑在已有预测mask或box范围。
后续候选二：在原训练源内增加空间/外观变化，单独对照；不同时修改分配/解码器。
前者是结构改进，后者是数据干预，不能混在一个ON里后归因给某一项。

具体结构、输入、初始化、参数范围和训练预算须在运行前独立锁定；当前本文是
路线筛选记录，不是已实现DAB-DETR/Mask2Former或已证明泛化收益。
截至本轮不新增依赖、克隆仓库、下载权重或修改生产head。
