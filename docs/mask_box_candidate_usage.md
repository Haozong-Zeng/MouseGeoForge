# Mask参考框候选：显式启用，不替换生产模型

两套候选：`output/mask_box_candidates_v1/seed17/`和`seed29/`。
每套仅有新增框分支`refiner.pt`与`manifest.json`，没有优化器或复制的基础模型。
必须与manifest中对应的温度ON checkpoint及encoder配对，不能混搭两个种子。
完整导出验证状态见`output/mask_box_candidates_v1/verification.json`；若文件缺失、
未通过或源哈希已变化，不把该包视为已验证候选。

## 调用约定

已有model必须正确加载对应温度分支及全部基础权重，并处于eval模式；不要为了冻结
而统一切换requires_grad标志。用no_grad执行基础前向，禁止将基础参数加入新优化器。
原已验证输入为768×768或768×432，B,V,3,H,W，实际1–4视角；不是把时间维当视角维。

```python
from pathlib import Path
import torch
from mousegeoforge.mask_box_candidate_safe import MaskBoxCandidate

root = Path('/home/haozong/projects/MouseGeoForge')
candidate = MaskBoxCandidate.load(
    root / 'output/mask_box_candidates_v1/seed17',
    base_checkpoint=root / 'output/cylinder_adaptive_temperature_v1/adaptive_temperature/final_heads.pt',
    encoder_checkpoint=root / 'vggt-omega/vggt_omega_1b_512.pt',
    repository_root=root,
)

# model/images由现有推理流程提供；这里不加载/替换任何生产模型。
# 关闭路径也可直接使用MaskBoxCandidate()，无需加载候选文件。
enabled = False
with candidate.capture(model, enabled=enabled):
    with torch.no_grad():
        prediction = model(images, coupling='off')
    prediction = candidate(prediction, tuple(images.shape[-2:]), enabled=enabled)
```

明确需要使用候选时才把enabled设为True。关闭时capture不注册hook、不读取模型状态；
后处理原样返回同一个prediction对象，不访问mask。启用时上下文入口校验实际基础
模型和encoder的参数摘要，防止“传了正确文件名但实际model加载错了”的情况。
加载时校验文件与源码哈希，用weights_only读取新增分支；不会自动查找或切换生产权重。

启用的每次基础forward必须紧接一次后处理，不能先forward多个样本再一起后处理。
ragged输入请在同一个capture上下文中逐个forward、逐个消费；不要直接收集
`forward_ragged`全部结果后再消费，这会被计数校验拒绝。上下文不可嵌套/并发共享。
异常退出也清理hook和已捕获token；不会调用基础model的train/eval/requires_grad_。

## 改变与不改变

仅替换`instance_boxes_cxcywh`，仍为归一化cxcywh、FP32。其他字典值沿用原对象：
关键点、objectness、mask、depth、相机等不变。启用时要求B,V,Q轴一致及真实正整数
图像HW。空mask只屏蔽显式参考残差，普通学习残差仍可修正原框；不是保证训练后
空mask必定等于原框。空query返回相应空形状。错误形状、底模或训练mode会报错。

该版本在CPU上计算连通域/形状及FP32框分支，再把框放回原device。
验证记录中的postprocess延迟不包含encoder、基础前向和上下文入口参数哈希检查，
不是端到端FPS或实时性承诺。推荐safe加载入口，保留调用者CPU Torch随机数状态，
包括加载失败路径；只在单线程初始化阶段加载，不在多线程并发加载。旧v1加载器
会因临时初始化线性层消耗CPU随机数，保留它仅为历史验证溯源，不推荐直接调用。
safe入口完全继承同一capture/forward函数，权重未改变；CPU加载守卫证明见
`output/mask_box_candidates_v1/loading_guard_verification.json`，不冒称重新跑了GPU验证。

## 证据与限制

- 两个种子均为同一合成开发集88记录，正常和固定四角噪点下原17门槛通过；不是
  独立真实视频测试，更不是在真实Cylinder上的验证。
- 框定位改善不代表关键点坐标改善。框改变还可能改变评估的实例匹配，必须同时
  看未门控NME和检测指标；不要据此宣称端到端3D获得了相同提升。
- 帧296顶部重复关注黑鼠仍未解决，强置信度不等于框可靠；相机/时间身份关联需
  独立方案。mask参考框是可见轮廓范围，不是amodal范围，不强制遮挡关键点入框。
- 不自动选择最好种子、不修改生产checkpoint、不修改数据或历史实验源代码。

固定示例：`output/mask_reference_box_seed17_v1/fixed_failures.png`、
`output/mask_reference_box_seed29_v1/fixed_failures.png`。完整过程见
`docs/mask_box_six_hour_cycle_20260911.md`。
