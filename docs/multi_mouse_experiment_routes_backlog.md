# 多鼠姿态待实验路线档案

状态：Backlog。以下两条不作为当前主实现，但保留为必须完成的对照实验，防止只与旧单鼠头比较。

## EXP-TD-001：Detector + Top-down ViTPose

目的：建立强、易诊断的工程基线。

方案：

- 使用相同 VGGT/DINO 多层特征训练 mouse detector 或 instance segmenter；
- 按实例框/掩码裁剪 ROI；
- 每个 ROI 单独运行 ViTPose 风格 Core23 解码器；
- 跨视角和时序关联独立完成。

优点：检测与关键点误差容易拆分；每只鼠的局部空间分辨率高；成熟组件多。

风险：严重接触时 detector 错误会直接传递；同一只鼠逐 ROI 重复计算；裁剪会丢失跨鼠关系和全局几何。

进入条件：首版多鼠 Blender 数据完成，主架构已有可复现训练曲线。

对照要求：使用同一 backbone 初始化、训练帧和增强；报告 detector oracle-box 与 predicted-box 两组结果。

## EXP-BU-001：Bottom-up DEKR / PAF Assembly

目的：对照 maDLC 类“全图关键点定位＋个体装配”路线。

方案：

- 每类关键点预测多峰 heatmap 或中心条件 offset；
- 使用 PAF、center-to-joint vector 或图匹配将关键点组装为实例；
- 身份跟踪作为后处理模块。

优点：实例数自然可变；在 detector 漏框时仍可能恢复局部关键点；与现有 DLC 数据生态接近。

风险：同色鼠接触时组装歧义最大；长尾部和左右对称点容易跨实例连错；三维和端到端身份建模不如 query 架构直接。

进入条件：建立接触/遮挡测试子集，并能输出 PAF 或中心向量监督。

对照要求：除 PCK/OKS 外，必须报告实例组装准确率、跨鼠错误连接率和接触帧子集表现。

## 统一实验纪律

- 数据划分按动物/场景隔离，不按相邻帧随机拆分；
- 单鼠 demo checkpoint 不参与公平性约束，所有路线统一从 VGGT/DINO backbone 初始化；
- 统一 Core23、可见性定义、图像分辨率和有效标注 mask；
- 统一报告 box AP、mask AP、pose AP/PCK、遮挡分层、3D MPJPE、HOTA/IDF1；
- 主架构失败时先区分检测、实例装配、关键点、跨视角和 tracking 五类错误，再决定是否提升待实验路线优先级。

