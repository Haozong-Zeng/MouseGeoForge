# PyMouseStudioRaw：真实视频源验证准备与缺口

2026-09-12。用户授权导入Windows Downloads新数据，建立独立视频源验证，
检查镜头畸变与标注，并研究多目标检测。没有启动训练或恢复旧自动化。

## 已交付

- Windows下载目录中本次20260912的12个包全部导入
  `dataset/PyMouseStudioRaw/<session>/`；399个原始文件，7,170,943,488字节
  （7.17GB/约6.68GiB）。解压检查路径/重复条目/符号链接及CRC，保存压缩包
  和逐文件SHA256；交付检查再次校验399个原文件。原Windows压缩包保留。
- 78个RGB视频、66个mask视频、66个DLC H5、12份标定；不执行随包pickle。
  H5只读取数值数据集及白名单基础类型元数据。shape_params.npy为对象负载，
  未反序列化；不能据文件名将它当逐帧3D真值。motion_score.npy是1560项分数。
- 所有已提供视频的元数据均为1080×1200（宽×高）、52fps、1560帧/30秒。
  相机标定尺寸完全一致；mask尺寸/帧数/fps与各自RGB一致，DLC索引0–1559。
  12个序列总计6分钟同步场景时间，不能把78路摄像机时长相加当独立样本。
- 生成轻量NPZ关键点副本、规范化相机JSON和可读取manifest，不重复导出整段
  图像。正式入口`dataset/PyMouseStudioRaw/_validation_v2/sequences.jsonl`。
  `mousegeoforge/raw_validation.py`可按帧按1–N个可用相机读取，原分辨率返回，
  RGB/mask/关键点同步选择是否矫正；无标注时显式NaN/valid=false，无深度/3D
  时返回None，不编造零深度监督。它是验证适配器，不是现有Core23训练格式。
- `configs/validation/independent_sources_v1.json`定义新真实验证及旧合成回归。
  实测读取1/2/3/7视角，以及实际只有5视角的AZR2L2，锁定来源访问检查通过。
  281单元测试通过；交付报告`output/pymousestudio_raw_delivery_v2/report.json`。

首次派生预检因本机OpenCV5将undistortPointsIter合并为带criteria的
undistortPoints而停止；已经增加OpenCV4/5兼容及往返测试。`_validation_v1`
保留失败说明，仅使用成功的`_validation_v2`。不涉及CUDA依赖修改。
新增H5读取依赖h5py3.14.0隔离放在output/pymousestudio_audit_deps，没有更改
训练venv的torch/numpy。原始文件目录本身没有被纠正、重编码或替换。

## 独立源划分

在查看新模型预测之前按保守名称家族分组；同一家族的帧、相机、短段及
后续派生数据必须保持同一split，不按帧或相机随机切分。

|用途|序列|
|---|---|
|真实开发验证，9段|AZL3、AZR2、AZR2L2、CR2、EL2、EL2L2、ER3L2、FR2、FR3|
|锁定真实holdout，3段|GR2、UR2、UR3|

AZ/C/E/F与G/U家族隔离；这些前缀不等于已确认动物ID，后续若发现同一动物/
同一连续源跨组，须合并来源重新划分。全部真实序列train_allowed=false。
新reader默认拒绝读取locked holdout，直到冻结checkpoint与评估协议后显式解锁。
本轮只检查holdout文件结构/元数据，不看其像素、不做模型推理/选checkpoint。
每序列固定8个均匀帧和中部32连续帧，分开评价帧级与时序；没有平滑插帧。

原88合成开发集仅保留为回归，明确不是新独立测试集。扫描现有configs、docs、
output顶层protocol/manifest/summary中本次精确project-date名称，未发现命中；
78个RGB SHA256无重复。此证据仅覆盖已扫描文件，不能证明全部历史训练、
上游DLC或VGGT预训练从未见过相关数据。所有序列同采集日期/同类装置，也
不构成跨实验室泛化评估。

## 视觉与标注缺口

已看AZL3第0/780/1559帧7视角、780帧畸变对照和B/D/F的mask叠加。
可以确认本例是单鼠圆筒：A从底部看到腹部/足掌，B–G是周围侧视，没有顶部
视角。其他序列沿用相同装置编号的假设要保留证据标记，不用世界Z轴武断
判断；原Anipose标定世界系未确认ground-aligned。

|项目|目前状况|对验证的影响|
|---|---|---|
|A底视关键点与mask|12段均缺失|不能直接报告有真值的底视PCK/IoU|
|部分侧视视频|AZR2L2缺B/G；EL2L2、ER3L2、UR2、UR3缺C|7相机标定不等于7路数据；总缺6路，不补假视图|
|2D关键点|DLC预测，31点，带likelihood|不是人工GT；likelihood不是可见性标签|
|mask|随包二值前景视频，生成方式未证明为人工|抽查贴合鼠体及尾巴，仍需标注来源/边界复核；无多实例ID|
|逐帧3D/深度真值|未发现可直接验证使用的文件|不做绝对3D/深度精度指标；三角化结果只能作伪标签/诊断|
|时间同步|帧索引/fps一致；无硬件时间戳或offset证明|不能用相等长度宣称严格硬同步|
|多鼠样本|当前DLC结构为单轨迹，本例视觉为单鼠|可做底视域差和单鼠误检回归，不能单独验证双鼠分离|

现有mask含尾巴，mask外接框并不天然等同于此前body-only GT框。必须先锁定
body-only与full-animal两种标注定义，不能改变GT定义后将IoU变化当模型进步。

### Core23映射

保留全部source31不静默转换。后续逐项核对配置发现：hind_paw 实际对应 Ankle，
hind_paw_digit_2b 才对应随包模型的 HindPaw。因此撤回最初“hind_paw 直接对应
paw_center”的候选。nose、tail_root、tail_end、fore_paw 和 hind_paw_digit_2b
仅为待复核部位候选，并非已验证的自动映射。
耳尖不是 Ear01 骨起点；ulna/humerus/tibia不是凭名字就可等同肘/膝；Spine0–4也
不能未经定义核对映射neck/spine_center/lumbar。tail_0–4不保证等弧长分位点。
head_center/abdomen等缺失点不得填成零且标valid=true，眼中点等派生点需标明
derived而非人工标签。31点保留有助于以后作正确映射与重投影检查。
进一步发现本批 Cylinder 导出器的 elbow 使用 Wrist.head，abdomen_center
与 spine_center 的 joint/uv 重合（表面支持不同）。详见
[人工标注语义核对](pymousestudio_core23_annotation_20260912.md)。

## 畸变矫正：参数有，但处理历史仍需确认

标定包含K、5项径向/切向系数、Rodrigues旋转、平移，metadata标注3Derror mm，
多份config也写unit:mm。按OpenCV world-to-camera规范保存[R|t]；保留毫米，
后续若接以米为单位的数据管线，t/点/深度须一起除1000，K不能随之缩放。

原图若未矫正，应使用每相机自己的K,D生成remap：RGB双线性、mask最近邻，
关键点同模型undistortPoints；输出选择同K/同尺寸，D置0，不额外裁剪。
以后resize/crop/pad须同时更新K和点。仅改D=0而不变换像素/点是错误的。
参见[OpenCV官方标定说明](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)。

实际镜头对照会改变边缘几何；本轮保持原视频不动，仅产生可视化和显式开关。
侧视DLC高置信点的leave-one-camera-out重投影并未一致支持一种处理历史：
例如AZL3按带畸变/已矫正假设中位误差10.60/10.52px，FR2为5.88/6.42px。
这些不是标定板误差或人工真值；DLC误差、时间偏移和玻璃折射等都可能影响。
故不能凭非零D或单幅图就断言输入一定未矫正，已询问用户确认以避免二次矫正。
透过圆筒玻璃的反光/折射也不是普通镜头畸变模型能自动消除的。

## 下一步标注与评估

已生成288个待人工复核条目（12段×8帧×A/D/E），仅路径/帧索引，不附模型
预测，文件在`output/pymousestudio_raw_delivery_v2/annotation_tasks.jsonl`。
优先补底视A的鼠体框、关键点和可见性；侧视抽查DLC与mask。锁定来源的标签
由独立标注过程完成，模型评估前不用于调阈值。之后才能计算真实accuracy。

真实双鼠验证还需从现有SBeA/CRIM13等审计未用过的独立动物/视频组并复核
实例ID/标注，不把本批单鼠伪装成多鼠测试。当前没有跑新模型推理或训练。

### 读取示例

```python
from mousegeoforge.raw_validation import RawVideoValidation
reader = RawVideoValidation(
    'dataset/PyMouseStudioRaw/_validation_v2/sequences.jsonl',
    '/home/haozong/projects/MouseGeoForge')
sample = reader.read('pymousestudio_raw/AZL3', 780,
                     cameras=['A', 'D', 'E'], undistort=False)
# 仅在确认输入尚未矫正后，使用undistort=True。
# images:[V,1200,1080,3]；source31:[V,1,31,2]；A无标签；depth/3D为None。
```
