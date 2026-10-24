# 独立分割训练引擎与本地 smoke（2026-10-24）

实现入口：[train_mouse_semseg_20261024.py](../tools/train_mouse_semseg_20261024.py)，模型：[mouse_semseg.py](../mousegeoforge/models/mouse_semseg.py)，数据/状态工具：[semseg_training.py](../mousegeoforge/semseg_training.py)，测试：[test_mouse_semseg_20261024.py](../tests/test_mouse_semseg_20261024.py)。这是[分割路线](mouse_semseg_depth_anything_route_20261024.md)的第一版可运行训练引擎；不依赖关键点代码、旧geometry脚本或MMSeg/MMCV CUDA扩展。

## 已实现与差异

- 只从固定发布checkpoint加载VGGT的DINO图像encoder；提取blocks20/21/22/23。L将最后层复制成四输入，M使用真实四层，F/U支持相同head父端点的固定预算续训。没有加载实际Depth Anything encoder权重。
- 纯PyTorch的多尺度可变形pixel attention使用`grid_sample`，query decoder采用masked cross-attention+self-attention；二类semantic scores排除no-object类；class/BCE/Dice权重2/5/5、逐图Hungarian匹配。head维度256、16queries、3+3层。
- **实现差异**：损失/匹配用完整的固定mask网格和辅助层损失均值，没有官方框架的不确定性点抽样或相同normalizer；这是Mask2Former-inspired轻量实现，不是官方mmseg checkpoint/config的精确复现。单层/多层主对照必须共用这一实现。
- 新head的query attention屏蔽key bias的理论零梯度，避免Adam放大归约舍入噪声。原失败的single/DDP参数对拍与修复回执均保留。
- 原图全画面letterbox，显式背景padding监督、ROI先裁再逆变换；保留未知255，不把未知标成背景。PNG旧标签默认binary0/255，新的class-index标签须在item指定`label_encoding=class_index_0_1_255`。
- 输入RGB/mask哈希检查，四层均衡的全局stateless采样，microbatch梯度累积；BF16/FP32，梯度有限/冻结字节检查；原生与ref512边界指标、raw/EMA二值mask和概率图。
- U按末1→2→4层解冻，LR末层1e-6、向前乘0.65，final norm同时开放；动态解冻时重建DDP reducer，不能漏掉新层的同步。head与encoder的warmup独立，未开放参数requires_grad=false。
- 原子checkpoint保存活动raw参数、EMA、AdamW和参数组、冻结标志、buffer、逐rank Python/NumPy/CPU/当前CUDA设备RNG、源/数据/日程pin、global_step和曝光记录；其余encoder参数由固定发布权重恢复。只读取当前设备RNG，不为每个rank初始化其余GPU上下文。采样无预取隐藏游标。BF16/FP32不使用GradScaler，协议明确记录。训练恢复要求原world size、数据、源代码、precision和日程一致，当前不支持elastic重分片；单进程evaluate允许读取多rank checkpoint，但不恢复训练器/其他rank RNG。

本引擎**不是整轮科学实验控制器**：不会自动选head共同收敛终点、启动多seed F/U、判定晋升或执行实际DA对照。`train`只执行显式的单臂固定预算，默认TRAIN监测每500步，输出完整序列供远端共同终点控制器使用。`--allow-unregistered-fixed-budget`要求操作者明确选择该单臂模式；不能把design_only JSON直接传入当已登记的完整实验。正式pair调度仍须按路线另写任务专用run protocol，不把smoke端点用于正式初始化。

## 本地验证范围

本机：PyTorch2.11.0+cu128，单张RTX PRO 6000 Blackwell Max-Q。8个unittest检查Hungarian query置换不变性、dense loss梯度累积、未知标签、padding与逆变换、源排除、四层/最终层接口一致，以及在新层开放边界的完整恢复。

真实发布权重+已准入TRAIN图像，512px/BF16，短L、M与U smoke均运行；U将2,000步阶段压缩为1步，仅用于覆盖1→2→4层工程路径，不代表学习率或分割效果验收。四层均为 `[1,1024,32,32]`，最后层与正常encoder输出一致，head/开放block有非零梯度，冻结参数不变。

两进程CPU/Gloo fixture验证DDP、全局batch分片和阶段切换，并与单进程同有效batch对拍；断点从第1步恢复，跨下一解冻边界继续至第4步。真实GPU也做一次跨第2→3阶段的恢复对拍。数值与哈希汇总见[紧凑smoke回执](experiments/mouse_semseg_smoke_20261024.json)。fixture是微型随机encoder，不可代替真实多GPU验收。

**本机没有多张GPU，H100/NCCL未实测**。已有torchrun NCCL代码分支与[单节点8卡Slurm模板](../scripts/train_mouse_semseg_h100.sbatch)，远端首次需先跑短DDP smoke，核对每rank参数/样本/显存再启动长作业。不将本机短步耗时当H100吞吐估计。

## 本地命令

以下Linux命令从仓库根执行；解释器替换为包含torch/scipy/Pillow/opencv的环境。依赖可通过本项目的`segmentation` extra准备。输出目录必须是新的，已有完成run不覆盖。

```bash
.venv-vggt-omega-eval/bin/python -m unittest discover -s tests -p test_mouse_semseg_20261024.py -v

# 验证原历史items的全部6435项RGB/mask pin，生成便携任务清单；不训练。
.venv-vggt-omega-eval/bin/python tools/train_mouse_semseg_20261024.py prepare \
  --items output/vggt_mouse_segmentation_16h_20261020/items.json \
  --output-manifest output/NEW_semseg_items.json

# 真权重/真TRAIN图片；自动仅取四个TRAIN stratum各一张，最多8个工程更新。
.venv-vggt-omega-eval/bin/python tools/train_mouse_semseg_20261024.py smoke \
  --items output/NEW_semseg_items.json --run-dir output/NEW_semseg_smoke \
  --arm U --steps 3 --global-batch 4 --micro-batch 1 --side 512 --device cuda --precision bf16

# 无模型/数据下载的两进程CPU工程fixture，压缩阶段；不作为精度实验。
.venv-vggt-omega-eval/bin/python -m torch.distributed.run --standalone --nproc-per-node=2 \
  tools/train_mouse_semseg_20261024.py smoke --fixture --side 64 --device cpu --precision fp32 \
  --arm U --steps 4 --global-batch 4 --save-every 1 --run-dir output/NEW_ddp_fixture
```

`--resume checkpoint.pt --run-dir NEW_DIR`可在相同arm/steps/数据/源版本下恢复；完成目录保持不变。`--parent`只初始化F/U的head EMA，两个分支分别重建optimizer，不能把resume与parent混为同一科学对照。`evaluate --resume ...`要求单进程和相同arm/steps/清单/precision；允许训练world size不同，单独输出DEV raw/EMA指标与checkpoint哈希；eval不会读取sealed。

## 潜在H100提交

模板只支持**单节点多卡DDP**，未实现多节点FSDP/弹性恢复。`SEMSEG_NPROC`须与分配GPU数一致，global batch8可用于1/2/4/8卡、microbatch1；不同world size不用于同一checkpoint的精确恢复。

```bash
# 由远端设置其路径和账户/partition；不要直接复制本机路径。
export MOUSEGEOFORGE_ROOT=/path/to/MouseGeoForge
export SEMSEG_PYTHON=/path/to/environment/bin/python
export SEMSEG_ITEMS=/data/manifests/mouse_semseg_items.json
export SEMSEG_RUN=/data/output/NEW_SEG_M_seed17
export SEMSEG_ARM=M
export SEMSEG_NPROC=8
mkdir -p /data/output/slurm_logs
# 在完成远端短NCCL预检、固定单臂run protocol后，才在远端显式执行：
sbatch --partition=YOUR_H100_PARTITION --account=YOUR_ACCOUNT \
  --output=/data/output/slurm_logs/%x-%j.out scripts/train_mouse_semseg_h100.sbatch
```

不在模板里硬编码账户或修改Slurm提供的CUDA_VISIBLE_DEVICES；每节点只有一个torchrun launcher。默认M/L18,000步、F/U6,000步，可通过SEMSEG_STEPS显式设置已登记预算。F/U还需在模板尾参传`--parent /data/.../head_endpoint.pt`。远端如只想做短工程预检，使用同一torchrun命令的`smoke`子命令，不运行上述长作业。当前本机未调用sbatch，也未启动正式分割训练。
