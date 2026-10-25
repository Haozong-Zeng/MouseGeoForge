"""Summarize completed and partial cycle evidence without changing experiments."""
from __future__ import annotations
from collections import defaultdict
import datetime as dt
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mousegeoforge.semseg_training import write_json
OUT=ROOT/'output/calibrated_mouse_10h_20261025'


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    plan=read(OUT/'plan.json');state=read(OUT/'state.json')
    depth={}
    for name in ['depth_fit300_eval_v2','depth_D1_6000_s17_v2_dev','depth_D2_6000_s17_v2_dev']:
        value=read(OUT/name/'evaluation.json')
        if value:depth[name]=value['summary']
    segmentation={};campaign=OUT/'semseg_campaign_s17_v1'
    for arm in ['L','M','F','U']:
        value=read(campaign/f'{arm}_dev/evaluation.json')
        if not value:continue
        metrics=defaultdict(list)
        for row in value['ema']:
            metrics[f"{row['domain']}/{'bottom' if row['bottom'] else 'other'}"].append(row)
        segmentation[arm]={group:dict(count=len(rows),iou=float(np.mean([r['ref512_tol2']['iou'] for r in rows])),
            boundary_f1_ref512_tol2=float(np.mean([r['ref512_tol2']['boundary_f1'] for r in rows])),
            empty_fraction=float(np.mean([r['ref512_tol2']['empty'] for r in rows]))) for group,rows in metrics.items()}
    pipeline=read(OUT/'pipeline_D1_M_dev_v1/evaluation.json')
    result=dict(time=dt.datetime.now(dt.timezone.utc).isoformat(),plan=plan,state=state,
        audit=read(OUT/'data_audit.json'),depth=depth,segmentation=segmentation,
        campaign=read(campaign/'completion.json') or read(campaign/'progress.json'),
        full_pipeline=pipeline['summary'] if pipeline else None,
        limits=['V5 six motion poses; rig/environment DEV not unseen-pose proof',
            'Real mask data not dense real depth truth; no real millimeter-accuracy claim',
            'Depth-only evaluation uses oracle support; full-pipeline evaluation uses predicted masks',
            'Hair/glass excluded from synthetic skin truth',
            'Single-mouse union bbox is not multi-mouse instance detection'])
    write_json(OUT/'experiment_report.json',result)
    write_json(ROOT/'docs/experiments/calibrated_mouse_local10h_20261025.json',result)
    lines=['# 本机已标定小鼠分割与毫米深度：10小时循环（2026-10-25）','',
        '用户授权：已标定多视角，优先毫米尺度深度与点云；10小时自动推进。',
        '固定窗口：洛杉矶时间2026-10-25 02:29:14–12:29:14，11:59:14停止优化。',
        '入口：`output/calibrated_mouse_10h_20261025/plan.json`、`queue.json`、`state.json`。',
        '单卡串行、共享GPU锁；独立watchdog执行绝对截止；历史循环保持暂停。','',
        '## 数据与方法','',
        'V5只准入TRAIN576场景/1728视图、DEV144场景/432视图。分割另用既有真实人工mask TRAIN889、DEV447；不混旧合成mask，不读取封存模型评分。',
        '深度是冻结VGGT末层图像特征＋已知K/E的射线代价头，96个毫米深度候选。固定capture AABB由TRAIN皮肤点范围外扩50mm得到。输入不含目标GT mask/depth/关键点；GT mask仅用于监督及明确标出的oracle-support评估。',
        'D1/D2共用seed17、初值、场景与射线曝光，固定6000更新；D1 logdepth，D2另加0.2局部logdepth梯度。首个smoke在任何优化前发现参数组接口错误；原版本和失败保留，v2重新smoke后运行。',
        '分割L/M：同head、同初值、同曝光，512px/global batch4，完整LR日程18000；每500更新成对暂停恢复；最少6000、最多18000。所有stratum的TRAIN loss/IoU/边界F1连续5次监测满足阈值，才视为共同收敛；DEV不选终点。',
        '仅共同收敛且剩余时间足够，才从同一M EMA head分叉F/U各6000更新；U末1→2→4层解冻，末层LR1e-6、向前乘0.65、head3e-5。',
        '完整链路固定使用M共同终点EMA与D1，预测mask生成bbox和深度支持域，输出逐视角NPZ及同一时刻毫米世界点云PLY。','',
        '## 当前证据','',f"生成时间UTC：{result['time']}。当前状态：{state.get('status') if state else 'unknown'}。",'']
    for name,value in depth.items():
        lines.append(f"- {name}：平均逐视图RMSE {value['rmse_mm']:.3f} mm，AbsRel {value['absrel']:.4f}，可见表面点云F1@1/2/5mm = "+'/'.join(f"{value['pointcloud_fscore'][str(t)]:.3f}" for t in [1,2,5])+'。此处使用GT mask评估支持域。')
    if pipeline:lines.append('- 预测mask完整链路：`'+json.dumps(pipeline['summary'],ensure_ascii=False)+'`。')
    lines += ['','分割逐stratum、所有失败/跳过/部分任务、源代码/输入哈希和计算时间见相邻JSON及output内不可覆盖的job/log/checkpoint。','',
        '## 解释边界','',*['- '+value for value in result['limits']],
        '- 这是首轮可学习性与对照实验；不自动晋升发布，不提交H100，不把短smoke耗时推算为完整训练预算。','']
    (ROOT/'docs/calibrated_mouse_local10h_20261025.md').write_text('\n'.join(lines))
    print(json.dumps(dict(depth=depth,segmentation=segmentation,full_pipeline=result['full_pipeline']),indent=2))


if __name__=='__main__':main()
