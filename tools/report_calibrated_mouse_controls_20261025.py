"""Companion evidence report; does not overwrite the pinned live core reporter."""
from __future__ import annotations
import datetime as dt
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from mousegeoforge.semseg_training import digest,write_json
OUT=ROOT/'output/calibrated_mouse_10h_20261025'


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    coverage=read(OUT/'coverage_audit_v1.json');sensitivity=read(OUT/'depth_train_sensitivity_v1.json')
    result=dict(time=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_sha256=digest(Path(__file__)),coverage=None,input_sensitivity=None,held_source_frame_pairs={})
    if coverage:
        result['coverage']={k:coverage[k] for k in ['summary','exact_train_dev_rgb_overlap_count',
            'instance_count_ground_truth_available','component_counts_do_not_establish_instance_counts','empty_background_fpr_evaluable']}
        result['coverage']['receipt_sha256']=digest(OUT/'coverage_audit_v1.json')
    if sensitivity:
        result['input_sensitivity']=dict(split=sensitivity['split'],device=sensitivity['device'],
            interpretation=sensitivity['interpretation'],arms={a:v['summary'] for a,v in sensitivity['arms'].items()},
            receipt_sha256=digest(OUT/'depth_train_sensitivity_v1.json'))
    for seed in [17,29,43]:
        paths={arm:OUT/f'depth_hold145_s{seed}_{arm}_v3' for arm in ['cost','reference']}
        values={arm:read(path.with_name(path.name+'_dev')/'evaluation.json') for arm,path in paths.items()}
        if not all(values.values()):continue
        protocols={arm:torch.load(path/'step_006000.pt',map_location='cpu',weights_only=False)['protocol'] for arm,path in paths.items()}
        required=['seed','steps','planes','rays','holdout_frame','gradient_weight','capture_aabb_mm',
                  'train_scene_indices','manifest_sha256','cache_receipt_sha256','source_sha256']
        different=[key for key in required if protocols['cost'][key]!=protocols['reference'][key]]
        comparable=not different and protocols['cost']['cost_enabled'] and not protocols['reference']['cost_enabled']
        pair=dict(comparable=bool(comparable),differing_required_fields=different,
            cost=values['cost']['breakdown'],reference=values['reference']['breakdown'])
        if comparable:
            pair['cost_minus_reference']={}
            for cohort in values['cost']['breakdown']:
                c,r=values['cost']['breakdown'][cohort],values['reference']['breakdown'][cohort]
                assert c['scenes']==r['scenes']
                pair['cost_minus_reference'][cohort]=dict(mean_view_rmse_mm=c['mean_view_rmse_mm']-r['mean_view_rmse_mm'],
                    pointcloud_fscore={str(t):c['pointcloud_fscore'][str(t)]-r['pointcloud_fscore'][str(t)] for t in [1,2,5]})
        result['held_source_frame_pairs'][str(seed)]=pair
    result['limits']=['No admitted empty-background frames; absence-of-mouse false-positive rate cannot be evaluated',
        'Mask components do not identify multiple mice; no instance-count claim',
        'Border contact is an observation, not a certificate of which anatomy is outside the camera',
        'TRAIN perturbation removes/changes trained inputs; it is not a matched retrained control',
        'Six correlated synthetic poses; held source-frame experiment is not independent real metric-depth validation']
    write_json(OUT/'controls_report.json',result)
    write_json(ROOT/'docs/experiments/calibrated_mouse_controls_20261025.json',result)
    lines=['# 已标定小鼠实验：数据覆盖与几何对照（2026-10-25）','',
        '这是本机10小时循环的补充证据；主训练、停止规则与数据清单均未改变。',
        '主记录：[calibrated_mouse_local10h_20261025.md](calibrated_mouse_local10h_20261025.md)。','',
        '## 数据覆盖','']
    if coverage:
        total=sum(v['images'] for v in coverage['summary'].values());empty=sum(v['empty'] for v in coverage['summary'].values())
        lines += [f"已核验{total}张TRAIN/DEV标签，其中空背景{empty}张；跨TRAIN/DEV完全相同RGB哈希重叠{coverage['exact_train_dev_rgb_overlap_count']}项。",'',
            '当前不能报告无鼠误检率。多连通区域可能来自遮挡或肢体分离，不作为实例数量。','',
            '| 数据层 | 张数 | 鼠体接触画面边缘 | 中位鼠体面积比例 |','|---|---:|---:|---:|']
        for name,value in coverage['summary'].items():
            lines.append(f"| {name} | {value['images']} | {value['boundary_touch']} | {value['median_foreground_fraction']:.2%} |")
    lines += ['','## TRAIN输入扰动','',
        '两种深度头各取12个TRAIN场景×3视角×64固定射线，全部CPU运行，未占用训练GPU。',
        '下表为探针逐视图RMSE的均值，不能与全量DEV指标直接比较。','',
        '| 模型 | 原始K/E及匹配（mm） | 源相机x平移+10mm（mm） | 关闭匹配输入（mm） |','|---|---:|---:|---:|']
    if sensitivity:
        for arm,value in sensitivity['arms'].items():
            s=value['summary'];lines.append(f"| {arm} | {s['mean_probe_rmse_mm']:.3f} | {s['source_camera_x_plus10mm']['mean_probe_rmse_mm']:.3f} | {s['matching_disabled']['mean_probe_rmse_mm']:.3f} |")
    lines += ['','扰动结果支持模型对标定/匹配输入有响应；关闭输入属于分布外诊断，不能据此证明跨视角训练优于重新训练的参考特征对照。','',
        '## 已登记的姿态留出对照','',
        'TRAIN仅用源帧49/97，第145帧从深度训练中留出；固定capture AABB也只由保留的TRAIN帧生成。',
        'seed17/29/43，各6000步；跨视角匹配与参考特征＋同一标定空间先验成对比较。分组报告DEV中已见/留出源帧；不用DEV选checkpoint。',
        f"当前已完成且可配对的seed数：{sum(bool(v['comparable']) for v in result['held_source_frame_pairs'].values())}/3。",'']
    for seed,pair in result['held_source_frame_pairs'].items():
        lines.append(f"- seed{seed}："+json.dumps(pair.get('cost_minus_reference',dict(comparable=False)),ensure_ascii=False))
    lines += ['','## 下一步边界','',
        '- 空背景必须作为另行登记的诊断数据建立并核验；不在当前L/M配对中插入新训练样本。',
        '- 若探索鼠体ROI提高几何分辨率，裁剪须来自预测分割或固定物理范围，并同步变换K；禁止用DEV真值框驱动推理。',
        '- 真实毫米精度仍需独立参考；三维完整性只针对同一时刻的可见表面。','']
    (ROOT/'docs/calibrated_mouse_controls_20261025.md').write_text('\n'.join(lines))
    print(json.dumps(dict(paired_seeds=list(result['held_source_frame_pairs']),coverage_images=total if coverage else 0),indent=2))


if __name__=='__main__':main()
