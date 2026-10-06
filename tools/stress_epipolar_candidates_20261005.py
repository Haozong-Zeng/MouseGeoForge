"""Fixed physical-view ablations, no GT-dependent support selection."""
import collections,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from mousegeoforge.epipolar_candidate_readout import probability_candidates,rerank_candidates
from mousegeoforge.detection_readout_loss import match_detection
from mousegeoforge.pilot_metrics import box_iou
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';SOURCE=CYCLE/'singleview_fusion_inputs_seed17_v1';MAPS=CYCLE/'singleview_probability_diagnostic_v1';OUT=CYCLE/'epipolar_candidate_stress_v1'
DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005');KP=[0,14,15,16,17]
VARIANTS=['single','paired','three','full','reverse','duplicate','drop_first_support']+[f'drop_support_cam{v}' for v in range(7)]


def main():
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2)
    data=torch.load(SOURCE/'validation.pt',map_location='cpu',weights_only=False)
    cp=CYCLE/'camera_geometry_audit_v2/validation_cameras.pt';scenes=torch.load(cp,map_location='cpu',weights_only=False)
    with np.load(MAPS/'validation_probabilities.npz') as a:prob=a['probability'];lookup={int(i):j for j,i in enumerate(a['row_indices'])}
    base=data['xy'].numpy();pred={name:base.copy() for name in VARIANTS};active={name:np.zeros(base.shape[:-1],bool) for name in VARIANTS};cohort=np.zeros(len(base),bool)
    for g in scenes:
        ids=g['row_indices'];cohort[ids]=True;views=g['physical_views'];score=data['objectness_logits'][ids].sigmoid().numpy();valid=(score>=.5).sum(-1)==1;queries=score.argmax(-1);pixel=data['pixels'][ids].numpy()
        candidates=np.empty((len(ids),5,5,2));masses=np.empty((len(ids),5,5))
        for v,i in enumerate(ids):
            for ki in range(5):candidates[v,ki],masses[v,ki]=probability_candidates(prob[lookup[i],queries[v],ki],pixel[v])
        for v,i in enumerate(ids):
            others=sorted([s for s in range(len(ids)) if s!=v],key=lambda s:views[s]);all_ids=[v]+others
            orders=dict(single=[v],paired=[v]+others[:1],three=[v]+others[:2],full=all_ids,reverse=list(reversed(all_ids)),duplicate=all_ids+all_ids,drop_first_support=[v]+others[1:])
            orders.update({f'drop_support_cam{view}':[v]+[s for s in others if views[s]!=view] for view in range(7)})
            for name,order in orders.items():
                target=order.index(v)
                for ki,kp in enumerate(KP):
                    q=int(queries[v]);r=rerank_candidates(g['projections'][order],candidates[order,ki],masses[order,ki],valid[order],np.array(views)[order],target,base[i,q,kp]*pixel[v],enabled=True)
                    if r['active']:pred[name][i,q,kp]=r['xy']/pixel[v];active[name][i,q,kp]=True
    for name in ['single','paired']:np.testing.assert_array_equal(pred[name],base)
    stress={}
    for name in ['reverse','duplicate']:
        delta=float(np.max(np.abs(pred[name]-pred['full'])));assert delta<1e-7;np.testing.assert_array_equal(active[name],active['full']);stress[name+'_max_delta']=delta
    with np.load(CYCLE/'epipolar_candidates_pilot_v1/validation_predictions.npz') as a:delta=float(np.max(np.abs(pred['full']-a['epipolar_local_xy'])));assert delta<1e-7;stress['full_replay_delta']=delta
    # Evaluation only after all variant predictions are finalized.
    matches=match_detection(dict(instance_boxes_cxcywh=data['boxes'],instance_objectness_logits=data['objectness_logits']),data['gt_boxes'],data['box_valid']&data['instance_valid']);accum=collections.defaultdict(lambda:dict(errors=[],hits=[],active=0))
    for i in np.flatnonzero(cohort):
        m=data['metadata'][i];kind='real' if m['record']['kind']=='real' else 'synthetic';cam='cam'+chr(65+int(m['view']))
        for qq,tt in zip(*matches[i]):
            q,t=int(qq),int(tt);detected=bool(data['objectness_logits'][i,q].sigmoid()>=.5 and box_iou(data['boxes'][i,q:q+1],data['gt_boxes'][i,t:t+1])[0,0]>=.5)
            for kp in KP:
                if not data['point_valid'][i,t,kp]:continue
                gt=data['gt_xy'][i,t,kp].numpy();pixel=data['pixels'][i].numpy()
                for name in VARIANTS:
                    err=float(np.linalg.norm((pred[name][i,q,kp]-gt)*pixel))
                    for group in [kind+'/reliable']+([kind+'/'+cam+'/feet'] if kp else []):
                        a=accum[(name,group)];a['errors'].append(err);a['hits'].append(detected and err<=32);a['active']+=int(active[name][i,q,kp])
    result=dict(stress=stress,metrics={name+'/'+group:dict(points=len(a['errors']),error_mean=float(np.mean(a['errors'])),gated_pck32=float(np.mean(a['hits'])),active_points=a['active']) for (name,group),a in accum.items()},
        protocol='Fixed physical-ID support selection, not GT. Drop_support_camX never removes the target camera; it only removes X when X is a support. All80cohortimages share evaluation denominators. Boxes/logits unchanged. Two views fallback because two OTHER support cameras required. No threshold selection or promotion.',script_sha256=fingerprint(Path(__file__)))
    write_json(OUT/'results.json',result);np.savez_compressed(OUT/'predictions.npz',image_indices=data['image_indices'],cohort=cohort,**{name+'_xy':pred[name] for name in VARIANTS},**{name+'_active':active[name] for name in VARIANTS})
    page=['<!doctype html><meta charset="utf-8"><title>候选融合视角压力测试</title><style>body{font:17px system-ui;margin:35px}td,th{padding:8px;border:1px solid #aaa}</style><h1>真实缓存的视角压力测试</h1><p>单视角/双视角精确回退；完整、倒序、重复视角结果一致。减少实际支持相机则可能改变候选，这是不同信息量，不是排列错误。所有方案固定，无验证阈值搜索。</p><table><tr><th>方案</th><th>可靠点误差</th><th>底视足点误差</th><th>底视激活点数</th></tr>']
    for name in VARIANTS:
        a=result['metrics'][name+'/real/reliable'];b=result['metrics'][name+'/real/camA/feet'];page.append(f'<tr><td>{name}</td><td>{a["error_mean"]:.3f}</td><td>{b["error_mean"]:.3f}</td><td>{b["active_points"]}/32</td></tr>')
    page.append('</table><p>模型输入像素；条件评价，不是AP或外部测试。drop_support_cam0–6对应A–G；仅删支持，始终保留待预测目标视角。两视角合成样本全部回退，不计为融合成果。</p>');(OUT/'index.html').write_text('\n'.join(page),encoding='utf-8');shutil.copytree(OUT,DEST/OUT.name);print(json.dumps(result))


if __name__=='__main__':main()
