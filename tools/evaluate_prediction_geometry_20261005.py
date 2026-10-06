"""Fixed-rule support-only geometry pilot; do not rewrite ground truth."""
import collections
import datetime as dt
import json
import shutil
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from PIL import Image,ImageDraw
from mousegeoforge.prediction_only_triangulation import leave_one_out_projection
from mousegeoforge.detection_readout_loss import match_detection
from mousegeoforge.pilot_metrics import box_iou
from run_depth_pose_ablation import fingerprint,write_json
from training.finetune_blendedmvs import load_scene

CYCLE=ROOT/'output/singleview_fusion_cycle_20261005'
SOURCE=CYCLE/'singleview_fusion_inputs_seed17_v1'
CAMERAS=CYCLE/'camera_geometry_audit_v2'
OUT=CYCLE/'prediction_geometry_pilot_v1'
DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')
KP=[0,14,15,16,17]
ARMS=['baseline','weighted_loo','robust_loo','bounded_blend']


def main():
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2)
    deadline=dt.datetime.fromisoformat(json.loads((CYCLE/'plan.json').read_text())['training_stop_utc'].replace('Z','+00:00'))
    protocol=dict(points=[k+1 for k in KP],score_threshold=.5,minimum_robust_inliers=3,
        support_reprojection_threshold_px=16,variance_floor_px=16,min_ray_angle_degrees=2,
        blend=.5,max_target_shift_px=48,rule='All numerical constants fixed before inspecting results, in model-input pixel units; no validation search.',
        scope='Same single-instance cohort. Only unique predicted confident query per camera; GT is eval-only. All target physical camera observations excluded from triangulation, including duplicate views. No runtime GT visibility/count/identity.',
        caveats='No correspondence guarantee for multiple animals; no real 3D GT; low own reprojection residual does not establish accuracy; production unchanged.')
    write_json(OUT/'protocol.json',protocol)
    cammanifest=json.loads((CAMERAS/'results.json').read_text());source=json.loads((SOURCE/'manifest.json').read_text())
    results=[];observations=[];valdata=None;valpred=None;valcohort=None
    for part in source['parts']:
        role=part['role'];assert fingerprint(ROOT/part['path'])==part['sha256']
        data=torch.load(ROOT/part['path'],map_location='cpu',weights_only=False)
        cp=next(p for p in cammanifest['camera_artifacts'] if p['role']==role)
        assert fingerprint(ROOT/cp['path'])==cp['sha256']
        cameras=torch.load(ROOT/cp['path'],map_location='cpu',weights_only=False)
        base=data['xy'].numpy();pred={arm:base.copy() for arm in ARMS}
        cohort=np.zeros(len(base),bool);active={arm:np.zeros(base.shape[:-1],bool) for arm in ARMS}
        failures=collections.Counter();supported=0
        for gi,g in enumerate(cameras):
            assert dt.datetime.now(dt.timezone.utc)<deadline
            ids=g['row_indices'];cohort[ids]=True;pixel=data['pixels'][ids].numpy()
            score=data['objectness_logits'][ids].sigmoid().numpy();unique=(score>=.5).sum(-1)==1;q=score.argmax(-1)
            uv=base[ids][np.arange(len(ids)),q]*pixel[:,None]
            variance=data['variance'][ids].numpy()[np.arange(len(ids)),q]
            weight=score[np.arange(len(ids)),q,None]/((variance*pixel[:,None]**2).sum(-1)+16**2)
            for vi,i in enumerate(ids):
                if not unique[vi]:continue
                for kp in KP:
                    for robust,arm in [(False,'weighted_loo'),(True,'robust_loo')]:
                        r=leave_one_out_projection(g['projections'],uv[:,kp],weight[:,kp],unique,g['physical_views'],vi,robust=robust)
                        if r is None:failures[arm]+=1;continue
                        target=r['uv'];delta=float(np.linalg.norm(target-uv[vi,kp]))
                        # No out-of-frame clamping masquerading as improvement.
                        if not ((target>=0).all() and (target<pixel[vi]).all()):failures[arm+'_outside']+=1;continue
                        pred[arm][i,q[vi],kp]=target/pixel[vi];active[arm][i,q[vi],kp]=True
                        if robust and delta<=48:
                            pred['bounded_blend'][i,q[vi],kp]=.5*(target/pixel[vi]+base[i,q[vi],kp]);active['bounded_blend'][i,q[vi],kp]=True
                        supported+=1
            if gi%30==0:print(role,gi,'/',len(cameras),flush=True)
        det=dict(instance_boxes_cxcywh=data['boxes'],instance_objectness_logits=data['objectness_logits'])
        matches=match_detection(det,data['gt_boxes'],data['box_valid']&data['instance_valid'])
        metric=collections.defaultdict(lambda:dict(errors=[],hit=[],changed=[],changed_base=[],changed_errors=[]))
        for i,m in enumerate(data['metadata']):
            if not cohort[i]:continue
            kind='real' if m['record']['kind']=='real' else 'synthetic';cam='cam'+chr(65+int(m['view']))
            for qq,tt in zip(*matches[i]):
                q,t=int(qq),int(tt)
                detected=bool(data['objectness_logits'][i,q].sigmoid()>=.5 and box_iou(data['boxes'][i,q:q+1],data['gt_boxes'][i,t:t+1])[0,0]>=.5)
                for kp in KP:
                    if not data['point_valid'][i,t,kp]:continue
                    gt=data['gt_xy'][i,t,kp].numpy();pixel=data['pixels'][i].numpy()
                    baseline=float(np.linalg.norm((base[i,q,kp]-gt)*pixel))
                    for arm in ARMS:
                        error=float(np.linalg.norm((pred[arm][i,q,kp]-gt)*pixel));changed=bool(active[arm][i,q,kp])
                        names=[f'{kind}/reliable']+([f'{kind}/{cam}/feet'] if kp in KP[1:] else [])
                        for name in names:
                            a=metric[(arm,name)];a['errors'].append(error);a['hit'].append(detected and error<=32);a['changed'].append(changed)
                            if changed:a['changed_base'].append(baseline);a['changed_errors'].append(error)
                        observations.append(dict(role=role,arm=arm,image_index=data['image_indices'][i],row=i,view=int(m['view']),kind=kind,point=kp+1,query=q,target=t,error=error,baseline_error=baseline,changed=changed))
        role_result=dict(role=role,cohort_images=int(cohort.sum()),scenes=len(cameras),abstentions=dict(failures),metrics={})
        for (arm,name),a in metric.items():
            role_result['metrics'][arm+'/'+name]=dict(points=len(a['errors']),error_mean=float(np.mean(a['errors'])),
                gated_pck32=float(np.mean(a['hit'])),changed_points=int(sum(a['changed'])),
                changed_baseline_mean=float(np.mean(a['changed_base'])) if a['changed_base'] else None,
                changed_error_mean=float(np.mean(a['changed_errors'])) if a['changed_errors'] else None)
        results.append(role_result)
        np.savez_compressed(OUT/f'{role}_predictions.npz',image_indices=data['image_indices'],cohort=cohort,
            **{arm+'_xy':pred[arm] for arm in ARMS},**{arm+'_active':active[arm] for arm in ARMS})
        if role=='validation':valdata=data;valpred=pred;valcohort=cohort
    write_json(OUT/'results.json',dict(results=results,protocol=protocol,script_sha256=fingerprint(Path(__file__))))
    write_json(OUT/'point_errors.json',observations)
    gallery=OUT/'gallery';gallery.mkdir();pages=[]
    scene=next(m['record']['scene'] for i,m in enumerate(valdata['metadata']) if valcohort[i] and m['record']['kind']=='real')
    for i,m in enumerate(valdata['metadata']):
        if m['record']['scene']!=scene:continue
        frame=load_scene(ROOT/scene).frames[int(m['view'])]
        w,h=map(int,valdata['pixels'][i]);original=Image.open(frame.image_path).convert('RGB').resize((w,h))
        canvas=Image.new('RGB',(3*w,h+42),'white');draw=ImageDraw.Draw(canvas)
        for col,arm in enumerate(['baseline','robust_loo','bounded_blend']):
            canvas.paste(original,(col*w,42));draw.text((col*w+10,12),arm,fill='black')
            score=valdata['objectness_logits'][i].sigmoid().numpy()
            colors=['#ff4040','#00ccff','#eecc00','#ee00cc','#00ee99','#ff9900']
            for q in np.flatnonzero(score>=.5):
                for kp in KP:
                    x,y=valpred[arm][i,q,kp]*[w,h];x+=col*w;y+=42
                    draw.ellipse((x-5,y-5,x+5,y+5),outline=colors[q],width=3);draw.text((x+5,y),str(kp+1),fill=colors[q])
            for t in range(2):
                for kp in KP:
                    if valdata['point_valid'][i,t,kp]:
                        x,y=valdata['gt_xy'][i,t,kp].numpy()*[w,h];x+=col*w;y+=42
                        draw.line((x-5,y-5,x+5,y+5),fill='white',width=2);draw.line((x-5,y+5,x+5,y-5),fill='white',width=2)
        name=f'cam{chr(65+int(m["view"]))}.jpg';canvas.save(gallery/name,quality=92);pages.append(name)
    page=['<!doctype html><meta charset="utf-8"><title>预测驱动几何辅助</title><style>body{font:17px system-ui;margin:30px}td,th{padding:8px;border:1px solid #aaa}img{max-width:100%}</style><h1>仅用其他相机预测的几何辅助</h1><p>目标视角不参与三角化。白叉为已有标注；彩色圆点为所有置信查询的预测，颜色不是跨相机身份。固定规则，不搜索验证集阈值。不改标注，不改正式模型。</p><table><tr><th>数据</th><th>方案</th><th>可靠点误差</th><th>底视足点误差</th><th>可靠点修正数</th></tr>']
    for row in results:
        for arm in ARMS:
            a=row['metrics'][arm+'/real/reliable'];b=row['metrics'][arm+'/real/camA/feet']
            page.append(f'<tr><td>{row["role"]}</td><td>{arm}</td><td>{a["error_mean"]:.2f}</td><td>{b["error_mean"]:.2f}</td><td>{a["changed_points"]}</td></tr>')
    page.append('</table><p>模型输入像素；GT框用于预测后评价关联，不是端到端AP。仅单实例实验队列；双鼠不在融合范围。只有两个相机的合成验证数据因缺少独立支持保持原预测，不计为融合成功。背部与尾巴不改。</p>')
    for name in pages:page.append(f'<h2>{name}</h2><img src="gallery/{name}">')
    (OUT/'index.html').write_text('\n'.join(page),encoding='utf-8');shutil.copytree(OUT,DEST/OUT.name)
    print(json.dumps(results),flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise
