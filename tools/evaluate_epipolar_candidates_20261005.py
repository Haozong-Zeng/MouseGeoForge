"""Fixed-weight spatial candidate control; no labels enter reranking."""
import collections,datetime as dt,json,os,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from PIL import Image,ImageDraw
from mousegeoforge.epipolar_candidate_readout import probability_candidates,rerank_candidates
from mousegeoforge.detection_readout_loss import match_detection
from mousegeoforge.pilot_metrics import box_iou
from training.finetune_blendedmvs import load_scene
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';SOURCE=CYCLE/'singleview_fusion_inputs_seed17_v1';MAPS=CYCLE/'singleview_probability_diagnostic_v1';CAMERAS=CYCLE/'camera_geometry_audit_v2'
OUT=CYCLE/'epipolar_candidates_pilot_v1';DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')
KP=[0,14,15,16,17];ARMS=['baseline','unary_local','epipolar_local']


def evaluate(data,pred,cohort,active):
    det=dict(instance_boxes_cxcywh=data['boxes'],instance_objectness_logits=data['objectness_logits'])
    matches=match_detection(det,data['gt_boxes'],data['box_valid']&data['instance_valid']);metrics=collections.defaultdict(lambda:dict(errors=[],hits=[],active=0));points=[]
    for i in np.flatnonzero(cohort):
        m=data['metadata'][i];kind='real' if m['record']['kind']=='real' else 'synthetic';cam='cam'+chr(65+int(m['view']))
        for qq,tt in zip(*matches[i]):
            q,t=int(qq),int(tt);hit=bool(data['objectness_logits'][i,q].sigmoid()>=.5 and box_iou(data['boxes'][i,q:q+1],data['gt_boxes'][i,t:t+1])[0,0]>=.5)
            for kp in KP:
                if not data['point_valid'][i,t,kp]:continue
                gt=data['gt_xy'][i,t,kp].numpy();pixel=data['pixels'][i].numpy()
                for arm in ARMS:
                    err=float(np.linalg.norm((pred[arm][i,q,kp]-gt)*pixel));changed=bool(active[arm][i,q,kp]);names=[kind+'/reliable']+([kind+'/'+cam+'/feet'] if kp!=0 else [])
                    for name in names:
                        a=metrics[(arm,name)];a['errors'].append(err);a['hits'].append(hit and err<=32);a['active']+=changed
                    points.append(dict(row=int(i),image_index=data['image_indices'][i],query=q,target=t,point=kp+1,view=int(m['view']),kind=kind,arm=arm,error=err,active=changed))
    return {arm+'/'+name:dict(points=len(a['errors']),error_mean=float(np.mean(a['errors'])),gated_pck32=float(np.mean(a['hits'])),active_points=a['active']) for (arm,name),a in metrics.items()},points


def main():
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2)
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'))
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='evaluating_epipolar_candidates',active_job=OUT.name,active_pid=os.getpid());write_json(CYCLE/'state.json',state)
    protocol=dict(candidate_count=5,nms_radius_cells=2,local_average_radius_cells=1,epipolar_sigma_px=16,geometry_weight=1,
        minimum_distinct_support_views=2,score_threshold=.5,control='unary_local: exact same eligibility, only geometry weight0',
        policy='Fixed settings before evaluation, no VAL search. Top2 per-candidate support log-evidence. Different source peaks may support different target candidates; no 3D identity or visibility guarantee.',
        restrictions='Only prediction-selected unique confident queries in knownsingle-instance experimental cohort; labels used only after forward. No GT count, crop, correspondence, visibility. Back/tail unchanged; production untouched.')
    write_json(OUT/'protocol.json',protocol);source=json.loads((SOURCE/'manifest.json').read_text());mm=json.loads((MAPS/'manifest.json').read_text());cm=json.loads((CAMERAS/'results.json').read_text());results=[];allpoints=[];gallery=[]
    for part in source['parts']:
        role=part['role'];assert fingerprint(ROOT/part['path'])==part['sha256'];data=torch.load(ROOT/part['path'],map_location='cpu',weights_only=False)
        mp=next(p for p in mm['parts'] if p['role']==role);assert fingerprint(ROOT/mp['path'])==mp['sha256']
        with np.load(ROOT/mp['path']) as a:prob=a['probability'];local={int(i):j for j,i in enumerate(a['row_indices'])}
        cp=next(p for p in cm['camera_artifacts'] if p['role']==role);assert fingerprint(ROOT/cp['path'])==cp['sha256'];scenes=torch.load(ROOT/cp['path'],map_location='cpu',weights_only=False)
        base=data['xy'].numpy();pred={a:base.copy() for a in ARMS};active={a:np.zeros(base.shape[:-1],bool) for a in ARMS};cohort=np.zeros(len(base),bool);decisions=[]
        for gi,g in enumerate(scenes):
            assert dt.datetime.now(dt.timezone.utc)<deadline
            ids=g['row_indices'];cohort[ids]=True;score=data['objectness_logits'][ids].sigmoid().numpy();valid=(score>=.5).sum(-1)==1;queries=score.argmax(-1)
            candidates=np.empty((len(ids),5,5,2));masses=np.empty((len(ids),5,5));pixels=data['pixels'][ids].numpy()
            for v,i in enumerate(ids):
                for ki in range(5):candidates[v,ki],masses[v,ki]=probability_candidates(prob[local[i],queries[v],ki],pixels[v])
            for v,i in enumerate(ids):
                q=int(queries[v])
                for ki,kp in enumerate(KP):
                    for arm,weight in [('unary_local',0.),('epipolar_local',1.)]:
                        r=rerank_candidates(g['projections'],candidates[:,ki],masses[:,ki],valid,g['physical_views'],v,
                            base[i,q,kp]*pixels[v],enabled=True,geometry_weight=weight)
                        if r['active']:
                            pred[arm][i,q,kp]=r['xy']/pixels[v];active[arm][i,q,kp]=True
                            decisions.append(dict(row=i,query=q,point=kp+1,arm=arm,choice=r['choice'],unary=r['unary'].tolist(),geometry=r['geometry'].tolist()))
            if gi%30==0:print(role,gi,'/',len(scenes),flush=True)
        metrics,points=evaluate(data,pred,cohort,active);results.append(dict(role=role,cohort_images=int(cohort.sum()),metrics=metrics));allpoints+= [dict(p,role=role) for p in points]
        write_json(OUT/f'{role}_decisions.json',decisions);np.savez_compressed(OUT/f'{role}_predictions.npz',image_indices=data['image_indices'],cohort=cohort,**{a+'_xy':pred[a] for a in ARMS},**{a+'_active':active[a] for a in ARMS})
        if role=='validation':
            scene=next(data['metadata'][i]['record']['scene'] for i in np.flatnonzero(cohort) if data['metadata'][i]['record']['kind']=='real')
            for i,m in enumerate(data['metadata']):
                if m['record']['scene']!=scene:continue
                physical=int(m['view']);frame=load_scene(ROOT/scene).frames[physical];w,h=map(int,data['pixels'][i]);rgb=Image.open(frame.image_path).convert('RGB').resize((w,h));canvas=Image.new('RGB',(3*w,h+42),'white');draw=ImageDraw.Draw(canvas)
                for col,arm in enumerate(ARMS):
                    canvas.paste(rgb,(col*w,42));draw.text((col*w+12,12),arm,fill='black');scores=data['objectness_logits'][i].sigmoid().numpy()
                    for q in np.flatnonzero(scores>=.5):
                        color=['#ff5555','#00ccff','#eecc00','#ee00cc','#00ee99','#ff9900'][q]
                        for kp in KP:
                            x,y=pred[arm][i,q,kp]*[w,h]+[col*w,42];draw.ellipse((x-5,y-5,x+5,y+5),outline=color,width=3);draw.text((x+5,y),str(kp+1),fill=color)
                    for t in range(2):
                        for kp in KP:
                            if data['point_valid'][i,t,kp]:
                                x,y=data['gt_xy'][i,t,kp].numpy()*[w,h]+[col*w,42];draw.line((x-5,y-5,x+5,y+5),fill='white',width=2);draw.line((x-5,y+5,x+5,y-5),fill='white',width=2)
                name=f'cam{chr(65+physical)}.jpg';canvas.save(OUT/name,quality=92);gallery.append(name)
    write_json(OUT/'results.json',dict(results=results,protocol=protocol,script_sha256=fingerprint(Path(__file__))));write_json(OUT/'point_errors.json',allpoints)
    page=['<!doctype html><meta charset="utf-8"><title>极线候选选择</title><style>body{font:17px system-ui;margin:30px}img{max-width:100%}td,th{padding:8px;border:1px solid #aaa}</style><h1>局部候选与极线重评分对照</h1><p>baseline为原概率期望；unary_local为局部质量最大的峰；epipolar_local同时使用其他相机的空间候选。后两组支持条件相同。固定权重1，未搜索验证集参数。彩色点为所有可信预测，白叉为事后叠加标注。</p><table><tr><th>数据</th><th>方案</th><th>真实可靠点</th><th>底视足点</th><th>gated PCK32</th></tr>']
    for r in results:
        for a in ARMS:
            x=r['metrics'][a+'/real/reliable'];y=r['metrics'][a+'/real/camA/feet'];page.append(f'<tr><td>{r["role"]}</td><td>{a}</td><td>{x["error_mean"]:.2f}</td><td>{y["error_mean"]:.2f}</td><td>{x["gated_pck32"]:.3f}</td></tr>')
    page.append('</table><p>模型输入像素；GT框仅用于前向后评价关联，不是端到端AP；验证是反复使用的开发集。双鼠未进入本实验；两视角不满足支持条件时原样回退，不计为融合成功。不修改标注和正式模型。</p>')
    for name in gallery:page.append(f'<h2>{name}</h2><img src="{name}">')
    (OUT/'index.html').write_text('\n'.join(page),encoding='utf-8');shutil.copytree(OUT,DEST/OUT.name)
    state.update(stage='epipolar_candidate_pilot_complete',active_job=None,active_pid=None,next='Inspect epipolar_candidates_pilot_v1 results and rawgallery before choosing follow-up; no promotion. Read matched unary control, TRAIN/VAL and camera tradeoffs; do not credit fallbacks.')
    state['completed'].append('fixed_epipolar_candidate_pilot');write_json(CYCLE/'state.json',state);print(json.dumps(results),flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise
