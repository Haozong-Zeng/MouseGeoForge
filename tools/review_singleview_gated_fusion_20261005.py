"""Fixed raw overlays plus physical-camera view-count stress for fusion pilot."""
import collections,html,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from PIL import Image,ImageDraw
from mousegeoforge.models.confidence_gated_view_residual import ConfidenceGatedViewResidual
from mousegeoforge.detection_readout_loss import match_detection
from mousegeoforge.pilot_metrics import box_iou
from run_surface_curve_pilot_20261003 import load_item
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';OLD=ROOT/'output/core23_spatial_cycle_20261004';BASE=CYCLE/'gated_fusion_matched_pilot_v1';SOURCE=CYCLE/'singleview_fusion_inputs_seed17_v1';OUT=CYCLE/'review_gated_fusion_pilot_v1';KP=[0,14,15,16,17]

@torch.no_grad()
def main():
    assert (BASE/'results.json').exists() and not OUT.exists();OUT.mkdir();torch.set_num_threads(2)
    part=next(p for p in json.loads((SOURCE/'manifest.json').read_text())['parts'] if p['role']=='validation');assert fingerprint(ROOT/part['path'])==part['sha256'];data=torch.load(ROOT/part['path'],map_location='cpu',weights_only=False)
    with np.load(BASE/'predictions.npz') as a:gold={arm:a[arm+'_xy'].copy() for arm in ['self','multiview']};in_cohort=a['in_cohort'].copy();assert data['image_indices']==a['image_indices'].tolist()
    groups=next(g['scenes'] for g in json.loads((SOURCE/'scene_groups.json').read_text()) if g['role']=='validation');groups=[g for g in groups if all(in_cohort[i] for i in g['row_indices'])]
    quality=torch.exp(-data['variance'].sum(-1).clamp_min(0).sqrt()/.05).clamp_min(1e-6)
    baseline=data['xy'].numpy();variants=['single','paired','full','full_reverse','full_duplicate','drop_one'];predictions={};coverage={};max_replay=0.
    for arm in ['self','multiview']:
        model=ConfidenceGatedViewResidual(mode=arm).eval();cp=torch.load(BASE/f'{arm}_latest.pt',map_location='cpu',weights_only=False);assert cp['step']==800;model.load_state_dict(cp['state'],strict=True)
        for variant in variants:
            xy=baseline.copy();active=np.zeros((len(xy),6,len(KP)),dtype=bool)
            for g in groups:
                rows=g['row_indices'];views=g['physical_views']
                for target in rows:
                    target_view=views[rows.index(target)];others=[i for i in rows if i!=target]
                    chosen=[target] if variant=='single' else [target,others[0]] if variant=='paired' else rows.copy()
                    if variant=='full_reverse':chosen=list(reversed(chosen))
                    if variant=='full_duplicate':chosen=chosen+[others[0]]
                    if variant=='drop_one':chosen=[i for i in rows if i!=others[0]]
                    ids=[views[rows.index(i)] for i in chosen];args=dict(xy=data['xy'][chosen][:,:,KP][None],tokens=data['tokens'][chosen][:,:,KP][None],objectness_logits=data['objectness_logits'][chosen][None],quality=quality[chosen][:,:,KP][None],view_ids=torch.tensor(ids)[None],view_valid=torch.ones(1,len(chosen),dtype=torch.bool));p=model(**args,enabled=True);j=chosen.index(target);xy[target,:,KP]=p['xy'][0,j].numpy().transpose(1,0,2);active[target]=p['active'][0,j].numpy()
            if variant=='single':np.testing.assert_array_equal(xy,baseline)
            if variant=='full':
                delta=float(np.max(np.abs(xy-gold[arm])));assert delta<2e-5;max_replay=max(max_replay,delta)
            predictions[(arm,variant)]=xy;coverage[(arm,variant)]=active
        for variant in ['full_reverse','full_duplicate']:
            np.testing.assert_allclose(predictions[(arm,variant)],predictions[(arm,'full')],atol=2e-6,rtol=0)
    matches=match_detection(dict(instance_boxes_cxcywh=data['boxes'],instance_objectness_logits=data['objectness_logits']),data['gt_boxes'],data['box_valid']&data['instance_valid']);summary=[]
    for (arm,variant),xy in predictions.items():
        accum=collections.defaultdict(lambda:dict(error=[],points=0,pck32=0,active_points=0))
        for i,m in enumerate(data['metadata']):
            if not in_cohort[i]:continue
            kind='real' if m['record']['kind']=='real' else 'synthetic';cam='cam'+chr(65+int(m['view']))
            for q,t in zip(*matches[i]):
                q,t=int(q),int(t);valid=data['point_valid'][i,t,KP].numpy();err=np.linalg.norm((xy[i,q,KP]-data['gt_xy'][i,t,KP].numpy())*data['pixels'][i].numpy(),axis=-1);hit=bool(data['objectness_logits'][i,q].sigmoid()>=.5 and box_iou(data['boxes'][i,q:q+1],data['gt_boxes'][i,t:t+1])[0,0]>=.5)
                for name,mask in [(kind,valid),(kind+'/'+cam+'/feet',valid&np.array([False,True,True,True,True]))]:
                    a=accum[name];a['error']+=err[mask].tolist();a['points']+=int(mask.sum());a['active_points']+=int(coverage[(arm,variant)][i,q,mask].sum());a['pck32']+=int((err[mask]<=32).sum())*hit
        metrics={}
        for name,a in accum.items():
            values=a.pop('error');a.update(error_mean=float(np.mean(values)) if values else None,gated_pck32=a['pck32']/max(a['points'],1),active_fraction=a['active_points']/max(a['points'],1));metrics[name]=a
        summary.append(dict(arm=arm,variant=variant,metrics=metrics))
    fixed=list(dict.fromkeys(r['image_index'] for r in json.loads((OLD/'review_view_context_control_v1/manifest.json').read_text())));lookup={i:j for j,i in enumerate(data['image_indices'])};colors=['#ff6a6a','#66ff88','#55ccff','#ffff55','#ee77ff','#ffa544'];gallery=[];pages=['<!doctype html><meta charset="utf-8"><title>融合残差首轮</title><style>body{font:17px sans-serif;background:#20232b;color:white}img{max-width:100%}</style><h1>仅自身特征 vs 多视角语义特征</h1><p>左：固定独立V1基线；中：self残差；右：multiview残差。均为固定seed17、800步，不代表正式模型。仅鼻尖/四足改动，其他18点和检测输出不变。全部confidence≥.5查询绘出，无GT数量过滤/NMS。双鼠不属于此融合实验，不用不变的双鼠结果冒充成功。</p>']
    for image_index in fixed:
        j=lookup[image_index]
        if not in_cohort[j]:continue
        m=data['metadata'][j];record=m['record'];rgb=load_item(dict(record,views=[m['view']]))['batch']['images'][0,0];pw=460;ph=round(rgb.shape[-2]*pw/rgb.shape[-1]);im=Image.fromarray((rgb.permute(1,2,0).numpy()*255).clip(0,255).astype('uint8')).resize((pw,ph));canvas=Image.new('RGB',(pw*3,ph+38),(22,24,30));draw=ImageDraw.Draw(canvas);scores=data['objectness_logits'][j].sigmoid().numpy()
        for col,arm in enumerate(['baseline','self','multiview']):
            canvas.paste(im,(col*pw,38));draw.text((col*pw+5,6),arm+' / full available V1 contexts',fill='white');xy=baseline if arm=='baseline' else predictions[(arm,'full')]
            for q in np.flatnonzero(scores>=.5):
                cx,cy,bw,bh=data['boxes'][j,q].numpy();lo=np.array([cx-bw/2,cy-bh/2])*[pw,ph]+[col*pw,38];hi=np.array([cx+bw/2,cy+bh/2])*[pw,ph]+[col*pw,38];draw.rectangle(tuple(lo)+tuple(hi),outline=colors[q],width=2);draw.text((lo[0],lo[1]-11),f'q{q}:{scores[q]:.2f}',fill=colors[q])
                for k in KP:
                    x,y=xy[j,q,k]*[pw,ph]+[col*pw,38];draw.ellipse((x-2,y-2,x+2,y+2),fill=colors[q]);draw.text((x+3,y),str(k+1),fill=colors[q])
            for target in range(2):
                for k in KP:
                    if not data['point_valid'][j,target,k]:continue
                    x,y=data['gt_xy'][j,target,k].numpy()*[pw,ph]+[col*pw,38];draw.line((x-4,y,x+4,y),fill='#fff06a',width=2);draw.line((x,y-4,x,y+4),fill='#fff06a',width=2)
        name=f'{len(gallery):02d}_{record["kind"]}_cam{chr(65+int(m["view"]))}.jpg';canvas.save(OUT/name,quality=95);gallery.append(dict(file=name,image_index=image_index,metadata=m));pages.append('<h2>'+html.escape(record['scene'])+'</h2><img src="'+name+'">')
    write_json(OUT/'results.json',dict(summary=summary,max_cpu_replay_delta=max_replay,physical_camera_permutation_and_duplicate_checked=True,single_exact=True,other18_points_and_dual_unchanged=True,dual_outside_experiment=True,visual_review_pending=True,RGB_integration_not_tested=True));write_json(OUT/'manifest.json',gallery);(OUT/'index.html').write_text('\n'.join(pages),encoding='utf-8');dest=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')/OUT.name;assert not dest.exists();shutil.copytree(OUT,dest);print(json.dumps(dict(summary=summary,max_replay=max_replay),indent=2))

if __name__=='__main__':main()
