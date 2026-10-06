"""All-query validation: separate box-prior changes, assignment switches and gating."""
import argparse, collections, datetime as dt, fcntl, html, json, os, shutil, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw
from mousegeoforge.models.core23_spatial_readout import Core23SpatialReadout
from mousegeoforge.detection_readout_loss import match_detection
from mousegeoforge.pilot_metrics import box_iou
from run_depth_pose_ablation import fingerprint,write_json
from run_surface_curve_pilot_20261003 import load_item
CYCLE=ROOT/'output/core23_spatial_cycle_20261004';SOURCE=CYCLE/'detection_features_v1';DET=CYCLE/'detection_readout_pilot_v1';OUT=CYCLE/'detection_pose_audit_v1'

def main():
    global DET,OUT
    parser=argparse.ArgumentParser();parser.add_argument('--retention',action='store_true');parser.add_argument('--context',choices=['roi','global']);args=parser.parse_args()
    if args.retention:DET=CYCLE/'detection_readout_retention_v1';OUT=CYCLE/'detection_pose_retention_audit_v1'
    if args.context:DET=CYCLE/f'detection_context_{args.context}_v1';OUT=CYCLE/f'detection_pose_context_{args.context}_audit_v1'
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'))
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='auditing_detection_to_pose',active_job=OUT.name,active_pid=os.getpid());write_json(CYCLE/'state.json',state)
    rows=[];manifest=json.loads((SOURCE/'manifest.json').read_text())
    for r in manifest['records']:
        assert fingerprint(ROOT/r['path'])==r['sha256'];rows.extend(torch.load(ROOT/r['path'],map_location='cpu',weights_only=False)['observations'])
    assert [r['metadata'] for r in rows]==json.loads((DET/'observations.json').read_text())
    ids=[i for i,r in enumerate(rows) if r['metadata']['record']['role']=='validation' and r['metadata']['detection_complete']]
    for i in ids:assert rows[i]['metadata']['record'].get('session') not in ['GR2','UR2','UR3']
    arrays=np.load(DET/'predictions.npz');results=[];saved={};visuals={}
    base_matches={};adapt_matches={}
    for seed in [17,29,43]:
        cp=CYCLE/f'readout_post4800_control_v1/spatial_seed{seed}_latest.pt';model=Core23SpatialReadout().cuda().eval();model.load_state_dict(torch.load(cp,map_location='cuda',weights_only=False)['state'])
        agg=collections.defaultdict(lambda:dict(errors=[],detected_errors=[],total_points=0,detected_points=0,pck16=0,pck32=0,instances=0,detected=0,switches=0));preds=[];details=[]
        for n,i in enumerate(ids):
            assert dt.datetime.now(dt.timezone.utc)<deadline
            o=rows[i];m=o['metadata'];r=m['record'];p=torch.load(ROOT/m['memory_part'],map_location='cpu',weights_only=False)
            memory=F.interpolate(p['memory'][m['view_index']:m['view_index']+1].float(),(48,48),mode='bilinear',align_corners=False).cuda().expand(6,-1,-1,-1)
            base={'instance_boxes_cxcywh':o['original_boxes'][None].cuda(),'instance_objectness_logits':o['original_objectness'][None].cuda()}
            adapted={k:torch.tensor(arrays[f'seed{seed}__{k}'][i:i+1],device='cuda') for k in base}
            boxes=o['boxes'][None].cuda();valid=(o['box_valid']&o['instance_valid'])[None].cuda()
            bm=match_detection(base,boxes,valid)[0];am=match_detection(adapted,boxes,valid)[0];bmap=dict(zip(bm[1].tolist(),bm[0].tolist()));amap=dict(zip(am[1].tolist(),am[0].tolist()))
            with torch.no_grad():
                oldxy=model(o['pose_tokens'].cuda(),o['coarse_xy'].cuda(),memory,predicted_boxes=base['instance_boxes_cxcywh'][0]).cpu()
                newxy=model(o['pose_tokens'].cuda(),o['coarse_xy'].cuda(),memory,predicted_boxes=adapted['instance_boxes_cxcywh'][0]).cpu()
            preds.append(torch.stack([oldxy,newxy]).numpy())
            for arm,xy,det,mapping in [('original',oldxy,base,bmap),('adapted_prior_fixed_query',newxy,adapted,bmap),('adapted_end_to_end',newxy,adapted,amap)]:
                mat=box_iou(det['instance_boxes_cxcywh'][0],boxes[0]).cpu();scores=det['instance_objectness_logits'][0].sigmoid().cpu()
                for instance,q in mapping.items():
                    vv=o['point_valid'][instance].clone()
                    if m['expected_instances']>1:vv[7:14]=False
                    err=((xy[q]-o['gt_xy'][instance])*o['pixels']).norm(dim=-1);hit=bool(scores[q]>=.5 and mat[q,instance]>=.5)
                    subsets={'all':vv,'reliable':vv & torch.tensor([k in [0,14,15,16,17] for k in range(23)]),'back':vv & torch.tensor([k in [5,6] for k in range(23)])}
                    kind='real' if r['kind']=='real' else 'synthetic';instance_type='dual' if m['expected_instances']>1 else 'single'
                    for name,mask in subsets.items():
                        for group in [kind+'/'+name,kind+'/'+instance_type+'/'+name,kind+f'/view{m["view"]}/'+name]:
                            a=agg[arm+'/'+group];e=err[mask].tolist();a['errors']+=e;a['total_points']+=len(e);a['instances']+=1;a['detected']+=int(hit);a['switches']+=int(q!=bmap[instance])
                            if hit:a['detected_errors']+=e;a['detected_points']+=len(e);a['pck16']+=sum(v<=16 for v in e);a['pck32']+=sum(v<=32 for v in e)
                    details.append(dict(image=i,arm=arm,instance=instance,query=q,baseline_query=bmap[instance],detected=hit,score=float(scores[q]),iou=float(mat[q,instance])))
            if seed==17:visuals[i]=dict(oldxy=oldxy,newxy=newxy,base=base,adapted=adapted)
            if n%20==0:write_json(OUT/'status.json',dict(stage='auditing',seed=seed,completed=n,total=len(ids),pid=os.getpid()))
        summary={}
        for name,a in agg.items():
            for field in ['errors','detected_errors']:
                v=a.pop(field);a[field+'_mean']=float(np.mean(v)) if v else None;a[field+'_median']=float(np.median(v)) if v else None
            a['detection_gated_pck16']=a['pck16']/max(a['total_points'],1);a['detection_gated_pck32']=a['pck32']/max(a['total_points'],1);summary[name]=a
        results.append(dict(seed=seed,checkpoint_sha256=fingerprint(cp),metrics=summary));write_json(OUT/f'seed{seed}_details.json',details);saved[f'seed{seed}']=np.stack(preds)
    np.savez_compressed(OUT/'predictions.npz',image_indices=np.array(ids),**saved)
    write_json(OUT/'results.json',dict(results=results,protocol='Frozen spatial readout at6400. ALL6 queries use predicted boxes only. Original versus adapted prior with original assignment versus adapted assignment. GT used after forward for scoring only. PCK denominators include missed detections. Pixel errors at training resolution, not original video. No NMS; confidence and IoU both .5. Development not external heldout.'))
    chosen=json.loads((CYCLE/'review_box_contract_v1/manifest.json').read_text());selected=[x['observation'] for x in chosen if x['observation'] in visuals]
    # Real views use integer A=0,... IDs. Show first example per camera,
    # without inferring physical camera orientation or mining largest errors.
    seen=set()
    for i in ids:
        m=rows[i]['metadata']
        if m['record']['kind']=='real' and m['view'] not in seen:
            seen.add(m['view'])
            if i not in selected:selected.append(i)
    page=['<!doctype html><meta charset="utf-8"><title>Detection and pose audit</title><style>body{background:#20232b;color:white;font:16px sans-serif}img{max-width:100%}</style><h1>Frozen spatial pose with adapted predicted boxes</h1><p>Left original detector, right adapted detector seed17. Every query with score >=.5 shown; no GT-based query filtering, no NMS. Green rectangles=GT; colored rectangles/points=predictions (IDs1,15-18); yellow crosses=observed labels. This is development data, not independent test.</p>']
    for n,i in enumerate(selected):
        o=rows[i];m=o['metadata'];r=m['record'];cache=ROOT/f'output/surface_curve_cycle_20261003/training_v1/pilot/cache/{r["key"]}.pt';item=torch.load(cache,map_location='cpu',weights_only=False) if cache.exists() else load_item(r)
        pix=(item['batch']['images'][0,m['view_index']].permute(1,2,0).numpy()*255).clip(0,255).astype('uint8');h,w=pix.shape[:2];pw=480;ph=round(h*pw/w);im=Image.fromarray(pix).resize((pw,ph));canvas=Image.new('RGB',(pw*2,ph+30),(22,24,30));draw=ImageDraw.Draw(canvas);v=visuals[i]
        for col,(arm,xy) in enumerate([('base',v['oldxy']),('adapted',v['newxy'])]):
            canvas.paste(im,(col*pw,30));draw.text((col*pw+5,5),arm,fill='white');det=v[arm]
            def rect(box,color):
                cx,cy,bw,bh=box;draw.rectangle(((cx-bw/2)*pw+col*pw,(cy-bh/2)*ph+30,(cx+bw/2)*pw+col*pw,(cy+bh/2)*ph+30),outline=color,width=2)
            for j in torch.where(o['box_valid'])[0]:rect(o['boxes'][j].tolist(),'#66ff88')
            for q in torch.where(det['instance_objectness_logits'][0].sigmoid()>=.5)[0].tolist():
                color=['#ff77ff','#44ddff','#ff8844','#aa88ff','#ffffff','#ff5555'][q];rect(det['instance_boxes_cxcywh'][0,q].cpu().tolist(),color)
                for k in [0,14,15,16,17]:
                    x,y=xy[q,k].numpy()*[pw,ph]+[col*pw,30];draw.ellipse((x-3,y-3,x+3,y+3),outline=color,width=2);draw.text((x+3,y),str(k+1),fill=color)
            for j in torch.where(o['instance_valid'])[0]:
                for k in [0,14,15,16,17]:
                    if o['point_valid'][j,k]:
                        x,y=o['gt_xy'][j,k].numpy()*[pw,ph]+[col*pw,30];draw.line((x-4,y,x+4,y),fill='#ffee55',width=2);draw.line((x,y-4,x,y+4),fill='#ffee55',width=2)
        name=f'{n:02d}_{r["kind"]}.jpg';canvas.save(OUT/name,quality=95);page.append('<h2>'+html.escape(str(m))+'</h2><img src="'+name+'">')
    (OUT/'index.html').write_text('\n'.join(page),encoding='utf-8');write_json(OUT/'status.json',dict(stage='complete',pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat()))
    dest=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-Core23-Spatial-20261004')/OUT.name;assert not dest.exists();shutil.copytree(OUT,dest)
    state.update(stage='detection_pose_audit_ready',active_job=None,active_pid=None,updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),next=f'Visually inspect {OUT.name} and compare downstream errors/gated coverage/query switches against pilot and original. No production promotion. Decide from retention tradeoff evidence, not blind step extension.');state['completed'].append(OUT.name+'_validation');write_json(CYCLE/'state.json',state)

if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()));raise
