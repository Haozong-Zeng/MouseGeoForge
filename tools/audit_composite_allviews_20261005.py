"""Predetermined all-camera RGB audit; labels only used after prediction."""
import datetime as dt
import fcntl
import json
import os
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from PIL import Image
from singleview_candidate_runtime import FrozenSingleViewCandidateRuntime
from audit_composite_candidate_cache_20261005 import load_wrapper
from audit_independent_unary_20261005 import metrics
from mousegeoforge.candidate_inputs import local_candidate_inputs
from mousegeoforge.multiview_camera_geometry import projection_matrix,project_points
from mousegeoforge.detection_readout_loss import match_detection
from mousegeoforge.pilot_metrics import box_iou
from run_surface_curve_pilot_20261003 import load_item,load_scene
from training.instance_pose_losses import instance_targets_from_batch
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005'
OUT=CYCLE/'composite_allview_association_audit_v1'
KP=[0,14,15,16,17]


@torch.no_grad()
def main(requested_counts=None,source_role='validation'):
    plan=json.loads((CYCLE/'plan.json').read_text())
    cutoff=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'))
    assert dt.datetime.now(dt.timezone.utc)<cutoff
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists();OUT.mkdir();(OUT/'rgb').mkdir()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    state=json.loads((CYCLE/'state.json').read_text())
    state.update(stage='composite_allviews_running',active_job=OUT.name,active_pid=os.getpid())
    write_json(CYCLE/'state.json',state)
    def status(stage,**kw):
        write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    status('loading');started=time.monotonic()
    assert source_role in ['train','validation']
    source=CYCLE/f'singleview_fusion_inputs_seed17_v1/{source_role}.pt'
    d=torch.load(source,map_location='cpu',weights_only=False)
    desired={1:1,2:2} if requested_counts is None else requested_counts
    selected=[];seen=set();counts={n:0 for n in desired}
    for m in d['metadata']:
        n=m['expected_instances'];r=m['record'];scene=r['scene']
        if r['kind']=='real' or scene in seen or n not in counts:continue
        if counts[n]>=desired[n]:continue
        counts[n]+=1;seen.add(scene)
        selected.append(dict(r,views=sorted(load_scene(ROOT/scene).frames)))
    assert counts==desired,(counts,desired)
    write_json(OUT/'protocol.json',dict(records=selected,source_sha256=fingerprint(source),
        selection=f'First {desired} distinct synthetic {source_role} scenes by instance-count cohort in fixed source order, all existing cameras. No error selection or training.',
        inference='Each RGB independently encoded, all6queries. Shared calibration from full scene batch; no GTboxes/count/identity enter forward. GT only posthoc projection/matching/metrics. Query index is not camera identity.',
        limits='Reused development synthetic scenes, not external video. Natural missed detections only; no fake identity oracle. Subset tests reuse fresh independent outputs.'))
    runtime=FrozenSingleViewCandidateRuntime();wrapper=load_wrapper();results=[];saved=[];calls=0
    lookup={(m['record']['scene'],int(m['view'])):i for i,m in enumerate(d['metadata'])}
    for gi,r in enumerate(selected):
        assert dt.datetime.now(dt.timezone.utc)<cutoff
        assert r['role']==source_role and r.get('session') not in ['GR2','UR2','UR3']
        item=load_item(r);b=item['batch'];views=r['views'];v=len(views);h,w=r['image_hw'];pixels=np.array([w,h],np.float32)
        projections=np.stack([projection_matrix(k.numpy(),e.numpy()) for k,e in zip(b['intrinsics'][0],b['extrinsics'][0])])
        predictions=[];images=[];parity=[];timings=[]
        for vi,view in enumerate(views):
            assert dt.datetime.now(dt.timezone.utc)<cutoff
            rgb=b['images'][:,vi:vi+1]
            torch.cuda.synchronize();t0=time.monotonic();raw=runtime.from_images(rgb);torch.cuda.synchronize();timings.append(time.monotonic()-t0);calls+=1
            aux=local_candidate_inputs(raw['probability'].cpu().numpy(),raw['xy'][:,KP].cpu().numpy(),raw['maps'])
            p={k:x.cpu() for k,x in raw.items() if k!='maps'};p.update(aux);predictions.append(p)
            old=lookup.get((r['scene'],view))
            if old is not None:
                delta={k:float((p[k]-d[k][old]).abs().max()) for k in ['xy','boxes','objectness_logits']}
                assert max(delta.values())<2e-5,delta
                parity.append(dict(view=view,row=old,delta=delta))
            name=f'rgb/group{gi:02d}_view{view}.png'
            Image.fromarray((rgb[0,0].permute(1,2,0).numpy().clip(0,1)*255).round().astype(np.uint8)).save(OUT/name);images.append(name)
            del raw
            status('encoding',group=gi,view=view,encoder_calls=calls)
        p={k:torch.stack([x[k] for x in predictions]) for k in predictions[0]}
        modes=dict(baseline=(False,False),local=(True,False),fusion_only=(False,True),composite=(True,True))
        def call(ids,mode='composite',**kw):
            local,fusion=modes[mode]
            return wrapper({k:x[ids] for k,x in p.items()},projections[ids],[views[i] for i in ids],pixels,local_enabled=local,fusion_enabled=fusion,**kw)
        outputs={name:call(list(range(v)),name) for name in modes}
        tests={}
        for name,ids in [('reverse',list(reversed(range(v)))),('duplicate',list(range(v))*2)]:
            xy=call(ids)['xy'];xy=xy.flip(0) if name=='reverse' else xy[:v]
            tests[name]=float((xy-outputs['composite']['xy']).abs().max());assert tests[name]==0
        for count in [1,2]:
            diffs=[]
            for target in range(v):
                ids=[target]+[i for i in range(v) if i!=target][:count-1]
                diffs.append(float((call(ids)['xy'][0]-outputs['local']['xy'][target]).abs().max()))
            tests[f'V{count}_local_fallback']=max(diffs);assert max(diffs)==0
        tests['all_invalid']=float((call(list(range(v)),view_valid=[False]*v)['xy']-p['xy']).abs().max());assert tests['all_invalid']==0
        # Forward is complete. Now inspect annotations, calibration and assignment.
        t=instance_targets_from_batch(b);projection_errors=[]
        for vi in range(v):
            for target in range(2):
                xyz=b['instance_keypoint_world_xyz'][0,target].numpy()
                uv,z=project_points(projections[vi],xyz)
                valid=t['keypoint_valid'][0,vi,target].numpy()&b['instance_world_valid'][0,target].numpy()&(z>0)
                error=np.linalg.norm(uv-t['keypoint_xy'][0,vi,target].numpy()*pixels,axis=-1)
                projection_errors.extend(error[valid].tolist())
        assert projection_errors and max(projection_errors)<.002,max(projection_errors)
        kind='synthetic_dual' if int(b['actual_instance_count'][0])==2 else 'synthetic_single'
        data=dict(base_xy=p['xy'],boxes=p['boxes'],logits=p['objectness_logits'],gt_xy=t['keypoint_xy'][0],point_valid=t['keypoint_valid'][0],
                  gt_boxes=t['boxes_cxcywh'][0],box_valid=t['box_valid'][0],instance_valid=t['instance_valid'][0],pixels=torch.tensor(pixels).repeat(v,1),
                  metadata=[dict(scene=r['scene'],view=view,kind=kind) for view in views])
        evaluations={}
        for mode,out in outputs.items():
            m,points=metrics(data,out['xy'],out['changed'],torch.zeros_like(out['changed'],dtype=torch.long))
            evaluations[mode]=m;write_json(OUT/f'group{gi:02d}_{mode}_points.json',points)
        matches=match_detection(dict(instance_boxes_cxcywh=p['boxes'],instance_objectness_logits=p['objectness_logits']),data['gt_boxes'],data['box_valid']&data['instance_valid'])
        detections=[]
        for vi,view in enumerate(views):
            assoc={int(q):int(tgt) for q,tgt in zip(*matches[vi])}
            confident=torch.where(p['objectness_logits'][vi].sigmoid()>=.5)[0].tolist()
            det=[]
            for q in confident:
                target=assoc.get(q)
                iou=None if target is None else float(box_iou(p['boxes'][vi,q:q+1],data['gt_boxes'][vi,target:target+1])[0,0])
                det.append(dict(query=q,score=float(p['objectness_logits'][vi,q].sigmoid()),evaluation_target=target,iou=iou))
            detections.append(dict(view=view,count=len(confident),predictions=det,
                local_eligible=int(outputs['composite']['local_eligible'][vi].sum()),fusion_eligible=int(outputs['composite']['fusion_eligible'][vi].sum())))
        result=dict(scene=r['scene'],kind=kind,views=views,detections=detections,metrics=evaluations,tests=tests,old_view_parity=parity,
            max_projection_px=max(projection_errors),encoder_seconds=timings,images=images,source_hashes=item['source_hashes'],
            calibration_scale=float(b['scale'][0]),instance_ids=t['instance_ids'][0].tolist())
        results.append(result);saved.append(dict(record=r,predictions=p,outputs=outputs,projections=projections,pixels=pixels,images=images,evaluation_data=data))
        print(kind,views,[x['count'] for x in detections],{mode:round(m[kind+'/reliable']['mean_error'],3) for mode,m in evaluations.items()},flush=True)
    torch.save(saved,OUT/'groups.pt')
    write_json(OUT/'results.json',dict(groups=results,encoder_calls=calls,wall_seconds=time.monotonic()-started,runtime_sha256=runtime.hashes,
        script_sha256=fingerprint(Path(__file__)),module_sha256=fingerprint(ROOT/'mousegeoforge/models/bounded_candidate_fusion.py'),
        head_sha256={a:fingerprint(CYCLE/f'bounded_prior_candidate_pilot_v1/{a}_latest.pt') for a in ['unary','geometry']},production_changed=False))
    status('complete',encoder_calls=calls)
    state.update(stage='composite_allviews_complete',active_job=None,active_pid=None,next='Review allview association results and visualize allqueries; then record next action, do not promote.')
    state['completed'].append('composite_allview_rgb_audit');write_json(CYCLE/'state.json',state)


if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise
