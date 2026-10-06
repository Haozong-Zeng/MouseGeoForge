"""Append pre-approved synthetic TRAIN poses; existing full23 prefix immutable."""
import collections
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'vggt-omega'),str(ROOT/'tools')]
import torch
from vggt_omega.models import VGGTOmega
from mousegeoforge.models.surface_curve_pose import SurfaceCurveDepthPoseModel
from training.instance_pose_losses import instance_targets_from_batch,hungarian_match_instances
from run_surface_curve_pilot_20261003 import load_item
from run_depth_pose_ablation import fingerprint,write_json,move_batch
from mousegeoforge.pilot_metrics import box_iou
CYCLE=ROOT/'output/core23_spatial_cycle_20261004'
OLD=ROOT/'output/surface_curve_cycle_20261003'
OUT=CYCLE/'features_expanded_synthetic_v1'


def main():
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'))
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert dt.datetime.now(dt.timezone.utc)<deadline and not OUT.exists()
    OUT.mkdir();(OUT/'parts').mkdir();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    def status(stage,**extra):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**extra))
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='expanding_synthetic_full23_train_features',active_job=OUT.name,active_pid=os.getpid());write_json(CYCLE/'state.json',state);status('loading')
    prefix_path=CYCLE/'features_v1/manifest.json';prefix=json.loads(prefix_path.read_text());rows=list(prefix['records']);seen={};counts=collections.Counter()
    for r in rows:
        assert fingerprint(ROOT/r['path'])==r['sha256']
        part=torch.load(ROOT/r['path'],map_location='cpu',weights_only=False)
        for o in part['observations']:
            m=o['metadata'];rec=m['record'];seen[(rec['scene'],m['view'],m['instance'])]=rec['role'];counts[(rec['kind'],rec['role'],m['instance_count'])]+=1
    prior_source=CYCLE/'instance_priors_v1/manifest.json';prior_meta=json.loads(prior_source.read_text())
    assert fingerprint(ROOT/prior_meta['path'])==prior_meta['sha256']
    priors=torch.load(ROOT/prior_meta['path'],map_location='cpu',weights_only=False)
    boxes=[x.clone() for x in priors['boxes_cxcywh']]
    source=OLD/'expanded_curve_features_v1';oldmeta=json.loads((source/'manifest.json').read_text())
    golden=torch.load(source/'frozen_inputs.pt',map_location='cpu',weights_only=False)
    groups=collections.defaultdict(list)
    for i,m in enumerate(golden['metadata']):groups[m['key']].append((i,m))
    initial=OLD/'training_v1/pilot/initial_heads.pt';encoder=ROOT/'vggt-omega/vggt_omega_1b_512.pt'
    assert fingerprint(initial)==oldmeta['source_initial_sha256'] and fingerprint(encoder)==oldmeta['source_encoder_sha256']
    spatial=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
    raw=torch.load(encoder,map_location='cpu',weights_only=True,mmap=True)
    chosen={k:v for k,v in raw.items() if k.startswith(('aggregator.','dense_head.'))}
    result=spatial.load_state_dict(chosen,strict=False)
    assert not result.unexpected_keys and all(k.startswith('instance_pose_head.') for k in result.missing_keys)
    del raw,chosen
    model=SurfaceCurveDepthPoseModel.from_spatial_model(spatial).cuda().eval().requires_grad_(False);del spatial
    result=model.load_state_dict(torch.load(initial,map_location='cpu',weights_only=True),strict=False)
    assert not result.unexpected_keys and all(k.startswith('aggregator.') for k in result.missing_keys)
    gold_count,maxdelta,added=0,0.,0
    for n,source_record in enumerate(oldmeta['source_records']):
        assert dt.datetime.now(dt.timezone.utc)<deadline
        r=source_record['record'];assert r['role']=='train' and r['kind']=='synthetic'
        assert '/BlenderPyMouse/' in r['scene']
        item=load_item(r);assert item['source_hashes']==source_record['source_hashes']
        b=move_batch(item['batch'],'cuda');memory=[];pose=[]
        a=model.instance_pose_head.instance_decoder.register_forward_pre_hook(lambda m,args:memory.append(args[1].detach()))
        hook=model.instance_pose_head.keypoint_decoder.register_forward_hook(lambda m,args,out:pose.append(out.detach()))
        try:
            with torch.no_grad():pred=model(b['images'],coupling='off')
        finally:a.remove();hook.remove()
        vcount=b['images'].shape[1];h,w=b['images'].shape[-2:]
        maps=memory[0].transpose(1,2).reshape(vcount,256,h//16,w//16).cpu();tokens=pose[0].reshape(vcount,6,23,256).cpu()
        assert torch.isfinite(maps).all() and torch.isfinite(tokens).all()
        for i,m in groups[r['key']]:
            v=r['views'].index(m['view']);q=m['query']
            delta=float((tokens[v,q,[4,5,6,7]].flatten()-golden['tensors']['pose'][i]).abs().max());assert delta<2e-4,(r['key'],delta)
            maxdelta=max(maxdelta,delta);gold_count+=1
        t=instance_targets_from_batch(b);mt=dict(t);mt['keypoint_valid']=torch.zeros_like(t['keypoint_valid']);matches=hungarian_match_instances(pred,mt)[0]
        observations=[]
        for v,(queries,instances) in enumerate(matches):
            for q,m in zip(queries.tolist(),instances.tolist()):
                identity=(r['scene'],r['views'][v],m)
                if identity in seen:
                    assert seen[identity]=='train';continue
                seen[identity]='train';added+=1;counts[(r['kind'],'train',len(instances))]+=1
                bx=pred['instance_boxes_cxcywh'][0,v,q].cpu();boxes.append(bx)
                iou=box_iou(pred['instance_boxes_cxcywh'][0,v,q:q+1],t['boxes_cxcywh'][0,v,m:m+1])[0,0]
                observations.append(dict(metadata=dict(record=r,view=r['views'][v],view_index=v,query=q,instance=m,instance_count=len(instances),box_iou=float(iou),objectness=float(pred['instance_objectness_logits'][0,v,q].sigmoid())),
                    pose=tokens[v,q].clone(),coarse_xy=pred['instance_keypoint_coarse_xy'][0,v,q].cpu(),original_xy=pred['instance_keypoint_xy'][0,v,q].cpu(),
                    gt_xy=t['keypoint_xy'][0,v,m].cpu(),valid=t['keypoint_valid'][0,v,m].cpu(),visible=b['instance_keypoint_visible'][0,v,m].cpu(),visibility_valid=b['instance_keypoint_visibility_valid'][0,v,m].cpu()))
        path=OUT/'parts'/f'{r["key"]}.pt';torch.save(dict(memory=maps.half(),image_hw=[h,w],observations=observations),path)
        rows.append(dict(key=r['key'],path=path.relative_to(ROOT).as_posix(),sha256=fingerprint(path),record=r,count=len(observations),source_hashes=item['source_hashes']))
        status('extracting',completed=n+1,total=len(oldmeta['source_records']),added_observations=added,golden_max_delta=maxdelta)
        print('EXPAND_FULL23',n+1,len(oldmeta['source_records']),'added',added,flush=True)
    assert gold_count==oldmeta['new_train_observations'] and len(boxes)==len(seen)
    # Existing parts/validation are referenced unchanged, not regenerated.
    assert rows[:len(prefix['records'])]==prefix['records']
    for r in prefix['records']:assert fingerprint(ROOT/r['path'])==r['sha256']
    info=dict(records=rows,observations=len(seen),counts={str(k):v for k,v in counts.items()},original_prefix_count=prefix['observations'],
              prefix_manifest_sha256=fingerprint(prefix_path),original_prefix_bitwise_unchanged=True,all_validation_unchanged=True,
              additional_train=added,golden_count=gold_count,max_golden_delta=maxdelta,initial_sha256=fingerprint(initial),encoder_sha256=fingerprint(encoder),
              semantics=prefix['semantics'],real_supported_ids=prefix['real_supported_ids'],no_production_promotion=True)
    write_json(OUT/'manifest.json',info)
    prior_path=OUT/'predicted_boxes.pt';torch.save(dict(boxes_cxcywh=torch.stack(boxes)),prior_path)
    write_json(OUT/'prior_manifest.json',dict(path=prior_path.relative_to(ROOT).as_posix(),sha256=fingerprint(prior_path),source_manifest_sha256=fingerprint(OUT/'manifest.json'),prediction_only=True))
    status('complete',observations=len(seen),additional_train=added,max_golden_delta=maxdelta)
    state.update(stage='expanded_synthetic_full23_ready',active_job=None,active_pid=None,next='Run same seeds1600steps global/spatial controls using --feature-set expanded_synthetic with none and soft_box priors; validation prefix is identical. Compare gains to original small-data controls before adding real TRAIN images or changing loss.')
    state['completed'].append('expanded_synthetic_full23_golden_verified');write_json(CYCLE/'state.json',state)


if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()));raise
