"""Append eligible real TRAIN groups with full23 tokens, no new labels/validation."""
import collections,datetime as dt,fcntl,json,os,sys
from pathlib import Path
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
OUT=CYCLE/'features_expanded_all_v1'


def main():
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'))
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert dt.datetime.now(dt.timezone.utc)<deadline and not OUT.exists()
    OUT.mkdir();(OUT/'parts').mkdir();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='expanding_real_full23_train_features',active_job=OUT.name,active_pid=os.getpid());write_json(CYCLE/'state.json',state);status('loading')
    prefix_path=CYCLE/'features_expanded_synthetic_v1/manifest.json';prefix=json.loads(prefix_path.read_text());rows=list(prefix['records']);counts=collections.Counter();seen=set()
    for r in rows:
        assert fingerprint(ROOT/r['path'])==r['sha256']
        p=torch.load(ROOT/r['path'],map_location='cpu',weights_only=False)
        for o in p['observations']:
            m=o['metadata'];rec=m['record'];counts[(rec['kind'],rec['role'],m['instance_count'])]+=1
            if rec['kind']=='real':seen.add((rec['session'],rec['frame'],m['view']))
    pm=json.loads((CYCLE/'features_expanded_synthetic_v1/prior_manifest.json').read_text());assert fingerprint(ROOT/pm['path'])==pm['sha256']
    boxes=[x.clone() for x in torch.load(ROOT/pm['path'],map_location='cpu',weights_only=False)['boxes_cxcywh']]
    source=OLD/'expanded_real_point_features_v1';records=json.loads((source/'records.json').read_text())
    original_memory=json.loads((source/'memory_manifest.json').read_text());hashes={r['key']:r['source_hashes'] for r in original_memory['records']}
    golden=torch.load(source/'real_inputs.pt',map_location='cpu',weights_only=False);groups=collections.defaultdict(list)
    for i,m in enumerate(golden['metadata']):groups[m['cache_key']].append((i,m))
    eligibility=json.loads((ROOT/'dataset/PyMouseStudio/annotation_versions/20261003_effective_v1/manifest.json').read_text())
    eligible={(g['session'],g['frame']) for g in eligibility['frame_groups'] if g['eligible_for_development'] and g['role']=='train' and not g['excluded']}
    assert len(records)==471 and all(r['role']=='train' and (r['session'],r['frame']) in eligible and r['session'] not in ('GR2','UR2','UR3','FR2','AZR2L2') for r in records)
    initial=OLD/'training_v1/pilot/initial_heads.pt';encoder=ROOT/'vggt-omega/vggt_omega_1b_512.pt'
    assert fingerprint(initial)==prefix['initial_sha256'] and fingerprint(encoder)==prefix['encoder_sha256']
    spatial=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
    raw=torch.load(encoder,map_location='cpu',weights_only=True,mmap=True);chosen={k:v for k,v in raw.items() if k.startswith(('aggregator.','dense_head.'))}
    result=spatial.load_state_dict(chosen,strict=False);assert not result.unexpected_keys and all(k.startswith('instance_pose_head.') for k in result.missing_keys);del raw,chosen
    model=SurfaceCurveDepthPoseModel.from_spatial_model(spatial).cuda().eval().requires_grad_(False);del spatial
    result=model.load_state_dict(torch.load(initial,map_location='cpu',weights_only=True),strict=False);assert not result.unexpected_keys and all(k.startswith('aggregator.') for k in result.missing_keys)
    gold_count,maxdelta,added=0,0.,0
    for n,r in enumerate(records):
        assert dt.datetime.now(dt.timezone.utc)<deadline
        item=load_item(r);assert item['source_hashes']==hashes[r['key']]
        b=move_batch(item['batch'],'cuda');torch.backends.cudnn.allow_tf32=False
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):features,start=model.aggregator(b['images'])
        cached=[x if j in (4,11,17,23) else None for j,x in enumerate(features)];del features
        torch.backends.cudnn.allow_tf32=True;memory=[];pose=[]
        a=model.instance_pose_head.instance_decoder.register_forward_pre_hook(lambda m,args:memory.append(args[1].detach()))
        hook=model.instance_pose_head.keypoint_decoder.register_forward_hook(lambda m,args,out:pose.append(out.detach()))
        try:
            with torch.no_grad():pred=model.forward_cached(b['images'],cached,start,coupling='off')
        finally:a.remove();hook.remove()
        del cached
        vcount=b['images'].shape[1];h,w=b['images'].shape[-2:]
        maps=memory[0].transpose(1,2).reshape(vcount,256,h//16,w//16).cpu();tokens=pose[0].reshape(vcount,6,23,256).cpu()
        assert torch.isfinite(maps).all() and torch.isfinite(tokens).all()
        for i,m in groups[r['key']]:
            v=m['cache_view_index'];q=m['query'];assert r['views'][v]==m['view']
            delta=float((tokens[v,q,[4,5,6,7]].flatten()-golden['tensors']['pose'][i]).abs().max());assert delta<2e-4,(r['key'],delta)
            maxdelta=max(maxdelta,delta);gold_count+=1
        t=instance_targets_from_batch(b);mt=dict(t);mt['keypoint_valid']=torch.zeros_like(t['keypoint_valid']);matches=hungarian_match_instances(pred,mt)[0];observations=[]
        for v,(queries,instances) in enumerate(matches):
            assert len(instances)==1 and instances.tolist()==[0]
            identity=(r['session'],r['frame'],r['views'][v])
            if identity in seen:continue
            seen.add(identity);q=int(queries[0]);m=0;added+=1;counts[('real','train',1)]+=1
            valid=t['keypoint_valid'][0,v,m].cpu();allowed=torch.zeros(23,dtype=torch.bool);allowed[[0,5,6,14,15,16,17]]=True
            assert not (valid&~allowed).any()
            boxes.append(pred['instance_boxes_cxcywh'][0,v,q].cpu())
            iou=box_iou(pred['instance_boxes_cxcywh'][0,v,q:q+1],t['boxes_cxcywh'][0,v,m:m+1])[0,0]
            observations.append(dict(metadata=dict(record=r,view=r['views'][v],view_index=v,query=q,instance=m,instance_count=1,box_iou=float(iou),objectness=float(pred['instance_objectness_logits'][0,v,q].sigmoid())),
                pose=tokens[v,q].clone(),coarse_xy=pred['instance_keypoint_coarse_xy'][0,v,q].cpu(),original_xy=pred['instance_keypoint_xy'][0,v,q].cpu(),
                gt_xy=t['keypoint_xy'][0,v,m].cpu(),valid=valid,visible=b['instance_keypoint_visible'][0,v,m].cpu(),visibility_valid=b['instance_keypoint_visibility_valid'][0,v,m].cpu()))
        path=OUT/'parts'/f'{r["key"]}.pt';torch.save(dict(memory=maps.half(),image_hw=[h,w],observations=observations),path)
        rows.append(dict(key=r['key'],path=path.relative_to(ROOT).as_posix(),sha256=fingerprint(path),record=r,count=len(observations),source_hashes=item['source_hashes']))
        status('extracting',completed=n+1,total=len(records),added_observations=added,golden_max_delta=maxdelta)
        print('EXPAND_REAL_FULL23',n+1,len(records),'added',added,flush=True)
    assert gold_count==855 and added==855 and len(boxes)==prefix['observations']+added
    assert rows[:len(prefix['records'])]==prefix['records']
    for r in prefix['records']:assert fingerprint(ROOT/r['path'])==r['sha256']
    write_json(OUT/'manifest.json',dict(records=rows,observations=len(boxes),counts={str(k):v for k,v in counts.items()},
        original_prefix_count=prefix['observations'],prefix_manifest_sha256=fingerprint(prefix_path),original_prefix_bitwise_unchanged=True,all_validation_unchanged=True,
        additional_train=added,golden_count=gold_count,max_golden_delta=maxdelta,initial_sha256=fingerprint(initial),encoder_sha256=fingerprint(encoder),
        semantics=prefix['semantics'],real_supported_ids=prefix['real_supported_ids'],no_production_promotion=True))
    prior_path=OUT/'predicted_boxes.pt';torch.save(dict(boxes_cxcywh=torch.stack(boxes)),prior_path)
    write_json(OUT/'prior_manifest.json',dict(path=prior_path.relative_to(ROOT).as_posix(),sha256=fingerprint(prior_path),source_manifest_sha256=fingerprint(OUT/'manifest.json'),prediction_only=True))
    status('complete',observations=len(boxes),additional_train=added,max_golden_delta=maxdelta)
    state.update(stage='expanded_all_full23_ready',active_job=None,active_pid=None,next='Run same global/spatial seeds1600steps controls --feature-set expanded_all --instance-prior soft_box first, and ungated as control if useful. Compare against expanded_synthetic with identical validation. Real TRAIN188 to1043; retain unknown/ambiguous masks. Inspect bottom feet and wrong-instance failure cases before changing architecture/loss.')
    state['completed'].append('expanded_real_full23_golden_verified');write_json(CYCLE/'state.json',state)


if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()));raise
