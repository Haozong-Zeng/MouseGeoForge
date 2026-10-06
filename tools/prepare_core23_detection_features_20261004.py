"""Exact all-query tokens and mask-backed boxes; no GT-selected feature queries."""
import collections,datetime as dt,fcntl,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vggt-omega'),str(ROOT/'tools')]
import numpy as np
import torch
from vggt_omega.models import VGGTOmega
from mousegeoforge.models.surface_curve_pose import SurfaceCurveDepthPoseModel
from mousegeoforge.models.instance_detection_readout import InstanceDetectionReadout
from training.instance_pose_losses import instance_targets_from_batch
from training.finetune_blendedmvs import _normalized_box_from_mask
from run_surface_curve_pilot_20261003 import load_item
from run_depth_pose_ablation import fingerprint,write_json,move_batch
CYCLE=ROOT/'output/core23_spatial_cycle_20261004';OLD=ROOT/'output/surface_curve_cycle_20261003'
SOURCE=CYCLE/'features_expanded_all_v1';OUT=CYCLE/'detection_features_v1'


def main():
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'))
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists();OUT.mkdir();(OUT/'parts').mkdir();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='extracting_all_query_detection_features',active_job=OUT.name,active_pid=os.getpid());write_json(CYCLE/'state.json',state);status('loading')
    source=json.loads((SOURCE/'manifest.json').read_text());initial=OLD/'training_v1/pilot/initial_heads.pt';encoder=ROOT/'vggt-omega/vggt_omega_1b_512.pt'
    assert fingerprint(initial)==source['initial_sha256'] and fingerprint(encoder)==source['encoder_sha256']
    spatial=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
    raw=torch.load(encoder,map_location='cpu',weights_only=True,mmap=True);chosen={k:v for k,v in raw.items() if k.startswith(('aggregator.','dense_head.'))};result=spatial.load_state_dict(chosen,strict=False);assert not result.unexpected_keys and all(k.startswith('instance_pose_head.') for k in result.missing_keys);del raw,chosen
    model=SurfaceCurveDepthPoseModel.from_spatial_model(spatial).cuda().eval().requires_grad_(False);del spatial
    heads=torch.load(initial,map_location='cpu',weights_only=True);result=model.load_state_dict(heads,strict=False);assert not result.unexpected_keys and all(k.startswith('aggregator.') for k in result.missing_keys)
    detector=InstanceDetectionReadout().cuda().eval();detector.load_state_dict({k[len('instance_pose_head.'):]:v for k,v in heads.items() if k.startswith(('instance_pose_head.box.','instance_pose_head.objectness.'))},strict=True)
    torch.save({k:v.detach().cpu() for k,v in detector.state_dict().items()},OUT/'initial_detector.pt');del heads
    records=[];seen={};count=collections.Counter();audit=[];maxdelta=0.;gold_count=0;examples=[]
    for n,record in enumerate(source['records']):
        assert dt.datetime.now(dt.timezone.utc)<deadline
        r=record['record'];assert r.get('session') not in ('GR2','UR2','UR3')
        assert fingerprint(ROOT/record['path'])==record['sha256']
        reference=torch.load(ROOT/record['path'],map_location='cpu',weights_only=False)
        old_cache=OLD/f'training_v1/pilot/cache/{r["key"]}.pt'
        item=torch.load(old_cache,map_location='cpu',weights_only=False) if r['key'].isdigit() else load_item(r)
        assert item['source_hashes']==record['source_hashes'] and all(fingerprint(ROOT/p)==sha for p,sha in record['source_hashes'].items())
        b=move_batch(item['batch'],'cuda');inst=[];pose=[]
        a=model.instance_pose_head.instance_decoder.register_forward_hook(lambda m,args,out:inst.append(out.detach()))
        hook=model.instance_pose_head.keypoint_decoder.register_forward_hook(lambda m,args,out:pose.append(out.detach()))
        try:
            if 'cached' in item:
                torch.backends.cudnn.allow_tf32=True
                with torch.no_grad():pred=model.forward_cached(b['images'],[None if x is None else x.cuda() for x in item['cached']],item['patch_start'],coupling='off')
            else:
                torch.backends.cudnn.allow_tf32=False
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):features,start=model.aggregator(b['images'])
                cached=[x if j in (4,11,17,23) else None for j,x in enumerate(features)];del features
                torch.backends.cudnn.allow_tf32=not r['key'].startswith('x')
                with torch.no_grad():pred=model.forward_cached(b['images'],cached,start,coupling='off')
                del cached
        finally:a.remove();hook.remove()
        with torch.no_grad():direct=detector(inst[0])
        for key,value in direct.items():torch.testing.assert_close(value,pred[key][0],atol=0,rtol=0)
        h,w=b['images'].shape[-2:];tokens=pose[0].reshape(len(r['views']),6,23,256).cpu();t=instance_targets_from_batch(b)
        for o in reference['observations']:
            m=o['metadata'];delta=float((tokens[m['view_index'],m['query']]-o['pose']).abs().max());maxdelta=max(maxdelta,delta);assert delta<2e-4;gold_count+=1
        observations=[]
        for v,view in enumerate(r['views']):
            identity=(r['scene'],view)
            if identity in seen:
                assert seen[identity]==r['role'];continue
            seen[identity]=r['role'];valid=t['instance_valid'][0,v].cpu();box_valid=t['box_valid'][0,v].cpu();mask_valid=t['mask_valid'][0,v].cpu();expected=int(item['batch']['actual_instance_count'].reshape(-1)[0])
            complete=bool(((valid&box_valid&mask_valid).sum()==expected) and expected>0)
            # No sparse-points-to-box fallback becomes supervised detection GT.
            for m in torch.nonzero(valid&box_valid&mask_valid).flatten().tolist():
                mask=t['masks'][0,v,m].cpu().numpy();derived=torch.tensor(_normalized_box_from_mask(mask));box=t['boxes_cxcywh'][0,v,m].cpu();delta=float((derived-box).abs().max())
                assert delta<1e-6,('Review explicit/alternate box semantics before training',r['scene'],view,m,delta)
                audit.append(dict(kind=r['kind'],role=r['role'],scene=r['scene'],view=view,instance=m,box_area=float(box[2]*box[3]),mask_area_fraction=float(mask.mean()),mask_box_delta=delta))
            meta=dict(record=r,view=view,view_index=v,expected_instances=expected,detection_complete=complete,memory_part=record['path'])
            observations.append(dict(metadata=meta,instance_tokens=inst[0][v].cpu(),pose_tokens=tokens[v],
                original_boxes=pred['instance_boxes_cxcywh'][0,v].cpu(),original_objectness=pred['instance_objectness_logits'][0,v].cpu(),
                coarse_xy=pred['instance_keypoint_coarse_xy'][0,v].cpu(),original_xy=pred['instance_keypoint_xy'][0,v].cpu(),
                boxes=t['boxes_cxcywh'][0,v].cpu(),box_valid=box_valid&mask_valid,instance_valid=valid,
                gt_xy=t['keypoint_xy'][0,v].cpu(),point_valid=t['keypoint_valid'][0,v].cpu(),pixels=torch.tensor([w,h],dtype=torch.float32)))
            count[(r['kind'],r['role'],expected,complete)]+=1
            if len(examples)<12 and r['role']=='validation':examples.append(dict(metadata=meta,gt_boxes=t['boxes_cxcywh'][0,v].cpu().tolist(),gt_valid=box_valid.tolist(),pred_boxes=pred['instance_boxes_cxcywh'][0,v].cpu().tolist()))
        path=OUT/'parts'/f'{r["key"]}.pt';torch.save(dict(observations=observations),path);records.append(dict(key=r['key'],path=path.relative_to(ROOT).as_posix(),sha256=fingerprint(path),count=len(observations),record=r,source_hashes=item['source_hashes']))
        status('extracting',completed=n+1,total=len(source['records']),unique_images=len(seen),max_pose_delta=maxdelta);print('DETECTION_FEATURES',n+1,len(source['records']),len(seen),flush=True)
    assert gold_count==source['observations']
    write_json(OUT/'box_audit.json',dict(records=audit,counts={str(k):v for k,v in count.items()},examples=examples,
        semantics='Mask-supported full visible instance bounding rectangle, including whatever tail is present in accepted instance mask; real/synthetic use identical normalized pixel-edge bbox function. All supervised boxes audited against masks. Sparse points never promoted to boxGT. Missing mask views marked incomplete and excluded from first detection-only training, not negative.',
        caveat='This verifies geometric conversion/contract, not correctness of every source segmentation. ProtectedR12oldmask/source hashes preserved.'))
    write_json(OUT/'manifest.json',dict(records=records,images=len(seen),counts={str(k):v for k,v in count.items()},source_manifest_sha256=fingerprint(SOURCE/'manifest.json'),initial_detector_sha256=fingerprint(OUT/'initial_detector.pt'),gold_observations=gold_count,max_pose_delta=maxdelta,no_labels_changed=True,no_locked_holdout=True))
    status('complete',images=len(seen),gold_observations=gold_count,max_pose_delta=maxdelta)
    state.update(stage='detection_features_ready',active_job=None,active_pid=None,next='Review detection_features_v1/box_audit.json counts/visual examples. Train only box+objectness MLP from initial_detector on complete-mask views, original split, source-uniform real/synthetic. Encoder/instance+pointdecoder frozen. Report IoU/precision/recall, duplicate count and query switches; then recompute spatial points using PREDICTED adaptedboxes for allqueries.');state['completed'].append('all_query_detection_features_and_mask_box_audit');write_json(CYCLE/'state.json',state)


if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()));raise
