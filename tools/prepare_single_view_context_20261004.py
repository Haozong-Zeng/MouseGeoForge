"""Re-encode the same accepted TRAIN images without their second view.

No labels enter encoding, no pair-memory slicing substitutes for re-encoding.
Validation extraction is a separate explicit invocation and separate directory.
"""
import argparse, collections, datetime as dt, fcntl, json, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import torch
from torch.nn import functional as F
from vggt_omega.models import VGGTOmega
from mousegeoforge.models.surface_curve_pose import SurfaceCurveDepthPoseModel
from mousegeoforge.models.spatial_detection_readout import SpatialDetectionReadout, sample_predicted_roi
from training.instance_pose_losses import instance_targets_from_batch
from run_surface_curve_pilot_20261003 import load_item
from run_depth_pose_ablation import fingerprint, write_json, move_batch
CYCLE=ROOT/'output/core23_spatial_cycle_20261004';OLD=ROOT/'output/surface_curve_cycle_20261003';OUT=CYCLE/'single_view_train_context_v1'

def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--role',choices=['train','validation'],default='train');args=parser.parse_args()
    OUT=CYCLE/f'single_view_{args.role}_context_v1'
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'))
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists();OUT.mkdir();(OUT/'parts').mkdir();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='extracting_single_view_context',active_job=OUT.name,active_pid=os.getpid());write_json(CYCLE/'state.json',state);status('loading')
    source=CYCLE/'detection_features_v1/manifest.json';manifest=json.loads(source.read_text());tasks=[];index=0
    for rec in manifest['records']:
        assert fingerprint(ROOT/rec['path'])==rec['sha256'];obs=torch.load(ROOT/rec['path'],map_location='cpu',weights_only=False)['observations']
        chosen=[]
        for o in obs:
            m=o['metadata'];r=m['record'];assert r.get('session') not in ['GR2','UR2','UR3']
            if r['role']==args.role and m['detection_complete'] and bool(o['point_valid'].any()):chosen.append((index,o))
            index+=1
        if chosen:tasks.append((rec,chosen))
    total=sum(len(obs) for _,obs in tasks);assert index==1350 and total>0
    initial=OLD/'training_v1/pilot/initial_heads.pt';encoder=ROOT/'vggt-omega/vggt_omega_1b_512.pt';native=json.loads((CYCLE/'native_decoder_memory_v1/manifest.json').read_text())
    assert fingerprint(initial)==native['initial_sha256'] and fingerprint(encoder)==native['encoder_sha256']
    model0=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
    raw=torch.load(encoder,map_location='cpu',weights_only=True,mmap=True);selected={k:v for k,v in raw.items() if k.startswith(('aggregator.','dense_head.'))};result=model0.load_state_dict(selected,strict=False);assert not result.unexpected_keys and all(k.startswith('instance_pose_head.') for k in result.missing_keys);del raw,selected
    model=SurfaceCurveDepthPoseModel.from_spatial_model(model0).cuda().eval().requires_grad_(False);del model0
    result=model.load_state_dict(torch.load(initial,map_location='cpu',weights_only=True),strict=False);assert not result.unexpected_keys and all(k.startswith('aggregator.') for k in result.missing_keys)
    detectors={};detector_hashes={}
    for seed in [17,29,43]:
        path=CYCLE/f'detection_context_roi_v1/seed{seed}_latest.pt';cp=torch.load(path,map_location='cpu',weights_only=False);assert cp['seed']==seed and cp['step']==1600
        detectors[seed]=SpatialDetectionReadout().cuda().eval().requires_grad_(False);detectors[seed].load_state_dict(cp['state'],strict=True);detector_hashes[seed]=fingerprint(path)
    def forward(item,paired=False):
        b=move_batch(item['batch'],'cuda');memory=[];inst=[];pose=[];head=model.instance_pose_head
        hooks=[head.instance_decoder.register_forward_pre_hook(lambda m,a:memory.append(a[1].detach())),head.instance_decoder.register_forward_hook(lambda m,a,o:inst.append(o.detach())),head.keypoint_decoder.register_forward_hook(lambda m,a,o:pose.append(o.detach()))]
        try:
            if 'cached' in item:
                features=[None if x is None else x.cuda() for x in item['cached']];start=item['patch_start']
            else:
                torch.backends.cudnn.allow_tf32=False
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):features,start=model.aggregator(b['images'])
                features=[x if j in (4,11,17,23) else None for j,x in enumerate(features)]
            torch.backends.cudnn.allow_tf32=not (paired and item['record']['key'].startswith('x'))
            with torch.no_grad():pred=model.forward_cached(b['images'],features,start,coupling='off')
        finally:
            for hook in hooks:hook.remove()
        assert len(memory)==len(inst)==len(pose)==1
        h,w=b['images'].shape[-2:];v=b['images'].shape[1]
        maps=memory[0].transpose(1,2).reshape(v,256,h//head.patch_size,w//head.patch_size)
        return pred,inst[0],pose[0].reshape(v,6,23,256),maps
    write_json(OUT/'protocol.json',dict(role=args.role,total=total,source_sha256=fingerprint(source),initial_sha256=fingerprint(initial),encoder_sha256=fingerprint(encoder),detector_sha256=detector_hashes,script_sha256=fingerprint(Path(__file__)),
        contract='Exactly same accepted image, point/mask box labels, source split and normalization as paired pool; encoder actually rerun with V=1. FP32 native memory, all6instance queries. Single-view encoder bf16 TF32off, headfp32 TF32on matching live wrapper. ROI detector fp16roundtrip path; all3seed detectors frozen. No new images or annotation edits. Role-separated manifest prevents validation variants entering TRAIN.',
        scope='All eligible images of this role, not a selected difficult subset; source-complete set avoids an additional image-selection confound. Single maps expected about3GB for TRAIN, no large model download.'))
    records=[];audit=[];checked=set();counts=collections.Counter();started=time.monotonic()
    for rec,obs in tasks:
        assert dt.datetime.now(dt.timezone.utc)<deadline
        r=rec['record'];pair=load_item(r)
        assert pair['source_hashes']==rec['source_hashes']
        assert all(fingerprint(ROOT/p)==sha for p,sha in rec['source_hashes'].items())
        category='original' if r['key'].isdigit() else r['key'][0]
        if category not in checked:
            cache=OLD/f'training_v1/pilot/cache/{r["key"]}.pt';parity_item=torch.load(cache,map_location='cpu',weights_only=False) if r['key'].isdigit() else pair
            pp,pi,pt,pm=forward(parity_item,paired=True)
            for _,o in obs:
                v=o['metadata']['view_index'];torch.testing.assert_close(pi[v].cpu(),o['instance_tokens'],atol=0,rtol=0);torch.testing.assert_close(pt[v].cpu(),o['pose_tokens'],atol=0,rtol=0)
            audit.append(dict(category=category,key=r['key'],paired_instance_and_pose_exact=True));checked.add(category);del pp,pi,pt,pm
        for image_index,o in obs:
            assert dt.datetime.now(dt.timezone.utc)<deadline
            m=o['metadata'];v=m['view_index'];single=load_item(dict(r,views=[m['view']]))
            assert single['batch']['images'].shape[1]==1
            torch.testing.assert_close(single['batch']['images'][0,0],pair['batch']['images'][0,v],atol=0,rtol=0)
            assert all(rec['source_hashes'][p]==sha for p,sha in single['source_hashes'].items())
            target=instance_targets_from_batch(single['batch'])
            torch.testing.assert_close(target['keypoint_xy'][0,0],o['gt_xy'],atol=0,rtol=0,equal_nan=True)
            torch.testing.assert_close(target['keypoint_valid'][0,0],o['point_valid'],atol=0,rtol=0)
            torch.testing.assert_close(target['boxes_cxcywh'][0,0],o['boxes'],atol=0,rtol=0)
            pred,it,pt,maps=forward(single)
            with torch.no_grad():
                readout_maps=F.interpolate(maps.half().float(),(48,48),mode='bilinear',align_corners=False)
                roi,global_memory=sample_predicted_roi(readout_maps,pred['instance_boxes_cxcywh'][0]);adapted={}
                for seed,detector in detectors.items():
                    result=detector(it,roi.half().float(),global_memory.half().float(),pred['instance_boxes_cxcywh'][0],pred['instance_objectness_logits'][0])
                    adapted[seed]={k:value[0].cpu() for k,value in result.items()}
            path=OUT/'parts'/f'{image_index:04d}.pt'
            torch.save(dict(image_index=image_index,metadata=m,single_record=single['record'],memory=maps[0].cpu(),instance_tokens=it[0].cpu(),pose_tokens=pt[0].cpu(),coarse_xy=pred['instance_keypoint_coarse_xy'][0,0].cpu(),adapted=adapted),path)
            counts[(r['kind'],r['role'],m['view'])]+=1
            records.append(dict(image_index=image_index,path=path.relative_to(ROOT).as_posix(),sha256=fingerprint(path),record=single['record'],source_hashes=single['source_hashes'],metadata=m,instance_delta_from_pair=float((it[0].cpu()-o['instance_tokens']).abs().max())))
            status('extracting',completed=len(records),total=total,elapsed_seconds=time.monotonic()-started);print('SINGLE_CONTEXT',args.role,len(records),total,flush=True)
            if len(records)%25==0:write_json(OUT/'partial_manifest.json',dict(records=records,role=args.role,paired_audit=audit))
            del pred,it,pt,maps,single
    assert len(records)==total
    write_json(OUT/'manifest.json',dict(records=records,role=args.role,images=total,paired_audit=audit,counts={str(k):v for k,v in counts.items()},source_sha256=fingerprint(source),protocol_sha256=fingerprint(OUT/'protocol.json'),bytes=sum((ROOT/r['path']).stat().st_size for r in records),no_label_changes=True))
    status('complete',images=total,elapsed_seconds=time.monotonic()-started)
    state.update(stage='single_view_context_ready',active_job=None,active_pid=None,updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),next='Check single_view TRAIN/validation manifests, exact paired and label parity. Implement matched pair-only vs50percent single-context training from balanced2400, same image schedule/bottom exposure, frozen detector percontext, all138queries. Separate validation-only variants strictly. Start800steps3seeds, retain oldpaired and single validation metrics; then fullRGB1/2/all audit. No blind extension or production promotion.')
    state['completed'].append(OUT.name+'_extracted');write_json(CYCLE/'state.json',state)

if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise
