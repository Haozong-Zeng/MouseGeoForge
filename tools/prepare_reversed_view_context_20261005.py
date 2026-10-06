"""Same two images, reversed encoder reference order; role-separated caches."""
import argparse,datetime as dt,fcntl,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from torch.nn import functional as F
from vggt_omega.models import VGGTOmega
from mousegeoforge.models.surface_curve_pose import SurfaceCurveDepthPoseModel
from mousegeoforge.models.spatial_detection_readout import SpatialDetectionReadout,sample_predicted_roi
from mousegeoforge.models.view_token_probe import shared_first_view_tokens
from training.instance_pose_losses import instance_targets_from_batch
from run_surface_curve_pilot_20261003 import load_item
from run_depth_pose_ablation import fingerprint,write_json,move_batch
CYCLE=ROOT/'output/core23_spatial_cycle_20261004';OLD=ROOT/'output/surface_curve_cycle_20261003';OUT=CYCLE/'reversed_view_train_context_v1'

def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--role',choices=['train','validation'],default='train');parser.add_argument('--shared-tokens',action='store_true');parser.add_argument('--eligible-full',action='store_true');args=parser.parse_args();assert not args.eligible_full or args.shared_tokens;context_name='fullshared' if args.eligible_full else 'shared' if args.shared_tokens else 'reversed';OUT=CYCLE/f'{context_name}_view_{args.role}_context_v1'
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists();OUT.mkdir();(OUT/'parts').mkdir();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='extracting_reversed_view_context',active_job=OUT.name,active_pid=os.getpid());write_json(CYCLE/'state.json',state);status('loading')
    source=CYCLE/'detection_features_v1/manifest.json';manifest=json.loads(source.read_text());tasks=[];index=0
    for rec in manifest['records']:
        assert fingerprint(ROOT/rec['path'])==rec['sha256'];observations=torch.load(ROOT/rec['path'],map_location='cpu',weights_only=False)['observations'];selected=[]
        for o in observations:
            m=o['metadata'];assert m['record'].get('session') not in ['GR2','UR2','UR3']
            if m['record']['role']==args.role and m['detection_complete'] and bool(o['point_valid'].any()):selected.append((index,o))
            index+=1
        if selected:tasks.append((rec,selected))
    total=sum(len(obs) for _,obs in tasks);assert index==1350
    full_gold={};full_golden_cases=0;max_full_golden=0.
    if args.eligible_full:
        by_index={i:o for _,obs in tasks for i,o in obs};fullplan_path=CYCLE/f'eligible_fullview_context_plan_v1/{args.role}.json';fullplan=json.loads(fullplan_path.read_text());assert fullplan['role']==args.role and fullplan['original_source_sha256']==fingerprint(source)
        tasks=[]
        for task in fullplan['tasks']:
            selected=[(i,by_index[i]) for i in task['image_indices']];r=task['record'];assert set(r['views'])=={o['metadata']['view'] for _,o in selected}
            # Keep the exact original ordered pair where full context has only2.
            if len(r['views'])==2:
                old=selected[0][1]['metadata']['record']['views'];assert set(old)==set(r['views']);r=dict(r,views=old)
            tasks.append((dict(record=r,source_hashes=task['known_source_hashes']),selected))
        assert sum(len(obs) for _,obs in tasks)==total
        for rec in json.loads((CYCLE/f'shared_view_{args.role}_context_v1/manifest.json').read_text())['records']:full_gold[rec['image_index']]=rec
        assert set(full_gold)==set(by_index)
    initial=OLD/'training_v1/pilot/initial_heads.pt';encoder=ROOT/'vggt-omega/vggt_omega_1b_512.pt';native=json.loads((CYCLE/'native_decoder_memory_v1/manifest.json').read_text());assert fingerprint(initial)==native['initial_sha256'] and fingerprint(encoder)==native['encoder_sha256']
    spatial=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23);raw=torch.load(encoder,map_location='cpu',weights_only=True,mmap=True);chosen={k:v for k,v in raw.items() if k.startswith(('aggregator.','dense_head.'))};loaded=spatial.load_state_dict(chosen,strict=False);assert not loaded.unexpected_keys and all(k.startswith('instance_pose_head.') for k in loaded.missing_keys);del raw,chosen
    model=SurfaceCurveDepthPoseModel.from_spatial_model(spatial).cuda().eval().requires_grad_(False);del spatial;loaded=model.load_state_dict(torch.load(initial,map_location='cpu',weights_only=True),strict=False);assert not loaded.unexpected_keys and all(k.startswith('aggregator.') for k in loaded.missing_keys)
    detectors={};detector_hashes={}
    for seed in [17,29,43]:
        path=CYCLE/f'detection_view_context_control_v1/mixed_seed{seed}_latest.pt';cp=torch.load(path,map_location='cuda',weights_only=False);assert cp['step']==800 and cp['seed']==seed and cp['arm']=='mixed';detectors[seed]=SpatialDetectionReadout().cuda().eval().requires_grad_(False);detectors[seed].load_state_dict(cp['state'],strict=True);detector_hashes[seed]=fingerprint(path)
    token_snapshot={name:getattr(model.aggregator,name).detach().clone() for name in ['camera_token','register_token']}
    def forward(item,original=False,intervention=None):
        enabled=args.shared_tokens and not original if intervention is None else intervention
        batch=move_batch(item['batch'],'cuda');memory=[];inst=[];pose=[];head=model.instance_pose_head
        hooks=[head.instance_decoder.register_forward_pre_hook(lambda m,a:memory.append(a[1].detach())),head.instance_decoder.register_forward_hook(lambda m,a,o:inst.append(o.detach())),head.keypoint_decoder.register_forward_hook(lambda m,a,o:pose.append(o.detach()))]
        try:
            if 'cached' in item:features=[None if v is None else v.cuda() for v in item['cached']];start=item['patch_start']
            else:
                torch.backends.cudnn.allow_tf32=False
                with torch.no_grad(),shared_first_view_tokens(model.aggregator,enabled=enabled),torch.autocast('cuda',dtype=torch.bfloat16):features,start=model.aggregator(batch['images'])
                for name,value in token_snapshot.items():torch.testing.assert_close(getattr(model.aggregator,name),value,atol=0,rtol=0)
                features=[v if j in (4,11,17,23) else None for j,v in enumerate(features)]
            torch.backends.cudnn.allow_tf32=not (original and item['record']['key'].startswith('x'))
            with torch.no_grad():pred=model.forward_cached(batch['images'],features,start,coupling='off')
        finally:
            for hook in hooks:hook.remove()
        assert len(memory)==len(inst)==len(pose)==1;h,w=batch['images'].shape[-2:];views=len(item['record']['views']);maps=memory[0].transpose(1,2).reshape(views,256,h//head.patch_size,w//head.patch_size)
        return pred,inst[0],pose[0].reshape(views,6,23,256),maps
    write_json(OUT/'protocol.json',dict(role=args.role,images=total,source_sha256=fingerprint(source),initial_sha256=fingerprint(initial),encoder_sha256=fingerprint(encoder),detector_sha256=detector_hashes,script_sha256=fingerprint(Path(__file__)),contract='Original pretrained reference-dependent encoder kept intact. Actual2view input reversed, sameRGB pixels/labels/hash set. Physical camera alignment verified exactly; metadata retains old view_index, context_view_index records reversed index. No shared-token intervention. Encoderbf16 TF32off,headfp32 TF32on; nativeFP32memory,ROIhalf48x48half path. Frozen mixed800ROI detector predictions all3seeds; no GT in feature/ROI sampling. Train and validation caches separate.'))
    records=[];checked=set();audit=[];started=time.monotonic()
    if args.shared_tokens:
        protocol=json.loads((OUT/'protocol.json').read_text());protocol.update(context='shared_first_special_tokens_original_image_order',contract='Runtime-only first camera/register slots copied toother slots while extracting2Dbranch features. ORIGINAL imageorder, pixels/GT/sourcehashes unchanged. Restore tokenparameters aftereveryforward. No checkpointwrites. Encoderbf16TF32off/headfp32TF32on, nativeFP32memory andROIhalf48half path. Same frozenmixed800ROI detector weights, predictions maychange withfeaturecontext. Original geometry mustbe preserved by separateoriginaltokenbranch atintegration; geometry fromthiscache isNOT used asGT.',probe_sha256=fingerprint(ROOT/'mousegeoforge/models/view_token_probe.py'));write_json(OUT/'protocol.json',protocol)
    if args.eligible_full:
        protocol.update(context='shared_first_tokens_eligible_full_context',full_context_plan_sha256=fingerprint(fullplan_path),contract='Actual fulleligible scene re-encoding, sharedfirsttokens, noadded/excluded/incomplete cameraimages. Fullcontext order physicalcamera ascending except exactoriginal order for2viewgolden controls. Original metadata/view_index preserved; context_view_index mapsactual fullcontext. Sourcehashverifiedsubsetofplanunion; perphysicalcameraRGB/GT/boxes exact. Samefrozenmixed800ROI3seeds, originalgeometry nottaken fromsharedpass.');write_json(OUT/'protocol.json',protocol)
    for rec,obs in tasks:
        assert dt.datetime.now(dt.timezone.utc)<deadline;r=rec['record'];assert args.eligible_full or len(r['views'])==2
        original_record=obs[0][1]['metadata']['record'] if args.eligible_full else r
        original_item=load_item(original_record);reverse_record=dict(r,views=list(r['views']) if args.shared_tokens else list(reversed(r['views'])));item=load_item(reverse_record)
        if args.eligible_full:
            for p,sha in item['source_hashes'].items():assert rec['source_hashes'].get(p)==sha
            for _,o in obs:
                v=r['views'].index(o['metadata']['view']);single=load_item(dict(o['metadata']['record'],views=[o['metadata']['view']]));torch.testing.assert_close(item['batch']['images'][:,v],single['batch']['images'][:,0],atol=0,rtol=0)
        else:
            assert original_item['source_hashes']==item['source_hashes']==rec['source_hashes']
            torch.testing.assert_close(item['batch']['images'],original_item['batch']['images'] if args.shared_tokens else original_item['batch']['images'].flip(1),atol=0,rtol=0)
        assert all(fingerprint(ROOT/p)==sha for p,sha in item['source_hashes'].items())
        category='original' if r['key'].isdigit() else r['key'][0]
        if category not in checked:
            cache=OLD/f'training_v1/pilot/cache/{original_record["key"]}.pt';checkitem=torch.load(cache,map_location='cpu',weights_only=False) if original_record['key'].isdigit() else original_item;pp,pi,pt,pm=forward(checkitem,original=True)
            for _,o in obs:
                if args.eligible_full and o['metadata']['record']['key']!=original_record['key']:continue
                v=o['metadata']['view_index'];torch.testing.assert_close(pi[v].cpu(),o['instance_tokens'],atol=0,rtol=0);torch.testing.assert_close(pt[v].cpu(),o['pose_tokens'],atol=0,rtol=0)
            checked.add(category);audit.append(dict(category=category,key=r['key'],original_goldens_exact=True));del pp,pi,pt,pm
            if args.shared_tokens:
                one=load_item(dict(r,views=[r['views'][0]]));plain=forward(one,intervention=False);shared=forward(one,intervention=True)
                assert plain[0].keys()==shared[0].keys()
                for key in plain[0]:torch.testing.assert_close(plain[0][key],shared[0][key],atol=0,rtol=0)
                for a,b in zip(plain[1:],shared[1:]):torch.testing.assert_close(a,b,atol=0,rtol=0)
                audit[-1]['single_full_output_exact']=True;del plain,shared,one
        target=instance_targets_from_batch(item['batch']);pred,it,pt,maps=forward(item)
        with torch.no_grad():
            scaled=F.interpolate(maps.half().float(),(48,48),mode='bilinear',align_corners=False);roi,glob=sample_predicted_roi(scaled,pred['instance_boxes_cxcywh'][0]);adapted={s:head(it,roi.half().float(),glob.half().float(),pred['instance_boxes_cxcywh'][0],pred['instance_objectness_logits'][0]) for s,head in detectors.items()}
        for image_index,o in obs:
            m=o['metadata'];v=reverse_record['views'].index(m['view']);assert args.eligible_full or v==(m['view_index'] if args.shared_tokens else 1-m['view_index'])
            torch.testing.assert_close(target['keypoint_xy'][0,v],o['gt_xy'],atol=0,rtol=0,equal_nan=True);torch.testing.assert_close(target['keypoint_valid'][0,v],o['point_valid'],atol=0,rtol=0);torch.testing.assert_close(target['boxes_cxcywh'][0,v],o['boxes'],atol=0,rtol=0)
            if args.eligible_full and len(r['views'])==2:
                golden=full_gold[image_index];assert fingerprint(ROOT/golden['path'])==golden['sha256'];old=torch.load(ROOT/golden['path'],map_location='cpu',weights_only=False);assert golden['record']['views']==r['views']
                for value,key in [(maps[v],'memory'),(it[v],'instance_tokens'),(pt[v],'pose_tokens')]:
                    delta=float((value.cpu()-old[key]).abs().max());assert delta<2e-5;max_full_golden=max(max_full_golden,delta)
                for s,p in adapted.items():
                    for key,value in p.items():
                        delta=float((value[v].cpu()-old['adapted'][s][key]).abs().max());assert delta<2e-5;max_full_golden=max(max_full_golden,delta)
                full_golden_cases+=1
            path=OUT/'parts'/f'{image_index:04d}.pt';torch.save(dict(image_index=image_index,metadata=m,context_record=reverse_record,context_kind=context_name,context_view_index=v,memory=maps[v].cpu(),instance_tokens=it[v].cpu(),pose_tokens=pt[v].cpu(),original_boxes=pred['instance_boxes_cxcywh'][0,v].cpu(),original_objectness=pred['instance_objectness_logits'][0,v].cpu(),adapted={s:{k:values[v].cpu() for k,values in p.items()} for s,p in adapted.items()}),path)
            records.append(dict(image_index=image_index,path=path.relative_to(ROOT).as_posix(),sha256=fingerprint(path),record=reverse_record,metadata=m,context_view_index=v,source_hashes=item['source_hashes']))
        if len(records)%25<2:write_json(OUT/'partial_manifest.json',dict(records=records,role=args.role))
        status('extracting',completed=len(records),total=total,elapsed_seconds=time.monotonic()-started)
        if len(records)%100<2:print('REVERSE_CONTEXT',args.role,len(records),total,flush=True)
    assert len(records)==total;write_json(OUT/'manifest.json',dict(records=records,role=args.role,images=total,source_sha256=fingerprint(source),original_parity=audit,protocol_sha256=fingerprint(OUT/'protocol.json'),bytes=sum((ROOT/r['path']).stat().st_size for r in records),no_labels_changed=True));status('complete',images=total)
    if args.eligible_full:
        manifest_path=OUT/'manifest.json';result=json.loads(manifest_path.read_text());result.update(full2view_golden_cases=full_golden_cases,max_full2view_golden_delta=max_full_golden,context_plan_sha256=fingerprint(fullplan_path));assert full_golden_cases>0;write_json(manifest_path,result)
    state.update(stage='reversed_view_context_ready',active_job=None,active_pid=None,updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),next='Reversed2view features ready with physical-camera RGB/GT exactalignment. Build matched fixed-order vsrandom-reference augmentation for pose from mixed800pose, fixedmixed800detector. Same1/2view exposure and image schedule; reverse only half paired occurrences. Separate validationcache. Full138queries, no GTforward. Start800steps3seeds; evaluate original/reversed/single contexts and RGB1/2/all; no automaticpromotion.');state['completed'].append(OUT.name+'_extracted');write_json(CYCLE/'state.json',state)
    if args.shared_tokens:
        state.update(stage='shared_token_context_ready',next='Shared-first-token original-order2D cache ready. GeometrynotGT; preserveoriginalgeometry integration. Followshared-token adaptation protocol andplan, do notduplicate jobs.');write_json(CYCLE/'state.json',state)

if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise
