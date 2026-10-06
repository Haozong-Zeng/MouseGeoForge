"""Original RGB -> independent V1 encoder -> detector -> pose -> candidates.

Fixed first validation real scene and first synthetic single scene. Cached
features/predictions are used ONLY for comparisons after fresh inference.
"""
import datetime as dt,fcntl,json,os,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from torch.nn import functional as F
from vggt_omega.models import VGGTOmega
from mousegeoforge.models.surface_curve_pose import SurfaceCurveDepthPoseModel
from mousegeoforge.models.instance_detection_readout import InstanceDetectionReadout
from mousegeoforge.models.spatial_detection_readout import SpatialDetectionReadout,sample_predicted_roi
from mousegeoforge.models.decoder_spatial_pose_readout import DecoderSpatialPoseReadout
from mousegeoforge.epipolar_candidate_readout import probability_candidates,rerank_candidates
from run_surface_curve_pilot_20261003 import load_item
from run_depth_pose_ablation import fingerprint,write_json,move_batch
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';OLD=ROOT/'output/core23_spatial_cycle_20261004';SOURCE=CYCLE/'singleview_fusion_inputs_seed17_v1';MAPS=CYCLE/'singleview_probability_diagnostic_v1'
OUT=CYCLE/'epipolar_candidate_rgb_audit_v1';DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005');KP=[0,14,15,16,17]


def main():
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'))
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists() and torch.cuda.is_available();OUT.mkdir();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='validating_candidate_rgb',active_job=OUT.name,active_pid=os.getpid());write_json(CYCLE/'state.json',state)
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    status('loading');started=time.monotonic()
    initial=ROOT/'output/surface_curve_cycle_20261003/training_v1/pilot/initial_heads.pt';encoder=ROOT/'vggt-omega/vggt_omega_1b_512.pt';native_info=json.loads((OLD/'native_decoder_memory_v1/manifest.json').read_text())
    assert fingerprint(initial)==native_info['initial_sha256'] and fingerprint(encoder)==native_info['encoder_sha256']
    model0=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
    raw=torch.load(encoder,map_location='cpu',weights_only=True,mmap=True);selected={k:v for k,v in raw.items() if k.startswith(('aggregator.','dense_head.'))}
    result=model0.load_state_dict(selected,strict=False);assert not result.unexpected_keys and all(k.startswith('instance_pose_head.') for k in result.missing_keys);del raw,selected
    model=SurfaceCurveDepthPoseModel.from_spatial_model(model0).cuda().eval().requires_grad_(False);del model0
    result=model.load_state_dict(torch.load(initial,map_location='cpu',weights_only=True),strict=False);assert not result.unexpected_keys and all(k.startswith('aggregator.') for k in result.missing_keys)
    basepath=OLD/'detection_features_v1/initial_detector.pt';detpath=OLD/'detection_view_context_control_v1/mixed_seed17_latest.pt';posepath=OLD/'full_context_pose_control_v1/full_seed17_latest.pt'
    basehead=InstanceDetectionReadout().cuda().eval().requires_grad_(False);basehead.load_state_dict(torch.load(basepath,map_location='cpu',weights_only=True),strict=True)
    det=SpatialDetectionReadout().cuda().eval().requires_grad_(False);det.load_state_dict(torch.load(detpath,map_location='cpu',weights_only=False)['state'],strict=True)
    pose=DecoderSpatialPoseReadout().cuda().eval().requires_grad_(False);pose.load_state_dict(torch.load(posepath,map_location='cpu',weights_only=False)['state'],strict=True)
    data=torch.load(SOURCE/'validation.pt',map_location='cpu',weights_only=False)
    cameras=torch.load(CYCLE/'camera_geometry_audit_v2/validation_cameras.pt',map_location='cpu',weights_only=False)
    groups=[next(g for g in cameras if g['kind']==kind) for kind in ['real','synthetic']]
    native={r['image_index']:r for r in json.loads((OLD/'single_view_validation_context_v1/manifest.json').read_text())['records']}
    with np.load(MAPS/'validation_probabilities.npz') as a:goldprob=a['probability'];maprow={int(i):j for j,i in enumerate(a['row_indices'])}
    with np.load(CYCLE/'epipolar_candidates_pilot_v1/validation_predictions.npz') as a:goldcandidate=a['epipolar_local_xy']
    records=[];saved={};total=sum(len(g['row_indices']) for g in groups)
    write_json(OUT/'protocol.json',dict(encoder_sha256=fingerprint(encoder),initial_sha256=fingerprint(initial),base_detector_sha256=fingerprint(basepath),detector_sha256=fingerprint(detpath),pose_sha256=fingerprint(posepath),
        image_indices=[data['image_indices'][i] for g in groups for i in g['row_indices']],groups=[g['scene'] for g in groups],
        scope='Fixed first eligible real and synthetic development scenes, no GT selection. Independent V1 encoder forward per image; not compute-matched to a joint multiview encoder. Cached outputs ONLY for parity, never fresh inference inputs.'))
    for g in groups:
        live=[];ids=g['row_indices']
        for vi,i in enumerate(ids):
            assert dt.datetime.now(dt.timezone.utc)<deadline
            m=data['metadata'][i];assert m['record']['role']=='validation' and m['record'].get('session') not in ['GR2','UR2','UR3']
            item=load_item(dict(m['record'],views=[int(m['view'])]));batch=move_batch(item['batch'],'cuda');images=batch['images']
            memory=[];instance=[];head=model.instance_pose_head
            hooks=[head.instance_decoder.register_forward_pre_hook(lambda module,args:memory.append(args[1].detach())),head.instance_decoder.register_forward_hook(lambda module,args,out:instance.append(out.detach()))]
            torch.cuda.synchronize();t0=time.monotonic()
            try:
                torch.backends.cudnn.allow_tf32=False
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):features,start=model.aggregator(images)
                features=[x if j in (4,11,17,23) else None for j,x in enumerate(features)]
                torch.backends.cudnn.allow_tf32=True
                with torch.no_grad():original=model.forward_cached(images,features,start,coupling='off')
            finally:
                for hook in hooks:hook.remove()
            assert len(memory)==len(instance)==1
            h,w=images.shape[-2:];native_memory=memory[0].transpose(1,2).reshape(1,256,h//head.patch_size,w//head.patch_size);tokens=instance[0]
            with torch.no_grad():
                original_det=basehead(tokens);maps=F.interpolate(native_memory.half().float(),(48,48),mode='bilinear',align_corners=False)
                roi,glob=sample_predicted_roi(maps,original_det['instance_boxes_cxcywh'])
                d=det(tokens,roi.half().float(),glob.half().float(),original_det['instance_boxes_cxcywh'],original_det['instance_objectness_logits'])
                p=pose(tokens,native_memory,d['instance_boxes_cxcywh'])
                _,prob=pose.spatial_readout(p['tokens'].reshape(6,23,256),p['coarse'].reshape(6,23,2),maps.expand(6,-1,-1,-1),return_probability=True,predicted_boxes=d['instance_boxes_cxcywh'].reshape(6,4),detach_pose_tokens=False)
            torch.cuda.synchronize();elapsed=time.monotonic()-t0
            # All comparisons below are after fresh inference and never feed it.
            rec=native[data['image_indices'][i]];assert item['source_hashes']==rec['source_hashes'];assert fingerprint(ROOT/rec['path'])==rec['sha256'];gold=torch.load(ROOT/rec['path'],map_location='cpu',weights_only=False)
            delta=dict(memory=float((native_memory[0].cpu()-gold['memory']).abs().max()),instance_tokens=float((tokens[0].cpu()-gold['instance_tokens']).abs().max()),
                boxes=float((d['instance_boxes_cxcywh'][0].cpu()-data['boxes'][i]).abs().max()),logits=float((d['instance_objectness_logits'][0].cpu()-data['objectness_logits'][i]).abs().max()),
                pose=float((p['xy'][0].cpu()-data['xy'][i]).abs().max()),probability=float(np.max(np.abs(prob[:,KP].cpu().numpy()-goldprob[maprow[i]]))))
            assert delta['memory']<1e-4 and delta['instance_tokens']<1e-4,delta
            assert max(delta['boxes'],delta['logits'],delta['pose'])<2e-5 and delta['probability']<1e-4,delta
            result=dict(row=i,view=int(m['view']),image_index=data['image_indices'][i],delta=delta,seconds=elapsed,source_hashes=item['source_hashes']);records.append(result)
            live.append(dict(xy=p['xy'][0].cpu().numpy(),boxes=d['instance_boxes_cxcywh'][0].cpu().numpy(),logits=d['instance_objectness_logits'][0].cpu().numpy(),prob=prob[:,KP].cpu().numpy()))
            status('forward',completed=len(records),total=total,elapsed_seconds=time.monotonic()-started);print(json.dumps(result),flush=True)
            del features,original,native_memory,tokens,memory,instance,images,batch,p,d,prob,gold,item
        outputs=np.stack([p['xy'] for p in live]);scores=1/(1+np.exp(-np.stack([p['logits'] for p in live])));valid=(scores>=.5).sum(-1)==1;queries=scores.argmax(-1);pixel=data['pixels'][ids].numpy()
        candidates=np.empty((len(ids),5,5,2));masses=np.empty((len(ids),5,5))
        for v,entry in enumerate(live):
            for ki in range(5):candidates[v,ki],masses[v,ki]=probability_candidates(entry['prob'][queries[v],ki],pixel[v])
        for v,i in enumerate(ids):
            for ki,kp in enumerate(KP):
                q=int(queries[v]);r=rerank_candidates(g['projections'],candidates[:,ki],masses[:,ki],valid,g['physical_views'],v,outputs[v,q,kp]*pixel[v],enabled=True)
                if r['active']:outputs[v,q,kp]=r['xy']/pixel[v]
            delta=float(np.max(np.abs(outputs[v]-goldcandidate[i])));assert delta<2e-5,(i,delta)
            records[-len(ids)+v]['candidate_delta']=delta
            saved[str(data['image_indices'][i])]=dict(candidate_xy=outputs[v],**live[v])
    torch.save(saved,OUT/'fresh_outputs.pt');write_json(OUT/'results.json',dict(records=records,images=len(records),max_candidate_delta=max(r['candidate_delta'] for r in records),elapsed_seconds=time.monotonic()-started,
        script_sha256=fingerprint(Path(__file__)),claim='Engineering replay only; not a new accuracy evaluation or independent dataset. No new architecture promotion.'))
    status('complete',images=len(records));shutil.copy2(OUT/'results.json',DEST/'epipolar_candidate_rgb_audit_v1.json')
    state.update(stage='candidate_rgb_replay_complete',active_job=None,active_pid=None,next='Read fresh RGB parity and epipolar_candidate_stress_v1. Candidate order/duplicate stable but adding views not monotonic; no selection onVAL. Plan independent seed29/43 map replay or TRAIN-only learned candidate gate, preserve baseline/semantics.');state['completed'].append('candidate_rgb_replay');write_json(CYCLE/'state.json',state)


if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise
