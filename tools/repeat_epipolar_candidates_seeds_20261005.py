"""Fixed candidate rule replicated on existing seed29/43 detector+pose heads.

No new training or hyperparameter selection. Single-view caches are immutable;
all138 pose tokens are freshly decoded for each seed. Seed17 reference is the
already completed pilot, not rerun or silently mixed with other checkpoints.
"""
import collections,datetime as dt,fcntl,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from torch.nn import functional as F
from mousegeoforge.models.decoder_spatial_pose_readout import DecoderSpatialPoseReadout
from mousegeoforge.models.spatial_detection_readout import SpatialDetectionReadout
from mousegeoforge.epipolar_candidate_readout import probability_candidates,rerank_candidates
from evaluate_epipolar_candidates_20261005 import evaluate,KP,ARMS
from run_depth_pose_ablation import fingerprint,write_json
OLD=ROOT/'output/core23_spatial_cycle_20261004';CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';SOURCE=CYCLE/'singleview_fusion_inputs_seed17_v1';OUT=CYCLE/'epipolar_candidates_seed_repeats_v1'


def main():
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'))
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists() and torch.cuda.is_available();OUT.mkdir();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='replicating_candidate_seeds',active_job=OUT.name,active_pid=os.getpid());write_json(CYCLE/'state.json',state)
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    status('loading');manifest=json.loads((SOURCE/'manifest.json').read_text());cm=json.loads((CYCLE/'camera_geometry_audit_v2/results.json').read_text())
    dm=json.loads((OLD/'single_detection_context_v1/manifest.json').read_text());assert fingerprint(ROOT/dm['path'])==dm['sha256'];features=torch.load(ROOT/dm['path'],map_location='cpu',weights_only=False)
    with np.load(OLD/'detection_view_context_control_v1/predictions.npz') as a:dg={s:{k:a[f'mixed_seed{s}_single__{k}'].copy() for k in ['instance_boxes_cxcywh','instance_objectness_logits']} for s in [29,43]}
    with np.load(OLD/'full_context_pose_control_v1/predictions.npz') as a:pg={s:{int(i):a[f'full_seed{s}_single'][j].copy() for j,i in enumerate(a['image_indices'])} for s in [29,43]}
    results=[];hashes={};audits=[]
    for seed in [29,43]:
        detpath=OLD/f'detection_view_context_control_v1/mixed_seed{seed}_latest.pt';posepath=OLD/f'full_context_pose_control_v1/full_seed{seed}_latest.pt';hashes[seed]=dict(detector=fingerprint(detpath),pose=fingerprint(posepath))
        det=SpatialDetectionReadout().cuda().eval().requires_grad_(False);det.load_state_dict(torch.load(detpath,map_location='cuda',weights_only=False)['state'],strict=True)
        pose=DecoderSpatialPoseReadout().cuda().eval().requires_grad_(False);pose.load_state_dict(torch.load(posepath,map_location='cuda',weights_only=False)['state'],strict=True)
        for part in manifest['parts']:
            role=part['role'];assert fingerprint(ROOT/part['path'])==part['sha256'];data=torch.load(ROOT/part['path'],map_location='cpu',weights_only=False)
            cp=next(r for r in cm['camera_artifacts'] if r['role']==role);assert fingerprint(ROOT/cp['path'])==cp['sha256'];groups=torch.load(ROOT/cp['path'],map_location='cpu',weights_only=False)
            ids=sorted(i for g in groups for i in g['row_indices']);local={i:j for j,i in enumerate(ids)};cohort=np.zeros(len(data['xy']),bool);cohort[ids]=True
            native={r['image_index']:r for r in json.loads((OLD/f'single_view_{role}_context_v1/manifest.json').read_text())['records']};buckets=collections.defaultdict(list)
            for i in ids:
                assert data['metadata'][i]['record'].get('session') not in ['GR2','UR2','UR3'];buckets[tuple(data['metadata'][i]['record']['image_hw'])].append(i)
            probability=np.empty((len(ids),6,5,48,48),np.float32);boxes=data['boxes'].clone();logits=data['objectness_logits'].clone();xy=data['xy'].clone();maximumdet=maximumpose=0.;goldcases=0;done=0
            for rows in buckets.values():
                for start in range(0,len(rows),8):
                    assert dt.datetime.now(dt.timezone.utc)<deadline
                    batch=rows[start:start+8];globalids=[data['image_indices'][i] for i in batch];memory=[];instance=[]
                    for i,gi in zip(batch,globalids):
                        rec=native[gi];assert fingerprint(ROOT/rec['path'])==rec['sha256'];raw=torch.load(ROOT/rec['path'],map_location='cpu',weights_only=False);assert raw['metadata']==data['metadata'][i]==features['metadata'][gi]
                        memory.append(raw['memory']);instance.append(raw['instance_tokens'])
                    memory=torch.stack(memory).cuda();instance=torch.stack(instance).cuda();n=len(batch)
                    inputs={k:features[k][globalids].float().cuda() for k in ['instance_tokens','roi','global_memory','original_boxes','original_objectness']};torch.testing.assert_close(instance,inputs['instance_tokens'],rtol=0,atol=0)
                    with torch.no_grad():
                        d=det(instance,inputs['roi'],inputs['global_memory'],inputs['original_boxes'],inputs['original_objectness']);p=pose(instance,memory,d['instance_boxes_cxcywh'])
                        maps=F.interpolate(memory.half().float(),(48,48),mode='bilinear',align_corners=False);expanded=maps[:,None].expand(-1,6,-1,-1,-1).reshape(n*6,256,48,48)
                        replay,prob=pose.spatial_readout(p['tokens'].reshape(n*6,23,256),p['coarse'].reshape(n*6,23,2),expanded,return_probability=True,predicted_boxes=d['instance_boxes_cxcywh'].reshape(n*6,4),detach_pose_tokens=False)
                    assert float((replay.reshape(n,6,23,2)-p['xy']).abs().max())<1e-7
                    for k in d:
                        delta=float(np.max(np.abs(d[k].cpu().numpy()-dg[seed][k][globalids])));assert delta<2e-5;maximumdet=max(maximumdet,delta)
                    boxes[batch]=d['instance_boxes_cxcywh'].cpu();logits[batch]=d['instance_objectness_logits'].cpu();xy[batch]=p['xy'].cpu();prob=prob.reshape(n,6,23,48,48)[:,:,KP].cpu().numpy()
                    for j,(i,gi) in enumerate(zip(batch,globalids)):
                        probability[local[i]]=prob[j]
                        if gi in pg[seed]:
                            delta=float(np.max(np.abs(xy[i].numpy()-pg[seed][gi])));assert delta<2e-5;maximumpose=max(maximumpose,delta);goldcases+=1
                    done+=len(batch);status('decoding',seed=seed,role=role,done=done,total=len(ids))
            # Replace only runtime prediction fields, never labels or source files.
            data=dict(data,boxes=boxes,objectness_logits=logits,xy=xy);base=xy.numpy();pred={arm:base.copy() for arm in ARMS};active={arm:np.zeros(base.shape[:-1],bool) for arm in ARMS}
            for g in groups:
                assert dt.datetime.now(dt.timezone.utc)<deadline
                rows=g['row_indices'];score=logits[rows].sigmoid().numpy();valid=(score>=.5).sum(-1)==1;queries=score.argmax(-1);pixel=data['pixels'][rows].numpy();candidates=np.empty((len(rows),5,5,2));masses=np.empty((len(rows),5,5))
                for v,i in enumerate(rows):
                    for ki in range(5):candidates[v,ki],masses[v,ki]=probability_candidates(probability[local[i],queries[v],ki],pixel[v])
                for v,i in enumerate(rows):
                    for ki,kp in enumerate(KP):
                        q=int(queries[v])
                        for arm,weight in [('unary_local',0.),('epipolar_local',1.)]:
                            r=rerank_candidates(g['projections'],candidates[:,ki],masses[:,ki],valid,g['physical_views'],v,base[i,q,kp]*pixel[v],enabled=True,geometry_weight=weight)
                            if r['active']:pred[arm][i,q,kp]=r['xy']/pixel[v];active[arm][i,q,kp]=True
            metrics,points=evaluate(data,pred,cohort,active);results.append(dict(seed=seed,role=role,metrics=metrics));audits.append(dict(seed=seed,role=role,images=len(ids),max_detector_delta=maximumdet,max_pose_delta=maximumpose,pose_golden_cases=goldcases))
            np.savez_compressed(OUT/f'seed{seed}_{role}_predictions.npz',image_indices=data['image_indices'],cohort=cohort,boxes=boxes.numpy(),logits=logits.numpy(),**{a+'_xy':pred[a] for a in ARMS},**{a+'_active':active[a] for a in ARMS});write_json(OUT/f'seed{seed}_{role}_point_errors.json',points)
            if role=='validation':np.savez_compressed(OUT/f'seed{seed}_validation_probabilities.npz',probability=probability,row_indices=ids,keypoint_indices=KP)
            write_json(OUT/'partial_results.json',dict(results=results,audits=audits));print(seed,role,{a:metrics[a+'/real/reliable']['error_mean'] for a in ARMS},flush=True)
    assert all(a['pose_golden_cases']==80 for a in audits if a['role']=='validation')
    write_json(OUT/'results.json',dict(results=results,audits=audits,head_sha256=hashes,script_sha256=fingerprint(Path(__file__)),
        reference_seed17='epipolar_candidates_pilot_v1/results.json',protocol='Same fixed48x48 spatial candidates, unary0/geometry1, sigma16,2supports. Seed29/43 immutable mixed800 detector/full800pose. TRAIN+VAL evaluated, no training or thresholdsearch. Out-of-cohort arrays retainseed17 values and MUSTNOTbeevaluated; cohortmask mandatory. Existing development split not external test. No production promotion.'))
    status('complete');state.update(stage='candidate_seed_repeats_complete',active_job=None,active_pid=None,next='Review all3seeds paired baseline/unary/epipolar and per-camera tradeoffs. No automatic promotion; original labels remain untouched.');state['completed'].append('candidate_seed29_43_repeats');write_json(CYCLE/'state.json',state)


if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise
