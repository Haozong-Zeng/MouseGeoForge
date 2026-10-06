"""Recover existing TRAIN dual contexts and diagnose attainable candidate error.

GT-assisted best-candidate metrics are postforward oracle diagnostics ONLY;
they are not predictions or data annotations and cannot be used in inference.
"""
import datetime as dt,fcntl,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from torch.nn import functional as F
from mousegeoforge.models.decoder_spatial_pose_readout import DecoderSpatialPoseReadout
from mousegeoforge.candidate_inputs import local_candidate_inputs
from mousegeoforge.candidate_selector_features import build_selector_inputs,KP
from mousegeoforge.detection_readout_loss import match_detection
from mousegeoforge.pilot_metrics import box_iou
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';OLD=ROOT/'output/core23_spatial_cycle_20261004';OUT=CYCLE/'dual_train_candidate_audit_v1'


@torch.no_grad()
def main():
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='dual_train_candidate_audit_running',active_job=OUT.name,active_pid=os.getpid());write_json(CYCLE/'state.json',state)
    source=CYCLE/'singleview_fusion_inputs_seed17_v1/train.pt';d=torch.load(source,map_location='cpu',weights_only=False)
    sm=json.loads((source.parent/'manifest.json').read_text());assert fingerprint(source)==next(x['sha256'] for x in sm['parts'] if x['role']=='train')
    ids=[i for i,m in enumerate(d['metadata']) if m['expected_instances']>1];assert len(ids)==42
    records={r['image_index']:r for r in json.loads((OLD/'single_view_train_context_v1/manifest.json').read_text())['records']}
    posepath=OLD/'full_context_pose_control_v1/full_seed17_latest.pt';assert fingerprint(posepath)==sm['pose_sha256'];pose=DecoderSpatialPoseReadout().cuda().eval();pose.load_state_dict(torch.load(posepath,map_location='cpu',weights_only=False)['state'])
    features=[];native_hashes={};replay=0.
    for done,i in enumerate(ids,1):
        assert dt.datetime.now(dt.timezone.utc)<deadline
        m=d['metadata'][i];assert m['record']['role']=='train' and m['record'].get('session') not in ['GR2','UR2','UR3']
        r=records[int(d['image_indices'][i])];assert r['metadata']==m and fingerprint(ROOT/r['path'])==r['sha256'];native_hashes[r['path']]=r['sha256'];raw=torch.load(ROOT/r['path'],map_location='cpu',weights_only=False)
        maps=F.interpolate(raw['memory'][None].cuda().half().float(),(48,48),mode='bilinear',align_corners=False)
        xy,prob=pose.spatial_readout(d['tokens'][i].cuda(),d['coarse'][i].cuda(),maps.expand(6,-1,-1,-1),return_probability=True,predicted_boxes=d['boxes'][i].cuda(),detach_pose_tokens=False)
        delta=float((xy.cpu()-d['xy'][i]).abs().max());replay=max(replay,delta);assert delta<2e-5
        local=local_candidate_inputs(prob[:,KP].cpu().numpy(),d['xy'][i][:,KP].numpy(),maps);pred={k:v[None] for k,v in local.items()};pred.update(xy=d['xy'][i:i+1],tokens=d['tokens'][i:i+1][:,:,KP],objectness_logits=d['objectness_logits'][i:i+1],probability_entropy=d['entropy'][i:i+1][:,:,KP],probability_variance=d['variance'][i:i+1][:,:,KP])
        feat=build_selector_inputs(pred,np.zeros((1,3,4)),[0],d['pixels'][i].numpy());features.append({k:v[0] for k,v in feat.items()})
        write_json(OUT/'status.json',dict(stage='extracting',pid=os.getpid(),completed=done,total=len(ids)))
    packed={k:d[k][ids] for k in ['gt_xy','point_valid','gt_boxes','box_valid','instance_valid','pixels']};packed.update(base_xy=d['xy'][ids],boxes=d['boxes'][ids],logits=d['objectness_logits'][ids],inputs={k:torch.stack([f[k] for f in features]) for k in features[0]},metadata=[dict(scene=d['metadata'][i]['record']['scene'],view=d['metadata'][i]['view'],kind='synthetic',role='train',source_row=i,image_index=int(d['image_indices'][i])) for i in ids]);packed['inputs']['local']=packed['inputs']['local'].half();torch.save(packed,OUT/'train_inputs.pt')
    del pose;torch.cuda.empty_cache()
    datasets=dict(train=packed,validation=torch.load(CYCLE/'dual_unary_diagnostic_v1/inputs.pt',map_location='cpu',weights_only=False));allresults={}
    assert not ({m['scene'] for m in datasets['train']['metadata']}&{m['scene'] for m in datasets['validation']['metadata']})
    for role,data in datasets.items():
        matches=match_detection(dict(instance_boxes_cxcywh=data['boxes'],instance_objectness_logits=data['logits']),data['gt_boxes'],data['box_valid']&data['instance_valid']);points=[];det_tp=0
        for i,(qs,ts) in enumerate(matches):
            for qq,tt in zip(qs,ts):
                q,t=int(qq),int(tt);conf=bool(data['logits'][i,q].sigmoid()>=.5);hit=bool(conf and box_iou(data['boxes'][i,q:q+1],data['gt_boxes'][i,t:t+1])[0,0]>=.5);det_tp+=int(hit)
                for ki,k in enumerate(KP):
                    if not data['point_valid'][i,t,k]:continue
                    e=((data['inputs']['candidate_xy'][i,q,ki]-data['gt_xy'][i,t,k])*data['pixels'][i]).norm(dim=-1)
                    allowed=(data['inputs']['scalars'][i,q,ki,:,0]*20+2)>.001;allowed[0]=True
                    if not conf:allowed[1:]=False
                    oracle=float(e.min()) if conf else float(e[0]);bounded=float(e.masked_fill(~allowed,float('inf')).min())
                    points.append(dict(row=i,point=k+1,query=q,target=t,detector_hit=hit,confident=conf,baseline_error=float(e[0]),oracle_all_candidates=oracle,oracle_bounded_prior=bounded,oracle_bounded_choice=int(e.masked_fill(~allowed,float('inf')).argmin()),candidate_errors=e.tolist(),admissible=allowed.tolist()))
        groups={}
        for name,selected in [('all',points),('detector_hit',[p for p in points if p['detector_hit']]),('detector_miss',[p for p in points if not p['detector_hit']])]:
            groups[name]=dict(points=len(selected),**{k:float(np.mean([p[k] for p in selected])) if selected else None for k in ['baseline_error','oracle_all_candidates','oracle_bounded_prior']},bounded_oracle_pck32=float(np.mean([p['oracle_bounded_prior']<=32 for p in selected])) if selected else None)
        allresults[role]=dict(images=len(data['base_xy']),scenes=len({m['scene'] for m in data['metadata']}),gt=int((data['box_valid']&data['instance_valid']).sum()),pred=int((data['logits'].sigmoid()>=.5).sum()),tp=det_tp,groups=groups)
        write_json(OUT/f'{role}_oracle_diagnostic_points.json',points);print(role,allresults[role],flush=True)
    write_json(OUT/'results.json',dict(results=allresults,source_sha256=fingerprint(source),native_sha256=native_hashes,script_sha256=fingerprint(Path(__file__)),train_inputs_sha256=fingerprint(OUT/'train_inputs.pt'),max_probability_replay_difference=replay,protocol='Only existing TRAIN42 dual V1 contexts recovered; no new frames/splits/annotations. All6query23 probability reconstructed without GT; camera context absent. Oracle uses GT only after predictions to assess candidate coverage, not achieved inference. bounded oracle allows baseline and relative mass with logmass+2>.001 under predeclared prior head; this is candidate-set optimistic error, not necessarily jointly learnable. Confidence gating remains prediction-only. No training or model selection. VAL diagnostic never used to fit.',production_changed=False))
    write_json(OUT/'status.json',dict(stage='complete',pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat()));state.update(stage='dual_train_candidate_audit_complete',active_job=None,active_pid=None,next='Inspect dual_train_candidate_audit_v1 results and boundedprior repeat review; plan one controlled inclusion of existing dualTRAIN if correct candidates reachable. No new architecture/VALthresholds. Save handoff.');state['completed'].append('dual_train_context42_and_reachability_audit');write_json(CYCLE/'state.json',state)


if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise
