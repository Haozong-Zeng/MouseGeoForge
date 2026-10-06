"""Frozen unary selector on held-out development dual-instance images."""
import collections
import datetime as dt
import fcntl
import json
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from torch.nn import functional as F
from mousegeoforge.models.decoder_spatial_pose_readout import DecoderSpatialPoseReadout
from mousegeoforge.models.local_candidate_selector import LocalCandidateSelector
from mousegeoforge.candidate_inputs import local_candidate_inputs
from mousegeoforge.candidate_selector_features import build_selector_inputs,KP
from mousegeoforge.detection_readout_loss import match_detection
from mousegeoforge.pilot_metrics import box_iou
from audit_independent_unary_20261005 import metrics,predict
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005'
OLD=ROOT/'output/core23_spatial_cycle_20261004'
OUT=CYCLE/'dual_unary_diagnostic_v1'


@torch.no_grad()
def main():
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'))
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='dual_unary_diagnostic_running',active_job=OUT.name,active_pid=os.getpid());write_json(CYCLE/'state.json',state)
    def status(stage,**extra):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**extra))
    status('extracting')
    source=CYCLE/'singleview_fusion_inputs_seed17_v1/validation.pt';data=torch.load(source,map_location='cpu',weights_only=False)
    source_manifest=json.loads((source.parent/'manifest.json').read_text());assert fingerprint(source)==next(r['sha256'] for r in source_manifest['parts'] if r['role']=='validation')
    records={r['image_index']:r for r in json.loads((OLD/'single_view_validation_context_v1/manifest.json').read_text())['records']}
    ids=[i for i,m in enumerate(data['metadata']) if m['expected_instances']>1]
    assert len(ids)==32 and all(data['metadata'][i]['record']['role']=='validation' for i in ids)
    posepath=OLD/'full_context_pose_control_v1/full_seed17_latest.pt'
    assert fingerprint(posepath)==source_manifest['pose_sha256']
    pose=DecoderSpatialPoseReadout().cuda().eval();pose.load_state_dict(torch.load(posepath,map_location='cpu',weights_only=False)['state'])
    features=[];replay=0.;native_hashes={}
    for done,i in enumerate(ids,1):
        assert dt.datetime.now(dt.timezone.utc)<deadline
        r=records[int(data['image_indices'][i])];assert r['metadata']==data['metadata'][i]
        assert fingerprint(ROOT/r['path'])==r['sha256'];native_hashes[r['path']]=r['sha256']
        raw=torch.load(ROOT/r['path'],map_location='cpu',weights_only=False)
        maps=F.interpolate(raw['memory'][None].cuda().half().float(),(48,48),mode='bilinear',align_corners=False)
        # All six queries and 23 semantics are processed, before any GT matching.
        xy,prob=pose.spatial_readout(data['tokens'][i].cuda(),data['coarse'][i].cuda(),maps.expand(6,-1,-1,-1),return_probability=True,predicted_boxes=data['boxes'][i].cuda(),detach_pose_tokens=False)
        delta=float((xy.cpu()-data['xy'][i]).abs().max());replay=max(replay,delta);assert delta<2e-5,delta
        local=local_candidate_inputs(prob[:,KP].cpu().numpy(),data['xy'][i][:,KP].numpy(),maps)
        predictions={k:v[None] for k,v in local.items()}
        predictions.update(xy=data['xy'][i:i+1],tokens=data['tokens'][i:i+1][:,:,KP],objectness_logits=data['objectness_logits'][i:i+1],probability_entropy=data['entropy'][i:i+1][:,:,KP],probability_variance=data['variance'][i:i+1][:,:,KP])
        # V1/no camera context: dummy projection is unused; geometry must stay zero.
        feat=build_selector_inputs(predictions,np.zeros((1,3,4)),[0],data['pixels'][i].numpy())
        assert not feat['eligible'].any() and not feat['geometry'].any()
        features.append({k:v[0] for k,v in feat.items()})
        status('extracting',completed=done,total=len(ids))
    del pose;torch.cuda.empty_cache()
    packed={k:data[k][ids] for k in ['gt_xy','point_valid','gt_boxes','box_valid','instance_valid','pixels']}
    packed.update(base_xy=data['xy'][ids],boxes=data['boxes'][ids],logits=data['objectness_logits'][ids],inputs={k:torch.stack([f[k] for f in features]) for k in features[0]},metadata=[dict(scene=data['metadata'][i]['record']['scene'],view=data['metadata'][i]['view'],kind='synthetic',role='validation',source_row=i,image_index=int(data['image_indices'][i])) for i in ids])
    packed['inputs']['local']=packed['inputs']['local'].half();torch.save(packed,OUT/'inputs.pt')
    checkpoint=CYCLE/'local_candidate_selector_pilot_v1/unary_latest.pt';selector=LocalCandidateSelector('unary').eval();selector.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['state'])
    confident=packed['logits'].sigmoid()>=.5;forward={}
    for arm,gate in [('unique_query',confident&(confident.sum(-1,keepdim=True)==1)),('per_query',confident)]:
        inputs=dict(packed['inputs'],eligible=gate[...,None].expand(-1,-1,5))
        forward[arm]=predict(selector,inputs)
    # All forward complete. Labels below are evaluation-only.
    results=[];arrays={};association=[]
    for arm in ['baseline','unique_query','per_query']:
        xy=packed['base_xy'].clone();active=torch.zeros_like(packed['inputs']['eligible']);choices=torch.zeros_like(active,dtype=torch.long)
        if arm!='baseline':
            out=forward[arm];xy[:,:,KP]=out['xy'];active,choices=out['active'],out['choice']
        values,points=metrics(packed,xy,active,choices)
        for p in points:
            i,q,t,k=p['row'],p['query'],p['target'],p['point']-1;other=1-t
            if packed['point_valid'][i,other,k]:
                other_error=float(((xy[i,q,k]-packed['gt_xy'][i,other,k])*packed['pixels'][i]).norm())
                old_other=float(((packed['base_xy'][i,q,k]-packed['gt_xy'][i,other,k])*packed['pixels'][i]).norm())
                p['other_instance_same_point_error']=other_error
                p['other_closer_than_matched']=other_error<p['error']
                p['new_other_closer']=other_error<p['error'] and not old_other<p['baseline_error']
        association.append(dict(arm=arm,eligible_label_pairs=sum('other_closer_than_matched' in p for p in points),other_closer=sum(p.get('other_closer_than_matched',False) for p in points),new_other_closer=sum(p.get('new_other_closer',False) for p in points)))
        results.append(dict(arm=arm,metrics=values,changed_all_queries=int(active.sum())))
        arrays[arm]=xy.numpy();arrays[arm+'_active']=active.numpy();write_json(OUT/f'{arm}_points.json',points)
        print(arm,values['synthetic/reliable'],flush=True)
    matches=match_detection(dict(instance_boxes_cxcywh=packed['boxes'],instance_objectness_logits=packed['logits']),packed['gt_boxes'],packed['box_valid']&packed['instance_valid'])
    counts=dict(images=len(ids),gt=int((packed['box_valid']&packed['instance_valid']).sum()),pred=int(confident.sum()),tp=0,pred_count_histogram=dict(collections.Counter(map(int,confident.sum(-1)))))
    for i,(qs,ts) in enumerate(matches):
        for q,t in zip(qs,ts):counts['tp']+=int(confident[i,q] and box_iou(packed['boxes'][i,q:q+1],packed['gt_boxes'][i,t:t+1])[0,0]>=.5)
    report=dict(results=results,unchanged_detector_counts=counts,association_diagnostic=association,max_probability_replay_difference=replay,source_sha256=fingerprint(source),checkpoint_sha256=fingerprint(checkpoint),pose_sha256=fingerprint(posepath),native_sha256=native_hashes,script_sha256=fingerprint(Path(__file__)),training=False,production_changed=False,protocol='Frozen single-cohort-trained unary; synthetic dual development diagnostic. Cohort expected_instances only selects offline evaluation set, never forward. All 6 predicted queries, 23 probabilities reconstructed from immutable genuine V1 inputs; only nose/fourfoot readout updated. Unique query versus every score>=.5 query; no GT count, box, matching, camera correspondence, or visibility in forward. Labels only after all forward. Other-instance-nearer metric is a diagnostic, not identity-swap ground truth. GTbox-conditioned pose, reused development not external test. No new training or threshold search.')
    np.savez_compressed(OUT/'predictions.npz',**arrays);write_json(OUT/'results.json',report);status('complete')
    state.update(stage='dual_unary_diagnostic_complete',active_job=None,active_pid=None,next='Read dual_unary_diagnostic_v1 results and inspect raw allquery overlays before updating handoff. No model promotion.');state['completed'].append('dual_unary_frozen_diagnostic');write_json(CYCLE/'state.json',state)


if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise
