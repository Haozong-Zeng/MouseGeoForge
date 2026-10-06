"""Frozen real V1 baseline tokens and descriptive uncertainty, no fake 3D/labels."""
import collections,datetime as dt,fcntl,json,math,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from torch.nn import functional as F
from scipy.stats import spearmanr
from mousegeoforge.models.decoder_spatial_pose_readout import DecoderSpatialPoseReadout
from mousegeoforge.models.spatial_detection_readout import SpatialDetectionReadout
from mousegeoforge.detection_readout_loss import match_detection
from run_depth_pose_ablation import fingerprint,write_json
OLD=ROOT/'output/core23_spatial_cycle_20261004';CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';OUT=CYCLE/'singleview_fusion_inputs_seed17_v1'

def main():
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists() and torch.cuda.is_available();OUT.mkdir();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='preparing_singleview_fusion_tokens',active_job=OUT.name,active_pid=os.getpid());write_json(CYCLE/'state.json',state)
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    status('loading');rows=[];source=OLD/'detection_features_v1/manifest.json'
    for rec in json.loads(source.read_text())['records']:
        assert fingerprint(ROOT/rec['path'])==rec['sha256'];rows.extend(torch.load(ROOT/rec['path'],map_location='cpu',weights_only=False)['observations'])
    cm=json.loads((OLD/'single_detection_context_v1/manifest.json').read_text());assert fingerprint(ROOT/cm['path'])==cm['sha256'];detfeatures=torch.load(ROOT/cm['path'],map_location='cpu',weights_only=False)
    assert detfeatures['metadata']==[o['metadata'] for o in rows]
    detpath=OLD/'detection_view_context_control_v1/mixed_seed17_latest.pt';posepath=OLD/'full_context_pose_control_v1/full_seed17_latest.pt'
    det=SpatialDetectionReadout().cuda().eval().requires_grad_(False);det.load_state_dict(torch.load(detpath,map_location='cuda',weights_only=False)['state'],strict=True)
    pose=DecoderSpatialPoseReadout().cuda().eval().requires_grad_(False);pose.load_state_dict(torch.load(posepath,map_location='cuda',weights_only=False)['state'],strict=True)
    with np.load(OLD/'detection_view_context_control_v1/predictions.npz') as a:dg={k:a[f'mixed_seed17_single__{k}'].copy() for k in ['instance_boxes_cxcywh','instance_objectness_logits']}
    with np.load(OLD/'full_context_pose_control_v1/predictions.npz') as a:pg={int(i):a['full_seed17_single'][j].copy() for j,i in enumerate(a['image_indices'])}
    maxdet=maxpose=maxprob=0.;goldcases=0;parts=[];error_rows=[];scene_roles={};groups_manifest=[]
    axis=(torch.arange(48,device='cuda',dtype=torch.float32)+.5)/48;y,x=torch.meshgrid(axis,axis,indexing='ij');grid=torch.stack([x,y],-1).reshape(2304,2)
    for role in ['train','validation']:
        mp=OLD/f'single_view_{role}_context_v1/manifest.json';records=json.loads(mp.read_text())['records'];groups=collections.defaultdict(list);collected=[]
        for r in records:
            i=r['image_index'];m=rows[i]['metadata'];assert m==r['metadata'] and m['record']['role']==role and m['detection_complete'] and detfeatures['available'][i]
            assert m['record'].get('session') not in ['GR2','UR2','UR3'];scene=m['record']['scene'];assert scene_roles.setdefault(scene,role)==role
            groups[tuple(m['record']['image_hw'])].append(r)
        completed=0
        for recs in groups.values():
            for start in range(0,len(recs),8):
                assert dt.datetime.now(dt.timezone.utc)<deadline;batch=recs[start:start+8];ids=[r['image_index'] for r in batch];native=[];inst=[]
                for rec,i in zip(batch,ids):
                    assert fingerprint(ROOT/rec['path'])==rec['sha256'];raw=torch.load(ROOT/rec['path'],map_location='cpu',weights_only=False);assert raw['metadata']==rows[i]['metadata'];native.append(raw['memory']);inst.append(raw['instance_tokens'])
                memory=torch.stack(native).cuda();instance=torch.stack(inst).cuda();features={k:detfeatures[k][ids].float().cuda() for k in ['instance_tokens','roi','global_memory','original_boxes','original_objectness']}
                torch.testing.assert_close(instance,features['instance_tokens'],atol=0,rtol=0)
                with torch.no_grad():
                    d=det(**dict(tokens=features['instance_tokens'],roi=features['roi'],global_memory=features['global_memory'],original_boxes=features['original_boxes'],original_logits=features['original_objectness']))
                    for k in d:
                        delta=float((d[k].cpu()-torch.tensor(dg[k][ids])).abs().max());assert delta<2e-5;maxdet=max(maxdet,delta)
                    p=pose(instance,memory,d['instance_boxes_cxcywh']);b=len(ids);maps=F.interpolate(memory.half().float(),(48,48),mode='bilinear',align_corners=False);expanded=maps[:,None].expand(-1,6,-1,-1,-1).reshape(b*6,256,48,48)
                    xy,prob=pose.spatial_readout(p['tokens'].reshape(b*6,23,256),p['coarse'].reshape(b*6,23,2),expanded,return_probability=True,predicted_boxes=d['instance_boxes_cxcywh'].reshape(b*6,4),detach_pose_tokens=False)
                    delta=float((xy.reshape(b,6,23,2)-p['xy']).abs().max());assert delta<1e-7;maxprob=max(maxprob,delta);prob=prob.reshape(b,6,23,2304)
                    entropy=-(prob*prob.clamp_min(1e-12).log()).sum(-1)/math.log(2304);variance=(prob[...,None]*(grid[None,None,None]-p['xy'][...,None,:]).square()).sum(-2);peak=prob.max(-1)
                    extra=dict(tokens=p['tokens'],xy=p['xy'],coarse=p['coarse'],boxes=d['instance_boxes_cxcywh'],objectness_logits=d['instance_objectness_logits'],entropy=entropy,variance=variance,peak_probability=peak.values,peak_xy=grid[peak.indices])
                boxes=torch.stack([rows[i]['boxes'] for i in ids]).cuda();valid=torch.stack([rows[i]['box_valid']&rows[i]['instance_valid'] for i in ids]).cuda();matches=match_detection(d,boxes,valid)
                for j,i in enumerate(ids):
                    m=rows[i]['metadata']
                    if i in pg:
                        delta=float((p['xy'][j].cpu()-torch.tensor(pg[i])).abs().max());assert delta<2e-5;maxpose=max(maxpose,delta);goldcases+=1
                    entry=dict(image_index=i,metadata=m,**{k:v[j].cpu() for k,v in extra.items()},**{k:rows[i][k] for k in ['gt_xy','point_valid','boxes','box_valid','instance_valid','pixels'] if k!='boxes'},gt_boxes=rows[i]['boxes'])
                    collected.append(entry)
                    for q,t in zip(*matches[j]):
                        point_valid=rows[i]['point_valid'][t].clone()
                        if m['expected_instances']>1:point_valid[7:14]=False
                        err=((p['xy'][j,q].cpu()-rows[i]['gt_xy'][t])*rows[i]['pixels']).norm(dim=-1)
                        for k in torch.where(point_valid)[0].tolist():
                            if m['record']['kind']=='real' and k not in [0,5,6,14,15,16,17]:continue
                            error_rows.append(dict(role=role,kind='real' if m['record']['kind']=='real' else 'synthetic',view=int(m['view']),image_index=i,query=int(q),target=int(t),point=k+1,error=float(err[k]),entropy=float(entropy[j,q,k]),variance_px=float((variance[j,q,k]*rows[i]['pixels'].cuda().square()).sum().sqrt()),score=float(d['instance_objectness_logits'][j,q].sigmoid())))
                completed+=len(ids);status('extracting',role=role,completed=completed,total=len(records))
        collected.sort(key=lambda e:e['image_index']);keys=['tokens','xy','coarse','boxes','objectness_logits','entropy','variance','peak_probability','peak_xy','gt_xy','point_valid','gt_boxes','box_valid','instance_valid','pixels'];packed={k:torch.stack([e[k] for e in collected]) for k in keys};packed.update(image_indices=[e['image_index'] for e in collected],metadata=[e['metadata'] for e in collected]);path=OUT/f'{role}.pt';torch.save(packed,path)
        scenes=collections.defaultdict(list)
        for j,e in enumerate(collected):scenes[e['metadata']['record']['scene']].append(j)
        groups_manifest.append(dict(role=role,scenes=[dict(scene=s,row_indices=js,physical_views=[int(collected[j]['metadata']['view']) for j in js]) for s,js in scenes.items()]));parts.append(dict(role=role,path=path.relative_to(ROOT).as_posix(),sha256=fingerprint(path),images=len(collected),source_manifest_sha256=fingerprint(mp)))
    assert goldcases==112;correlations=[]
    for role in ['train','validation']:
        for kind in ['real','synthetic']:
            for part in ['reliable','bottom_feet','back']:
                selected=[r for r in error_rows if r['role']==role and r['kind']==kind and (r['point'] in [1,15,16,17,18] if part=='reliable' else r['view']==0 and r['point'] in [15,16,17,18] if part=='bottom_feet' else r['point'] in [6,7])]
                if not selected:continue
                correlations.append(dict(role=role,kind=kind,part=part,points=len(selected),error_mean=float(np.mean([r['error'] for r in selected])),entropy_error_spearman=float(spearmanr([r['entropy'] for r in selected],[r['error'] for r in selected]).statistic),variance_error_spearman=float(spearmanr([r['variance_px'] for r in selected],[r['error'] for r in selected]).statistic)))
    write_json(OUT/'uncertainty_audit.json',dict(correlations=correlations,observations=error_rows,warning='Entropy/variance are descriptive uncertainty, NOT calibrated confidence or visibility. Error uses postforward GTbox assignment. Do not choose confidence thresholds on locked data or fabricate visibility.'))
    write_json(OUT/'scene_groups.json',groups_manifest);write_json(OUT/'manifest.json',dict(parts=parts,detector_sha256=fingerprint(detpath),pose_sha256=fingerprint(posepath),script_sha256=fingerprint(Path(__file__)),golden_cases=goldcases,max_detector_delta=maxdet,max_pose_delta=maxpose,max_probability_replay_delta=maxprob,contract='All1350pool eligible1335 genuineV1 outputs, seed17 fixed. Same immutable baseline detector/pose as priorfullpose control. Float32 semantic tokens, predicted boxes and descriptivemap statistics. GT stored separately for loss/eval only. Role/scenes grouped, physical views retained. No crossview query correspondence established. No real curves/3D generated.',production_changed=False));status('complete');state.update(stage='singleview_fusion_inputs_ready',active_job=None,active_pid=None,updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),next='Inspect uncertainty correlation and actual badfeet. Use TRAIN only for any calibration, not VALLABELS inforward. Implement opt-in confidence-gated semantic residual with fixed singleviewbaseline, zeroinit/off/V1/no-support tests. Matchself-only parametercontrol. No hard query-ID correspondence acrosscameras; restrict initialsingle-instance experiment or use prediction-only association anddual abstention. Baseline detector unchanged because negative04 hurts camArecall.');state['completed'].append('singleview_baseline_tokens_1335_and_uncertainty_audit');write_json(CYCLE/'state.json',state);print(json.dumps(dict(parts=parts,correlations=correlations,golden_cases=goldcases,max_pose_delta=maxpose)))

if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise
