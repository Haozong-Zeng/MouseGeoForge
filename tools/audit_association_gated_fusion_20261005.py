"""Offline frozen prediction-only association -> grouped candidate refinement."""
import datetime as dt,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from torch.nn import functional as F
from mousegeoforge.models.association_projection import AssociationProjection
from mousegeoforge.models.grouped_candidate_fusion import GroupedCandidateFusion
from mousegeoforge.partial_camera_association import conservative_camera_groups
from audit_composite_candidate_cache_20261005 import load_wrapper
from audit_association_faults_20261005 import choose,match,MODES
from mousegeoforge.pilot_metrics import box_iou
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';OUT=CYCLE/'association_gated_fusion_audit_v1'
SOURCES=dict(train='dual_train_allview_affinity_context_v1',validation='dual_allscene_association_audit_v1')
KP=[0,14,15,16,17]
ARMS=['baseline','local','unsafe_composite','grouped_local','grouped_geometry','grouped_composite']


@torch.no_grad()
def infer_groups(p,views,descriptors,model,cutoff):
    nodes=[]
    for vi in sorted(range(len(views)),key=lambda i:views[i]):
        for q in torch.where(p['objectness_logits'][vi].sigmoid()>=.5)[0].tolist():
            if not torch.isfinite(descriptors[vi,q]).all():continue
            nodes.append(dict(view_index=vi,view=views[vi],query=q))
    if not nodes:return [],nodes,dict(groups=[],edges=[])
    # No annotation keys, identities, GT count or GT boxes enter this function.
    x=torch.stack([descriptors[n['view_index'],n['query']] for n in nodes])
    z=F.normalize(model(x,enabled=True),dim=-1);distance=(1-z@z.T).clamp(0,2).numpy()
    associated=conservative_camera_groups(nodes,distance,cutoff,enabled=True)
    groups=[[(nodes[j]['view'],nodes[j]['query']) for j in group] for group in associated['groups']]
    return groups,nodes,associated


def summarize(rows):
    result={}
    for name,select in [
        ('all_matched',lambda p:True),('confident_iou50',lambda p:p['hit']),
        ('associated',lambda p:p['associated']),('associated_iou50',lambda p:p['associated'] and p['hit']),
        ('retained',lambda p:p['retained']),('retained_iou50',lambda p:p['retained'] and p['hit']),
        ('feet_all',lambda p:p['point']!=1),('feet_associated',lambda p:p['associated'] and p['point']!=1)]:
        ps=[p for p in rows if select(p)];result[name]={}
        for arm in ARMS:
            errors=np.array([p['errors'][arm] for p in ps]);base=np.array([p['errors']['baseline'] for p in ps]);local=np.array([p['errors']['grouped_local'] for p in ps])
            result[name][arm]=dict(points=len(ps),mean=float(errors.mean()) if len(ps) else None,p95=float(np.percentile(errors,95)) if len(ps) else None,
                gated_pck32=float(np.mean([p['hit'] and p['errors'][arm]<=32 for p in ps])) if ps else None,
                harms_vs_original_over16=int(((errors-base)>16).sum()),worst_harm_vs_original=float((errors-base).max()) if ps else None,
                harms_vs_grouped_local_over16=int(((errors-local)>16).sum()),worst_harm_vs_grouped_local=float((errors-local).max()) if ps else None)
    return result


def evaluate(p,d,outputs,matches,associated,retained):
    rows=[]
    for vi,(queries,targets) in enumerate(matches):
        for q,t in zip(queries.tolist(),targets.tolist()):
            iou=float(box_iou(p['boxes'][vi,q:q+1],d['gt_boxes'][vi,t:t+1])[0,0])
            hit=bool(p['objectness_logits'][vi,q].sigmoid()>=.5 and iou>=.5)
            for kp in KP:
                if not d['point_valid'][vi,t,kp]:continue
                errors={name:float(((xy[vi,q,kp]-d['gt_xy'][vi,t,kp])*d['pixels'][vi]).norm()) for name,xy in outputs.items()}
                rows.append(dict(view_index=vi,query=q,target=t,point=kp+1,iou=iou,hit=hit,
                    associated=bool(associated[vi,q,KP.index(kp)]),retained=retained is None or q==retained[vi],errors=errors))
    return rows


@torch.no_grad()
def main():
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2);started=time.monotonic()
    plan=json.loads((CYCLE/'plan.json').read_text());cutoff_time=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));assert dt.datetime.now(dt.timezone.utc)<cutoff_time
    modelpath=CYCLE/'association_projection_pilot_v1/rgb_seed17_step0800.pt';calpath=CYCLE/'learned_partial_association_v1/calibration.json'
    cutoff=json.loads(calpath.read_text())['models']['rgb_seed17_trained']['cutoff']
    metric=AssociationProjection().eval();metric.load_state_dict(torch.load(modelpath,map_location='cpu',weights_only=False)['state'])
    wrapper=load_wrapper();grouped=GroupedCandidateFusion(wrapper)
    cropfile=CYCLE/'predicted_crop_encoder_v1/descriptors.pt';crops=torch.load(cropfile,map_location='cpu',weights_only=False);croplookup={(r['role'],r['group']):r for r in crops}
    expected=json.loads((CYCLE/'learned_partial_association_v1/results.json').read_text())['details']
    expected={(r['role'],r['mode'],r['group_index']):r for r in expected if r['model']=='rgb_seed17_trained'}
    write_json(OUT/'protocol.json',dict(seed=17,checkpoint_sha256=fingerprint(modelpath),calibration_sha256=fingerprint(calpath),cutoff=cutoff,
        arms=ARMS,device='cpu',encoder_calls=0,training=False,scope='Fixed historical seed17 final800, not best VAL seed. Natural and3logit faults separately. Prediction-only RGB crop metric, calibrated partial groups. Grouped refinements copy only accepted nodes5reliable points; all else original. Local-only within same groups separates local versus geometry. Evaluate GT original/operational assignment AFTER forward. No model promotion or threshold tuning.'))
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='association_gated_fusion_auditing',active_job=OUT.name,active_pid=os.getpid());write_json(CYCLE/'state.json',state)
    details=[];saved=[];allpoints={};sourcehashes={}
    for role,folder in SOURCES.items():
        source=CYCLE/folder/'groups.pt';sourcehashes[role]=fingerprint(source);groups=torch.load(source,map_location='cpu',weights_only=False)
        for gi,g in enumerate(groups):
            assert dt.datetime.now(dt.timezone.utc)<cutoff_time
            original=g['predictions'];views=g['record']['views'];descriptors=croplookup[role,gi]['descriptors']['rgb_cls'];assert croplookup[role,gi]['scene']==g['record']['scene']
            for mode in ['natural']+(MODES if role=='validation' else []):
                p=dict(original);keep=None
                if mode!='natural':
                    keep=choose(original,views,mode);p['objectness_logits']=torch.full_like(original['objectness_logits'],-20)
                    for vi,q in enumerate(keep):p['objectness_logits'][vi,q]=original['objectness_logits'][vi,q]
                associations,nodes,assoc=infer_groups(p,views,descriptors,metric,cutoff)
                # Validate replay parity against earlier inference-only groups, NOT a selection rule.
                if (role,mode,gi) in expected:
                    e=expected[role,mode,gi];exp=[sorted((e['node_list'][j]['view'],e['node_list'][j]['query']) for j in group['nodes']) for group in e['groups']]
                    assert sorted(map(sorted,associations))==sorted(exp),(role,gi,mode)
                params=dict(projections=g['projections'],physical_views=views,pixels=g['pixels'])
                outputs=dict(baseline=p['xy'],local=wrapper(p,**params,local_enabled=True)['xy'],
                             unsafe_composite=wrapper(p,**params,local_enabled=True,fusion_enabled=True)['xy'])
                grouped_outputs={}
                for arm,local,fusion in [('grouped_local',True,False),('grouped_geometry',False,True),('grouped_composite',True,True)]:
                    refined=grouped(p,associations,**params,enabled=True,local_enabled=local,fusion_enabled=fusion)
                    grouped_outputs[arm]=refined;outputs[arm]=refined['xy']
                    others=[k for k in range(23) if k not in KP];assert torch.equal(refined['xy'][:,:,others],p['xy'][:,:,others])
                    noassign=~refined['associated'].any(-1);assert torch.equal(refined['xy'][noassign],p['xy'][noassign])
                associated=grouped_outputs['grouped_geometry']['associated'];tests={}
                assert grouped(p,associations,**params)['xy'] is p['xy']
                if associations:
                    ids=list(reversed(range(len(views))))
                    perm=grouped({k:x[ids] for k,x in p.items()},list(reversed(associations)),g['projections'][ids],[views[i] for i in ids],g['pixels'],enabled=True,local_enabled=True)
                    tests['reverse_max']=float((perm['xy'].flip(0)-outputs['grouped_composite']).abs().max());assert tests['reverse_max']==0
                for count in [1,2]:
                    sub={k:x[:count] for k,x in p.items()};out=grouped(sub,associations,g['projections'][:count],views[:count],g['pixels'],enabled=True)
                    assert torch.equal(out['xy'],sub['xy'])
                # Annotation-dependent evaluation begins after all candidate forward passes.
                d=g['evaluation_data'];fixed=match(original,d);operational=match(p,d);pointsets={}
                for name,matches in [('fixed_original_assignment',fixed),('operational_assignment',operational)]:
                    ps=evaluate(p,d,outputs,matches,associated,keep)
                    for point in ps:point.update(role=role,mode=mode,group_index=gi)
                    allpoints.setdefault(role+'/'+mode+'/'+name,[]).extend(ps);pointsets[name]=ps
                changed={arm:int((xy[:,:,KP]!=p['xy'][:,:,KP]).any(-1).sum()) for arm,xy in outputs.items()}
                row=dict(role=role,mode=mode,group_index=gi,scene=g['record']['scene'],associations=associations,associated_points=int(associated.sum()),
                    changed=changed,tests=tests,grouped_rejected=grouped_outputs['grouped_geometry']['rejected_groups'],
                    metrics=summarize(pointsets['fixed_original_assignment']))
                details.append(row);saved.append(dict(role=role,mode=mode,group_index=gi,outputs=outputs,associated=associated,associations=associations,points=pointsets))
            write_json(OUT/'status.json',dict(stage='auditing',role=role,group=gi,pid=os.getpid()))
        assert fingerprint(source)==sourcehashes[role]
    summary={key:summarize(points) for key,points in allpoints.items()}
    torch.save(saved,OUT/'predictions.pt');write_json(OUT/'points.json',allpoints)
    write_json(OUT/'results.json',dict(summary=summary,details=details,source_sha256=sourcehashes,script_sha256=fingerprint(Path(__file__)),
        module_sha256=fingerprint(ROOT/'mousegeoforge/models/grouped_candidate_fusion.py'),elapsed_seconds=time.monotonic()-started,production_changed=False))
    write_json(OUT/'status.json',dict(stage='complete',pid=os.getpid()))
    state.update(stage='association_gated_fusion_audit_complete',active_job=None,active_pid=None,next='Review all-cohort and assigned/foot effects, worst harms and RGB overlays; do not promote grouped fusion.');state['completed'].append('association_gated_fusion_offline_audit');write_json(CYCLE/'state.json',state)
    print(json.dumps({key:{subset:summary[key][subset] for subset in ['all_matched','associated']} for key in summary if 'validation/natural/fixed' in key},indent=2))


if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise

