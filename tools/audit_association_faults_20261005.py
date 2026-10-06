"""Prediction-only logit fault injection, NOT natural detection evaluation."""
import datetime as dt,json,sys,time,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from audit_composite_candidate_cache_20261005 import load_wrapper
from mousegeoforge.detection_readout_loss import match_detection
from mousegeoforge.pilot_metrics import box_iou
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';SOURCE=CYCLE/'dual_allscene_association_audit_v1';OUT=CYCLE/'association_logit_fault_audit_v1'
KP=[0,14,15,16,17];MODES=['highest_confidence','leftmost_box','alternating_confidence_rank']


def choose(p,views,mode):
    keep=[]
    for vi,view in enumerate(views):
        ids=torch.where(p['objectness_logits'][vi].sigmoid()>=.5)[0]
        assert len(ids)>=1
        if mode=='leftmost_box':q=ids[p['boxes'][vi,ids,0].argmin()]
        else:
            order=ids[p['objectness_logits'][vi,ids].argsort(descending=True,stable=True)]
            rank=int(view)%2 if mode=='alternating_confidence_rank' else 0
            q=order[min(rank,len(order)-1)]
        keep.append(int(q))
    return keep


def match(p,d):
    return match_detection(dict(instance_boxes_cxcywh=p['boxes'],instance_objectness_logits=p['objectness_logits']),d['gt_boxes'],d['box_valid']&d['instance_valid'])


def evaluate(d,p,outputs,matches,keep):
    rows=[]
    for vi,(qs,ts) in enumerate(matches):
        for q,t in zip(qs.tolist(),ts.tolist()):
            iou=float(box_iou(p['boxes'][vi,q:q+1],d['gt_boxes'][vi,t:t+1])[0,0])
            hit=bool(p['objectness_logits'][vi,q].sigmoid()>=.5 and iou>=.5)
            for kp in KP:
                if not d['point_valid'][vi,t,kp]:continue
                errors={m:float(((xy[vi,q,kp]-d['gt_xy'][vi,t,kp])*d['pixels'][vi]).norm()) for m,xy in outputs.items()}
                rows.append(dict(view_index=vi,query=q,target=t,point=kp+1,retained=q==keep[vi],hit=hit,iou=iou,errors=errors))
    return rows


def summarize(rows):
    result={}
    for subset in ['all_matched','retained','retained_iou50']:
        ps=[p for p in rows if subset=='all_matched' or p['retained'] and (subset=='retained' or p['iou']>=.5)]
        result[subset]={}
        for arm in ['baseline','local','composite']:
            a=np.array([p['errors'][arm] for p in ps]);delta=a-np.array([p['errors']['local'] for p in ps])
            result[subset][arm]=dict(points=len(ps),mean=float(a.mean()) if len(a) else None,p95=float(np.percentile(a,95)) if len(a) else None,
                gated_pck32=float(np.mean([p['hit'] and p['errors'][arm]<=32 for p in ps])) if ps else None,
                harms_vs_local_over16=int((delta>16).sum()),worst_harm_vs_local=float(delta.max()) if len(delta) else None)
    return result


@torch.no_grad()
def main():
    plan=json.loads((CYCLE/'plan.json').read_text());assert dt.datetime.now(dt.timezone.utc)<dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'))
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2);start=time.monotonic()
    source=SOURCE/'groups.pt';source_hash=fingerprint(source);groups=torch.load(source,map_location='cpu',weights_only=False)
    provenance=json.loads((SOURCE/'results.json').read_text());wrapper=load_wrapper()
    write_json(OUT/'protocol.json',dict(modes=MODES,suppressed_logit=-20,source_sha256=source_hash,
        scope='Synthetic output fault injection only: RGB, tokens, boxes, keypoints, probabilities unchanged. Keep one ORIGINAL confident query via prediction-only fixed rule; suppress all other logits. No label/count/identity selects queries. Not natural occlusion or detection statistics. No training.',
        evaluation='After forward: both original assignment held fixed and perturbed-logit operational Hungarian assignment. Identity diagnostics require original box assignment IoU>=.5; ambiguous views remain unknown. Same subject cannot be inferred from query ID.'))
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='association_fault_running',active_job=OUT.name,active_pid=os.getpid());write_json(CYCLE/'state.json',state)
    results=[];saved=[];allpoints={mode:{m:[] for m in ['fixed_original_assignment','operational_assignment']} for mode in MODES}
    for gi,g in enumerate(groups):
        original=g['predictions'];d=g['evaluation_data'];views=g['record']['views'];v=len(views)
        origmatch=match(original,d);idlist=provenance['groups'][gi]['instance_ids'];assert all(ids==idlist[0] for ids in idlist)
        for mode in MODES:
            keep=choose(original,views,mode);p=dict(original);p['objectness_logits']=torch.full_like(original['objectness_logits'],-20.)
            for vi,q in enumerate(keep):p['objectness_logits'][vi,q]=original['objectness_logits'][vi,q]
            def call(ids,local=True,fusion=True,**kw):
                return wrapper({k:x[ids] for k,x in p.items()},g['projections'][ids],[views[i] for i in ids],g['pixels'],local_enabled=local,fusion_enabled=fusion,**kw)
            local=call(list(range(v)),fusion=False);composite=call(list(range(v)))
            outputs=dict(baseline=p['xy'],local=local['xy'],composite=composite['xy'])
            tests={}
            assert wrapper(p)['xy'] is original['xy']
            for label,ids in [('reverse',list(reversed(range(v)))),('duplicate',list(range(v))*2)]:
                xy=call(ids)['xy'];xy=xy.flip(0) if label=='reverse' else xy[:v]
                tests[label]=float((xy-composite['xy']).abs().max())
            for count in [1,2]:
                tests[f'V{count}']=max(float((call(([t]+[j for j in range(v) if j!=t])[:count])['xy'][0]-local['xy'][t]).abs().max()) for t in range(v))
            tests['all_invalid']=float((call(list(range(v)),view_valid=[False]*v)['xy']-original['xy']).abs().max())
            tests['untargeted']=float((composite['xy'][:,:,[k for k in range(23) if k not in KP]]-original['xy'][:,:,[k for k in range(23) if k not in KP]]).abs().max())
            assert all(x==0 for x in tests.values()),tests
            # All label-dependent operations below this point are diagnostic only.
            identities=[]
            for vi,q in enumerate(keep):
                assoc={int(a):int(b) for a,b in zip(*origmatch[vi])};t=assoc.get(q)
                iou=None if t is None else float(box_iou(p['boxes'][vi,q:q+1],d['gt_boxes'][vi,t:t+1])[0,0])
                identity=idlist[vi][t] if t is not None and iou>=.5 else None
                identities.append(dict(view=views[vi],query=q,original_assignment=t,iou=iou,reliable_identity=identity))
            validpairs=disagree=0
            for a in range(v):
                for b in range(a+1,v):
                    ia,ib=identities[a]['reliable_identity'],identities[b]['reliable_identity']
                    if ia is not None and ib is not None:validpairs+=1;disagree+=int(ia!=ib)
            pointsets={}
            for label,matches in [('fixed_original_assignment',origmatch),('operational_assignment',match(p,d))]:
                rows=evaluate(d,p,outputs,matches,keep);pointsets[label]=rows;allpoints[mode][label].extend(rows)
            operational=match(p,d);switches=sum(dict(zip(a.tolist(),b.tolist()))!=dict(zip(c.tolist(),e.tolist())) for (a,b),(c,e) in zip(origmatch,operational))
            result=dict(group=gi,scene=g['record']['scene'],fault=mode,keep_queries=keep,identities=identities,reliable_identity_pairs=validpairs,mixed_identity_pairs=disagree,
                fusion_eligible=int(composite['fusion_eligible'].sum()),changed_from_original=int(composite['changed'].sum()),changed_vs_local=int((composite['xy']!=local['xy']).any(-1).sum()),
                operational_assignment_changed_views=switches,tests=tests,metrics={k:summarize(ps) for k,ps in pointsets.items()})
            results.append(result);saved.append(dict(group=gi,fault=mode,keep_queries=keep,logits=p['objectness_logits'],outputs=outputs,points=pointsets))
        write_json(OUT/'status.json',dict(stage='auditing',groups_done=gi+1,pid=os.getpid()))
    assert fingerprint(source)==source_hash
    summary={}
    for mode in MODES:
        rr=[r for r in results if r['fault']==mode]
        summary[mode]=dict(scenes=len(rr),scenes_with_mixed_reliable_identities=sum(r['mixed_identity_pairs']>0 for r in rr),
            mixed_identity_pairs=sum(r['mixed_identity_pairs'] for r in rr),reliable_identity_pairs=sum(r['reliable_identity_pairs'] for r in rr),
            fusion_eligible=sum(r['fusion_eligible'] for r in rr),changed_vs_local=sum(r['changed_vs_local'] for r in rr),
            operational_assignment_changed_views=sum(r['operational_assignment_changed_views'] for r in rr),metrics={k:summarize(ps) for k,ps in allpoints[mode].items()})
    torch.save(saved,OUT/'predictions.pt');write_json(OUT/'results.json',dict(summary=summary,scenes=results,source_unchanged=True,source_sha256=source_hash,
        elapsed_seconds=time.monotonic()-start,device='cpu',encoder_calls=0,training=False,script_sha256=fingerprint(Path(__file__))))
    write_json(OUT/'status.json',dict(stage='complete',pid=os.getpid()))
    state.update(stage='association_fault_complete',active_job=None,active_pid=None,next='Inspect fault-injection results and raw RGB counterexample; record association limitation and next research. Not natural accuracy.')
    state['completed'].append('association_logit_fault_injection');write_json(CYCLE/'state.json',state)
    print(json.dumps({k:{name:value for name,value in v.items() if name!='metrics'} for k,v in summary.items()},indent=2))


if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise
