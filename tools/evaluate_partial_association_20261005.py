"""Freeze TRAIN-only cutoff, then evaluate natural and isolated fault contexts."""
import datetime as dt,json,sys,time,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from mousegeoforge.association_evidence import pairwise_affinity_evidence
from mousegeoforge.partial_camera_association import conservative_camera_groups
from audit_association_faults_20261005 import choose,match,MODES
from mousegeoforge.pilot_metrics import box_iou
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';OUT=CYCLE/'partial_association_control_v1'
SOURCES=dict(train='dual_train_allview_affinity_context_v1',validation='dual_allscene_association_audit_v1')


def summarize(rows):
    groups=[g for r in rows for g in r['groups']]
    return dict(scenes=len(rows),predicted_nodes=sum(r['nodes'] for r in rows),assigned_nodes=sum(r['assigned_nodes'] for r in rows),
        groups=len(groups),pure_known_groups=sum(g['status']=='pure_known' for g in groups),mixed_groups=sum(g['status']=='mixed' for g in groups),
        unknown_groups=sum(g['status']=='unknown' for g in groups),recovered_gt_instances=sum(r['recovered_gt_instances'] for r in rows),
        gt_instances=sum(r['gt_instances'] for r in rows),scenes_with_group=sum(bool(r['groups']) for r in rows),
        groups_with_ambiguity_rejected=sum(r['ambiguous_groups'] for r in rows),pair_edges=sum(r['pair_edges'] for r in rows))


@torch.no_grad()
def main():
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2);started=time.monotonic()
    plan=json.loads((CYCLE/'plan.json').read_text());assert dt.datetime.now(dt.timezone.utc)<dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'))
    # No development/fault data is loaded before this threshold is persisted.
    path=CYCLE/'cross_query_affinity_diagnostic_v1/train_edges.json';edges=json.loads(path.read_text())
    negatives=sorted(e['evidence']['epipolar_mean_px'] for e in edges if e['known'] and not e['same_identity'] and e['evidence']['epipolar_mean_px'] is not None)
    positives=[e['evidence']['epipolar_mean_px'] for e in edges if e['known'] and e['same_identity'] and e['evidence']['epipolar_mean_px'] is not None]
    budget=int(np.floor(.01*len(negatives)));cutoff=float(negatives[budget]);accepted=sum(x<cutoff for x in negatives);assert accepted<=budget
    calibration=dict(channel='epipolar_mean_px',cutoff_px=cutoff,strict_comparison='distance < cutoff',train_negative_pairs=len(negatives),
        train_positive_pairs=len(positives),negative_accept_budget=budget,actual_negative_accepted=accepted,train_positive_accepted=sum(x<cutoff for x in positives),
        source_sha256=fingerprint(path),frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        protocol='Predeclared1% empirical TRAIN negative-edge budget; cutoff is next negative order statistic, strict less-than handles ties. Not a statistical safety guarantee. Mutual nearest by physical camera, reject ties, maximal complete groups>=3 cameras, reject all overlapping cliques. No count input, no pose fusion, no VAL fitting.')
    write_json(OUT/'calibration.json',calibration);calibration_hash=fingerprint(OUT/'calibration.json')
    print('FROZEN CALIBRATION',json.dumps(calibration),flush=True)
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='partial_association_running',active_job=OUT.name,active_pid=os.getpid());write_json(CYCLE/'state.json',state)
    details=[];sources={}
    for role,folder in SOURCES.items():
        source=CYCLE/folder/'groups.pt';sources[role]=fingerprint(source);groups=torch.load(source,map_location='cpu',weights_only=False)
        provenance=json.loads((source.parent/'results.json').read_text())
        for gi,g in enumerate(groups):
            original=g['predictions'];d=g['evaluation_data'];views=g['record']['views']
            for mode in ['natural']+(MODES if role=='validation' else []):
                p=dict(original)
                if mode!='natural':
                    keep=choose(original,views,mode);p['objectness_logits']=torch.full_like(original['objectness_logits'],-20.)
                    for vi,q in enumerate(keep):p['objectness_logits'][vi,q]=original['objectness_logits'][vi,q]
                evidence=pairwise_affinity_evidence(p,g['projections'],views,g['pixels']);nodes=evidence['nodes'];matrix=evidence['evidence']['epipolar_mean_px']
                out=conservative_camera_groups(nodes,matrix,cutoff,enabled=True)
                assert not out['overflow']
                # Exact index permutation invariance, including ambiguity decisions.
                perm=list(reversed(range(len(nodes))));rev=conservative_camera_groups([nodes[j] for j in perm],matrix[np.ix_(perm,perm)],cutoff,enabled=True)
                assert sorted(tuple(sorted(perm[j] for j in group)) for group in rev['groups'])==sorted(tuple(group) for group in out['groups'])
                assert not conservative_camera_groups(nodes,matrix,cutoff)['groups']
                for subset in [views[:1],views[:2]]:
                    ids=[i for i,node in enumerate(nodes) if node['view'] in subset]
                    assert not conservative_camera_groups([nodes[i] for i in ids],matrix[np.ix_(ids,ids)],cutoff,enabled=True)['groups']
                # Only after the prediction-only association inspect target labels.
                matches=match(original,d);identities=[];ious=[];actual_ids=provenance['groups'][gi]['instance_ids']
                for node in nodes:
                    vi,q=node['view_index'],node['query'];mapping=dict(zip(matches[vi][0].tolist(),matches[vi][1].tolist()));target=mapping.get(q)
                    iou=None if target is None else float(box_iou(original['boxes'][vi,q:q+1],d['gt_boxes'][vi,target:target+1])[0,0]);ious.append(iou)
                    identities.append(actual_ids[vi][target] if target is not None and iou>=.5 else None)
                interpreted=[];recovered=set()
                for group in out['groups']:
                    known={identities[i] for i in group if identities[i] is not None};unknown=any(identities[i] is None for i in group)
                    status='mixed' if len(known)>1 else 'unknown' if unknown else 'pure_known'
                    if status=='pure_known':recovered.update(known)
                    interpreted.append(dict(nodes=group,status=status,identities=[identities[i] for i in group]))
                details.append(dict(role=role,mode=mode,group_index=gi,scene=g['record']['scene'],nodes=len(nodes),node_list=nodes,identity_diagnostic=identities,ious=ious,
                    groups=interpreted,assigned_nodes=len(nodes)-len(out['unassigned']),unassigned=out['unassigned'],edges=out['edges'],pair_edges=len(out['edges']),
                    ambiguous_groups=len(out['ambiguous_groups']),recovered_gt_instances=len(recovered),gt_instances=int(d['instance_valid'][0].sum())))
        assert fingerprint(source)==sources[role]
    summary={}
    for role in SOURCES:
        for mode in ['natural']+(MODES if role=='validation' else []):summary[role+'/'+mode]=summarize([x for x in details if x['role']==role and x['mode']==mode])
    assert fingerprint(OUT/'calibration.json')==calibration_hash
    write_json(OUT/'results.json',dict(calibration=calibration,calibration_sha256=calibration_hash,summary=summary,details=details,source_sha256=sources,
        script_sha256=fingerprint(Path(__file__)),module_sha256=fingerprint(ROOT/'mousegeoforge/partial_camera_association.py'),elapsed_seconds=time.monotonic()-started,
        inference_claim='Association only, no poses changed. Natural and fault-injection kept separate. IoU-qualified identity evaluation has unknown coverage; few calibrated negatives cannot certify safety.'))
    write_json(OUT/'status.json',dict(stage='complete',pid=os.getpid()))
    state.update(stage='partial_association_complete',active_job=None,active_pid=None,next='Inspect strict partial association results and coverage, visualize accepted/rejected cases; do not loosen cutoff on VAL.');state['completed'].append('partial_association_train_calibrated_control');write_json(CYCLE/'state.json',state)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise
