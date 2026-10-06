"""Freeze internal-calibration thresholds before development and fault audits."""
import datetime as dt,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from torch.nn import functional as F
from mousegeoforge.partial_camera_association import conservative_camera_groups
from audit_association_faults_20261005 import choose,MODES
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';SOURCE=CYCLE/'association_projection_pilot_v1';OUT=CYCLE/'learned_partial_association_v1'


def distances(z):
    z=F.normalize(z.float(),dim=-1)
    return (1-z@z.T).clamp(0,2).numpy()


def main():
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2);started=time.monotonic()
    plan=json.loads((CYCLE/'plan.json').read_text());assert dt.datetime.now(dt.timezone.utc)<dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'))
    protocol=json.loads((SOURCE/'protocol.json').read_text());data=torch.load(SOURCE/'inputs.pt',map_location='cpu',weights_only=False)
    embedded=torch.load(SOURCE/'embeddings.pt',map_location='cpu',weights_only=False)
    evidence=torch.load(CYCLE/'cross_query_affinity_diagnostic_v1/evidence.pt',map_location='cpu',weights_only=False)
    lookup={(x['role'],x['group']):x for x in evidence};cal=protocol['calibration_indices'];models={}
    # Raw reference and all three final seeds, not best-VAL model selection.
    for arm in ['rgb','gray']:
        for seed,phase in [(17,'raw')]+[(s,'trained') for s in [17,29,43]]:
            key=f'{arm}_seed{seed}_{phase}';z=embedded[key+'_train'];negative=[];positive=[]
            for gi in cal:
                matrix=distances(z[gi]);d=data['train'];labels=d['labels'][gi];views=d['views'][gi]
                for a in range(8):
                    for b in range(a+1,8):
                        if views[a]==views[b] or min(labels[a],labels[b])<0:continue
                        (positive if labels[a]==labels[b] else negative).append(float(matrix[a,b]))
            negative.sort();assert negative and positive
            budget=int(np.floor(.01*len(negative)));cutoff=negative[budget]
            models[key]=dict(cutoff=cutoff,calibration_negatives=len(negative),calibration_positives=len(positive),negative_budget=budget,
                negative_accepted=sum(x<cutoff for x in negative),positive_accepted=sum(x<cutoff for x in positive))
            assert models[key]['negative_accepted']<=budget
    frozen=dict(models=models,calibration_indices=cal,source_protocol_sha256=fingerprint(SOURCE/'protocol.json'),frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        scope='Distance=1-cosine. Same predeclared1% empirical negative-pair budget as geometry control, now on reserved5 association-stage TRAIN calibration scenes. Strict cutoff; mutual nearest; >=3 physical-view complete cliques; reject overlap. Unknown labels excluded from calibration, NOT from inference. All seeds and raw controls. No pose change or VAL threshold choice. Small dependent calibration sample cannot certify safety.')
    write_json(OUT/'calibration.json',frozen);calhash=fingerprint(OUT/'calibration.json');details=[]
    # Development groups are accessed only after calibration has been frozen.
    valgroups=torch.load(CYCLE/'dual_allscene_association_audit_v1/groups.pt',map_location='cpu',weights_only=False)
    for key,setting in models.items():
        for role in ['train','validation']:
            z=embedded[key+'_'+role];groups=range(len(z)) if role=='validation' else cal
            for gi in groups:
                e=lookup[role,gi];nodes=e['nodes'];identity=e['identity_diagnostic'];full=distances(z[gi])
                for mode in ['natural']+(MODES if role=='validation' else []):
                    ids=list(range(len(nodes)))
                    if mode!='natural':
                        g=valgroups[gi];kept=choose(g['predictions'],g['record']['views'],mode)
                        ids=[j for j,n in enumerate(nodes) if kept[n['view_index']]==n['query']]
                    ns=[nodes[j] for j in ids];matrix=full[np.ix_(ids,ids)]
                    out=conservative_camera_groups(ns,matrix,setting['cutoff'],enabled=True)
                    assert not out['overflow'] and not conservative_camera_groups(ns,matrix,setting['cutoff'])['groups']
                    perm=list(reversed(range(len(ns))));reverse=conservative_camera_groups([ns[j] for j in perm],matrix[np.ix_(perm,perm)],setting['cutoff'],enabled=True)
                    assert sorted(tuple(sorted(perm[j] for j in group)) for group in reverse['groups'])==sorted(tuple(group) for group in out['groups'])
                    interpreted=[]
                    for group in out['groups']:
                        labels=[identity[ids[j]] for j in group];known={x for x in labels if x is not None}
                        status='mixed' if len(known)>1 else 'unknown' if None in labels else 'pure_known'
                        interpreted.append(dict(nodes=[ids[j] for j in group],identities=labels,status=status))
                    edges=[]
                    for a,b in out['edges']:
                        labels=[identity[ids[a]],identity[ids[b]]];known=None not in labels
                        edges.append(dict(nodes=[ids[a],ids[b]],known=known,correct=labels[0]==labels[1] if known else None,distance=float(matrix[a,b])))
                    details.append(dict(model=key,role=role,mode=mode,group_index=gi,scene=e['scene'],node_indices=ids,node_list=nodes,
                        identity_diagnostic=identity,groups=interpreted,edges=edges,nodes=len(ids),assigned_nodes=len(ids)-len(out['unassigned']),
                        rejected_ambiguous=len(out['ambiguous_groups'])))
    summary={}
    for key in models:
        for role in ['train','validation']:
            for mode in ['natural']+(MODES if role=='validation' else []):
                rows=[r for r in details if (r['model'],r['role'],r['mode'])==(key,role,mode)];gs=[g for r in rows for g in r['groups']];es=[e for r in rows for e in r['edges']]
                summary[key+'/'+role+'/'+mode]=dict(scenes=len(rows),nodes=sum(r['nodes'] for r in rows),assigned_nodes=sum(r['assigned_nodes'] for r in rows),groups=len(gs),
                    pure_known=sum(g['status']=='pure_known' for g in gs),mixed=sum(g['status']=='mixed' for g in gs),unknown=sum(g['status']=='unknown' for g in gs),
                    pair_edges=len(es),pair_known=sum(e['known'] for e in es),pair_correct=sum(e['correct'] is True for e in es),pair_unknown=sum(not e['known'] for e in es),
                    rejected_ambiguous=sum(r['rejected_ambiguous'] for r in rows))
    assert fingerprint(OUT/'calibration.json')==calhash
    write_json(OUT/'results.json',dict(summary=summary,details=details,calibration_sha256=calhash,elapsed_seconds=time.monotonic()-started,
        source_sha256={name:fingerprint(SOURCE/name) for name in ['inputs.pt','embeddings.pt','results.json']},script_sha256=fingerprint(Path(__file__)),production_changed=False))
    write_json(OUT/'status.json',dict(stage='complete'))
    print(json.dumps({k:v for k,v in summary.items() if 'validation/natural' in k},indent=2))


if __name__=='__main__':main()
