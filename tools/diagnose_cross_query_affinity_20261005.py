"""Threshold-free affinity diagnostics; no learned weights or inference labels."""
import datetime as dt,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from scipy.stats import rankdata
from mousegeoforge.association_evidence import pairwise_affinity_evidence
from audit_association_faults_20261005 import match
from mousegeoforge.pilot_metrics import box_iou
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';OUT=CYCLE/'cross_query_affinity_diagnostic_v1'
SOURCES=dict(train='dual_train_allview_affinity_context_v1',validation='dual_allscene_association_audit_v1')


def auc(scores,labels):
    scores=np.asarray(scores);labels=np.asarray(labels,bool);good=np.isfinite(scores);scores=scores[good];labels=labels[good]
    n1=int(labels.sum());n0=len(labels)-n1
    return float((rankdata(scores)[labels].sum()-n1*(n1+1)/2)/(n1*n0)) if n1 and n0 else None


@torch.no_grad()
def main():
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2);started=time.monotonic();allresults={};source_hashes={};scenes={};saved=[]
    for role,folder in SOURCES.items():
        path=CYCLE/folder/'groups.pt';source_hashes[role]=fingerprint(path);groups=torch.load(path,map_location='cpu',weights_only=False);provenance=json.loads((path.parent/'results.json').read_text())
        scenes[role]={g['record']['scene'] for g in groups};edges=[];rankings=[];nodes_total=nodes_known=0
        for gi,g in enumerate(groups):
            assert g['record']['role']==role
            p=g['predictions'];d=g['evaluation_data'];ev=pairwise_affinity_evidence(p,g['projections'],g['record']['views'],g['pixels'])
            # Evidence and pair rankings are fully computed before identity labels.
            forward_rank=[]
            for channel,matrix in ev['evidence'].items():
                sign=-1 if channel.startswith('epipolar') else 1;scores=matrix*sign;best={}
                for a,na in enumerate(ev['nodes']):
                    for view in sorted({x['view'] for x in ev['nodes']}):
                        if view==na['view']:continue
                        candidates=[j for j,n in enumerate(ev['nodes']) if n['view']==view and np.isfinite(scores[a,j])]
                        if not candidates:continue
                        vals=scores[a,candidates];mx=float(vals.max());ties=[j for j in candidates if np.isclose(scores[a,j],mx,rtol=0,atol=1e-12)]
                        if len(ties)==1:best[(a,view)]=ties[0]
                for (a,view),b in best.items():forward_rank.append(dict(channel=channel,a=a,b=b,mutual=best.get((b,ev['nodes'][a]['view']))==a))
            matches=match(p,d);identities=[];ious=[];ids=provenance['groups'][gi]['instance_ids']
            assert all(x==ids[0] for x in ids)
            for node in ev['nodes']:
                vi,q=node['view_index'],node['query'];mapping={int(a):int(b) for a,b in zip(*matches[vi])};target=mapping.get(q)
                iou=0. if target is None else float(box_iou(p['boxes'][vi,q:q+1],d['gt_boxes'][vi,target:target+1])[0,0]);ious.append(iou)
                identities.append(ids[vi][target] if target is not None and iou>=.5 else None)
            nodes_total+=len(identities);nodes_known+=sum(x is not None for x in identities)
            for a,na in enumerate(ev['nodes']):
                for b in range(a+1,len(ev['nodes'])):
                    nb=ev['nodes'][b]
                    if na['view']==nb['view']:continue
                    known=identities[a] is not None and identities[b] is not None
                    edges.append(dict(group=gi,a=a,b=b,view_a=na['view'],query_a=na['query'],view_b=nb['view'],query_b=nb['query'],
                        known=known,same_identity=identities[a]==identities[b] if known else None,same_query=na['query']==nb['query'],
                        evidence={k:float(x[a,b]) if np.isfinite(x[a,b]) else None for k,x in ev['evidence'].items()}))
            for x in forward_rank:
                a,b=x['a'],x['b'];known=identities[a] is not None and identities[b] is not None
                rankings.append(dict(x,group=gi,known=known,same_identity=identities[a]==identities[b] if known else None,same_query=ev['nodes'][a]['query']==ev['nodes'][b]['query']))
            saved.append(dict(role=role,group=gi,scene=g['record']['scene'],nodes=ev['nodes'],evidence=ev['evidence'],identity_diagnostic=identities,ious=ious))
        summaries={}
        for channel in ev['evidence']:
            rows=[e for e in edges if e['known'] and e['evidence'][channel] is not None];sign=-1 if channel.startswith('epipolar') else 1
            values=[sign*x['evidence'][channel] for x in rows];labels=[x['same_identity'] for x in rows]
            top=[x for x in rankings if x['channel']==channel];mutual=[x for x in top if x['mutual'] and x['a']<x['b']]
            def rank_summary(items):
                known=[x for x in items if x['known']]
                return dict(proposed=len(items),known=len(known),unknown=len(items)-len(known),correct=sum(x['same_identity'] for x in known),
                    precision_known=float(np.mean([x['same_identity'] for x in known])) if known else None,same_query_fraction=float(np.mean([x['same_query'] for x in items])) if items else None)
            summaries[channel]=dict(known_pairs=len(rows),same=sum(labels),different=len(labels)-sum(labels),same_identity_auc=auc(values,labels),
                same_query_auc=auc(values,[x['same_query'] for x in rows]),top1=rank_summary(top),mutual=rank_summary(mutual),
                same_median=float(np.median([x['evidence'][channel] for x in rows if x['same_identity']])),different_median=float(np.median([x['evidence'][channel] for x in rows if not x['same_identity']])))
        write_json(OUT/f'{role}_edges.json',edges);write_json(OUT/f'{role}_rankings.json',rankings)
        allresults[role]=dict(images=sum(len(g['record']['views']) for g in groups),scenes=len(groups),nodes=nodes_total,known_nodes=nodes_known,unknown_nodes=nodes_total-nodes_known,channels=summaries)
    assert scenes['train'].isdisjoint(scenes['validation'])
    torch.save(saved,OUT/'evidence.pt');write_json(OUT/'results.json',dict(results=allresults,source_sha256=source_hashes,elapsed_seconds=time.monotonic()-started,
        module_sha256=fingerprint(ROOT/'mousegeoforge/association_evidence.py'),script_sha256=fingerprint(Path(__file__)),training=False,
        protocol='All confident queries, fixed corresponding5point geometry and frozen semantic/candidate0local cosine. No score combination, threshold fitting, VAL selection or fusion. Mutual top1 is a ranking diagnostic, cannot reject an incorrect singleton. Posthoc identity requires IoU>=.5 assignment; unknown detections counted. Same-query AUC audits shortcut. Disjoint frames, not independent source videos; black/brown color cue not same-color proof.'))
    for role,value in allresults.items():print(role,json.dumps(value,indent=2),flush=True)


if __name__=='__main__':main()
