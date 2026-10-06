"""Separate frozen ROI descriptor diagnostics, no fit or model selection."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from diagnose_cross_query_affinity_20261005 import auc
from run_depth_pose_ablation import write_json,fingerprint
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';SOURCE=CYCLE/'prequery_roi_descriptors_v1';OUT=CYCLE/'prequery_roi_affinity_diagnostic_v1'


def rank_summary(items):
    known=[x for x in items if x['known']]
    return dict(proposed=len(items),known=len(known),unknown=len(items)-len(known),correct=sum(x['same_identity'] for x in known),
        precision_known=float(np.mean([x['same_identity'] for x in known])) if known else None,same_query_fraction=float(np.mean([x['same_query'] for x in items])) if items else None)


def main(channels=None):
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2)
    source=SOURCE/'descriptors.pt';data=torch.load(source,map_location='cpu',weights_only=False)
    evidence=torch.load(CYCLE/'cross_query_affinity_diagnostic_v1/evidence.pt',map_location='cpu',weights_only=False);lookup={(x['role'],x['group']):x for x in evidence}
    channels=['image_backbone','projected_memory'] if channels is None else channels
    allresults={};saved=[]
    for role in ['train','validation']:
        edges=[];rankings=[]
        for g in [x for x in data if x['role']==role]:
            ev=lookup[(role,g['group'])];nodes=ev['nodes'];n=len(nodes);matrices={}
            for channel in channels:
                vectors=np.stack([g['descriptors'][channel][x['view_index'],x['query']].numpy() for x in nodes]);norm=np.linalg.norm(vectors,axis=-1)
                valid=g.get('valid',g['descriptors'].get('valid'))
                good=np.array([bool(valid[x['view_index'],x['query']]) for x in nodes])&(norm>1e-12)&np.isfinite(vectors).all(-1)
                vectors=vectors/norm.clip(1e-12)[:,None];matrix=vectors@vectors.T
                for a,na in enumerate(nodes):
                    for b,nb in enumerate(nodes):
                        if na['view']==nb['view'] or not good[a] or not good[b]:matrix[a,b]=np.nan
                matrices[channel]=matrix
            # Features complete; labels below are evaluation-only.
            labels=ev['identity_diagnostic']
            for a,na in enumerate(nodes):
                for b in range(a+1,n):
                    nb=nodes[b]
                    if na['view']==nb['view']:continue
                    known=labels[a] is not None and labels[b] is not None
                    edges.append(dict(group=g['group'],a=a,b=b,known=known,same_identity=labels[a]==labels[b] if known else None,same_query=na['query']==nb['query'],
                        scores={k:float(m[a,b]) if np.isfinite(m[a,b]) else None for k,m in matrices.items()}))
            for channel,matrix in matrices.items():
                best={}
                for a,na in enumerate(nodes):
                    for view in g['views']:
                        candidates=[i for i,node in enumerate(nodes) if node['view']==view and np.isfinite(matrix[a,i])]
                        if not candidates:continue
                        maximum=max(matrix[a,i] for i in candidates);ties=[i for i in candidates if abs(matrix[a,i]-maximum)<1e-12]
                        if len(ties)==1:best[a,view]=ties[0]
                for (a,view),b in best.items():
                    known=labels[a] is not None and labels[b] is not None
                    rankings.append(dict(channel=channel,group=g['group'],a=a,b=b,mutual=best.get((b,nodes[a]['view']))==a,known=known,
                        same_identity=labels[a]==labels[b] if known else None,same_query=nodes[a]['query']==nodes[b]['query']))
            saved.append(dict(role=role,group=g['group'],scene=g['scene'],nodes=nodes,evidence=matrices))
        summary={}
        for channel in channels:
            rows=[x for x in edges if x['known'] and x['scores'][channel] is not None];scores=[x['scores'][channel] for x in rows]
            top=[x for x in rankings if x['channel']==channel];mutual=[x for x in top if x['mutual'] and x['a']<x['b']]
            summary[channel]=dict(known_pairs=len(rows),same_identity_auc=auc(scores,[x['same_identity'] for x in rows]),same_query_auc=auc(scores,[x['same_query'] for x in rows]),
                top1=rank_summary(top),mutual=rank_summary(mutual),same_median=float(np.median([x['scores'][channel] for x in rows if x['same_identity']])),
                different_median=float(np.median([x['scores'][channel] for x in rows if not x['same_identity']])))
        allresults[role]=summary;write_json(OUT/f'{role}_edges.json',edges);write_json(OUT/f'{role}_rankings.json',rankings)
    torch.save(saved,OUT/'evidence.pt');write_json(OUT/'results.json',dict(results=allresults,descriptor_sha256=fingerprint(source),script_sha256=fingerprint(Path(__file__)),
        protocol='Separate fixed descriptor channels, no threshold/weight fit. Predicted-box descriptors computed before identity evaluation; no GT crops. Current Omega image feature stream, not a new re-ID model. See source extraction protocol for ROI pooling versus crop re-encoding. Black/brown synthetic cohort only; same-color/real generalization untested.'))
    print(json.dumps(allresults,indent=2))


if __name__=='__main__':main()
