"""GT-assisted union geometry bound; not a deployed detection result."""
import json
import torch
from collections import defaultdict
from run_depth_pose_ablation import ROOT,fingerprint,write_json
from mousegeoforge.candidate_ceiling import geometric_ceiling
from mousegeoforge.candidate_pool import merge_candidate_pools


def run():
    out=ROOT/'output/candidate_pool_union_audit_v1';out.mkdir(exist_ok=True)
    if (out/'report.json').exists():raise RuntimeError('Preserve audit')
    source=ROOT/'output/dense_center_context_diversity_v1/on';spec=json.loads((source/'protocol.json').read_text())
    groups=defaultdict(lambda:dict(gt=0,original=0,center=0,union=0));rows=[];inputs={}
    for i,e in enumerate(spec['evaluation']):
        p=source/f'candidates_{i:03d}.pt';inputs[str(p)]=fingerprint(p);x=torch.load(p,map_location='cpu',weights_only=True)
        union,_=merge_candidate_pools(x['original_boxes'],x['original_scores'],x['decoded']['boxes'],x['decoded']['scores'],True)
        for v,role in enumerate(x['record']['physical_roles']):
            gt=x['boxes'][v,x['valid'][v]]
            rr={name:geometric_ceiling(boxes,gt) for name,boxes in (('original',x['original_boxes'][v]),('center',x['decoded']['boxes'][v]),('union',union[v]))}
            g=groups[e['group']];g['gt']+=len(gt)
            for name in rr:g[name]+=rr[name]['covered']
            rows.append(dict(index=i,group=e['group'],frame=x['record']['frame'],role=role,bounds=rr))
    total={k:sum(g[k] for g in groups.values()) for k in ('gt','original','center','union')}
    write_json(out/'report.json',dict(passed=True,total=total,groups=dict(groups),rows=rows,input_sha256=inputs,model_forwards=0,optimizer_steps=0,
        code_sha256={p:fingerprint(ROOT/p) for p in ('tools/audit_candidate_pool_union.py','mousegeoforge/candidate_pool.py','mousegeoforge/candidate_ceiling.py')},
        limitation='GT-assisted upper bound, not deployable scoring, recall or evidence that a learned union selector will work.',production_promoted=False))
    print(json.dumps(dict(total=total,dual=groups['old_dual_pair']),indent=2))


if __name__=='__main__':run()
