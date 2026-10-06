"""Zero-forward frozen-candidate geometry bound and score-route comparison."""
import json
import time
from collections import defaultdict
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_depth_pose_ablation import ROOT,fingerprint,write_json
from mousegeoforge.candidate_ceiling import geometric_ceiling
from mousegeoforge.dense_center_probe import candidate_metrics

OUT=ROOT/'output/frozen_center_ceiling_audit_v1'
SOURCE=ROOT/'output/dense_center_context_diversity_v1/on'


def run():
    started=time.monotonic();torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'report.json').exists():raise RuntimeError('Preserve existing audit')
    report=json.loads((SOURCE/'report.json').read_text());assert report['passed']
    groups=defaultdict(lambda:dict(gt=0,center_ceiling=0,original_ceiling=0,center_tp=0,original_tp=0));rows=[];inputs={}
    evaluation=json.loads((SOURCE/'protocol.json').read_text())['evaluation']
    for index,e in enumerate(evaluation):
        path=SOURCE/f'candidates_{index:03d}.pt';inputs[str(path)]=fingerprint(path)
        item=torch.load(path,map_location='cpu',weights_only=True);r=item['record']
        for v,role in enumerate(r['physical_roles']):
            gt=item['boxes'][v,item['valid'][v]]
            center=geometric_ceiling(item['decoded']['boxes'][v],gt);original=geometric_ceiling(item['original_boxes'][v],gt)
            cm=candidate_metrics(item['decoded']['boxes'][v],item['decoded']['scores'][v],gt)
            om=candidate_metrics(item['original_boxes'][v],item['original_scores'][v],gt)
            for key in (e['group'],e['group']+'/'+role):
                g=groups[key];g['gt']+=len(gt);g['center_ceiling']+=center['covered'];g['original_ceiling']+=original['covered'];g['center_tp']+=cm['tp'];g['original_tp']+=om['tp']
            rows.append(dict(index=index,group=e['group'],frame=r['frame'],role=role,view=r['views'][v],center=center,original=original))
    stages=[('Original',None,None),('Center16','dense_center_context_diversity_v1/on','candidate'),('ROI quality','roi_candidate_quality_v1','on'),
            ('Unique independent','roi_set_objectness_pair_v1/off','on'),('Unique relational','roi_set_objectness_pair_v1/on','on'),
            ('Geometry OFF','roi_geometric_objectness_pair_v1/off','on'),('Geometry ON','roi_geometric_objectness_pair_v1/on','on')]
    comparison=[]
    for name,path,arm in stages:
        source_path=ROOT/'output'/path/'report.json' if path else SOURCE/'report.json'
        r=json.loads(source_path.read_text());inputs[str(source_path)]=fingerprint(source_path)
        summary=r['summary'][arm] if path else r['summary']['original']
        dev={g:v for g,v in summary.items() if not g.startswith('train_')}
        comparison.append(dict(name=name,dual_tp=dev['old_dual_pair']['tp'],dual_fp=dev['old_dual_pair']['fp'],
            tp=sum(v['tp'] for v in dev.values()),fp=sum(v['fp'] for v in dev.values()),duplicate_views=sum(v['duplicate_views'] for v in dev.values())))
    total={k:sum(v[k] for g,v in groups.items() if '/' not in g) for k in ('gt','center_ceiling','original_ceiling','center_tp','original_tp')}
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for row in comparison:
        for ax,x,y in ((axes[0],row['fp'],row['tp']),(axes[1],row['dual_fp'],row['dual_tp'])):
            ax.scatter(x,y,s=45);ax.annotate(row['name'],(x,y),xytext=(4,4),textcoords='offset points',fontsize=8)
    for ax,title,bound in ((axes[0],'All184 development views /216GT',total['center_ceiling']),
                           (axes[1],'Dual32 development views /64GT',groups['old_dual_pair']['center_ceiling'])):
        ax.axhline(bound,ls='--',color='gray',label='Frozen center geometry bound (uses GT, not achievable claim)')
        ax.set(title=title,xlabel='False positives (lower better)',ylabel='True positives (higher better)');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(OUT/'candidate_score_tradeoff.png',dpi=140);plt.close(fig)
    for p,h in inputs.items():assert fingerprint(__import__('pathlib').Path(p))==h
    write_json(OUT/'report.json',dict(passed=True,model_forwards=0,optimizer_steps=0,groups=dict(groups),total=total,rows=rows,score_routes=comparison,
        input_sha256=inputs,code_sha256={p:fingerprint(ROOT/p) for p in ('tools/audit_frozen_center_ceiling.py','mousegeoforge/candidate_ceiling.py','tests/test_candidate_ceiling.py')},
        limitations=['GT-assisted maximum bipartite IoU>=.5 matching is a geometry capacity bound, not deployable scoring or the standard metric.',
                     'Realizing the bound while preserving score-independent assignment and lowFP is not guaranteed.',
                     'Original learned-query boxes differ; all other plotted routes share exactly the same frozen16pose center boxes.',
                     'No keypoint predictions changed or PCK improvements established.'],production_promoted=False,elapsed_s=time.monotonic()-started))
    print(json.dumps(dict(total=total,dual=groups['old_dual_pair'],comparison=comparison),indent=2))


if __name__=='__main__':run()
