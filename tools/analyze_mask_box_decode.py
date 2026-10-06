"""Fixed-mask-box audit: unchanged gates, fixed assignments and visual diagnostics."""
import json
from collections import Counter
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from run_depth_pose_ablation import ROOT, torch, write_json, fingerprint
from analyze_cylinder_view_balance import gate_results
from mousegeoforge.retention_geometry_audit import matched_views

OUT=ROOT/'output/mask_box_decode_v1'


def main():
    torch.set_num_threads(2)
    assert json.loads((OUT/'verification.json').read_text())['passed']
    names=('results.json','box_rows.json','assignment_changes.json','mask_availability.json','verification.json','protocol.json')
    hashes={n:fingerprint(OUT/n) for n in names}
    result=json.loads((OUT/'results.json').read_text())
    base=json.loads((ROOT/'output/cylinder_source_mix_v1/results.json').read_text())['baseline']
    control=json.loads((ROOT/'output/cylinder_view_balance_v1/results.json').read_text())['arms']['side_control']
    rows=json.loads((OUT/'box_rows.json').read_text())
    changes=json.loads((OUT/'assignment_changes.json').read_text())
    masks=json.loads((OUT/'mask_availability.json').read_text())
    gates={};summary=[];remaining=[];parts=[]
    for seed,arms in result['seeds'].items():
        gates[seed]={}
        for arm,candidate in arms.items():
            gates[seed][arm]=gate_results(dict(config=result['config'],baseline=base,arms=dict(side_control=control,balanced_views=candidate)))['balanced_views']
        for group in sorted({r['group'] for r in rows}):
            subset=[r for r in rows if r['seed']==seed and r['group']==group]
            a=np.array([r['old_iou'] for r in subset]);b=np.array([r['mask_iou'] for r in subset]);score=np.array([r['score'] for r in subset])
            old=(score>=.5)&(a>=.5);new=(score>=.5)&(b>=.5)
            summary.append(dict(seed=seed,group=group,n=len(subset),old_mean_iou=float(a.mean()),mask_box_mean_iou=float(b.mean()),
                old_iou_quantiles=np.quantile(a,[0,.1,.5,.9,1]).tolist(),mask_box_iou_quantiles=np.quantile(b,[0,.1,.5,.9,1]).tolist(),
                improved=int((b>a+1e-7).sum()),worse=int((b<a-1e-7).sum()),old_detected=int(old.sum()),mask_detected=int(new.sum()),
                gained=int((new&~old).sum()),lost=int((old&~new).sum()),empty_matched_masks=sum(not r['mask_used'] for r in subset)))
        for idx,r in enumerate(arms['regressed']['records']):
            if r['group']!='old_dual_pair':continue
            for arm in arms:
                s=torch.load(OUT/seed/arm/f'prediction_{idx:03d}.pt',map_location='cpu',weights_only=True)
                for item in matched_views(s):
                    if arm=='mask' and not item['found']:
                        remaining.append(dict(seed=seed,index=idx,scene=r['scene'],view=item['view'],role=item['role'],query=item['query'],instance=item['instance'],score=item['score'],iou=item['iou']))
                    for part,inds in [('body',list(range(7))+list(range(14,23))),('tail',list(range(7,14)))]:
                        valid=item['valid'][inds];error=item['errors'][inds][valid]
                        if len(error):parts.append(dict(seed=seed,arm=arm,part=part,role=item['role'],nme=float(error.mean())))
        fig,axes=plt.subplots(4,2,figsize=(13,15),layout='constrained')
        for row,frame in enumerate((246,296,352,383)):
            idx=next(i for i,r in enumerate(arms['regressed']['records']) if r['group']=='old_dual_pair' and r['scene'].endswith(f'{frame:06d}'))
            old=torch.load(OUT/seed/'regressed'/f'prediction_{idx:03d}.pt',map_location='cpu',weights_only=True)
            new=torch.load(OUT/seed/'mask'/f'prediction_{idx:03d}.pt',map_location='cpu',weights_only=True)
            r=old['record'];pixel=np.array(r['image_hw'][::-1])
            for col,role in enumerate(('side','top')):
                v=r['physical_roles'].index(role);ax=axes[row,col]
                ax.imshow(Image.open(ROOT/r['scene']/'blended_images'/f'{r["views"][v]:08d}.jpg').resize(tuple(pixel)))
                labels=[]
                for item in [x for x in rows if x['seed']==seed and x['index']==idx and x['view']==v]:
                    q,m=item['query'],item['instance']
                    boxes=[(old['targets']['boxes_cxcywh'][0,v,m],'lime','-'),(old['predictions']['instance_boxes_cxcywh'][0,v,q],'magenta','--'),(new['predictions']['instance_boxes_cxcywh'][0,v,q],'cyan',':')]
                    for box,color,style in boxes:
                        b=box.numpy();ax.add_patch(Rectangle((b[:2]-b[2:]/2)*pixel,*(b[2:]*pixel),fill=False,edgecolor=color,linestyle=style,lw=1.5))
                    labels.append(f'm{m}/q{q}: {item["old_iou"]:.2f} -> {item["mask_iou"]:.2f}')
                ax.set_title(f'Frame {frame} / {role} | '+', '.join(labels),fontsize=10);ax.axis('off')
        fig.suptitle(f'{seed}: GT green / regressed magenta / mask-box cyan\nFixed original assignment; unchanged coordinates and scores; full input shown')
        fig.savefig(OUT/f'dual_fixed_{seed}.png',dpi=135);plt.close(fig)
    part_summary=[]
    for seed in result['seeds']:
        for arm in ('regressed','mask'):
            for part in ('body','tail'):
                values=[x['nme'] for x in parts if (x['seed'],x['arm'],x['part'])==(seed,arm,part)]
                part_summary.append(dict(seed=seed,arm=arm,part=part,nme=float(np.mean(values))))
    report=dict(gates=gates,box_summary=summary,dual_parts=part_summary,remaining_dual_misses=remaining,
        changed_assignment_records=dict(Counter(x['seed'] for x in changes)),
        mask_availability={seed:{k:sum(x[k] for x in masks if x['seed']==seed) for k in ('queries','nonempty','high_score_empty')} for seed in result['seeds']},
        production_promoted=False,optimizer_steps=0,source_sha256=hashes,analysis_code_sha256=fingerprint(ROOT/'tools/analyze_mask_box_decode.py'),
        limitations=['Same reused synthetic development set and two adaptation-order seeds, not independent real test data.',
            'Boxes alone change; detection-gated PCK gains are not keypoint coordinate gains. Rematching may change coordinate metrics.',
            'box_summary uses fixed original GT assignment; gates rerun original box-only assignment.',
            'High-score query counts are not FP counts; full detection precision requires a separate explicit detection protocol.',
            'Visible mask boxes are not amodal extents and include tails and disconnected foreground.'])
    assert all(fingerprint(OUT/n)==h for n,h in hashes.items())
    write_json(OUT/'acceptance.json',report)
    print(json.dumps(dict(gates={s:{a:g['passed_count'] for a,g in arms.items()} for s,arms in gates.items()},dual=[x for x in summary if x['group']=='old_dual_pair'],remaining=remaining,changes=report['changed_assignment_records']),indent=2))


if __name__=='__main__':main()
