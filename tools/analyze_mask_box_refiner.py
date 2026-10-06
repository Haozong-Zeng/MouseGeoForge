"""Unchanged development gates and fixed visuals for frozen-feature box pilot."""
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from train_mask_box_refiner import ROOT,OUT,ARMS
from run_depth_pose_ablation import torch,write_json,fingerprint
from analyze_cylinder_view_balance import gate_results
from mousegeoforge.retention_geometry_audit import matched_views


def main():
    torch.set_num_threads(2)
    assert json.loads((OUT/'verification.json').read_text())['passed']
    results=json.loads((OUT/'results.json').read_text())
    baseline=json.loads((ROOT/'output/cylinder_source_mix_v1/results.json').read_text())['baseline']
    control=json.loads((ROOT/'output/cylinder_view_balance_v1/results.json').read_text())['arms']['side_control']
    gates={a:gate_results(dict(config=results['config'],baseline=baseline,arms=dict(side_control=control,balanced_views=value)))['balanced_views'] for a,value in results['arms'].items()}
    records=results['arms']['regressed']['records'];stats=[];noise=[];changes=[];part=[];hashes={}
    for i,r in enumerate(records):
        saved={}
        for arm in results['arms']:
            path=OUT/arm/f'prediction_{i:03d}.pt';hashes[str(path.relative_to(OUT))]=fingerprint(path)
            saved[arm]=torch.load(path,map_location='cpu',weights_only=True)
            rows=matched_views(saved[arm]);stats.extend(dict(arm=arm,group=r['group'],scene=r['scene'],iou=x['iou'],found=x['found'],score=x['score']) for x in rows)
            if saved[arm]['oracle_assignment']!=saved['regressed']['oracle_assignment']:changes.append(dict(arm=arm,index=i,scene=r['scene']))
            if r['group']=='old_dual_pair':
                for row in rows:
                    for name,inds in [('body',list(range(7))+list(range(14,23))),('tail',list(range(7,14)))]:
                        valid=row['valid'][inds];errors=row['errors'][inds][valid]
                        if len(errors):part.append(dict(arm=arm,part=name,nme=float(errors.mean())))
        for arm in ARMS:
            a=saved[arm]['predictions']['instance_boxes_cxcywh'];b=saved[arm+'_corner_stress']['predictions']['instance_boxes_cxcywh']
            if arm=='token_only':torch.testing.assert_close(a,b,atol=0,rtol=0)
            noise.append(dict(arm=arm,group=r['group'],box_max_difference=float((a-b).abs().max())))
    box_summary=[]
    for arm in results['arms']:
        for group in sorted({r['group'] for r in records}):
            rows=[s for s in stats if s['arm']==arm and s['group']==group]
            box_summary.append(dict(arm=arm,group=group,n=len(rows),mean_iou=float(np.mean([s['iou'] for s in rows])),found=sum(s['found'] for s in rows)))
    parts=[dict(arm=arm,part=part_name,nme=float(np.mean([x['nme'] for x in part if x['arm']==arm and x['part']==part_name]))) for arm in results['arms'] for part_name in ('body','tail')]
    histories={a:json.loads((OUT/f'{a}_history.json').read_text()) for a in ARMS}
    training={a:dict(first200=float(np.mean([x['loss'] for x in h[:200]])),last200=float(np.mean([x['loss'] for x in h[-200:]])),max_grad_norm=max(x['grad_norm'] for x in h)) for a,h in histories.items()}
    report=dict(gates=gates,box_summary=box_summary,dual_parts=parts,assignment_changes=changes,noise_sensitivity=noise,training=training,
        production_promoted=False,results_sha256=fingerprint(OUT/'results.json'),prediction_sha256=hashes,analysis_code_sha256=fingerprint(ROOT/'tools/analyze_mask_box_refiner.py'),
        limitations=['Only boxes changed; coordinates and scores fixed. Assignment can still alter matched pose/3D metrics.',
            'Same reused synthetic development, one frozen seed17 checkpoint. No real Cylinder generalization claim.',
            'OFF and ON both learn a new box residual; this comparison isolates mask features, not extra training vs no training.',
            'Corner stress is a fixed synthetic perturbation, not proof of robustness to arbitrary segmentation failures.'])
    write_json(OUT/'acceptance.json',report)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for arm,h in histories.items():axes[0].plot(np.arange(50,801),np.convolve([x['loss'] for x in h],np.ones(50)/50,mode='valid'),label=arm)
    axes[0].set(title='50-slot mean box loss',xlabel='step');axes[0].legend()
    groups=('old_dual_pair','old_cylinder_side_pair')
    for j,arm in enumerate(('regressed',*ARMS)):
        axes[1].bar(np.arange(2)+(j-1)*.25,[results['arms'][arm]['summary'][g]['detection_aware_pck_0.1'] for g in groups],.25,label=arm)
    axes[1].set(title='Detection-aware PCK (coordinates unchanged)',xticks=np.arange(2),xticklabels=['dual','old single side']);axes[1].legend()
    fig.savefig(OUT/'comparison.png',dpi=145);plt.close(fig)
    cases=[('old_dual_pair',246,0),('old_dual_pair',296,1),('old_dual_pair',218,0),('old_cylinder_side_pair',31,0),('old_cylinder_side_pair',73,1),('old_cylinder_side_pair',91,1)]
    fig,axes=plt.subplots(2,3,figsize=(15,9),layout='constrained')
    for ax,(group,frame,v) in zip(axes.flat,cases):
        idx=next(i for i,r in enumerate(records) if r['group']==group and r['scene'].endswith(f'{frame:06d}'))
        files={a:torch.load(OUT/a/f'prediction_{idx:03d}.pt',map_location='cpu',weights_only=True) for a in ('regressed',*ARMS)}
        base=files['regressed'];r=base['record'];pixel=np.array(r['image_hw'][::-1]);ax.imshow(Image.open(ROOT/r['scene']/'blended_images'/f'{r["views"][v]:08d}.jpg').resize(tuple(pixel)))
        def draw(b,color,style):
            b=b.numpy();ax.add_patch(Rectangle((b[:2]-b[2:]/2)*pixel,*(b[2:]*pixel),fill=False,edgecolor=color,lw=1.2,linestyle=style))
        qs,ms=base['oracle_assignment'][v]
        for q,m in zip(qs,ms):
            draw(base['targets']['boxes_cxcywh'][0,v,m],'lime','-')
            for arm,color,style in [('regressed','magenta','--'),('token_only','orange','-.'),('mask_conditioned','cyan',':')]:draw(files[arm]['predictions']['instance_boxes_cxcywh'][0,v,q],color,style)
        ax.set_title(f'Frame {frame} / {r["physical_roles"][v]} cam{r["views"][v]}',fontsize=10);ax.axis('off')
    fig.suptitle('GT green / regressed magenta / token-only orange / mask-conditioned cyan\nFixed original query assignment for illustration; gates rematch with original evaluator')
    fig.savefig(OUT/'fixed_failures.png',dpi=135);plt.close(fig)
    assert all(fingerprint(OUT/p)==h for p,h in hashes.items())
    print(json.dumps(dict(gates={a:{k:v for k,v in g.items() if k!='gates'} for a,g in gates.items()},failed={a:[x for x in g['gates'] if not x['passed']] for a,g in gates.items()},box_summary=[x for x in box_summary if x['group'] in groups],training=training),indent=2))


if __name__=='__main__':main()
