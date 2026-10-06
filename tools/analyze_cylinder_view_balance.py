"""Fixed gates and diagnostic figures for the synthetic Cylinder pilot."""
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from train_cylinder_view_balance import ROOT, OUTPUT, torch, write_json, fingerprint


def gate_results(results):
    baseline = results['baseline']['summary']
    cfg = results['config']['acceptance']
    report = {}
    for arm, result in results['arms'].items():
        values = result['summary']
        gates = []
        def check(group, metric, limit, relation='<='):
            value = values[group][metric]
            gates.append(dict(group=group,metric=metric,value=value,limit=limit,
                              relation=relation,passed=value is not None and (value<=limit if relation=='<=' else value>=limit)))
        for group in ('bottom_only','side_bottom'):
            key = 'bottom_body_keypoint_nme'
            check(group,key,baseline[group][key]*cfg['bottom_body_nme_ratio_max'])
            if arm=='balanced_views':
                key_control = results['arms']['side_control']['summary'][group][key]
                check(group,key,key_control*cfg['bottom_vs_control_ratio_max'])
        for group in ('old_dual_pair','old_cylinder_side_pair'):
            check(group,'box_matched_nme',baseline[group]['box_matched_nme']*cfg['retention_nme_ratio_max'])
            check(group,'detection_aware_pck_0.1',baseline[group]['detection_aware_pck_0.1']-cfg['retention_pck_drop_max'],'>=')
            check(group,'body_depth_absrel',baseline[group]['body_depth_absrel']*cfg['body_depth_ratio_max'])
        for group in ('bottom_only','side_bottom','sides_bottom','sides_bottom_top'):
            key = 'bottom_body_depth_absrel'
            check(group,key,baseline[group][key]*cfg['body_depth_ratio_max'])
        for group in ('top_only','sides_top','sides_bottom_top'):
            key = 'top_body_keypoint_nme'
            check(group,key,baseline[group][key]*cfg['top_body_nme_ratio_max'])
        report[arm] = dict(passed=all(g['passed'] for g in gates),
                          passed_count=sum(g['passed'] for g in gates),total=len(gates),gates=gates)
    return report


def main():
    result = json.loads((OUTPUT/'results.json').read_text())
    report = gate_results(result)
    write_json(OUTPUT/'acceptance.json',dict(arms=report,production_promoted=False,
        results_sha256=fingerprint(OUTPUT/'results.json'),
        limitations=['One seed, same synthetic rig/animal asset; reused development poses',
                     '3D metrics use GT-assisted association and conditional supported subsets',
                     'No real Cylinder evaluation']))
    names = ['initial']+list(result['arms'])
    summaries = [result['baseline']['summary']]+[result['arms'][a]['summary'] for a in result['arms']]
    colors = ['#708090','#c18d42','#278a8b']
    fig,axes = plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    panels = [(['bottom_only','side_bottom','sides_bottom','sides_bottom_top'],'bottom_body_keypoint_nme','Bottom body-keypoint NME'),
              (['old_cylinder_side_pair','old_dual_pair'],'box_matched_nme','Original pair Core23 NME'),
              (['bottom_only','side_bottom','sides_bottom','sides_bottom_top'],'bottom_body_depth_spearman','Bottom depth rank correlation')]
    for ax,(groups,key,title) in zip(axes,panels):
        x=np.arange(len(groups))
        for i,(name,summary,color) in enumerate(zip(names,summaries,colors)):
            y=[summary[g][key] for g in groups]
            ax.bar(x+(i-1)*.25,y,.25,label=name,color=color)
        ax.set_title(title); ax.set_xticks(x,groups,rotation=25,ha='right')
        ax.axhline(0,color='gray',lw=.5)
    axes[0].legend(fontsize=8)
    fig.suptitle('Synthetic Cylinder | 768 px | 2 x 800 steps | frozen encoder | one seed')
    fig.savefig(OUTPUT/'comparison.png',dpi=150); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for arm,color in zip(result['arms'],colors[1:]):
        history=json.loads((OUTPUT/arm/'history.json').read_text())
        for ax,key in zip(axes,('pose_loss','depth_loss')):
            y=np.array([r[key] for r in history]); window=50
            smooth=np.convolve(y,np.ones(window)/window,mode='valid')
            ax.plot(np.arange(window,len(y)+1),smooth,label=arm,color=color)
            ax.set(title=key+' (50-step mean)',xlabel='optimizer step'); ax.legend()
    fig.suptitle('Loss values reflect different view content; not a convergence claim')
    fig.savefig(OUTPUT/'training_curves.png',dpi=150); plt.close(fig)
    initial=result['baseline']['records']
    worst=max((r for r in initial if r['group']=='bottom_only'),key=lambda r:r['metrics']['bottom_body_keypoint_nme'])
    scene=worst['scene']; frame=int(scene.rsplit('_',1)[-1])
    groups=['bottom_only','side_bottom','sides_bottom']
    fig,axes=plt.subplots(3,3,figsize=(12,12),layout='constrained')
    for i,name in enumerate(names):
        for j,group in enumerate(groups):
            idx=next(k for k,r in enumerate(initial) if r['scene']==scene and r['group']==group)
            item=torch.load(OUTPUT/name/f'prediction_{idx:03d}.pt',weights_only=True,map_location='cpu')
            record=item['record']; v=record['physical_roles'].index('bottom'); camera=record['views'][v]
            image=Image.open(ROOT/scene/'blended_images'/f'{camera:08d}.jpg').resize((768,768))
            p,t=item['predictions'],item['targets']; qs,ms=item['oracle_assignment'][v]; q=qs[ms.index(0)]
            xy=p['instance_keypoint_xy'][0,v,q].numpy()*768
            gt=t['keypoint_xy'][0,v,0].numpy()*768
            valid=t['keypoint_valid'][0,v,0].numpy().copy(); valid[7:14]=False
            ax=axes[i,j]; ax.imshow(image)
            ax.scatter(gt[valid,0],gt[valid,1],c='lime',s=14,label='GT body')
            ax.scatter(xy[valid,0],xy[valid,1],c='magenta',s=20,marker='x',label='pred body')
            for g,pred in zip(gt[valid],xy[valid]):
                ax.plot([g[0],pred[0]],[g[1],pred[1]],c='magenta',lw=.5,alpha=.5)
            lo=np.maximum(gt[valid].min(0)-90,0); hi=np.minimum(gt[valid].max(0)+90,768)
            ax.set_xlim(lo[0],hi[0]); ax.set_ylim(hi[1],lo[1]); ax.axis('off')
            nme=item['metrics']['bottom_body_keypoint_nme']
            ax.set_title(f'{name} / {group}\nbody NME={nme:.3f}',fontsize=10)
    axes[0,0].legend(fontsize=8)
    fig.suptitle(f'Synthetic frame {frame}: hardest baseline bottom-only of 8 development frames\nSame frame/input/crop per column; body points only; predictions outside crop remain counted in metrics')
    fig.savefig(OUTPUT/'hardest_bottom_pose.png',dpi=150); plt.close(fig)
    print(json.dumps({a:{k:v for k,v in r.items() if k!='gates'} for a,r in report.items()},indent=2))


if __name__=='__main__': main()
