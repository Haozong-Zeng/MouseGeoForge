"""Trainable-scope comparison with unchanged V1 acceptance thresholds."""
import copy
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from train_cylinder_pose_only import ROOT, OUTPUT, PARENT, torch
from run_depth_pose_ablation import write_json, fingerprint
from analyze_cylinder_view_balance import gate_results


def compare_results(current,parent):
    pose=current['arms']['pose_head_only']
    joint=parent['arms']['balanced_views']
    # Keep the prior gates, including the old equal-budget side-only reference.
    assembled=dict(baseline=parent['baseline'],config=current['config'],
        arms=dict(side_control=parent['arms']['side_control'],balanced_views=pose))
    gates=gate_results(assembled)['balanced_views']
    groups=current['config']['isolation_questions']['multiview_bottom_groups']
    ratios=[]
    for group in groups:
        key='bottom_body_keypoint_nme'
        a,b=joint['summary'][group][key],pose['summary'][group][key]
        matched=[(x['metrics'][key],y['metrics'][key]) for x,y in zip(joint['records'],pose['records']) if x['group']==group]
        assert all(x['scene']==y['scene'] and x['views']==y['views'] and x['group']==y['group']
                   for x,y in zip(joint['records'],pose['records']))
        ratios.append(dict(group=group,joint=a,pose_only=b,ratio=b/a,
            paired_scenes=len(matched),scenes_improved=sum(y<x for x,y in matched),
            meets_gain=b<=a*(1-current['config']['isolation_questions']['minimum_nme_gain_vs_joint_update'])))
    return dict(original_gates=gates,scope_comparison=ratios,
        all_multiview_bottom_groups_improved=all(r['meets_gain'] for r in ratios),
        production_promoted=False,one_seed=True,real_validation=False)


def main():
    current=json.loads((OUTPUT/'results.json').read_text())
    parent=json.loads((PARENT/'results.json').read_text())
    report=compare_results(current,parent)
    write_json(OUTPUT/'acceptance.json',dict(**report,
        results_sha256=fingerprint(OUTPUT/'results.json'),parent_results_sha256=fingerprint(PARENT/'results.json')))
    names=['original','all_heads','pose_only']
    summaries=[parent['baseline']['summary'],parent['arms']['balanced_views']['summary'],current['arms']['pose_head_only']['summary']]
    colors=['#708090','#b27649','#218d89']
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    panels=[(['bottom_only','side_bottom','sides_bottom','sides_bottom_top'],'bottom_body_keypoint_nme','Bottom body-keypoint NME'),
            (['old_cylinder_side_pair','old_dual_pair'],'detection_aware_pck_0.1','Original pair detection-aware PCK'),
            (['bottom_only','side_bottom','sides_bottom','sides_bottom_top'],'bottom_body_depth_absrel','Bottom depth AbsRel (pose-only frozen)')]
    for ax,(groups,key,title) in zip(axes,panels):
        for i,(name,summary,color) in enumerate(zip(names,summaries,colors)):
            ax.bar(np.arange(len(groups))+(i-1)*.25,[summary[g][key] for g in groups],.25,color=color,label=name)
        ax.set_title(title); ax.set_xticks(np.arange(len(groups)),groups,rotation=25,ha='right')
    axes[0].legend(fontsize=8)
    fig.suptitle('Synthetic Cylinder | exact balanced-view schedule | 800 steps | trainable scope only')
    fig.savefig(OUTPUT/'comparison.png',dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for folder,arm,name,color in ((PARENT,'balanced_views','all_heads',colors[1]),(OUTPUT,'pose_head_only','pose_only',colors[2])):
        history=json.loads((folder/arm/'history.json').read_text())
        loss=np.array([r['pose_loss'] for r in history]); window=50
        ax.plot(np.arange(window,len(loss)+1),np.convolve(loss,np.ones(window)/window,mode='valid'),label=name,color=color)
    ax.set(xlabel='step',ylabel='pose loss (50-step mean)',title='Identical scene/view sequence; no convergence claim');ax.legend()
    fig.savefig(OUTPUT/'pose_loss.png',dpi=150);plt.close(fig)
    # Same preselected hard baseline frame as V1, no candidate-dependent selection.
    baseline=parent['baseline']['records']
    worst=max((r for r in baseline if r['group']=='bottom_only'),key=lambda r:r['metrics']['bottom_body_keypoint_nme'])
    scene=worst['scene']; groups=['bottom_only','side_bottom','sides_bottom']
    fig,axes=plt.subplots(3,3,figsize=(12,12),layout='constrained')
    for i,(folder,arm,name) in enumerate(((PARENT,'initial','original'),(PARENT,'balanced_views','all_heads'),(OUTPUT,'pose_head_only','pose_only'))):
        for j,group in enumerate(groups):
            idx=next(k for k,r in enumerate(baseline) if r['scene']==scene and r['group']==group)
            item=torch.load(folder/arm/f'prediction_{idx:03d}.pt',weights_only=True,map_location='cpu')
            record=item['record']; v=record['physical_roles'].index('bottom');camera=record['views'][v]
            p,t=item['predictions'],item['targets'];qs,ms=item['oracle_assignment'][v];q=qs[ms.index(0)]
            xy=p['instance_keypoint_xy'][0,v,q].numpy()*768;gt=t['keypoint_xy'][0,v,0].numpy()*768
            valid=t['keypoint_valid'][0,v,0].numpy().copy();valid[7:14]=False
            ax=axes[i,j]
            ax.imshow(Image.open(ROOT/scene/'blended_images'/f'{camera:08d}.jpg').resize((768,768)))
            ax.scatter(gt[valid,0],gt[valid,1],c='lime',s=14,label='GT body')
            ax.scatter(xy[valid,0],xy[valid,1],c='magenta',s=20,marker='x',label='pred body')
            for a,b in zip(gt[valid],xy[valid]):ax.plot([a[0],b[0]],[a[1],b[1]],c='magenta',lw=.5,alpha=.5)
            lo=np.maximum(gt[valid].min(0)-90,0);hi=np.minimum(gt[valid].max(0)+90,768)
            ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1]);ax.axis('off')
            ax.set_title(f'{name} / {group}\nbody NME={item["metrics"]["bottom_body_keypoint_nme"]:.3f}',fontsize=10)
    axes[0,0].legend(fontsize=8)
    fig.suptitle(f'Synthetic {scene.rsplit("/",1)[-1]} | same hard baseline pose as V1\nGT green / prediction magenta; all off-crop points remain counted')
    fig.savefig(OUTPUT/'hardest_bottom_pose.png',dpi=150);plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
