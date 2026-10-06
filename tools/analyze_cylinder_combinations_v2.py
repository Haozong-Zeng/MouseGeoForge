"""Unchanged acceptance gates and fixed-frame visual comparison for sampling V2."""
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from train_cylinder_combinations_v2 import ROOT, OUTPUT, PARENT, ARM
from run_depth_pose_ablation import torch, write_json, fingerprint
from analyze_cylinder_view_balance import gate_results


def main():
    current=json.loads((OUTPUT/'results.json').read_text())
    parent=json.loads((PARENT/'results.json').read_text())
    protocol=json.loads((OUTPUT/'protocol.json').read_text())
    candidate=current['arms'][ARM]; old=parent['arms']['balanced_views']
    assembled=dict(baseline=parent['baseline'],config=current['config'],
        arms=dict(side_control=parent['arms']['side_control'],balanced_views=candidate))
    gates=gate_results(assembled)['balanced_views']
    comparisons=[]
    for group in ('bottom_only','side_bottom','sides_bottom','sides_bottom_top'):
        key='bottom_body_keypoint_nme'
        pairs=[(a['metrics'][key],b['metrics'][key]) for a,b in zip(old['records'],candidate['records']) if a['group']==group]
        a,b=old['summary'][group][key],candidate['summary'][group][key]
        comparisons.append(dict(group=group,old=a,new=b,relative_change=b/a-1,
            paired_scenes=len(pairs),improved_scenes=sum(y<x for x,y in pairs)))
    training_orders=[protocol['records'][x[0]]['views'] for i,x in enumerate(protocol['arms'][ARM])
                     if protocol['slots'][i]['kind']=='cylinder']
    evaluation_coverage={}
    for item in protocol['evaluation']:
        if item['group'].startswith('old_'): continue
        order=protocol['records'][item['key']]['views']
        evaluation_coverage[item['group']]=dict(views=order,
            exact_order_slots=sum(x==order for x in training_orders),
            unordered_subset_slots=sum(sorted(x)==sorted(order) for x in training_orders),
            reference_and_subset_slots=sum(x[0]==order[0] and sorted(x)==sorted(order) for x in training_orders))
    report=dict(original_gates=gates,comparison=comparisons,development_combination_coverage=evaluation_coverage,production_promoted=False,
        results_sha256=fingerprint(OUTPUT/'results.json'),parent_results_sha256=fingerprint(PARENT/'results.json'),
        limitations=['Single seed and reused synthetic development rig/assets; no real Cylinder evaluation.',
            'Sampling changes subset diversity, reference order and physical-role frequencies jointly.',
            'Triangulation is GT-box-assisted and conditional on support; not end-to-end multi-instance 3D.'])
    write_json(OUTPUT/'acceptance.json',report)
    audit=json.loads((OUTPUT/'coverage.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(10,5),layout='constrained')
    for ax,key,title in zip(axes,('previous','new'),('Old V2-view sampling','New V2-view sampling')):
        counts=np.zeros((8,8),dtype=int)
        for pair,count in audit[key]['ordered_pairs'].items():
            a,b=map(int,pair.split(','));counts[a,b]=count
        ax.imshow(counts,cmap='Blues',vmin=0,vmax=24)
        for (a,b),count in np.ndenumerate(counts):
            if count:ax.text(b,a,str(count),ha='center',va='center',color='white' if count>14 else 'black',fontsize=9)
        labels=['S0','S1','S2','S3','S4','S5','Bottom','Top']
        ax.set_xticks(range(8),labels,rotation=45,ha='right');ax.set_yticks(range(8),labels)
        ax.set(title=title,xlabel='Second camera',ylabel='First / reference camera')
    fig.suptitle('Cylinder dual-view slot coverage (100 slots each; zeros blank)')
    fig.savefig(OUTPUT/'pair_coverage.png',dpi=150);plt.close(fig)
    names=['original','old sampling','new sampling']
    summaries=[parent['baseline']['summary'],old['summary'],candidate['summary']]
    colors=['#708090','#b27649','#218d89']
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    panels=[(['bottom_only','side_bottom','sides_bottom','sides_bottom_top'],'bottom_body_keypoint_nme','Bottom body-keypoint NME (lower better)'),
        (['old_cylinder_side_pair','old_dual_pair'],'detection_aware_pck_0.1','Original-pair detection-aware PCK (higher better)'),
        (['bottom_only','side_bottom','sides_bottom','sides_bottom_top'],'bottom_body_depth_absrel','Bottom depth AbsRel (lower better)')]
    for ax,(groups,key,title) in zip(axes,panels):
        for i,(name,summary,color) in enumerate(zip(names,summaries,colors)):
            ax.bar(np.arange(len(groups))+(i-1)*.25,[summary[g][key] for g in groups],.25,color=color,label=name)
        ax.set_title(title,fontsize=10);ax.set_xticks(np.arange(len(groups)),groups,rotation=25,ha='right')
    axes[0].legend(fontsize=8)
    fig.suptitle('Synthetic Cylinder | same 800-step source/V budget | new combinations and reference order')
    fig.savefig(OUTPUT/'comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for folder,arm,name,color in ((PARENT,'balanced_views','old sampling',colors[1]),(OUTPUT,ARM,'new sampling',colors[2])):
        history=json.loads((folder/arm/'history.json').read_text())
        for ax,key in zip(axes,('pose_loss','depth_loss')):
            loss=np.array([r[key] for r in history]);window=50
            ax.plot(np.arange(window,len(loss)+1),np.convolve(loss,np.ones(window)/window,mode='valid'),label=name,color=color)
            ax.set(xlabel='step',ylabel=key+' (50-step mean)');ax.legend()
    fig.suptitle('View content differs: loss values alone are not a convergence comparison')
    fig.savefig(OUTPUT/'training_curves.png',dpi=150);plt.close(fig)
    baseline=parent['baseline']['records']
    worst=max((r for r in baseline if r['group']=='bottom_only'),key=lambda r:r['metrics']['bottom_body_keypoint_nme'])
    scene=worst['scene'];groups=['bottom_only','side_bottom','sides_bottom']
    fig,axes=plt.subplots(3,3,figsize=(12,12),layout='constrained')
    for i,(folder,arm,name) in enumerate(((PARENT,'initial','original'),(PARENT,'balanced_views','old sampling'),(OUTPUT,ARM,'new sampling'))):
        for j,group in enumerate(groups):
            idx=next(k for k,r in enumerate(baseline) if r['scene']==scene and r['group']==group)
            item=torch.load(folder/arm/f'prediction_{idx:03d}.pt',weights_only=True,map_location='cpu')
            record=item['record'];v=record['physical_roles'].index('bottom');cam=record['views'][v]
            p,t=item['predictions'],item['targets'];qs,ms=item['oracle_assignment'][v];q=qs[ms.index(0)]
            h,w=record['image_hw'];scale=np.array([w,h])
            xy=p['instance_keypoint_xy'][0,v,q].numpy()*scale;gt=t['keypoint_xy'][0,v,0].numpy()*scale
            valid=t['keypoint_valid'][0,v,0].numpy().copy();valid[7:14]=False
            ax=axes[i,j];ax.imshow(Image.open(ROOT/scene/'blended_images'/f'{cam:08d}.jpg').resize((w,h)))
            ax.scatter(gt[valid,0],gt[valid,1],c='lime',s=14,label='GT body')
            ax.scatter(xy[valid,0],xy[valid,1],c='magenta',s=20,marker='x',label='pred body')
            for a,b in zip(gt[valid],xy[valid]):ax.plot([a[0],b[0]],[a[1],b[1]],c='magenta',lw=.5,alpha=.5)
            lo=np.maximum(gt[valid].min(0)-90,0);hi=np.minimum(gt[valid].max(0)+90,scale)
            ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1]);ax.axis('off')
            ax.set_title(f'{name} / {group}\nbody NME={item["metrics"]["bottom_body_keypoint_nme"]:.3f}',fontsize=10)
    axes[0,0].legend(fontsize=8)
    fig.suptitle(f'Synthetic {scene.rsplit("/",1)[-1]} | same preselected V1 hard pose\nGT green / prediction magenta; off-crop points remain counted')
    fig.savefig(OUTPUT/'hardest_bottom_pose.png',dpi=150);plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
