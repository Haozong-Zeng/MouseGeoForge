"""Fixed acceptance and visuals for coarse auxiliary supervision."""
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_cylinder_coarse_aux import ROOT,OUTPUT,PARENT,ARM
from run_depth_pose_ablation import torch,write_json,fingerprint
from analyze_cylinder_view_balance import gate_results


def main():
    current=json.loads((OUTPUT/'results.json').read_text());parent=json.loads((PARENT/'results.json').read_text())
    oldv1=json.loads((ROOT/'output/cylinder_view_balance_v1/results.json').read_text())
    candidate=current['arms'][ARM];reference=parent['arms']['combination_coverage']
    assembled=dict(config=current['config'],baseline=parent['baseline'],
        arms=dict(side_control=oldv1['arms']['side_control'],balanced_views=candidate))
    gates=gate_results(assembled)['balanced_views'];comparison=[]
    groups=['bottom_only','side_bottom','sides_bottom','sides_bottom_top']
    for group in groups:
        k='bottom_body_keypoint_nme';a=reference['summary'][group][k];b=candidate['summary'][group][k]
        pairs=[(x['metrics'][k],y['metrics'][k]) for x,y in zip(reference['records'],candidate['records']) if x['group']==group]
        comparison.append(dict(group=group,off=a,on=b,relative_change=b/a-1,paired_scenes=len(pairs),improved_scenes=sum(y<x for x,y in pairs)))
    report=dict(original_gates=gates,comparison=comparison,production_promoted=False,
        results_sha256=fingerprint(OUTPUT/'results.json'),limitations=['One seed, reused synthetic development, GT-box-assisted pose matching; no real Cylinder.'])
    write_json(OUTPUT/'acceptance.json',report)
    names=['original','coarse aux OFF','coarse aux ON'];summaries=[parent['baseline']['summary'],reference['summary'],candidate['summary']]
    colors=['#708090','#b27649','#218d89']
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
    for ax,(gs,k,title) in zip(axes,[(groups,'bottom_body_keypoint_nme','Bottom body NME (lower better)'),
        (['old_cylinder_side_pair','old_dual_pair'],'detection_aware_pck_0.1','Original-pair detection-aware PCK'),
        (groups,'bottom_body_depth_absrel','Bottom depth AbsRel (lower better)')]):
        for i,(name,s,color) in enumerate(zip(names,summaries,colors)):
            ax.bar(np.arange(len(gs))+(i-1)*.25,[s[g][k] for g in gs],.25,color=color,label=name)
        ax.set_title(title,fontsize=10);ax.set_xticks(range(len(gs)),gs,rotation=25,ha='right')
    axes[0].legend(fontsize=8);fig.suptitle('Synthetic Cylinder | same 800-step schedule | coarse-body auxiliary weight 0 vs 2.5')
    fig.savefig(OUTPUT/'comparison.png',dpi=150);plt.close(fig)
    off_history=json.loads((PARENT/'combination_coverage/history.json').read_text())
    on_history=json.loads((OUTPUT/ARM/'coarse_loss_history.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');window=50
    smooth=lambda x:np.convolve(x,np.ones(window)/window,mode='valid')
    steps=np.arange(window,len(on_history)+1)
    axes[0].plot(steps,smooth([r['pose_loss'] for r in off_history]),label='aux OFF',color=colors[1])
    axes[0].plot(steps,smooth([r['base_pose_loss'] for r in on_history]),label='aux ON: base loss only',color=colors[2])
    axes[0].set(title='Original pose objective, excluding new auxiliary',xlabel='step');axes[0].legend(fontsize=8)
    axes[1].plot(steps,smooth([r['coarse_body_loss'] for r in on_history]),color=colors[2])
    axes[1].set(title='Raw coarse-body auxiliary (before weight 2.5)',xlabel='step')
    fig.suptitle('50-step moving means; total loss is not directly comparable across objectives')
    fig.savefig(OUTPUT/'training_curves.png',dpi=150);plt.close(fig)
    baseline=parent['baseline']['records'];scene=next(r['scene'] for r in baseline if r['group']=='bottom_only' and r['scene'].endswith('000055'))
    fig,axes=plt.subplots(2,3,figsize=(12,8),layout='constrained')
    for i,(folder,arm,name) in enumerate(((PARENT,'combination_coverage','aux OFF'),(OUTPUT,ARM,'aux ON'))):
        for j,group in enumerate(groups[:3]):
            idx=next(k for k,r in enumerate(baseline) if r['scene']==scene and r['group']==group)
            saved=torch.load(folder/arm/f'prediction_{idx:03d}.pt',weights_only=True,map_location='cpu')
            r=saved['record'];v=r['physical_roles'].index('bottom');qs,ms=saved['oracle_assignment'][v];q=qs[ms.index(0)]
            p,t=saved['predictions'],saved['targets'];h,w=r['image_hw'];pixel=np.array([w,h])
            xy=p['instance_keypoint_xy'][0,v,q].numpy()*pixel;gt=t['keypoint_xy'][0,v,0].numpy()*pixel
            valid=t['keypoint_valid'][0,v,0].numpy().copy();valid[7:14]=False
            ax=axes[i,j];ax.imshow(Image.open(ROOT/scene/'blended_images'/f'{r["views"][v]:08d}.jpg').resize((w,h)))
            ax.scatter(gt[valid,0],gt[valid,1],c='lime',s=14);ax.scatter(xy[valid,0],xy[valid,1],c='magenta',marker='x',s=20)
            for a,b in zip(gt[valid],xy[valid]):ax.plot([a[0],b[0]],[a[1],b[1]],color='magenta',lw=.5,alpha=.5)
            lo=np.maximum(gt[valid].min(0)-90,0);hi=np.minimum(gt[valid].max(0)+90,pixel)
            ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1]);ax.axis('off')
            ax.set_title(f'{name} / {group}\nNME={saved["metrics"]["bottom_body_keypoint_nme"]:.3f}')
    fig.suptitle('Fixed hard frame 55 | GT green, prediction magenta; off-crop points counted')
    fig.savefig(OUTPUT/'hardest_bottom_pose.png',dpi=150);plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
