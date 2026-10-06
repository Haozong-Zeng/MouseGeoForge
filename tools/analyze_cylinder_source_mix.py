"""Fixed 17-gate evaluation and preselected visual comparisons for source mixing."""
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_cylinder_source_mix import ROOT,OUTPUT,PARENT,ARM
from run_depth_pose_ablation import torch,write_json,fingerprint
from analyze_cylinder_view_balance import gate_results


def main():
    assert json.loads((OUTPUT/'delivery_verification.json').read_text())['passed']
    current=json.loads((OUTPUT/'results.json').read_text());parent=json.loads((PARENT/'results.json').read_text())
    oldv1=json.loads((ROOT/'output/cylinder_view_balance_v1/results.json').read_text())
    candidate=current['arms'][ARM];reference=parent['arms']['combination_coverage']
    assembled=dict(config=current['config'],baseline=parent['baseline'],arms=dict(side_control=oldv1['arms']['side_control'],balanced_views=candidate))
    gates=gate_results(assembled)['balanced_views'];comparison=[]
    groups=['bottom_only','side_bottom','sides_bottom','sides_bottom_top'];key='bottom_body_keypoint_nme'
    for group in groups:
        a=reference['summary'][group][key];b=candidate['summary'][group][key]
        pairs=[]
        for x,y in zip(reference['records'],candidate['records']):
            assert all(x[k]==y[k] for k in ('group','scene','views'))
            if x['group']==group:pairs.append(dict(scene=x['scene'],off=x['metrics'][key],on=y['metrics'][key]))
        comparison.append(dict(group=group,off=a,on=b,relative_change=b/a-1,paired_scenes=len(pairs),
            improved_scenes=sum(p['on']<p['off'] for p in pairs),pairs=pairs))
    ratio=float(np.mean([r['on'] for r in comparison[1:]])/np.mean([r['off'] for r in comparison[1:]]))
    report=dict(original_gates=gates,comparison=comparison,multi_bottom_mean_ratio=ratio,
        substantive_improvement=ratio<=.95,consecutive_no_improvement_methods=0 if ratio<=.95 else 2,
        production_promoted=False,results_sha256=fingerprint(OUTPUT/'results.json'),
        limitations=['One seed; repeatedly reused synthetic development, no real Cylinder evaluation.',
            'Training source mixture changes pose, rig, asset and native resolution together; not a pure pose-diversity effect.',
            'GT-box-assisted pose matching and known-calibration conditional 3D; not end-to-end association.'])
    write_json(OUTPUT/'acceptance.json',report)
    names=['original','ClimbWall-only V2','source mixture'];summaries=[parent['baseline']['summary'],reference['summary'],candidate['summary']]
    colors=['#708090','#b27649','#218d89']
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
    for ax,(gs,k,title) in zip(axes,[(groups,key,'Bottom body NME (lower better)'),
        (['old_cylinder_side_pair','old_dual_pair'],'detection_aware_pck_0.1','Original-pair detection-aware PCK'),
        (groups,'bottom_body_depth_absrel','Bottom depth AbsRel (lower better)')]):
        for i,(name,s,color) in enumerate(zip(names,summaries,colors)):
            ax.bar(np.arange(len(gs))+(i-1)*.25,[s[g][k] for g in gs],.25,color=color,label=name)
        ax.set_title(title,fontsize=10);ax.set_xticks(range(len(gs)),gs,rotation=25,ha='right')
    axes[0].legend(fontsize=8);fig.suptitle('Synthetic Cylinder | 768 | fixed 800 steps, V/H/W and replay | source mixture only')
    fig.savefig(OUTPUT/'comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');window=50
    for folder,arm,name,color in ((PARENT,'combination_coverage',names[1],colors[1]),(OUTPUT,ARM,names[2],colors[2])):
        history=json.loads((folder/arm/'history.json').read_text())
        for ax,k in zip(axes,('pose_loss','depth_loss')):
            ax.plot(np.arange(window,len(history)+1),np.convolve([r[k] for r in history],np.ones(window)/window,mode='valid'),label=name,color=color)
            ax.set(title=k+' / 50-step mean',xlabel='step');ax.legend(fontsize=8)
    fig.suptitle('Training observations differ: raw losses do not establish relative convergence')
    fig.savefig(OUTPUT/'training_curves.png',dpi=150);plt.close(fig)
    baseline=parent['baseline']['records'];scene=next(r['scene'] for r in baseline if r['group']=='bottom_only' and r['scene'].endswith('000055'))
    fig,axes=plt.subplots(3,4,figsize=(15,11),layout='constrained')
    for i,(folder,arm,name) in enumerate(((PARENT,'initial','original'),(PARENT,'combination_coverage','V2'),(OUTPUT,ARM,'mixture'))):
        for j,group in enumerate(groups):
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
            ax.set_title(f'{name} / {group}\nNME={saved["metrics"][key]:.3f}',fontsize=9)
    fig.suptitle('Preselected hard frame 55 | GT green, predicted magenta | off-crop errors still counted')
    fig.savefig(OUTPUT/'hardest_bottom_pose.png',dpi=150);plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k!='comparison'},indent=2))
    print(json.dumps([{k:v for k,v in r.items() if k!='pairs'} for r in comparison],indent=2))


if __name__=='__main__':main()
