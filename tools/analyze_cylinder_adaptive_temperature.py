"""Unchanged performance gates, matched foreground scales and fixed-frame image."""
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_cylinder_adaptive_temperature import ROOT,OUTPUT,PARENT,ARM
from run_depth_pose_ablation import torch,write_json,fingerprint
from analyze_cylinder_view_balance import gate_results


def main():
    torch.set_num_threads(2)
    assert json.loads((OUTPUT/'delivery_verification.json').read_text())['passed']
    current=json.loads((OUTPUT/'results.json').read_text());prior=json.loads((PARENT/'results.json').read_text())
    old=json.loads((ROOT/'output/cylinder_view_balance_v1/results.json').read_text())
    candidate=current['arms'][ARM];reference=prior['arms']['source_mix']
    gates=gate_results(dict(config=current['config'],baseline=prior['baseline'],arms=dict(side_control=old['arms']['side_control'],balanced_views=candidate)))['balanced_views']
    groups=['bottom_only','side_bottom','sides_bottom','sides_bottom_top'];key='bottom_body_keypoint_nme';comparison=[]
    for group in groups:
        a=reference['summary'][group][key];b=candidate['summary'][group][key]
        pairs=[]
        for x,y in zip(reference['records'],candidate['records']):
            assert all(x[k]==y[k] for k in ('group','scene','views'))
            if x['group']==group:pairs.append(dict(scene=x['scene'],off=x['metrics'][key],on=y['metrics'][key]))
        comparison.append(dict(group=group,off=a,on=b,relative_change=b/a-1,improved_scenes=sum(p['on']<p['off'] for p in pairs),pairs=pairs))
    ratio=float(np.mean([r['on'] for r in comparison[1:]])/np.mean([r['off'] for r in comparison[1:]]))
    scales=[]
    for idx,row in enumerate(candidate['records']):
        stored=torch.load(OUTPUT/ARM/f'temperature_{idx:03d}.pt',weights_only=True,map_location='cpu')
        saved=torch.load(OUTPUT/ARM/f'prediction_{idx:03d}.pt',weights_only=True,map_location='cpu')
        assert stored['oracle_assignment']==saved['oracle_assignment']
        for v,(qs,ms) in enumerate(saved['oracle_assignment']):
            for q,m in zip(qs,ms):
                valid=saved['targets']['keypoint_valid'][0,v,m]
                for k in torch.where(valid)[0].tolist():
                    scales.append(dict(group=row['group'],scene=row['scene'],role=saved['record']['physical_roles'][v],instance=m,
                        keypoint=k,part='tail' if 7<=k<14 else 'body',scale=float(stored['scale'][v,q,k])))
    scale_summary=[]
    for group in sorted({r['group'] for r in scales}):
        for role in sorted({r['role'] for r in scales if r['group']==group}):
            for part in ('body','tail'):
                vals=[r['scale'] for r in scales if (r['group'],r['role'],r['part'])==(group,role,part)]
                if vals:scale_summary.append(dict(group=group,role=role,part=part,count=len(vals),mean=float(np.mean(vals)),min=min(vals),max=max(vals)))
    history=json.loads((OUTPUT/ARM/'history.json').read_text());before=json.loads((PARENT/'source_mix/history.json').read_text())
    convergence=[]
    for tag,rows in [('off',before),('on',history)]:
        for kind in sorted({r['slot_kind'] for r in rows}):
            early=[r for r in rows if 401<=r['step']<=600 and r['slot_kind']==kind]
            late=[r for r in rows if 601<=r['step']<=800 and r['slot_kind']==kind]
            convergence.append(dict(arm=tag,kind=kind,**{k:dict(previous=float(np.mean([r[k] for r in early])),last=float(np.mean([r[k] for r in late]))) for k in ('loss','pose_loss','depth_loss')}))
    report=dict(original_gates=gates,comparison=comparison,multi_bottom_mean_ratio=ratio,substantive_improvement_at_800=ratio<=.95,
        scale_summary=scale_summary,convergence=convergence,method_decision='Pending convergence review; no automatic production promotion or added steps.',
        production_promoted=False,results_sha256=fingerprint(OUTPUT/'results.json'))
    write_json(OUTPUT/'acceptance_800.json',report);write_json(OUTPUT/'foreground_scales.json',scales)
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,(gs,k,title) in zip(axes,[(groups,key,'Bottom body NME'),(['old_cylinder_side_pair','old_dual_pair'],'detection_aware_pck_0.1','Original-pair PCK'),(groups,'bottom_body_depth_absrel','Bottom depth AbsRel')]):
        for j,(s,label,color) in enumerate([(reference['summary'],'OFF','#698fa9'),(candidate['summary'],'ON','#c38a4c')]):
            ax.bar(np.arange(len(gs))+(j-.5)*.35,[s[g][k] for g in gs],.35,label=label,color=color)
        ax.set_title(title);ax.set_xticks(range(len(gs)),gs,rotation=25,ha='right')
    axes[0].legend();fig.suptitle('Conditional temperature | same 800 steps, 768 | synthetic reused development')
    fig.savefig(OUTPUT/'comparison.png',dpi=145);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(14,7),layout='constrained')
    for row,(folder,label) in enumerate([(PARENT/'source_mix','OFF'),(OUTPUT/ARM,'ON')]):
        for col,g in enumerate(groups):
            idx=next(i for i,r in enumerate(reference['records']) if r['group']==g and r['scene'].endswith('000055'))
            saved=torch.load(folder/f'prediction_{idx:03d}.pt',weights_only=True,map_location='cpu');r=saved['record'];v=r['physical_roles'].index('bottom')
            qs,ms=saved['oracle_assignment'][v];q=qs[ms.index(0)];pixel=np.array(r['image_hw'][::-1]);t=saved['targets'];p=saved['predictions']
            xy=p['instance_keypoint_xy'][0,v,q].numpy()*pixel;gt=t['keypoint_xy'][0,v,0].numpy()*pixel
            valid=t['keypoint_valid'][0,v,0].numpy().copy();valid[7:14]=False;ax=axes[row,col]
            ax.imshow(Image.open(ROOT/r['scene']/'blended_images'/f'{r["views"][v]:08d}.jpg').resize(tuple(pixel)))
            ax.scatter(gt[valid,0],gt[valid,1],c='lime',s=12);ax.scatter(xy[valid,0],xy[valid,1],c='magenta',marker='x',s=16)
            for a,b in zip(gt[valid],xy[valid]):ax.plot([a[0],b[0]],[a[1],b[1]],c='magenta',lw=.5)
            lo=np.maximum(gt[valid].min(0)-90,0);hi=np.minimum(gt[valid].max(0)+90,pixel)
            ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1]);ax.axis('off');ax.set_title(f'{label} / {g}\nNME={saved["metrics"][key]:.3f}',fontsize=9)
    fig.suptitle('Fixed frame55 | GT green / predictions magenta')
    fig.savefig(OUTPUT/'hard_frame55.png',dpi=150);plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k not in ('comparison','scale_summary')},indent=2))
    print(json.dumps([{k:v for k,v in r.items() if k!='pairs'} for r in comparison],indent=2))


if __name__=='__main__':main()
