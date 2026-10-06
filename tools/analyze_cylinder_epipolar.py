"""Original acceptance gates plus saved-output geometric mechanism checks."""
from collections import Counter
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_cylinder_epipolar import ROOT,OUTPUT,PARENT,ARM
from run_depth_pose_ablation import torch,write_json,fingerprint
from analyze_cylinder_view_balance import gate_results
from mousegeoforge.retention_geometry_audit import geometry_points


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
    geo=[];sources={}
    for tag,folder in (('off',PARENT/'source_mix'),('on',OUTPUT/ARM)):
        for i,r in enumerate(reference['records']):
            if r['group'] not in groups[1:]:continue
            path=folder/f'prediction_{i:03d}.pt';sources[str(path.relative_to(ROOT))]=fingerprint(path)
            saved=torch.load(path,weights_only=True,map_location='cpu')
            geo.extend(dict(checkpoint=tag,group=r['group'],scene=r['scene'],**point) for point in geometry_points(saved))
    mechanism=[]
    for g in groups[1:]:
        a={(p['scene'],p['keypoint']):p for p in geo if p['group']==g and p['checkpoint']=='off'}
        b={(p['scene'],p['keypoint']):p for p in geo if p['group']==g and p['checkpoint']=='on'}
        common=[k for k in a if a[k]['supported'] and b[k]['supported']]
        row=dict(group=g,off_supported=sum(p['supported'] for p in a.values()),on_supported=sum(p['supported'] for p in b.values()),
            total=len(a),common_points=len(common),transitions=dict(Counter(f'{int(a[k]["supported"])}->{int(b[k]["supported"])}' for k in a)),
            off_common_error=float(np.mean([a[k]['error_native'] for k in common])) if common else None,
            on_common_error=float(np.mean([b[k]['error_native'] for k in common])) if common else None)
        if g=='side_bottom':
            row.update(off_epipolar_px=float(np.mean([p['epipolar_distance_px'] for p in a.values()])),on_epipolar_px=float(np.mean([p['epipolar_distance_px'] for p in b.values()])))
        mechanism.append(row)
    report=dict(original_gates=gates,comparison=comparison,geometry=mechanism,multi_bottom_mean_ratio=ratio,
        substantive_improvement=ratio<=.95,consecutive_no_improvement_methods=0 if ratio<=.95 else 1,production_promoted=False,
        results_sha256=fingerprint(OUTPUT/'results.json'),geometry_input_sha256=sources,
        limitations=['One seed, reused synthetic development; no real Cylinder.', 'Known calibration and GT-assisted instance correspondence; no end-to-end association.',
            'Common-support 3D is conditional and point-weighted; do not equate to whole-population scene-macro error.'])
    write_json(OUTPUT/'acceptance.json',report);write_json(OUTPUT/'geometry_points.json',geo)
    names=['original','epipolar OFF','epipolar ON'];summaries=[prior['baseline']['summary'],reference['summary'],candidate['summary']];colors=['#708090','#b27649','#218d89']
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for ax,(gs,k,title) in zip(axes,[(groups,key,'Bottom body NME'),(['old_cylinder_side_pair','old_dual_pair'],'detection_aware_pck_0.1','Original-pair Core23 PCK'),(groups,'bottom_body_depth_absrel','Bottom depth AbsRel')]):
        for j,(name,s,color) in enumerate(zip(names,summaries,colors)):
            ax.bar(np.arange(len(gs))+(j-1)*.25,[s[g][k] for g in gs],.25,label=name,color=color)
        ax.set_title(title);ax.set_xticks(range(len(gs)),gs,rotation=25,ha='right')
    axes[0].legend(fontsize=8);fig.suptitle('768 | exact same 800 source-mixture steps | epipolar auxiliary weight 0 vs 1')
    fig.savefig(OUTPUT/'comparison.png',dpi=145);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for folder,arm,name,color in ((PARENT,'source_mix',names[1],colors[1]),(OUTPUT,ARM,names[2],colors[2])):
        history=json.loads((folder/arm/'history.json').read_text())
        if arm==ARM:
            aux=json.loads((folder/arm/'epipolar_loss_history.json').read_text())
            pose=[p['base_pose_loss'] for p in aux]
            axes[1].plot(range(50,801),np.convolve([p['auxiliary_loss'] for p in aux],np.ones(50)/50,mode='valid'),color=color)
        else:pose=[p['pose_loss'] for p in history]
        axes[0].plot(range(50,801),np.convolve(pose,np.ones(50)/50,mode='valid'),color=color,label=name)
    axes[0].set(title='Base pose loss (excludes auxiliary)',xlabel='step');axes[0].legend();axes[1].set(title='Epipolar auxiliary (weight1)',xlabel='step')
    fig.suptitle('50-step moving means; no validation checkpoint selection')
    fig.savefig(OUTPUT/'training_curves.png',dpi=145);plt.close(fig)
    fig,axes=plt.subplots(3,4,figsize=(15,11),layout='constrained')
    for row,(folder,name) in enumerate(((PARENT/'initial','original'),(PARENT/'source_mix','OFF'),(OUTPUT/ARM,'ON'))):
        for col,g in enumerate(groups):
            idx=next(i for i,r in enumerate(reference['records']) if r['group']==g and r['scene'].endswith('000055'))
            s=torch.load(folder/f'prediction_{idx:03d}.pt',weights_only=True,map_location='cpu');r=s['record'];v=r['physical_roles'].index('bottom')
            qs,ms=s['oracle_assignment'][v];q=qs[ms.index(0)];pixel=np.array(r['image_hw'][::-1]);t=s['targets'];p=s['predictions']
            xy=p['instance_keypoint_xy'][0,v,q].numpy()*pixel;gt=t['keypoint_xy'][0,v,0].numpy()*pixel
            valid=t['keypoint_valid'][0,v,0].numpy().copy();valid[7:14]=False;ax=axes[row,col]
            ax.imshow(Image.open(ROOT/r['scene']/'blended_images'/f'{r["views"][v]:08d}.jpg').resize(tuple(pixel)))
            ax.scatter(gt[valid,0],gt[valid,1],c='lime',s=13);ax.scatter(xy[valid,0],xy[valid,1],c='magenta',marker='x',s=18)
            for a,b in zip(gt[valid],xy[valid]):ax.plot([a[0],b[0]],[a[1],b[1]],c='magenta',lw=.5)
            lo=np.maximum(gt[valid].min(0)-90,0);hi=np.minimum(gt[valid].max(0)+90,pixel)
            ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1]);ax.axis('off');ax.set_title(f'{name} / {g}\nNME={s["metrics"][key]:.3f}',fontsize=9)
    fig.suptitle('Fixed frame55 | GT green, predictions magenta; off-crop errors included')
    fig.savefig(OUTPUT/'hard_frame55.png',dpi=145);plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k not in ('comparison','geometry_input_sha256')},indent=2))
    print(json.dumps([{k:v for k,v in r.items() if k!='pairs'} for r in comparison],indent=2))


if __name__=='__main__':main()
