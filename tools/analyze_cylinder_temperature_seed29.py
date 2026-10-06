"""Paired seed repeat, fixed performance gates and support-aware 3D diagnostics."""
import argparse
from collections import Counter,defaultdict
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from train_cylinder_temperature_seed29 import ROOT,OUTPUT,ARMS,first
from run_depth_pose_ablation import torch,write_json,fingerprint
from analyze_cylinder_view_balance import gate_results
from mousegeoforge.retention_geometry_audit import geometry_points,matched_views
from mousegeoforge.localization_audit import refinement_statistics
from retry_cylinder_epipolar_localization import summarize_default_temperature

GROUPS=('bottom_only','side_bottom','sides_bottom','sides_bottom_top')
KEY='bottom_body_keypoint_nme'


def geometry(folders,records,out):
    points=[];sources={};parity=[]
    for arm,folder in folders.items():
        for idx,r in enumerate(records):
            if r['group'] not in GROUPS[1:]:continue
            path=folder/f'prediction_{idx:03d}.pt';sources[str(path.relative_to(ROOT))]=fingerprint(path)
            s=torch.load(path,weights_only=True,map_location='cpu')
            local=geometry_points(s)
            parity.append(abs(sum(p['supported'] for p in local)/len(local)-s['metrics']['calibrated_body3d_coverage']))
            errors=[p['error_native'] for p in local if p['supported']]
            if errors:parity.append(abs(float(np.mean(errors))-s['metrics']['calibrated_body3d_error_native']))
            points.extend(dict(arm=arm,group=r['group'],scene=r['scene'],**p) for p in local)
    summary=[]
    for group in GROUPS[1:]:
        a={(p['scene'],p['keypoint']):p for p in points if p['group']==group and p['arm']=='off'}
        b={(p['scene'],p['keypoint']):p for p in points if p['group']==group and p['arm']=='on'}
        assert a.keys()==b.keys()
        common=[k for k in a if a[k]['supported'] and b[k]['supported']]
        row=dict(group=group,total=len(a),off_supported=sum(p['supported'] for p in a.values()),on_supported=sum(p['supported'] for p in b.values()),
            common=len(common),off_common_error_native=float(np.mean([a[k]['error_native'] for k in common])) if common else None,
            on_common_error_native=float(np.mean([b[k]['error_native'] for k in common])) if common else None,
            transitions=dict(Counter(f'{int(a[k]["supported"])}->{int(b[k]["supported"])}' for k in a)))
        if group=='side_bottom':row.update(off_epipolar_px=float(np.mean([p['epipolar_distance_px'] for p in a.values()])),on_epipolar_px=float(np.mean([p['epipolar_distance_px'] for p in b.values()])))
        summary.append(row)
    assert max(parity)<1e-5
    assert all(fingerprint(ROOT/p)==h for p,h in sources.items())
    report=dict(summary=summary,source_sha256=sources,optimizer_steps=0,encoder_forwards=0,saved_metric_max_difference=max(parity),
        code_sha256={p:fingerprint(ROOT/p) for p in ('tools/analyze_cylinder_temperature_seed29.py','mousegeoforge/retention_geometry_audit.py','vggt-omega/vggt_omega/utils/robust_pose_geometry.py')},
        limitations=['Known GT calibration and GT-assisted instance association; not end-to-end multi-mouse reconstruction.',
            'Conditional common-support point error must be read together with support coverage.'])
    write_json(out/'paired_geometry.json',report);write_json(out/'paired_geometry_points.json',points)
    return summary


def fixed_visual(folders,records,out):
    fig,axes=plt.subplots(2,4,figsize=(14,7),layout='constrained')
    for row,(arm,folder) in enumerate(folders.items()):
        for col,g in enumerate(GROUPS):
            idx=next(i for i,r in enumerate(records) if r['group']==g and r['scene'].endswith('000055'))
            s=torch.load(folder/f'prediction_{idx:03d}.pt',weights_only=True,map_location='cpu');r=s['record'];v=r['physical_roles'].index('bottom')
            qs,ms=s['oracle_assignment'][v];q=qs[ms.index(0)];pixel=np.array(r['image_hw'][::-1]);t=s['targets'];p=s['predictions']
            xy=p['instance_keypoint_xy'][0,v,q].numpy()*pixel;gt=t['keypoint_xy'][0,v,0].numpy()*pixel
            valid=t['keypoint_valid'][0,v,0].numpy().copy();valid[7:14]=False;ax=axes[row,col]
            ax.imshow(Image.open(ROOT/r['scene']/'blended_images'/f'{r["views"][v]:08d}.jpg').resize(tuple(pixel)))
            ax.scatter(gt[valid,0],gt[valid,1],c='lime',s=12);ax.scatter(xy[valid,0],xy[valid,1],c='magenta',marker='x',s=16)
            for a,b in zip(gt[valid],xy[valid]):ax.plot([a[0],b[0]],[a[1],b[1]],c='magenta',lw=.5)
            lo=np.maximum(gt[valid].min(0)-90,0);hi=np.minimum(gt[valid].max(0)+90,pixel)
            ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1]);ax.axis('off');ax.set_title(f'{arm.upper()} / {g}\nNME={s["metrics"][KEY]:.3f}',fontsize=9)
    fig.suptitle('Seed29 fixed frame55 | GT green / prediction magenta')
    fig.savefig(out/'hard_frame55.png',dpi=150);plt.close(fig)


def dual_visual(folders,records,out):
    scene=json.loads((ROOT/'output/cylinder_retention_geometry_v1/report.json').read_text())['dual_visual_scene']
    idx=next(i for i,r in enumerate(records) if r['group']=='old_dual_pair' and r['scene']==scene)
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for row,(arm,folder) in enumerate(folders.items()):
        s=torch.load(folder/f'prediction_{idx:03d}.pt',weights_only=True,map_location='cpu');r=s['record'];pixel=np.array(r['image_hw'][::-1])
        for col,role in enumerate(('side','top')):
            v=r['physical_roles'].index(role);ax=axes[row,col];ax.imshow(Image.open(ROOT/scene/'blended_images'/f'{r["views"][v]:08d}.jpg').resize(tuple(pixel)))
            labels=[]
            for item in matched_views(s):
                if item['view']!=v:continue
                q,m=item['query'],item['instance'];t=s['targets'];p=s['predictions'];valid=item['valid'].numpy().copy();valid[7:14]=False
                gt=t['keypoint_xy'][0,v,m].numpy()*pixel;xy=p['instance_keypoint_xy'][0,v,q].numpy()*pixel
                ax.scatter(gt[valid,0],gt[valid,1],c='lime',s=10);ax.scatter(xy[valid,0],xy[valid,1],c='magenta',s=16,marker='x')
                for b,color in [(t['boxes_cxcywh'][0,v,m],'lime'),(p['instance_boxes_cxcywh'][0,v,q],'magenta')]:
                    b=b.numpy();ax.add_patch(Rectangle((b[:2]-b[2:]/2)*pixel,*(b[2:]*pixel),fill=False,edgecolor=color,lw=.8))
                labels.append(f'm{m}: IoU {item["iou"]:.2f}, score {item["score"]:.2f}')
            ax.axis('off');ax.set_title(f'{arm.upper()} / {role}\n'+', '.join(labels),fontsize=9)
    fig.suptitle('Fixed dual frame246 | body16 shown, all23 retained in gates | GT green / prediction magenta')
    fig.savefig(out/'dual_frame246.png',dpi=145);plt.close(fig)


def localization(folders,records,out):
    points=[];sources={}
    for arm,folder in folders.items():
        for idx,record in enumerate(records):
            path=folder/f'diagnostics_{idx:03d}.pt';sources[str(path.relative_to(ROOT))]=fingerprint(path)
            s=torch.load(path,weights_only=True,map_location='cpu');r=s['record'];p=s['predictions'];t=s['targets']
            for v,(qs,ms) in enumerate(s['oracle_assignment']):
                for q,m in zip(qs,ms):
                    for k in list(range(7))+list(range(14,23)):
                        if not t['keypoint_valid'][0,v,m,k]:continue
                        vis='unknown' if not t['keypoint_visibility_valid'][0,v,m,k] else 'visible' if t['keypoint_visible'][0,v,m,k] else 'occluded'
                        part='head' if k<4 else 'limbs' if 14<=k<22 else 'trunk'
                        stats=refinement_statistics(p['instance_keypoint_coarse_xy'][0,v,q,k].numpy(),p['instance_keypoint_xy'][0,v,q,k].numpy(),
                            t['keypoint_xy'][0,v,m,k].numpy(),p['instance_keypoint_local_heatmap_logits'][0,v,q,k].numpy(),r['image_hw'],t['boxes_cxcywh'][0,v,m,2:].numpy(),scales=(1,))
                        points.append(dict(arm=arm,group=record['group'],role=r['physical_roles'][v],part=part,visibility=vis,
                            scene=r['scene'],instance=m,keypoint=k,temperature_scale=float(s['temperature_scale'][v,q,k]),**stats))
    grouped=defaultdict(list)
    for p in points:
        for part in ('body',p['part']):
            for vis in ('all',p['visibility']):grouped[p['arm'],p['group'],p['role'],part,vis].append(p)
    summary=[dict(arm=k[0],group=k[1],role=k[2],part=k[3],visibility=k[4],temperature_scale=float(np.mean([r['temperature_scale'] for r in rows])),**summarize_default_temperature(rows)) for k,rows in sorted(grouped.items())]
    write_json(out/'localization_points.json',points);write_json(out/'localization_summary.json',summary)
    write_json(out/'localization_inputs.json',sources)
    return [r for r in summary if r['group'] in GROUPS and (r['role'],r['part'],r['visibility'])==('bottom','body','all')]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--first-geometry-only',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2)
    if args.first_geometry_only:
        results=json.loads((first.OUTPUT/'results.json').read_text())
        summary=geometry(dict(off=first.PARENT/'source_mix',on=first.OUTPUT/first.ARM),results['baseline']['records'],first.OUTPUT)
        print(json.dumps(summary,indent=2));return
    assert json.loads((OUTPUT/'delivery_verification.json').read_text())['passed']
    result=json.loads((OUTPUT/'results.json').read_text());old=json.loads((ROOT/'output/cylinder_view_balance_v1/results.json').read_text())
    gates={}
    for arm in ARMS:
        gates[arm]=gate_results(dict(config=result['config'],baseline=result['baseline'],arms=dict(side_control=old['arms']['side_control'],balanced_views=result['arms'][arm])))['balanced_views']
    off,on=[result['arms'][a] for a in ARMS];comparison=[]
    for g in GROUPS:
        a=off['summary'][g][KEY];b=on['summary'][g][KEY];pairs=[]
        for x,y in zip(off['records'],on['records']):
            assert all(x[k]==y[k] for k in ('group','scene','views'))
            if x['group']==g:pairs.append(dict(scene=x['scene'],off=x['metrics'][KEY],on=y['metrics'][KEY]))
        comparison.append(dict(group=g,off=a,on=b,relative_change=b/a-1,improved_scenes=sum(p['on']<p['off'] for p in pairs),pairs=pairs))
    ratio=float(np.mean([r['on'] for r in comparison[1:]])/np.mean([r['off'] for r in comparison[1:]]))
    seed17=json.loads((first.OUTPUT/'acceptance_800.json').read_text())
    folders=dict(off=OUTPUT/ARMS[0],on=OUTPUT/ARMS[1]);geo=geometry(folders,off['records'],OUTPUT)
    parts=[]
    for arm,folder in folders.items():
        for idx,r in enumerate(off['records']):
            if r['group']!='old_dual_pair':continue
            s=torch.load(folder/f'prediction_{idx:03d}.pt',weights_only=True,map_location='cpu')
            for item in matched_views(s):
                for part,inds in [('body',list(range(7))+list(range(14,23))),('tail',list(range(7,14)))]:
                    valid=item['valid'][inds];err=item['errors'][inds][valid]
                    if len(err):parts.append(dict(arm=arm,role=item['role'],part=part,scene=r['scene'],instance=item['instance'],nme=float(err.mean()),found=item['found'],iou=item['iou']))
    write_json(OUTPUT/'dual_parts.json',parts)
    paired_repeat_passed=bool(gates[ARMS[1]]['passed'] and ratio<=.95)
    report=dict(gates=gates,comparison=comparison,multi_bottom_mean_ratio=ratio,
        seed17_multi_bottom_mean_ratio=seed17['multi_bottom_mean_ratio'],paired_repeat_passed=paired_repeat_passed,
        geometry=geo,production_promoted=False,results_sha256=fingerprint(OUTPUT/'results.json'),
        limitations=['Two adaptation-order seeds on the same reused synthetic development set; not independent test data or statistical significance.',
            'No local real Cylinder data; no claim of real-world generalization.', '17 unchanged gates are small-scale development gates, not a production certification.'])
    report['localization']=localization(folders,off['records'],OUTPUT)
    write_json(OUTPUT/'acceptance.json',report);fixed_visual(folders,off['records'],OUTPUT);dual_visual(folders,off['records'],OUTPUT)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for j,(s,label,color) in enumerate([(off['summary'],'OFF','#698fa9'),(on['summary'],'ON','#c38a4c')]):
        axes[0].bar(np.arange(4)+(j-.5)*.35,[s[g][KEY] for g in GROUPS],.35,label=label,color=color)
    axes[0].set(title='Seed29 bottom body NME',xticks=np.arange(4),xticklabels=GROUPS);axes[0].tick_params(axis='x',labelrotation=25);axes[0].legend()
    for arm,color in zip(ARMS,('#698fa9','#c38a4c')):
        h=json.loads((OUTPUT/arm/'history.json').read_text());axes[1].plot(range(50,801),np.convolve([r['pose_loss'] for r in h],np.ones(50)/50,mode='valid'),label=arm,color=color)
    axes[1].set(title='50-step mean pose loss',xlabel='step');axes[1].legend()
    fig.suptitle('Fixed 800-step paired repeat | identical input multiset | 768')
    fig.savefig(OUTPUT/'comparison.png',dpi=145);plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k!='comparison'},indent=2))
    print(json.dumps([{k:v for k,v in r.items() if k!='pairs'} for r in comparison],indent=2))


if __name__=='__main__':main()
