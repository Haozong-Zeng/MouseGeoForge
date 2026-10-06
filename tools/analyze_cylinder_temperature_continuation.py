"""Equal-budget staged temperature results, localization and common-support 3D."""
import argparse
from collections import defaultdict,Counter
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_depth_pose_ablation import ROOT,torch,write_json,fingerprint
from analyze_cylinder_view_balance import gate_results
from mousegeoforge.retention_geometry_audit import geometry_points
from mousegeoforge.localization_audit import refinement_statistics


def main(step):
    folder=ROOT/f'output/cylinder_temperature_continuation_v1/step_{step}'
    assert json.loads((folder/'delivery_verification.json').read_text())['passed']
    result=json.loads((folder/'results.json').read_text())
    old=json.loads((ROOT/'output/cylinder_view_balance_v1/results.json').read_text())
    groups=['bottom_only','side_bottom','sides_bottom','sides_bottom_top'];key='bottom_body_keypoint_nme'
    gates={a:gate_results(dict(config=result['config'],baseline=result['baseline'],arms=dict(side_control=old['arms']['side_control'],balanced_views=r)))['balanced_views'] for a,r in result['arms'].items()}
    comparison=[]
    for group in groups:
        off=result['arms']['off']['summary'][group][key];on=result['arms']['on']['summary'][group][key]
        comparison.append(dict(group=group,off=off,on=on,relative_change=on/off-1))
    ratio=float(np.mean([r['on'] for r in comparison[1:]])/np.mean([r['off'] for r in comparison[1:]]))
    points=[];geo=[];sources={}
    for arm in ('off','on'):
        for idx,row in enumerate(result['arms'][arm]['records']):
            path=folder/arm/'final'/f'prediction_{idx:03d}.pt';sources[str(path.relative_to(ROOT))]=fingerprint(path)
            saved=torch.load(path,weights_only=True,map_location='cpu');r=saved['record'];p,t=saved['predictions'],saved['targets']
            if row['group'] in groups[1:]:geo.extend(dict(arm=arm,scene=row['scene'],group=row['group'],**g) for g in geometry_points(saved))
            for v,(qs,ms) in enumerate(saved['oracle_assignment']):
                for q,m in zip(qs,ms):
                    for k in torch.where(t['keypoint_valid'][0,v,m])[0].tolist():
                        stats=refinement_statistics(p['instance_keypoint_coarse_xy'][0,v,q,k].numpy(),p['instance_keypoint_xy'][0,v,q,k].numpy(),
                            t['keypoint_xy'][0,v,m,k].numpy(),p['instance_keypoint_local_heatmap_logits'][0,v,q,k].numpy(),r['image_hw'],t['boxes_cxcywh'][0,v,m,2:].numpy())
                        visibility='unknown'
                        if t['keypoint_visibility_valid'][0,v,m,k]:visibility='visible' if t['keypoint_visible'][0,v,m,k] else 'occluded'
                        part='tail' if 7<=k<14 else 'head' if k<4 else 'limbs' if 14<=k<22 else 'trunk'
                        points.append(dict(arm=arm,record=idx,scene=row['scene'],group=row['group'],role=r['physical_roles'][v],instance=m,keypoint=k,
                            part=part,visibility=visibility,scale=float(saved['temperature_scale'][v,q,k]),
                            **{key:stats[key] for key in ('coarse_nme','refined_nme','inside_window','entropy_normalized','max_probability')}))
    strata=defaultdict(list)
    for p in points:
        for subset in (p['part'], 'body' if p['part']!='tail' else 'all_tail'):
            strata[p['arm'],p['group'],p['role'],subset].append(p)
        if p['part']!='tail':strata[p['arm'],p['group'],p['role'],'body_'+p['visibility']].append(p)
    localization=[dict(arm=k[0],group=k[1],role=k[2],subset=k[3],points=len(v),
        **{name:float(np.mean([p[name] for p in v])) for name in ('coarse_nme','refined_nme','inside_window','entropy_normalized','scale')}) for k,v in sorted(strata.items())]
    geometry=[]
    for group in groups[1:]:
        a={(r['scene'],r['keypoint']):r for r in geo if r['arm']=='off' and r['group']==group}
        b={(r['scene'],r['keypoint']):r for r in geo if r['arm']=='on' and r['group']==group}
        common=[k for k in a if a[k]['supported'] and b[k]['supported']]
        geometry.append(dict(group=group,off_supported=sum(p['supported'] for p in a.values()),on_supported=sum(p['supported'] for p in b.values()),total=len(a),
            common_points=len(common),off_common_error=float(np.mean([a[k]['error_native'] for k in common])) if common else None,
            on_common_error=float(np.mean([b[k]['error_native'] for k in common])) if common else None,
            transitions=dict(Counter(f'{int(a[k]["supported"])}->{int(b[k]["supported"])}' for k in a))))
    report=dict(step=step,original_gates=gates,comparison=comparison,multi_bottom_mean_ratio=ratio,substantive_improvement=ratio<=.95,
        geometry=geometry,localization=localization,results_sha256=fingerprint(folder/'results.json'),prediction_sha256=sources,
        limitations='Reused synthetic development; GT-box matching and known calibration, not real Cylinder or end-to-end3D; optimizer reset at800 is symmetric.',production_promoted=False)
    write_json(folder/'acceptance.json',report);write_json(folder/'localization_points.json',points);write_json(folder/'geometry_points.json',geo)
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,(gs,k,title) in zip(axes,[(groups,key,'Bottom body NME'),(['old_dual_pair','old_cylinder_side_pair'],'detection_aware_pck_0.1','Original-pair PCK'),(groups,'bottom_body_depth_absrel','Bottom depth AbsRel')]):
        for j,arm in enumerate(('off','on')):ax.bar(np.arange(len(gs))+(j-.5)*.35,[result['arms'][arm]['summary'][g][k] for g in gs],.35,label=arm)
        ax.set_title(title);ax.set_xticks(range(len(gs)),gs,rotation=25,ha='right')
    axes[0].legend();fig.suptitle(f'Conditional temperature, equal {step} cumulative steps; symmetric optimizer reset at800')
    fig.savefig(folder/'comparison.png',dpi=150);plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k not in ('localization','prediction_sha256')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--step',type=int,required=True);args=parser.parse_args();main(args.step)
