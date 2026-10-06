"""Post-hoc known-calibration fusion of already-saved independent predictions.

No new training/forward, no learned fusion, no GT XYZ used by triangulation.
Associations still use GT boxes. This is diagnostic, not an end-to-end result.
"""
import json
import numpy as np
from train_cylinder_view_balance import ROOT, OUTPUT, torch, write_json, fingerprint
from vggt_omega.utils.robust_pose_geometry import triangulate_robust


def samples(saved, view):
    p,t=saved['predictions'],saved['targets']
    qs,ms=saved['oracle_assignment'][view]; q=qs[ms.index(0)]
    h,w=saved['record']['image_hw']
    return dict(uv=p['instance_keypoint_xy'][0,view,q]*torch.tensor([w,h]),
        score=p['instance_objectness_logits'][0,view,q].sigmoid(),
        valid=t['keypoint_valid'][0,view,0],
        projection=t['intrinsics'][0,view] @ t['extrinsics'][0,view,:3],
        world=t['keypoint_world_xyz'][0,0],scale=saved['record']['normalization_scale'])


def main():
    result=json.loads((OUTPUT/'results.json').read_text())
    records=result['baseline']['records']
    scenes=[r['scene'] for r in records if r['group']=='side_bottom']
    rows=[]
    for arm in ('initial','side_control','balanced_views'):
        for scene in scenes:
            items={}
            for group in ('side_only','bottom_only','side_bottom'):
                idx=next(i for i,r in enumerate(records) if r['scene']==scene and r['group']==group)
                items[group]=torch.load(OUTPUT/arm/f'prediction_{idx:03d}.pt',weights_only=True,map_location='cpu')
            joint=[samples(items['side_bottom'],i) for i in (0,1)]
            independent=[samples(items[g],0) for g in ('side_only','bottom_only')]
            for a,b in zip(joint,independent):
                torch.testing.assert_close(a['projection'],b['projection'],atol=0,rtol=0)
                torch.testing.assert_close(a['world'],b['world'],atol=0,rtol=0)
                assert a['scale']==b['scale']
            for k in list(range(7))+list(range(14,23)):
                row=dict(arm=arm,scene=scene,keypoint=k)
                for name,views in (('joint',joint),('independent',independent)):
                    uv=torch.stack([v['uv'][k] for v in views])
                    projection=torch.stack([v['projection'] for v in views])
                    valid=torch.stack([v['valid'][k] & (v['score']>=.5) for v in views])
                    out=triangulate_robust(uv,projection,valid,huber_delta_px=3.)
                    row[name]=dict(supported=out.supported,reason=out.reason,
                        error_native=float((out.xyz-views[0]['world'][k]).norm())*views[0]['scale'] if out.supported else None)
                rows.append(row)
    summary={}
    for arm in ('initial','side_control','balanced_views'):
        selected=[r for r in rows if r['arm']==arm]
        common=[r for r in selected if r['joint']['supported'] and r['independent']['supported']]
        report={}
        for name in ('joint','independent'):
            errors=[r[name]['error_native'] for r in selected if r[name]['supported']]
            report[name]=dict(coverage=len(errors)/len(selected),supported=len(errors),points=len(selected),
                mean_error_native=float(np.mean(errors)) if errors else None,
                common_points=len(common),common_mean_error_native=float(np.mean([r[name]['error_native'] for r in common])) if common else None)
        summary[arm]=report
    write_json(OUTPUT/'independent_view_fusion.json',dict(summary=summary,rows=rows,
        results_sha256=fingerprint(OUTPUT/'results.json'),posthoc=True,weights_updated=False,
        caveat='GT box-assisted association; same known K/E and thresholds; supports differ; no real validation or production promotion'))
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
