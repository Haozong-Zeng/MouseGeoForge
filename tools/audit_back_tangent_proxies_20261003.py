"""TRAIN-only synthetic audit of uncertain tangent proxies; no loss changes."""
import collections,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np
import torch
from PIL import Image
from run_depth_pose_ablation import write_json,fingerprint
CYCLE=ROOT/'output/surface_curve_cycle_20261003'
OUT=CYCLE/'back_tangent_proxy_audit_v1'


def main():
    assert not OUT.exists();OUT.mkdir()
    saved=torch.load(CYCLE/'expanded_curve_features_v1/frozen_inputs.pt',map_location='cpu',weights_only=False)
    scenes=sorted({m['scene'] for m in saved['metadata'] if m['role']=='train'})
    sources={}
    for name in ['diverse_curve_gt_v1','diverse_curve_gt_v2']:
        for r in json.loads((CYCLE/name/'manifest.json').read_text())['records']:
            if r['scene'] in scenes and r['accepted_for_training']:sources[r['scene']]=r['sidecar']
    for scene in scenes:
        if (ROOT/scene/'surface_curve.npz').exists():sources[scene]=str(Path(scene)/'surface_curve.npz')
    rows=[];skipped=collections.Counter()
    for scene in scenes:
        with np.load(ROOT/scene/'keypoints.npz') as z:kp=dict(z)
        with np.load(ROOT/sources[scene]) as z:curve=dict(z)
        xyz=kp['joint_world_xyz'];uv=kp['uv']
        if xyz.ndim==2:xyz=xyz[None]
        if uv.ndim==3:uv=uv[:,None]
        world=curve['world_xyz'];curveuv=curve['uv']
        assert xyz.shape[0]==world.shape[0]
        view_ids=kp['view_ids'].tolist()
        for v,vid in enumerate(view_ids):
            images=list((ROOT/scene/'blended_images').glob(f'{vid:08d}.*'));assert len(images)==1
            with Image.open(images[0]) as im:scale=768/max(im.size)
            cv=curve['view_ids'].tolist().index(vid)
            for m in range(len(xyz)):
                length=np.linalg.norm(np.diff(world[m],axis=0),axis=1).sum()
                for k,lo,hi in [(5,4,6),(6,5,7)]:
                    d=np.linalg.norm(world[m]-xyz[m,k],axis=1);idx=int(d.argmin())
                    if d[idx]>.025*length or idx<2 or idx>62:
                        skipped['anchor_not_near_interior_material_curve']+=1;continue
                    tangent=(curveuv[cv,m,idx+2]-curveuv[cv,m,idx-2])*scale
                    if np.linalg.norm(tangent)<1:
                        skipped['foreshortened_true_tangent']+=1;continue
                    for method,a,b in [('legacy_neighbors',lo,hi),('back06to07_chord',5,6)]:
                        chord=(uv[v,m,b]-uv[v,m,a])*scale
                        if not np.isfinite(chord).all() or np.linalg.norm(chord)<10:
                            skipped[method+'_short_or_missing']+=1;continue
                        cosine=abs(np.dot(chord,tangent)/(np.linalg.norm(chord)*np.linalg.norm(tangent)))
                        angle=float(np.degrees(np.arccos(np.clip(cosine,0,1))))
                        rows.append(dict(scene=scene,view=int(vid),instance=m,point_id=k+1,method=method,
                                         axis_error_degrees=angle,chord_length_px=float(np.linalg.norm(chord)),
                                         nearest_material_index=idx,anchor_distance_fraction=float(d[idx]/length)))
    summary=[]
    for method in ['legacy_neighbors','back06to07_chord']:
        for point in [6,7]:
            r=[x for x in rows if x['method']==method and x['point_id']==point];a=np.array([x['axis_error_degrees'] for x in r])
            summary.append(dict(method=method,point_id=point,n=len(a),mean=float(a.mean()),median=float(np.median(a)),
                                p90=float(np.quantile(a,.9)),fraction_over30deg=float((a>30).mean())))
    write_json(OUT/'report.json',dict(summary=summary,rows=rows,skipped=dict(skipped),train_scenes=len(scenes),
        source_feature_sha256=fingerprint(CYCLE/'expanded_curve_features_v1/frozen_inputs.pt'),
        protocol='Only89TRAINscenes; known synthetic material curve gives local +/-2sample chord near06/07. Unsigned axis angle. Anchor-distance<2.5%curve length, local tangent>=1px, proxy chord>=10px at768longside.',
        limits=['Amodeled geometry, not a visibility or annotation confidence measure',
                'Nearest sampled curve point is only approximate surface tangent',
                'Both chord proxies can disagree with local curved surface; manual06/07are themselves uncertain'],
        no_validation_or_real_holdout_used=True,no_training_loss_changed=True))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
