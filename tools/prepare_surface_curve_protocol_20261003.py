"""Freeze source-separated experiment inventory before looking at model outcomes."""
import json,hashlib,collections,datetime as dt
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/surface_curve_cycle_20261003/training_v1'
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def evenly(rows,n):
    idx=np.linspace(0,len(rows)-1,min(n,len(rows))).round().astype(int);return [rows[i] for i in idx]
def main():
    assert not OUT.exists();OUT.mkdir()
    assert load(ROOT/'output/surface_curve_cycle_20261003/stress_v2/report.json')['passed']
    labels=ROOT/'dataset/PyMouseStudio/annotation_versions/20261003_effective_v1/manifest.json';m=load(labels)
    real=[]
    for session in sorted({g['session'] for g in m['frame_groups']}):
        groups=sorted([g for g in m['frame_groups'] if g['session']==session and g['eligible_for_development']],key=lambda g:g['frame'])
        for g in evenly(groups,4):
            views=[r for r in m['provenance'] if (r['session'],r['frame'])==(g['session'],g['frame'])]
            real.append(dict(session=session,frame=g['frame'],role=g['role'],normalized=g['effective_normalized_file'],
              normalized_sha256=sha(ROOT/g['effective_normalized_file']),views=[{k:r[k] for k in ['camera','annotation','source_image','mask','mask_valid','annotation_sha256','mask_sha256']} for r in views]))
    assert not any(r['session'] in ['GR2','UR2','UR3'] for r in real)
    parent=load(ROOT/'output/cylinder_source_mix_v1/protocol.json')
    bydataset=collections.defaultdict(dict)
    for choices in parent['arms']['source_mix']:
        for key in choices:
            r=parent['records'][key];assert r['split']=='train';bydataset[r['dataset']][r['scene']]=r
    old=[]
    for dataset,records in sorted(bydataset.items()):
        for r in evenly(sorted(records.values(),key=lambda x:x['scene']),4):
            folder=ROOT/r['scene'];paths=[folder/'keypoints.npz',folder/'scene_meta.json']
            old.append(dict(scene=r['scene'],dataset=dataset,frame=r['frame'],expected_instances=r['expected_instances'],
                            source_group=r['source_group'],normalization_scale=r['normalization_scale'],role='train',
                            annotation_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in paths if p.exists()}))
    development=[]
    seen=set()
    for e in parent['evaluation']:
        r=parent['records'][e['key']]
        if r['scene'] in seen:continue
        seen.add(r['scene']);development.append(dict(scene=r['scene'],views=r['views'],role='validation',group=e['group'],source_group=r['source_group']))
    assert not {r['scene'] for r in old}&{r['scene'] for r in development}
    new=ROOT/'dataset/BlenderPyMouse/PyMouseStudio-RealCapture-SurfaceCurve-Multiview'
    synthetic_new=[dict(scene=p.relative_to(ROOT).as_posix(),role='train',keypoints_sha256=sha(p/'keypoints.npz'),curve_sha256=sha(p/'surface_curve.npz')) for p in sorted(new.glob('frame_*'))]
    cfg=dict(schema=1,status='inventory_frozen_runner_not_started',created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
      deadline_utc='2026-10-04T08:35:16Z',seed=17,long_side=768,real_input_wh=[704,768],real_affine=[[691/1080,0,6],[0,768/1200,0],[0,0,1]],
      initial_steps_per_arm=600,evaluate_every=100,checkpoint_every=100,extend_policy='Only equal increments for all arms after convergence review; do not train beyond deadline minus30min',
      arms=['strict_points','weak_points','weak_points_curve6'],point_loss_scale=768.,weak_ids_1based=[6,7],along_tolerance=4.,unknown_tangent_weight=.25,
      tangent_policy='Compute chord estimates from all finite original observed05-08 before restricting strong loss to seven body IDs. Preserve separate observed/tangent/strong masks; otherwise both real tangents would always be unavailable. Never turn these auxiliary observations into strong endpoint GT.',
      curve_representation='clamped_cubic_bspline_6_controls_33_samples',curve_ground_truth='Only actual synthetic material curve; sparse manual points are NOT full curve GT',
      real_supervised_ids_1based=[1,6,7,15,16,17,18],synthetic_supervised_ids_1based=list(range(1,24)),
      strict_loss_replacement_required=True,temporal='off_fixed',depth_pose_coupling='off_fixed',encoder_frozen=True,
      learning_rate=3e-6,curve_head_learning_rate=3e-4,curve_aux_weight_provisional=1.,gradient_clip=1.,
      sampling_mixture={'real_development_train':.5,'old_diverse_synthetic_train':1/3,'new_surface_curve_synthetic_train':1/6},
      primary_checkpoint='output/pose_diverse_core23_768_v1/seed17_off/final_heads.pt',
      checkpoint_reason='Common known-compatible768 baseline used by previous source-mixture controls, not claiming it is the latest best model',
      shared_matching='Boxes/masks/reliable points; exclude ambiguous real06/07 from assignment cost in all three arms; use one match per loss call',
      heldout_policy='GR2/UR2/UR3 are not loaded or tuned; validation sessions AZR2L2/FR2 retain old split',
      validation_limitation='Replica look/pose development previously used frames from FR2/AZR2L2. These remain development diagnostics, NOT a clean never-seen external generalization benchmark.',
      new_dataset_limitation='Three static poses only, used for auxiliary-curve feasibility; no temporal or broad-generalization claims',
      real=real,synthetic_train=old,new_synthetic_train=synthetic_new,synthetic_development=development,
      label_manifest_sha256=sha(labels),checkpoint_sha256=sha(ROOT/'output/pose_diverse_core23_768_v1/seed17_off/final_heads.pt'),
      acceptance=['Finite gradients/checkpoints and loss replacement verified','Reliable real nose/paw error and P90 do not regress materially','Separate original back point error from along/perpendicular curve diagnostics','Synthetic curve/order/endpoint errors improve, no collapse','Mask/detection/depth retention reported','No production promotion from this pilot'])
    save(OUT/'protocol.json',cfg)
    print(json.dumps(dict(real_groups=len(real),real_images=sum(len(r['views']) for r in real),real_roles=dict(collections.Counter(r['role'] for r in real)),synthetic_train=len(old),synthetic_development=len(development),new_synthetic=len(synthetic_new)),indent=2))
if __name__=='__main__':main()
