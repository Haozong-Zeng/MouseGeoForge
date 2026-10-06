"""Read-only camera/target-coordinate audit before prediction-only fusion.

No labels are corrected or generated. The cached single-instance cohort is
experimental selection, not a claim that confident queries establish identity.
"""
import argparse
import collections
import datetime as dt
import itertools
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tools'), str(ROOT/'vggt-omega')]
import numpy as np
import torch
from PIL import Image
from mousegeoforge.multiview_camera_geometry import (
    projection_matrix, project_points, fundamental_from_projections,
    symmetric_epipolar_distance,
)
from run_depth_pose_ablation import fingerprint, write_json
from run_surface_curve_pilot_20261003 import load_item
from training.finetune_blendedmvs import load_scene, read_cam
from training.instance_pose_losses import instance_targets_from_batch

CYCLE = ROOT/'output/singleview_fusion_cycle_20261005'
SOURCE = CYCLE/'singleview_fusion_inputs_seed17_v1'
OUT = CYCLE/'camera_geometry_audit_v2'
RELIABLE = [0, 14, 15, 16, 17]
BACK = [5, 6]


def summary(values):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    return dict(n=len(a), mean=float(a.mean()) if len(a) else None,
                median=float(np.median(a)) if len(a) else None,
                p90=float(np.quantile(a, .9)) if len(a) else None,
                max=float(a.max()) if len(a) else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    plan = json.loads((CYCLE/'plan.json').read_text())
    deadline = dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z', '+00:00'))
    if not args.probe:
        assert not OUT.exists(), 'Keep previous audit immutable'
        OUT.mkdir()
        state = json.loads((CYCLE/'state.json').read_text())
        state.update(stage='auditing_camera_geometry', active_job=OUT.name, active_pid=os.getpid())
        write_json(CYCLE/'state.json', state)
    groups = json.loads((SOURCE/'scene_groups.json').read_text())
    manifest = json.loads((SOURCE/'manifest.json').read_text())
    metrics = collections.defaultdict(list)
    details = []
    artifacts = []
    max_xy = max_k = max_e = 0.
    checked = 0
    for part in manifest['parts']:
        role = part['role']
        assert fingerprint(ROOT/part['path']) == part['sha256']
        data = torch.load(ROOT/part['path'], map_location='cpu', weights_only=False)
        selected = []
        for g in next(g['scenes'] for g in groups if g['role'] == role):
            ids = g['row_indices']
            if len(ids) >= 2 and all(data['metadata'][i]['expected_instances'] == 1 for i in ids):
                selected.append(g)
        if args.probe:
            selected = [next(g for g in selected if data['metadata'][g['row_indices'][0]]['record']['kind'] == kind)
                        for kind in ['real', 'synthetic']] if role == 'train' else []
        records = []
        for g in selected:
            assert dt.datetime.now(dt.timezone.utc) < deadline
            ids = g['row_indices']
            record = dict(data['metadata'][ids[0]]['record'], views=g['physical_views'])
            assert record.get('session') not in ['GR2', 'UR2', 'UR3']
            assert record['role'] == role and len(set(record['views'])) == len(ids)
            kind = 'real' if record['kind'] == 'real' else 'synthetic'
            item = load_item(record)
            b = item['batch']
            t = instance_targets_from_batch(b)
            if args.probe:
                print(json.dumps(dict(role=role, kind=kind, record=record,
                    shapes={k:list(v.shape) for k,v in b.items() if isinstance(v,torch.Tensor)},
                    packed_shapes={k:list(data[k][ids].shape) for k in ['gt_xy','point_valid','pixels']},
                    scale=float(b['scale'][0])), default=str), flush=True)
                continue
            h,w = record['image_hw']
            pixel = np.array([w,h], float)
            np.testing.assert_array_equal(data['pixels'][ids].numpy(), np.tile(pixel,(len(ids),1)))
            np.testing.assert_array_equal(t['keypoint_valid'][0], data['point_valid'][ids])
            valid = t['keypoint_valid'][0].numpy()
            delta = np.abs((t['keypoint_xy'][0] - data['gt_xy'][ids]).numpy())[valid]
            max_xy = max(max_xy, float(delta.max()) if len(delta) else 0.)
            assert max_xy < 2e-6, (record, max_xy)
            k = b['intrinsics'][0].numpy().astype(float)
            e = b['extrinsics'][0].numpy().astype(float)
            p = np.stack([projection_matrix(ki,ei) for ki,ei in zip(k,e)])
            scene = load_scene(ROOT/g['scene'])
            source_sizes = []
            # load_item explicitly uses canonicalize_camera=False. Only a
            # shared scalar world normalization is applied to translations.
            for vi, physical in enumerate(record['views']):
                frame = scene.frames[physical]
                with Image.open(frame.image_path) as im: sw,sh = im.size
                source_sizes.append([sw,sh])
                raw_e, raw_k = read_cam(frame.cam_path)
                expected_k = raw_k.copy(); expected_k[0] *= w/sw; expected_k[1] *= h/sh
                expected_e = raw_e[:3].copy(); expected_e[:3,3] /= float(b['scale'][0])
                max_k = max(max_k, float(np.abs(k[vi]-expected_k).max()))
                max_e = max(max_e, float(np.abs(e[vi]-expected_e).max()))
                np.testing.assert_allclose(k[vi], expected_k, atol=2e-4, rtol=2e-6)
                np.testing.assert_allclose(e[vi], expected_e, atol=2e-5, rtol=2e-6)
            gt = data['gt_xy'][ids].numpy()*pixel
            score = data['objectness_logits'][ids].sigmoid().numpy()
            unique = (score >= .5).sum(-1) == 1
            query = score.argmax(-1)  # Used only where unique is true.
            pred = data['xy'][ids].numpy()[np.arange(len(ids)),query]*pixel
            for ai,bi in itertools.combinations(range(len(ids)),2):
                f = fundamental_from_projections(p[ai],p[bi])
                # Camera-algebra round trip does not validate real calibration.
                xyz = np.array([[0.,0.,0.],[.1,.3,.2],[-.2,.1,.5]])
                ua,_ = project_points(p[ai],xyz); ub,_ = project_points(p[bi],xyz)
                algebra = symmetric_epipolar_distance(f,ua,ub)
                assert np.nanmax(algebra) < 1e-5
                metrics['camera_algebra_px'].extend(algebra.tolist())
                distances = symmetric_epipolar_distance(f,gt[ai,0],gt[bi,0])
                pd = symmetric_epipolar_distance(f,pred[ai],pred[bi])
                common = valid[ai,0]&valid[bi,0]
                for name,kps in [('reliable',RELIABLE),('feet',[14,15,16,17]),('back',BACK)]:
                    take = np.zeros(23,bool); take[kps] = True; take &= common
                    prefix = f'{role}/{kind}/{name}'
                    metrics[prefix+'/accepted_pair_px'].extend(distances[take].tolist())
                    if unique[ai] and unique[bi]:
                        # Prediction-only availability, with no GT visibility
                        # filter; these counts deliberately differ from GT.
                        metrics[prefix+'/prediction_pair_px'].extend(pd[kps].tolist())
                    pair = f"cam{chr(65+record['views'][ai])}-cam{chr(65+record['views'][bi])}"
                    metrics[prefix+'/'+pair].extend(distances[take].tolist())
                    for kp in np.flatnonzero(take):
                        details.append(dict(role=role,kind=kind,scene=g['scene'],pair=pair,point=int(kp+1),
                            accepted_epipolar_px=float(distances[kp]),
                            prediction_epipolar_px=float(pd[kp]) if unique[ai] and unique[bi] else None))
            if kind == 'synthetic':
                for name,field,validfield in [('surface','surface_world_xyz_instances','instance_surface_valid'),
                                               ('joint','instance_keypoint_world_xyz','instance_world_valid')]:
                    xyz = b[field][0,0].numpy()
                    world_valid = b[validfield][0,0].numpy()
                    for vi in range(len(ids)):
                        uv,z = project_points(p[vi],xyz)
                        take = valid[vi,0]&world_valid&(z>0)
                        err = np.linalg.norm(uv-gt[vi,0],axis=-1)
                        metrics[f'{role}/synthetic/{name}_projection_px'].extend(err[take].tolist())
            records.append(dict(scene=g['scene'],row_indices=ids,physical_views=record['views'],kind=kind,
                image_hw=[h,w],source_sizes=source_sizes,scale=float(b['scale'][0]),intrinsics=k,
                extrinsics=e,projections=p,source_hashes=item['source_hashes']))
            checked += 1
            write_json(OUT/'status.json',dict(stage='auditing',scenes_done=checked,role=role,pid=os.getpid()))
            if checked%20 == 0: print(f'audited {checked} scenes',flush=True)
        if not args.probe:
            path = OUT/f'{role}_cameras.pt'; torch.save(records,path)
            artifacts.append(dict(role=role,path=path.relative_to(ROOT).as_posix(),sha256=fingerprint(path),scenes=len(records)))
    if args.probe:return
    report = dict(scenes=checked,max_target_xy_normalized_delta=max_xy,max_K_delta=max_k,max_E_delta=max_e,
        metrics={k:summary(v) for k,v in metrics.items()},camera_artifacts=artifacts,
        source_manifest_sha256=fingerprint(SOURCE/'manifest.json'),script_sha256=fingerprint(Path(__file__)),
        caveats=['Read-only accepted-label diagnostics, not new ground truth.',
                 'Repeated development validation is not an external test.',
                 'Prediction pairs use unique confident query per view, not GT count/visibility/identity.',
                 'Low epipolar residual is necessary but not sufficient for accurate anatomical localization.',
                 'Pinhole calibration may not fully capture glass/refraction; no calibration is fitted here.'])
    write_json(OUT/'results.json',report); write_json(OUT/'pair_details.json',details)
    write_json(OUT/'status.json',dict(stage='complete',scenes_done=checked,pid=os.getpid()))
    state.update(stage='camera_geometry_audit_complete',active_job=None,active_pid=None,
        next='Read camera_geometry_audit_v2/results.json before prediction-only geometry experiment; do not modify labels or assume small epipolar residual implies correct anatomy.')
    state['completed'].append('camera_geometry_audit'); write_json(CYCLE/'state.json',state)
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    try:main()
    except Exception:
        if OUT.exists():write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise
