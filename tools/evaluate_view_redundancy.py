"""Frozen 2-input/2-ray, 4-input/2-ray, 4-input/4-ray diagnostic.

Reuses V3 development scenes; no new-test/generalization claim. All methods use
uniform precision, the same pixel kernel and GT box-assisted instance matching.
The original two cameras retain their order and reference-camera position.
"""
import argparse
from functools import partial
import json
from pathlib import Path

from diagnose_occlusion_pose import ROOT, torch, infer, fingerprint, write_json, load_batch
from evaluate_pixel_robust_fusion import summarize, compare
from vggt_omega.utils.robust_pose_geometry import triangulate_robust

OLD = ROOT/'output/occlusion_pose_v3/pixel_fusion'
METHODS = ('A_two_input_two_rays', 'B_four_input_two_rays', 'C_four_input_four_rays', 'D_four_min_three_inliers')


def view_indices(item, cameras):
    actual = list(map(int, item['actual_view_ids'][0].split(',')))
    if len(set(actual)) != len(actual) or len(set(cameras)) != len(cameras):
        raise ValueError('Duplicate camera IDs')
    if not set(cameras).issubset(actual):
        raise ValueError('Requested physical camera absent from cache')
    return [actual.index(v) for v in cameras]


def verify_shared_targets(old, new):
    """Compare native-unit labels/calibration by physical ID, not tensor position."""
    cameras = list(map(int, old['actual_view_ids'][0].split(',')))
    indices = view_indices(new, cameras)
    a, b = old['targets'], new['targets']
    sa, sb = float(old['scale'].flatten()[0]), float(new['scale'].flatten()[0])
    for field in ('keypoint_xy', 'keypoint_valid', 'keypoint_visible', 'instance_ids', 'intrinsics'):
        torch.testing.assert_close(a[field], b[field][:, indices], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(a['keypoint_world_xyz']*sa, b['keypoint_world_xyz']*sb, rtol=1e-5, atol=1e-3)
    torch.testing.assert_close(a['world_valid'], b['world_valid'])
    ea, eb = a['extrinsics'].clone(), b['extrinsics'][:, indices].clone()
    ea[..., :3, 3] *= sa; eb[..., :3, 3] *= sb
    torch.testing.assert_close(ea, eb, rtol=1e-5, atol=1e-3)


def protocol():
    previous = json.loads((OLD/'protocol.json').read_text())
    checkpoint = ROOT/'output/pose_diverse_core23_768_v1/seed17_off/final_heads.pt'
    if fingerprint(checkpoint) != previous['head_sha256']:
        raise ValueError('Checkpoint changed')
    records, hashes = [], []
    for i, r in enumerate(previous['records']):
        cache = OLD/'predictions'/f'{i:03d}.pt'
        item = torch.load(cache, weights_only=True)
        if item['record'] != r or fingerprint(ROOT/r['scene']/'keypoints.npz') != r['annotation_sha256']:
            raise ValueError('V3 source cache/annotation changed')
        cameras = list(map(int, item['actual_view_ids'][0].split(',')))
        if set(cameras) != {0, 2}: raise ValueError('Unexpected V3 cameras')
        order = cameras + [v for v in range(4) if v not in cameras]
        records.append(dict(r, split='development_diagnostic', views=list(range(4)), diagnostic_view_order=order))
        hashes.append(fingerprint(cache))
    return dict(records=records, reference_prediction_sha256=hashes,
        mode='reused_V3_development_scenes_not_new_test', independent_session=False,
        head_sha256=previous['head_sha256'], previous_protocol_sha256=fingerprint(OLD/'protocol.json'),
        code_sha256={p:fingerprint(ROOT/p) for p in ('tools/run_depth_pose_ablation.py',
            'tools/diagnose_occlusion_pose.py', 'vggt-omega/vggt_omega/utils/robust_pose_geometry.py')},
        methods=METHODS, primary='C versus B: same four-view forward, different geometry observations',
        context_control='B versus A: same physical rays, different image context',
        secondary='D keeps only C solutions with at least three final inliers; no refit',
        weights='uniform; no sigma or visibility filtering',
        solver=dict(huber_delta_px=3., reprojection_threshold_px=8., iterations=5),
        gates=dict(max_coverage_drop=.03, max_paired_mean_ratio=1.01, max_paired_p95_ratio=1.05),
        association='GT box-assisted query matching; instance identities verified across views',
        production_enabled=False)


def oracle(item):
    return [{m:q for q,m in zip(qs,ms)} for qs,ms in item['oracle_assignment']]


def point_solve(item, indices, m, k):
    p, t = item['predictions'], item['targets']; matches = oracle(item)
    h,w = item['record']['image_hw']; scale = torch.tensor([w,h])
    uv = torch.stack([p['instance_keypoint_xy'][0,v,matches[v][m],k] for v in indices])*scale
    proj = (t['intrinsics'][0] @ t['extrinsics'][0,:,:3])[indices]
    out = triangulate_robust(uv, proj, t['keypoint_valid'][0,indices,m,k], huber_delta_px=3.)
    native = float(item['scale'].flatten()[0])
    return dict(error_mm=float((out.xyz-t['keypoint_world_xyz'][0,m,k]).norm())*native if out.supported else None,
        reason=out.reason, inliers=int(out.inliers.sum()),
        xyz_mm=(out.xyz*native).tolist() if out.supported else None,
        inlier_cameras=[int(item['actual_view_ids'][0].split(',')[indices[j]]) for j in torch.where(out.inliers)[0].tolist()])


def two_dimensional(old, new):
    rows = []; cameras = list(map(int, old['actual_view_ids'][0].split(',')))
    h,w = old['record']['image_hw']; scale = torch.tensor([w,h])
    for camera in cameras:
        vs = [view_indices(x,[camera])[0] for x in (old,new)]
        for m in range(2):
            a = old['targets']; v = vs[0]
            for k in torch.where(a['keypoint_valid'][0,v,m])[0].tolist():
                row = dict(camera=camera, instance=m, keypoint=k,
                           visible=bool(a['keypoint_visible'][0,v,m,k]))
                for name, item, vi in zip(('two_input','four_input'), (old,new), vs):
                    t = item['targets']; q = oracle(item)[vi][m]
                    error = float(((item['predictions']['instance_keypoint_xy'][0,vi,q,k]-t['keypoint_xy'][0,vi,m,k])*scale).norm())
                    diag = float((t['boxes_cxcywh'][0,vi,m,2:]*scale).norm().clamp_min(1.))
                    row[name] = dict(error_px=error, nme=error/diag)
                rows.append(row)
    return rows


def summarize_2d(rows):
    result = {}
    for group in ('all','visible','occluded'):
        subset = [r for r in rows if group=='all' or r['visible']==(group=='visible')]
        result[group] = {method:dict(points=len(subset),
            mean_px=sum(r[method]['error_px'] for r in subset)/len(subset),
            nme=sum(r[method]['nme'] for r in subset)/len(subset),
            pck_01=sum(r[method]['nme']<=.1 for r in subset)/len(subset)) for method in ('two_input','four_input')}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT/'output/occlusion_pose_v4/view_redundancy')
    args = parser.parse_args(); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    spec = json.loads(json.dumps(protocol())); args.output.mkdir(parents=True, exist_ok=True)
    lock = args.output/'protocol.json'
    if lock.exists() and json.loads(lock.read_text()) != spec: raise ValueError('Protocol changed')
    write_json(lock,spec)
    print('LOCKED 24 reused development scenes; reference order retained', flush=True)
    if args.prepare_only: return
    if not args.report_only:
        infer(ROOT/'output/pose_diverse_core23_768_v1', args.output, spec['records'],
              batch_loader=partial(load_batch,num_views=4))
    rows, rows2d = [], []
    reference = json.loads((OLD/'report.json').read_text())['rows']
    for i,r in enumerate(spec['records']):
        old = torch.load(OLD/'predictions'/f'{i:03d}.pt',weights_only=True)
        new = torch.load(args.output/'predictions'/f'{i:03d}.pt',weights_only=True)
        if new['record'] != r: raise ValueError('Four-view cache mismatch')
        verify_shared_targets(old,new)
        for item in (old,new):
            ids = item['targets']['instance_ids'][0]
            if not torch.all(ids == ids[:1]): raise ValueError('Instance slots reordered across views')
        pair = list(map(int,old['actual_view_ids'][0].split(',')))
        rows2d.extend(dict(x,scene=r['scene']) for x in two_dimensional(old,new))
        for m in range(2):
            for k in range(23):
                if not bool(new['targets']['world_valid'][0,m,k]): raise ValueError('Missing 3D truth')
                row = dict(scene=r['scene'],instance=m,keypoint=k)
                for name,item,cameras in ((METHODS[0],old,pair),(METHODS[1],new,pair),(METHODS[2],new,list(range(4)))):
                    row[name] = point_solve(item,view_indices(item,cameras),m,k)
                c = row[METHODS[2]]
                row[METHODS[3]] = dict(c) if c['inliers']>=3 else dict(c,error_mm=None,xyz_mm=None,reason='fewer_than_three_inliers')
                base = reference[len(rows)]['uniform_pixel']
                if row[METHODS[0]]['error_mm'] != base['error_mm']: raise AssertionError('V3 baseline changed')
                rows.append(row)
        print('Geometry',r['frame'],flush=True)
    gates = spec['gates']
    result = dict(protocol_sha256=fingerprint(lock), rows=rows, rows_2d=rows2d,
        summary={key:summarize(rows,key) for key in METHODS}, context_2d=summarize_2d(rows2d),
        comparisons={name:compare(rows,a,b,gates) for name,a,b in (
            ('context',METHODS[0],METHODS[1]),('primary_geometry',METHODS[1],METHODS[2]),
            ('combined',METHODS[0],METHODS[2]),('three_inlier_filter',METHODS[2],METHODS[3]))},
        four_view_inlier_histogram={str(n):sum(r[METHODS[2]]['inliers']==n for r in rows) for n in range(5)},
        production_enabled=False, model_weights_updated=False,
        scope='GT-assisted, reused development poses, same source session; not deployment evaluation')
    for ref in METHODS[:2]:
        recovered = [r for r in rows if r[ref]['error_mm'] is None and r[METHODS[2]]['error_mm'] is not None]
        result.setdefault('newly_supported',{})[ref] = summarize(recovered,METHODS[2]) if recovered else None
    write_json(args.output/'report.json',result)
    print(json.dumps({k:v for k,v in result.items() if not k.startswith('rows')},indent=2),flush=True)


if __name__ == '__main__': main()
