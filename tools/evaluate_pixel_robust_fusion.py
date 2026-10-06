"""Locked, diagnostic-only fusion comparison on 24 previously unused pose scenes.

No parameter fitting. Reuses V2 calibration temperatures, not its test predictions.
Both old and new evaluation blocks derive from the same source session; this is
not independent-animal validation or authorization to enable production fusion.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from diagnose_occlusion_pose import ROOT, torch, infer, fingerprint, write_json
from vggt_omega.utils.robust_pose_geometry import triangulate_robust
from mousegeoforge.uncertainty_calibration import uncertainty_metrics
from calibrate_pose_heldout import points


VARIANTS = {
    'uniform_legacy': ('uniform', None), 'uniform_pixel': ('uniform', 3.),
    'original_legacy': ('original', None), 'original_pixel': ('original', 3.),
    'learned_legacy': ('fitted', None), 'learned_pixel': ('fitted', 3.),
}


def protocol():
    folder = ROOT/'output/occlusion_pose_v2/calibration'
    old = json.loads((folder/'protocol.json').read_text())
    report = json.loads((folder/'calibration_report.json').read_text())
    if fingerprint(ROOT/'output/pose_diverse_core23_768_v1/seed17_off/final_heads.pt') != old['head_checkpoint_sha256']:
        raise ValueError('Pose checkpoint changed since calibration')
    if fingerprint(ROOT/'output/occlusion_pose_v1/sigma_only_pilot.pt') != report['secondary_sigma_checkpoint_sha256']:
        raise ValueError('Sigma checkpoint changed since calibration')
    manifest = json.loads((ROOT/'output/pose_diverse_core23_768_v1/manifest.json').read_text())
    plan = json.loads((ROOT/'dataset/Homecage/LightningPose-CRIM13-Diverse400.plan.json').read_text())
    prior = old['records'] + manifest['records']
    used = {r['scene'] for r in prior}
    template = old['records'][0]
    nearby = [r['source_frame'] for r in prior if r['source_group'] == template['source_group']]
    candidates = []
    for k in plan['keyframes']:
        if Path(k['source_image'].replace('\\', '/')).parent.name != old['source_session']:
            continue
        scene = f'dataset/BlenderPyMouse/{template["dataset"]}/frame_{k["scene_frame"]:06d}'
        if scene in used or min(abs(k['source_image_frame']-s) for s in nearby) <= 100:
            continue
        if not (ROOT/scene/'keypoint_occlusion.npz').exists():
            continue
        candidates.append(dict(template, scene=scene, frame=k['scene_frame'], split='test',
            source_frame=k['source_image_frame'], annotation_sha256=fingerprint(ROOT/scene/'keypoints.npz')))
    candidates.sort(key=lambda r: r['source_frame'])
    if len(candidates) < 24:
        raise ValueError(f'Need 24 unused scenes, found {len(candidates)}; do not silently relax exclusion')
    records = [candidates[round(i*(len(candidates)-1)/23)] for i in range(24)]
    if len({r['scene'] for r in records}) != 24:
        raise ValueError('Duplicate selected scenes')
    return dict(records=records, candidates=len(candidates), exclusion_source_frames=100,
        variants=VARIANTS, temperatures=report['temperatures'],
        source_session=old['source_session'], independent_session=False,
        primary='learned_pixel versus learned_legacy (kernel only)',
        utility_reference='uniform_pixel',
        gates=dict(max_coverage_drop=.03, max_paired_mean_ratio=1.01, max_paired_p95_ratio=1.05),
        thresholds=dict(huber_delta_px=3., reprojection_threshold_px=8., iterations=5),
        head_sha256=fingerprint(ROOT/'output/pose_diverse_core23_768_v1/seed17_off/final_heads.pt'),
        sigma_sha256=fingerprint(ROOT/'output/occlusion_pose_v1/sigma_only_pilot.pt'),
        prior_protocol_sha256=fingerprint(folder/'protocol.json'),
        calibration_report_sha256=fingerprint(folder/'calibration_report.json'),
        geometry_code_sha256=fingerprint(ROOT/'vggt-omega/vggt_omega/utils/robust_pose_geometry.py'),
        mode='diagnostic_only_no_production_promotion',
        association='GT box-assisted; not end-to-end instance association')


def summarize(rows, key):
    errors = torch.tensor([r[key]['error_mm'] for r in rows if r[key]['error_mm'] is not None])
    return dict(points=len(rows), supported=len(errors), coverage=len(errors)/len(rows),
        mean_mm=float(errors.mean()) if len(errors) else None,
        p95_mm=float(torch.quantile(errors, .95)) if len(errors) else None,
        reasons=dict(Counter(r[key]['reason'] for r in rows)))


def compare(rows, reference, candidate, gates):
    common = [r for r in rows if all(r[k]['error_mm'] is not None for k in (reference, candidate))]
    paired = {}
    for key in (reference, candidate):
        e = torch.tensor([r[key]['error_mm'] for r in common])
        paired[key] = dict(mean_mm=float(e.mean()) if len(e) else None,
                          p95_mm=float(torch.quantile(e, .95)) if len(e) else None)
    a, b = paired[reference], paired[candidate]
    drop = summarize(rows, reference)['coverage']-summarize(rows, candidate)['coverage']
    passed = bool(common) and drop <= gates['max_coverage_drop'] and (
        b['mean_mm'] <= a['mean_mm']*gates['max_paired_mean_ratio'] and
        b['p95_mm'] <= a['p95_mm']*gates['max_paired_p95_ratio'])
    return dict(reference=reference, candidate=candidate, common_points=len(common),
                paired=paired, coverage_drop=drop, gate_passed=passed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT/'output/occlusion_pose_v3/pixel_fusion')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    spec = json.loads(json.dumps(protocol()))
    args.output.mkdir(parents=True, exist_ok=True)
    lock = args.output/'protocol.json'
    if lock.exists() and json.loads(lock.read_text()) != spec:
        raise ValueError('Locked protocol changed; use a new output directory')
    write_json(lock, spec)
    print('LOCKED', len(spec['records']), 'new test scenes;', spec['candidates'], 'eligible', flush=True)
    if args.prepare_only: return
    if not args.report_only:
        infer(ROOT/'output/pose_diverse_core23_768_v1', args.output, spec['records'])
    data = [torch.load(args.output/'predictions'/f'{i:03d}.pt', weights_only=True) for i in range(24)]
    head = torch.nn.Linear(data[0]['sigma_tokens'].shape[-1], 1)
    head.load_state_dict(torch.load(ROOT/'output/occlusion_pose_v1/sigma_only_pilot.pt', weights_only=True)['state_dict'])
    rows = []
    for item, record in zip(data, spec['records']):
        if item['record'] != record or fingerprint(ROOT/record['scene']/'keypoints.npz') != record['annotation_sha256']:
            raise ValueError('Cache or original annotations changed')
        p, t = item['predictions'], item['targets']
        h, w = record['image_hw']; scale = torch.tensor([w, h])
        proj = t['intrinsics'][0] @ t['extrinsics'][0, :, :3]
        oracle = [{m:q for q,m in zip(qs,ms)} for qs,ms in item['oracle_assignment']]
        native = float(item['scale'].flatten()[0])
        with torch.no_grad(): learned = 1e-3 + torch.nn.functional.softplus(head(item['sigma_tokens']).squeeze(-1))
        for m in range(2):
            qs = [o[m] for o in oracle]
            for k in range(23):
                uv = torch.stack([p['instance_keypoint_xy'][0,v,q,k] for v,q in enumerate(qs)])*scale
                scales = dict(uniform=None,
                    original=torch.stack([p['instance_keypoint_sigma'][0,v,q,k] for v,q in enumerate(qs)])[:,None]*scale*spec['temperatures']['original'],
                    fitted=torch.stack([learned[v,q,k] for v,q in enumerate(qs)])[:,None]*scale*spec['temperatures']['fitted'])
                row = dict(scene=record['scene'], instance=m, keypoint=k)
                for key, (mode, delta) in VARIANTS.items():
                    s = scales[mode]
                    out = triangulate_robust(uv, proj, t['keypoint_valid'][0,:,m,k],
                        sigma_px=s, uncertainty_calibrated=s is not None, huber_delta_px=delta)
                    row[key] = dict(error_mm=float((out.xyz-t['keypoint_world_xyz'][0,m,k]).norm())*native if out.supported else None,
                                    reason=out.reason)
                rows.append(row)
        print('Geometry', record['frame'], flush=True)
    # The uniform control must remain numerically identical after adding the option.
    if any(r['uniform_legacy'] != r['uniform_pixel'] for r in rows):
        raise AssertionError('Uniform baseline changed')
    x = points(data, head)
    result = dict(protocol_sha256=fingerprint(lock), rows=rows,
        summary={key:summarize(rows,key) for key in VARIANTS},
        comparisons={name:compare(rows,a,b,spec['gates']) for name,a,b in (
            ('primary_kernel', 'learned_legacy', 'learned_pixel'),
            ('utility', 'uniform_pixel', 'learned_pixel'),
            ('original_kernel', 'original_legacy', 'original_pixel'))},
        uncertainty={key:uncertainty_metrics(x['residual'],x[key]*spec['temperatures'][key]) for key in ('original','fitted')},
        production_enabled=False, model_weights_updated=False, independent_session=False)
    write_json(args.output/'report.json', result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}, indent=2), flush=True)


if __name__ == '__main__': main()
