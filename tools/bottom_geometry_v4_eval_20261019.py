"""v4 development evaluator. Every invocation is explicit; sealed inference is absent."""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import bottom_geometry_v4_core_20261019 as core
import bottom_geometry_v4_runtime_20261019 as rt
import bottom_geometry_v3_losses_20261018 as v3
import bottom_geometry_v3_data_20261018 as data


def pixel_errors(actual, expected, valid):
    error = ((actual-expected) / 512 * actual.new_tensor([704, 768])).norm(dim=-1)
    return [float(x) if ok and torch.isfinite(x) else None for x, ok in zip(error, valid)]


def score(errors):
    n = len(errors); finite = sorted(x for x in errors if x is not None)
    rank = math.ceil(.95*n)-1
    return dict(support=n, coverage32=sum(x <= 32 for x in finite)/n if n else None,
                invalid=n-len(finite), p95=finite[rank] if 0 <= rank < len(finite) else None)


def dense_errors(e, k, z, b):
    result = dict(bottom=[], other=[])
    ge, gk, gz = [b[name][0].double() for name in ('E', 'K', 'depth')]
    e, k, z = e.double(), k.double(), z.double()
    yy, xx = torch.meshgrid(torch.arange(8, 512, 16, device=z.device),
                            torch.arange(8, 512, 16, device=z.device), indexing='ij')
    for source in range(len(e)):
        mask = b['valid'][0, source, yy, xx] & b['body'][0, source, yy, xx]
        x, y = xx[mask], yy[mask]
        if not len(x):
            continue
        xy = torch.stack((x, y), -1).double()
        for target in range(len(e)):
            if source == target:
                continue
            expected, tz, good = core.project_depth(ge, gk, gz[source, y, x], source, target, xy)
            good &= (expected >= 0).all(-1) & (expected < 511).all(-1)
            u, v = expected.nan_to_num().round().long().clamp(0, 511).T
            target_z = gz[target, v, u]
            good &= b['valid'][0, target, v, u] & ((target_z-tz).abs() < .01*target_z+.001)
            if not good.any():
                continue
            actual, _, valid = core.project_depth(e, k, z[source, y[good], x[good]],
                                                  source, target, xy[good])
            role = 'bottom' if 'bottom' in (b['roles'][source], b['roles'][target]) else 'other'
            result[role].extend(pixel_errors(actual, expected[good], valid))
    return result


def sparse_errors(z, b):
    result = dict(bottom=[], other=[])
    for s in range(len(b['roles'])):
        xy = b['xy'][0, s]
        depth = F.grid_sample(z[s][None, None], ((xy+.5)/512*2-1)[None, None],
                              align_corners=False, padding_mode='border').reshape(-1)
        for t in range(len(b['roles'])):
            if s == t:
                continue
            mask = b['known'][0, s] & b['known'][0, t]
            actual, _, good = core.project_depth(b['E'][0], b['K'][0], depth, s, t, xy)
            role = 'bottom' if 'bottom' in (b['roles'][s], b['roles'][t]) else 'other'
            result[role].extend(pixel_errors(actual[mask], b['xy'][0, t, mask], good[mask]))
    return result


def selected_records(manifest, sources):
    grouped = collections.defaultdict(list)
    for index, rec in enumerate(manifest['records']):
        if rec['split'] == 'val' and rec['evaluation_role'] in sources:
            grouped[rec['evaluation_role'], rec['group']].append(index)
    selected = []
    for (source, _), indices in sorted(grouped.items()):
        count = min(8, len(indices)) if source == 'free_motion_replay' else len(indices)
        selected.extend(indices[int(j)] for j in sorted(set(np.linspace(0, len(indices)-1, count).round())))
    return selected


@torch.no_grad()
def train_scale(model, manifest):
    indices = [i for i, r in enumerate(manifest['records']) if r['kind'] == 'real_labeled' and r['split'] == 'train']
    if not indices:
        return None, dict(available=False)
    indices = [indices[int(j)] for j in sorted(set(np.linspace(0, len(indices)-1, min(24, len(indices))).round()))]
    grid = torch.linspace(math.log(.05), math.log(20), 129, device='cuda').exp()
    cost = torch.zeros_like(grid); identities = []
    for index in indices:
        rec = manifest['records'][index]; pack = next(rt.evaluation_packs(rec, manifest))
        b = rt.load(rec, manifest, pack); pred = model(b['images'])
        cost += v3.sparse_grid_cost(v3.sparse_coefficients(pred, b), grid)/len(indices)
        identities.append(dict(index=index, group=rec['group'], frame=rec.get('frame'), views=pack))
    j = int(cost.argmin()); scalar = float(grid[j])
    return scalar, dict(available=True, scale=scalar, samples=identities, boundary=j in (0, len(grid)-1),
                        purpose='TRAIN-only global calibration diagnostic; matched first-pack input order, not universal SI alignment')


def camera_rows(e, b, pack):
    result = []
    for s in range(len(pack)):
        for t in range(s+1, len(pack)):
            pr, pt = core.relative_pose(e, s, t); gr, gt = core.relative_pose(b['E'][0], s, t)
            direction = None
            if pt.norm() > 1e-8 and gt.norm() > 1e-8:
                direction = float(torch.acos((F.normalize(pt, dim=0)*F.normalize(gt, dim=0)).sum().clamp(-1, 1))*180/math.pi)
            result.append(dict(source=pack[s], target=pack[t], star=s == 0,
                               pair_role='bottom' if 'bottom' in (b['roles'][s], b['roles'][t]) else 'other',
                               side_side=all(b['roles'][j] in ('side', 'other') for j in (s, t)),
                               gt_angle_deg=float(core.angle_degrees(gr)),
                               rotation_deg=float(core.angle_degrees(pr @ gr.T)), direction_deg=direction))
    return result


def aggregate(rows):
    result = {}
    for role in ('bottom', 'other'):
        pairs = [c for r in rows for c in r['cameras'] if c['pair_role'] == role]
        metrics = dict(star_rotation=core.rotation_summary([c['rotation_deg'] for c in pairs if c['star']]),
                       all_pair_rotation=core.rotation_summary([c['rotation_deg'] for c in pairs]),
                       star_direction=core.rotation_summary([c['direction_deg'] for c in pairs if c['star'] and c['direction_deg'] is not None]),
                       undefined_star_directions=sum(c['star'] and c['direction_deg'] is None for c in pairs))
        metrics['gt_angle_bins_star'] = {
            f'{lo}-{min(180, hi)}': core.rotation_summary([c['rotation_deg'] for c in pairs if c['star'] and lo <= c['gt_angle_deg'] < hi])
            for lo, hi in [(0, 45), (45, 90), (90, 135), (135, 180.001)]}
        names = {'raw_joint', 'known_camera_body_pack_oracle', 'known_camera_sparse',
                 'matched_train_scale_sparse', 'known_baseline_scale_sparse'}
        names.update(name for r in rows for name in r['errors'])
        for name in sorted(names):
            metrics[name] = score([v for r in rows for v in r['errors'].get(name, {}).get(role, [])])
        result[role] = metrics
    return result


@torch.no_grad()
def evaluate(model, manifest, output, *, intervention='original', sources=None, limit=None, provenance=None, train_probes=False, decompose=False):
    sources = sources or ['unseen_rig_gate', 'free_motion_replay', 'known_rig_diagnostic']
    probes = {i['index']: i['views'][1:]+i['views'][:1] for i in data.calibration_items(manifest)
              if i['sampling_role'] == 'new_rig'} if train_probes else {}
    indices = list(probes) if train_probes else selected_records(manifest, sources)
    if train_probes:
        sources = ['randomized_rig_train']
    if limit is not None:
        indices = indices[:limit]
    scalar, calibration = train_scale(model, manifest) if intervention == 'original' and not train_probes else (None, dict(available=False, reason='probe/background diagnostic; global scale fitting disabled'))
    rows = []; skipped = []
    for index in indices:
        rec = manifest['records'][index]
        for pack in [probes[index]] if train_probes else rt.evaluation_packs(rec, manifest):
            b = rt.load(rec, manifest, pack)
            try:
                b = rt.background(b, intervention)
            except ValueError as error:
                skipped.append(dict(index=index, views=pack, reason=str(error))); continue
            pred = model(b['images']); e, k = rt.geo.encoding_to_camera(pred['pose_enc'], (512, 512))
            z = pred['depth'][0, ..., 0]
            errors = dict(raw_joint=dense_errors(e[0], k[0], z, b))
            scale = core.body_scale(z, b['depth'][0], b['valid'][0], b['body'][0])
            per_view = core.body_scale(z, b['depth'][0], b['valid'][0], b['body'][0], per_view=True)
            depth_rows = []
            for v in range(len(pack)):
                mask = b['valid'][0, v] & b['body'][0, v]
                if mask.any():
                    gt, zz = b['depth'][0, v][mask], z[v][mask]
                    row = dict(view=pack[v], body_pixels=int(mask.sum()), role=b['roles'][v],
                               unaligned_absrel_diagnostic=float(((zz-gt).abs()/gt.clamp_min(1e-8)).mean()))
                    for name, ss in [('common_pack_body_oracle', scale), ('per_view_shape_oracle', per_view[v])]:
                        row[name] = dict(scale=float(ss), absrel=float(((ss*zz-gt).abs()/gt.clamp_min(1e-8)).mean()))
                    depth_rows.append(row)
            if scale is not None:
                errors['known_camera_body_pack_oracle'] = dense_errors(b['E'][0], b['K'][0], z*scale, b)
                if decompose:
                    for name, (oe, ok, oz) in core.component_oracles(e[0], k[0], z, b['E'][0], b['K'][0], b['depth'][0], scale).items():
                        errors[name] = dense_errors(oe, ok, oz, b)
            baseline = core.baseline_scale(e[0], b['E'][0], b['roles'])
            if rec['kind'] == 'real_labeled':
                errors['known_camera_sparse'] = sparse_errors(z, b)
                if scalar is not None:
                    errors['matched_train_scale_sparse'] = sparse_errors(z*scalar, b)
                if baseline is not None:
                    errors['known_baseline_scale_sparse'] = sparse_errors(z*baseline, b)
                else:
                    # Keep failed-estimation samples in the annotation denominator.
                    errors['known_baseline_scale_sparse'] = {role: [None]*len(v) for role, v in errors['known_camera_sparse'].items()}
            rows.append(dict(index=index, group=rec['group'], rig=rec.get('rig'), frame=rec.get('frame'),
                             scene=rec.get('scene'), context=pack, evaluation_role=rec['evaluation_role'],
                             cameras=camera_rows(e[0], b, pack), body_depth=depth_rows, errors=errors,
                             intrinsic_diagnostics=[dict(view=pack[v], fx_ratio=float(k[0,v,0,0]/b['K'][0,v,0,0]),
                                                         fy_ratio=float(k[0,v,1,1]/b['K'][0,v,1,1]),
                                                         principal_offset_px=float((k[0,v,:2,2]-b['K'][0,v,:2,2]).norm())) for v in range(len(pack))],
                             baseline_scale=float(baseline) if baseline is not None else None))
        print('V4_EVAL', index, len(rows), flush=True)
    rt.write_new(output, dict(version=4, intervention=intervention, provenance=provenance,
                             sealed_opened=False, optimizer_steps=0, partial=limit is not None, fixed_train_probes=train_probes, component_oracles_enabled=decompose,
                             train_scale_diagnostic=calibration, rows=rows, skipped=skipped,
                             summary={source: aggregate([r for r in rows if r['evaluation_role'] == source]) for source in sources},
                             per_rig={rig: aggregate([r for r in rows if r['rig'] == rig]) for rig in sorted({r['rig'] for r in rows if r['rig']})},
                             per_source={g: aggregate([r for r in rows if r['group'] == g]) for g in sorted({r['group'] for r in rows})},
                             warnings=['Oracle fits use development GT; shape-only per-view scales cannot promote joint geometry.',
                                       'P95=null with positive support means the nearest-rank error is infinite/unavailable, never zero error.',
                                       'Real collections share one physical rig; per-source rows are not independent rigs.',
                                       'Background conditions require matched complete masks; skipped samples must be matched across conditions.',
                                       'Global TRAIN scale and baseline scale are diagnostics, not guarantees of SI scale recovery.']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--release', required=True, type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--state', choices=['ema', 'raw'], default='ema')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--background', choices=['original', 'gray', 'blur'], default='original')
    parser.add_argument('--sources', nargs='+', choices=['unseen_rig_gate', 'free_motion_replay', 'known_rig_diagnostic'])
    parser.add_argument('--limit', type=int, help='Engineering smoke only; output marked partial')
    parser.add_argument('--train-probes', action='store_true', help='48 fixed TRAIN-rig probes; separate from development')
    parser.add_argument('--decompose', action='store_true', help='Diagnostic GT R/centers/K/Z swaps; never a deployable result')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; historical evaluation cannot be overwritten')
    if args.train_probes and args.sources:
        parser.error('--train-probes cannot be combined with development --sources')
    manifest = rt.read_manifest(args.manifest)
    torch.set_num_threads(4); rt.geo.old.cv2.setNumThreads(1)
    lock_path = rt.gpu_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    import fcntl
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        model = rt.model(args.release, checkpoint=args.checkpoint, state=args.state)
        evaluate(model, manifest, args.output, intervention=args.background, sources=args.sources, limit=args.limit, train_probes=args.train_probes, decompose=args.decompose,
                 provenance=dict(manifest_sha256=rt.sha(args.manifest), release_sha256=rt.sha(args.release),
                                 checkpoint_sha256=rt.sha(args.checkpoint) if args.checkpoint else None, state=args.state))


if __name__ == '__main__':
    main()
