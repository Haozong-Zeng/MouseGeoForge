"""Audit all synthetic keypoint archives; optionally add non-destructive validity sidecars.

No RGB, depth, cameras, coordinates, or original keypoints.npz are changed.
Missing anatomical channels are reported, never filled with invented coordinates.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def derive_validity(a):
    uv = a['uv']
    inside = a['in_frame'].astype(bool)
    finite = np.isfinite(uv).all(-1)
    return {
        'xy_label_valid': a.get('xy_label_valid', finite).astype(bool) & finite,
        'visibility_valid': a.get('visibility_valid', inside).astype(bool),
        'world_valid': a.get('world_valid', np.isfinite(a['joint_world_xyz']).all(-1)).astype(bool),
        'surface_valid': a.get('surface_valid', np.isfinite(a['surface_world_xyz']).all(-1)).astype(bool),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT/'dataset/BlenderPyMouse')
    p.add_argument('--output', type=Path, default=ROOT/'output/occlusion_pose_v1/label_audit.json')
    p.add_argument('--apply', action='store_true')
    args = p.parse_args()
    if args.root.resolve() != (ROOT/'dataset/BlenderPyMouse').resolve() and args.apply:
        raise ValueError('Automatic derivation is restricted to audited synthetic data')
    reports = []
    for ds in sorted(args.root.iterdir()):
        if not ds.is_dir() or ds.name.startswith('_'):
            continue
        counts, layouts, missing, issues = Counter(), Counter(), [], []
        for frame in sorted(ds.glob('frame_*')):
            if not frame.is_dir():
                continue
            counts['frames'] += 1
            path = frame/'keypoints.npz'
            if not path.exists():
                missing.append(frame.name)
                continue
            with np.load(path, allow_pickle=False) as archive:
                a = dict(archive)
            needed = ('names', 'uv', 'in_frame', 'surface_visible', 'joint_world_xyz', 'surface_world_xyz')
            absent = [k for k in needed if k not in a]
            if absent:
                issues.append({'frame': frame.name, 'missing_arrays': absent})
                continue
            flags = derive_validity(a)
            inside = a['in_frame'].astype(bool)
            valid = flags['xy_label_valid'] & inside
            visible = a['surface_visible'].astype(bool)
            layouts[','.join(map(str, a['names']))] += 1
            counts['npz'] += 1
            counts['valid_2d'] += int(valid.sum())
            counts['known_occluded'] += int((valid & flags['visibility_valid'] & ~visible).sum())
            counts['unknown_visibility'] += int((valid & ~flags['visibility_valid']).sum())
            counts['in_frame_nonfinite_uv'] += int((inside & ~np.isfinite(a['uv']).all(-1)).sum())
            for k in flags:
                counts['missing_field_'+k] += int(k not in a)
            sidecar = frame/'keypoint_validity.npz'
            if args.apply and not sidecar.exists():
                if counts['in_frame_nonfinite_uv']:
                    raise ValueError(f'Invalid in-frame coordinates: {path}')
                with sidecar.with_suffix('.npz.tmp').open('wb') as f:
                    np.savez_compressed(f, **flags, schema_version=np.asarray(1),
                                        provenance=np.asarray('derived_from_existing_blender_geometry'))
                sidecar.with_suffix('.npz.tmp').replace(sidecar)
                counts['sidecars_created'] += 1
            elif sidecar.exists():
                with np.load(sidecar, allow_pickle=False) as saved:
                    for key, expected in flags.items():
                        if not np.array_equal(saved[key], expected):
                            raise ValueError(f'Stale validity sidecar: {sidecar}, {key}')
                counts['sidecars_verified'] += 1
        reports.append(dict(dataset=ds.name, counts=dict(counts), layouts=dict(layouts),
                            missing_npz=missing, issues=issues))
        print(ds.name, dict(counts), flush=True)
    payload = dict(schema_version=1, applied=args.apply, datasets=reports,
                   totals=dict(sum((Counter(r['counts']) for r in reports), Counter())),
                   policy='Existing geometry only; no inferred anatomical channels; source backfill required for missing coordinates.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2)+'\n')


if __name__ == '__main__':
    main()
