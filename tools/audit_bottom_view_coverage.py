"""Read-only camera orientation and training coverage audit (Blender Z-up only)."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/bottom_view_audit_v1'


def orientation_from_extrinsic(extrinsic):
    """Return elevation/orientation from an OpenCV world-to-camera pose, Z-up."""
    e = np.asarray(extrinsic, dtype=float)
    if e.shape not in ((3, 4), (4, 4)) or not np.isfinite(e).all():
        raise ValueError('Expected finite 3x4 or 4x4 world-to-camera extrinsic')
    r = e[:3, :3]
    if not np.allclose(r @ r.T, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(r), 1, atol=1e-4):
        raise ValueError('Expected a proper rotation, not a scaled Blender matrix')
    direction = r.T @ np.array([0., 0., 1.])
    elevation = float(np.degrees(np.arcsin(np.clip(-direction[2], -1, 1))))
    orientation = ('top' if elevation >= 60 else 'bottom' if elevation <= -60
                   else 'oblique_above' if elevation >= 20 else
                   'oblique_below' if elevation <= -20 else 'side')
    return elevation, orientation


def cameras(scene):
    meta = json.loads((scene / 'scene_meta.json').read_text())
    rows = []
    for v in meta['views']:
        e = np.asarray(v['extrinsic'], dtype=float)
        center = -e[:3, :3].T @ e[:3, 3]
        direction = e[:3, :3].T @ np.array([0., 0., 1.])
        # Blender world is Z-up; direction is from camera into the scene.
        elevation, orientation = orientation_from_extrinsic(e)
        rows.append(dict(view_id=v['view_id'], camera=v['camera'],
                         elevation_deg=elevation, orientation=orientation,
                         center=center.tolist(), forward=direction.tolist()))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / 'output/pose_diverse_core23_768_v1/manifest.json').read_text())
    rigs = {}
    count = {s: Counter() for s in ('train', 'val')}
    by_dataset = {}
    # Audit every manifest frame, not just one representative rig.
    for rec in manifest['records']:
        rows = cameras(ROOT / rec['scene'])
        rigs.setdefault(rec['dataset'], dict(scene=rec['scene'], cameras=rows))
        selected = [r for r in rows if r['view_id'] in rec['views']]
        assert len(selected) == len(rec['views'])
        for row in selected:
            count[rec['split']][row['orientation']] += 1
            key = f"{rec['split']}:{rec['dataset']}"
            by_dataset.setdefault(key, Counter())[row['orientation']] += 1
    for path in sorted((ROOT / 'dataset/BlenderPyMouse').iterdir()):
        if not path.is_dir() or path.name in rigs:
            continue
        candidates = sorted(path.glob('frame_*/scene_meta.json'))
        if not candidates:
            continue
        try:
            rows = cameras(candidates[0].parent)
        except (KeyError, TypeError):
            continue
        rigs[path.name] = dict(scene=str(candidates[0].parent.relative_to(ROOT)), cameras=rows)
    result = dict(scope='Blender world Z-up; not valid for unaligned real calibrations',
                  elevation_definition='asin(-camera_forward_world_z); >=60 top, <=-60 bottom',
                  training_manifest='output/pose_diverse_core23_768_v1/manifest.json',
                  selected_image_counts={k: dict(v) for k, v in count.items()},
                  selected_by_dataset={k: dict(v) for k, v in by_dataset.items()},
                  representative_rigs=rigs,
                  weights_or_data_modified=False)
    (OUT / 'coverage.json').write_text(json.dumps(result, indent=2) + '\n')
    chosen = [k for k in rigs if k.startswith('Homecage-BlackBrown')]
    chosen += ['PyMouseStudio-BrownFur-WashFace-Core23']
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), layout='constrained')
    for ax in axes.flat:
        ax.axis('off')
    for j, name in enumerate(chosen):
        rig = rigs[name]
        rows = rig['cameras'] if j == 0 else [r for r in rig['cameras'] if r['view_id'] in (0, 2, 6, 7)]
        for ax, row in zip(axes[j], rows):
            path = ROOT / rig['scene'] / 'blended_images' / f"{row['view_id']:08d}.jpg"
            ax.imshow(Image.open(path))
            ax.set_title(f"{row['camera']} / id={row['view_id']}\n{row['orientation']} ({row['elevation_deg']:.1f} deg)")
    fig.suptitle('Actual physical camera directions (Blender Z-up)\nHomecage dual mouse / Cylinder single mouse')
    fig.savefig(OUT / 'camera_orientation.png', dpi=130)
    plt.close(fig)
    print(json.dumps({k: result[k] for k in ('selected_image_counts', 'selected_by_dataset')}, indent=2))
    for name, rig in rigs.items():
        print(name, [(r['view_id'], r['orientation'], round(r['elevation_deg'], 1)) for r in rig['cameras']])


if __name__ == '__main__':
    main()
