"""Final integrity and saved geometry roundtrip verification; no label edits."""
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from ingest_pymousestudio_full_return_20260930 import BASE, PACK, digest
from audit_pymousestudio_manual_multiview import OUT, projection, solve


def main():
    manifest = json.loads((BASE / 'ingestion_manifest.json').read_text())
    for row in manifest['files']:
        assert digest(BASE / row['path']) == row['sha256'], row['path']
    for row in manifest['calibration']:
        assert digest(BASE / row['path']) == row['sha256']
        assert digest(PACK / row['path']) == row['sha256']
    assert len(list((BASE / 'normalized_observations').rglob('*.npz'))) == 300
    assert len(list((BASE / 'geometry_diagnostics_NOT_GT').rglob('*.npz'))) == 300
    points = observations = loo_checks = 0
    max_roundtrip_error = 0.
    for group in manifest['frame_groups']:
        with np.load(BASE / group['normalized_file']) as z:
            norm = dict(z)
        path = BASE / 'geometry_diagnostics_NOT_GT' / group['split'] / group['session'] / f"frame_{group['frame']:06d}.npz"
        with np.load(path) as z:
            geom = dict(z)
        uv = geom['uv_observed']
        assert norm['uv'].shape == (len(group['cameras']), 1, 23, 2)
        np.testing.assert_equal(norm['uv'][:, 0], uv)
        np.testing.assert_equal(norm['xy_observed'][:, 0], np.isfinite(uv).all(-1))
        np.testing.assert_equal(norm['semantic_names'], geom['names'])
        np.testing.assert_equal(norm['camera_names'], geom['camera_names'])
        assert not norm['train_allowed'] and not geom['train_allowed'] and geom['not_ground_truth']
        assert not norm['tail_edge_filter_applied'] and norm['instance_ids'].tolist() == [0]
        cal = json.loads((BASE / 'calibration' / group['session'] / 'rectified_calibration.json').read_text())
        K, E = geom['K'], geom['world_to_camera_mm']
        np.testing.assert_equal(K, np.array([cal[c]['K'] for c in group['cameras']]))
        np.testing.assert_equal(E, np.array([cal[c]['world_to_camera_mm'] for c in group['cameras']]))
        for key, errkey in [('xyz_all_ls_mm', 'error_all_ls_px'), ('xyz_consensus_mm', 'error_consensus_px')]:
            assert geom[key].shape == (23, 3)
            for k, xyz in enumerate(geom[key]):
                if not np.isfinite(xyz).all():
                    assert np.isnan(geom[errkey][:, k]).all()
                    continue
                pred, _ = projection(xyz, K, E)
                error = np.linalg.norm(pred - uv[:, k], axis=-1)
                np.testing.assert_allclose(error, geom[errkey][:, k], atol=1e-8, rtol=1e-9)
                max_roundtrip_error = max(max_roundtrip_error, float(np.nanmax(np.abs(error - geom[errkey][:, k]))))
        looe = np.linalg.norm(geom['uv_leave_one_view_out'] - uv, axis=-1)
        np.testing.assert_allclose(looe, geom['error_leave_one_view_out_px'], atol=1e-8, rtol=1e-9)
        # One independently recomputed held-out prediction for each of 300 groups.
        candidates = np.flatnonzero(np.isfinite(uv).all(-1).sum(0) >= 3)
        if len(candidates):
            k = candidates[0]
            ids = np.flatnonzero(np.isfinite(uv[:, k]).all(-1))
            v, rest = ids[0], ids[1:]
            xyz = solve(uv[rest, k], K[rest], E[rest])
            pred, _ = projection(xyz, K[v:v+1], E[v:v+1])
            np.testing.assert_allclose(pred[0], geom['uv_leave_one_view_out'][v, k], atol=1e-7)
            loo_checks += 1
        points += int(np.isfinite(geom['xyz_all_ls_mm']).all(-1).sum())
        observations += int(np.isfinite(uv).all(-1).sum())
    images = [p for p in OUT.rglob('*') if p.suffix.lower() in ('.jpg', '.png')]
    for p in images:
        with Image.open(p) as im:
            im.verify()
    for relative in re.findall(r'(?:src|href)=["\']([^"\']+)["\']', (OUT / 'index.html').read_text()):
        if not relative.startswith(('http:', 'https:', '#')):
            assert (OUT / relative).is_file(), relative
    result = dict(status='passed', raw_files_sha256_verified=len(manifest['files']),
                  calibration_files_unchanged=len(manifest['calibration']), groups_verified=300,
                  normalized_observations=observations, triangulated_points=points,
                  saved_projection_max_difference_px=max_roundtrip_error,
                  independent_loo_rechecks=loo_checks, visualization_images_decoded=len(images),
                  original_labels_unchanged=True, calibration_unchanged=True, training_disabled=True)
    (OUT / 'verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
