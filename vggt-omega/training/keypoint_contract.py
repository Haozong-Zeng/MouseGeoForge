"""Explicit availability masks; annotation absence is not physical occlusion."""
from pathlib import Path
import hashlib

import numpy as np


def read_validity(archive, path: Path):
    """Explicit archive flags take precedence over additive, geometry-derived sidecars."""
    flags = {}
    sidecar = path.with_name('keypoint_validity.npz')
    if sidecar.exists():
        with np.load(sidecar, allow_pickle=False) as saved:
            if int(saved['schema_version']) != 1:
                raise ValueError(f'Unsupported validity schema: {sidecar}')
            flags.update({k: saved[k] for k in ('xy_label_valid', 'visibility_valid',
                                               'world_valid', 'surface_valid') if k in saved})
    for key in ('xy_label_valid', 'visibility_valid', 'world_valid', 'surface_valid'):
        if key in archive:
            flags[key] = archive[key]
    defaults = dict(xy_label_valid=np.isfinite(archive['uv']).all(-1),
                    visibility_valid=archive['in_frame'],
                    world_valid=np.isfinite(archive['joint_world_xyz']).all(-1),
                    surface_valid=np.isfinite(archive['surface_world_xyz']).all(-1))
    for key, default in defaults.items():
        value = np.asarray(flags.get(key, default))
        if value.shape != default.shape or value.dtype != np.bool_:
            raise ValueError(f'{path}: {key} must be bool {default.shape}, got {value.shape}/{value.dtype}')
        flags[key] = value.copy()
    flags['xy_label_valid'] &= np.isfinite(archive['uv']).all(-1)
    flags['world_valid'] &= np.isfinite(archive['joint_world_xyz']).all(-1)
    flags['surface_valid'] &= np.isfinite(archive['surface_world_xyz']).all(-1)
    return flags


def legacy_point(item, width, height):
    """Only explicit coordinate validity can rescue an invisible legacy JSON point.

    Many older exporters put (0,0) in missing labels. Do not reinterpret those
    placeholders as ground truth merely because they are finite.
    """
    uv = item.get('uv', [item.get('x', np.nan), item.get('y', np.nan)])
    uv = np.asarray(uv, dtype=np.float32)
    if uv.shape != (2,):
        uv = np.full(2, np.nan, dtype=np.float32)
    finite = bool(np.isfinite(uv).all())
    inside = finite and 0 <= uv[0] < width and 0 <= uv[1] < height
    visible = bool(item.get('surface_visible', item.get('visible', False)))
    label = finite and bool(item.get('xy_label_valid', item.get('occluded_but_annotated', visible)))
    known = bool(item.get('visibility_valid', 'visible' in item or 'surface_visible' in item))
    return np.nan_to_num(uv), inside, label, visible and inside, known


def read_occlusion(archive, path):
    """Optional source-verified cause labels; absent is UNKNOWN, never self-occluded."""
    shape = archive['in_frame'].shape
    result = np.full(shape, -1, dtype=np.int8)
    sidecar = path.with_name('keypoint_occlusion.npz')
    if not sidecar.exists():
        return result
    with np.load(sidecar, allow_pickle=False) as a:
        if int(a['schema_version']) != 1:
            raise ValueError(f'Unsupported occlusion schema: {sidecar}')
        if 'annotation_sha256' not in a or str(a['annotation_sha256']) != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f'Stale or unverified occlusion annotation: {sidecar}')
        if not np.array_equal(a['names'], archive['names']) or not np.array_equal(a['view_ids'], archive['view_ids']):
            raise ValueError(f'Occlusion keypoint/view order mismatch: {sidecar}')
        if 'instance_ids' in archive and not np.array_equal(a['instance_ids'], archive['instance_ids']):
            raise ValueError(f'Occlusion instance order mismatch: {sidecar}')
        result = a['occlusion_cause'].astype(np.int8)
        if result.shape != shape or not np.isin(result, [-1, 0, 1, 2, 3]).all():
            raise ValueError(f'Invalid occlusion cause layout: {sidecar}')
        if not np.array_equal(a['occlusion_cause_valid'], result >= 0):
            raise ValueError(f'Invalid occlusion validity: {sidecar}')
    return result
