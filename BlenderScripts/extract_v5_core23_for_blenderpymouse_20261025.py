"""Read-only source replay for six V5 poses and Core23 source/body-ray checks.

No RGB rendering, GPU work, .blend save, training, or model inference.
"""
import hashlib
import json
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/v5_blenderpymouse_conversion_20261025'
SURFACE_IDS = [5, 6, 22]


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def main():
    plan = json.loads((OUT / 'conversion_plan.json').read_text(encoding='utf-8'))
    reports = []
    for source in plan['sources']:
        path = ROOT / source['blend']
        assert sha(path) == source['source_sha256']
        bpy.ops.wm.open_mainfile(filepath=str(path))
        for obj in bpy.data.objects:
            if obj.type == 'CURVES':
                obj.hide_viewport = True
        scene = bpy.context.scene
        reference_ds = ROOT / source['reference_dataset']
        definition = json.loads((reference_ds / 'surface_keypoint_definition.json').read_text())
        assert definition['source_blend_sha256'] == source['source_sha256']
        with np.load(reference_ds / 'frame_000049/keypoints.npz', allow_pickle=False) as z:
            reference = dict(z)
        with (reference_ds / 'frame_000049/keypoints.json').open() as f:
            old_json = json.load(f)
        arm = bpy.data.objects[old_json['armature']]
        mesh = bpy.data.objects[old_json['mesh']]
        ids = reference['support_vertex_ids']
        assert ids.shape == (23, 16) and (ids >= 0).all()
        for frame in (49, 97, 145):
            scene.frame_set(frame)
            bpy.context.view_layer.update()
            scene.frame_set(frame)
            bpy.context.view_layer.update()
            joint = np.array([arm.matrix_world @ arm.pose.bones[str(b)].head for b in reference['bones']], dtype=np.float64)
            ev = mesh.evaluated_get(bpy.context.evaluated_depsgraph_get())
            em = ev.to_mesh()
            try:
                xyz = np.array([ev.matrix_world @ v.co for v in em.vertices], dtype=np.float64)
                faces = [list(p.vertices) for p in em.polygons]
            finally:
                ev.to_mesh_clear()
            assert ids.max() < len(xyz)
            support = xyz[ids]
            surface = support.mean(axis=1)
            for anchor in definition['anchors']:
                k = anchor['core23_id']
                assert k in SURFACE_IDS and abs(sum(anchor['barycentric_weights']) - 1) < 1e-8
                point = np.asarray(anchor['barycentric_weights']) @ xyz[anchor['triangle_vertex_ids']]
                joint[k] = surface[k] = point
            baseline_path = reference_ds / f'frame_{frame:06d}/keypoints.npz'
            replay = dict(source=source['id'], frame=frame, source_sha256=source['source_sha256'],
                          reference_source_sha256=sha(reference_ds / 'frame_000049/keypoints.npz'))
            if baseline_path.exists():
                with np.load(baseline_path, allow_pickle=False) as baseline:
                    errors = {key: float(np.linalg.norm(actual - baseline[key], axis=-1).max())
                              for key, actual in [('joint_world_xyz', joint), ('surface_world_xyz', surface), ('support_world_xyz', support)]}
                assert max(errors.values()) < .002, (source['id'], frame, errors)
                replay.update(existing_frame_geometry_verified=True, errors_mm=errors, baseline_sha256=sha(baseline_path))
            else:
                replay.update(existing_frame_geometry_verified=False,
                              method='Same pinned source, bones, fixed REST triangle anchors and support IDs; frame49 replay verified against existing dataset.')
            dest = OUT / 'source_geometry' / source['id']
            dest.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest / f'frame_{frame:06d}.npz',
                **{key: reference[key] for key in ('names', 'semantic_names', 'groups', 'bones', 'endpoints', 'coordinate_kind', 'definition_version')},
                joint_world_xyz=joint.astype(np.float32), surface_world_xyz=surface.astype(np.float32),
                support_vertex_ids=ids, support_world_xyz=support.astype(np.float32),
                source_frame=np.asarray(frame, np.int32), source_sha256=np.asarray(source['source_sha256']),
                reference_annotation_sha256=np.asarray(replay['reference_source_sha256']))
            tree = BVHTree.FromPolygons([Vector(v) for v in xyz], faces)
            camera_cache = {}
            records = [r for r in plan['records'] if r['motion_id'] == source['id'] and r['frame'] == frame]
            for record in records:
                anchors_visible, self_blocked = [], []
                for view in record['views']:
                    e = np.asarray(view['E'], dtype=float)
                    signature = e.tobytes()
                    if signature not in camera_cache:
                        origin = Vector((-e[:3, :3].T @ e[:3, 3]).tolist())
                        def blocked(point, epsilon):
                            ray = Vector(point) - origin
                            distance = ray.length
                            if distance <= 1e-6:
                                return False
                            return tree.ray_cast(origin, ray.normalized(), max(0, distance - epsilon(distance)))[0] is not None
                        visible = np.zeros(23, bool)
                        for k in SURFACE_IDS:
                            visible[k] = not blocked(joint[k], lambda d: max(.0002, d * 2e-6))
                        blockers = np.array([[blocked(p, lambda d: .05) for p in points] for points in support], bool)
                        camera_cache[signature] = (visible, blockers)
                    visible, blockers = camera_cache[signature]
                    anchors_visible.append(visible)
                    self_blocked.append(blockers)
                target = OUT / 'body_ray_visibility' / record['export_dataset']
                target.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(target / (record['export_scene'] + '.npz'),
                                    anchor_body_visible=np.array(anchors_visible), self_support_blocked=np.array(self_blocked))
            reports.append(replay)
            write(OUT / 'source_geometry_progress.json', dict(completed=reports, latest_views=len(camera_cache)))
            print('CORE23_SOURCE_REPLAY', source['id'], frame, replay.get('errors_mm'), len(records), flush=True)
        assert sha(path) == source['source_sha256']
    assert len(reports) == 6
    write(OUT / 'source_geometry_receipt.json', dict(passed=True, sources_unchanged=True, poses=reports,
                extraction_script_sha256=sha(Path(__file__)), model_calls=0, renders=0,
                meaning='World coordinates replayed; visibility here is body-only geometry, full-scene raster truth is checked by converter.'))


if __name__ == '__main__':
    main()
