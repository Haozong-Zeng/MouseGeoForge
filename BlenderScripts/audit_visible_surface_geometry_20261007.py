"""Read-only source geometry probe; writes only to a new diagnostic directory."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Matrix


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def camera_parameters(scene, camera, width, height):
    c = camera.data
    assert c.type == 'PERSP', c.type
    aspect = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    fit = c.sensor_fit
    if fit == 'AUTO':
        fit = 'HORIZONTAL' if width >= aspect * height else 'VERTICAL'
    sensor = c.sensor_height if c.sensor_fit == 'VERTICAL' else c.sensor_width
    factor = width if fit == 'HORIZONTAL' else aspect * height
    fx = c.lens * factor / sensor
    k = np.array([[fx, 0, width / 2 - c.shift_x * factor],
                  [0, fx / aspect, height / 2 + c.shift_y * factor / aspect], [0, 0, 1.]])
    loc, rot, _ = camera.matrix_world.decompose()
    rotation = Matrix(((1., 0, 0), (0, -1., 0), (0, 0, -1.))) @ rot.to_matrix().transposed()
    e = np.eye(4)
    e[:3, :3] = np.asarray(rotation)
    e[:3, 3] = np.asarray(-(rotation @ loc))
    return k, e


def mesh_arrays(obj, depsgraph):
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        vertices = np.empty((len(mesh.vertices), 3), np.float64)
        mesh.vertices.foreach_get('co', vertices.ravel())
        transform = np.asarray(evaluated.matrix_world, dtype=np.float64)
        vertices = vertices @ transform[:3, :3].T + transform[:3, 3]
        triangles = np.empty((len(mesh.loop_triangles), 3), np.int32)
        mesh.loop_triangles.foreach_get('vertices', triangles.ravel())
        return vertices, triangles
    finally:
        evaluated.to_mesh_clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    root = Path(protocol['windows_root'])
    dest = root / protocol['output']
    results = []
    opened = None
    for rec in protocol['records']:
        source = root / rec['source_blend']
        if str(source) != opened:
            bpy.ops.wm.open_mainfile(filepath=str(source))
            opened = str(source)
            for obj in bpy.data.objects:
                if obj.type == 'CURVES':
                    obj.hide_viewport = True
        scene = bpy.context.scene
        scene.frame_set(rec['frame'])
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        ann = dict(np.load(root / rec['scene'] / 'keypoints.npz'))
        probes = []
        for instance, name in enumerate(rec['meshes']):
            obj = bpy.data.objects[name]
            vertices, triangles = mesh_arrays(obj, dg)
            ids = ann['support_vertex_ids'] if rec['kind'] == 'single' else ann['support_vertex_ids'][instance]
            reference = ann['support_world_xyz'] if rec['kind'] == 'single' else ann['support_world_xyz'][instance]
            errors = np.linalg.norm(vertices[ids] - reference, axis=-1)
            probes.append(dict(mesh=name, vertices=len(vertices), triangles=len(triangles),
                               support_max=float(errors.max()), support_median=float(np.median(errors)),
                               bbox_min=vertices.min(0).tolist(), bbox_max=vertices.max(0).tolist(),
                               modifiers=[dict(name=m.name, type=m.type, viewport=m.show_viewport, render=m.show_render)
                                          for m in obj.modifiers]))
        cameras = []
        for view in rec['views']:
            camera = bpy.data.objects[str(ann['camera_names'][view])]
            k, e = camera_parameters(scene, camera, *rec['source_size_wh'])
            cameras.append(dict(view=view, camera=camera.name,
                                k_max_abs=float(np.abs(k - ann['intrinsics'][view]).max()),
                                e_max_abs=float(np.abs(e - ann['extrinsics'][view]).max()),
                                intrinsic=k.tolist(), extrinsic=e.tolist()))
        inventory = [dict(name=o.name, type=o.type, hide_render=o.hide_render,
                          collections=[c.name for c in o.users_collection],
                          properties={k:str(o[k]) for k in o.keys() if 'depth' in k.lower()},
                          materials=[s.material.name if s.material else None for s in o.material_slots])
                     for o in scene.objects if o.type in ('MESH', 'CURVE', 'CURVES', 'SURFACE', 'VOLUME', 'META', 'FONT')]
        result = dict(index=rec['index'], phase=rec['phase'], kind=rec['kind'],
                      source_blend=rec['source_blend'], frame=rec['frame'], blender=bpy.app.version_string,
                      unit_scale_meters=scene.unit_settings.scale_length, meshes=probes, cameras=cameras,
                      object_inventory=inventory)
        write(dest / 'probe' / f'{rec["index"]:02d}.json', result)
        results.append({k:v for k,v in result.items() if k != 'object_inventory'})
        write(dest / 'probe_results.json', results)
        print('SURFACE_PROBE', rec['index'], 'support', [p['support_max'] for p in probes],
              'K/E', [(c['k_max_abs'], c['e_max_abs']) for c in cameras], flush=True)


if __name__ == '__main__':
    main()
