#!/usr/bin/env python3
"""Inspect evaluated Openfield mouse mesh projection and anomalous triangles."""

import json
import sys
from pathlib import Path

import bpy
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import convert_cylinder_renders_to_blenderpymouse_core23 as core23


def main():
    output = Path(sys.argv[sys.argv.index("--") + 1])
    scene = bpy.context.scene
    scene.frame_set(1)
    bpy.context.view_layer.update()
    mesh_obj = bpy.data.objects["MS_Mouse.001"]
    camera = bpy.data.objects["CameraOrbbec"]
    intrinsic = np.asarray(((camera.data.lens * 1024 / camera.data.sensor_width, 0, 512), (0, camera.data.lens * 1024 / camera.data.sensor_width, 512), (0, 0, 1)), dtype=np.float64)
    extrinsic = core23.camera_extrinsic(camera)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = mesh_obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        world = np.asarray([evaluated.matrix_world @ vertex.co for vertex in mesh.vertices], dtype=np.float64)
        camera_xyz, uv = core23.project_points(world, extrinsic, intrinsic)
        anomalies = []
        for index, triangle in enumerate(mesh.loop_triangles):
            ids = np.asarray(triangle.vertices, dtype=np.int64)
            points = uv[ids]
            extent = np.ptp(points, axis=0)
            area = abs(np.cross(points[1] - points[0], points[2] - points[0])) * 0.5
            if area > 100 or np.max(extent) > 30:
                anomalies.append({"triangle": index, "vertices": ids.tolist(), "uv": points.tolist(), "area_px2": float(area), "extent_px": extent.tolist(), "world": world[ids].tolist()})
        payload = {
            "vertex_count": len(mesh.vertices),
            "triangle_count": len(mesh.loop_triangles),
            "world_min": world.min(axis=0).tolist(),
            "world_max": world.max(axis=0).tolist(),
            "uv_min": np.nanmin(uv, axis=0).tolist(),
            "uv_max": np.nanmax(uv, axis=0).tolist(),
            "camera_z_min": float(camera_xyz[:, 2].min()),
            "camera_z_max": float(camera_xyz[:, 2].max()),
            "anomaly_count": len(anomalies),
            "largest_anomalies": sorted(anomalies, key=lambda item: item["area_px2"], reverse=True)[:50],
        }
    finally:
        evaluated.to_mesh_clear()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "largest_anomalies"}, indent=2))


main()
