#!/usr/bin/env python3
"""Diagnose view-layer and camera-frustum visibility without rendering."""

from __future__ import annotations

import json

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


CAMERA_NAMES = tuple(f"Camera.{index:03d}" for index in range(1, 9))
OBJECT_NAMES = (
    "Part_GlassCylinder160mm",
    "Building_Floor",
    "Part_WhiteHexagonBackground",
    "MS_Mouse.001",
    "MS_Mouse.002",
)


def layer_tree(layer_collection):
    return {
        "name": layer_collection.name,
        "exclude": bool(layer_collection.exclude),
        "hide_viewport": bool(layer_collection.hide_viewport),
        "holdout": bool(layer_collection.holdout),
        "indirect_only": bool(layer_collection.indirect_only),
        "children": [layer_tree(child) for child in layer_collection.children],
    }


def world_bbox(obj):
    return [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]


def projection_record(scene, camera, obj):
    points = [world_to_camera_view(scene, camera, point) for point in world_bbox(obj)]
    xyz = np.asarray([(point.x, point.y, point.z) for point in points], dtype=float)
    return {
        "ndc_x": [float(xyz[:, 0].min()), float(xyz[:, 0].max())],
        "ndc_y": [float(xyz[:, 1].min()), float(xyz[:, 1].max())],
        "depth": [float(xyz[:, 2].min()), float(xyz[:, 2].max())],
        "overlaps_frame": bool(
            xyz[:, 0].max() >= 0
            and xyz[:, 0].min() <= 1
            and xyz[:, 1].max() >= 0
            and xyz[:, 1].min() <= 1
            and xyz[:, 2].max() > 0
        ),
    }


scene = bpy.context.scene
scene.frame_set(scene.frame_start)
bpy.context.view_layer.update()
payload = {
    "blend": bpy.data.filepath,
    "view_layer": bpy.context.view_layer.name,
    "view_layer_tree": layer_tree(bpy.context.view_layer.layer_collection),
    "film_transparent": bool(scene.render.film_transparent),
    "world_color": [float(value) for value in scene.world.color] if scene.world else None,
    "objects": {},
    "lights": {},
    "camera_projections": {},
}
for name in OBJECT_NAMES:
    obj = bpy.data.objects.get(name)
    if obj is None:
        continue
    payload["objects"][name] = {
        "hide_render": bool(obj.hide_render),
        "visible_get": bool(obj.visible_get(view_layer=bpy.context.view_layer)),
        "world_location": [float(value) for value in obj.matrix_world.translation],
        "world_bbox_min": [float(value) for value in np.min(np.asarray(world_bbox(obj)), axis=0)],
        "world_bbox_max": [float(value) for value in np.max(np.asarray(world_bbox(obj)), axis=0)],
    }
for obj in bpy.data.objects:
    if obj.type != "LIGHT":
        continue
    payload["lights"][obj.name] = {
        "type": obj.data.type,
        "energy": float(obj.data.energy),
        "hide_render": bool(obj.hide_render),
        "visible_get": bool(obj.visible_get(view_layer=bpy.context.view_layer)),
        "world_location": [float(value) for value in obj.matrix_world.translation],
        "world_direction_negative_z": [float(value) for value in -(obj.matrix_world.to_3x3().col[2])],
    }
for camera_name in CAMERA_NAMES:
    camera = bpy.data.objects.get(camera_name)
    if camera is None:
        continue
    payload["camera_projections"][camera_name] = {
        name: projection_record(scene, camera, bpy.data.objects[name])
        for name in OBJECT_NAMES
        if name in bpy.data.objects
    }
print("MGF_VISIBILITY_DIAGNOSTIC_BEGIN")
print(json.dumps(payload, indent=2))
print("MGF_VISIBILITY_DIAGNOSTIC_END")
