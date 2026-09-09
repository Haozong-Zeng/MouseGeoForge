#!/usr/bin/env python3
"""Audit cameras and animated fur before the SceneOnly Cycles review render."""

from __future__ import annotations

import json

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


FRAME = 100
MOUSE = "MS_Mouse.001"


scene = bpy.context.scene
scene.frame_set(FRAME)
bpy.context.view_layer.update()
mouse = bpy.data.objects[MOUSE]
evaluated_mouse = mouse.evaluated_get(bpy.context.evaluated_depsgraph_get())
mouse_corners = [evaluated_mouse.matrix_world @ Vector(corner) for corner in evaluated_mouse.bound_box]
mouse_min = Vector(tuple(min(point[axis] for point in mouse_corners) for axis in range(3)))
mouse_max = Vector(tuple(max(point[axis] for point in mouse_corners) for axis in range(3)))
mouse_center = (mouse_min + mouse_max) * 0.5

cameras = []
for obj in sorted((obj for obj in bpy.data.objects if obj.type == "CAMERA"), key=lambda item: item.name):
    if not obj.name.startswith("Camera.0"):
        continue
    direction = (mouse_center - obj.matrix_world.translation).normalized()
    forward = -(obj.matrix_world.to_quaternion() @ Vector((0.0, 0.0, 1.0)))
    projected = [world_to_camera_view(scene, obj, corner) for corner in mouse_corners]
    projected_center = world_to_camera_view(scene, obj, mouse_center)
    cameras.append(
        {
            "name": obj.name,
            "location": list(obj.matrix_world.translation),
            "lens_mm": obj.data.lens,
            "sensor_width_mm": obj.data.sensor_width,
            "clip": [obj.data.clip_start, obj.data.clip_end],
            "mouse_forward_dot": float(forward.dot(direction)),
            "projected_mouse_center": list(projected_center),
            "projected_bbox_xy": [
                [min(point.x for point in projected), min(point.y for point in projected)],
                [max(point.x for point in projected), max(point.y for point in projected)],
            ],
            "hidden_render": obj.hide_render,
            "collections": [collection.name for collection in obj.users_collection],
        }
    )

fur = []
for name in (
    "BaseFur.001",
    "EarFur.001",
    "EyesFur.001",
    "LegFur.001",
    "Mouth.001",
    "TailFur.001",
    "Whisker.001",
    "MGF_AnimatedMouse_GuardShort",
    "MGF_AnimatedMouse_GuardLong",
):
    obj = bpy.data.objects.get(name)
    fur.append(
        {
            "name": name,
            "exists": obj is not None,
            "type": obj.type if obj else None,
            "hide_render": obj.hide_render if obj else None,
            "hide_viewport": obj.hide_viewport if obj else None,
            "materials": [material.name for material in obj.data.materials if material] if obj else [],
        }
    )

print(
    json.dumps(
        {
            "blend": bpy.data.filepath,
            "frame": FRAME,
            "mouse_center": list(mouse_center),
            "mouse_bounds": [list(mouse_min), list(mouse_max)],
            "active_camera": scene.camera.name if scene.camera else None,
            "render_engine": scene.render.engine,
            "resolution": [scene.render.resolution_x, scene.render.resolution_y],
            "cameras": cameras,
            "fur": fur,
        },
        indent=2,
    ),
    flush=True,
)
