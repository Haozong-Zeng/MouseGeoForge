#!/usr/bin/env python3
"""Dump scene units and world-space object extents for scale/alignment audits."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def world_bounds(obj: bpy.types.Object):
    if not obj.bound_box:
        return None
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    minimum = [min(point[index] for point in corners) for index in range(3)]
    maximum = [max(point[index] for point in corners) for index in range(3)]
    return {
        "min": minimum,
        "max": maximum,
        "center": [(lo + hi) * 0.5 for lo, hi in zip(minimum, maximum)],
        "size": [hi - lo for lo, hi in zip(minimum, maximum)],
    }


def object_dump(obj: bpy.types.Object):
    world_scale = obj.matrix_world.to_scale()
    return {
        "name": obj.name,
        "type": obj.type,
        "parent": obj.parent.name if obj.parent else None,
        "location": list(obj.location),
        "scale": list(obj.scale),
        "world_location": list(obj.matrix_world.translation),
        "world_scale": list(world_scale),
        "bounds": world_bounds(obj),
        "hide_viewport": obj.hide_viewport,
        "hide_render": obj.hide_render,
        "collections": [collection.name for collection in obj.users_collection],
        "materials": [slot.material.name if slot.material else None for slot in obj.material_slots],
        "action": obj.animation_data.action.name
        if obj.animation_data and obj.animation_data.action
        else None,
    }


def main() -> None:
    args = args_after_dash()
    if len(args) != 1:
        raise SystemExit("Expected output JSON path after --")
    scene = bpy.context.scene
    keywords = (
        "mouse",
        "cylinder",
        "glass",
        "floor",
        "ground",
        "plane",
        "armature",
        "deformation",
        "fur",
    )
    selected = []
    for obj in bpy.data.objects:
        haystack = " ".join(
            [
                obj.name,
                *(collection.name for collection in obj.users_collection),
                *(slot.material.name for slot in obj.material_slots if slot.material),
            ]
        ).lower()
        if any(keyword in haystack for keyword in keywords):
            selected.append(object_dump(obj))
    large_meshes = sorted(
        (obj for obj in bpy.data.objects if obj.type == "MESH" and world_bounds(obj)),
        key=lambda obj: max(world_bounds(obj)["size"]),
        reverse=True,
    )[:40]
    payload = {
        "blend": bpy.data.filepath,
        "unit_settings": {
            "system": scene.unit_settings.system,
            "scale_length": scene.unit_settings.scale_length,
            "length_unit": scene.unit_settings.length_unit,
        },
        "frame_range": [scene.frame_start, scene.frame_end],
        "current_frame": scene.frame_current,
        "active_camera": scene.camera.name if scene.camera else None,
        "objects_total": len(bpy.data.objects),
        "collections": [collection.name for collection in bpy.data.collections],
        "armatures": [object_dump(obj) for obj in bpy.data.objects if obj.type == "ARMATURE"],
        "root_empties": [
            object_dump(obj)
            for obj in bpy.data.objects
            if obj.type == "EMPTY" and obj.parent is None
        ],
        "cameras": [object_dump(obj) for obj in bpy.data.objects if obj.type == "CAMERA"],
        "selected_objects": selected,
        "largest_meshes": [object_dump(obj) for obj in large_meshes],
    }
    Path(args[0]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": args[0], "selected": len(selected)}, indent=2))


if __name__ == "__main__":
    main()
