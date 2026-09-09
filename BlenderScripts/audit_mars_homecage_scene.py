#!/usr/bin/env python3
"""Export hierarchy, bounds, collections, and materials from a Blender scene."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def world_bounds(obj: bpy.types.Object) -> dict[str, list[float]] | None:
    if obj.type not in {"MESH", "CURVE", "CURVES", "FONT", "SURFACE", "META"}:
        return None
    points = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    return {
        "min": [min(point[axis] for point in points) for axis in range(3)],
        "max": [max(point[axis] for point in points) for axis in range(3)],
    }


def object_record(obj: bpy.types.Object) -> dict[str, object]:
    bounds = world_bounds(obj)
    dimensions = None
    center = None
    if bounds is not None:
        dimensions = [bounds["max"][i] - bounds["min"][i] for i in range(3)]
        center = [(bounds["max"][i] + bounds["min"][i]) * 0.5 for i in range(3)]
    return {
        "name": obj.name,
        "type": obj.type,
        "parent": obj.parent.name if obj.parent else None,
        "collections": [collection.name for collection in obj.users_collection],
        "scenes": [scene.name for scene in bpy.data.scenes if obj.name in scene.objects],
        "location": list(obj.location),
        "scale": list(obj.scale),
        "bounds": bounds,
        "dimensions": dimensions,
        "center": center,
        "materials": [slot.material.name if slot.material else None for slot in obj.material_slots],
        "hide_viewport": obj.hide_viewport,
        "hide_render": obj.hide_render,
    }


def main() -> None:
    args = args_after_dash()
    if len(args) != 1:
        raise SystemExit("Usage: blender -b scene.blend -P audit_mars_homecage_scene.py -- output.json")

    payload = {
        "blend": bpy.data.filepath,
        "scene": bpy.context.scene.name,
        "scenes": [
            {
                "name": scene.name,
                "root_children": [collection.name for collection in scene.collection.children],
                "object_count": len(scene.objects),
                "camera": scene.camera.name if scene.camera else None,
            }
            for scene in bpy.data.scenes
        ],
        "unit_system": bpy.context.scene.unit_settings.system,
        "unit_scale": bpy.context.scene.unit_settings.scale_length,
        "collections": [],
        "objects": [object_record(obj) for obj in bpy.data.objects],
    }
    for collection in bpy.data.collections:
        payload["collections"].append(
            {
                "name": collection.name,
                "parents": [
                    candidate.name
                    for candidate in bpy.data.collections
                    if collection.name in candidate.children
                ],
                "objects": [obj.name for obj in collection.objects],
                "all_object_count": len(collection.all_objects),
                "children": [child.name for child in collection.children],
                "hide_viewport": collection.hide_viewport,
                "hide_render": collection.hide_render,
            }
        )
    Path(args[0]).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
