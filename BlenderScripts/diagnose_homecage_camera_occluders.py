#!/usr/bin/env python3
"""Ray-cast homecage camera sightlines to identify foreground occluders."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def main() -> None:
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(args) != 1:
        raise SystemExit("Usage: blender -b scene.blend -P script.py -- output.json")
    output = Path(args[0])
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    rows = []
    for camera in bpy.data.objects:
        if camera.type != "CAMERA" or not camera.name.startswith("HomecageCam_"):
            continue
        target = Vector(camera.get("MGF_target_mm", (0.0, 0.0, 322.5)))
        for z_offset in (-70.0, -35.0, 0.0, 35.0, 70.0):
            sample = target + Vector((0.0, 0.0, z_offset))
            direction = sample - camera.location
            max_distance = direction.length
            direction.normalize()
            hit, location, _normal, _index, obj, _matrix = scene.ray_cast(
                depsgraph,
                camera.location,
                direction,
                distance=max_distance,
            )
            rows.append(
                {
                    "camera": camera.name,
                    "z_offset": z_offset,
                    "hit": hit,
                    "object": obj.name if obj else None,
                    "collections": [collection.name for collection in obj.users_collection]
                    if obj
                    else [],
                    "distance": (location - camera.location).length if hit else None,
                }
            )
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
