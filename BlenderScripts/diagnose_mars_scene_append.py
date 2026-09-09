#!/usr/bin/env python3
"""Check whether scene-level library append preserves MARS rig transforms."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy


def main() -> None:
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(args) != 2:
        raise SystemExit("Usage: blender -b base.blend -P script.py -- donor.blend output.json")
    donor, output = map(Path, args)
    before = set(bpy.data.scenes)
    with bpy.data.libraries.load(str(donor.resolve()), link=False) as (available, requested):
        if "Scene.002" not in available.scenes:
            raise KeyError("Scene.002")
        requested.scenes = ["Scene.002"]
    appended = next(scene for scene in bpy.data.scenes if scene not in before)
    rows = []
    for obj in appended.objects:
        if obj.name.startswith(("Group.", "MS_Mouse", "SPT_cage")):
            rows.append(
                {
                    "name": obj.name,
                    "type": obj.type,
                    "parent": obj.parent.name if obj.parent else None,
                    "location": list(obj.location),
                    "scale": list(obj.scale),
                    "dimensions": list(obj.dimensions),
                    "matrix_translation": list(obj.matrix_world.translation),
                }
            )
    output.write_text(
        json.dumps(
            {
                "scene": appended.name,
                "object_count": len(appended.objects),
                "objects": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
