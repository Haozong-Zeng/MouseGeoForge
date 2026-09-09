#!/usr/bin/env python3
"""List object actions and frame ranges in Blender 5 layered-action projects."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def curve_count(action) -> int:
    if hasattr(action, "fcurves"):
        return len(action.fcurves)
    return sum(
        len(channelbag.fcurves)
        for layer in action.layers
        for strip in layer.strips
        for channelbag in getattr(strip, "channelbags", [])
    )


def main() -> None:
    args = args_after_dash()
    if len(args) != 1:
        raise SystemExit("Expected output JSON path")
    rows = []
    for obj in bpy.data.objects:
        action = obj.animation_data.action if obj.animation_data else None
        if not action:
            continue
        rows.append(
            {
                "object": obj.name,
                "type": obj.type,
                "parent": obj.parent.name if obj.parent else None,
                "collections": [collection.name for collection in obj.users_collection],
                "action": action.name,
                "frame_range": list(action.frame_range),
                "curve_count": curve_count(action),
            }
        )
    rows.sort(key=lambda row: (row["frame_range"][1] - row["frame_range"][0], row["curve_count"]), reverse=True)
    Path(args[0]).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"output": args[0], "animated_objects": len(rows), "top": rows[:10]}, indent=2))


if __name__ == "__main__":
    main()
