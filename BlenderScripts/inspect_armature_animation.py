#!/usr/bin/env python3
"""Export compact armature rest-pose and action metadata."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def main() -> None:
    args = args_after_dash()
    if len(args) != 2:
        raise SystemExit("Expected armature name and output JSON path")
    armature = bpy.data.objects[args[0]]
    action = armature.animation_data.action if armature.animation_data else None
    paths = []
    if action:
        curves = []
        if hasattr(action, "fcurves"):
            curves.extend(action.fcurves)
        else:
            for layer in action.layers:
                for strip in layer.strips:
                    for channelbag in getattr(strip, "channelbags", []):
                        curves.extend(channelbag.fcurves)
        for curve in curves:
            paths.append(
                {
                    "data_path": curve.data_path,
                    "array_index": curve.array_index,
                    "key_count": len(curve.keyframe_points),
                    "frame_range": list(curve.range()),
                }
            )
    payload = {
        "blend": bpy.data.filepath,
        "armature": armature.name,
        "bones": {
            bone.name: {
                "parent": bone.parent.name if bone.parent else None,
                "head": list(bone.head_local),
                "tail": list(bone.tail_local),
                "length": bone.length,
                "matrix_local": [list(row) for row in bone.matrix_local],
            }
            for bone in armature.data.bones
        },
        "pose": {
            bone.name: {
                "rotation_mode": bone.rotation_mode,
                "constraints": [constraint.type for constraint in bone.constraints],
            }
            for bone in armature.pose.bones
        },
        "action": {
            "name": action.name,
            "frame_range": list(action.frame_range),
            "curve_count": len(paths),
            "paths": paths,
        }
        if action
        else None,
    }
    Path(args[1]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": args[1],
                "bones": len(payload["bones"]),
                "action": payload["action"]["name"] if payload["action"] else None,
                "curves": payload["action"]["curve_count"] if payload["action"] else 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
