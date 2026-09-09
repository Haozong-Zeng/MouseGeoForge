#!/usr/bin/env python3
"""Hide dual-mouse hair in the viewport while keeping final rendering enabled."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(args_after_dash())

    prefixes = ("LP_Black_", "LP_Brown_")
    hair_objects = []
    for obj in bpy.data.objects:
        if obj is None or obj.type != "CURVES" or not obj.name.startswith(prefixes):
            continue
        obj.hide_viewport = True
        obj.hide_render = False
        hair_objects.append(obj.name)

    if not hair_objects:
        raise RuntimeError("No dual-mouse CURVES hair objects were found")
    scene = bpy.context.scene
    scene["MGF_fur_viewport_default"] = "hidden"
    scene["MGF_fur_render_enabled"] = True
    scene["MGF_fur_viewport_objects"] = json.dumps(hair_objects)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output), check_existing=False)
    print(json.dumps({
        "output": str(args.output),
        "viewport_hidden": True,
        "render_enabled": True,
        "hair_object_count": len(hair_objects),
        "hair_objects": hair_objects,
    }, indent=2))


if __name__ == "__main__":
    main()
