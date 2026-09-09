#!/usr/bin/env python3
"""Rename a Blender project and separate dataset identity from processing method."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--processing-method", default="Keypoint-MoSeq retargeting")
    parser.add_argument("--audit-path")
    args = parser.parse_args(args_after_dash())

    scene = bpy.context.scene
    old_token = "PyMouseStudio-KeypointMoseq"
    for key in ("MGF_mouse_segment", "MGF_mouse_rig"):
        value = scene.get(key)
        if isinstance(value, str):
            scene[key] = value.replace(old_token, args.dataset_name)
    scene["MGF_dataset_name"] = args.dataset_name
    scene["MGF_pose_processing_method"] = args.processing_method
    scene["MGF_project_name"] = args.output.stem
    if args.audit_path:
        scene["MGF_flexible_tail_audit"] = args.audit_path
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output))
    print(f"Saved renamed dataset project: {args.output}")


if __name__ == "__main__":
    main()

