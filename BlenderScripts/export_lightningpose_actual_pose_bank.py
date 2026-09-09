#!/usr/bin/env python3
"""Export the currently evaluated Blender seven-point pose bank."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np


BONES = {
    "nose": "HeadEnd_M",
    "left_ear": "Ear01_L",
    "right_ear": "Ear01_R",
    "neck": "Neck3_M",
    "left_knee": "Knee_L",
    "right_knee": "Knee_R",
    "tail_base": "Root_M",
}


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--armature", default="DeformationSystem.001")
    parser.add_argument("--minimum-body-length", type=float, default=50.0)
    parser.add_argument("--maximum-body-length", type=float, default=100.0)
    args = parser.parse_args(args_after_dash())
    scene = bpy.context.scene
    armature = bpy.data.objects[args.armature]
    for obj in bpy.data.objects:
        if obj.type == "CURVES":
            obj.hide_viewport = True
    poses = []
    rejected = []
    for frame in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        points = {
            name: np.asarray(
                armature.matrix_world @ armature.pose.bones[bone].head,
                dtype=np.float64,
            )
            for name, bone in BONES.items()
        }
        body_length = float(np.linalg.norm(points["nose"] - points["tail_base"]))
        z_span = float(max(point[2] for point in points.values()) - min(point[2] for point in points.values()))
        record = {
            "frame": frame,
            "body_length_mm": body_length,
            "z_span_mm": z_span,
            "keypoints_world_xyz": {name: value.tolist() for name, value in points.items()},
        }
        if args.minimum_body_length <= body_length <= args.maximum_body_length:
            poses.append(record)
        else:
            rejected.append({"frame": frame, "body_length_mm": body_length})
    payload = {
        "schema": "mousegeoforge.lightningpose_actual_pose_bank.v1",
        "source_blend": bpy.data.filepath,
        "armature": armature.name,
        "keypoint_names": list(BONES),
        "accepted_count": len(poses),
        "rejected_count": len(rejected),
        "body_length_range_mm": [args.minimum_body_length, args.maximum_body_length],
        "poses": poses,
        "rejected": rejected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "accepted": len(poses),
        "rejected": len(rejected),
    }, indent=2))


if __name__ == "__main__":
    main()
