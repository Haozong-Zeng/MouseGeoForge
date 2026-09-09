#!/usr/bin/env python3
"""Audit palm-facing semantics and evaluated finger lengths."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--side", choices=("L", "R"), default="R")
    return parser.parse_args(args_after_dash())


def bone_tail_world(armature, bone) -> Vector:
    return armature.matrix_world @ (
        bone.matrix @ Vector((0.0, bone.bone.length, 0.0))
    )


def main() -> None:
    args = parse_args()
    scene = bpy.context.scene
    armature = bpy.data.objects["DeformationSystem"]
    controller = bpy.data.objects[f"CTRL_Hand_{args.side}"]
    mouth = bpy.data.objects["DLC3D_mouth_tip_A"]
    world_left = Vector((-1.0, 0.0, 0.0))
    rows = []
    previous_quaternion = None

    for frame in range(args.start, args.end + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        quaternion = controller.matrix_world.to_quaternion().normalized()
        palm = controller.matrix_world.to_3x3().col[0].normalized()
        mouth_direction = (
            mouth.matrix_world.translation - controller.matrix_world.translation
        ).normalized()
        angular_step = (
            0.0
            if previous_quaternion is None
            else math.degrees(
                2.0
                * math.acos(
                    max(
                        -1.0,
                        min(1.0, abs(previous_quaternion.dot(quaternion))),
                    )
                )
            )
        )
        rows.append(
            {
                "frame": frame,
                "palm_dot_mouth": palm.dot(mouth_direction),
                "palm_dot_world_left": palm.dot(world_left),
                "angular_step_deg": angular_step,
            }
        )
        previous_quaternion = quaternion

    scene.frame_set((args.start + args.end) // 2)
    bpy.context.view_layer.update()
    lengths = {}
    for prefix in ("Index", "Middle", "Ring", "Pinky"):
        for segment in (1, 2, 3):
            name = f"{prefix}Finger{segment}_{args.side}"
            bone = armature.pose.bones[name]
            head = armature.matrix_world @ bone.matrix.translation
            lengths[name] = {
                "world_length": (bone_tail_world(armature, bone) - head).length,
                "pose_scale": [float(value) for value in bone.scale],
            }

    output = {
        "blend_file": bpy.data.filepath,
        "frame_range": [args.start, args.end],
        "side": args.side,
        "summary": {
            "min_palm_dot_mouth": min(
                rows, key=lambda row: row["palm_dot_mouth"]
            ),
            "min_palm_dot_world_left": min(
                rows, key=lambda row: row["palm_dot_world_left"]
            ),
            "max_angular_step_deg": max(
                rows, key=lambda row: row["angular_step_deg"]
            ),
            "nonpositive_mouth_frames": [
                row["frame"] for row in rows if row["palm_dot_mouth"] <= 0.0
            ],
            "nonpositive_left_frames": [
                row["frame"]
                for row in rows
                if row["palm_dot_world_left"] <= 0.0
            ],
        },
        "finger_lengths": lengths,
        "frames": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
