#!/usr/bin/env python3
"""Inspect evaluated bilateral controller and finger poses at selected frames."""

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
    parser.add_argument("--frame", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bind-frame", type=int, default=228)
    return parser.parse_args(args_after_dash())


def vector(value) -> list[float]:
    return [float(component) for component in value]


def bone_tail_world(armature, bone) -> Vector:
    return armature.matrix_world @ (
        bone.matrix @ Vector((0.0, bone.bone.length, 0.0))
    )


def main() -> None:
    args = parse_args()
    scene = bpy.context.scene
    armature = bpy.data.objects["DeformationSystem"]
    scene.frame_set(args.bind_frame)
    bpy.context.view_layer.update()
    bind_poses = {}
    for side in ("L", "R"):
        controller = bpy.data.objects[f"CTRL_Hand_{side}"]
        wrist = armature.pose.bones[f"Wrist_{side}"]
        wrist_head = armature.matrix_world @ wrist.matrix.translation
        wrist_tail = bone_tail_world(armature, wrist)
        bind_poses[side] = {
            "controller_quaternion": controller.matrix_world.to_quaternion().copy(),
            "wrist_axis": (wrist_tail - wrist_head).normalized(),
        }
    output = {"blend_file": bpy.data.filepath, "frames": []}

    for frame in args.frame:
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        frame_row = {"frame": frame, "hands": {}}
        for side in ("L", "R"):
            controller = bpy.data.objects[f"CTRL_Hand_{side}"]
            wrist = armature.pose.bones[f"Wrist_{side}"]
            fingers = armature.pose.bones[f"Fingers1_{side}"]
            wrist_head = armature.matrix_world @ wrist.matrix.translation
            wrist_tail = bone_tail_world(armature, wrist)
            wrist_axis = (wrist_tail - wrist_head).normalized()
            bind_pose = bind_poses[side]
            swing = bind_pose["wrist_axis"].rotation_difference(wrist_axis)
            reference = swing @ bind_pose["controller_quaternion"]
            delta = controller.matrix_world.to_quaternion() @ reference.inverted()
            roll_sign = 1.0 if delta.axis.dot(wrist_axis) >= 0.0 else -1.0
            forearm_roll = math.degrees(delta.angle) * roll_sign
            while forearm_roll > 180.0:
                forearm_roll -= 360.0
            while forearm_roll < -180.0:
                forearm_roll += 360.0
            frame_row["hands"][side] = {
                "controller_location": vector(controller.matrix_world.translation),
                "controller_quaternion_wxyz": vector(
                    controller.matrix_world.to_quaternion()
                ),
                "controller_curl_deg": {
                    finger: float(controller[f"{side}_{finger}CurlDeg"])
                    for finger in ("Index", "Middle", "Ring", "Pinky")
                    if f"{side}_{finger}CurlDeg" in controller
                },
                "wrist_axis": vector(wrist_axis),
                "forearm_roll_from_bind_deg": forearm_roll,
                "palm_head": vector(armature.matrix_world @ fingers.matrix.translation),
                "palm_tail": vector(bone_tail_world(armature, fingers)),
                "finger_bones": {},
            }
            for prefix in ("Index", "Middle", "Ring", "Pinky"):
                for segment in (1, 2, 3):
                    name = f"{prefix}Finger{segment}_{side}"
                    bone = armature.pose.bones[name]
                    frame_row["hands"][side]["finger_bones"][name] = {
                        "rotation_euler_deg": [
                            math.degrees(float(value))
                            for value in bone.rotation_euler
                        ],
                        "scale": vector(bone.scale),
                        "head": vector(armature.matrix_world @ bone.matrix.translation),
                        "tail": vector(bone_tail_world(armature, bone)),
                    }
        output["frames"].append(frame_row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
