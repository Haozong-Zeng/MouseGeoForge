"""Sample hand controllers, wrists, and solved finger angles for comparison."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


FINGERS = ("Index", "Middle", "Ring", "Pinky")


def blender_args() -> list[str]:
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(blender_args())


def world_point(armature: bpy.types.Object, point: Vector) -> Vector:
    return armature.matrix_world @ point


def bend_degrees(first: Vector, second: Vector) -> float:
    if first.length < 1e-8 or second.length < 1e-8:
        return 0.0
    return math.degrees(first.angle(second))


def main() -> None:
    args = parse_args()
    scene = bpy.context.scene
    armature = bpy.data.objects["DeformationSystem"]
    curves = [obj for obj in bpy.data.objects if obj.type == "CURVES"]
    curve_state = [(obj, obj.hide_viewport) for obj in curves]
    for obj in curves:
        obj.hide_viewport = True

    frames = []
    try:
        for frame in range(scene.frame_start, scene.frame_end + 1):
            scene.frame_set(frame)
            bpy.context.view_layer.update()
            row: dict[str, object] = {"frame": frame, "sides": {}}
            for side in ("L", "R"):
                controller = bpy.data.objects[f"CTRL_Hand_{side}"]
                quaternion = controller.matrix_world.to_quaternion()
                wrist = armature.pose.bones[f"Wrist_{side}"]
                side_row = {
                    "controller_location": list(
                        controller.matrix_world.translation
                    ),
                    "controller_quaternion": list(quaternion),
                    "wrist_tail": list(world_point(armature, wrist.tail)),
                    "curl_properties": {
                        finger: float(
                            controller[f"{side}_{finger}CurlDeg"]
                        )
                        for finger in FINGERS
                    },
                    "fingers": {},
                }
                for finger in FINGERS:
                    bones = [
                        armature.pose.bones[f"{finger}Finger{index}_{side}"]
                        for index in (1, 2, 3)
                    ]
                    vectors = [
                        world_point(armature, bone.tail)
                        - world_point(armature, bone.head)
                        for bone in bones
                    ]
                    side_row["fingers"][finger] = {
                        "rotation_x_deg": [
                            math.degrees(float(bone.rotation_euler.x))
                            for bone in bones
                        ],
                        "bend_1_2_deg": bend_degrees(vectors[0], vectors[1]),
                        "bend_2_3_deg": bend_degrees(vectors[1], vectors[2]),
                        "tip": list(world_point(armature, bones[-1].tail)),
                    }
                row["sides"][side] = side_row
            frames.append(row)
    finally:
        for obj, hidden in curve_state:
            obj.hide_viewport = hidden

    report = {
        "blend": bpy.data.filepath,
        "frame_range": [scene.frame_start, scene.frame_end],
        "frames": frames,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=True),
        encoding="utf-8",
    )
    print(f"MGF_FINGER_AUDIT={args.output}")


if __name__ == "__main__":
    main()
