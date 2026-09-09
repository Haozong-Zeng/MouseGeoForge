#!/usr/bin/env python3
"""Inspect bilateral hand controls, armatures, constraints, and cameras in Blender."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy


KEYWORDS = (
    "arm",
    "finger",
    "fingers",
    "hand",
    "ik",
    "palm",
    "scapula",
    "wrist",
)

APPARATUS_KEYWORDS = (
    "bottle",
    "cup",
    "point",
    "port",
    "spout",
    "tube",
    "water",
)


def blender_args() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def vector(value) -> list[float]:
    return [float(component) for component in value]


def matrix(value) -> list[list[float]]:
    return [[float(component) for component in row] for row in value]


def matches(name: str) -> bool:
    lowered = name.lower()
    return any(keyword in lowered for keyword in KEYWORDS)


def apparatus_matches(name: str) -> bool:
    lowered = name.lower()
    return any(keyword in lowered for keyword in APPARATUS_KEYWORDS)


def serialize_constraint(constraint) -> dict:
    data = {
        "name": constraint.name,
        "type": constraint.type,
        "mute": bool(constraint.mute),
        "influence": float(constraint.influence),
    }
    for attr in (
        "chain_count",
        "head_tail",
        "ik_type",
        "max_x",
        "max_y",
        "max_z",
        "min_x",
        "min_y",
        "min_z",
        "mix_mode",
        "owner_space",
        "pole_angle",
        "subtarget",
        "target_space",
        "track_axis",
        "up_axis",
        "use_location",
        "use_limit_x",
        "use_limit_y",
        "use_limit_z",
        "use_rotation",
        "use_stretch",
    ):
        if hasattr(constraint, attr):
            value = getattr(constraint, attr)
            if isinstance(value, (str, bool, int, float)):
                data[attr] = value
    target = getattr(constraint, "target", None)
    pole_target = getattr(constraint, "pole_target", None)
    data["target"] = target.name if target is not None else None
    data["pole_target"] = pole_target.name if pole_target is not None else None
    return data


def inspect_object(obj) -> dict:
    item = {
        "name": obj.name,
        "type": obj.type,
        "parent": obj.parent.name if obj.parent else None,
        "location": vector(obj.location),
        "rotation_mode": obj.rotation_mode,
        "rotation_euler": vector(obj.rotation_euler),
        "scale": vector(obj.scale),
        "dimensions": vector(obj.dimensions),
        "matrix_world": matrix(obj.matrix_world),
        "world_location": vector(obj.matrix_world.translation),
        "collections": [collection.name for collection in obj.users_collection],
        "constraints": [serialize_constraint(constraint) for constraint in obj.constraints],
    }
    if obj.type == "ARMATURE":
        item["pose_bones"] = []
        for pose_bone in obj.pose.bones:
            if not matches(pose_bone.name):
                continue
            bone = pose_bone.bone
            item["pose_bones"].append(
                {
                    "name": pose_bone.name,
                    "parent": pose_bone.parent.name if pose_bone.parent else None,
                    "head_local": vector(bone.head_local),
                    "tail_local": vector(bone.tail_local),
                    "head_world": vector(obj.matrix_world @ pose_bone.head),
                    "tail_world": vector(obj.matrix_world @ pose_bone.tail),
                    "length": float(bone.length),
                    "matrix_local": matrix(bone.matrix_local),
                    "rotation_mode": pose_bone.rotation_mode,
                    "rotation_euler": vector(pose_bone.rotation_euler),
                    "scale": vector(pose_bone.scale),
                    "lock_location": list(pose_bone.lock_location),
                    "lock_rotation": list(pose_bone.lock_rotation),
                    "lock_scale": list(pose_bone.lock_scale),
                    "use_ik_limit_x": bool(pose_bone.use_ik_limit_x),
                    "use_ik_limit_y": bool(pose_bone.use_ik_limit_y),
                    "use_ik_limit_z": bool(pose_bone.use_ik_limit_z),
                    "ik_min": [
                        float(pose_bone.ik_min_x),
                        float(pose_bone.ik_min_y),
                        float(pose_bone.ik_min_z),
                    ],
                    "ik_max": [
                        float(pose_bone.ik_max_x),
                        float(pose_bone.ik_max_y),
                        float(pose_bone.ik_max_z),
                    ],
                    "constraints": [
                        serialize_constraint(constraint)
                        for constraint in pose_bone.constraints
                    ],
                }
            )
    return item


def main() -> None:
    args = blender_args()
    output = Path(args[0]) if args else Path(bpy.path.abspath("//bilateral_hand_rig_inspection.json"))
    objects = [
        inspect_object(obj)
        for obj in bpy.data.objects
        if matches(obj.name) or apparatus_matches(obj.name) or obj.type == "ARMATURE"
    ]
    cameras = []
    for obj in bpy.data.objects:
        if obj.type != "CAMERA":
            continue
        cameras.append(
            {
                "name": obj.name,
                "matrix_world": matrix(obj.matrix_world),
                "location": vector(obj.matrix_world.translation),
                "lens": float(obj.data.lens),
                "sensor_width": float(obj.data.sensor_width),
                "sensor_height": float(obj.data.sensor_height),
            }
        )
    payload = {
        "blend_file": bpy.data.filepath,
        "scene": bpy.context.scene.name,
        "frame_start": int(bpy.context.scene.frame_start),
        "frame_end": int(bpy.context.scene.frame_end),
        "object_count": len(bpy.data.objects),
        "objects": objects,
        "cameras": cameras,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"inspection_written={output}")


if __name__ == "__main__":
    main()
