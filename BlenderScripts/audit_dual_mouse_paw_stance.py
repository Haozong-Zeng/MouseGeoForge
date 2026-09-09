#!/usr/bin/env python3
"""Audit dual-mouse paw support and left/right stance symmetry in Blender."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import bpy


ARMATURES = {
    "Black": "LP_Black_Armature",
    "Brown": "LP_Brown_Armature",
}
PAW_BONES = {
    "left_forepaw": "Fingers1_L",
    "right_forepaw": "Fingers1_R",
    "left_hindpaw": "Toes1_L",
    "right_hindpaw": "Toes1_R",
}
PAIR_BONES = ("Hip", "Knee", "Ankle", "Wrist", "Fingers1", "Toes1")


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(args_after_dash())


def world_head(armature: bpy.types.Object, bone_name: str):
    return armature.matrix_world @ armature.pose.bones[bone_name].head


def digit_extremity_z(armature: bpy.types.Object, token: str, side: str) -> list[float]:
    values = []
    if token == "Finger":
        pattern = re.compile(rf"^(Fingers1|(?:Index|Middle|Ring|Pinky)Finger[123])_{side}$")
    else:
        pattern = re.compile(rf"^(Toes1|(?:Index|Middle|Ring|Pinky)Toe[123])_{side}$")
    for bone in armature.pose.bones:
        if pattern.fullmatch(bone.name):
            values.append(float((armature.matrix_world @ bone.head).z))
            values.append(float((armature.matrix_world @ bone.tail).z))
    if not values:
        raise RuntimeError(f"No canonical {token} chain for side {side} on {armature.name}")
    return values


def support_state(minimum_height: float) -> str:
    if minimum_height < -2.0:
        return "penetrating"
    if minimum_height <= 3.0:
        return "ground_contact"
    return "raised"


def extrema(rows: list[dict], path: tuple[str, ...]) -> tuple[float, float]:
    values = []
    for row in rows:
        value = row
        for key in path:
            value = value[key]
        values.append(float(value))
    return min(values), max(values)


def main() -> None:
    args = parse_args()
    scene = bpy.context.scene
    # Hair evaluation dominates frame changes and is irrelevant to skeletal
    # paw geometry. This is process-local; the audit does not save the blend.
    for obj in bpy.data.objects:
        if obj.type == "CURVES":
            obj.hide_viewport = True
    floor_z = float(scene.get("MGF_ground_contact_floor_z_mm", -237.19223022460938))
    frames = list(range(scene.frame_start, scene.frame_end + 1))
    report = {
        "schema": "mousegeoforge.dual_mouse_paw_stance_audit.v1",
        "blend": bpy.data.filepath,
        "frame_range": [scene.frame_start, scene.frame_end],
        "floor_z_mm": floor_z,
        "paw_semantics": {
            "forepaw_center": "Fingers1_L/R bone head",
            "hindpaw_center": "Toes1_L/R bone head",
            "extremity": "minimum world z over all matching finger/toe bone heads and tails",
        },
        "instances": {},
    }

    for label, armature_name in ARMATURES.items():
        armature = bpy.data.objects[armature_name]
        rows = []
        for frame in frames:
            scene.frame_set(frame)
            bpy.context.view_layer.update()
            paws = {}
            for paw_name, bone_name in PAW_BONES.items():
                point = world_head(armature, bone_name)
                token = "Finger" if "forepaw" in paw_name else "Toe"
                side = "L" if paw_name.startswith("left") else "R"
                extremities = digit_extremity_z(armature, token, side)
                paws[paw_name] = {
                    "world_xyz_mm": [float(value) for value in point],
                    "center_height_above_floor_mm": float(point.z - floor_z),
                    "minimum_extremity_height_above_floor_mm": min(extremities) - floor_z,
                }
            paired = {}
            for stem in PAIR_BONES:
                left = world_head(armature, f"{stem}_L")
                right = world_head(armature, f"{stem}_R")
                xy_separation = math.hypot(left.x - right.x, left.y - right.y)
                dz = float(left.z - right.z)
                paired[stem] = {
                    "left_minus_right_z_mm": dz,
                    "xy_separation_mm": xy_separation,
                    "roll_proxy_degrees": math.degrees(math.atan2(dz, max(xy_separation, 1.0e-9))),
                }
            rows.append({"frame": frame, "paws": paws, "paired_joints": paired})

        paw_summary = {}
        for paw_name in PAW_BONES:
            center_min, center_max = extrema(rows, ("paws", paw_name, "center_height_above_floor_mm"))
            tip_min, tip_max = extrema(
                rows, ("paws", paw_name, "minimum_extremity_height_above_floor_mm")
            )
            paw_summary[paw_name] = {
                "center_height_range_mm": [center_min, center_max],
                "minimum_extremity_height_range_mm": [tip_min, tip_max],
                "support_state": support_state(tip_min),
            }
        pair_summary = {}
        for stem in PAIR_BONES:
            dz_values = [row["paired_joints"][stem]["left_minus_right_z_mm"] for row in rows]
            roll_values = [row["paired_joints"][stem]["roll_proxy_degrees"] for row in rows]
            index = max(range(len(frames)), key=lambda item: abs(dz_values[item]))
            pair_summary[stem] = {
                "maximum_absolute_z_difference_mm": abs(dz_values[index]),
                "signed_z_difference_at_max_mm": dz_values[index],
                "frame_at_max": frames[index],
                "maximum_absolute_roll_proxy_degrees": max(abs(value) for value in roll_values),
            }
        hip = pair_summary["Hip"]
        report["instances"][label] = {
            "armature": armature_name,
            "paws": paw_summary,
            "paired_joint_symmetry": pair_summary,
            "systematic_body_side_lean": (
                hip["maximum_absolute_z_difference_mm"] > 2.0
                or hip["maximum_absolute_roll_proxy_degrees"] > 5.0
            ),
            "rows": rows,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "instances": {
            label: {
                "paws": data["paws"],
                "paired_joint_symmetry": data["paired_joint_symmetry"],
                "systematic_body_side_lean": data["systematic_body_side_lean"],
            }
            for label, data in report["instances"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
