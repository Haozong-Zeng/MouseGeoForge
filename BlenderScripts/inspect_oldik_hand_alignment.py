"""Inspect hand bones, proxy targets, and constraints in a migrated old-IK file."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy


BONES = [
    "Scapula_L",
    "Wrist_L",
    "Fingers1_L",
    "IndexFinger1_L",
    "IndexFinger2_L",
    "IndexFinger3_L",
    "MiddleFinger1_L",
    "MiddleFinger2_L",
    "MiddleFinger3_L",
    "RingFinger1_L",
    "RingFinger2_L",
    "RingFinger3_L",
    "PinkyFinger1_L",
    "PinkyFinger2_L",
    "PinkyFinger3_L",
    "Scapula_R",
    "Wrist_R",
    "Fingers1_R",
    "IndexFinger1_R",
    "IndexFinger2_R",
    "IndexFinger3_R",
    "MiddleFinger1_R",
    "MiddleFinger2_R",
    "MiddleFinger3_R",
    "RingFinger1_R",
    "RingFinger2_R",
    "RingFinger3_R",
    "PinkyFinger1_R",
    "PinkyFinger2_R",
    "PinkyFinger3_R",
]

PROXIES = [
    "MGF_HandSphere_L_palm",
    "MGF_HandSphere_L_index_tip",
    "MGF_HandSphere_L_middle_tip",
    "MGF_HandSphere_L_ring_tip",
    "MGF_HandSphere_L_pinky_tip",
    "MGF_HandSphere_R_palm",
    "MGF_HandSphere_R_index_tip",
    "MGF_HandSphere_R_middle_tip",
    "MGF_HandSphere_R_ring_tip",
    "MGF_HandSphere_R_pinky_tip",
]


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--armature", default="DeformationSystem")
    parser.add_argument("--action", default="MIG_BrowmFur_FGcamB_v44_smooth60")
    parser.add_argument("--frame", type=int, action="append", default=[])
    return parser.parse_args(args_after_dash())


def rounded_vec(vec) -> list[float]:
    return [round(float(vec[i]), 5) for i in range(3)]


def rotation_degrees(pb) -> list[float]:
    if pb.rotation_mode == "QUATERNION":
        euler = pb.rotation_quaternion.to_euler()
    elif pb.rotation_mode == "AXIS_ANGLE":
        euler = pb.rotation_axis_angle.to_quaternion().to_euler()
    else:
        euler = pb.rotation_euler
    return [round(math.degrees(float(v)), 3) for v in euler]


def constraint_row(constraint) -> dict:
    row = {
        "name": constraint.name,
        "type": constraint.type,
        "mute": bool(getattr(constraint, "mute", False)),
        "influence": round(float(getattr(constraint, "influence", 0.0)), 5),
        "target": getattr(getattr(constraint, "target", None), "name", None),
        "subtarget": getattr(constraint, "subtarget", ""),
    }
    for attr in ("owner_space", "target_space", "min_x", "max_x", "min_y", "max_y", "min_z", "max_z"):
        if hasattr(constraint, attr):
            value = getattr(constraint, attr)
            if attr.startswith(("min_", "max_")):
                value = round(math.degrees(float(value)), 3)
            row[attr] = value
    return row


def bone_row(armature, bone_name: str) -> dict:
    pb = armature.pose.bones.get(bone_name)
    if pb is None:
        return {"missing": True}
    head = armature.matrix_world @ pb.head
    tail = armature.matrix_world @ pb.tail
    z_axis_tip = armature.matrix_world @ (pb.head + pb.matrix.to_3x3().col[2].normalized() * pb.length)
    return {
        "missing": False,
        "parent": pb.parent.name if pb.parent else None,
        "rotation_mode": pb.rotation_mode,
        "rotation_euler_deg": rotation_degrees(pb),
        "scale": [round(float(v), 5) for v in pb.scale],
        "head_world": rounded_vec(head),
        "tail_world": rounded_vec(tail),
        "local_z_axis_tip_world": rounded_vec(z_axis_tip),
        "constraints": [constraint_row(c) for c in pb.constraints],
    }


def object_row(name: str) -> dict:
    obj = bpy.data.objects.get(name)
    if obj is None:
        return {"missing": True}
    return {
        "missing": False,
        "location_world": rounded_vec(obj.matrix_world.translation),
        "hide_render": bool(obj.hide_render),
        "hide_viewport": bool(obj.hide_viewport),
    }


def main() -> None:
    args = parse_args()
    bpy.ops.wm.open_mainfile(filepath=str(args.input.resolve()))
    armature = bpy.data.objects.get(args.armature)
    if armature is None:
        raise RuntimeError(f"Armature {args.armature!r} not found.")
    action = bpy.data.actions.get(args.action)
    if action is not None:
        armature.animation_data_create()
        armature.animation_data.action = action

    frames = args.frame or [int(bpy.context.scene.frame_current)]
    report = {
        "input": str(args.input.resolve()),
        "armature": armature.name,
        "action": action.name if action else None,
        "constraint_targets": {},
        "candidate_control_objects": [],
        "frames": {},
    }
    for pb in armature.pose.bones:
        for constraint in pb.constraints:
            target = getattr(constraint, "target", None)
            if target is None:
                continue
            target_row = report["constraint_targets"].setdefault(target.name, [])
            target_row.append(
                {
                    "bone": pb.name,
                    "constraint": constraint.name,
                    "type": constraint.type,
                    "mute": bool(getattr(constraint, "mute", False)),
                    "influence": round(float(getattr(constraint, "influence", 0.0)), 5),
                    "subtarget": getattr(constraint, "subtarget", ""),
                }
            )
    for obj in bpy.data.objects:
        lower = obj.name.lower()
        if any(token in lower for token in ("wrist", "finger", "hand", "ik", "target", "ctrl", "control", "sphere")):
            report["candidate_control_objects"].append(
                {
                    "name": obj.name,
                    "type": obj.type,
                    "location_world": rounded_vec(obj.matrix_world.translation),
                    "hide_viewport": bool(obj.hide_viewport),
                    "hide_render": bool(obj.hide_render),
                }
            )
    for frame in frames:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        report["frames"][str(frame)] = {
            "bones": {name: bone_row(armature, name) for name in BONES},
            "proxies": {name: object_row(name) for name in PROXIES},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
