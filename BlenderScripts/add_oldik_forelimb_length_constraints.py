"""Add fixed local-Y length constraints to the old IK forelimb arm bones.

This keeps the rest skeleton unchanged.  The fixed reach is expressed as pose
scale on Scapula/Wrist and enforced with LIMIT_SCALE constraints.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector


FORELIMB_ARM_BASES = ("Scapula", "Wrist")
DEFAULT_CONSTRAINT_NAME = "MGF_ForelimbLengthLock"


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blend", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--armature", default="DeformationSystem")
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Default fixed local-Y pose scale for Scapula and Wrist.",
    )
    parser.add_argument("--scapula-scale", type=float, default=None)
    parser.add_argument("--wrist-scale", type=float, default=None)
    parser.add_argument("--constraint-name", default=DEFAULT_CONSTRAINT_NAME)
    return parser.parse_args(args_after_dash())


def pose_bone_tail_world(armature, pose_bone):
    bone = armature.data.bones[pose_bone.name]
    return armature.matrix_world @ (pose_bone.matrix @ Vector((0.0, bone.length, 0.0)))


def pose_bone_head_world(armature, pose_bone):
    return armature.matrix_world @ pose_bone.matrix.translation


def constraint_payload(constraint) -> dict:
    row = {
        "name": constraint.name,
        "type": constraint.type,
        "mute": bool(getattr(constraint, "mute", False)),
        "influence": float(getattr(constraint, "influence", 0.0)),
    }
    for attr in (
        "use_min_x",
        "use_min_y",
        "use_min_z",
        "use_max_x",
        "use_max_y",
        "use_max_z",
        "min_x",
        "min_y",
        "min_z",
        "max_x",
        "max_y",
        "max_z",
        "owner_space",
    ):
        if hasattr(constraint, attr):
            value = getattr(constraint, attr)
            row[attr] = float(value) if isinstance(value, (int, float)) else value
    return row


def bone_payload(armature, bone_name: str) -> dict:
    pb = armature.pose.bones[bone_name]
    bone = armature.data.bones[bone_name]
    head = pose_bone_head_world(armature, pb)
    tail = pose_bone_tail_world(armature, pb)
    return {
        "bone": bone_name,
        "rest_length": float(bone.length),
        "pose_length_world": float((tail - head).length),
        "pose_scale": [float(value) for value in pb.scale],
        "lock_scale": [bool(value) for value in pb.lock_scale],
        "constraints": [constraint_payload(constraint) for constraint in pb.constraints],
    }


def configure_limit_scale_constraint(pb, constraint_name: str, scale_y: float) -> None:
    for constraint in list(pb.constraints):
        if constraint.name == constraint_name:
            pb.constraints.remove(constraint)

    constraint = pb.constraints.new(type="LIMIT_SCALE")
    constraint.name = constraint_name
    constraint.mute = False
    constraint.influence = 1.0
    for attr in ("use_min_x", "use_min_y", "use_min_z", "use_max_x", "use_max_y", "use_max_z"):
        if hasattr(constraint, attr):
            setattr(constraint, attr, True)
    constraint.min_x = 1.0
    constraint.max_x = 1.0
    constraint.min_y = float(scale_y)
    constraint.max_y = float(scale_y)
    constraint.min_z = 1.0
    constraint.max_z = 1.0
    constraint.owner_space = "LOCAL"


def apply_forelimb_length_locks(armature, scales_by_base: dict[str, float], constraint_name: str) -> dict:
    changed = {}
    for side in ("L", "R"):
        for base in FORELIMB_ARM_BASES:
            bone_name = f"{base}_{side}"
            pb = armature.pose.bones.get(bone_name)
            if pb is None:
                changed[bone_name] = {"missing": True}
                continue
            scale_y = float(scales_by_base[base])
            before = bone_payload(armature, bone_name)
            pb.scale = (1.0, scale_y, 1.0)
            configure_limit_scale_constraint(pb, constraint_name, scale_y)
            changed[bone_name] = {
                "missing": False,
                "requested_scale_y": scale_y,
                "before": before,
            }
    bpy.context.view_layer.update()
    for bone_name, row in changed.items():
        if not row.get("missing"):
            row["after"] = bone_payload(armature, bone_name)
    return changed


def main() -> None:
    args = parse_args()
    bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()))
    armature = bpy.data.objects.get(args.armature)
    if armature is None or armature.type != "ARMATURE":
        raise RuntimeError(f"Missing armature {args.armature!r}")

    scale = float(args.scale)
    scales_by_base = {
        "Scapula": float(args.scapula_scale) if args.scapula_scale is not None else scale,
        "Wrist": float(args.wrist_scale) if args.wrist_scale is not None else scale,
    }
    changed = apply_forelimb_length_locks(armature, scales_by_base, args.constraint_name)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output.resolve()))

    report = {
        "input": str(args.blend.resolve()),
        "output": str(args.output.resolve()),
        "armature": armature.name,
        "constraint_name": args.constraint_name,
        "scales_by_base": scales_by_base,
        "changed_bones": changed,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OLDIK_FORELIMB_LENGTH_LOCK_BEGIN")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("OLDIK_FORELIMB_LENGTH_LOCK_END")


if __name__ == "__main__":
    main()
