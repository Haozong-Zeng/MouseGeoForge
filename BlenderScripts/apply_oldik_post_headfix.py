"""Post-process an old-IK migrated action with a true world-space headfix.

The migration script's --preserve-head restores head bones before the final
Tail0/root lock, so the head can still move in world space.  This script runs
after migration and re-bakes a selected head/neck cluster to a reference pose.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import migrate_animations_to_ik as mig  # noqa: E402


DEFAULT_HEADFIX_BONES = (
    "Chest_M",
    "Neck3_M",
    "Neck1_M",
    "Neck2_M",
    "Head1_M",
    "HeadEnd_M",
    "Jaw_M",
    "JawEnd_M",
)


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blend", type=Path, required=True)
    parser.add_argument("--reference-blend", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--armature", default="DeformationSystem")
    parser.add_argument("--reference-armature", default="DeformationSystem")
    parser.add_argument("--action", default="MIG_BrowmFur_FGcamB")
    parser.add_argument("--reference-action", default=None)
    parser.add_argument("--frame-start", type=float, default=None)
    parser.add_argument("--frame-end", type=float, default=None)
    parser.add_argument(
        "--bone",
        action="append",
        default=[],
        help="Bone to lock. Defaults to Chest/Neck/Head/Jaw headfix cluster.",
    )
    return parser.parse_args(args_after_dash())


def action_fcurves(action) -> list:
    fcurves = []
    if getattr(action, "fcurves", None) is not None:
        fcurves.extend(action.fcurves)
    elif getattr(action, "is_action_layered", False):
        for layer in getattr(action, "layers", []):
            for strip in getattr(layer, "strips", []):
                for channelbag in getattr(strip, "channelbags", []):
                    fcurves.extend(channelbag.fcurves)
    return fcurves


def action_frames(action) -> list[float]:
    frames = set()
    for fcurve in action_fcurves(action):
        for key in fcurve.keyframe_points:
            frames.add(round(float(key.co.x), 6))
    if frames:
        return sorted(frames)
    start, end = action.frame_range
    return [float(frame) for frame in range(math.ceil(start), math.floor(end) + 1)]


def set_action(armature, action_name: str | None):
    armature.animation_data_create()
    if action_name:
        action = bpy.data.actions.get(action_name)
        if action is None:
            raise RuntimeError(f"Missing action {action_name!r}")
        armature.animation_data.action = action
        return action
    if armature.animation_data and armature.animation_data.action:
        return armature.animation_data.action
    return None


def set_frame(frame: float) -> None:
    whole = math.floor(frame)
    bpy.context.scene.frame_set(whole, subframe=frame - whole)
    bpy.context.view_layer.update()


def pose_bone_world_matrix(armature, bone_name: str) -> Matrix:
    return armature.matrix_world @ armature.pose.bones[bone_name].matrix


def pose_tail_world(armature, bone_name: str) -> Vector:
    pose_bone = armature.pose.bones[bone_name]
    local_tail = pose_bone.matrix @ Vector((0.0, pose_bone.bone.length, 0.0))
    return armature.matrix_world @ local_tail


def vector_payload(vector: Vector) -> list[float]:
    return [float(value) for value in vector]


def collect_reference_world_matrices(args: argparse.Namespace, bone_names: list[str]) -> dict[str, Matrix]:
    bpy.ops.wm.open_mainfile(filepath=str(args.reference_blend.resolve()))
    armature = bpy.data.objects.get(args.reference_armature)
    if armature is None or armature.type != "ARMATURE":
        raise RuntimeError(f"Missing reference armature {args.reference_armature!r}")
    set_action(armature, args.reference_action)
    set_frame(1.0)
    missing = [name for name in bone_names if name not in armature.pose.bones]
    if missing:
        raise RuntimeError(f"Reference is missing bones: {missing}")
    return {name: pose_bone_world_matrix(armature, name).copy() for name in bone_names}


def choose_frames(args: argparse.Namespace, action) -> list[float]:
    frames = action_frames(action)
    if args.frame_start is not None:
        frames = [frame for frame in frames if frame >= args.frame_start]
    if args.frame_end is not None:
        frames = [frame for frame in frames if frame <= args.frame_end]
    return frames


def apply_reference_pose(
    armature,
    reference_world: dict[str, Matrix],
    bone_names: list[str],
) -> None:
    world_to_armature = armature.matrix_world.inverted()
    for bone_name in bone_names:
        desired_local = world_to_armature @ reference_world[bone_name]
        mig.set_pose_matrix_via_basis(armature, bone_name, desired_local)
        bpy.context.view_layer.update()


def key_bones(armature, bone_names: list[str], frame: float) -> None:
    for bone_name in bone_names:
        pose_bone = armature.pose.bones[bone_name]
        pose_bone.keyframe_insert(data_path="location", frame=frame)
        pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame)
        pose_bone.keyframe_insert(data_path="scale", frame=frame)


def main() -> None:
    args = parse_args()
    bone_names = args.bone or list(DEFAULT_HEADFIX_BONES)
    reference_world = collect_reference_world_matrices(args, bone_names)

    bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()))
    armature = bpy.data.objects.get(args.armature)
    if armature is None or armature.type != "ARMATURE":
        raise RuntimeError(f"Missing armature {args.armature!r}")
    missing = [name for name in bone_names if name not in armature.pose.bones]
    if missing:
        raise RuntimeError(f"Target is missing bones: {missing}")
    action = set_action(armature, args.action)
    if action is None:
        raise RuntimeError("No target action found")

    frames = choose_frames(args, action)
    rows = []
    for index, frame in enumerate(frames, start=1):
        set_frame(frame)
        before_head = pose_tail_world(armature, "HeadEnd_M") if "HeadEnd_M" in armature.pose.bones else None
        apply_reference_pose(armature, reference_world, bone_names)
        after_head = pose_tail_world(armature, "HeadEnd_M") if "HeadEnd_M" in armature.pose.bones else None
        key_bones(armature, bone_names, frame)
        if index == 1 or index == len(frames) or index % 50 == 0:
            print(f"[headfix] baked {index}/{len(frames)} frame={frame}")
        rows.append(
            {
                "frame": float(frame),
                "headend_tail_before": None if before_head is None else vector_payload(before_head),
                "headend_tail_after": None if after_head is None else vector_payload(after_head),
                "headend_tail_delta": None
                if before_head is None or after_head is None
                else float((after_head - before_head).length),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output.resolve()))

    report = {
        "input": str(args.blend.resolve()),
        "reference": str(args.reference_blend.resolve()),
        "output": str(args.output.resolve()),
        "action": action.name,
        "reference_action": args.reference_action,
        "bone_names": bone_names,
        "frames": [float(frame) for frame in frames],
        "headend_tail_reference": vector_payload(reference_world["HeadEnd_M"] @ Vector((0.0, bpy.data.objects[args.armature].pose.bones["HeadEnd_M"].bone.length, 0.0)))
        if "HeadEnd_M" in reference_world and args.armature in bpy.data.objects and "HeadEnd_M" in bpy.data.objects[args.armature].pose.bones
        else None,
        "rows_sample": rows[:5] + rows[-5:] if len(rows) > 10 else rows,
        "max_headend_tail_delta": max((row["headend_tail_delta"] or 0.0 for row in rows), default=0.0),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OLDIK_POST_HEADFIX_BEGIN")
    print(json.dumps({key: report[key] for key in ("output", "bone_names", "frames", "max_headend_tail_delta")}, indent=2))
    print("OLDIK_POST_HEADFIX_END")


if __name__ == "__main__":
    main()
