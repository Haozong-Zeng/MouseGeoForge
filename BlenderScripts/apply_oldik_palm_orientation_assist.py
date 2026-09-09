"""Bake a conservative palm-orientation assist for the old-IK hand.

The hand proxy spheres are visual references in V49; they do not actively drive
the pose. This script creates PalmFrame empties from the proxy layout and bakes
Fingers1_L/R rotation toward those frames so finger curl no longer has to
compensate for a missing palm flip.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Quaternion, Vector


SIDES = ("L", "R")


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--armature", default="DeformationSystem")
    parser.add_argument("--action", default="MIG_BrowmFur_FGcamB_v44_smooth60")
    parser.add_argument("--new-action", default="MIG_BrowmFur_FGcamB_v50_palm_orient")
    parser.add_argument("--frame-start", type=int, default=1)
    parser.add_argument("--frame-end", type=int, default=240)
    parser.add_argument("--influence", type=float, default=0.65)
    parser.add_argument("--inspect-frame", type=int, action="append", default=[])
    return parser.parse_args(args_after_dash())


def get_obj(name: str):
    obj = bpy.data.objects.get(name)
    if obj is None:
        raise RuntimeError(f"Object {name!r} not found.")
    return obj


def get_pb(armature, name: str):
    pb = armature.pose.bones.get(name)
    if pb is None:
        raise RuntimeError(f"Pose bone {name!r} not found.")
    return pb


def mean_vec(points: list[Vector]) -> Vector:
    total = Vector((0.0, 0.0, 0.0))
    for point in points:
        total += point
    return total / max(1, len(points))


def safe_normalized(vec: Vector, fallback: Vector) -> Vector:
    if vec.length < 1e-6:
        return fallback.normalized()
    return vec.normalized()


def matrix_from_columns(x_axis: Vector, y_axis: Vector, z_axis: Vector) -> Matrix:
    return Matrix(
        (
            (x_axis.x, y_axis.x, z_axis.x),
            (x_axis.y, y_axis.y, z_axis.y),
            (x_axis.z, y_axis.z, z_axis.z),
        )
    )


def quaternion_angle_degrees(a: Quaternion, b: Quaternion) -> float:
    return round(math.degrees(a.rotation_difference(b).angle), 4)


def palm_proxy_positions(side: str) -> dict[str, Vector]:
    names = {
        "palm": f"MGF_HandSphere_{side}_palm",
        "index": f"MGF_HandSphere_{side}_index_tip",
        "middle": f"MGF_HandSphere_{side}_middle_tip",
        "ring": f"MGF_HandSphere_{side}_ring_tip",
        "pinky": f"MGF_HandSphere_{side}_pinky_tip",
    }
    return {key: get_obj(name).matrix_world.translation.copy() for key, name in names.items()}


def make_palm_frame(side: str, current_y: Vector, current_z: Vector) -> tuple[Matrix, dict]:
    pts = palm_proxy_positions(side)
    palm = pts["palm"]
    tips = [pts["index"], pts["middle"], pts["ring"], pts["pinky"]]
    tip_center = mean_vec(tips)
    forward = safe_normalized(tip_center - palm, current_y)
    radial = safe_normalized(mean_vec([pts["index"], pts["middle"]]) - mean_vec([pts["ring"], pts["pinky"]]), current_y.cross(current_z))

    raw_normal = radial.cross(forward)
    if raw_normal.length < 1e-6:
        raw_normal = current_z.copy()
    candidates = [raw_normal.normalized(), -raw_normal.normalized()]
    best = None
    best_score = -1e9
    for normal in candidates:
        x_axis = safe_normalized(forward.cross(normal), radial)
        normal = safe_normalized(x_axis.cross(forward), normal)
        # Prefer palm-up in Blender world Z, but keep radial consistency too.
        score = 2.0 * normal.z + 0.8 * x_axis.dot(radial) + 0.25 * normal.dot(current_z)
        if score > best_score:
            best_score = score
            best = (x_axis, forward, normal)
    x_axis, y_axis, z_axis = best
    rot = matrix_from_columns(x_axis, y_axis, z_axis)
    mat = rot.to_4x4()
    mat.translation = palm
    diagnostics = {
        "palm": [round(float(v), 5) for v in palm],
        "forward": [round(float(v), 5) for v in y_axis],
        "radial": [round(float(v), 5) for v in radial],
        "normal": [round(float(v), 5) for v in z_axis],
        "normal_world_z": round(float(z_axis.z), 5),
        "score": round(float(best_score), 5),
    }
    return mat, diagnostics


def ensure_empty(name: str, display_size: float = 1.0):
    obj = bpy.data.objects.get(name)
    if obj is None:
        obj = bpy.data.objects.new(name, None)
        bpy.context.scene.collection.objects.link(obj)
    obj.empty_display_type = "ARROWS"
    obj.empty_display_size = display_size
    obj.hide_render = True
    obj.hide_viewport = False
    return obj


def relax_palm_parent_rotation_window(pb) -> list[dict]:
    changes = []
    for constraint in pb.constraints:
        if constraint.type != "LIMIT_ROTATION":
            continue
        before = {
            "name": constraint.name,
            "min_x": round(math.degrees(constraint.min_x), 3),
            "max_x": round(math.degrees(constraint.max_x), 3),
            "min_y": round(math.degrees(constraint.min_y), 3),
            "max_y": round(math.degrees(constraint.max_y), 3),
            "min_z": round(math.degrees(constraint.min_z), 3),
            "max_z": round(math.degrees(constraint.max_z), 3),
        }
        constraint.use_limit_x = True
        constraint.use_limit_y = True
        constraint.use_limit_z = True
        constraint.min_x = constraint.min_y = constraint.min_z = -math.pi
        constraint.max_x = constraint.max_y = constraint.max_z = math.pi
        after = {
            "name": constraint.name,
            "min_x": -180.0,
            "max_x": 180.0,
            "min_y": -180.0,
            "max_y": 180.0,
            "min_z": -180.0,
            "max_z": 180.0,
        }
        changes.append({"before": before, "after": after})
    return changes


def set_pose_bone_world_rotation(armature, pb, world_quat: Quaternion) -> None:
    world_mat = world_quat.to_matrix().to_4x4()
    world_mat.translation = armature.matrix_world @ pb.head
    pose_mat = armature.matrix_world.inverted() @ world_mat
    pb.matrix = pose_mat


def bone_world_rotation(armature, pb) -> Quaternion:
    return (armature.matrix_world @ pb.matrix).to_quaternion()


def bone_world_axes(armature, pb) -> tuple[Vector, Vector, Vector]:
    mat = (armature.matrix_world @ pb.matrix).to_3x3()
    return mat.col[0].normalized(), mat.col[1].normalized(), mat.col[2].normalized()


def set_linear_interpolation(action, bone_names: list[str]) -> None:
    if not hasattr(action, "fcurves"):
        return
    for fcurve in action.fcurves:
        if not any(f'pose.bones["{name}"].rotation_euler' in fcurve.data_path for name in bone_names):
            continue
        for key in fcurve.keyframe_points:
            key.interpolation = "LINEAR"


def main() -> None:
    args = parse_args()
    influence = max(0.0, min(1.0, args.influence))
    bpy.ops.wm.open_mainfile(filepath=str(args.input.resolve()))

    armature = get_obj(args.armature)
    if armature.type != "ARMATURE":
        raise RuntimeError(f"{args.armature!r} is not an armature.")

    action = bpy.data.actions.get(args.action)
    if action is None:
        raise RuntimeError(f"Action {args.action!r} not found.")
    new_action = action.copy()
    new_action.name = args.new_action
    armature.animation_data_create()
    armature.animation_data.action = new_action

    frame_objs = {side: ensure_empty(f"MGF_PalmFrame_{side}", 2.0) for side in SIDES}
    palm_bones = {side: get_pb(armature, f"Fingers1_{side}") for side in SIDES}
    for pb in palm_bones.values():
        pb.rotation_mode = "XYZ"

    constraint_changes = {side: relax_palm_parent_rotation_window(pb) for side, pb in palm_bones.items()}
    inspect_frames = set(args.inspect_frame or [169, 170, 179, 239])
    diagnostics = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "source_action": action.name,
        "new_action": new_action.name,
        "influence": influence,
        "constraint_changes": constraint_changes,
        "frames": {},
    }
    previous_target = {side: None for side in SIDES}
    bone_names = [f"Fingers1_{side}" for side in SIDES]

    for frame in range(args.frame_start, args.frame_end + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        frame_diag = {}
        for side, pb in palm_bones.items():
            _, current_y, current_z = bone_world_axes(armature, pb)
            target_mat, palm_diag = make_palm_frame(side, current_y, current_z)
            target_quat = target_mat.to_quaternion()
            if previous_target[side] is not None and previous_target[side].dot(target_quat) < 0:
                target_quat.negate()
            previous_target[side] = target_quat.copy()

            empty = frame_objs[side]
            empty.matrix_world = target_mat
            empty.rotation_mode = "QUATERNION"
            empty.rotation_quaternion = target_quat
            empty.keyframe_insert(data_path="location", frame=frame)
            empty.keyframe_insert(data_path="rotation_quaternion", frame=frame)

            before_quat = bone_world_rotation(armature, pb)
            blended_quat = before_quat.slerp(target_quat, influence)
            set_pose_bone_world_rotation(armature, pb, blended_quat)
            bpy.context.view_layer.update()
            pb.keyframe_insert(data_path="rotation_euler", frame=frame)

            if frame in inspect_frames:
                after_quat = bone_world_rotation(armature, pb)
                frame_diag[side] = {
                    **palm_diag,
                    "angle_to_palm_before_deg": quaternion_angle_degrees(before_quat, target_quat),
                    "angle_to_palm_after_deg": quaternion_angle_degrees(after_quat, target_quat),
                    "rotation_euler_deg_after": [round(math.degrees(float(v)), 4) for v in pb.rotation_euler],
                }
        if frame_diag:
            diagnostics["frames"][str(frame)] = frame_diag

    set_linear_interpolation(new_action, bone_names)
    bpy.context.scene.frame_set(args.frame_start)
    bpy.context.view_layer.update()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output.resolve()))

    report_path = args.report or args.output.with_suffix(".palm-orientation-report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(diagnostics, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "report": str(report_path.resolve())}, indent=2))


if __name__ == "__main__":
    main()
