"""Bake a conservative two-bone wrist/palm position assist for old-IK hands.

The proxy palm spheres mark a palm surface point, not the wrist joint. This
script first calibrates the local surface-to-Fingers1-root offset from stable
early frames, then uses a two-bone analytic IK pass on Scapula/Wrist to move
the hand root toward the inferred target while preserving the V50 palm
orientation assist.
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
    parser.add_argument("--action", default="MIG_BrowmFur_FGcamB_v50_palm_orient")
    parser.add_argument("--new-action", default="MIG_BrowmFur_FGcamB_v51_wrist_position")
    parser.add_argument("--frame-start", type=int, default=1)
    parser.add_argument("--frame-end", type=int, default=240)
    parser.add_argument("--position-influence", type=float, default=0.45)
    parser.add_argument("--max-step", type=float, default=8.0)
    parser.add_argument("--side", action="append", choices=SIDES, default=[])
    parser.add_argument("--calibration-frame", type=int, action="append", default=[])
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


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def rounded_vec(vec: Vector) -> list[float]:
    return [round(float(vec[i]), 5) for i in range(3)]


def bone_world_matrix(armature, pb) -> Matrix:
    return armature.matrix_world @ pb.matrix


def bone_world_rotation(armature, pb) -> Quaternion:
    return bone_world_matrix(armature, pb).to_quaternion()


def set_pose_bone_world_matrix(armature, pb, world_matrix: Matrix) -> None:
    pb.matrix = armature.matrix_world.inverted() @ world_matrix


def set_pose_bone_world_rotation(armature, pb, world_quat: Quaternion) -> None:
    world_mat = world_quat.to_matrix().to_4x4()
    world_mat.translation = armature.matrix_world @ pb.head
    set_pose_bone_world_matrix(armature, pb, world_mat)


def make_oriented_segment_matrix(head: Vector, tail: Vector, current_matrix: Matrix) -> Matrix:
    y_axis = tail - head
    if y_axis.length < 1e-6:
        y_axis = current_matrix.to_3x3().col[1]
    y_axis.normalize()

    z_ref = current_matrix.to_3x3().col[2].normalized()
    z_axis = z_ref - y_axis * z_ref.dot(y_axis)
    if z_axis.length < 1e-6:
        x_ref = current_matrix.to_3x3().col[0].normalized()
        z_axis = y_axis.cross(x_ref)
    if z_axis.length < 1e-6:
        z_axis = Vector((0.0, 0.0, 1.0))
    z_axis.normalize()
    x_axis = y_axis.cross(z_axis)
    if x_axis.length < 1e-6:
        x_axis = Vector((1.0, 0.0, 0.0))
    x_axis.normalize()
    z_axis = x_axis.cross(y_axis).normalized()

    mat = Matrix(
        (
            (x_axis.x, y_axis.x, z_axis.x, head.x),
            (x_axis.y, y_axis.y, z_axis.y, head.y),
            (x_axis.z, y_axis.z, z_axis.z, head.z),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    return mat


def solve_two_bone(
    shoulder: Vector,
    elbow_current: Vector,
    wrist_current: Vector,
    target: Vector,
    upper_len: float,
    lower_len: float,
) -> tuple[Vector, Vector, dict]:
    to_target = target - shoulder
    dist = to_target.length
    min_reach = abs(upper_len - lower_len) + 1e-4
    max_reach = upper_len + lower_len - 1e-4
    clamped = False
    if dist < 1e-6:
        direction = (wrist_current - shoulder).normalized()
    else:
        direction = to_target.normalized()
    if dist > max_reach:
        target = shoulder + direction * max_reach
        dist = max_reach
        clamped = True
    elif dist < min_reach:
        target = shoulder + direction * min_reach
        dist = min_reach
        clamped = True

    pole = elbow_current - (shoulder + direction * (elbow_current - shoulder).dot(direction))
    if pole.length < 1e-6:
        old_direction = wrist_current - shoulder
        if old_direction.length > 1e-6:
            pole = elbow_current - (shoulder + old_direction.normalized() * (elbow_current - shoulder).dot(old_direction.normalized()))
    if pole.length < 1e-6:
        pole = Vector((0.0, 0.0, 1.0))
        if abs(pole.dot(direction)) > 0.95:
            pole = Vector((1.0, 0.0, 0.0))
        pole = pole - direction * pole.dot(direction)
    pole.normalize()

    along = (upper_len * upper_len - lower_len * lower_len + dist * dist) / max(2.0 * dist, 1e-6)
    height_sq = max(0.0, upper_len * upper_len - along * along)
    elbow_new = shoulder + direction * along + pole * math.sqrt(height_sq)
    return elbow_new, target, {"clamped": clamped, "target_distance_from_shoulder": round(float(dist), 5)}


def ensure_empty(name: str, display_size: float = 1.5):
    obj = bpy.data.objects.get(name)
    if obj is None:
        obj = bpy.data.objects.new(name, None)
        bpy.context.scene.collection.objects.link(obj)
    obj.empty_display_type = "SPHERE"
    obj.empty_display_size = display_size
    obj.hide_render = True
    obj.hide_viewport = False
    return obj


def palm_proxy(side: str) -> Vector:
    return get_obj(f"MGF_HandSphere_{side}_palm").matrix_world.translation.copy()


def calibrate_offsets(armature, frames: list[int], sides: tuple[str, ...]) -> dict[str, dict]:
    offsets: dict[str, dict] = {}
    for side in sides:
        pb = get_pb(armature, f"Fingers1_{side}")
        samples = []
        for frame in frames:
            bpy.context.scene.frame_set(frame)
            bpy.context.view_layer.update()
            quat = bone_world_rotation(armature, pb)
            head = armature.matrix_world @ pb.head
            local_offset = quat.inverted() @ (palm_proxy(side) - head)
            samples.append(local_offset)
        offset = Vector((0.0, 0.0, 0.0))
        for sample in samples:
            offset += sample
        offset /= max(1, len(samples))
        offsets[side] = {
            "frames": frames,
            "local_offset": offset,
            "sample_lengths": [round(float(sample.length), 5) for sample in samples],
        }
    return offsets


def keyframe_rotation(pb, frame: int) -> None:
    pb.keyframe_insert(data_path="rotation_euler", frame=frame)
    if pb.rotation_mode == "QUATERNION":
        pb.keyframe_insert(data_path="rotation_quaternion", frame=frame)


def main() -> None:
    args = parse_args()
    position_influence = clamp(args.position_influence, 0.0, 1.0)
    active_sides = tuple(args.side) if args.side else SIDES
    calibration_frames = args.calibration_frame or [1, 50, 100]
    inspect_frames = set(args.inspect_frame or [169, 170, 179, 239])

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

    bones = {
        side: {
            "upper": get_pb(armature, f"Scapula_{side}"),
            "lower": get_pb(armature, f"Wrist_{side}"),
            "palm": get_pb(armature, f"Fingers1_{side}"),
        }
        for side in active_sides
    }
    for side_bones in bones.values():
        for pb in side_bones.values():
            pb.rotation_mode = "XYZ"

    offsets = calibrate_offsets(armature, calibration_frames, active_sides)
    target_empties = {side: ensure_empty(f"MGF_WristPalmTarget_{side}") for side in active_sides}

    diagnostics = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "source_action": action.name,
        "new_action": new_action.name,
        "position_influence": position_influence,
        "max_step": args.max_step,
        "active_sides": list(active_sides),
        "calibration": {
            side: {
                "frames": offsets[side]["frames"],
                "local_offset": rounded_vec(offsets[side]["local_offset"]),
                "sample_lengths": offsets[side]["sample_lengths"],
            }
            for side in active_sides
        },
        "frames": {},
    }

    for frame in range(args.frame_start, args.frame_end + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        frame_diag = {}

        for side in active_sides:
            upper = bones[side]["upper"]
            lower = bones[side]["lower"]
            palm = bones[side]["palm"]

            palm_world_quat = bone_world_rotation(armature, palm)
            shoulder = armature.matrix_world @ upper.head
            elbow = armature.matrix_world @ upper.tail
            wrist = armature.matrix_world @ lower.tail
            current_upper_mat = bone_world_matrix(armature, upper)
            current_lower_mat = bone_world_matrix(armature, lower)

            surface_target = palm_proxy(side)
            inferred_root_target = surface_target - (palm_world_quat @ offsets[side]["local_offset"])
            wanted_delta = inferred_root_target - wrist
            wanted_delta_len = wanted_delta.length
            if wanted_delta_len > args.max_step > 0.0:
                inferred_root_target = wrist + wanted_delta.normalized() * args.max_step
                wanted_delta = inferred_root_target - wrist
                wanted_delta_len = wanted_delta.length
            applied_target = wrist.lerp(inferred_root_target, position_influence)

            target_empty = target_empties[side]
            target_empty.location = applied_target
            target_empty.keyframe_insert(data_path="location", frame=frame)

            upper_len = (elbow - shoulder).length
            lower_len = (wrist - elbow).length
            elbow_new, wrist_new, solve_diag = solve_two_bone(
                shoulder=shoulder,
                elbow_current=elbow,
                wrist_current=wrist,
                target=applied_target,
                upper_len=upper_len,
                lower_len=lower_len,
            )

            upper_mat = make_oriented_segment_matrix(shoulder, elbow_new, current_upper_mat)
            set_pose_bone_world_matrix(armature, upper, upper_mat)
            bpy.context.view_layer.update()
            lower_mat = make_oriented_segment_matrix(elbow_new, wrist_new, current_lower_mat)
            set_pose_bone_world_matrix(armature, lower, lower_mat)
            bpy.context.view_layer.update()

            # Keep the V50 palm orientation after moving the arm chain.
            set_pose_bone_world_rotation(armature, palm, palm_world_quat)
            bpy.context.view_layer.update()

            keyframe_rotation(upper, frame)
            keyframe_rotation(lower, frame)
            keyframe_rotation(palm, frame)

            if frame in inspect_frames:
                after_wrist = armature.matrix_world @ lower.tail
                frame_diag[side] = {
                    "surface_proxy": rounded_vec(surface_target),
                    "current_wrist_before": rounded_vec(wrist),
                    "inferred_root_target": rounded_vec(inferred_root_target),
                    "applied_target": rounded_vec(applied_target),
                    "wrist_after": rounded_vec(after_wrist),
                    "delta_to_inferred_before": round(float((inferred_root_target - wrist).length), 5),
                    "delta_to_inferred_after": round(float((inferred_root_target - after_wrist).length), 5),
                    "requested_delta_len_after_max_step": round(float(wanted_delta_len), 5),
                    **solve_diag,
                }
        if frame_diag:
            diagnostics["frames"][str(frame)] = frame_diag

    bpy.context.scene.frame_set(args.frame_start)
    bpy.context.view_layer.update()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output.resolve()))

    report_path = args.report or args.output.with_suffix(".wrist-position-report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(diagnostics, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "report": str(report_path.resolve())}, indent=2))


if __name__ == "__main__":
    main()
