#!/usr/bin/env python3
"""Drive bilateral Blender hand controls from aligned VGGT/DLC keypoints.

Run with Blender:
  blender source.blend --background --python this_script.py -- \
      --descriptors hand_descriptors.csv \
      --keypoints aligned_keypoints_blender.csv \
      --output output.blend
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import bpy
from mathutils import Matrix, Quaternion, Vector


SIDE_CONFIG = {
    "l": {
        "controller": "CTRL_Hand_L",
        "color": (0.15, 0.55, 1.0, 1.0),
    },
    "r": {
        "controller": "CTRL_Hand_R",
        "color": (1.0, 0.28, 0.12, 1.0),
    },
}


def blender_args() -> list[str]:
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptors", type=Path, required=True)
    parser.add_argument("--keypoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--bind-frame",
        type=int,
        default=None,
        help="Source zero-based frame. Auto-selected when omitted.",
    )
    parser.add_argument("--frame-offset", type=int, default=1)
    parser.add_argument("--smooth-radius", type=int, default=2)
    parser.add_argument("--max-position-jump", type=float, default=10.0)
    parser.add_argument("--max-rotation-jump-deg", type=float, default=55.0)
    parser.add_argument(
        "--palm-orientation-mode",
        choices=("full-frame", "forearm-roll"),
        default="full-frame",
    )
    parser.add_argument("--remove-elbow-poles", action="store_true")
    parser.add_argument(
        "--elbow-pole-mode",
        choices=("keep", "remove", "rest-prior"),
        default="keep",
        help="rest-prior transports the bind pole with the shoulder-to-hand axis.",
    )
    parser.add_argument("--compact-fingers", action="store_true")
    parser.add_argument("--cup-object", default="Part_PrintedMouseCup")
    parser.add_argument(
        "--cup-front-clearance",
        type=float,
        default=0.0,
        help="Push palm/tip probes in front of the cup maximum world Y plane.",
    )
    parser.add_argument("--cup-front-max-push", type=float, default=8.0)
    parser.add_argument("--cup-xz-margin", type=float, default=1.5)
    parser.add_argument(
        "--cup-rim-clearance",
        type=float,
        default=0.0,
        help="Minimum resting palm height above the detected front rim.",
    )
    parser.add_argument("--cup-rim-max-push", type=float, default=7.5)
    parser.add_argument("--resting-mouth-distance", type=float, default=20.0)
    parser.add_argument(
        "--allow-cup-correction-in-active-window",
        action="store_true",
        help=(
            "Apply cup front/rim position correction during active hand "
            "windows while preserving the active palm and finger animation."
        ),
    )
    parser.add_argument("--max-palm-roll-deg", type=float, default=135.0)
    parser.add_argument("--max-palm-roll-step-deg", type=float, default=18.0)
    parser.add_argument(
        "--principal-prebind-palm-branch",
        action="store_true",
        help="Resolve pre-bind palm-normal ambiguity to the roll branch nearest zero.",
    )
    parser.add_argument(
        "--active-window",
        action="append",
        default=[],
        help="Side/source-frame range, for example r:62:100.",
    )
    parser.add_argument(
        "--palm-roll-offset-anchor",
        action="append",
        default=[],
        help="Side:source_frame:offset_degrees, linearly interpolated.",
    )
    parser.add_argument(
        "--palm-roll-target-anchor",
        action="append",
        default=[],
        help=(
            "Side:source_frame:absolute_degrees. Between the first and last "
            "anchor, replace noisy descriptor roll with this interpolated target."
        ),
    )
    parser.add_argument(
        "--palm-face-mouth-window",
        action="append",
        default=[],
        help=(
            "Side/source-frame range where the controller palm +X axis is "
            "rolled toward the mouth, for example r:100:180."
        ),
    )
    parser.add_argument("--palm-face-mouth-left-weight", type=float, default=0.35)
    parser.add_argument("--palm-face-mouth-blend-frames", type=int, default=5)
    parser.add_argument("--palm-face-mouth-max-step-deg", type=float, default=12.0)
    parser.add_argument("--palm-roll-median-radius", type=int, default=0)
    parser.add_argument(
        "--finger-length-compensation",
        type=float,
        default=1.0,
        help=(
            "Divide local Y scale of each first finger bone by this factor. "
            "Use 1.2 when the fingers inherited a 1.2 forelimb extension."
        ),
    )
    parser.add_argument("--drive-finger-grip", action="store_true")
    parser.add_argument(
        "--drive-normalized-fingers",
        action="store_true",
        help=(
            "Retarget scale-free Anipose finger bend and spread onto the "
            "model's existing finger lengths and curl drivers."
        ),
    )
    parser.add_argument("--normalized-finger-curl-max-deg", type=float, default=32.0)
    parser.add_argument("--normalized-finger-spread-gain", type=float, default=0.45)
    parser.add_argument("--normalized-finger-spread-max-deg", type=float, default=5.0)
    parser.add_argument("--normalized-finger-smooth-radius", type=int, default=3)
    parser.add_argument("--normalized-finger-max-step-deg", type=float, default=4.0)
    parser.add_argument("--normalized-finger-quantile-low", type=float, default=0.10)
    parser.add_argument("--normalized-finger-quantile-high", type=float, default=0.90)
    parser.add_argument("--grip-open-distance", type=float, default=6.2)
    parser.add_argument("--grip-closed-distance", type=float, default=3.2)
    parser.add_argument("--grip-smooth-radius", type=int, default=2)
    parser.add_argument("--grip-open-curl-deg", type=float, default=10.0)
    parser.add_argument("--grip-closed-curl-deg", type=float, default=38.0)
    parser.add_argument("--grip-ring-open-curl-deg", type=float, default=0.0)
    parser.add_argument("--grip-ring-closed-curl-deg", type=float, default=8.0)
    parser.add_argument("--grip-open-spread-deg", type=float, default=2.5)
    parser.add_argument("--grip-closed-spread-deg", type=float, default=1.0)
    parser.add_argument("--grip-boundary-blend-frames", type=int, default=4)
    parser.add_argument(
        "--no-markers",
        action="store_true",
        help="Do not create viewport-only aligned keypoint markers.",
    )
    return parser.parse_args(blender_args())


def parse_active_windows(values: list[str]) -> dict[str, list[tuple[int, int]]]:
    windows: dict[str, list[tuple[int, int]]] = {
        side: [] for side in SIDE_CONFIG
    }
    for value in values:
        side, start, end = value.lower().split(":")
        if side not in windows:
            raise ValueError(f"Unknown active-window side: {side}")
        windows[side].append((int(start), int(end)))
    return windows


def parse_roll_anchors(
    values: list[str],
) -> dict[str, list[tuple[int, float]]]:
    anchors: dict[str, list[tuple[int, float]]] = {
        side: [] for side in SIDE_CONFIG
    }
    for value in values:
        side, frame, offset = value.lower().split(":")
        if side not in anchors:
            raise ValueError(f"Unknown palm-roll anchor side: {side}")
        anchors[side].append((int(frame), float(offset)))
    for side in anchors:
        anchors[side].sort()
    return anchors


def frame_in_windows(
    side: str,
    frame: int,
    windows: dict[str, list[tuple[int, int]]],
) -> bool:
    return any(start <= frame <= end for start, end in windows[side])


def finite_float(value: str | None) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def descriptor_rotation(row: dict[str, str]) -> Matrix | None:
    columns = []
    for axis in ("across", "forward", "normal"):
        values = [finite_float(row.get(f"{axis}_{component}")) for component in "xyz"]
        if any(value is None for value in values):
            return None
        columns.append(Vector(values))
    rotation = Matrix((columns[0], columns[1], columns[2])).transposed()
    if abs(rotation.determinant()) < 1e-6:
        return None
    # Filtering matrix entries can slightly break orthogonality.
    return rotation.to_quaternion().to_matrix()


def descriptor_position(row: dict[str, str]) -> Vector | None:
    values = [finite_float(row.get(f"palm_{component}")) for component in "xyz"]
    if any(value is None for value in values):
        return None
    return Vector(values)


def read_descriptors(path: Path) -> dict[str, dict[int, dict[str, object]]]:
    by_side: dict[str, dict[int, dict[str, object]]] = {
        side: {} for side in SIDE_CONFIG
    }
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            side = row.get("side", "").lower()
            if side not in by_side or row.get("valid", "").lower() != "true":
                continue
            frame = int(row["frame"])
            position = descriptor_position(row)
            rotation = descriptor_rotation(row)
            if position is None or rotation is None:
                continue
            by_side[side][frame] = {
                "position": position,
                "rotation": rotation,
                "raw": row,
            }
    return by_side


def read_hand_keypoints(
    path: Path,
) -> tuple[dict[str, dict[int, dict[str, Vector]]], dict[int, Vector]]:
    output: dict[str, dict[int, dict[str, Vector]]] = {
        side: defaultdict(dict) for side in SIDE_CONFIG
    }
    mouth_positions: dict[int, Vector] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            frame = int(float(row.get("fnum", row_index)))
            mouth_values = [
                finite_float(row.get(f"mouth_tip_A_{axis}")) for axis in "xyz"
            ]
            if not any(value is None for value in mouth_values):
                mouth_positions[frame] = Vector(mouth_values)
            for side in SIDE_CONFIG:
                names = [f"{side}_mid_A"]
                names.extend(
                    f"{side}{finger}_{landmark}_A"
                    for finger in range(1, 5)
                    for landmark in ("joint", "mid", "tip")
                )
                for name in names:
                    values = [finite_float(row.get(f"{name}_{axis}")) for axis in "xyz"]
                    score = finite_float(row.get(f"{name}_score"))
                    ncams = finite_float(row.get(f"{name}_ncams"))
                    if (
                        any(value is None for value in values)
                        or (score is not None and score < 0.5)
                        or (ncams is not None and ncams < 2.0)
                    ):
                        continue
                    output[side][frame][name] = Vector(values)
    return output, mouth_positions


def source_frame_range(path: Path) -> tuple[int, int]:
    frames = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            value = finite_float(row.get("fnum"))
            frames.append(int(value) if value is not None else row_index)
    if not frames:
        raise RuntimeError(f"No source frames in {path}")
    return min(frames), max(frames)


def quaternion_average(quaternions: list[Quaternion]) -> Quaternion:
    reference = quaternions[len(quaternions) // 2].normalized()
    accumulator = [0.0, 0.0, 0.0, 0.0]
    for quaternion in quaternions:
        current = quaternion.normalized()
        if reference.dot(current) < 0.0:
            current.negate()
        for index, component in enumerate(
            (current.w, current.x, current.y, current.z)
        ):
            accumulator[index] += component
    magnitude = math.sqrt(sum(component * component for component in accumulator))
    if magnitude < 1e-8:
        return reference
    return Quaternion(tuple(component / magnitude for component in accumulator))


def validate_rotation(rotation: Matrix, label: str) -> None:
    identity_error = max(
        abs(value - (1.0 if row == column else 0.0))
        for row, values in enumerate(rotation.transposed() @ rotation)
        for column, value in enumerate(values)
    )
    determinant_error = abs(rotation.determinant() - 1.0)
    if identity_error > 1e-4 or determinant_error > 1e-4:
        raise RuntimeError(
            f"{label} is not a proper rotation: "
            f"orthogonality error={identity_error:.6g}, "
            f"determinant={rotation.determinant():.6g}"
        )


def smooth_descriptors(
    by_side: dict[str, dict[int, dict[str, object]]],
    radius: int,
) -> dict[str, dict[int, dict[str, object]]]:
    smoothed: dict[str, dict[int, dict[str, object]]] = {
        side: {} for side in SIDE_CONFIG
    }
    for side, rows in by_side.items():
        frames = sorted(rows)
        for frame in frames:
            neighbors = [
                rows[other]
                for other in frames
                if abs(other - frame) <= radius
            ]
            positions = [item["position"] for item in neighbors]
            coordinates = [
                sorted(position[axis] for position in positions)
                for axis in range(3)
            ]
            median_position = Vector(
                coordinate[len(coordinate) // 2] for coordinate in coordinates
            )
            rotation = quaternion_average(
                [item["rotation"].to_quaternion() for item in neighbors]
            ).to_matrix()
            validate_rotation(rotation, f"{side} frame {frame} smoothed palm")
            smoothed[side][frame] = {
                "position": median_position,
                "rotation": rotation,
                "raw": rows[frame]["raw"],
            }
    return smoothed


def controller_world_pose(controller: bpy.types.Object) -> tuple[Vector, Matrix]:
    return (
        controller.matrix_world.translation.copy(),
        controller.matrix_world.to_quaternion().to_matrix(),
    )


def select_bind_frame(
    descriptors: dict[str, dict[int, dict[str, object]]],
    controllers: dict[str, bpy.types.Object],
    requested: int | None,
) -> tuple[int, list[dict[str, float]]]:
    common_frames = sorted(set(descriptors["l"]) & set(descriptors["r"]))
    if not common_frames:
        raise RuntimeError("No frame has valid descriptors for both hands.")
    if requested is not None:
        if requested not in common_frames:
            raise RuntimeError(
                f"Requested bind frame {requested} is not valid for both hands."
            )
        return requested, []

    ranked = []
    for frame in common_frames:
        side_distances = {}
        for side in SIDE_CONFIG:
            controller_position, _ = controller_world_pose(controllers[side])
            palm_position = descriptors[side][frame]["position"]
            side_distances[side] = (controller_position - palm_position).length
        ranked.append(
            {
                "frame": frame,
                "left_distance": side_distances["l"],
                "right_distance": side_distances["r"],
                "sum_distance": side_distances["l"] + side_distances["r"],
            }
        )
    ranked.sort(key=lambda item: item["sum_distance"])
    return int(ranked[0]["frame"]), ranked[:10]


def rotation_angle_degrees(first: Matrix, second: Matrix) -> float:
    delta = first.transposed() @ second
    return math.degrees(delta.to_quaternion().angle)


def gate_temporal_outliers(
    descriptors: dict[str, dict[int, dict[str, object]]],
    max_position_jump: float,
    max_rotation_jump_deg: float,
) -> tuple[dict[str, dict[int, dict[str, object]]], dict[str, list[int]]]:
    accepted: dict[str, dict[int, dict[str, object]]] = {
        side: {} for side in SIDE_CONFIG
    }
    rejected: dict[str, list[int]] = {side: [] for side in SIDE_CONFIG}
    for side, rows in descriptors.items():
        previous_frame = None
        previous = None
        for frame in sorted(rows):
            current = rows[frame]
            if previous is not None and previous_frame is not None:
                gap = max(1, frame - previous_frame)
                position_jump = (
                    current["position"] - previous["position"]
                ).length / gap
                rotation_jump = rotation_angle_degrees(
                    previous["rotation"], current["rotation"]
                ) / gap
                if (
                    position_jump > max_position_jump
                    or rotation_jump > max_rotation_jump_deg
                ):
                    rejected[side].append(frame)
                    continue
            accepted[side][frame] = current
            previous_frame = frame
            previous = current
    return accepted, rejected


def clear_object_animation(obj: bpy.types.Object) -> None:
    obj.animation_data_clear()
    obj.rotation_mode = "QUATERNION"


def keyframe_controller(
    controller: bpy.types.Object,
    source_frame: int,
    blender_frame: int,
    descriptor: dict[str, object],
    bind_descriptor: dict[str, object],
    bind_controller_position: Vector,
    bind_controller_rotation: Matrix,
    previous_quaternion: Quaternion | None,
) -> Quaternion:
    bind_palm_position = bind_descriptor["position"]
    bind_palm_rotation = bind_descriptor["rotation"]
    local_position_offset = (
        bind_palm_rotation.transposed()
        @ (bind_controller_position - bind_palm_position)
    )
    local_rotation_offset = bind_palm_rotation.transposed() @ bind_controller_rotation

    palm_position = descriptor["position"]
    palm_rotation = descriptor["rotation"]
    controller_position = palm_position + palm_rotation @ local_position_offset
    controller_rotation = palm_rotation @ local_rotation_offset
    quaternion = controller_rotation.to_quaternion().normalized()
    if previous_quaternion is not None and previous_quaternion.dot(quaternion) < 0.0:
        quaternion.negate()

    controller.location = controller_position
    controller.rotation_quaternion = quaternion
    controller.keyframe_insert("location", frame=blender_frame, group="VGGT_DLC_Hand")
    controller.keyframe_insert(
        "rotation_quaternion", frame=blender_frame, group="VGGT_DLC_Hand"
    )
    controller["vggt_dlc_source_frame"] = source_frame
    return quaternion


def controller_target_position(
    descriptor: dict[str, object],
    bind_descriptor: dict[str, object],
    bind_controller_position: Vector,
) -> Vector:
    bind_palm_position = bind_descriptor["position"]
    bind_palm_rotation = bind_descriptor["rotation"]
    local_position_offset = (
        bind_palm_rotation.transposed()
        @ (bind_controller_position - bind_palm_position)
    )
    return descriptor["position"] + descriptor["rotation"] @ local_position_offset


def object_world_bounds(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    points = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    minimum = Vector(min(point[axis] for point in points) for axis in range(3))
    maximum = Vector(max(point[axis] for point in points) for axis in range(3))
    return minimum, maximum


def cup_front_pushes(
    descriptors: dict[str, dict[int, dict[str, object]]],
    hand_keypoints: dict[str, dict[int, dict[str, Vector]]],
    mouth_positions: dict[int, Vector],
    cup: bpy.types.Object | None,
    clearance: float,
    max_push: float,
    xz_margin: float,
    smooth_radius: int,
    rim_clearance: float,
    rim_max_push: float,
    resting_mouth_distance: float,
    active_windows: dict[str, list[tuple[int, int]]],
) -> tuple[dict[str, dict[int, Vector]], dict[str, object]]:
    pushes: dict[str, dict[int, Vector]] = {side: {} for side in SIDE_CONFIG}
    if cup is None or (clearance <= 0.0 and rim_clearance <= 0.0):
        return pushes, {"enabled": False}

    bbox_min, bbox_max = object_world_bounds(cup)
    target_y = bbox_max.y + clearance
    cup_vertices = [cup.matrix_world @ vertex.co for vertex in cup.data.vertices]
    front_vertices = [
        vertex for vertex in cup_vertices if vertex.y >= bbox_max.y - 1.5
    ]
    rim_top_z = max((vertex.z for vertex in front_vertices), default=bbox_max.z)
    target_rest_z = rim_top_z + rim_clearance
    raw: dict[str, dict[int, Vector]] = {side: {} for side in SIDE_CONFIG}
    for side, rows in descriptors.items():
        for frame in rows:
            if frame_in_windows(side, frame, active_windows):
                raw[side][frame] = Vector((0.0, 0.0, 0.0))
                continue
            probes = hand_keypoints[side].get(frame, {})
            relevant = [
                point
                for name, point in probes.items()
                if (name == f"{side}_mid_A" or name.endswith("_tip_A"))
                if bbox_min.x - xz_margin <= point.x <= bbox_max.x + xz_margin
                and bbox_min.z - xz_margin <= point.z <= bbox_max.z + xz_margin
            ]
            push_y = 0.0
            if relevant:
                push_y = max(0.0, target_y - min(point.y for point in relevant))
                if max_push > 0.0:
                    push_y = min(push_y, max_push)

            push_z = 0.0
            palm = probes.get(f"{side}_mid_A")
            mouth = mouth_positions.get(frame)
            is_resting = (
                palm is not None
                and mouth is not None
                and (palm - mouth).length >= resting_mouth_distance
                and bbox_min.x - xz_margin <= palm.x <= bbox_max.x + xz_margin
                and palm.y <= target_y + 5.0
            )
            if is_resting and rim_clearance > 0.0:
                push_z = max(0.0, target_rest_z - palm.z)
                if rim_max_push > 0.0:
                    push_z = min(push_z, rim_max_push)
            raw[side][frame] = Vector((0.0, push_y, push_z))

    for side, rows in raw.items():
        frames = sorted(rows)
        for frame in frames:
            values = [
                rows[other]
                for other in frames
                if abs(other - frame) <= smooth_radius
            ]
            pushes[side][frame] = Vector(
                sorted(value[axis] for value in values)[len(values) // 2]
                for axis in range(3)
            )

    nonzero = {
        side: [value for value in rows.values() if value.length > 1e-6]
        for side, rows in pushes.items()
    }
    return pushes, {
        "enabled": True,
        "cup": cup.name,
        "bbox_min": list(bbox_min),
        "bbox_max": list(bbox_max),
        "target_front_y": target_y,
        "front_rim_top_z": rim_top_z,
        "target_resting_palm_z": target_rest_z,
        "resting_mouth_distance": resting_mouth_distance,
        "active_windows_excluded": active_windows,
        "corrected_frame_count": {
            side: len(values) for side, values in nonzero.items()
        },
        "max_push_y": {
            side: max((value.y for value in values), default=0.0)
            for side, values in nonzero.items()
        },
        "max_push_z": {
            side: max((value.z for value in values), default=0.0)
            for side, values in nonzero.items()
        },
        "directions": ["+world_Y front clearance", "+world_Z resting rim support"],
        "semantic_basis": (
            "CamC/CamB show paws outside or resting on the front face of the cup."
        ),
    }


def find_hand_ik_constraints(
    armature: bpy.types.Object,
) -> dict[str, bpy.types.Constraint]:
    constraints = {}
    for side in SIDE_CONFIG:
        suffix = side.upper()
        bone = armature.pose.bones.get(f"Wrist_{suffix}")
        if bone is None:
            raise RuntimeError(f"Missing Wrist_{suffix}")
        constraint = bone.constraints.get("MGF_ManualHandIK")
        if constraint is None or constraint.type != "IK":
            raise RuntimeError(f"Missing wrist IK on Wrist_{suffix}")
        constraints[side] = constraint
    return constraints


def remove_elbow_poles(
    constraints: dict[str, bpy.types.Constraint],
) -> dict[str, object]:
    report = {}
    for side, constraint in constraints.items():
        report[side] = {
            "previous_target": (
                None if constraint.pole_target is None else constraint.pole_target.name
            ),
            "previous_angle": float(constraint.pole_angle),
        }
        constraint.pole_target = None
        constraint.pole_subtarget = ""
        constraint.pole_angle = 0.0
    bpy.context.view_layer.update()
    return report


def bone_head_world(armature: bpy.types.Object, name: str) -> Vector:
    bone = armature.pose.bones[name]
    return armature.matrix_world @ bone.matrix.translation


def dynamic_rest_prior_poles(
    scene: bpy.types.Scene,
    armature: bpy.types.Object,
    controllers: dict[str, bpy.types.Object],
    constraints: dict[str, bpy.types.Constraint],
    target_positions: dict[str, dict[int, Vector]],
    bind_frame: int,
    frame_offset: int,
) -> dict[str, object]:
    report = {}
    for side in SIDE_CONFIG:
        constraint = constraints[side]
        pole = constraint.pole_target
        if pole is None:
            raise RuntimeError(f"{side} wrist IK has no pole object for rest-prior mode")
        if pole.parent is not None:
            raise RuntimeError(f"{pole.name} has a parent; rest-prior expects world space")
        pole.animation_data_clear()

        suffix = side.upper()
        shoulder_bind = bone_head_world(armature, f"Scapula_{suffix}")
        bind_target = target_positions[side][bind_frame]
        bind_reach = bind_target - shoulder_bind
        if bind_reach.length < 1e-8:
            raise RuntimeError(f"Degenerate bind shoulder-to-hand axis for {side}")
        bind_reach.normalize()
        pole_bind_position = pole.matrix_world.translation.copy()
        bind_offset = pole_bind_position - shoulder_bind

        keyed = []
        for frame in sorted(target_positions[side]):
            target = target_positions[side][frame]
            current_reach = target - shoulder_bind
            if current_reach.length < 1e-8:
                continue
            current_reach.normalize()
            transport = bind_reach.rotation_difference(current_reach)
            pole.location = shoulder_bind + transport @ bind_offset
            pole.keyframe_insert(
                "location",
                frame=frame + frame_offset,
                group="VGGT_DLC_RestPole",
            )
            keyed.append(frame)
        set_linear_interpolation(pole)
        report[side] = {
            "pole": pole.name,
            "bind_position": list(pole_bind_position),
            "bind_offset_length": float(bind_offset.length),
            "keyed_frame_count": len(keyed),
            "method": "minimal rotation of bind pole around shoulder-to-hand reach axis",
        }
    scene.frame_set(bind_frame + frame_offset)
    bpy.context.view_layer.update()
    return report


def compact_finger_constraints(
    armature: bpy.types.Object,
) -> dict[str, dict[str, list[float]]]:
    report = {}
    finger_prefixes = ("Index", "Middle", "Ring", "Pinky")
    for side in ("L", "R"):
        for prefix in finger_prefixes:
            for segment in (1, 2, 3):
                name = f"{prefix}Finger{segment}_{side}"
                bone = armature.pose.bones.get(name)
                if bone is None:
                    continue
                if segment == 1:
                    x_limits = (
                        (math.radians(0.0), math.radians(30.0))
                        if side == "L"
                        else (math.radians(-30.0), math.radians(0.0))
                    )
                    limits = (
                        x_limits[0],
                        x_limits[1],
                        math.radians(-4.0),
                        math.radians(4.0),
                        math.radians(-80.0),
                        math.radians(25.0),
                    )
                else:
                    limits = (
                        math.radians(-12.0),
                        math.radians(12.0),
                        math.radians(-5.0),
                        math.radians(5.0),
                        math.radians(-85.0),
                        math.radians(25.0),
                    )
                constraint = bone.constraints.get("MGF_DLCCompactRotation")
                if constraint is None:
                    constraint = bone.constraints.new("LIMIT_ROTATION")
                    constraint.name = "MGF_DLCCompactRotation"
                constraint.owner_space = "LOCAL"
                constraint.use_limit_x = True
                constraint.use_limit_y = True
                constraint.use_limit_z = True
                (
                    constraint.min_x,
                    constraint.max_x,
                    constraint.min_y,
                    constraint.max_y,
                    constraint.min_z,
                    constraint.max_z,
                ) = limits
                bone.use_ik_limit_x = True
                bone.use_ik_limit_y = True
                bone.use_ik_limit_z = True
                bone.ik_min_x, bone.ik_max_x = limits[0], limits[1]
                bone.ik_min_y, bone.ik_max_y = limits[2], limits[3]
                bone.ik_min_z, bone.ik_max_z = limits[4], limits[5]
                report[name] = {
                    "x_deg": (
                        [0.0, 30.0]
                        if segment == 1 and side == "L"
                        else [-30.0, 0.0]
                        if segment == 1
                        else [-12.0, 12.0]
                    ),
                    "y_deg": [-4.0, 4.0] if segment == 1 else [-5.0, 5.0],
                    "z_deg": [-80.0, 25.0] if segment == 1 else [-85.0, 25.0],
                }
    bpy.context.view_layer.update()
    return report


def compensate_first_finger_lengths(
    armature: bpy.types.Object,
    factor: float,
) -> dict[str, object]:
    if factor <= 0.0:
        raise ValueError("finger-length-compensation must be positive")
    if abs(factor - 1.0) < 1e-8:
        return {"enabled": False, "factor": factor, "bones": {}}

    report = {}
    for side in ("L", "R"):
        for prefix in ("Index", "Middle", "Ring", "Pinky"):
            name = f"{prefix}Finger1_{side}"
            bone = armature.pose.bones[name]
            before = float(bone.scale.y)
            bone.scale.y = before / factor
            length_lock = bone.constraints.get("MGF_StretchLengthLock")
            if length_lock is not None and length_lock.type == "LIMIT_SCALE":
                length_lock.min_y = float(bone.scale.y)
                length_lock.max_y = float(bone.scale.y)
            report[name] = {
                "scale_y_before": before,
                "scale_y_after": float(bone.scale.y),
                "updated_length_lock": (
                    length_lock.name if length_lock is not None else None
                ),
            }
    bpy.context.view_layer.update()
    return {"enabled": True, "factor": factor, "bones": report}


def forearm_axis_world(armature: bpy.types.Object, side: str) -> Vector:
    bone = armature.pose.bones[f"Wrist_{side.upper()}"]
    head = armature.matrix_world @ bone.matrix.translation
    tail = armature.matrix_world @ (
        bone.matrix @ Vector((0.0, bone.bone.length, 0.0))
    )
    axis = tail - head
    if axis.length < 1e-8:
        raise RuntimeError(f"Degenerate forearm axis for {side}")
    return axis.normalized()


def projected_normal(normal: Vector, axis: Vector) -> Vector | None:
    projected = normal - axis * normal.dot(axis)
    if projected.length < 1e-8:
        return None
    return projected.normalized()


def signed_angle_around_axis(first: Vector, second: Vector, axis: Vector) -> float:
    return math.atan2(axis.dot(first.cross(second)), first.dot(second))


def unwrap_near(angle: float, reference: float) -> float:
    while angle - reference > math.pi:
        angle -= 2.0 * math.pi
    while angle - reference < -math.pi:
        angle += 2.0 * math.pi
    return angle


def interpolated_anchor_offset(
    frame: int,
    anchors: list[tuple[int, float]],
) -> float:
    if not anchors:
        return 0.0
    if frame <= anchors[0][0]:
        return anchors[0][1]
    if frame >= anchors[-1][0]:
        return anchors[-1][1]
    for (left_frame, left_value), (right_frame, right_value) in zip(
        anchors, anchors[1:]
    ):
        if left_frame <= frame <= right_frame:
            alpha = (frame - left_frame) / max(right_frame - left_frame, 1)
            return left_value * (1.0 - alpha) + right_value * alpha
    return 0.0


def median_smooth_scalars(
    values: dict[int, float],
    radius: int,
) -> dict[int, float]:
    if radius <= 0:
        return dict(values)
    frames = sorted(values)
    output = {}
    for frame in frames:
        neighbors = [
            values[other] for other in frames if abs(other - frame) <= radius
        ]
        output[frame] = sorted(neighbors)[len(neighbors) // 2]
    return output


def enforce_roll_continuity_from_bind(
    values: dict[int, float],
    bind_frame: int,
    max_step: float,
) -> dict[int, float]:
    output = dict(values)
    output[bind_frame] = values.get(bind_frame, 0.0)
    for frames in (
        sorted(frame for frame in values if frame > bind_frame),
        sorted((frame for frame in values if frame < bind_frame), reverse=True),
    ):
        previous_frame = bind_frame
        previous_value = output[bind_frame]
        for frame in frames:
            gap = max(1, abs(frame - previous_frame))
            limit = max_step * gap
            value = max(
                previous_value - limit,
                min(previous_value + limit, values[frame]),
            )
            output[frame] = value
            previous_frame = frame
            previous_value = value
    return output


def keyframe_forearm_roll_orientations(
    scene: bpy.types.Scene,
    armature: bpy.types.Object,
    controller: bpy.types.Object,
    side: str,
    descriptors: dict[int, dict[str, object]],
    bind_frame: int,
    frame_offset: int,
    bind_controller_rotation: Matrix,
    bind_forearm_axis: Vector,
    max_roll_deg: float,
    max_step_deg: float,
    principal_prebind_branch: bool,
    roll_offset_anchors: list[tuple[int, float]],
    roll_target_anchors: list[tuple[int, float]],
    roll_median_radius: int,
    mouth_positions: dict[int, Vector],
    controller_positions: dict[int, Vector],
    face_mouth_windows: list[tuple[int, int]],
    face_mouth_left_weight: float,
    face_mouth_blend_frames: int,
    face_mouth_max_step_deg: float,
) -> dict[str, object]:
    bind_normal = descriptors[bind_frame]["rotation"].col[2].normalized()
    roll_by_frame = {bind_frame: 0.0}
    normal_by_frame = {bind_frame: bind_normal.copy()}
    axis_by_frame = {}

    scene.frame_set(bind_frame + frame_offset)
    bpy.context.view_layer.update()
    axis_by_frame[bind_frame] = forearm_axis_world(armature, side)

    def solve_direction(frames: list[int]) -> None:
        previous_frame = bind_frame
        previous_roll = 0.0
        previous_normal = bind_normal.copy()
        previous_axis = axis_by_frame[bind_frame]
        for frame in frames:
            scene.frame_set(frame + frame_offset)
            bpy.context.view_layer.update()
            current_axis = forearm_axis_world(armature, side)
            axis_by_frame[frame] = current_axis

            transported_previous = (
                previous_axis.rotation_difference(current_axis) @ previous_normal
            )
            raw_normal = descriptors[frame]["rotation"].col[2].normalized()
            if raw_normal.dot(transported_previous) < 0.0:
                raw_normal.negate()

            swing = bind_forearm_axis.rotation_difference(current_axis)
            reference_normal = projected_normal(swing @ bind_normal, current_axis)
            desired_normal = projected_normal(raw_normal, current_axis)
            if reference_normal is None or desired_normal is None:
                roll = previous_roll
            else:
                roll = signed_angle_around_axis(
                    reference_normal, desired_normal, current_axis
                )
                if principal_prebind_branch and frame < bind_frame:
                    candidates = [
                        roll + offset * math.pi for offset in range(-2, 3)
                    ]
                    roll = min(candidates, key=abs)
                else:
                    roll = unwrap_near(roll, previous_roll)
                max_roll = math.radians(max_roll_deg)
                roll = max(-max_roll, min(max_roll, roll))
                gap = max(1, abs(frame - previous_frame))
                max_step = math.radians(max_step_deg) * gap
                roll = max(
                    previous_roll - max_step,
                    min(previous_roll + max_step, roll),
                )
            roll_by_frame[frame] = roll
            normal_by_frame[frame] = raw_normal
            previous_frame = frame
            previous_roll = roll
            previous_normal = raw_normal
            previous_axis = current_axis

    forward = sorted(frame for frame in descriptors if frame > bind_frame)
    backward = sorted(
        (frame for frame in descriptors if frame < bind_frame), reverse=True
    )
    solve_direction(forward)
    solve_direction(backward)

    roll_by_frame = median_smooth_scalars(
        roll_by_frame, max(0, roll_median_radius)
    )
    max_roll = math.radians(max_roll_deg)
    for frame in roll_by_frame:
        offset = math.radians(
            interpolated_anchor_offset(frame, roll_offset_anchors)
        )
        roll_by_frame[frame] = max(
            -max_roll, min(max_roll, roll_by_frame[frame] + offset)
        )
    roll_by_frame = enforce_roll_continuity_from_bind(
        roll_by_frame,
        bind_frame,
        math.radians(max_step_deg),
    )
    if roll_target_anchors:
        # Absolute targets are an explicit semantic correction. Apply them
        # after generic continuity filtering so the backward pass from the
        # bind frame cannot pull a grasping palm onto the wrong roll branch.
        target_start = roll_target_anchors[0][0]
        target_end = roll_target_anchors[-1][0]
        for frame in roll_by_frame:
            if target_start <= frame <= target_end:
                roll_by_frame[frame] = math.radians(
                    interpolated_anchor_offset(frame, roll_target_anchors)
                )

    mouth_facing_report = {}
    mouth_facing_rotations = {}
    palm_local_axis = Vector((1.0, 0.0, 0.0))
    world_left = Vector((-1.0, 0.0, 0.0))
    for start, end in face_mouth_windows:
        for frame in sorted(
            frame
            for frame in roll_by_frame
            if start <= frame <= end
            and frame in mouth_positions
            and frame in controller_positions
        ):
            current_axis = axis_by_frame.get(frame)
            if current_axis is None:
                continue
            mouth_direction = mouth_positions[frame] - controller_positions[frame]
            if mouth_direction.length < 1e-8:
                continue
            mouth_unit = mouth_direction.normalized()
            mouth_projected = projected_normal(mouth_unit, current_axis)
            left_projected = projected_normal(world_left, current_axis)
            if mouth_projected is None or left_projected is None:
                continue

            desired_palm = (
                mouth_projected
                + left_projected * max(0.0, face_mouth_left_weight)
            )
            if desired_palm.length < 1e-8:
                continue
            desired_palm.normalize()
            if (
                desired_palm.dot(mouth_unit) <= 0.0
                or desired_palm.dot(world_left) <= 0.0
            ):
                # The equal-angle bisector is the minimax solution for the
                # two sign constraints whenever their projections are not
                # exactly opposite.
                desired_palm = mouth_projected + left_projected
                if desired_palm.length < 1e-8:
                    continue
                desired_palm.normalize()
            alternate = -desired_palm
            desired_score = min(
                desired_palm.dot(mouth_unit),
                desired_palm.dot(world_left),
            )
            alternate_score = min(
                alternate.dot(mouth_unit),
                alternate.dot(world_left),
            )
            if alternate_score > desired_score:
                desired_palm = alternate

            swing = bind_forearm_axis.rotation_difference(current_axis)
            reference_rotation = (
                swing @ bind_controller_rotation.to_quaternion()
            ).normalized()
            reference_palm = projected_normal(
                reference_rotation @ palm_local_axis, current_axis
            )
            if reference_palm is None:
                continue
            semantic_roll = signed_angle_around_axis(
                reference_palm, desired_palm, current_axis
            )
            blend = 1.0
            if face_mouth_blend_frames > 0:
                blend = min(
                    1.0,
                    (frame - start + 1) / face_mouth_blend_frames,
                    (end - frame + 1) / face_mouth_blend_frames,
                )
            semantic_roll = unwrap_near(semantic_roll, roll_by_frame[frame])
            roll_by_frame[frame] = semantic_roll

            palm_z = desired_palm.cross(current_axis)
            if palm_z.length < 1e-8:
                continue
            palm_z.normalize()
            semantic_rotation = Matrix(
                (desired_palm, current_axis, palm_z)
            ).transposed().to_quaternion()
            base_rotation = (
                Quaternion(current_axis, roll_by_frame[frame])
                @ reference_rotation
            ).normalized()
            if base_rotation.dot(semantic_rotation) < 0.0:
                semantic_rotation.negate()
            solved_rotation = base_rotation.slerp(semantic_rotation, blend)
            mouth_facing_rotations[frame] = solved_rotation
            solved_palm = (solved_rotation @ palm_local_axis).normalized()
            mouth_facing_report[str(frame)] = {
                "blend": blend,
                "equivalent_roll_deg": math.degrees(semantic_roll),
                "palm_dot_mouth": solved_palm.dot(mouth_direction.normalized()),
                "palm_dot_world_left": solved_palm.dot(world_left),
            }

    previous_frame = None
    previous_rotation = None
    for frame in sorted(mouth_facing_rotations):
        rotation = mouth_facing_rotations[frame]
        if previous_rotation is not None:
            if previous_rotation.dot(rotation) < 0.0:
                rotation.negate()
            dot = max(-1.0, min(1.0, previous_rotation.dot(rotation)))
            angle = 2.0 * math.acos(dot)
            gap = max(1, frame - previous_frame)
            max_angle = math.radians(face_mouth_max_step_deg) * gap
            if angle > max_angle and angle > 1e-8:
                rotation = previous_rotation.slerp(
                    rotation, max_angle / angle
                )
                mouth_facing_rotations[frame] = rotation
        mouth_direction = mouth_positions[frame] - controller_positions[frame]
        solved_palm = (rotation @ palm_local_axis).normalized()
        mouth_facing_report[str(frame)].update(
            {
                "palm_dot_mouth": solved_palm.dot(
                    mouth_direction.normalized()
                ),
                "palm_dot_world_left": solved_palm.dot(world_left),
            }
        )
        previous_frame = frame
        previous_rotation = rotation

    previous_quaternion = None
    for frame in sorted(descriptors):
        scene.frame_set(frame + frame_offset)
        bpy.context.view_layer.update()
        current_axis = axis_by_frame.get(frame) or forearm_axis_world(armature, side)
        swing = bind_forearm_axis.rotation_difference(current_axis)
        semantic_rotation = mouth_facing_rotations.get(frame)
        if semantic_rotation is None:
            roll = Quaternion(current_axis, roll_by_frame.get(frame, 0.0))
            quaternion = (
                roll @ swing @ bind_controller_rotation.to_quaternion()
            ).normalized()
        else:
            quaternion = semantic_rotation.copy()
        if previous_quaternion is not None and previous_quaternion.dot(quaternion) < 0.0:
            quaternion.negate()
        controller.rotation_quaternion = quaternion
        controller.keyframe_insert(
            "rotation_quaternion",
            frame=frame + frame_offset,
            group="VGGT_DLC_ForearmRoll",
        )
        previous_quaternion = quaternion

    values = [math.degrees(value) for value in roll_by_frame.values()]
    saturation_threshold = max_roll_deg - 0.5
    saturated_frames = sorted(
        frame
        for frame, value in roll_by_frame.items()
        if abs(math.degrees(value)) >= saturation_threshold
    )
    saturation_spans = []
    for frame in saturated_frames:
        if not saturation_spans or frame > saturation_spans[-1]["end"] + 1:
            saturation_spans.append({"start": frame, "end": frame, "count": 1})
        else:
            saturation_spans[-1]["end"] = frame
            saturation_spans[-1]["count"] += 1
    return {
        "roll_min_deg": min(values),
        "roll_max_deg": max(values),
        "roll_abs_p95_deg": sorted(abs(value) for value in values)[
            int(0.95 * (len(values) - 1))
        ],
        "saturated_frame_count": len(saturated_frames),
        "saturated_frame_spans": saturation_spans,
        "principal_prebind_branch": principal_prebind_branch,
        "roll_offset_anchors": roll_offset_anchors,
        "roll_target_anchors": roll_target_anchors,
        "face_mouth_windows": face_mouth_windows,
        "face_mouth_left_weight": face_mouth_left_weight,
        "face_mouth_blend_frames": face_mouth_blend_frames,
        "face_mouth_max_step_deg": face_mouth_max_step_deg,
        "face_mouth_samples": {
            frame: values
            for frame, values in mouth_facing_report.items()
            if int(frame) % 5 == 0
            or any(int(frame) in (start, end) for start, end in face_mouth_windows)
        },
        "roll_median_radius": roll_median_radius,
        "sampled_roll_deg": {
            str(frame): math.degrees(roll_by_frame[frame])
            for frame in sorted(roll_by_frame)
            if frame % 5 == 0 or any(frame == anchor[0] for anchor in roll_offset_anchors)
        },
    }


def fingertip_pair_distance(
    points: dict[str, Vector],
    side: str,
) -> float | None:
    tips = [
        points[f"{side}{finger}_tip_A"]
        for finger in range(1, 5)
        if f"{side}{finger}_tip_A" in points
    ]
    if len(tips) < 3:
        return None
    distances = [
        (first - second).length
        for index, first in enumerate(tips)
        for second in tips[index + 1 :]
    ]
    distances.sort()
    middle = len(distances) // 2
    return (
        distances[middle]
        if len(distances) % 2
        else (distances[middle - 1] + distances[middle]) * 0.5
    )


def interpolate_scalar_curve(
    values: dict[int, float],
    frames: list[int],
) -> dict[int, float]:
    valid_frames = sorted(values)
    if not valid_frames:
        return {}
    output = {}
    for frame in frames:
        if frame in values:
            output[frame] = values[frame]
            continue
        left = max((item for item in valid_frames if item < frame), default=None)
        right = min((item for item in valid_frames if item > frame), default=None)
        if left is None:
            output[frame] = values[right]
        elif right is None:
            output[frame] = values[left]
        else:
            alpha = (frame - left) / (right - left)
            output[frame] = values[left] * (1.0 - alpha) + values[right] * alpha
    return output


def scalar_percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        raise ValueError("Cannot compute a percentile from no finite values")
    position = min(max(fraction, 0.0), 1.0) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    alpha = position - lower
    return ordered[lower] * (1.0 - alpha) + ordered[upper] * alpha


def wrapped_angle_degrees(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def limit_scalar_steps(
    values: dict[int, float],
    max_step: float,
) -> dict[int, float]:
    if max_step <= 0.0 or not values:
        return dict(values)
    output = {}
    previous = None
    previous_frame = None
    for frame in sorted(values):
        value = values[frame]
        if previous is not None and previous_frame is not None:
            allowed = max_step * max(1, frame - previous_frame)
            value = min(max(value, previous - allowed), previous + allowed)
        output[frame] = value
        previous = value
        previous_frame = frame
    return output


def normalized_finger_bend(
    points: dict[str, Vector],
    side: str,
    finger_index: int,
) -> float | None:
    names = [
        f"{side}{finger_index}_joint_A",
        f"{side}{finger_index}_mid_A",
        f"{side}{finger_index}_tip_A",
    ]
    if any(name not in points for name in names):
        return None
    joint, middle, tip = (points[name] for name in names)
    proximal = middle - joint
    distal = tip - middle
    if proximal.length < 1e-8 or distal.length < 1e-8:
        return None
    # Unit vectors remove the 1.3-1.8x Anipose/model finger-length mismatch.
    proximal.normalize()
    distal.normalize()
    return math.degrees(proximal.angle(distal))


def normalized_finger_spread(
    points: dict[str, Vector],
    side: str,
    finger_index: int,
) -> float | None:
    palm_name = f"{side}_mid_A"
    joint_names = [
        f"{side}{index}_joint_A" for index in range(1, 5)
    ]
    middle_name = f"{side}{finger_index}_mid_A"
    if (
        palm_name not in points
        or middle_name not in points
        or any(name not in points for name in joint_names)
    ):
        return None
    palm = points[palm_name]
    joints = [points[name] for name in joint_names]
    across = joints[-1] - joints[0]
    if across.length < 1e-8:
        return None
    across.normalize()
    forward = sum(joints, Vector()) / len(joints) - palm
    forward -= across * forward.dot(across)
    if forward.length < 1e-8:
        return None
    forward.normalize()
    proximal = points[middle_name] - joints[finger_index - 1]
    if proximal.length < 1e-8:
        return None
    proximal.normalize()
    return math.degrees(
        math.atan2(proximal.dot(across), proximal.dot(forward))
    )


def model_hand_spread_frame(
    armature: bpy.types.Object,
    side: str,
) -> tuple[Vector, Vector]:
    roots = []
    directions = []
    for finger_name in ("Index", "Middle", "Ring", "Pinky"):
        bone = armature.pose.bones[f"{finger_name}Finger1_{side}"]
        roots.append(armature.matrix_world @ bone.head)
        direction = armature.matrix_world.to_3x3() @ (bone.tail - bone.head)
        if direction.length > 1e-8:
            directions.append(direction.normalized())
    across = roots[-1] - roots[0]
    if across.length < 1e-8:
        raise RuntimeError(f"Degenerate model hand spread axis for {side}")
    across.normalize()
    forward = sum(directions, Vector()) / max(len(directions), 1)
    forward -= across * forward.dot(across)
    if forward.length < 1e-8:
        raise RuntimeError(f"Degenerate model hand forward axis for {side}")
    forward.normalize()
    return across, forward


def model_finger_spread_angle(
    armature: bpy.types.Object,
    bone: bpy.types.PoseBone,
    across: Vector,
    forward: Vector,
) -> float:
    direction = armature.matrix_world.to_3x3() @ (bone.tail - bone.head)
    if direction.length < 1e-8:
        return 0.0
    direction.normalize()
    return math.degrees(
        math.atan2(direction.dot(across), direction.dot(forward))
    )


def calibrate_spread_controls(
    scene: bpy.types.Scene,
    armature: bpy.types.Object,
    bind_frame: int,
    frame_offset: int,
) -> dict[tuple[str, str], dict[str, float]]:
    scene.frame_set(bind_frame + frame_offset)
    bpy.context.view_layer.update()
    result = {}
    epsilon = math.radians(2.0)
    for side in ("L", "R"):
        across, forward = model_hand_spread_frame(armature, side)
        for finger_name in ("Index", "Middle", "Ring", "Pinky"):
            bone = armature.pose.bones[f"{finger_name}Finger1_{side}"]
            constraint = bone.constraints.get("MGF_DLCCompactRotation")
            lower = (
                float(constraint.min_x)
                if constraint is not None and constraint.use_limit_x
                else -math.pi
            )
            upper = (
                float(constraint.max_x)
                if constraint is not None and constraint.use_limit_x
                else math.pi
            )
            original = float(bone.rotation_euler.x)
            neutral = min(max(original, lower), upper)
            bone.rotation_euler.x = neutral
            bpy.context.view_layer.update()
            base_angle = model_finger_spread_angle(
                armature, bone, across, forward
            )
            candidates = []
            for delta in (epsilon, -epsilon):
                perturbed = min(max(neutral + delta, lower), upper)
                actual_delta = perturbed - neutral
                if abs(actual_delta) < 1e-8:
                    continue
                bone.rotation_euler.x = perturbed
                bpy.context.view_layer.update()
                angle = model_finger_spread_angle(
                    armature, bone, across, forward
                )
                derivative = wrapped_angle_degrees(angle - base_angle) / math.degrees(
                    actual_delta
                )
                candidates.append(derivative)
            bone.rotation_euler.x = original
            bpy.context.view_layer.update()
            sensitivity = max(candidates, key=abs) if candidates else 1.0
            if abs(sensitivity) < 0.05:
                sensitivity = 1.0
            result[(side.lower(), finger_name)] = {
                "base_x": neutral,
                "min_x": lower,
                "max_x": upper,
                "spread_sensitivity": sensitivity,
            }
    scene.frame_set(bind_frame + frame_offset)
    bpy.context.view_layer.update()
    return result


def drive_normalized_finger_pose(
    scene: bpy.types.Scene,
    armature: bpy.types.Object,
    controllers: dict[str, bpy.types.Object],
    hand_keypoints: dict[str, dict[int, dict[str, Vector]]],
    bind_frame: int,
    frame_offset: int,
    source_frame_start: int,
    source_frame_end: int,
    curl_max_deg: float,
    spread_gain: float,
    spread_max_deg: float,
    smooth_radius: int,
    max_step_deg: float,
    quantile_low: float,
    quantile_high: float,
) -> dict[str, object]:
    if curl_max_deg < 0.0 or spread_max_deg < 0.0:
        raise ValueError("Normalized finger ranges must be non-negative")
    if not 0.0 <= quantile_low < quantile_high <= 1.0:
        raise ValueError("Normalized finger quantiles must satisfy 0 <= low < high <= 1")
    source_frames = list(range(source_frame_start, source_frame_end + 1))
    finger_names = ("Index", "Middle", "Ring", "Pinky")
    spread_controls = calibrate_spread_controls(
        scene, armature, bind_frame, frame_offset
    )
    report: dict[str, object] = {
        "method": (
            "unit Anipose segment directions; robust per-trial bend "
            "normalization; model-native curl drivers and constrained spread"
        ),
        "curl_max_deg": curl_max_deg,
        "spread_gain": spread_gain,
        "spread_max_deg": spread_max_deg,
        "smooth_radius": smooth_radius,
        "max_step_deg": max_step_deg,
        "quantiles": [quantile_low, quantile_high],
        "sides": {},
    }

    for side in SIDE_CONFIG:
        suffix = side.upper()
        controller = controllers[side]
        side_report = {}
        for finger_index, finger_name in enumerate(finger_names, start=1):
            raw_bend = {}
            raw_spread = {}
            for frame, points in hand_keypoints[side].items():
                bend = normalized_finger_bend(
                    points,
                    side,
                    finger_index,
                )
                spread = normalized_finger_spread(
                    points,
                    side,
                    finger_index,
                )
                if bend is not None and math.isfinite(bend):
                    raw_bend[frame] = bend
                if spread is not None and math.isfinite(spread):
                    raw_spread[frame] = spread
            if not raw_bend or not raw_spread:
                continue

            bend_low = scalar_percentile(list(raw_bend.values()), quantile_low)
            bend_high = scalar_percentile(list(raw_bend.values()), quantile_high)
            bend_width = max(bend_high - bend_low, 5.0)
            bend_curve = interpolate_scalar_curve(raw_bend, source_frames)
            bend_curve = median_smooth_scalars(
                bend_curve, max(0, smooth_radius)
            )
            curl_curve = {
                frame: curl_max_deg
                * min(max((value - bend_low) / bend_width, 0.0), 1.0)
                for frame, value in bend_curve.items()
            }
            curl_curve = limit_scalar_steps(curl_curve, max_step_deg)
            property_name = f"{suffix}_{finger_name}CurlDeg"
            if property_name not in controller:
                raise RuntimeError(
                    f"{controller.name} is missing {property_name}"
                )
            for frame, value in curl_curve.items():
                controller[property_name] = value
                controller.keyframe_insert(
                    f'["{property_name}"]',
                    frame=frame + frame_offset,
                    group=f"MGF_NormalizedFinger_{suffix}",
                )

            spread_center = scalar_percentile(list(raw_spread.values()), 0.5)
            spread_deviation = {
                frame: wrapped_angle_degrees(value - spread_center)
                for frame, value in raw_spread.items()
            }
            spread_curve = interpolate_scalar_curve(
                spread_deviation, source_frames
            )
            spread_curve = median_smooth_scalars(
                spread_curve, max(0, smooth_radius)
            )
            spread_curve = {
                frame: min(
                    max(value * spread_gain, -spread_max_deg),
                    spread_max_deg,
                )
                for frame, value in spread_curve.items()
            }
            spread_curve = limit_scalar_steps(spread_curve, max_step_deg)
            control = spread_controls[(side, finger_name)]
            bone = armature.pose.bones[f"{finger_name}Finger1_{suffix}"]
            bone.rotation_mode = "XYZ"
            sensitivity = control["spread_sensitivity"]
            keyed_spread = {}
            for frame, desired_delta_deg in spread_curve.items():
                input_delta = math.radians(desired_delta_deg / sensitivity)
                target_x = min(
                    max(control["base_x"] + input_delta, control["min_x"]),
                    control["max_x"],
                )
                bone.rotation_euler.x = target_x
                bone.keyframe_insert(
                    "rotation_euler",
                    index=0,
                    frame=frame + frame_offset,
                    group=f"MGF_NormalizedFinger_{suffix}",
                )
                keyed_spread[frame] = math.degrees(target_x)

            side_report[finger_name] = {
                "valid_bend_frames": len(raw_bend),
                "valid_spread_frames": len(raw_spread),
                "bend_low_deg": bend_low,
                "bend_high_deg": bend_high,
                "curl_min_deg": min(curl_curve.values()),
                "curl_max_deg": max(curl_curve.values()),
                "spread_source_center_deg": spread_center,
                "spread_sensitivity": sensitivity,
                "spread_x_min_deg": min(keyed_spread.values()),
                "spread_x_max_deg": max(keyed_spread.values()),
            }
        report["sides"][side] = side_report

    for controller in controllers.values():
        set_linear_interpolation(controller)
    set_linear_interpolation(armature)
    scene.frame_set(bind_frame + frame_offset)
    bpy.context.view_layer.update()
    return report


def drive_active_finger_grip(
    armature: bpy.types.Object,
    controllers: dict[str, bpy.types.Object],
    hand_keypoints: dict[str, dict[int, dict[str, Vector]]],
    active_windows: dict[str, list[tuple[int, int]]],
    frame_offset: int,
    open_distance: float,
    closed_distance: float,
    smooth_radius: int,
    open_curl_deg: float,
    closed_curl_deg: float,
    ring_open_curl_deg: float,
    ring_closed_curl_deg: float,
    open_spread_deg: float,
    closed_spread_deg: float,
    boundary_blend_frames: int,
) -> dict[str, object]:
    if open_distance <= closed_distance:
        raise ValueError("grip-open-distance must exceed grip-closed-distance")
    if closed_curl_deg < open_curl_deg:
        raise ValueError("grip-closed-curl-deg must be at least open curl")
    if ring_closed_curl_deg < ring_open_curl_deg:
        raise ValueError("ring closed curl must be at least ring open curl")
    if boundary_blend_frames < 1:
        raise ValueError("grip-boundary-blend-frames must be at least 1")

    finger_names = ("Index", "Middle", "Ring", "Pinky")
    spread_signs = (-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0)
    report = {}

    for side, windows in active_windows.items():
        if not windows:
            continue
        source_frames = sorted(
            {
                frame
                for start, end in windows
                for frame in range(start, end + 1)
            }
        )
        raw_distances = {}
        for frame in source_frames:
            distance = fingertip_pair_distance(
                hand_keypoints[side].get(frame, {}), side
            )
            if distance is not None and math.isfinite(distance):
                raw_distances[frame] = distance
        distances = interpolate_scalar_curve(raw_distances, source_frames)
        distances = median_smooth_scalars(distances, max(0, smooth_radius))
        grip = {
            frame: max(
                0.0,
                min(
                    1.0,
                    (open_distance - distance)
                    / (open_distance - closed_distance),
                ),
            )
            for frame, distance in distances.items()
        }

        suffix = side.upper()
        controller = controllers[side]
        keyed_bones = []
        keyed_properties = []
        base_xy = {}
        base_curl = {}
        for finger_name in finger_names:
            property_name = f"{suffix}_{finger_name}CurlDeg"
            if property_name not in controller:
                raise RuntimeError(
                    f"{controller.name} is missing driven curl property "
                    f"{property_name}"
                )
            keyed_properties.append(property_name)
            base_curl[property_name] = float(controller[property_name])
            for segment in (1, 2, 3):
                bone_name = f"{finger_name}Finger{segment}_{suffix}"
                bone = armature.pose.bones[bone_name]
                base_xy[bone_name] = (
                    float(bone.rotation_euler.x),
                    float(bone.rotation_euler.y),
                )

        boundary_frames = sorted(
            {
                boundary
                for start, end in windows
                for boundary in (
                    start - boundary_blend_frames,
                    end + boundary_blend_frames,
                )
            }
        )
        for finger_index, finger_name in enumerate(finger_names):
            property_name = f"{suffix}_{finger_name}CurlDeg"
            for frame in boundary_frames:
                controller[property_name] = base_curl[property_name]
                controller.keyframe_insert(
                    f'["{property_name}"]',
                    frame=frame + frame_offset,
                    group=f"VGGT_DLC_Grip_{suffix}",
                )
            for frame in source_frames:
                amount = grip[frame]
                if finger_name == "Ring":
                    curl_degrees = (
                        ring_open_curl_deg * (1.0 - amount)
                        + ring_closed_curl_deg * amount
                    )
                else:
                    curl_degrees = (
                        open_curl_deg * (1.0 - amount)
                        + closed_curl_deg * amount
                    )
                controller[property_name] = curl_degrees
                controller.keyframe_insert(
                    f'["{property_name}"]',
                    frame=frame + frame_offset,
                    group=f"VGGT_DLC_Grip_{suffix}",
                )

            for segment in (1, 2, 3):
                bone_name = f"{finger_name}Finger{segment}_{suffix}"
                bone = armature.pose.bones[bone_name]
                bone.rotation_mode = "XYZ"
                keyed_bones.append(bone_name)
                for frame in boundary_frames:
                    bone.rotation_euler.x = base_xy[bone_name][0]
                    bone.rotation_euler.y = base_xy[bone_name][1]
                    bone.keyframe_insert(
                        "rotation_euler",
                        index=0,
                        frame=frame + frame_offset,
                        group=f"VGGT_DLC_Grip_{suffix}",
                    )
                    bone.keyframe_insert(
                        "rotation_euler",
                        index=1,
                        frame=frame + frame_offset,
                        group=f"VGGT_DLC_Grip_{suffix}",
                    )
                for frame in source_frames:
                    amount = grip[frame]
                    if segment == 1:
                        spread = (
                            open_spread_deg * (1.0 - amount)
                            + closed_spread_deg * amount
                        )
                        x_degrees = spread_signs[finger_index] * spread
                    else:
                        x_degrees = 0.0
                    bone.rotation_euler.x = math.radians(x_degrees)
                    bone.rotation_euler.y = 0.0
                    bone.keyframe_insert(
                        "rotation_euler",
                        index=0,
                        frame=frame + frame_offset,
                        group=f"VGGT_DLC_Grip_{suffix}",
                    )
                    bone.keyframe_insert(
                        "rotation_euler",
                        index=1,
                        frame=frame + frame_offset,
                        group=f"VGGT_DLC_Grip_{suffix}",
                    )

        report[side] = {
            "windows": windows,
            "keyed_bones": keyed_bones,
            "keyed_controller_properties": keyed_properties,
            "boundary_frames": boundary_frames,
            "boundary_blend_frames": boundary_blend_frames,
            "open_distance": open_distance,
            "closed_distance": closed_distance,
            "open_curl_deg": open_curl_deg,
            "closed_curl_deg": closed_curl_deg,
            "ring_open_curl_deg": ring_open_curl_deg,
            "ring_closed_curl_deg": ring_closed_curl_deg,
            "open_spread_deg": open_spread_deg,
            "closed_spread_deg": closed_spread_deg,
            "distance_min": min(distances.values()),
            "distance_max": max(distances.values()),
            "grip_min": min(grip.values()),
            "grip_max": max(grip.values()),
            "sampled_grip": {
                str(frame): grip[frame]
                for frame in source_frames
                if frame % 5 == 0 or any(
                    frame in (start, end) for start, end in windows
                )
            },
        }
    return report


def set_linear_interpolation(obj: bpy.types.Object) -> None:
    if obj.animation_data is None or obj.animation_data.action is None:
        return
    action = obj.animation_data.action
    fcurves = getattr(action, "fcurves", None)
    if fcurves is None:
        # Blender 4.4+ layered actions do not expose the legacy collection.
        # Default Bezier interpolation is suitable for this first-pass motion.
        return
    for fcurve in fcurves:
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = "BEZIER"
            keyframe.handle_left_type = "AUTO_CLAMPED"
            keyframe.handle_right_type = "AUTO_CLAMPED"


def keypoint_names(fieldnames: list[str]) -> list[str]:
    return [
        field[:-2]
        for field in fieldnames
        if field.endswith("_x")
        and f"{field[:-2]}_y" in fieldnames
        and f"{field[:-2]}_z" in fieldnames
    ]


def marker_color(name: str) -> tuple[float, float, float, float]:
    if name.startswith("l"):
        return SIDE_CONFIG["l"]["color"]
    if name.startswith("r"):
        return SIDE_CONFIG["r"]["color"]
    if name in {"snout_A", "mouth_tip_A"}:
        return (0.2, 1.0, 0.25, 1.0)
    return (1.0, 0.85, 0.1, 1.0)


def marker_size(name: str) -> float:
    if name in {"l_mid_A", "r_mid_A", "snout_A", "mouth_tip_A", "spout_tip_A"}:
        return 0.9
    return 0.42


def create_keypoint_markers(
    path: Path,
    frame_offset: int,
) -> tuple[int, int]:
    old_collection = bpy.data.collections.get("MGF_VGGTDLC_Keypoints")
    if old_collection is not None:
        for obj in list(old_collection.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(old_collection)

    collection = bpy.data.collections.new("MGF_VGGTDLC_Keypoints")
    bpy.context.scene.collection.children.link(collection)
    markers: dict[str, bpy.types.Object] = {}
    keyed = 0
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        names = keypoint_names(reader.fieldnames or [])
        for name in names:
            marker = bpy.data.objects.new(f"DLC3D_{name}", None)
            marker.empty_display_type = "SPHERE"
            marker.empty_display_size = marker_size(name)
            marker.color = marker_color(name)
            marker.show_in_front = True
            marker.hide_render = True
            collection.objects.link(marker)
            markers[name] = marker

        for row_index, row in enumerate(reader):
            source_frame = int(float(row.get("fnum", row_index)))
            blender_frame = source_frame + frame_offset
            for name, marker in markers.items():
                values = [finite_float(row.get(f"{name}_{axis}")) for axis in "xyz"]
                score = finite_float(row.get(f"{name}_score"))
                ncams = finite_float(row.get(f"{name}_ncams"))
                if (
                    any(value is None for value in values)
                    or (score is not None and score < 0.5)
                    or (ncams is not None and ncams < 2.0)
                ):
                    continue
                marker.location = Vector(values)
                marker.keyframe_insert("location", frame=blender_frame)
                keyed += 1
    return len(markers), keyed


def main() -> None:
    args = parse_args()
    active_windows = parse_active_windows(args.active_window)
    roll_anchors = parse_roll_anchors(args.palm_roll_offset_anchor)
    roll_targets = parse_roll_anchors(args.palm_roll_target_anchor)
    face_mouth_windows = parse_active_windows(args.palm_face_mouth_window)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report_path = args.report or args.output.with_suffix(".json")
    source_blend = bpy.data.filepath

    controllers = {}
    for side, config in SIDE_CONFIG.items():
        controller = bpy.data.objects.get(config["controller"])
        if controller is None:
            raise RuntimeError(f"Missing controller {config['controller']}")
        if controller.parent is not None:
            raise RuntimeError(
                f"{controller.name} has a parent; world-space assignment needs review."
            )
        controllers[side] = controller
    armature = bpy.data.objects.get("DeformationSystem")
    if armature is None or armature.type != "ARMATURE":
        raise RuntimeError("Missing DeformationSystem armature")
    hand_ik_constraints = find_hand_ik_constraints(armature)

    raw_descriptors = read_descriptors(args.descriptors)
    smoothed = smooth_descriptors(raw_descriptors, max(0, args.smooth_radius))
    gated, rejected = gate_temporal_outliers(
        smoothed,
        max_position_jump=args.max_position_jump,
        max_rotation_jump_deg=args.max_rotation_jump_deg,
    )
    # Select the bind from observations that actually survive temporal gating.
    # Otherwise a geometrically close but discontinuous frame can abort an
    # otherwise usable trial after it has already been ranked first.
    bind_frame, bind_candidates = select_bind_frame(
        gated, controllers, args.bind_frame
    )

    bind_controller_poses = {
        side: controller_world_pose(controller)
        for side, controller in controllers.items()
    }
    pole_mode = "remove" if args.remove_elbow_poles else args.elbow_pole_mode
    pole_report = {}
    if pole_mode == "remove":
        pole_report = remove_elbow_poles(hand_ik_constraints)
    compact_report = {}
    if args.compact_fingers:
        compact_report = compact_finger_constraints(armature)
    finger_length_report = compensate_first_finger_lengths(
        armature, args.finger_length_compensation
    )
    bind_forearm_axes = {
        side: forearm_axis_world(armature, side) for side in SIDE_CONFIG
    }

    hand_keypoints, mouth_positions = read_hand_keypoints(args.keypoints)
    source_frame_start, source_frame_end = source_frame_range(args.keypoints)
    cup = bpy.data.objects.get(args.cup_object)
    cup_exclusion_windows = (
        {side: [] for side in SIDE_CONFIG}
        if args.allow_cup_correction_in_active_window
        else active_windows
    )
    cup_pushes, cup_report = cup_front_pushes(
        gated,
        hand_keypoints,
        mouth_positions,
        cup,
        clearance=args.cup_front_clearance,
        max_push=args.cup_front_max_push,
        xz_margin=args.cup_xz_margin,
        smooth_radius=max(0, args.smooth_radius),
        rim_clearance=args.cup_rim_clearance,
        rim_max_push=args.cup_rim_max_push,
        resting_mouth_distance=args.resting_mouth_distance,
        active_windows=cup_exclusion_windows,
    )
    cup_report["correction_allowed_in_active_window"] = (
        args.allow_cup_correction_in_active_window
    )
    bind_offsets = {}
    for side in SIDE_CONFIG:
        bind_palm = gated[side][bind_frame]
        controller_position, controller_rotation = bind_controller_poses[side]
        local_position = (
            bind_palm["rotation"].transposed()
            @ (controller_position - bind_palm["position"])
        )
        local_rotation = (
            bind_palm["rotation"].transposed() @ controller_rotation
        )
        validate_rotation(
            local_rotation, f"{side} bind-frame controller rotation offset"
        )
        bind_offsets[side] = {
            "local_position": list(local_position),
            "local_rotation": [list(row) for row in local_rotation],
        }

    for controller in controllers.values():
        clear_object_animation(controller)
        controller["vggt_dlc_bind_frame"] = bind_frame
        controller["vggt_dlc_descriptors"] = str(args.descriptors)

    keyed_frames = {}
    palm_roll_report = {}
    target_positions: dict[str, dict[int, Vector]] = {
        side: {} for side in SIDE_CONFIG
    }
    if args.palm_orientation_mode == "forearm-roll":
        for side, controller in controllers.items():
            keyed_frames[side] = []
            for source_frame in sorted(gated[side]):
                blender_frame = source_frame + args.frame_offset
                target = controller_target_position(
                    descriptor=gated[side][source_frame],
                    bind_descriptor=gated[side][bind_frame],
                    bind_controller_position=bind_controller_poses[side][0],
                )
                target += cup_pushes[side].get(
                    source_frame, Vector((0.0, 0.0, 0.0))
                )
                target_positions[side][source_frame] = target.copy()
                controller.location = target
                controller.keyframe_insert(
                    "location", frame=blender_frame, group="VGGT_DLC_Hand"
                )
                keyed_frames[side].append(source_frame)
            set_linear_interpolation(controller)

        if pole_mode == "rest-prior":
            pole_report = dynamic_rest_prior_poles(
                scene=bpy.context.scene,
                armature=armature,
                controllers=controllers,
                constraints=hand_ik_constraints,
                target_positions=target_positions,
                bind_frame=bind_frame,
                frame_offset=args.frame_offset,
            )

        # Location IK is evaluated first. Palm orientation then receives only
        # the roll around the evaluated lower-arm Y axis.
        scene = bpy.context.scene
        for side, controller in controllers.items():
            palm_roll_report[side] = keyframe_forearm_roll_orientations(
                scene=scene,
                armature=armature,
                controller=controller,
                side=side,
                descriptors=gated[side],
                bind_frame=bind_frame,
                frame_offset=args.frame_offset,
                bind_controller_rotation=bind_controller_poses[side][1],
                bind_forearm_axis=bind_forearm_axes[side],
                max_roll_deg=args.max_palm_roll_deg,
                max_step_deg=args.max_palm_roll_step_deg,
                principal_prebind_branch=args.principal_prebind_palm_branch,
                roll_offset_anchors=roll_anchors[side],
                roll_target_anchors=roll_targets[side],
                roll_median_radius=args.palm_roll_median_radius,
                mouth_positions=mouth_positions,
                controller_positions=target_positions[side],
                face_mouth_windows=face_mouth_windows[side],
                face_mouth_left_weight=args.palm_face_mouth_left_weight,
                face_mouth_blend_frames=args.palm_face_mouth_blend_frames,
                face_mouth_max_step_deg=args.palm_face_mouth_max_step_deg,
            )
            set_linear_interpolation(controller)
    else:
        for side, controller in controllers.items():
            previous_quaternion = None
            keyed_frames[side] = []
            for source_frame in sorted(gated[side]):
                blender_frame = source_frame + args.frame_offset
                previous_quaternion = keyframe_controller(
                    controller=controller,
                    source_frame=source_frame,
                    blender_frame=blender_frame,
                    descriptor=gated[side][source_frame],
                    bind_descriptor=gated[side][bind_frame],
                    bind_controller_position=bind_controller_poses[side][0],
                    bind_controller_rotation=bind_controller_poses[side][1],
                    previous_quaternion=previous_quaternion,
                )
                keyed_frames[side].append(source_frame)
            set_linear_interpolation(controller)

    if args.drive_finger_grip and args.drive_normalized_fingers:
        raise ValueError(
            "--drive-finger-grip and --drive-normalized-fingers are mutually exclusive"
        )
    normalized_finger_report = {}
    if args.drive_normalized_fingers:
        normalized_finger_report = drive_normalized_finger_pose(
            scene=bpy.context.scene,
            armature=armature,
            controllers=controllers,
            hand_keypoints=hand_keypoints,
            bind_frame=bind_frame,
            frame_offset=args.frame_offset,
            source_frame_start=source_frame_start,
            source_frame_end=source_frame_end,
            curl_max_deg=args.normalized_finger_curl_max_deg,
            spread_gain=args.normalized_finger_spread_gain,
            spread_max_deg=args.normalized_finger_spread_max_deg,
            smooth_radius=args.normalized_finger_smooth_radius,
            max_step_deg=args.normalized_finger_max_step_deg,
            quantile_low=args.normalized_finger_quantile_low,
            quantile_high=args.normalized_finger_quantile_high,
        )

    grip_report = {}
    if args.drive_finger_grip:
        grip_report = drive_active_finger_grip(
            armature=armature,
            controllers=controllers,
            hand_keypoints=hand_keypoints,
            active_windows=active_windows,
            frame_offset=args.frame_offset,
            open_distance=args.grip_open_distance,
            closed_distance=args.grip_closed_distance,
            smooth_radius=args.grip_smooth_radius,
            open_curl_deg=args.grip_open_curl_deg,
            closed_curl_deg=args.grip_closed_curl_deg,
            ring_open_curl_deg=args.grip_ring_open_curl_deg,
            ring_closed_curl_deg=args.grip_ring_closed_curl_deg,
            open_spread_deg=args.grip_open_spread_deg,
            closed_spread_deg=args.grip_closed_spread_deg,
            boundary_blend_frames=args.grip_boundary_blend_frames,
        )

    marker_count = 0
    marker_keyframes = 0
    if not args.no_markers:
        marker_count, marker_keyframes = create_keypoint_markers(
            args.keypoints, args.frame_offset
        )

    all_frames = [frame for side_frames in keyed_frames.values() for frame in side_frames]
    scene = bpy.context.scene
    scene.frame_start = source_frame_start + args.frame_offset
    scene.frame_end = source_frame_end + args.frame_offset
    scene.frame_set(bind_frame + args.frame_offset)
    scene["vggt_dlc_coordinate_formula"] = (
        "X_blender = scale * R_vggt_to_blender @ X_vggt + translation"
    )
    scene["vggt_dlc_bind_frame"] = bind_frame
    scene["vggt_dlc_drive_stage"] = (
        "hand_position_forearm_roll_cup_clearance"
        if args.palm_orientation_mode == "forearm-roll"
        else "hand_controller_position_and_palm_orientation"
    )

    bpy.ops.wm.save_as_mainfile(filepath=str(args.output))

    report = {
        "source_blend": source_blend,
        "output_blend": str(args.output),
        "descriptors": str(args.descriptors),
        "keypoints": str(args.keypoints),
        "bind_frame_source_zero_based": bind_frame,
        "bind_frame_blender": bind_frame + args.frame_offset,
        "source_frame_range": [source_frame_start, source_frame_end],
        "scene_frame_range": [scene.frame_start, scene.frame_end],
        "auto_bind_candidates": bind_candidates,
        "bind_offsets": bind_offsets,
        "palm_orientation_mode": args.palm_orientation_mode,
        "palm_roll": palm_roll_report,
        "elbow_pole_mode": pole_mode,
        "elbow_poles": pole_report,
        "compact_finger_constraints": compact_report,
        "finger_length_compensation": finger_length_report,
        "active_windows": active_windows,
        "active_finger_grip": grip_report,
        "normalized_finger_pose": normalized_finger_report,
        "cup_front_correction": cup_report,
        "keyed_frame_count": {
            side: len(frames) for side, frames in keyed_frames.items()
        },
        "keyed_frame_ranges": {
            side: [min(frames), max(frames)] for side, frames in keyed_frames.items()
        },
        "rejected_temporal_outliers": rejected,
        "marker_count": marker_count,
        "marker_keyframes": marker_keyframes,
        "notes": [
            "Controllers use bind-relative skin-to-control offsets.",
            (
                "Palm orientation is limited to continuous roll around the "
                "evaluated lower-arm Y axis."
                if args.palm_orientation_mode == "forearm-roll"
                else "Palm orientation uses the complete DLC palm frame."
            ),
            (
                "Finger base spread and twist limits were tightened."
                if args.compact_fingers
                else "Finger constraints were not changed."
            ),
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
