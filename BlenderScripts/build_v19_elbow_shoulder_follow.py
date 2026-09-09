"""Add a small, smoothed shoulder follow layer on top of ElbowReachClamp V1.

The raw hand controllers remain the semantic targets.  A shoulder helper absorbs
only a bounded fraction of overreach, after which ReachSafe and StablePole are
rebaked from the translated shoulder.  This preserves the minimum elbow bend
while allowing the scapula to participate naturally in long reaches.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import bpy
from mathutils import Quaternion, Vector


OUTPUT_BLEND_NAME = (
    "PyMouseStudio-BrowmFur-WaterReaching-IK-BilateralHand-Control-"
    "HairGN-Experiment-V19-ElbowReachClamp-ShoulderFollow-V2.blend"
)
FRAME_START = 59
FRAME_END = 185
SOFT_REACH_RATIO = 0.92
HARD_REACH_RATIO = 0.965
MIN_REACH_RATIO = 0.20
POLE_MIN_RATIO = 0.50
POLE_MAX_RATIO = 0.75
POLE_DIRECTION_BLEND = 0.35
SHOULDER_FOLLOW_FRACTION = 0.18
MAX_SHOULDER_FOLLOW_RATIO = 0.045
SMOOTH_WEIGHTS = (1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0)
MIN_ELBOW_BEND_DEG = 20.0
TARGET_ELBOW_BEND_DEG = 24.0
ELBOW_REFINEMENT_SHRINK = 0.94
ELBOW_REFINEMENT_PASSES = 8


def blender_args() -> list[str]:
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--pack-resources", action="store_true")
    parser.add_argument("--trial-name")
    parser.add_argument("--source-anipose", type=Path)
    parser.add_argument("--calibration", type=Path)
    return parser.parse_args(blender_args())


def project_perpendicular(vector: Vector, axis: Vector) -> Vector:
    return vector - axis * vector.dot(axis)


def smooth_reach(distance: float, total_length: float) -> float:
    lower = MIN_REACH_RATIO * total_length
    soft = SOFT_REACH_RATIO * total_length
    hard = HARD_REACH_RATIO * total_length
    if distance < lower:
        return lower
    if distance <= soft:
        return distance
    width = max(hard - soft, 1e-8)
    return soft + width * (1.0 - math.exp(-(distance - soft) / width))


def action_fcurves(action: bpy.types.Action | None) -> list[bpy.types.FCurve]:
    if action is None:
        return []
    legacy = list(getattr(action, "fcurves", []))
    if legacy:
        return legacy
    result: list[bpy.types.FCurve] = []
    for layer in getattr(action, "layers", []):
        for strip in getattr(layer, "strips", []):
            for channelbag in getattr(strip, "channelbags", []):
                result.extend(channelbag.fcurves)
    return result


def set_linear_interpolation(obj: bpy.types.Object) -> None:
    action = obj.animation_data.action if obj.animation_data else None
    for curve in action_fcurves(action):
        for point in curve.keyframe_points:
            point.interpolation = "LINEAR"


def get_or_create_helper(
    name: str,
    source: bpy.types.Object,
    display_type: str,
    color: tuple[float, float, float, float],
    clear_animation: bool = False,
) -> bpy.types.Object:
    obj = bpy.data.objects.get(name)
    if obj is None:
        obj = bpy.data.objects.new(name, None)
        collection = source.users_collection[0] if source.users_collection else bpy.context.scene.collection
        collection.objects.link(obj)
    if clear_animation:
        obj.animation_data_clear()
    obj.parent = None
    obj.empty_display_type = display_type
    obj.empty_display_size = 0.70
    obj.color = color
    obj.hide_render = True
    obj.hide_viewport = False
    obj.hide_set(False)
    obj.rotation_mode = "QUATERNION"
    return obj


def rig_specs() -> list[dict[str, str]]:
    specs = []
    for suffix in ("", ".001"):
        armature_name = "DeformationSystem" + suffix
        if armature_name in bpy.data.objects:
            specs.append({"suffix": suffix, "armature": armature_name})
    if not specs:
        raise RuntimeError("No DeformationSystem armature found")
    return specs


def hide_curve_evaluation() -> list[tuple]:
    state = []
    for obj in bpy.data.objects:
        if obj.type != "CURVES":
            continue
        modifiers = [(modifier, modifier.show_viewport) for modifier in obj.modifiers]
        state.append((obj, obj.hide_viewport, modifiers))
        obj.hide_viewport = True
        for modifier, _ in modifiers:
            modifier.show_viewport = False
    return state


def restore_curve_evaluation(state: list[tuple]) -> None:
    for obj, hidden, modifiers in state:
        obj.hide_viewport = hidden
        for modifier, visible in modifiers:
            modifier.show_viewport = visible


def smooth_offsets(raw_offsets: list[Vector], caps: list[float]) -> list[Vector]:
    radius = len(SMOOTH_WEIGHTS) // 2
    result = []
    for index in range(len(raw_offsets)):
        weighted = Vector((0.0, 0.0, 0.0))
        weight_sum = 0.0
        for kernel_index, weight in enumerate(SMOOTH_WEIGHTS):
            source_index = min(max(index + kernel_index - radius, 0), len(raw_offsets) - 1)
            weighted += raw_offsets[source_index] * weight
            weight_sum += weight
        offset = weighted / weight_sum
        cap = caps[index]
        if offset.length > cap > 0.0:
            offset = offset.normalized() * cap
        result.append(offset)
    return result


def add_shoulder_constraint(
    armature: bpy.types.Object,
    side: str,
    target: bpy.types.Object,
) -> bpy.types.Constraint:
    bone = armature.pose.bones[f"Scapula_{side}"]
    existing = bone.constraints.get("MGF_ShoulderFollow")
    if existing is not None:
        bone.constraints.remove(existing)
    constraint = bone.constraints.new("COPY_LOCATION")
    constraint.name = "MGF_ShoulderFollow"
    constraint.target = target
    constraint.owner_space = "WORLD"
    constraint.target_space = "WORLD"
    constraint.influence = 1.0
    constraint.mute = False
    if hasattr(constraint, "use_offset"):
        constraint.use_offset = False
    if hasattr(constraint, "mix_mode"):
        constraint.mix_mode = "REPLACE"
    bone["MGF_ShoulderFollowFraction"] = SHOULDER_FOLLOW_FRACTION
    bone["MGF_ShoulderFollowMaxArmRatio"] = MAX_SHOULDER_FOLLOW_RATIO
    return constraint


def elbow_bend_deg(armature: bpy.types.Object, side: str) -> float:
    upper = armature.pose.bones[f"Scapula_{side}"]
    lower = armature.pose.bones[f"Wrist_{side}"]
    shoulder = armature.matrix_world @ upper.head
    elbow = armature.matrix_world @ upper.tail
    wrist = armature.matrix_world @ lower.tail
    return math.degrees((elbow - shoulder).angle(wrist - elbow))


def main() -> None:
    args = parse_args()
    source_path = Path(bpy.data.filepath)
    if args.output is None and "V19-ElbowReachClamp" not in source_path.name:
        raise RuntimeError(f"Expected an ElbowReachClamp source branch, got: {source_path.name}")

    output_path = args.output or source_path.with_name(OUTPUT_BLEND_NAME)
    project_root = source_path.parent.parent
    report_path = args.report or (
        project_root
        / "output"
        / "bilateral_hand_control"
        / "V19_ElbowReachClamp_ShoulderFollow_V2"
        / "build_report.json"
    )
    os.makedirs(output_path.parent, exist_ok=True)
    os.makedirs(report_path.parent, exist_ok=True)

    scene = bpy.context.scene
    frame_start = args.frame_start if args.frame_start is not None else scene.frame_start
    frame_end = args.frame_end if args.frame_end is not None else scene.frame_end
    if frame_start > frame_end:
        raise ValueError(f"Invalid frame range: {frame_start}..{frame_end}")
    display_frame = min(max(63, frame_start), frame_end)
    original_frame = scene.frame_current
    # The visible brown-fur character is the primary rig.  The hidden .001 rig
    # is retained as an untouched reference because hidden pose constraints are
    # not evaluated reliably during the bake/audit pass.
    specs = [spec for spec in rig_specs() if spec["suffix"] == ""]
    items: dict[tuple[str, str], dict[str, object]] = {}
    samples: dict[tuple[str, str], list[dict[str, object]]] = {}
    expected_offsets: dict[tuple[str, str], list[Vector]] = {}

    for spec in specs:
        suffix = spec["suffix"]
        for side in ("R", "L"):
            raw_hand = bpy.data.objects[f"CTRL_Hand_{side}{suffix}"]
            raw_pole = bpy.data.objects[f"CTRL_ElbowPole_{side}{suffix}"]
            safe = bpy.data.objects[f"CTRL_HandReachSafe_{side}{suffix}"]
            stable_pole = bpy.data.objects[f"CTRL_ElbowPoleStable_{side}{suffix}"]
            shoulder_helper = get_or_create_helper(
                f"CTRL_ShoulderFollow_{side}{suffix}",
                raw_hand,
                "SPHERE",
                (0.15, 1.0, 0.22, 1.0),
                clear_animation=True,
            )
            shoulder_helper["mgf_role"] = "bounded_smoothed_shoulder_follow"
            shoulder_helper["mgf_side"] = side
            items[(suffix, side)] = {
                "armature": bpy.data.objects[spec["armature"]],
                "raw_hand": raw_hand,
                "raw_pole": raw_pole,
                "safe": safe,
                "stable_pole": stable_pole,
                "shoulder": shoulder_helper,
            }
            samples[(suffix, side)] = []

    curve_state = hide_curve_evaluation()
    try:
        # Pass 1: collect the unmodified V1 shoulder motion and continuous poles.
        for frame in range(frame_start, frame_end + 1):
            scene.frame_set(frame)
            for key, item in items.items():
                side = key[1]
                armature = item["armature"]
                upper = armature.pose.bones[f"Scapula_{side}"]
                lower = armature.pose.bones[f"Wrist_{side}"]
                shoulder = armature.matrix_world @ upper.head
                elbow = armature.matrix_world @ upper.tail
                wrist = armature.matrix_world @ lower.tail
                desired = item["raw_hand"].matrix_world.translation.copy()
                reference_pole = item["stable_pole"].matrix_world.translation.copy()
                total_length = (elbow - shoulder).length + (wrist - elbow).length
                target_vector = desired - shoulder
                raw_distance = target_vector.length
                overreach = max(0.0, raw_distance - SOFT_REACH_RATIO * total_length)
                follow_distance = min(
                    SHOULDER_FOLLOW_FRACTION * overreach,
                    MAX_SHOULDER_FOLLOW_RATIO * total_length,
                )
                direction = (
                    target_vector.normalized()
                    if raw_distance > 1e-8
                    else Vector((0.0, 0.0, 0.0))
                )
                samples[key].append(
                    {
                        "frame": frame,
                        "base_shoulder": shoulder.copy(),
                        "base_elbow": elbow.copy(),
                        "desired": desired,
                        "reference_pole": reference_pole,
                        "hand_rotation": item["raw_hand"].matrix_world.to_quaternion().copy(),
                        "total_length": total_length,
                        "raw_offset": direction * follow_distance,
                    }
                )

        for key, key_samples in samples.items():
            expected_offsets[key] = smooth_offsets(
                [sample["raw_offset"] for sample in key_samples],
                [MAX_SHOULDER_FOLLOW_RATIO * sample["total_length"] for sample in key_samples],
            )

        # Rebuild all helper animation from the translated shoulder trajectory.
        for item in items.values():
            item["safe"].animation_data_clear()
            item["stable_pole"].animation_data_clear()

        previous_pole_direction: dict[tuple[str, str], Vector | None] = {
            key: None for key in items
        }
        previous_pole_distance: dict[tuple[str, str], float | None] = {
            key: None for key in items
        }
        build_stats: dict[str, dict[str, object]] = {}

        for key, item in items.items():
            suffix, side = key
            stat_key = f"{item['armature'].name}:{side}"
            build_stats[stat_key] = {
                "follow_frames": [],
                "max_shoulder_follow": 0.0,
                "max_shoulder_follow_ratio": 0.0,
                "max_raw_reach_ratio_after_follow": 0.0,
                "max_safe_reach_ratio": 0.0,
                "max_target_correction": 0.0,
            }
            for sample, offset in zip(samples[key], expected_offsets[key]):
                frame = sample["frame"]
                moved_shoulder = sample["base_shoulder"] + offset
                desired = sample["desired"]
                total_length = sample["total_length"]
                target_vector = desired - moved_shoulder
                raw_distance = target_vector.length
                axis = (
                    target_vector.normalized()
                    if raw_distance > 1e-8
                    else Vector((0.0, 1.0, 0.0))
                )
                safe_distance = smooth_reach(raw_distance, total_length)
                safe_position = moved_shoulder + axis * safe_distance

                pole_vector = sample["reference_pole"] - moved_shoulder
                raw_direction = project_perpendicular(pole_vector, axis)
                if raw_direction.length < 1e-6:
                    raw_direction = project_perpendicular(
                        sample["base_elbow"] - sample["base_shoulder"], axis
                    )
                if raw_direction.length < 1e-6:
                    raw_direction = Vector((1.0 if side == "R" else -1.0, 0.0, 0.0))
                    raw_direction = project_perpendicular(raw_direction, axis)
                raw_direction.normalize()

                previous = previous_pole_direction[key]
                if previous is not None:
                    previous = project_perpendicular(previous, axis)
                    if previous.length > 1e-6:
                        previous.normalize()
                        if raw_direction.dot(previous) < 0.0:
                            raw_direction.negate()
                        blended = (
                            previous * (1.0 - POLE_DIRECTION_BLEND)
                            + raw_direction * POLE_DIRECTION_BLEND
                        )
                        if blended.length > 1e-6:
                            raw_direction = blended.normalized()
                previous_pole_direction[key] = raw_direction.copy()

                raw_pole_distance = project_perpendicular(pole_vector, axis).length
                pole_distance = min(
                    max(raw_pole_distance, POLE_MIN_RATIO * total_length),
                    POLE_MAX_RATIO * total_length,
                )
                previous_distance = previous_pole_distance[key]
                if previous_distance is not None:
                    pole_distance = 0.7 * previous_distance + 0.3 * pole_distance
                previous_pole_distance[key] = pole_distance
                axial = min(
                    max(pole_vector.dot(axis), 0.15 * total_length),
                    0.85 * total_length,
                )
                stable_pole_position = (
                    moved_shoulder + axis * axial + raw_direction * pole_distance
                )

                shoulder_helper = item["shoulder"]
                shoulder_helper.location = moved_shoulder
                shoulder_helper.keyframe_insert(
                    "location", frame=frame, group="MGF Shoulder Follow"
                )

                safe = item["safe"]
                safe.location = safe_position
                safe.rotation_quaternion = Quaternion(sample["hand_rotation"])
                safe.keyframe_insert("location", frame=frame, group="MGF Reach Safe")
                safe.keyframe_insert(
                    "rotation_quaternion", frame=frame, group="MGF Reach Safe"
                )

                stable_pole = item["stable_pole"]
                stable_pole.location = stable_pole_position
                stable_pole.keyframe_insert(
                    "location", frame=frame, group="MGF Stable Pole"
                )

                stat = build_stats[stat_key]
                follow_ratio = offset.length / total_length
                stat["max_shoulder_follow"] = max(
                    stat["max_shoulder_follow"], offset.length
                )
                stat["max_shoulder_follow_ratio"] = max(
                    stat["max_shoulder_follow_ratio"], follow_ratio
                )
                stat["max_raw_reach_ratio_after_follow"] = max(
                    stat["max_raw_reach_ratio_after_follow"], raw_distance / total_length
                )
                stat["max_safe_reach_ratio"] = max(
                    stat["max_safe_reach_ratio"], safe_distance / total_length
                )
                correction = (desired - safe_position).length
                stat["max_target_correction"] = max(
                    stat["max_target_correction"], correction
                )
                if offset.length > 1e-4:
                    stat["follow_frames"].append(frame)

        for item in items.values():
            set_linear_interpolation(item["shoulder"])
            set_linear_interpolation(item["safe"])
            set_linear_interpolation(item["stable_pole"])

        for key, item in items.items():
            suffix, side = key
            armature = item["armature"]
            add_shoulder_constraint(armature, side, item["shoulder"])
            wrist = armature.pose.bones[f"Wrist_{side}"]
            upper = armature.pose.bones[f"Scapula_{side}"]
            ik = wrist.constraints.get("MGF_ManualHandIK")
            if not ik or ik.type != "IK":
                raise RuntimeError(f"Missing MGF_ManualHandIK on {armature.name}:Wrist_{side}")
            ik.target = item["safe"]
            ik.pole_target = item["stable_pole"]
            ik.chain_count = 2
            ik.influence = 1.0
            ik.mute = False
            wrist.ik_stretch = 0.0
            upper.ik_stretch = 0.0

        # The theoretical reach cap does not include every controller-pivot and
        # constraint-space offset. Measure the solved bones and gently pull only
        # unsafe frames inward until the actual elbow has a physical margin.
        refinement: dict[str, dict[str, object]] = {
            f"{item['armature'].name}:{key[1]}": {
                "frames_adjusted": set(),
                "minimum_bend_before_deg": 180.0,
                "passes_used": 0,
            }
            for key, item in items.items()
        }
        for pass_index in range(ELBOW_REFINEMENT_PASSES):
            adjusted_count = 0
            for frame in range(frame_start, frame_end + 1):
                scene.frame_set(frame)
                bpy.context.view_layer.update()
                for key, item in items.items():
                    side = key[1]
                    armature = item["armature"]
                    stat_key = f"{armature.name}:{side}"
                    bend_deg = elbow_bend_deg(armature, side)
                    if pass_index == 0:
                        refinement[stat_key]["minimum_bend_before_deg"] = min(
                            refinement[stat_key]["minimum_bend_before_deg"],
                            bend_deg,
                        )
                    if bend_deg >= TARGET_ELBOW_BEND_DEG:
                        continue
                    upper = armature.pose.bones[f"Scapula_{side}"]
                    shoulder = armature.matrix_world @ upper.head
                    safe = item["safe"]
                    target_vector = safe.matrix_world.translation - shoulder
                    if target_vector.length < 1e-8:
                        continue
                    safe.location = (
                        shoulder + target_vector * ELBOW_REFINEMENT_SHRINK
                    )
                    safe.keyframe_insert(
                        "location", frame=frame, group="MGF Reach Safe"
                    )
                    refinement[stat_key]["frames_adjusted"].add(frame)
                    refinement[stat_key]["passes_used"] = pass_index + 1
                    adjusted_count += 1
            for item in items.values():
                set_linear_interpolation(item["safe"])
            if adjusted_count == 0:
                break

        for stat in refinement.values():
            stat["frames_adjusted"] = sorted(stat["frames_adjusted"])

        # Evaluate the finished rig, including the new shoulder constraint.
        audit: dict[str, dict[str, object]] = {}
        for key, item in items.items():
            suffix, side = key
            stat_key = f"{item['armature'].name}:{side}"
            audit[stat_key] = {
                "minimum_elbow_bend_deg": 180.0,
                "minimum_elbow_bend_frame": None,
                "frames_below_20_deg": [],
                "maximum_shoulder_helper_error": 0.0,
                "maximum_shoulder_frame_step": 0.0,
            }
        previous_shoulder: dict[tuple[str, str], Vector | None] = {
            key: None for key in items
        }
        for frame in range(frame_start, frame_end + 1):
            scene.frame_set(frame)
            for key, item in items.items():
                side = key[1]
                armature = item["armature"]
                upper = armature.pose.bones[f"Scapula_{side}"]
                shoulder = armature.matrix_world @ upper.head
                bend_deg = elbow_bend_deg(armature, side)
                helper_error = (
                    shoulder - item["shoulder"].matrix_world.translation
                ).length
                stat = audit[f"{armature.name}:{side}"]
                if bend_deg < stat["minimum_elbow_bend_deg"]:
                    stat["minimum_elbow_bend_deg"] = bend_deg
                    stat["minimum_elbow_bend_frame"] = frame
                if bend_deg < MIN_ELBOW_BEND_DEG:
                    stat["frames_below_20_deg"].append(frame)
                stat["maximum_shoulder_helper_error"] = max(
                    stat["maximum_shoulder_helper_error"], helper_error
                )
                previous = previous_shoulder[key]
                if previous is not None:
                    stat["maximum_shoulder_frame_step"] = max(
                        stat["maximum_shoulder_frame_step"],
                        (shoulder - previous).length,
                    )
                previous_shoulder[key] = shoulder.copy()

        scene["MGF_ElbowReachClampVersion"] = 2
        scene["MGF_ShoulderFollowEnabled"] = True
        scene["MGF_ShoulderFollowFraction"] = SHOULDER_FOLLOW_FRACTION
        scene["MGF_ShoulderFollowMaxArmRatio"] = MAX_SHOULDER_FOLLOW_RATIO
        scene["MGF_ShoulderFollowSmoothingFrames"] = len(SMOOTH_WEIGHTS)
        scene.frame_start = frame_start
        scene.frame_end = frame_end
        scene.frame_set(display_frame)
        if args.trial_name:
            scene["MGF_AniposeTrial"] = args.trial_name
        if args.source_anipose:
            scene["MGF_AniposePoseCSV"] = str(args.source_anipose)
        if args.calibration:
            scene["MGF_AniposeCalibration"] = str(args.calibration)

        report = {
            "source": str(source_path),
            "output": str(output_path),
            "frames": [frame_start, frame_end],
            "shoulder_follow_fraction": SHOULDER_FOLLOW_FRACTION,
            "maximum_shoulder_follow_arm_ratio": MAX_SHOULDER_FOLLOW_RATIO,
            "smoothing_weights": list(SMOOTH_WEIGHTS),
            "soft_reach_ratio": SOFT_REACH_RATIO,
            "hard_reach_ratio": HARD_REACH_RATIO,
            "elbow_refinement": refinement,
            "minimum_elbow_bend_deg": MIN_ELBOW_BEND_DEG,
            "target_elbow_bend_deg": TARGET_ELBOW_BEND_DEG,
            "build_stats": build_stats,
            "audit": audit,
        }
        # Persist the user's original hair visibility/modifier state, not the
        # temporary fast-evaluation state used by the bake and audit loops.
        restore_curve_evaluation(curve_state)
        curve_state = []
        if args.pack_resources:
            bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=str(output_path))
        report_path.write_text(
            json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8"
        )
        print("MGF_SHOULDER_FOLLOW_REPORT=" + json.dumps(report, ensure_ascii=True))
    finally:
        restore_curve_evaluation(curve_state)
        scene.frame_set(
            original_frame if frame_start <= original_frame <= frame_end else display_frame
        )


if __name__ == "__main__":
    main()
