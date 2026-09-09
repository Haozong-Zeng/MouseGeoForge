"""Create a V19 branch with reach-safe hand IK and continuous elbow poles.

The raw hand controllers remain the semantic/keypoint targets. New baked helper
controllers become the actual IK targets, so unreachable measurements cannot
force the two-bone chain into a nearly straight or ambiguous elbow pose.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import bpy
from mathutils import Vector


OUTPUT_BLEND_NAME = (
    "PyMouseStudio-BrowmFur-WaterReaching-IK-BilateralHand-Control-"
    "HairGN-Experiment-V19-ElbowReachClamp-V1.blend"
)
FRAME_START = 59
FRAME_END = 185
SOFT_REACH_RATIO = 0.92
HARD_REACH_RATIO = 0.965
MIN_REACH_RATIO = 0.20
POLE_MIN_RATIO = 0.50
POLE_MAX_RATIO = 0.75
POLE_DIRECTION_BLEND = 0.35


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
) -> bpy.types.Object:
    obj = bpy.data.objects.get(name)
    if obj is None:
        obj = bpy.data.objects.new(name, None)
        collection = source.users_collection[0] if source.users_collection else bpy.context.scene.collection
        collection.objects.link(obj)
    obj.animation_data_clear()
    obj.parent = None
    obj.empty_display_type = display_type
    obj.empty_display_size = 0.75
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
        if armature_name not in bpy.data.objects:
            continue
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


def main() -> None:
    args = parse_args()
    source_path = Path(bpy.data.filepath)
    if (
        args.output is None
        and "V19-PalmToMouth-CompactFingers" not in source_path.name
        and "V19-ElbowReachClamp" not in source_path.name
    ):
        raise RuntimeError(f"Expected the V19 source branch, got: {source_path.name}")

    output_path = args.output or source_path.with_name(OUTPUT_BLEND_NAME)
    project_root = source_path.parent.parent
    report_path = args.report or (
        project_root
        / "output"
        / "bilateral_hand_control"
        / "V19_ElbowReachClamp_V1"
        / "build_report.json"
    )
    os.makedirs(output_path.parent, exist_ok=True)
    os.makedirs(report_path.parent, exist_ok=True)

    specs = rig_specs()
    helpers: dict[tuple[str, str], dict[str, bpy.types.Object]] = {}
    previous_pole_direction: dict[tuple[str, str], Vector | None] = {}
    previous_pole_distance: dict[tuple[str, str], float | None] = {}
    stats: dict[str, dict[str, object]] = {}

    for spec in specs:
        suffix = spec["suffix"]
        for side in ("R", "L"):
            raw_hand = bpy.data.objects[f"CTRL_Hand_{side}{suffix}"]
            raw_pole = bpy.data.objects[f"CTRL_ElbowPole_{side}{suffix}"]
            safe = get_or_create_helper(
                f"CTRL_HandReachSafe_{side}{suffix}",
                raw_hand,
                "CUBE",
                (1.0, 0.28, 0.05, 1.0),
            )
            stable_pole = get_or_create_helper(
                f"CTRL_ElbowPoleStable_{side}{suffix}",
                raw_pole,
                "SPHERE",
                (0.05, 0.55, 1.0, 1.0),
            )
            safe["mgf_role"] = "actual_reach_safe_ik_target"
            safe["mgf_raw_target"] = raw_hand.name
            stable_pole["mgf_role"] = "continuous_elbow_pole"
            stable_pole["mgf_raw_pole"] = raw_pole.name
            helpers[(suffix, side)] = {
                "raw_hand": raw_hand,
                "raw_pole": raw_pole,
                "safe": safe,
                "stable_pole": stable_pole,
            }
            previous_pole_direction[(suffix, side)] = None
            previous_pole_distance[(suffix, side)] = None
            stats[f"{spec['armature']}:{side}"] = {
                "clamped_frames": [],
                "max_raw_reach_ratio": 0.0,
                "max_safe_reach_ratio": 0.0,
                "max_target_correction": 0.0,
            }

    scene = bpy.context.scene
    frame_start = args.frame_start if args.frame_start is not None else scene.frame_start
    frame_end = args.frame_end if args.frame_end is not None else scene.frame_end
    if frame_start > frame_end:
        raise ValueError(f"Invalid frame range: {frame_start}..{frame_end}")
    display_frame = min(max(63, frame_start), frame_end)
    original_frame = scene.frame_current
    curve_state = hide_curve_evaluation()
    try:
        for frame in range(frame_start, frame_end + 1):
            scene.frame_set(frame)
            for spec in specs:
                suffix = spec["suffix"]
                armature = bpy.data.objects[spec["armature"]]
                for side in ("R", "L"):
                    key = (suffix, side)
                    item = helpers[key]
                    upper = armature.pose.bones[f"Scapula_{side}"]
                    lower = armature.pose.bones[f"Wrist_{side}"]
                    shoulder = armature.matrix_world @ upper.head
                    elbow = armature.matrix_world @ upper.tail
                    wrist = armature.matrix_world @ lower.tail
                    desired = item["raw_hand"].matrix_world.translation.copy()
                    raw_pole = item["raw_pole"].matrix_world.translation.copy()

                    length_1 = (elbow - shoulder).length
                    length_2 = (wrist - elbow).length
                    total_length = length_1 + length_2
                    target_vector = desired - shoulder
                    raw_distance = target_vector.length
                    axis = target_vector.normalized() if raw_distance > 1e-8 else Vector((0.0, 1.0, 0.0))
                    safe_distance = smooth_reach(raw_distance, total_length)
                    safe_position = shoulder + axis * safe_distance

                    pole_vector = raw_pole - shoulder
                    raw_direction = project_perpendicular(pole_vector, axis)
                    if raw_direction.length < 1e-6:
                        raw_direction = project_perpendicular(elbow - shoulder, axis)
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
                            blended = previous * (1.0 - POLE_DIRECTION_BLEND) + raw_direction * POLE_DIRECTION_BLEND
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
                    stable_pole_position = shoulder + axis * axial + raw_direction * pole_distance

                    safe = item["safe"]
                    safe.location = safe_position
                    safe.rotation_quaternion = item["raw_hand"].matrix_world.to_quaternion()
                    safe.keyframe_insert("location", frame=frame, group="MGF Reach Safe")
                    safe.keyframe_insert("rotation_quaternion", frame=frame, group="MGF Reach Safe")

                    stable_pole = item["stable_pole"]
                    stable_pole.location = stable_pole_position
                    stable_pole.keyframe_insert("location", frame=frame, group="MGF Stable Pole")

                    stat = stats[f"{spec['armature']}:{side}"]
                    raw_ratio = raw_distance / total_length
                    safe_ratio = safe_distance / total_length
                    correction = (desired - safe_position).length
                    stat["max_raw_reach_ratio"] = max(stat["max_raw_reach_ratio"], raw_ratio)
                    stat["max_safe_reach_ratio"] = max(stat["max_safe_reach_ratio"], safe_ratio)
                    stat["max_target_correction"] = max(stat["max_target_correction"], correction)
                    if correction > 1e-4:
                        stat["clamped_frames"].append(frame)

        for item in helpers.values():
            set_linear_interpolation(item["safe"])
            set_linear_interpolation(item["stable_pole"])

        for spec in specs:
            suffix = spec["suffix"]
            armature = bpy.data.objects[spec["armature"]]
            for side in ("R", "L"):
                wrist = armature.pose.bones[f"Wrist_{side}"]
                upper = armature.pose.bones[f"Scapula_{side}"]
                constraint = wrist.constraints.get("MGF_ManualHandIK")
                if not constraint or constraint.type != "IK":
                    raise RuntimeError(f"Missing MGF_ManualHandIK on {armature.name}:Wrist_{side}")
                constraint.target = helpers[(suffix, side)]["safe"]
                constraint.pole_target = helpers[(suffix, side)]["stable_pole"]
                constraint.chain_count = 2
                constraint.influence = 1.0
                constraint.mute = False
                wrist.ik_stretch = 0.0
                upper.ik_stretch = 0.0
                wrist["MGF_MinElbowBendDeg"] = 20.0
                wrist["MGF_ReachSoftRatio"] = SOFT_REACH_RATIO
                wrist["MGF_ReachHardRatio"] = HARD_REACH_RATIO

        scene["MGF_ElbowReachClampVersion"] = 1
        scene["MGF_ElbowReachSoftRatio"] = SOFT_REACH_RATIO
        scene["MGF_ElbowReachHardRatio"] = HARD_REACH_RATIO
        scene["MGF_ElbowPoleContinuity"] = True
        scene.frame_start = frame_start
        scene.frame_end = frame_end
        scene.frame_set(display_frame)
    finally:
        restore_curve_evaluation(curve_state)
        scene.frame_set(
            original_frame if frame_start <= original_frame <= frame_end else display_frame
        )

    report = {
        "source": str(source_path),
        "output": str(output_path),
        "frames": [frame_start, frame_end],
        "soft_reach_ratio": SOFT_REACH_RATIO,
        "hard_reach_ratio": HARD_REACH_RATIO,
        "minimum_semantic_elbow_bend_deg": 20.0,
        "ik_stretch_disabled": True,
        "helpers": {
            f"{suffix or 'primary'}:{side}": {
                "reach_safe": item["safe"].name,
                "stable_pole": item["stable_pole"].name,
                "raw_hand": item["raw_hand"].name,
                "raw_pole": item["raw_pole"].name,
            }
            for (suffix, side), item in helpers.items()
        },
        "stats": stats,
    }
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path))
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print("MGF_ELBOW_REACH_CLAMP_REPORT=" + json.dumps(report, ensure_ascii=True))


if __name__ == "__main__":
    main()
