"""Build a V20 branch whose wrist IK follows calibrated palm landmarks.

The V19 elbow poles, reach safeguards, shoulder helpers, and palm orientation
animation remain intact. Only the positional Wrist IK target is replaced by a
target constrained to the aligned DLC palm marker and clamped to the arm's
reachable sphere around the existing shoulder helper.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import bpy


OUTPUT_BLEND_NAME = (
    "PyMouseStudio-BrowmFur-WaterReaching-IK-BilateralHand-Control-"
    "HairGN-Experiment-V20-CalibratedPalmTarget-V1.blend"
)
REPORT_DIR_NAME = "V20_CalibratedPalmTarget_V1"
HARD_REACH_RATIO = 0.955
AUDIT_FRAMES = (63, 113, 163)


def remove_named_constraints(obj: bpy.types.Object, names: tuple[str, ...]) -> None:
    for name in names:
        constraint = obj.constraints.get(name)
        if constraint is not None:
            obj.constraints.remove(constraint)


def get_or_create_target(
    name: str,
    source_collection: bpy.types.Collection,
    color: tuple[float, float, float, float],
) -> bpy.types.Object:
    target = bpy.data.objects.get(name)
    if target is None:
        target = bpy.data.objects.new(name, None)
        source_collection.objects.link(target)
    target.animation_data_clear()
    target.parent = None
    target.empty_display_type = "CUBE"
    target.empty_display_size = 0.9
    target.color = color
    target.hide_render = True
    target.hide_viewport = False
    target.hide_set(False)
    target.rotation_mode = "QUATERNION"
    remove_named_constraints(
        target,
        (
            "MGF_CalibratedPalmLocation",
            "MGF_CalibratedReachLimit",
            "MGF_PreservePalmRotation",
        ),
    )
    return target


def world_point(armature: bpy.types.Object, pose_bone, endpoint: str):
    value = pose_bone.head if endpoint == "head" else pose_bone.tail
    return armature.matrix_world @ value


def main() -> None:
    source_path = Path(bpy.data.filepath)
    if "V19-ElbowReachClamp-ShoulderFollow-V2" not in source_path.name:
        raise RuntimeError(f"Expected the V19 ShoulderFollow source, got: {source_path.name}")

    output_path = source_path.with_name(OUTPUT_BLEND_NAME)
    project_root = source_path.parent.parent
    report_path = (
        project_root
        / "output"
        / "bilateral_hand_control"
        / REPORT_DIR_NAME
        / "build_report.json"
    )
    os.makedirs(report_path.parent, exist_ok=True)

    scene = bpy.context.scene
    armature = bpy.data.objects.get("DeformationSystem")
    if armature is None or armature.type != "ARMATURE":
        raise RuntimeError("Missing primary DeformationSystem armature")

    source_collection = bpy.data.collections.get("MGF_VGGTDLC_Keypoints")
    if source_collection is None:
        source_collection = scene.collection

    original_frame = scene.frame_current
    scene.frame_set(AUDIT_FRAMES[0])
    targets: dict[str, bpy.types.Object] = {}
    setup: dict[str, dict[str, object]] = {}

    for side, suffix, color in (
        ("l", "L", (0.15, 0.95, 1.0, 1.0)),
        ("r", "R", (1.0, 0.65, 0.1, 1.0)),
    ):
        marker = bpy.data.objects.get(f"DLC3D_{side}_mid_A")
        raw_hand = bpy.data.objects.get(f"CTRL_Hand_{suffix}")
        shoulder_helper = bpy.data.objects.get(f"CTRL_ShoulderFollow_{suffix}")
        stable_pole = bpy.data.objects.get(f"CTRL_ElbowPoleStable_{suffix}")
        if any(item is None for item in (marker, raw_hand, shoulder_helper, stable_pole)):
            raise RuntimeError(f"Missing V19 helper or calibrated marker for side {suffix}")

        upper = armature.pose.bones.get(f"Scapula_{suffix}")
        lower = armature.pose.bones.get(f"Wrist_{suffix}")
        if upper is None or lower is None:
            raise RuntimeError(f"Missing forelimb bones for side {suffix}")
        shoulder = world_point(armature, upper, "head")
        elbow = world_point(armature, upper, "tail")
        wrist = world_point(armature, lower, "tail")
        arm_length = (elbow - shoulder).length + (wrist - elbow).length
        reach_distance = HARD_REACH_RATIO * arm_length

        target = get_or_create_target(
            f"CTRL_CalibratedPalmTarget_{suffix}", source_collection, color
        )
        target.matrix_world = raw_hand.matrix_world.copy()

        copy_location = target.constraints.new("COPY_LOCATION")
        copy_location.name = "MGF_CalibratedPalmLocation"
        copy_location.target = marker
        copy_location.owner_space = "WORLD"
        copy_location.target_space = "WORLD"
        copy_location.influence = 1.0

        reach_limit = target.constraints.new("LIMIT_DISTANCE")
        reach_limit.name = "MGF_CalibratedReachLimit"
        reach_limit.target = shoulder_helper
        reach_limit.distance = reach_distance
        reach_limit.limit_mode = "LIMITDIST_INSIDE"
        reach_limit.owner_space = "WORLD"
        reach_limit.target_space = "WORLD"
        reach_limit.influence = 1.0
        if hasattr(reach_limit, "use_transform_limit"):
            reach_limit.use_transform_limit = True

        copy_rotation = target.constraints.new("COPY_ROTATION")
        copy_rotation.name = "MGF_PreservePalmRotation"
        copy_rotation.target = raw_hand
        copy_rotation.owner_space = "WORLD"
        copy_rotation.target_space = "WORLD"
        copy_rotation.influence = 1.0

        ik = lower.constraints.get("MGF_ManualHandIK")
        if ik is None or ik.type != "IK":
            raise RuntimeError(f"Missing MGF_ManualHandIK on Wrist_{suffix}")
        ik.target = target
        ik.pole_target = stable_pole
        ik.chain_count = 2
        ik.influence = 1.0
        ik.mute = False
        lower.ik_stretch = 0.0
        upper.ik_stretch = 0.0

        target["mgf_role"] = "calibrated_reach_limited_palm_ik_target"
        target["mgf_side"] = side
        target["mgf_marker"] = marker.name
        target["mgf_raw_rotation_source"] = raw_hand.name
        target["mgf_shoulder_center"] = shoulder_helper.name
        target["mgf_reach_ratio"] = HARD_REACH_RATIO
        target["mgf_reach_distance"] = reach_distance
        targets[suffix] = target
        setup[suffix] = {
            "marker": marker.name,
            "target": target.name,
            "raw_rotation_source": raw_hand.name,
            "shoulder_helper": shoulder_helper.name,
            "stable_pole": stable_pole.name,
            "arm_length": arm_length,
            "reach_distance": reach_distance,
        }

    audit: dict[str, dict[str, dict[str, float | list[float]]]] = {}
    for frame in AUDIT_FRAMES:
        scene.frame_set(frame)
        frame_result: dict[str, dict[str, float | list[float]]] = {}
        for side, suffix in (("l", "L"), ("r", "R")):
            marker = bpy.data.objects[f"DLC3D_{side}_mid_A"]
            target = targets[suffix]
            lower = armature.pose.bones[f"Wrist_{suffix}"]
            wrist = world_point(armature, lower, "tail")
            marker_position = marker.matrix_world.translation
            target_position = target.matrix_world.translation
            frame_result[suffix] = {
                "marker": [float(value) for value in marker_position],
                "limited_target": [float(value) for value in target_position],
                "wrist_tail": [float(value) for value in wrist],
                "target_to_marker": float((target_position - marker_position).length),
                "wrist_to_marker": float((wrist - marker_position).length),
                "wrist_to_target": float((wrist - target_position).length),
            }
        audit[str(frame)] = frame_result

    scene["MGF_CalibratedPalmTargetVersion"] = 1
    scene["MGF_CalibratedPalmHardReachRatio"] = HARD_REACH_RATIO
    scene["MGF_CalibratedPalmCoordinateSystem"] = "anipose_calibration_to_blender"
    scene.frame_set(original_frame if scene.frame_start <= original_frame <= scene.frame_end else 63)

    report = {
        "source": str(source_path),
        "output": str(output_path),
        "hard_reach_ratio": HARD_REACH_RATIO,
        "position_source": "DLC3D_[l|r]_mid_A",
        "rotation_source": "existing CTRL_Hand_[L|R] animation",
        "elbow_pole_source": "existing CTRL_ElbowPoleStable_[L|R] animation",
        "setup": setup,
        "audit": audit,
    }
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path))
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print("MGF_CALIBRATED_PALM_TARGET_REPORT=" + json.dumps(report, ensure_ascii=True))


if __name__ == "__main__":
    main()
