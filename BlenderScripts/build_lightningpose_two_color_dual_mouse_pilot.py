#!/usr/bin/env python3
"""Build a black/brown two-mouse Homecage pilot from a LightningPose plan."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from apply_flexible_tail_animation import (  # noqa: E402
    body_frame,
    key_tail_curve,
    rebuild_tail_spline,
    tail_lengths_world,
)
from flexible_tail_solver import (  # noqa: E402
    ArenaBounds,
    EllipsoidProxy,
    TailSolverConfig,
    solve_tail_sequence,
)


SOURCE_MOUSE_COLLECTION = "MGF_First200_Mouse"
SOURCE_TARGET_COLLECTION = "MGF_First200_IK_Targets"
SOURCE_CONTROL_COLLECTION = "MGF_First200_Trajectory_Controls"
SOURCE_ROOT = "MGF_SourcePose_PerFramePlacement"
SOURCE_ARMATURE = "DeformationSystem.001"
SOURCE_MESH = "MS_Mouse.001"
SOURCE_SPINE_CURVE = "Spine_NURBSCurve.001"
SOURCE_TAIL_CURVE = "Tail_NURBSCurve.001"
TAIL_BONES = ["TailStart_M"] + [f"Tail{index}_M" for index in range(12)]
CONDITIONING_BONES = {
    "nose": "HeadEnd_M",
    "left_ear": "Ear01_L",
    "right_ear": "Ear01_R",
    "neck": "Neck3_M",
    "left_knee": "Knee_L",
    "right_knee": "Knee_R",
    "tail_base": "Root_M",
}


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--brown-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--brown-source-material", default="MGF_BrownFur_KeypointMoseq")
    parser.add_argument("--seed", type=int, default=5706)
    return parser.parse_args(args_after_dash())


def all_source_objects() -> list[bpy.types.Object]:
    objects = {bpy.data.objects[SOURCE_ROOT]}
    for name in (SOURCE_MOUSE_COLLECTION, SOURCE_TARGET_COLLECTION, SOURCE_CONTROL_COLLECTION):
        collection = bpy.data.collections.get(name)
        if collection is not None:
            objects.update(collection.objects)
    return sorted(objects, key=lambda obj: obj.name)


def clear_animation(identifier) -> None:
    if getattr(identifier, "animation_data", None) is not None:
        identifier.animation_data_clear()


def retarget_pointer_properties(owner, mapping: dict[bpy.types.Object, bpy.types.Object]) -> None:
    for attribute in ("object", "target", "mirror_object", "offset_object", "origin"):
        if not hasattr(owner, attribute):
            continue
        value = getattr(owner, attribute)
        try:
            replacement = mapping.get(value)
        except TypeError:
            replacement = None
        if replacement is not None:
            try:
                setattr(owner, attribute, replacement)
            except (AttributeError, TypeError):
                pass
    if hasattr(owner, "keys"):
        try:
            keys = list(owner.keys())
        except TypeError:
            keys = []
        for key in keys:
            try:
                value = owner[key]
            except (KeyError, TypeError):
                continue
            try:
                replacement = mapping.get(value)
            except TypeError:
                replacement = None
            if replacement is not None:
                try:
                    owner[key] = replacement
                except (KeyError, TypeError):
                    pass


def clone_mouse_system(label: str) -> dict:
    source_objects = all_source_objects()
    collection = bpy.data.collections.new(f"MGF_LP_{label}_System")
    bpy.context.scene.collection.children.link(collection)
    mapping: dict[bpy.types.Object, bpy.types.Object] = {}
    for source in source_objects:
        clone = source.copy()
        if source.data is not None:
            clone.data = source.data.copy()
            clear_animation(clone.data)
        clone.name = f"LP_{label}_{source.name}"
        clear_animation(clone)
        if clone.type == "CURVES":
            clone.hide_viewport = True
        collection.objects.link(clone)
        mapping[source] = clone

    for source, clone in mapping.items():
        clone.parent = mapping.get(source.parent, source.parent)
        clone.matrix_parent_inverse = source.matrix_parent_inverse.copy()
        for modifier in clone.modifiers:
            retarget_pointer_properties(modifier, mapping)
        for constraint in clone.constraints:
            retarget_pointer_properties(constraint, mapping)
        if clone.type == "CURVES" and getattr(clone.data, "surface", None) in mapping:
            clone.data.surface = mapping[clone.data.surface]
        if clone.type == "ARMATURE":
            for pose_bone in clone.pose.bones:
                for constraint in pose_bone.constraints:
                    retarget_pointer_properties(constraint, mapping)

    root = mapping[bpy.data.objects[SOURCE_ROOT]]
    root.name = f"LP_{label}_Placement"
    armature = mapping[bpy.data.objects[SOURCE_ARMATURE]]
    armature.name = f"LP_{label}_Armature"
    mesh = mapping[bpy.data.objects[SOURCE_MESH]]
    mesh.name = f"LP_{label}_Mesh"
    spine_curve = mapping[bpy.data.objects[SOURCE_SPINE_CURVE]]
    spine_curve.name = f"LP_{label}_SpineCurve"
    tail_curve = mapping[bpy.data.objects[SOURCE_TAIL_CURVE]]
    tail_curve.name = f"LP_{label}_TailCurve"
    return {
        "label": label,
        "collection": collection,
        "mapping": mapping,
        "root": root,
        "armature": armature,
        "mesh": mesh,
        "spine_curve": spine_curve,
        "tail_curve": tail_curve,
    }


def append_brown_material(source: Path, source_name: str) -> bpy.types.Material:
    output_name = "MGF_BrownFur_LightningPosePilot"
    old = bpy.data.materials.get(output_name)
    if old is not None:
        bpy.data.materials.remove(old)
    with bpy.data.libraries.load(str(source), link=False) as (available, requested):
        if source_name not in available.materials:
            raise KeyError(f"{source_name!r} missing from {source}")
        requested.materials = [source_name]
    material = requested.materials[0]
    if material is None:
        raise RuntimeError("failed to append brown fur material")
    material.name = output_name
    ramp = next((node for node in material.node_tree.nodes if node.type == "VALTORGB"), None)
    if ramp is not None and len(ramp.color_ramp.elements) >= 2:
        ramp.color_ramp.elements[0].color = (0.014, 0.0065, 0.003, 1.0)
        ramp.color_ramp.elements[-1].color = (0.14, 0.055, 0.022, 1.0)
    hair = next(
        (node for node in material.node_tree.nodes if node.type == "BSDF_HAIR_PRINCIPLED"),
        None,
    )
    if hair is not None:
        hair.inputs["Roughness"].default_value = 0.72
        if hair.inputs.get("Random Color") is not None:
            hair.inputs["Random Color"].default_value = 0.08
    return material


def assign_brown_fur(system: dict, material: bpy.types.Material) -> list[str]:
    assigned = []
    for clone in system["mapping"].values():
        if clone.type != "CURVES" or getattr(clone.data, "surface", None) != system["mesh"]:
            continue
        clone.data.materials.clear()
        clone.data.materials.append(material)
        assigned.append(clone.name)
    if not assigned:
        raise RuntimeError("no cloned brown fur objects found")
    return assigned


def key_object_transform(obj: bpy.types.Object, frame: int) -> None:
    obj.keyframe_insert(data_path="location", frame=frame)
    if obj.rotation_mode == "QUATERNION":
        obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)
    elif obj.rotation_mode == "AXIS_ANGLE":
        obj.keyframe_insert(data_path="rotation_axis_angle", frame=frame)
    else:
        obj.keyframe_insert(data_path="rotation_euler", frame=frame)
    obj.keyframe_insert(data_path="scale", frame=frame)


def key_pose_bone(pose_bone: bpy.types.PoseBone, frame: int) -> None:
    pose_bone.keyframe_insert(data_path="location", frame=frame)
    if pose_bone.rotation_mode == "QUATERNION":
        pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame)
    elif pose_bone.rotation_mode == "AXIS_ANGLE":
        pose_bone.keyframe_insert(data_path="rotation_axis_angle", frame=frame)
    else:
        pose_bone.keyframe_insert(data_path="rotation_euler", frame=frame)
    pose_bone.keyframe_insert(data_path="scale", frame=frame)


def copy_source_pose(system: dict, source_frame: int, destination_frame: int) -> tuple[np.ndarray, np.ndarray]:
    scene = bpy.context.scene
    scene.frame_set(source_frame)
    bpy.context.view_layer.update()
    source_root = bpy.data.objects[SOURCE_ROOT]
    source_armature = bpy.data.objects[SOURCE_ARMATURE]
    inverse_root = source_root.matrix_world.inverted()
    base_world = source_armature.matrix_world @ source_armature.pose.bones["Root_M"].head
    nose_world = source_armature.matrix_world @ source_armature.pose.bones["HeadEnd_M"].head
    base_local = np.asarray(inverse_root @ base_world, dtype=np.float64)
    nose_local = np.asarray(inverse_root @ nose_world, dtype=np.float64)

    for source, clone in system["mapping"].items():
        if source == source_root:
            continue
        if source.animation_data is not None and source.animation_data.action is not None:
            clone.matrix_basis = source.matrix_basis.copy()
            key_object_transform(clone, destination_frame)
        if source.type == "CURVE" and source.data.animation_data is not None:
            source_spline = source.data.splines[0]
            clone_spline = clone.data.splines[0]
            if len(source_spline.points) != len(clone_spline.points):
                raise RuntimeError(f"curve point mismatch for {source.name}")
            for source_point, clone_point in zip(source_spline.points, clone_spline.points, strict=True):
                clone_point.co = source_point.co.copy()
                clone_point.keyframe_insert(data_path="co", frame=destination_frame)

    clone_armature = system["armature"]

    for source_bone in source_armature.pose.bones:
        clone_bone = clone_armature.pose.bones[source_bone.name]
        clone_bone.matrix_basis = source_bone.matrix_basis.copy()
        key_pose_bone(clone_bone, destination_frame)
    return base_local, nose_local


def place_system(
    system: dict,
    destination_frame: int,
    base_local: np.ndarray,
    nose_local: np.ndarray,
    target: dict,
    source_base_z: float,
    floor_z: float,
    physical_scale: float,
) -> None:
    source_heading = nose_local[:2] - base_local[:2]
    target_heading = np.asarray(target["target_heading_xy"], dtype=np.float64)
    source_angle = math.atan2(source_heading[1], source_heading[0])
    target_angle = math.atan2(target_heading[1], target_heading[0])
    rotation = target_angle - source_angle
    target_xy = target["target_tail_base_xy_mm"]
    target_base_z = floor_z + (source_base_z - floor_z) * physical_scale
    target_base = Vector((float(target_xy[0]), float(target_xy[1]), target_base_z))
    matrix = (
        Matrix.Translation(target_base)
        @ Matrix.Rotation(rotation, 4, "Z")
        @ Matrix.Scale(physical_scale, 4)
        @ Matrix.Translation(-Vector(base_local))
    )
    root = system["root"]
    root.matrix_world = matrix
    key_object_transform(root, destination_frame)


def action_fcurves(action) -> list:
    if hasattr(action, "fcurves"):
        return list(action.fcurves)
    output = []
    for layer in getattr(action, "layers", ()):
        for strip in getattr(layer, "strips", ()):
            for channelbag in getattr(strip, "channelbags", ()):
                output.extend(channelbag.fcurves)
    return output


def set_all_interpolation(systems: list[dict], mode: str = "BEZIER") -> None:
    identifiers = []
    for system in systems:
        for obj in system["mapping"].values():
            identifiers.append(obj)
            if obj.data is not None:
                identifiers.append(obj.data)
    seen = set()
    for identifier in identifiers:
        action = identifier.animation_data.action if getattr(identifier, "animation_data", None) else None
        if action is None or action.as_pointer() in seen:
            continue
        seen.add(action.as_pointer())
        for fcurve in action_fcurves(action):
            for key in fcurve.keyframe_points:
                key.interpolation = mode


def set_armature_pose_interpolation(systems: list[dict], mode: str = "CONSTANT") -> None:
    for system in systems:
        armature = system["armature"]
        action = armature.animation_data.action if armature.animation_data else None
        if action is None:
            continue
        for fcurve in action_fcurves(action):
            for key in fcurve.keyframe_points:
                key.interpolation = mode


def fixed_physical_scales(plan: dict, safe_source_frames: tuple[int, int]) -> list[float]:
    """Estimate one stable scale per identity from all sparse keyframes."""
    source_root = bpy.data.objects[SOURCE_ROOT]
    source_armature = bpy.data.objects[SOURCE_ARMATURE]
    ratios = [[], []]
    for keyframe in plan["keyframes"]:
        for instance_index, target in enumerate(keyframe["instances"]):
            bpy.context.scene.frame_set(safe_source_frames[instance_index])
            bpy.context.view_layer.update()
            inverse = source_root.matrix_world.inverted()
            base = np.asarray(
                inverse @ (source_armature.matrix_world @ source_armature.pose.bones["Root_M"].head),
                dtype=np.float64,
            )
            nose = np.asarray(
                inverse @ (source_armature.matrix_world @ source_armature.pose.bones["HeadEnd_M"].head),
                dtype=np.float64,
            )
            source_length = float(np.linalg.norm(nose[:2] - base[:2]))
            desired = target["target_keypoints_xy_mm"]
            target_length = float(np.linalg.norm(
                np.asarray(desired["nose"]) - np.asarray(desired["tail_base"])
            ))
            ratios[instance_index].append(target_length / max(source_length, 1.0e-6))
    return [float(np.clip(np.median(values), 0.65, 1.10)) for values in ratios]


def align_sparse_conditioning_roots(systems: list[dict], plan: dict) -> dict:
    """Rigidly re-align cloned evaluated tail-base/head axes at real keyframes."""
    rows = []
    for keyframe in plan["keyframes"]:
        frame = int(keyframe["scene_frame"])
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        for system, target in zip(systems, keyframe["instances"], strict=True):
            armature = system["armature"]
            base = armature.matrix_world @ armature.pose.bones["Root_M"].head
            nose = armature.matrix_world @ armature.pose.bones["HeadEnd_M"].head
            actual_heading = np.asarray((nose.x - base.x, nose.y - base.y), dtype=np.float64)
            target_heading = np.asarray(target["target_heading_xy"], dtype=np.float64)
            actual_angle = math.atan2(actual_heading[1], actual_heading[0])
            target_angle = math.atan2(target_heading[1], target_heading[0])
            delta_angle = target_angle - actual_angle
            desired_xy = (
                np.asarray(target["target_keypoints_xy_mm"]["tail_base"], dtype=np.float64)
                + np.asarray(target["body_contact_shift_xy_mm"], dtype=np.float64)
            )
            desired_base = Vector((float(desired_xy[0]), float(desired_xy[1]), base.z))
            correction = (
                Matrix.Translation(desired_base)
                @ Matrix.Rotation(delta_angle, 4, "Z")
                @ Matrix.Translation(-base)
            )
            root = system["root"]
            root.matrix_world = correction @ root.matrix_world
            key_object_transform(root, frame)
            rows.append({
                "frame": frame,
                "instance": system["label"],
                "translation_mm": float(
                    np.linalg.norm(desired_xy - np.asarray((base.x, base.y)))
                ),
                "rotation_degrees": math.degrees(delta_angle),
            })
    return {
        "rows": rows,
        "maximum_translation_mm": max(row["translation_mm"] for row in rows),
    }


def bake_dense_root_alignment(systems: list[dict], plan: dict, frames: list[int]) -> dict:
    """Bake tail-base/heading alignment every frame around the real body root."""
    key_frames = np.asarray([row["scene_frame"] for row in plan["keyframes"]], dtype=np.float64)
    snapshots = []
    for instance_index, system in enumerate(systems):
        desired_bases = np.asarray([
            np.asarray(
                row["instances"][instance_index]["target_keypoints_xy_mm"]["tail_base"],
                dtype=np.float64,
            )
            + np.asarray(row["instances"][instance_index]["body_contact_shift_xy_mm"], dtype=np.float64)
            for row in plan["keyframes"]
        ])
        heading_angles = np.unwrap(np.asarray([
            math.atan2(
                row["instances"][instance_index]["target_heading_xy"][1],
                row["instances"][instance_index]["target_heading_xy"][0],
            )
            for row in plan["keyframes"]
        ], dtype=np.float64))
        for frame in frames:
            bpy.context.scene.frame_set(frame)
            bpy.context.view_layer.update()
            armature = system["armature"]
            base = armature.matrix_world @ armature.pose.bones["Root_M"].head
            nose = armature.matrix_world @ armature.pose.bones["HeadEnd_M"].head
            actual_angle = math.atan2(nose.y - base.y, nose.x - base.x)
            desired_xy = np.asarray((
                np.interp(frame, key_frames, desired_bases[:, 0]),
                np.interp(frame, key_frames, desired_bases[:, 1]),
            ))
            desired_angle = float(np.interp(frame, key_frames, heading_angles))
            desired_base = Vector((float(desired_xy[0]), float(desired_xy[1]), base.z))
            correction = (
                Matrix.Translation(desired_base)
                @ Matrix.Rotation(desired_angle - actual_angle, 4, "Z")
                @ Matrix.Translation(-base)
            )
            snapshots.append({
                "frame": frame,
                "system": system,
                "matrix": correction @ system["root"].matrix_world,
                "translation_mm": float(np.linalg.norm(desired_xy - np.asarray((base.x, base.y)))),
            })
    for snapshot in snapshots:
        root = snapshot["system"]["root"]
        root.matrix_world = snapshot["matrix"]
        key_object_transform(root, snapshot["frame"])
    return {
        "keyed_frames_per_instance": len(frames),
        "maximum_translation_mm": max(row["translation_mm"] for row in snapshots),
    }


def oriented_body_proxy(armature: bpy.types.Object, radii=(20.0, 12.0, 10.0), margin=0.4) -> EllipsoidProxy:
    base = np.asarray(armature.matrix_world @ armature.pose.bones["Root_M"].head, dtype=np.float64)
    reference = np.asarray(armature.matrix_world @ armature.pose.bones["Spine2_M"].head, dtype=np.float64)
    forward, rotation = body_frame(base, reference)
    return EllipsoidProxy(
        center=tuple(base + forward * (radii[0] * 0.72) + rotation[:, 2]),
        radii=radii,
        name=f"{armature.name}:body",
        margin=margin,
        rotation=tuple(rotation.reshape(-1)),
    )


def enforce_actual_body_spacing(
    systems: list[dict],
    frames: list[int],
    minimum_distance: float,
) -> dict:
    corrections = []
    minimum_before = float("inf")
    snapshots = []
    for frame in frames:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        centers = [
            np.asarray(oriented_body_proxy(system["armature"]).center, dtype=np.float64)
            for system in systems
        ]
        delta = centers[1][:2] - centers[0][:2]
        distance = float(np.linalg.norm(delta))
        minimum_before = min(minimum_before, distance)
        snapshots.append({
            "frame": frame,
            "centers": centers,
            "distance": distance,
            "matrices": [system["root"].matrix_world.copy() for system in systems],
        })

    # Write in a second pass so a key inserted at frame N cannot change the
    # trajectory sampled at frame N+1.
    for snapshot in snapshots:
        frame = snapshot["frame"]
        centers = snapshot["centers"]
        distance = snapshot["distance"]
        delta = centers[1][:2] - centers[0][:2]
        if distance >= minimum_distance:
            amount = 0.0
            direction = np.zeros(2, dtype=np.float64)
        else:
            if distance < 1.0e-6:
                delta = np.asarray((1.0, 0.0))
                distance = 1.0
            amount = minimum_distance - distance
            direction = delta / distance
        for index, (sign, system) in enumerate(((-1.0, systems[0]), (1.0, systems[1]))):
            root = system["root"]
            matrix = snapshot["matrices"][index]
            matrix.translation.x += float(sign * direction[0] * amount * 0.5)
            matrix.translation.y += float(sign * direction[1] * amount * 0.5)
            root.matrix_world = matrix
            root.keyframe_insert(data_path="location", frame=frame)
        if amount > 0.0:
            corrections.append({"frame": frame, "distance_before_mm": distance, "push_mm": amount})
    return {
        "minimum_requested_mm": minimum_distance,
        "minimum_before_mm": minimum_before,
        "corrected_frames": len(corrections),
        "maximum_total_push_mm": max((row["push_mm"] for row in corrections), default=0.0),
        "rows": corrections,
    }


def solve_system_tail(
    system: dict,
    other: dict,
    frames: list[int],
    seed: int,
    floor_z: float,
    arena: ArenaBounds,
) -> dict:
    armature = system["armature"]
    other_armature = other["armature"]
    bases = []
    tangents = []
    proxies = []
    for frame in frames:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        base = np.asarray(armature.matrix_world @ armature.pose.bones["Root_M"].head, dtype=np.float64)
        reference = np.asarray(armature.matrix_world @ armature.pose.bones["Spine2_M"].head, dtype=np.float64)
        forward, _rotation = body_frame(base, reference)
        bases.append(base)
        tangents.append(-forward)
        proxies.append((
            oriented_body_proxy(armature),
            oriented_body_proxy(other_armature, radii=(18.0, 10.5, 8.5), margin=0.8),
        ))
    bpy.context.scene.frame_set(frames[0])
    lengths = np.asarray([
        (
            (armature.matrix_world @ armature.data.bones[name].tail_local)
            - (armature.matrix_world @ armature.data.bones[name].head_local)
        ).length
        for name in TAIL_BONES
    ], dtype=np.float64)
    config = TailSolverConfig(
        body_collision_start_point=3,
        temporal_ar=0.97,
        correction_iterations=80,
        maximum_bend_degrees=34.0,
    )
    best = None
    attempts = 12
    for attempt in range(attempts):
        attempt_seed = seed + attempt * 7919
        candidate_chains, candidate_audits = solve_tail_sequence(
            np.asarray(bases),
            np.asarray(tangents),
            lengths,
            seed=attempt_seed,
            floor_z=floor_z,
            arena=arena,
            collision_proxies_by_frame=proxies,
            config=config,
        )
        score = (
            sum(not audit.accepted for audit in candidate_audits),
            max(audit.maximum_residual_penetration for audit in candidate_audits),
        )
        if best is None or score < best[0]:
            best = (score, attempt_seed, candidate_chains, candidate_audits, attempt + 1)
        if score[0] == 0:
            break
    _score, accepted_seed, chains, audits, attempts_used = best
    rebuild_tail_spline(system["tail_curve"], chains.shape[1], "POLY")
    key_tail_curve(system["tail_curve"], frames, chains)
    return {
        "accepted_frames": sum(audit.accepted for audit in audits),
        "rejected_frames": sum(not audit.accepted for audit in audits),
        "maximum_residual_penetration_mm": max(a.maximum_residual_penetration for a in audits),
        "maximum_length_error_mm": max(a.length_error_max for a in audits),
        "accepted_seed": accepted_seed,
        "resampling_attempts": attempts_used,
        "base_bounds_mm": {
            "minimum": np.min(np.asarray(bases), axis=0).tolist(),
            "maximum": np.max(np.asarray(bases), axis=0).tolist(),
        },
        "rejected_details": [
            {
                "frame": frames[index],
                "base": np.asarray(bases[index]).tolist(),
                "residual_mm": audit.maximum_residual_penetration,
            }
            for index, audit in enumerate(audits)
            if not audit.accepted
        ],
        "corrections": {
            "floor": sum(a.floor_pushes for a in audits),
            "wall": sum(a.wall_pushes for a in audits),
            "body": sum(a.body_pushes for a in audits),
        },
    }


def conditioning_world_xy(system: dict) -> dict[str, list[float]]:
    armature = system["armature"]
    output = {}
    for name, bone in CONDITIONING_BONES.items():
        point = armature.matrix_world @ armature.pose.bones[bone].head
        output[name] = [float(point.x), float(point.y)]
    return output


def pose_audit(
    systems: list[dict],
    plan: dict,
    floor_z: float,
    baked_internal_pose_frames: tuple[int, int],
) -> dict:
    rows = []
    minimum_center_distance = float("inf")
    maximum_conditioning_error = 0.0
    for keyframe in plan["keyframes"]:
        frame = int(keyframe["scene_frame"])
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        centers = []
        for instance_index, (system, target) in enumerate(
            zip(systems, keyframe["instances"], strict=True)
        ):
            actual = conditioning_world_xy(system)
            desired = target["target_keypoints_xy_mm"]
            # The physical-contact shift applies to the entire target instance.
            shift = np.asarray(target["body_contact_shift_xy_mm"], dtype=np.float64)
            errors = {
                name: float(np.linalg.norm(np.asarray(actual[name]) - (np.asarray(desired[name]) + shift)))
                for name in CONDITIONING_BONES
            }
            maximum_conditioning_error = max(maximum_conditioning_error, max(errors.values()))
            base = np.asarray(actual["tail_base"])
            neck = np.asarray(actual["neck"])
            centers.append((base + neck) * 0.5)
            rows.append({
                "frame": frame,
                "instance": system["label"],
                "matched_pose_bank_frame": target["source_pose_frame"],
                "baked_internal_pose_frame": baked_internal_pose_frames[instance_index],
                "conditioning_error_mm": errors,
            })
        minimum_center_distance = min(minimum_center_distance, float(np.linalg.norm(centers[0] - centers[1])))
    return {
        "minimum_proxy_center_distance_mm": minimum_center_distance,
        "maximum_conditioning_point_error_mm": maximum_conditioning_error,
        "rows": rows,
        "floor_z_mm": floor_z,
    }


def main() -> None:
    args = parse_args()
    source_blend_path = bpy.data.filepath
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    scene = bpy.context.scene
    source_mouse_collection = bpy.data.collections[SOURCE_MOUSE_COLLECTION]
    source_mouse_collection.hide_render = True
    source_mouse_collection.hide_viewport = False
    for source_object in source_mouse_collection.objects:
        if source_object.type == "CURVES":
            source_object.hide_viewport = True

    black = clone_mouse_system("Black")
    brown = clone_mouse_system("Brown")
    systems = [black, brown]
    brown_material = append_brown_material(args.brown_source, args.brown_source_material)
    brown_fur = assign_brown_fur(brown, brown_material)
    # Visually reviewed, root-compatible poses.  Frame 15 is the original
    # grounded black stance; frame 98 replaces the unjustified rearing brown
    # stance (frame 14) with a low-head four-foot pose selected by this
    # segment's matcher.
    safe_source_frames = (15, 98)
    physical_scales = fixed_physical_scales(plan, safe_source_frames)

    source_armature = bpy.data.objects[SOURCE_ARMATURE]
    floor_z = float(scene.get("MGF_ground_contact_floor_z_mm", -237.19223022460938))
    for keyframe in plan["keyframes"]:
        destination_frame = int(keyframe["scene_frame"])
        for instance_index, (system, target) in enumerate(
            zip(systems, keyframe["instances"], strict=True)
        ):
            source_frame = safe_source_frames[instance_index]
            base_local, nose_local = copy_source_pose(system, source_frame, destination_frame)
            source_base_z = float(
                (source_armature.matrix_world @ source_armature.pose.bones["Root_M"].head).z
            )
            place_system(
                system,
                destination_frame,
                base_local,
                nose_local,
                target,
                source_base_z,
                floor_z,
                physical_scales[instance_index],
            )
    frame_start, frame_end = (int(value) for value in plan["scene_frame_range"])
    scene.frame_start = frame_start
    scene.frame_end = frame_end
    scene.render.fps = int(plan.get("scene_fps", 8))
    frames = list(range(frame_start, frame_end + 1))
    set_all_interpolation(systems, "LINEAR")
    set_armature_pose_interpolation(systems, "CONSTANT")
    dense_alignment = bake_dense_root_alignment(systems, plan, frames)
    spacing_audit = enforce_actual_body_spacing(systems, frames, 44.0)
    arena = ArenaBounds((-132.0, -98.0, floor_z), (132.0, 98.0, floor_z + 150.0))
    tail_audits = {
        black["label"]: solve_system_tail(black, brown, frames, args.seed, floor_z, arena),
        brown["label"]: solve_system_tail(brown, black, frames, args.seed + 1009, floor_z, arena),
    }
    if any(value["rejected_frames"] for value in tail_audits.values()):
        raise RuntimeError(f"tail solver rejected frames: {tail_audits}")

    audit = pose_audit(systems, plan, floor_z, safe_source_frames)
    if audit["minimum_proxy_center_distance_mm"] < 18.0:
        raise RuntimeError(f"mouse body proxies remain too close: {audit}")
    if audit["maximum_conditioning_point_error_mm"] > 50.0:
        raise RuntimeError(f"LightningPose conditioning drift is too large: {audit}")
    scene.frame_set(frame_start)
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene["MGF_project_kind"] = "LightningPose CRIM13 two-color dual-mouse Homecage pilot"
    scene["MGF_lightningpose_plan"] = str(args.plan)
    scene["MGF_dual_mouse_instances"] = json.dumps([system["armature"].name for system in systems])
    scene["MGF_appearance_mapping"] = "black_mouse=black_c57;white_mouse=brown_agouti"
    scene["MGF_sparse_conditioning_warning"] = "interpolated frames are synthetic, not real frame labels"
    scene["MGF_flexible_tail_profile"] = "tail-base-conditioned-collision-corrected-v1-dual"
    viewport_hidden_fur = []
    for system in systems:
        for clone in system["mapping"].values():
            if clone.type == "CURVES":
                clone.hide_viewport = True
                clone.hide_render = False
                viewport_hidden_fur.append(clone.name)
    scene["MGF_fur_viewport_default"] = "hidden"
    scene["MGF_fur_render_enabled"] = True
    scene["MGF_fur_viewport_objects"] = json.dumps(viewport_hidden_fur)
    source_mouse_collection.hide_viewport = True

    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output), check_existing=False)
    report = {
        "schema": "mousegeoforge.lightningpose_two_color_blender_pilot.audit.v1",
        "output_blend": str(args.output),
        "plan": str(args.plan),
        "source_blend": source_blend_path,
        "source_session": plan["source_session"],
        "source_label_frames": plan["source_label_frames"],
        "scene_frame_range": plan["scene_frame_range"],
        "instances": [{
            "label": system["label"],
            "armature": system["armature"].name,
            "mesh": system["mesh"].name,
            "tail_curve": system["tail_curve"].name,
        } for system in systems],
        "brown_material": brown_material.name,
        "physical_scales": physical_scales,
        "safe_internal_pose_frames": list(safe_source_frames),
        "actual_body_spacing": spacing_audit,
        "dense_root_alignment": dense_alignment,
        "brown_fur_objects": brown_fur,
        "tail_audits": tail_audits,
        "pose_audit": audit,
        "limitations": [
            "LightningPose annotations are sparse; only eight keyframes are real 2D constraints",
            f"the pilot bakes stable internal poses {safe_source_frames[0]}/{safe_source_frames[1]}; LightningPose drives root position, heading, contact layout, and tail base rather than per-joint body articulation",
            "pose-bank matches are retained as experimental diagnostics but are not baked because direct sparse rig retargeting was numerically unstable",
            "body contact is regularized to a minimum center distance instead of exact mesh dynamics",
        ],
    }
    report_path = args.report or args.output.with_suffix(".audit.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "frames": plan["scene_frame_range"],
        "tail_audits": tail_audits,
        "pose_audit": {key: value for key, value in audit.items() if key != "rows"},
    }, indent=2))


if __name__ == "__main__":
    main()
