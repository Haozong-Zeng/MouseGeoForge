#!/usr/bin/env python3
"""Bake MoSeq poses onto the SceneOnly furry mouse and convert the scene to raw millimeters."""

from __future__ import annotations

import json
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector


POSE_ARCHIVE = Path(
    r"\\wsl.localhost\Ubuntu\home\haozong\projects\MouseGeoForge\dataset\Cylinder\review\moseq_sample10_first200_evaluated_pose.npz"
)
OUTPUT_BLEND = Path(
    r"\\wsl.localhost\Ubuntu\home\haozong\projects\MouseGeoForge\BlenderPyMouseProjects\Cylinder\PyMouseStudio-fur-SceneOnly-MoSeqPose-Sample10-First200-mm.blend"
)
TARGET_ARMATURE = "DeformationSystem.002"
TARGET_MESH = "MS_Mouse.002"
TARGET_GROUP = "Group.001"
TARGET_CYLINDER = "Part_GlassCylinder160mm.001"
TARGET_PLATFORM = "Part_AntiReflectiveGlass.001"
MILLIMETER_SCENE_COLLECTION = "Collection"
RAW_MM_PLATFORM_CENTER_Z = 338.5
SCENE_SCALE = 1000.0
GROUND_CLEARANCE_M = 0.0005


def evaluated_bounds(obj: bpy.types.Object) -> np.ndarray:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    xyz = np.asarray(
        [evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box],
        dtype=np.float64,
    )
    return np.stack((xyz.min(axis=0), xyz.max(axis=0)))


def hierarchy_depth(pose_bone: bpy.types.PoseBone) -> int:
    depth = 0
    parent = pose_bone.parent
    while parent:
        depth += 1
        parent = parent.parent
    return depth


def mute_pose_constraints(armature: bpy.types.Object) -> int:
    count = 0
    for bone in armature.pose.bones:
        for constraint in bone.constraints:
            constraint.mute = True
            count += 1
    return count


def set_action_interpolation_linear(obj: bpy.types.Object) -> None:
    action = obj.animation_data.action if obj.animation_data else None
    if not action:
        return
    if hasattr(action, "fcurves"):
        curves = action.fcurves
    else:
        curves = [
            curve
            for layer in action.layers
            for strip in layer.strips
            for channelbag in getattr(strip, "channelbags", [])
            for curve in channelbag.fcurves
        ]
    for curve in curves:
        for point in curve.keyframe_points:
            point.interpolation = "LINEAR"


def hide_legacy_mm_scene() -> list[str]:
    collection = bpy.data.collections.get(MILLIMETER_SCENE_COLLECTION)
    if collection is None:
        return []
    kept_collection_names = {"Collection.001", "Collection.002", "集合 2", "RigidBodyWorld.001"}
    hidden = []
    for obj in collection.all_objects:
        if obj is None:
            continue
        if any(user.name in kept_collection_names for user in obj.users_collection):
            continue
        obj.hide_viewport = True
        obj.hide_render = True
        hidden.append(obj.name)
    return hidden


def bake_pose(armature: bpy.types.Object, archive) -> tuple[list[int], list[str], list[str]]:
    source_names = [str(name) for name in archive["bone_names"].tolist()]
    source_index = {name: index for index, name in enumerate(source_names)}
    common = [name for name in source_names if name in armature.pose.bones]
    missing = [name for name in source_names if name not in armature.pose.bones]
    common.sort(key=lambda name: hierarchy_depth(armature.pose.bones[name]))
    frames = [int(frame) for frame in archive["frame_numbers"].tolist()]
    matrices = archive["visual_matrix_basis"]

    armature.animation_data_clear()
    for bone in armature.pose.bones:
        bone.matrix_basis.identity()
    for frame_index, frame in enumerate(frames):
        bpy.context.scene.frame_set(frame)
        for name in common:
            pose_bone = armature.pose.bones[name]
            pose_bone.rotation_mode = "QUATERNION"
            pose_bone.matrix_basis = Matrix(matrices[frame_index, source_index[name]].tolist())
        bpy.context.view_layer.update()
        for name in common:
            pose_bone = armature.pose.bones[name]
            pose_bone.keyframe_insert("location", frame=frame, group=name)
            pose_bone.keyframe_insert("rotation_quaternion", frame=frame, group=name)
            pose_bone.keyframe_insert("scale", frame=frame, group=name)
        if frame == frames[0] or frame == frames[-1] or frame % 25 == 0:
            print(f"baked furry pose frame {frame}/{frames[-1]}", flush=True)
    set_action_interpolation_linear(armature)
    return frames, common, missing


def center_mouse_per_frame(
    frames: list[int],
    mouse: bpy.types.Object,
    group: bpy.types.Object,
    cylinder: bpy.types.Object,
    platform: bpy.types.Object,
) -> tuple[list[list[float]], float]:
    cylinder_bounds = evaluated_bounds(cylinder)
    platform_bounds = evaluated_bounds(platform)
    center_xy = (cylinder_bounds[0, :2] + cylinder_bounds[1, :2]) * 0.5
    platform_top = float(platform_bounds[1, 2])
    group.animation_data_clear()
    translations = []
    max_radius = 0.0
    for frame in frames:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        before = evaluated_bounds(mouse)
        mouse_center_xy = (before[0, :2] + before[1, :2]) * 0.5
        delta = np.asarray(
            [
                center_xy[0] - mouse_center_xy[0],
                center_xy[1] - mouse_center_xy[1],
                platform_top + GROUND_CLEARANCE_M - before[0, 2],
            ],
            dtype=np.float64,
        )
        group.location = Vector(group.location) + Vector(delta.tolist())
        group.keyframe_insert("location", frame=frame)
        bpy.context.view_layer.update()
        after = evaluated_bounds(mouse)
        corners_xy = np.asarray(
            [
                [x, y]
                for x in (after[0, 0], after[1, 0])
                for y in (after[0, 1], after[1, 1])
            ]
        )
        max_radius = max(max_radius, float(np.linalg.norm(corners_xy - center_xy, axis=1).max()))
        translations.append(delta.tolist())
    set_action_interpolation_linear(group)
    return translations, max_radius


def convert_meter_scene_to_raw_mm(
    cylinder: bpy.types.Object,
    platform: bpy.types.Object,
    excluded_objects: set[bpy.types.Object],
) -> bpy.types.Object:
    bpy.context.scene.frame_set(1)
    bpy.context.view_layer.update()
    cylinder_bounds = evaluated_bounds(cylinder)
    platform_bounds = evaluated_bounds(platform)
    cylinder_center = (cylinder_bounds[0] + cylinder_bounds[1]) * 0.5
    platform_center_z = float((platform_bounds[0, 2] + platform_bounds[1, 2]) * 0.5)

    existing = bpy.data.objects.get("MGF_MeterSceneToMillimeter")
    if existing:
        bpy.data.objects.remove(existing, do_unlink=True)
    root = bpy.data.objects.new("MGF_MeterSceneToMillimeter", None)
    bpy.context.scene.collection.objects.link(root)

    roots = [
        obj
        for obj in bpy.data.objects
        if obj is not root and obj.parent is None and obj not in excluded_objects
    ]
    for obj in roots:
        world = obj.matrix_world.copy()
        obj.parent = root
        obj.matrix_world = world

    root.scale = (SCENE_SCALE, SCENE_SCALE, SCENE_SCALE)
    root.location = (
        -SCENE_SCALE * float(cylinder_center[0]),
        -SCENE_SCALE * float(cylinder_center[1]),
        RAW_MM_PLATFORM_CENTER_Z - SCENE_SCALE * platform_center_z,
    )

    for camera in bpy.data.cameras:
        camera.clip_start *= SCENE_SCALE
        camera.clip_end *= SCENE_SCALE
        if camera.dof:
            camera.dof.focus_distance *= SCENE_SCALE
    for light in bpy.data.lights:
        if light.type != "SUN":
            light.energy *= SCENE_SCALE * SCENE_SCALE
            if hasattr(light, "shadow_soft_size"):
                light.shadow_soft_size *= SCENE_SCALE
            if light.type == "AREA":
                light.shape = light.shape
                light.size *= SCENE_SCALE
                if hasattr(light, "size_y"):
                    light.size_y *= SCENE_SCALE
    return root


def main() -> None:
    archive = np.load(POSE_ARCHIVE, allow_pickle=False)
    armature = bpy.data.objects[TARGET_ARMATURE]
    mouse = bpy.data.objects[TARGET_MESH]
    group = bpy.data.objects[TARGET_GROUP]
    cylinder = bpy.data.objects[TARGET_CYLINDER]
    platform = bpy.data.objects[TARGET_PLATFORM]

    hidden_names = hide_legacy_mm_scene()
    hidden_objects = {bpy.data.objects[name] for name in hidden_names if name in bpy.data.objects}
    muted_constraints = mute_pose_constraints(armature)
    frames, common_bones, missing_bones = bake_pose(armature, archive)
    translations, max_radius_m = center_mouse_per_frame(frames, mouse, group, cylinder, platform)
    conversion_root = convert_meter_scene_to_raw_mm(cylinder, platform, hidden_objects)

    scene = bpy.context.scene
    scene.frame_start = frames[0]
    scene.frame_end = frames[-1]
    scene.frame_set(frames[0])
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.length_unit = "MILLIMETERS"
    scene.unit_settings.scale_length = 0.001
    scene["MGF_pose_source"] = str(POSE_ARCHIVE)
    scene["MGF_scene_source"] = bpy.data.filepath
    scene["MGF_scene_scale_to_raw_mm"] = SCENE_SCALE
    scene["MGF_pose_translation_mode"] = "per-frame bbox center to cylinder; bbox min to glass"
    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_BLEND))

    cylinder_radius_m = min(evaluated_bounds(cylinder)[1, :2] - evaluated_bounds(cylinder)[0, :2]) * 0.5 / SCENE_SCALE
    translations_np = np.asarray(translations) * SCENE_SCALE
    audit = {
        "format": "sceneonly_moseq_pose_raw_mm_v1",
        "output_blend": str(OUTPUT_BLEND),
        "scene_source": "PyMouseStudio-fur-SceneOnly.blend",
        "pose_source": str(POSE_ARCHIVE),
        "frames": [frames[0], frames[-1]],
        "common_bones": len(common_bones),
        "missing_target_bones": missing_bones,
        "muted_target_constraints": muted_constraints,
        "hidden_legacy_mm_objects": len(hidden_names),
        "scene_scale": SCENE_SCALE,
        "raw_unit": "millimeter",
        "per_frame_translation_mm_min": translations_np.min(axis=0).tolist(),
        "per_frame_translation_mm_max": translations_np.max(axis=0).tolist(),
        "max_bbox_corner_radius_mm": max_radius_m * SCENE_SCALE,
        "cylinder_radius_mm": float(cylinder_radius_m * SCENE_SCALE),
        "conversion_root": conversion_root.name,
    }
    OUTPUT_BLEND.with_suffix(".alignment.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
