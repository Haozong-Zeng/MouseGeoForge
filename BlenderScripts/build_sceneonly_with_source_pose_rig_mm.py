#!/usr/bin/env python3
"""Combine the SceneOnly lab with the original evaluated MoSeq mouse rig in raw millimeters."""

from __future__ import annotations

import json
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


SOURCE_BLEND = Path(
    r"\\wsl.localhost\Ubuntu\home\haozong\projects\MouseGeoForge\dataset\Cylinder\PyMouseStudio-WhiteFur-Controller-moseq-FullTrajectorySample10-First200.blend"
)
OUTPUT_BLEND = Path(
    r"\\wsl.localhost\Ubuntu\home\haozong\projects\MouseGeoForge\dataset\Cylinder\PyMouseStudio-fur-SceneOnly-SourceRigPose-Sample10-First200-mm.blend"
)
SOURCE_COLLECTIONS = ["Mouse_White", "PseudoKeypoints.001", "MGF_FullTrajectory_Controls"]
TARGET_MOUSE_COLLECTION = "集合 2"
LEGACY_MM_COLLECTION = "Collection"
TARGET_CYLINDER = "Part_GlassCylinder160mm.001"
TARGET_PLATFORM = "Part_AntiReflectiveGlass.001"
TARGET_MOUSE = "MS_Mouse.002"
SOURCE_MOUSE = "MS_Mouse.001"
SOURCE_ARMATURE = "DeformationSystem.001"
SCENE_SCALE = 1000.0
RAW_MM_PLATFORM_CENTER_Z = 338.5
GROUND_CLEARANCE_MM = 0.5


def evaluated_bounds(obj: bpy.types.Object) -> np.ndarray:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    xyz = np.asarray(
        [evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box],
        dtype=np.float64,
    )
    return np.stack((xyz.min(axis=0), xyz.max(axis=0)))


def collection_objects(collections: list[bpy.types.Collection]) -> set[bpy.types.Object]:
    objects = set()
    for collection in collections:
        objects.update(obj for obj in collection.all_objects if obj is not None)
    return objects


def append_source_collections() -> list[bpy.types.Collection]:
    with bpy.data.libraries.load(str(SOURCE_BLEND), link=False) as (data_from, data_to):
        missing = [name for name in SOURCE_COLLECTIONS if name not in data_from.collections]
        if missing:
            raise KeyError(f"Missing source collections: {missing}")
        data_to.collections = SOURCE_COLLECTIONS
    loaded = list(data_to.collections)
    for collection in loaded:
        if collection.name not in bpy.context.scene.collection.children:
            bpy.context.scene.collection.children.link(collection)
    return loaded


def hide_collection_objects(name: str, keep: set[bpy.types.Object] | None = None) -> set[bpy.types.Object]:
    collection = bpy.data.collections.get(name)
    if collection is None:
        return set()
    keep = keep or set()
    hidden = set()
    for obj in collection.all_objects:
        if obj is None or obj in keep:
            continue
        obj.hide_viewport = True
        obj.hide_render = True
        hidden.add(obj)
    return hidden


def retarget_modifier_objects(
    obj: bpy.types.Object,
    source_mesh: bpy.types.Object,
    target_mesh: bpy.types.Object,
) -> None:
    for modifier in obj.modifiers:
        for key in modifier.keys():
            try:
                if modifier[key] == source_mesh:
                    modifier[key] = target_mesh
            except (ReferenceError, TypeError):
                continue


def assign_material(obj: bpy.types.Object, material: bpy.types.Material) -> None:
    obj.data.materials.clear()
    obj.data.materials.append(material)


def duplicate_guard_hair(
    donor_name: str,
    output_name: str,
    donor_mesh: bpy.types.Object,
    source_mesh: bpy.types.Object,
    material: bpy.types.Material,
    destination: bpy.types.Collection,
) -> bpy.types.Object:
    old = bpy.data.objects.get(output_name)
    if old:
        bpy.data.objects.remove(old, do_unlink=True)
    donor = bpy.data.objects[donor_name]
    duplicate = donor.copy()
    duplicate.data = donor.data.copy()
    duplicate.name = output_name
    duplicate.data.name = f"{output_name}_Curves"
    destination.objects.link(duplicate)
    duplicate.parent = source_mesh
    duplicate.data.surface = source_mesh
    retarget_modifier_objects(duplicate, donor_mesh, source_mesh)
    assign_material(duplicate, material)
    duplicate.hide_viewport = False
    duplicate.hide_render = False
    return duplicate


def apply_sceneonly_mouse_appearance(
    source_mesh: bpy.types.Object,
    source_mouse_collection: bpy.types.Collection,
) -> list[str]:
    donor_mesh = bpy.data.objects[TARGET_MOUSE]
    if len(source_mesh.material_slots) != len(donor_mesh.material_slots):
        raise RuntimeError("Source and SceneOnly mouse material-slot counts differ")
    for index, slot in enumerate(donor_mesh.material_slots):
        source_mesh.material_slots[index].material = slot.material

    dense = bpy.data.materials["Black_Whiskers.004"]
    whisker = bpy.data.materials["Black_Whiskers.003"]
    for name in (
        "BaseFur.001",
        "EarFur.001",
        "EyesFur.001",
        "LegFur.001",
        "Mouth.001",
        "TailFur.001",
    ):
        assign_material(bpy.data.objects[name], dense)
    assign_material(bpy.data.objects["Whisker.001"], whisker)
    guard_short = duplicate_guard_hair(
        "BaseFur.003",
        "MGF_SourceRig_GuardShort",
        donor_mesh,
        source_mesh,
        bpy.data.materials["Black_Whiskers.005"],
        source_mouse_collection,
    )
    guard_long = duplicate_guard_hair(
        "BaseFur.004",
        "MGF_SourceRig_GuardLong",
        donor_mesh,
        source_mesh,
        bpy.data.materials["Black_Whiskers.006"],
        source_mouse_collection,
    )
    return [guard_short.name, guard_long.name]


def convert_sceneonly_meter_objects_to_mm(
    original_objects: set[bpy.types.Object],
    excluded: set[bpy.types.Object],
    original_cameras: set[bpy.types.Camera],
    original_lights: set[bpy.types.Light],
) -> bpy.types.Object:
    cylinder = bpy.data.objects[TARGET_CYLINDER]
    platform = bpy.data.objects[TARGET_PLATFORM]
    cylinder_bounds = evaluated_bounds(cylinder)
    platform_bounds = evaluated_bounds(platform)
    cylinder_center = (cylinder_bounds[0] + cylinder_bounds[1]) * 0.5
    platform_center_z = float((platform_bounds[0, 2] + platform_bounds[1, 2]) * 0.5)

    root = bpy.data.objects.new("MGF_SceneOnly_MeterToMillimeter", None)
    bpy.context.scene.collection.objects.link(root)
    for obj in original_objects:
        if obj in excluded or obj.parent is not None:
            continue
        world = obj.matrix_world.copy()
        obj.parent = root
        obj.matrix_world = world
    root.scale = (SCENE_SCALE, SCENE_SCALE, SCENE_SCALE)
    root.location = (
        -SCENE_SCALE * float(cylinder_center[0]),
        -SCENE_SCALE * float(cylinder_center[1]),
        RAW_MM_PLATFORM_CENTER_Z - SCENE_SCALE * platform_center_z,
    )

    for camera in original_cameras:
        camera.clip_start *= SCENE_SCALE
        camera.clip_end *= SCENE_SCALE
        camera.dof.focus_distance *= SCENE_SCALE
    for light in original_lights:
        if light.type == "SUN":
            continue
        light.energy *= SCENE_SCALE * SCENE_SCALE
        if hasattr(light, "shadow_soft_size"):
            light.shadow_soft_size *= SCENE_SCALE
        if light.type == "AREA":
            light.size *= SCENE_SCALE
            if hasattr(light, "size_y"):
                light.size_y *= SCENE_SCALE
    return root


def create_pose_placement_root(source_objects: set[bpy.types.Object]) -> bpy.types.Object:
    placement = bpy.data.objects.new("MGF_SourcePose_PerFramePlacement", None)
    bpy.context.scene.collection.objects.link(placement)
    for obj in source_objects:
        if obj.parent is not None:
            continue
        world = obj.matrix_world.copy()
        obj.parent = placement
        obj.matrix_world = world
    return placement


def align_source_pose_per_frame(
    source_mesh: bpy.types.Object,
    placement: bpy.types.Object,
    frames: range,
) -> tuple[np.ndarray, float]:
    cylinder = bpy.data.objects[TARGET_CYLINDER]
    platform = bpy.data.objects[TARGET_PLATFORM]
    cylinder_bounds = evaluated_bounds(cylinder)
    platform_bounds = evaluated_bounds(platform)
    center_xy = (cylinder_bounds[0, :2] + cylinder_bounds[1, :2]) * 0.5
    platform_top = float(platform_bounds[1, 2])
    placement.animation_data_clear()
    translations = []
    max_radius = 0.0
    for frame in frames:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        before = evaluated_bounds(source_mesh)
        mouse_center_xy = (before[0, :2] + before[1, :2]) * 0.5
        delta = np.asarray(
            [
                center_xy[0] - mouse_center_xy[0],
                center_xy[1] - mouse_center_xy[1],
                platform_top + GROUND_CLEARANCE_MM - before[0, 2],
            ],
            dtype=np.float64,
        )
        placement.location = Vector(placement.location) + Vector(delta.tolist())
        placement.keyframe_insert("location", frame=frame)
        bpy.context.view_layer.update()
        after = evaluated_bounds(source_mesh)
        corners_xy = np.asarray(
            [
                [x, y]
                for x in (after[0, 0], after[1, 0])
                for y in (after[0, 1], after[1, 1])
            ]
        )
        max_radius = max(max_radius, float(np.linalg.norm(corners_xy - center_xy, axis=1).max()))
        translations.append(delta)
    return np.asarray(translations), max_radius


def set_linear_interpolation(obj: bpy.types.Object) -> None:
    action = obj.animation_data.action if obj.animation_data else None
    if not action:
        return
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


def main() -> None:
    scene = bpy.context.scene
    scene.frame_set(1)
    bpy.context.view_layer.update()
    original_objects = set(bpy.data.objects)
    original_cameras = set(bpy.data.cameras)
    original_lights = set(bpy.data.lights)

    target_mouse_objects = hide_collection_objects(TARGET_MOUSE_COLLECTION)
    legacy_mm_objects = hide_collection_objects(LEGACY_MM_COLLECTION)
    excluded_scene_objects = target_mouse_objects | legacy_mm_objects

    scene_root = convert_sceneonly_meter_objects_to_mm(
        original_objects,
        excluded_scene_objects,
        original_cameras,
        original_lights,
    )
    source_collections = append_source_collections()
    source_objects = collection_objects(source_collections)
    source_mouse = bpy.data.objects[SOURCE_MOUSE]
    source_armature = bpy.data.objects[SOURCE_ARMATURE]
    source_mouse_collection = next(collection for collection in source_collections if collection.name.startswith("Mouse_White"))
    guard_hair = apply_sceneonly_mouse_appearance(source_mouse, source_mouse_collection)

    for collection in source_collections:
        if collection.name.startswith("PseudoKeypoints") or collection.name.startswith("MGF_FullTrajectory"):
            for obj in collection.all_objects:
                if obj:
                    obj.hide_render = True

    placement = create_pose_placement_root(source_objects)
    frames = range(1, 201)
    source_curves = [obj for obj in source_mouse_collection.all_objects if obj and obj.type == "CURVES"]
    curve_viewport_state = {obj: obj.hide_viewport for obj in source_curves}
    for obj in source_curves:
        obj.hide_viewport = True
    translations, max_radius_mm = align_source_pose_per_frame(source_mouse, placement, frames)
    for obj, was_hidden in curve_viewport_state.items():
        obj.hide_viewport = was_hidden
    set_linear_interpolation(placement)

    scene.frame_start = 1
    scene.frame_end = 200
    scene.frame_set(1)
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.length_unit = "MILLIMETERS"
    scene.unit_settings.scale_length = 0.001
    scene.camera = bpy.data.objects.get("Camera.009") or scene.camera
    scene["MGF_scene_source"] = "PyMouseStudio-fur-SceneOnly.blend"
    scene["MGF_pose_source"] = str(SOURCE_BLEND)
    scene["MGF_pose_transfer"] = "original source rig retained; per-frame whole-rig translation only"
    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_BLEND))

    cylinder_bounds = evaluated_bounds(bpy.data.objects[TARGET_CYLINDER])
    cylinder_radius = float(min(cylinder_bounds[1, :2] - cylinder_bounds[0, :2]) * 0.5)
    audit = {
        "format": "sceneonly_source_rig_pose_raw_mm_v1",
        "output_blend": str(OUTPUT_BLEND),
        "scene_source": "PyMouseStudio-fur-SceneOnly.blend",
        "pose_source": str(SOURCE_BLEND),
        "pose_mode": "source rig unchanged; per-frame whole-rig translation",
        "frames": [1, 200],
        "source_armature": source_armature.name,
        "source_mouse": source_mouse.name,
        "scene_scale": SCENE_SCALE,
        "raw_unit": "millimeter",
        "hidden_sceneonly_mouse_objects": len(target_mouse_objects),
        "hidden_legacy_mm_objects": len(legacy_mm_objects),
        "brown_guard_hair": guard_hair,
        "per_frame_translation_mm_min": translations.min(axis=0).tolist(),
        "per_frame_translation_mm_max": translations.max(axis=0).tolist(),
        "max_bbox_corner_radius_mm": max_radius_mm,
        "cylinder_radius_mm": cylinder_radius,
        "scene_conversion_root": scene_root.name,
        "pose_placement_root": placement.name,
    }
    OUTPUT_BLEND.with_suffix(".alignment.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
