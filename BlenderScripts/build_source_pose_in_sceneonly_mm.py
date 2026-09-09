#!/usr/bin/env python3
"""Keep the complete MoSeq pose rig and replace its environment with SceneOnly."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


SCENE_BLEND = Path(
    r"\\wsl.localhost\Ubuntu\home\haozong\projects\MouseGeoForge\dataset\Cylinder\PyMouseStudio-fur-SceneOnly.blend"
)
OUTPUT_BLEND = Path(
    r"\\wsl.localhost\Ubuntu\home\haozong\projects\MouseGeoForge\dataset\Cylinder\PyMouseStudio-fur-SceneOnly-MoSeqPose-Sample10-First200-mm.blend"
)
SCENE_COLLECTIONS = ["Collection.001", "Collection.002"]
KEEP_SOURCE_COLLECTIONS = {"Mouse_White", "PseudoKeypoints.001", "MGF_FullTrajectory_Controls"}
SOURCE_MOUSE_COLLECTION = "Mouse_White"
SOURCE_MOUSE = "MS_Mouse.001"
BROWN_DONOR_COLLECTION = "Mouse_Brown"
BROWN_DONOR_MOUSE = "MS_Mouse.002"
SCENE_CYLINDER_PREFIX = "Part_GlassCylinder160mm.001"
SCENE_PLATFORM_PREFIX = "Part_AntiReflectiveGlass.001"
SCENE_CAMERA_PREFIX = "Camera.009"
SCENE_SCALE = 1000.0
RAW_MM_PLATFORM_CENTER_Z = 338.5
GROUND_CLEARANCE_MM = 0.5


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-blend", type=Path, default=SCENE_BLEND)
    parser.add_argument("--output", type=Path, default=OUTPUT_BLEND)
    parser.add_argument("--frame-start", type=int, default=1)
    parser.add_argument("--frame-end", type=int)
    return parser.parse_args(args_after_dash())


def evaluated_bounds(obj: bpy.types.Object) -> np.ndarray:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    xyz = np.asarray(
        [evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box],
        dtype=np.float64,
    )
    return np.stack((xyz.min(axis=0), xyz.max(axis=0)))


def append_scene_collections(scene_blend: Path) -> list[bpy.types.Collection]:
    with bpy.data.libraries.load(str(scene_blend), link=False) as (data_from, data_to):
        missing = [name for name in SCENE_COLLECTIONS if name not in data_from.collections]
        if missing:
            raise KeyError(f"Missing SceneOnly collections: {missing}")
        data_to.collections = SCENE_COLLECTIONS
    loaded = list(data_to.collections)
    for collection in loaded:
        if collection.name not in bpy.context.scene.collection.children:
            bpy.context.scene.collection.children.link(collection)
    return loaded


def all_collection_objects(collections: list[bpy.types.Collection]) -> set[bpy.types.Object]:
    objects: set[bpy.types.Object] = set()
    for collection in collections:
        objects.update(obj for obj in collection.all_objects if obj is not None)
    return objects


def find_imported_object(objects: set[bpy.types.Object], prefix: str) -> bpy.types.Object:
    matches = [obj for obj in objects if obj.name == prefix or obj.name.startswith(f"{prefix}.")]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one imported object for {prefix}, got {[obj.name for obj in matches]}")
    return matches[0]


def convert_imported_scene_to_mm(
    imported_objects: set[bpy.types.Object],
    cylinder: bpy.types.Object,
    platform: bpy.types.Object,
) -> bpy.types.Object:
    cylinder_bounds = evaluated_bounds(cylinder)
    platform_bounds = evaluated_bounds(platform)
    cylinder_center = (cylinder_bounds[0] + cylinder_bounds[1]) * 0.5
    platform_center_z = float((platform_bounds[0, 2] + platform_bounds[1, 2]) * 0.5)

    root = bpy.data.objects.new("MGF_SceneOnly_MeterToMillimeter", None)
    bpy.context.scene.collection.objects.link(root)
    for obj in imported_objects:
        if obj.parent is not None:
            continue
        world = obj.matrix_world.copy()
        obj.parent = root
        obj.matrix_world = world
    root.scale = (SCENE_SCALE,) * 3
    root.location = (
        -SCENE_SCALE * float(cylinder_center[0]),
        -SCENE_SCALE * float(cylinder_center[1]),
        RAW_MM_PLATFORM_CENTER_Z - SCENE_SCALE * platform_center_z,
    )

    for obj in imported_objects:
        if obj.type == "CAMERA":
            obj.data.clip_start *= SCENE_SCALE
            obj.data.clip_end *= SCENE_SCALE
            obj.data.dof.focus_distance *= SCENE_SCALE
        elif obj.type == "LIGHT" and obj.data.type != "SUN":
            obj.data.energy *= SCENE_SCALE * SCENE_SCALE
            obj.data.shadow_soft_size *= SCENE_SCALE
            if obj.data.type == "AREA":
                obj.data.size *= SCENE_SCALE
                obj.data.size_y *= SCENE_SCALE
    return root


def assign_single_material(obj: bpy.types.Object, material: bpy.types.Material) -> None:
    obj.data.materials.clear()
    obj.data.materials.append(material)


def retarget_modifier_objects(
    obj: bpy.types.Object,
    donor_mesh: bpy.types.Object,
    source_mesh: bpy.types.Object,
) -> None:
    for modifier in obj.modifiers:
        for key in modifier.keys():
            try:
                if modifier[key] == donor_mesh:
                    modifier[key] = source_mesh
            except (ReferenceError, TypeError):
                continue


def duplicate_guard_hair(
    donor_name: str,
    output_name: str,
    donor_mesh: bpy.types.Object,
    source_mesh: bpy.types.Object,
    destination: bpy.types.Collection,
) -> str:
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
    if donor.data.materials:
        assign_single_material(duplicate, donor.data.materials[0])
    duplicate.hide_viewport = False
    duplicate.hide_render = False
    return duplicate.name


def apply_brown_appearance() -> list[str]:
    source_mesh = bpy.data.objects[SOURCE_MOUSE]
    donor_mesh = bpy.data.objects[BROWN_DONOR_MOUSE]
    if len(source_mesh.material_slots) != len(donor_mesh.material_slots):
        raise RuntimeError("Animated and brown donor mesh material-slot counts differ")
    for index, slot in enumerate(donor_mesh.material_slots):
        source_mesh.material_slots[index].material = slot.material

    dense_material = bpy.data.objects["BaseFur.002"].data.materials[0]
    whisker_material = bpy.data.objects["Whisker.002"].data.materials[0]
    for name in (
        "BaseFur.001",
        "EarFur.001",
        "EyesFur.001",
        "LegFur.001",
        "Mouth.001",
        "TailFur.001",
    ):
        assign_single_material(bpy.data.objects[name], dense_material)
    assign_single_material(bpy.data.objects["Whisker.001"], whisker_material)

    destination = bpy.data.collections[SOURCE_MOUSE_COLLECTION]
    return [
        duplicate_guard_hair(
            "BaseFur.003", "MGF_AnimatedMouse_GuardShort", donor_mesh, source_mesh, destination
        ),
        duplicate_guard_hair(
            "BaseFur.004", "MGF_AnimatedMouse_GuardLong", donor_mesh, source_mesh, destination
        ),
    ]


def hide_old_scene_collections(original_root_collections: list[bpy.types.Collection]) -> int:
    hidden: set[bpy.types.Object] = set()
    for collection in original_root_collections:
        keep = collection.name in KEEP_SOURCE_COLLECTIONS
        for obj in collection.all_objects:
            if obj is None:
                continue
            if keep:
                if collection.name != SOURCE_MOUSE_COLLECTION:
                    obj.hide_render = True
                continue
            obj.hide_viewport = True
            obj.hide_render = True
            hidden.add(obj)
    return len(hidden)


def create_complete_pose_placement(original_objects: set[bpy.types.Object]) -> bpy.types.Object:
    placement = bpy.data.objects.new("MGF_SourcePose_PerFramePlacement", None)
    bpy.context.scene.collection.objects.link(placement)
    for obj in original_objects:
        if obj.parent is not None:
            continue
        world = obj.matrix_world.copy()
        obj.parent = placement
        obj.matrix_world = world
    return placement


def align_pose_per_frame(
    mouse: bpy.types.Object,
    placement: bpy.types.Object,
    cylinder: bpy.types.Object,
    platform: bpy.types.Object,
    frames: range,
) -> tuple[np.ndarray, float, np.ndarray]:
    cylinder_bounds = evaluated_bounds(cylinder)
    platform_bounds = evaluated_bounds(platform)
    center_xy = (cylinder_bounds[0, :2] + cylinder_bounds[1, :2]) * 0.5
    platform_top = float(platform_bounds[1, 2])
    cylinder_radius = float(min(cylinder_bounds[1, :2] - cylinder_bounds[0, :2]) * 0.5)

    placement.animation_data_clear()
    translations = []
    frame_audits = []
    max_radius = 0.0
    for frame in frames:
        bpy.context.scene.frame_set(frame)
        placement.location = (0.0, 0.0, 0.0)
        bpy.context.view_layer.update()
        before = evaluated_bounds(mouse)
        mouse_center_xy = (before[0, :2] + before[1, :2]) * 0.5
        translation = np.asarray(
            [
                center_xy[0] - mouse_center_xy[0],
                center_xy[1] - mouse_center_xy[1],
                platform_top + GROUND_CLEARANCE_MM - before[0, 2],
            ],
            dtype=np.float64,
        )
        placement.location = translation.tolist()
        placement.keyframe_insert("location", frame=frame)
        bpy.context.view_layer.update()
        after = evaluated_bounds(mouse)
        corners_xy = np.asarray(
            [[x, y] for x in (after[0, 0], after[1, 0]) for y in (after[0, 1], after[1, 1])]
        )
        radius = float(np.linalg.norm(corners_xy - center_xy, axis=1).max())
        max_radius = max(max_radius, radius)
        translations.append(translation)
        frame_audits.append(
            [
                frame,
                *after[0].tolist(),
                *after[1].tolist(),
                radius,
            ]
        )
    return np.asarray(translations), max_radius, np.asarray(frame_audits)


def set_linear_interpolation(obj: bpy.types.Object) -> None:
    action = obj.animation_data.action if obj.animation_data else None
    if not action:
        return
    for layer in action.layers:
        for strip in layer.strips:
            for channelbag in getattr(strip, "channelbags", []):
                for curve in channelbag.fcurves:
                    for point in curve.keyframe_points:
                        point.interpolation = "LINEAR"


def main() -> None:
    args = parse_args()
    scene = bpy.context.scene
    source_pose_blend = Path(bpy.data.filepath)
    frame_end = args.frame_end if args.frame_end is not None else scene.frame_end
    if args.frame_start < 1 or frame_end < args.frame_start:
        raise ValueError(f"Invalid frame range: {args.frame_start}..{frame_end}")
    scene.frame_set(1)
    bpy.context.view_layer.update()
    original_objects = set(bpy.data.objects)
    original_root_collections = list(scene.collection.children)

    imported_collections = append_scene_collections(args.scene_blend)
    imported_objects = all_collection_objects(imported_collections)
    cylinder = find_imported_object(imported_objects, SCENE_CYLINDER_PREFIX)
    platform = find_imported_object(imported_objects, SCENE_PLATFORM_PREFIX)
    camera = find_imported_object(imported_objects, SCENE_CAMERA_PREFIX)
    scene_root = convert_imported_scene_to_mm(imported_objects, cylinder, platform)

    hidden_old_objects = hide_old_scene_collections(original_root_collections)
    guard_hair = apply_brown_appearance()
    placement = create_complete_pose_placement(original_objects)
    mouse = bpy.data.objects[SOURCE_MOUSE]

    animated_curves = [
        obj for obj in bpy.data.collections[SOURCE_MOUSE_COLLECTION].all_objects if obj and obj.type == "CURVES"
    ]
    curve_states = {obj: obj.hide_viewport for obj in animated_curves}
    for obj in animated_curves:
        obj.hide_viewport = True
    translations, max_radius, frame_audits = align_pose_per_frame(
        mouse, placement, cylinder, platform, range(args.frame_start, frame_end + 1)
    )
    for obj, was_hidden in curve_states.items():
        obj.hide_viewport = was_hidden
    set_linear_interpolation(placement)

    scene.frame_start = args.frame_start
    scene.frame_end = frame_end
    scene.frame_set(args.frame_start)
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.length_unit = "MILLIMETERS"
    scene.unit_settings.scale_length = 0.001
    scene.camera = camera
    scene["MGF_scene_source"] = str(args.scene_blend)
    scene["MGF_pose_source"] = bpy.data.filepath
    scene["MGF_pose_transfer"] = "source project retained; per-frame whole-project translation only"

    cylinder_bounds = evaluated_bounds(cylinder)
    cylinder_radius = float(min(cylinder_bounds[1, :2] - cylinder_bounds[0, :2]) * 0.5)
    if not np.isfinite(frame_audits).all():
        raise RuntimeError("Non-finite pose audit values")
    if max_radius >= cylinder_radius:
        raise RuntimeError(f"Mouse exceeds cylinder: {max_radius:.3f} >= {cylinder_radius:.3f} mm")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output), check_existing=False)
    sampled_audit_frames = sorted(
        {
            args.frame_start,
            args.frame_start + (frame_end - args.frame_start) // 4,
            args.frame_start + (frame_end - args.frame_start) // 2,
            args.frame_start + 3 * (frame_end - args.frame_start) // 4,
            frame_end,
        }
    )
    audit = {
        "format": "sceneonly_complete_source_pose_raw_mm_v1",
        "output_blend": str(output),
        "scene_source": str(args.scene_blend),
        "pose_source": source_pose_blend.name,
        "pose_mode": "complete source project retained; per-frame whole-project translation only",
        "frames": [args.frame_start, frame_end],
        "scene_scale": SCENE_SCALE,
        "raw_unit": "millimeter",
        "scene_camera": camera.name,
        "scene_cylinder": cylinder.name,
        "scene_platform": platform.name,
        "hidden_old_scene_objects": hidden_old_objects,
        "brown_guard_hair": guard_hair,
        "translation_mm_min": translations.min(axis=0).tolist(),
        "translation_mm_max": translations.max(axis=0).tolist(),
        "max_bbox_corner_radius_mm": max_radius,
        "cylinder_radius_mm": cylinder_radius,
        "scene_conversion_root": scene_root.name,
        "pose_placement_root": placement.name,
        "sample_frame_audits": {
            str(frame): frame_audits[frame - args.frame_start].tolist()
            for frame in sampled_audit_frames
        },
    }
    output.with_suffix(".alignment.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
