#!/usr/bin/env python3
"""Replace the MARS homecage mouse with the Keypoint-MoSeq IK mouse rig."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from math import radians
from pathlib import Path

import bpy
from mathutils import Vector


DEFAULT_DONOR = Path(
    r"\\wsl.localhost\Ubuntu\home\haozong\projects\MouseGeoForge\dataset\Cylinder\PyMouseStudio-WhiteFur-Controller-moseq.blend"
)
DEFAULT_OUTPUT = Path(
    r"\\wsl.localhost\Ubuntu\home\haozong\projects\MouseGeoForge\dataset\Homecage\PyMouseStudio-KeypointMoseqIK-BlackFur-MARSHomecage-Multiview.blend"
)
SOURCE_COLLECTION = "MGF_MARS_Mouse"
DONOR_MOUSE_COLLECTION = "Mouse_White"
DONOR_KEYPOINT_COLLECTION = "PseudoKeypoints.001"
TARGET_MOUSE_COLLECTION = "MGF_KeypointMoseq_IK_Mouse"
TARGET_KEYPOINT_COLLECTION = "MGF_KeypointMoseq_PseudoKeypoints"
SOURCE_MESH = "MS_Mouse.001"
SOURCE_ARMATURE = "DeformationSystem.001"


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor", type=Path, default=DEFAULT_DONOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--rotation-z",
        type=float,
        default=90.0,
        help="Whole-rig rotation used to match the previous mouse heading.",
    )
    return parser.parse_args(args_after_dash())


def evaluated_bounds(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    points = [evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box]
    if not all(math.isfinite(value) for point in points for value in point):
        raise RuntimeError(f"Non-finite evaluated bounds for {obj.name}")
    if max(abs(value) for point in points for value in point) > 1_000_000.0:
        raise RuntimeError(f"Implausible evaluated bounds for {obj.name}")
    return (
        Vector(tuple(min(point[axis] for point in points) for axis in range(3))),
        Vector(tuple(max(point[axis] for point in points) for axis in range(3))),
    )


def bounds_payload(bounds: tuple[Vector, Vector]) -> dict[str, list[float]]:
    minimum, maximum = bounds
    return {
        "min": [float(value) for value in minimum],
        "max": [float(value) for value in maximum],
        "dimensions": [float(value) for value in maximum - minimum],
    }


def material_copy(source: bpy.types.Material, name: str) -> bpy.types.Material:
    existing = bpy.data.materials.get(name)
    if existing is not None and existing.users == 0:
        bpy.data.materials.remove(existing)
    duplicate = source.copy()
    duplicate.name = name
    duplicate.use_fake_user = True
    duplicate["MGF_appearance_source"] = source.name
    return duplicate


def capture_black_appearance(
    source_collection: bpy.types.Collection,
) -> tuple[list[bpy.types.Material], bpy.types.Material, bpy.types.Material, bpy.types.Material]:
    mesh = next(
        obj
        for obj in source_collection.all_objects
        if obj.type == "MESH" and obj.name.startswith("MS_Mouse")
    )
    body = []
    for index, slot in enumerate(mesh.material_slots):
        if slot.material is None:
            raise RuntimeError(f"Source body slot {index} has no material")
        body.append(material_copy(slot.material, f"MGF_C57_MoSeq_BodySlot_{index:02d}"))

    coat_source = bpy.data.materials.get("MGF_C57_Coat")
    whisker_source = bpy.data.materials.get("MGF_C57_Whiskers")
    mouth_source = next(
        (
            slot.material
            for obj in source_collection.all_objects
            if obj.type == "CURVES" and obj.name.startswith("Mouth")
            for slot in obj.material_slots
            if slot.material is not None
        ),
        None,
    )
    if coat_source is None or whisker_source is None or mouth_source is None:
        raise RuntimeError("The current C57 coat, whisker, or mouth material is missing")
    return (
        body,
        material_copy(coat_source, "MGF_C57_MoSeq_Coat"),
        material_copy(whisker_source, "MGF_C57_MoSeq_Whiskers"),
        material_copy(mouth_source, "MGF_C57_MoSeq_Mouth"),
    )


def remove_collection_and_objects(collection: bpy.types.Collection) -> list[str]:
    names = sorted(obj.name for obj in collection.all_objects)
    for obj in list(collection.all_objects):
        if obj.name in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
    if collection.name in bpy.data.collections:
        bpy.data.collections.remove(collection)
    return names


def append_donor_collections(
    donor: Path,
) -> tuple[bpy.types.Collection, bpy.types.Collection]:
    with bpy.data.libraries.load(str(donor), link=False) as (available, requested):
        required = [DONOR_MOUSE_COLLECTION, DONOR_KEYPOINT_COLLECTION]
        missing = [name for name in required if name not in available.collections]
        if missing:
            raise KeyError(f"MoSeq donor is missing collections: {missing}")
        requested.collections = required
    mouse, keypoints = requested.collections
    if mouse is None or keypoints is None:
        raise RuntimeError("Failed to append the MoSeq IK collections")
    mouse.name = TARGET_MOUSE_COLLECTION
    keypoints.name = TARGET_KEYPOINT_COLLECTION
    root = bpy.data.collections.get("MGF_MARS_Homecage")
    if root is None:
        root = bpy.context.scene.collection
    root.children.link(mouse)
    root.children.link(keypoints)
    return mouse, keypoints


def assign_black_appearance(
    mouse: bpy.types.Collection,
    body_materials: list[bpy.types.Material],
    coat: bpy.types.Material,
    whiskers: bpy.types.Material,
    mouth: bpy.types.Material,
) -> tuple[bpy.types.Object, list[str]]:
    mesh = bpy.data.objects.get(SOURCE_MESH)
    if mesh is None or mouse.all_objects.get(mesh.name) is None:
        raise KeyError(f"Appended mouse does not contain {SOURCE_MESH}")
    if len(mesh.material_slots) != len(body_materials):
        raise RuntimeError(
            f"Body material slots differ: donor={len(mesh.material_slots)}, source={len(body_materials)}"
        )
    for index, material in enumerate(body_materials):
        mesh.material_slots[index].material = material

    assigned = []
    # Freeze the collection because changing Curves materials can invalidate
    # Blender's live all_objects iterator.
    for obj in list(mouse.all_objects):
        if obj.type != "CURVES":
            continue
        obj.data.materials.clear()
        if obj.name.startswith("Whisker"):
            obj.data.materials.append(whiskers)
        elif obj.name.startswith("Mouth"):
            obj.data.materials.append(mouth)
        else:
            obj.data.materials.append(coat)
        obj.hide_render = False
        obj.hide_viewport = False
        assigned.append(obj.name)
    if len(assigned) < 7:
        raise RuntimeError(f"Expected at least seven bound hair objects, found {assigned}")
    return mesh, sorted(assigned)


def create_placement(
    objects: set[bpy.types.Object],
    rotation_z_degrees: float,
) -> bpy.types.Object:
    placement = bpy.data.objects.new("MGF_KeypointMoseq_IK_Placement", None)
    bpy.context.scene.collection.objects.link(placement)
    armature = bpy.data.objects.get(SOURCE_ARMATURE)
    if armature is None:
        raise KeyError(SOURCE_ARMATURE)
    constraint_states = [
        (constraint, constraint.mute)
        for bone in armature.pose.bones
        for constraint in bone.constraints
    ]
    for constraint, _ in constraint_states:
        constraint.mute = True
    bpy.context.view_layer.update()

    roots = [obj for obj in objects if obj.parent is None]
    # Snapshot every root before changing the dependency graph. Reading a later
    # IK target after parenting an earlier one can otherwise evaluate a partly
    # migrated rig and poison its matrix with NaNs.
    root_world = {obj: obj.matrix_world.copy() for obj in roots}
    for obj in roots:
        obj.parent = placement
        obj.matrix_world = root_world[obj]
    placement.rotation_euler.z = radians(rotation_z_degrees)
    bpy.context.view_layer.update()
    for constraint, was_muted in constraint_states:
        constraint.mute = was_muted
    placement.empty_display_type = "ARROWS"
    placement.empty_display_size = 35.0
    placement["MGF_role"] = "whole-rig placement only; pose with DeformationSystem.001"
    bpy.context.view_layer.update()
    return placement


def align_to_previous_mouse(
    mesh: bpy.types.Object,
    placement: bpy.types.Object,
    previous_bounds: tuple[Vector, Vector],
) -> tuple[Vector, tuple[Vector, Vector]]:
    current_bounds = evaluated_bounds(mesh)
    previous_min, previous_max = previous_bounds
    current_min, current_max = current_bounds
    previous_center = 0.5 * (previous_min + previous_max)
    current_center = 0.5 * (current_min + current_max)
    delta = Vector(
        (
            previous_center.x - current_center.x,
            previous_center.y - current_center.y,
            previous_min.z - current_min.z,
        )
    )
    placement.location += delta
    bpy.context.view_layer.update()
    return delta, evaluated_bounds(mesh)


def configure_pose_rig(
    mouse: bpy.types.Collection,
    keypoints: bpy.types.Collection,
) -> tuple[bpy.types.Object, dict[str, int], int]:
    armature = bpy.data.objects.get(SOURCE_ARMATURE)
    if armature is None or mouse.all_objects.get(armature.name) is None:
        raise KeyError(f"Appended mouse does not contain {SOURCE_ARMATURE}")
    armature.hide_viewport = False
    armature.hide_render = False
    armature.show_in_front = True
    keypoint_count = 0
    for obj in keypoints.all_objects:
        obj.hide_render = True
        obj.hide_viewport = False
        keypoint_count += 1
    constraints = Counter(
        constraint.type
        for bone in armature.pose.bones
        for constraint in bone.constraints
    )
    custom_shapes = sum(1 for bone in armature.pose.bones if bone.custom_shape is not None)
    return armature, dict(sorted(constraints.items())), custom_shapes


def main() -> None:
    args = parse_args()
    scene = bpy.context.scene
    scene.frame_set(1)
    bpy.context.view_layer.update()

    source_collection = bpy.data.collections.get(SOURCE_COLLECTION)
    if source_collection is None:
        raise KeyError(f"Current scene is missing {SOURCE_COLLECTION}")
    source_mesh = next(
        obj
        for obj in source_collection.all_objects
        if obj.type == "MESH" and obj.name.startswith("MS_Mouse")
    )
    previous_bounds = evaluated_bounds(source_mesh)
    body, coat, whiskers, mouth = capture_black_appearance(source_collection)
    removed = remove_collection_and_objects(source_collection)

    mouse, keypoints = append_donor_collections(args.donor)
    target_mesh, fur_objects = assign_black_appearance(
        mouse, body, coat, whiskers, mouth
    )
    imported_objects = set(mouse.all_objects) | set(keypoints.all_objects)
    placement = create_placement(imported_objects, args.rotation_z)
    translation, final_bounds = align_to_previous_mouse(
        target_mesh, placement, previous_bounds
    )
    armature, constraints, custom_shapes = configure_pose_rig(mouse, keypoints)

    scene.frame_set(1)
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene["MGF_mouse_rig"] = "Keypoint-MoSeq IK DeformationSystem.001"
    scene["MGF_mouse_appearance"] = "C57 warm-black settings migrated from MARS mouse"
    scene["MGF_mouse_pose_donor"] = str(args.donor)
    scene["MGF_pose_root"] = placement.name
    scene["MGF_pose_instruction"] = (
        "Use DeformationSystem.001 pose controls; use placement only for whole-body alignment."
    )

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output), check_existing=False)
    report = {
        "output": str(output),
        "donor": str(args.donor),
        "mouse_collection": mouse.name,
        "keypoint_collection": keypoints.name,
        "mesh": target_mesh.name,
        "armature": armature.name,
        "bone_count": len(armature.pose.bones),
        "constraint_counts": constraints,
        "custom_shape_bones": custom_shapes,
        "fur_objects": fur_objects,
        "fur_materials": {
            "coat": coat.name,
            "whiskers": whiskers.name,
            "mouth": mouth.name,
        },
        "previous_mouse_bounds_mm": bounds_payload(previous_bounds),
        "final_mouse_bounds_mm": bounds_payload(final_bounds),
        "whole_rig_rotation_z_degrees": args.rotation_z,
        "whole_rig_translation_mm": [float(value) for value in translation],
        "removed_mars_mouse_objects": removed,
        "render_resolution": [scene.render.resolution_x, scene.render.resolution_y],
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
