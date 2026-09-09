#!/usr/bin/env python3
"""Install the exact rendered First200 mouse rig in the MARS homecage scene."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


DEFAULT_DONOR = Path(
    r"\\wsl.localhost\Ubuntu\home\haozong\projects\MouseGeoForge\BlenderPyMouseProjects\Cylinder\PyMouseStudio-fur-SceneOnly-MoSeqPose-Sample10-First200-mm.blend"
)
DEFAULT_OUTPUT = Path(
    r"\\wsl.localhost\Ubuntu\home\haozong\projects\MouseGeoForge\dataset\Homecage\PyMouseStudio-First200Rig-BlackFur-MARSHomecage-Multiview.blend"
)
SOURCE_MOUSE_COLLECTION = "MGF_MARS_Mouse"
DONOR_COLLECTIONS = (
    "Mouse_White",
    "PseudoKeypoints.001",
    "MGF_FullTrajectory_Controls",
)
TARGET_COLLECTION_NAMES = (
    "MGF_First200_Mouse",
    "MGF_First200_IK_Targets",
    "MGF_First200_Trajectory_Controls",
)
MESH_NAME = "MS_Mouse.001"
ARMATURE_NAME = "DeformationSystem.001"
DONOR_PLACEMENT = "MGF_SourcePose_PerFramePlacement"
OFFSET_NAME = "MGF_First200_Homecage_Offset"
AUDIT_FRAMES = (1, 25, 100, 200)


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor", type=Path, default=DEFAULT_DONOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--segment-label",
        default="Sample10-First200",
        help="Human-readable sampled trajectory segment stored in scene metadata.",
    )
    return parser.parse_args(args_after_dash())


def evaluated_bounds(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    bpy.context.view_layer.update()
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
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


def remove_collection_and_objects(collection: bpy.types.Collection) -> list[str]:
    objects = list(collection.all_objects)
    names = sorted(obj.name for obj in objects)
    for obj in objects:
        if obj.name in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
    if collection.name in bpy.data.collections:
        bpy.data.collections.remove(collection)
    return names


def material_copy(source: bpy.types.Material, name: str) -> bpy.types.Material:
    existing = bpy.data.materials.get(name)
    if existing is not None and existing.users == 0:
        bpy.data.materials.remove(existing)
    duplicate = source.copy()
    duplicate.name = name
    duplicate.use_fake_user = True
    duplicate["MGF_appearance_source"] = source.name
    return duplicate


def set_node_input(node: bpy.types.Node, name: str, value: object) -> None:
    socket = node.inputs.get(name)
    if socket is not None:
        socket.default_value = value


def tune_black_for_homecage_cycles(
    body: list[bpy.types.Material],
    coat: bpy.types.Material,
) -> None:
    body_shader = next(
        (
            node
            for node in body[0].node_tree.nodes
            if node.type == "BSDF_PRINCIPLED"
        ),
        None,
    )
    if body_shader is not None:
        set_node_input(body_shader, "Base Color", (0.0035, 0.0022, 0.0018, 1.0))
        set_node_input(body_shader, "Roughness", 0.68)
        set_node_input(body_shader, "Coat Weight", 0.02)
        set_node_input(body_shader, "Specular IOR Level", 0.25)

    hair = next(
        (
            node
            for node in coat.node_tree.nodes
            if node.type == "BSDF_HAIR_PRINCIPLED"
        ),
        None,
    )
    if hair is None:
        raise RuntimeError("C57 coat has no Principled Hair BSDF")
    hair.parametrization = "COLOR"
    set_node_input(hair, "Color", (0.0015, 0.0010, 0.0008, 1.0))
    set_node_input(hair, "Roughness", 0.58)
    set_node_input(hair, "Radial Roughness", 0.46)
    set_node_input(hair, "Coat", 0.03)
    set_node_input(hair, "Random Color", 0.06)
    set_node_input(hair, "Random Roughness", 0.15)
    coat["MGF_C57_homecage_cycles_tuned"] = True


def capture_homecage_black_appearance(
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
            raise RuntimeError(f"Homecage body slot {index} has no material")
        body.append(material_copy(slot.material, f"MGF_C57_First200_BodySlot_{index:02d}"))

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
        raise RuntimeError("The current Homecage C57 appearance is incomplete")
    return (
        body,
        material_copy(coat_source, "MGF_C57_First200_Coat"),
        material_copy(whisker_source, "MGF_C57_First200_Whiskers"),
        material_copy(mouth_source, "MGF_C57_First200_Mouth"),
    )


def apply_homecage_black_appearance(
    mouse: bpy.types.Collection,
    mesh: bpy.types.Object,
    body: list[bpy.types.Material],
    coat: bpy.types.Material,
    whiskers: bpy.types.Material,
    mouth: bpy.types.Material,
) -> None:
    if len(mesh.material_slots) != len(body):
        raise RuntimeError(
            f"Body material slots differ: First200={len(mesh.material_slots)}, Homecage={len(body)}"
        )
    for index, material in enumerate(body):
        mesh.material_slots[index].material = material
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


def append_first200_collections(
    donor: Path,
) -> tuple[bpy.types.Collection, bpy.types.Collection, bpy.types.Collection]:
    with bpy.data.libraries.load(str(donor), link=False) as (available, requested):
        missing = [name for name in DONOR_COLLECTIONS if name not in available.collections]
        if missing:
            raise KeyError(f"First200 donor is missing collections: {missing}")
        requested.collections = list(DONOR_COLLECTIONS)
    loaded = tuple(requested.collections)
    if any(collection is None for collection in loaded):
        raise RuntimeError("Failed to append all First200 collections")

    root = bpy.data.collections.get("MGF_MARS_Homecage")
    if root is None:
        root = bpy.context.scene.collection
    for collection, target_name in zip(loaded, TARGET_COLLECTION_NAMES, strict=True):
        collection.name = target_name
        root.children.link(collection)
    return loaded


def configure_controls(
    armature: bpy.types.Object,
    ik_targets: bpy.types.Collection,
    trajectory_controls: bpy.types.Collection,
) -> dict[str, int]:
    armature.hide_viewport = False
    armature.hide_render = False
    armature.show_in_front = True
    for collection in (ik_targets, trajectory_controls):
        collection.hide_viewport = False
        collection.hide_render = False
        for obj in list(collection.all_objects):
            obj.hide_viewport = False
            obj.hide_render = True
    return {
        "ik_target_objects": len(ik_targets.all_objects),
        "trajectory_control_objects": len(trajectory_controls.all_objects),
    }


def parent_donor_placement_to_offset(
    placement: bpy.types.Object,
) -> bpy.types.Object:
    if not placement.users_collection:
        bpy.context.scene.collection.objects.link(placement)
    offset = bpy.data.objects.new(OFFSET_NAME, None)
    bpy.context.scene.collection.objects.link(offset)
    offset.empty_display_type = "ARROWS"
    offset.empty_display_size = 35.0
    placement.parent = offset
    placement.matrix_parent_inverse = Matrix.Identity(4)
    offset["MGF_role"] = "constant homecage alignment outside the untouched First200 rig"
    return offset


def align_frame_one(
    mesh: bpy.types.Object,
    offset: bpy.types.Object,
    target_bounds: tuple[Vector, Vector],
) -> Vector:
    current_bounds = evaluated_bounds(mesh)
    target_min, target_max = target_bounds
    current_min, current_max = current_bounds
    target_center = 0.5 * (target_min + target_max)
    current_center = 0.5 * (current_min + current_max)
    delta = Vector(
        (
            target_center.x - current_center.x,
            target_center.y - current_center.y,
            target_min.z - current_min.z,
        )
    )
    offset.location = delta
    bpy.context.view_layer.update()
    return delta


def animation_audit(
    scene: bpy.types.Scene,
    mesh: bpy.types.Object,
) -> dict[str, dict[str, list[float]]]:
    records = {}
    for frame in AUDIT_FRAMES:
        scene.frame_set(frame)
        records[str(frame)] = bounds_payload(evaluated_bounds(mesh))
    return records


def main() -> None:
    args = parse_args()
    scene = bpy.context.scene
    scene.frame_set(1)
    bpy.context.view_layer.update()

    source_collection = bpy.data.collections.get(SOURCE_MOUSE_COLLECTION)
    if source_collection is None:
        raise KeyError(SOURCE_MOUSE_COLLECTION)
    source_mesh = next(
        obj
        for obj in source_collection.all_objects
        if obj.type == "MESH" and obj.name.startswith("MS_Mouse")
    )
    target_bounds = evaluated_bounds(source_mesh)
    body, coat, whiskers, mouth = capture_homecage_black_appearance(source_collection)
    tune_black_for_homecage_cycles(body, coat)
    removed = remove_collection_and_objects(source_collection)

    mouse, ik_targets, trajectory_controls = append_first200_collections(args.donor)
    mesh = bpy.data.objects.get(MESH_NAME)
    armature = bpy.data.objects.get(ARMATURE_NAME)
    placement = bpy.data.objects.get(DONOR_PLACEMENT)
    if mesh is None or armature is None or placement is None:
        raise KeyError(
            f"Missing First200 hierarchy: mesh={mesh}, armature={armature}, placement={placement}"
        )
    if placement.animation_data is None or placement.animation_data.action is None:
        raise RuntimeError("First200 placement animation was not appended")

    apply_homecage_black_appearance(
        mouse, mesh, body, coat, whiskers, mouth
    )

    offset = parent_donor_placement_to_offset(placement)
    scene.frame_set(1)
    bpy.context.view_layer.update()
    alignment = align_frame_one(mesh, offset, target_bounds)
    control_counts = configure_controls(armature, ik_targets, trajectory_controls)

    scene.frame_start = 1
    scene.frame_end = 200
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    audits = animation_audit(scene, mesh)
    scene.frame_set(1)
    bpy.context.view_layer.update()

    constraints = Counter(
        constraint.type
        for bone in armature.pose.bones
        for constraint in bone.constraints
    )
    hair_objects = sorted(obj.name for obj in mouse.all_objects if obj.type == "CURVES")
    hair_materials = sorted(
        {
            material.name
            for obj in mouse.all_objects
            if obj.type == "CURVES"
            for material in obj.data.materials
            if material is not None
        }
    )
    body_materials = [material.name for material in mesh.data.materials]

    scene["MGF_mouse_source"] = str(args.donor)
    scene["MGF_mouse_segment"] = args.segment_label
    scene["MGF_mouse_rig"] = f"exact {args.segment_label} rendered rig and model"
    scene["MGF_mouse_appearance"] = (
        f"Homecage C57 black materials on exact {args.segment_label} complete fur layers"
    )
    scene["MGF_first200_offset"] = offset.name
    scene["MGF_first200_internal_placement"] = placement.name

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output), check_existing=False)
    report = {
        "output": str(output),
        "donor": str(args.donor),
        "source_dataset": args.segment_label,
        "appearance_source": "current MARS homecage C57 black setup",
        "frame_range": [scene.frame_start, scene.frame_end],
        "mouse_collection": mouse.name,
        "mesh": mesh.name,
        "armature": armature.name,
        "bone_count": len(armature.pose.bones),
        "constraint_counts": dict(sorted(constraints.items())),
        "donor_placement": placement.name,
        "donor_placement_action": placement.animation_data.action.name,
        "homecage_offset": offset.name,
        "homecage_offset_translation_mm": [float(value) for value in alignment],
        "control_counts": control_counts,
        "body_materials": body_materials,
        "hair_materials": hair_materials,
        "hair_objects": hair_objects,
        "guard_hair_objects": [
            name for name in hair_objects if name.startswith("MGF_AnimatedMouse_Guard")
        ],
        "target_floor_bounds_mm": bounds_payload(target_bounds),
        "animation_bounds_mm": audits,
        "removed_mars_mouse_objects": removed,
        "render_resolution": [scene.render.resolution_x, scene.render.resolution_y],
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
