#!/usr/bin/env python3
"""Adaptively sample an animation and render multi-view bone-over-mesh audits."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


ARMATURE_NAME = "DeformationSystem.001"
MESH_NAME = "MS_Mouse.001"
AUDIT_COLLECTION = "MGF_BonePose_Audit"

AXIAL_BONES = (
    "RootPart1_M", "RootPart2_M", "Spine1_M", "Spine2_M", "Spine3_M",
    "Spine4_M", "Chest_M", "Neck3_M", "Neck1_M", "Neck2_M",
    "Head1_M", "HeadEnd_M",
)
TAIL_BONES = ("TailStart_M",) + tuple(f"Tail{i}_M" for i in range(12))
LEFT_BONES = (
    "Scapula_L", "Wrist_L", "Fingers1_L",
    "Hip_L", "Knee_L", "Ankle_L", "Toes1_L",
)
RIGHT_BONES = (
    "Scapula_R", "Wrist_R", "Fingers1_R",
    "Hip_R", "Knee_R", "Ankle_R", "Toes1_R",
)
AUDIT_BONES = AXIAL_BONES + TAIL_BONES + LEFT_BONES + RIGHT_BONES
FEATURE_BONES = (
    "RootPart1_M", "Spine2_M", "Chest_M", "Neck1_M", "Head1_M",
    "Scapula_L", "Wrist_L", "Fingers1_L", "Scapula_R", "Wrist_R",
    "Fingers1_R", "Hip_L", "Knee_L", "Ankle_L", "Toes1_L",
    "Hip_R", "Knee_R", "Ankle_R", "Toes1_R", "TailStart_M",
    "Tail3_M", "Tail7_M", "Tail11_M",
)
CHAIN_PAIRS = (
    ("RootPart1_M", "RootPart2_M"), ("RootPart2_M", "Spine1_M"),
    ("Spine1_M", "Spine2_M"), ("Spine2_M", "Spine3_M"),
    ("Spine3_M", "Spine4_M"), ("Spine4_M", "Chest_M"),
    ("Chest_M", "Neck3_M"), ("Neck3_M", "Neck1_M"),
    ("Neck1_M", "Neck2_M"), ("Neck2_M", "Head1_M"),
    ("Scapula_L", "Wrist_L"), ("Wrist_L", "Fingers1_L"),
    ("Scapula_R", "Wrist_R"), ("Wrist_R", "Fingers1_R"),
    ("Hip_L", "Knee_L"), ("Knee_L", "Ankle_L"),
    ("Ankle_L", "Toes1_L"), ("Hip_R", "Knee_R"),
    ("Knee_R", "Ankle_R"), ("Ankle_R", "Toes1_R"),
) + tuple((TAIL_BONES[i], TAIL_BONES[i + 1]) for i in range(len(TAIL_BONES) - 1))

VIEW_DIRECTIONS = {
    "top": Vector((0.0, 0.0, 1.0)),
    "side_x": Vector((1.0, 0.0, 0.0)),
    "side_y": Vector((0.0, 1.0, 0.0)),
    "oblique": Vector((1.0, -1.0, 0.75)).normalized(),
}


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--views", default="top,side_x,side_y,oblique")
    parser.add_argument(
        "--extra-frames",
        default="",
        help="Comma-separated diagnostic frames added after adaptive selection.",
    )
    parser.add_argument(
        "--only-frames",
        default="",
        help="Comma-separated frames that replace adaptive selection for targeted audits.",
    )
    return parser.parse_args(args_after_dash())


def world_joint(armature: bpy.types.Object, bone_name: str, tail: bool = False) -> Vector:
    bone = armature.pose.bones[bone_name]
    return armature.matrix_world @ (bone.tail if tail else bone.head)


def pose_feature(armature: bpy.types.Object) -> list[float]:
    points = []
    for name in FEATURE_BONES:
        points.append(world_joint(armature, name))
    points.append(world_joint(armature, "HeadEnd_M", tail=True))
    points.append(world_joint(armature, "Tail11_M", tail=True))
    center = world_joint(armature, "RootPart1_M")
    scale = max((world_joint(armature, "HeadEnd_M", True) - center).length, 1.0)
    return [coordinate / scale for point in points for coordinate in (point - center)]


def rms_distance(first: list[float], second: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)) / len(first))


def adaptive_frames(
    scene: bpy.types.Scene,
    armature: bpy.types.Object,
    sample_count: int,
) -> tuple[list[int], dict[int, float], list[list[float]]]:
    frames = list(range(scene.frame_start, scene.frame_end + 1))
    features = []
    for frame in frames:
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        features.append(pose_feature(armature))

    steps = [0.0]
    for index in range(1, len(features)):
        steps.append(rms_distance(features[index - 1], features[index]))
    cumulative = []
    total = 0.0
    for step in steps:
        total += step
        cumulative.append(total)

    if sample_count >= len(frames) or total <= 1e-12:
        selected = frames
    else:
        selected_indices = {0, len(frames) - 1}
        for rank in range(1, sample_count - 1):
            target = total * rank / (sample_count - 1)
            index = min(range(len(cumulative)), key=lambda i: abs(cumulative[i] - target))
            selected_indices.add(index)
        while len(selected_indices) < sample_count:
            candidates = [i for i in range(1, len(frames) - 1) if i not in selected_indices]
            if not candidates:
                break
            best = max(
                candidates,
                key=lambda i: steps[i] * min(abs(i - j) for j in selected_indices),
            )
            selected_indices.add(best)
        selected = [frames[index] for index in sorted(selected_indices)]
    return selected, dict(zip(frames, steps)), features


def material(name: str, color: tuple[float, float, float, float], emission: bool = False):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.use_nodes = True
    node = next(
        (candidate for candidate in mat.node_tree.nodes if candidate.type == "BSDF_PRINCIPLED"),
        None,
    )
    if node is None:
        raise RuntimeError(f"Material {name} has no Principled BSDF node")
    if emission:
        node.inputs["Base Color"].default_value = color
        node.inputs["Emission Color"].default_value = color
        node.inputs["Emission Strength"].default_value = 1.4
        node.inputs["Roughness"].default_value = 0.55
    else:
        node.inputs["Base Color"].default_value = color
        node.inputs["Alpha"].default_value = color[3]
        node.inputs["Roughness"].default_value = 0.8
        try:
            mat.surface_render_method = "DITHERED"
        except (AttributeError, TypeError):
            pass
    return mat


def clear_collection(collection: bpy.types.Collection) -> None:
    for obj in list(collection.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def link_only(obj: bpy.types.Object, collection: bpy.types.Collection) -> None:
    for source in list(obj.users_collection):
        source.objects.unlink(obj)
    collection.objects.link(obj)


def cylinder_between(
    start: Vector,
    end: Vector,
    radius: float,
    mat: bpy.types.Material,
    collection: bpy.types.Collection,
    name: str,
) -> None:
    delta = end - start
    if delta.length < 1e-6:
        return
    bpy.ops.mesh.primitive_cylinder_add(vertices=10, radius=radius, depth=delta.length)
    obj = bpy.context.object
    obj.name = name
    obj.location = (start + end) * 0.5
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = Vector((0.0, 0.0, 1.0)).rotation_difference(delta.normalized())
    obj.data.materials.append(mat)
    link_only(obj, collection)


def sphere(
    location: Vector,
    radius: float,
    mat: bpy.types.Material,
    collection: bpy.types.Collection,
    name: str,
) -> None:
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=radius, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(mat)
    link_only(obj, collection)


def build_bone_proxies(
    armature: bpy.types.Object,
    collection: bpy.types.Collection,
    materials: dict[str, bpy.types.Material],
) -> None:
    clear_collection(collection)
    joint_locations: dict[tuple[int, int, int], Vector] = {}
    for name in AUDIT_BONES:
        if name not in armature.pose.bones:
            continue
        start = world_joint(armature, name)
        end = world_joint(armature, name, tail=True)
        group = "tail" if name in TAIL_BONES else "left" if name.endswith("_L") else "right" if name.endswith("_R") else "axial"
        cylinder_between(start, end, 0.9, materials[group], collection, f"AuditBone_{name}")
        for point in (start, end):
            key = tuple(round(value * 10) for value in point)
            joint_locations.setdefault(key, point)
    for index, point in enumerate(joint_locations.values()):
        sphere(point, 1.35, materials["joint"], collection, f"AuditJoint_{index:03d}")


def mesh_bounds(mesh: bpy.types.Object) -> tuple[Vector, Vector]:
    evaluated = mesh.evaluated_get(bpy.context.evaluated_depsgraph_get())
    points = [evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box]
    low = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
    high = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
    return low, high


def fit_camera(
    camera: bpy.types.Object,
    direction: Vector,
    low: Vector,
    high: Vector,
    aspect: float,
) -> None:
    center = (low + high) * 0.5
    radius = (high - low).length
    camera.location = center + direction.normalized() * max(radius * 2.5, 250.0)
    camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    corners = [
        Vector((x, y, z))
        for x in (low.x, high.x)
        for y in (low.y, high.y)
        for z in (low.z, high.z)
    ]
    inverse = camera.matrix_world.inverted()
    local = [inverse @ point for point in corners]
    width = max(point.x for point in local) - min(point.x for point in local)
    height = max(point.y for point in local) - min(point.y for point in local)
    camera.data.ortho_scale = max(height * 1.28, width / aspect * 1.28, 40.0)


def numeric_audit(
    scene: bpy.types.Scene,
    armature: bpy.types.Object,
    selected: list[int],
    motion_steps: dict[int, float],
) -> dict:
    length_samples = {name: [] for name in AUDIT_BONES}
    max_gap = {f"{parent}->{child}": 0.0 for parent, child in CHAIN_PAIRS}
    frame_rows = []
    for frame in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        for name in AUDIT_BONES:
            length_samples[name].append((world_joint(armature, name, True) - world_joint(armature, name)).length)
        for parent, child in CHAIN_PAIRS:
            gap = (world_joint(armature, parent, True) - world_joint(armature, child)).length
            key = f"{parent}->{child}"
            max_gap[key] = max(max_gap[key], gap)
        if frame in selected:
            frame_rows.append({"frame": frame, "motion_step": motion_steps[frame]})

    length_drift = {}
    for name, values in length_samples.items():
        median = sorted(values)[len(values) // 2]
        min_value = min(values)
        max_value = max(values)
        length_drift[name] = {
            "min_mm": min_value,
            "min_frame": scene.frame_start + values.index(min_value),
            "max_mm": max_value,
            "max_frame": scene.frame_start + values.index(max_value),
            "max_relative_drift": max(abs(value / median - 1.0) for value in values) if median else 0.0,
        }
    return {
        "selected_frames": frame_rows,
        "top_motion_change_frames": [
            {"frame": frame, "motion_step": score}
            for frame, score in sorted(motion_steps.items(), key=lambda item: item[1], reverse=True)[:12]
        ],
        "bone_length_drift": length_drift,
        "max_chain_gap_mm": max_gap,
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    armature = bpy.data.objects.get(ARMATURE_NAME)
    mesh = bpy.data.objects.get(MESH_NAME)
    if armature is None or mesh is None:
        raise KeyError(f"Missing armature={armature} or mesh={mesh}")

    missing = [name for name in AUDIT_BONES if name not in armature.pose.bones]
    if missing:
        raise KeyError(f"Missing audit bones: {missing}")
    original_viewport_hide = {obj.name: obj.hide_viewport for obj in bpy.data.objects}
    for obj in bpy.data.objects:
        # Hair curves dominate dependency-graph updates. IK target curves and
        # controller empties must remain active or the audited pose changes.
        if obj.type == "CURVES":
            obj.hide_viewport = True
    selected, motion_steps, _ = adaptive_frames(scene, armature, args.sample_count)
    only_frames = {
        int(value.strip())
        for value in args.only_frames.split(",")
        if value.strip()
    }
    extra_frames = {
        int(value.strip())
        for value in args.extra_frames.split(",")
        if value.strip()
    }
    requested_frames = only_frames | extra_frames
    invalid_extra = [
        frame for frame in requested_frames
        if not scene.frame_start <= frame <= scene.frame_end
    ]
    if invalid_extra:
        raise ValueError(f"Extra frames outside scene range: {invalid_extra}")
    selected = sorted(only_frames if only_frames else set(selected) | extra_frames)
    audit = numeric_audit(scene, armature, selected, motion_steps)

    audit_collection = bpy.data.collections.get(AUDIT_COLLECTION)
    if audit_collection is None:
        audit_collection = bpy.data.collections.new(AUDIT_COLLECTION)
        scene.collection.children.link(audit_collection)

    materials = {
        "body": material("MGF_Audit_Body", (0.42, 0.44, 0.47, 0.22)),
        "axial": material("MGF_Audit_Axial", (1.0, 0.62, 0.05, 1.0), True),
        "tail": material("MGF_Audit_Tail", (0.18, 0.78, 0.30, 1.0), True),
        "left": material("MGF_Audit_Left", (0.04, 0.78, 0.92, 1.0), True),
        "right": material("MGF_Audit_Right", (0.92, 0.12, 0.52, 1.0), True),
        "joint": material("MGF_Audit_Joint", (1.0, 1.0, 1.0, 1.0), True),
    }

    original_materials = [slot.material for slot in mesh.material_slots]
    for slot in mesh.material_slots:
        slot.material = materials["body"]
    original_hide = {obj.name: obj.hide_render for obj in bpy.data.objects}
    keep = {mesh.name, armature.name}
    for obj in bpy.data.objects:
        obj.hide_render = obj.name not in keep and obj.name not in audit_collection.objects
    armature.hide_render = True
    mesh.hide_render = False

    camera_data = bpy.data.cameras.new("MGF_Audit_Camera")
    camera_data.type = "ORTHO"
    camera = bpy.data.objects.new("MGF_Audit_Camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    engine_items = {
        item.identifier
        for item in scene.bl_rna.properties["render"].fixed_type.properties["engine"].enum_items
    }
    scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engine_items else "BLENDER_EEVEE"
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.world.color = (0.055, 0.055, 0.055)
    scene.view_settings.look = "Medium High Contrast"

    requested_views = [name.strip() for name in args.views.split(",") if name.strip()]
    unknown = [name for name in requested_views if name not in VIEW_DIRECTIONS]
    if unknown:
        raise ValueError(f"Unknown views: {unknown}")

    renders = []
    for frame in selected:
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        build_bone_proxies(armature, audit_collection, materials)
        for obj in audit_collection.objects:
            obj.hide_render = False
        low, high = mesh_bounds(mesh)
        for view_name in requested_views:
            fit_camera(camera, VIEW_DIRECTIONS[view_name], low, high, args.width / args.height)
            output = output_dir / f"frame_{frame:04d}_{view_name}.png"
            scene.render.filepath = str(output)
            bpy.ops.render.render(write_still=True)
            renders.append({"frame": frame, "view": view_name, "path": str(output)})

    report = {
        "blend": bpy.data.filepath,
        "frame_range": [scene.frame_start, scene.frame_end],
        "sampling": "cumulative normalized pose motion with change-aware gap filling",
        "sample_count": len(selected),
        "selected_frames": selected,
        "views": requested_views,
        "legend": {
            "axial": "orange", "tail": "green", "left": "cyan",
            "right": "magenta", "joints": "white", "mesh": "transparent gray",
        },
        "audit": audit,
        "renders": renders,
    }
    (output_dir / "bone_pose_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    for index, material_slot in enumerate(mesh.material_slots):
        material_slot.material = original_materials[index]
    for name, hidden in original_hide.items():
        obj = bpy.data.objects.get(name)
        if obj is not None:
            obj.hide_render = hidden
    for name, hidden in original_viewport_hide.items():
        obj = bpy.data.objects.get(name)
        if obj is not None:
            obj.hide_viewport = hidden


if __name__ == "__main__":
    main()
