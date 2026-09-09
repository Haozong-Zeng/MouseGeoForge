"""Add visible, non-rendering hand contact sphere proxies.

The proxies are independent mesh spheres with per-frame location keys sampled
from the current hand pose.  They are meant as inspectable contact/control
targets for later palm/fingertip refinement, so they stay visible in the
viewport but are disabled for final renders.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


TIP_BONES = {
    "index_tip": "IndexFinger3",
    "middle_tip": "MiddleFinger3",
    "ring_tip": "RingFinger3",
    "pinky_tip": "PinkyFinger3",
}


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--armature", default="DeformationSystem")
    parser.add_argument("--mesh", default="MS_Mouse")
    parser.add_argument("--action", default=None)
    parser.add_argument("--new-action", default=None)
    parser.add_argument("--collection", default="MGF_HandContactSpheres")
    parser.add_argument("--prefix", default="MGF_HandSphere")
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--palm-radius", type=float, default=1.15)
    parser.add_argument("--tip-radius", type=float, default=0.55)
    parser.add_argument("--min-weight", type=float, default=0.001)
    parser.add_argument("--tip-projection-quantile", type=float, default=0.62)
    parser.add_argument("--segments", type=int, default=24)
    parser.add_argument("--rings", type=int, default=12)
    parser.add_argument("--no-overwrite", action="store_true")
    return parser.parse_args(args_after_dash())


def set_frame(frame: int) -> None:
    bpy.context.scene.frame_set(int(frame))
    bpy.context.view_layer.update()


def choose_action(armature, action_name: str | None):
    if action_name:
        action = bpy.data.actions.get(action_name)
        if action is None:
            raise RuntimeError(f"Action {action_name!r} not found.")
    elif armature.animation_data and armature.animation_data.action:
        action = armature.animation_data.action
    elif bpy.data.actions:
        action = sorted(bpy.data.actions, key=lambda item: item.name)[0]
    else:
        raise RuntimeError("No action found.")
    armature.animation_data_create()
    armature.animation_data.action = action
    return action


def action_frame_range(action, scene, start: int | None, end: int | None, step: int) -> list[int]:
    first = int(math.floor(action.frame_range[0])) if start is None else int(start)
    last = int(math.ceil(action.frame_range[1])) if end is None else int(end)
    if last < first:
        first, last = int(scene.frame_start), int(scene.frame_end)
    return list(range(first, last + 1, max(int(step), 1)))


def ensure_collection(name: str):
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)
    collection.hide_viewport = False
    collection.hide_render = True
    return collection


def unlink_from_other_collections(obj, keep_collection) -> None:
    for collection in list(obj.users_collection):
        if collection != keep_collection:
            collection.objects.unlink(obj)
    if obj.name not in keep_collection.objects.keys():
        keep_collection.objects.link(obj)


def remove_existing(prefix: str, collection_name: str) -> int:
    removed = 0
    for obj in list(bpy.data.objects):
        if obj.name.startswith(prefix + "_"):
            bpy.data.objects.remove(obj, do_unlink=True)
            removed += 1
    collection = bpy.data.collections.get(collection_name)
    if collection is not None:
        for obj in list(collection.objects):
            if obj.name.startswith(prefix + "_"):
                bpy.data.objects.remove(obj, do_unlink=True)
                removed += 1
    return removed


def make_material(name: str, color: tuple[float, float, float, float]):
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name)
    material.diffuse_color = color
    return material


def pose_bone_head_world(armature, bone_name: str) -> Vector | None:
    pose_bone = armature.pose.bones.get(bone_name)
    if pose_bone is None:
        return None
    return armature.matrix_world @ pose_bone.head


def pose_bone_tail_world(armature, bone_name: str) -> Vector | None:
    pose_bone = armature.pose.bones.get(bone_name)
    if pose_bone is None:
        return None
    return armature.matrix_world @ pose_bone.tail


def evaluated_vertices_world(mesh_obj) -> list[Vector]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = mesh_obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    try:
        return [eval_obj.matrix_world @ vertex.co for vertex in mesh.vertices]
    finally:
        eval_obj.to_mesh_clear()


def weighted_group_points(mesh_obj, vertices: list[Vector], group_name: str, min_weight: float) -> list[tuple[Vector, float]]:
    group = mesh_obj.vertex_groups.get(group_name)
    if group is None:
        return []
    out = []
    for vertex in mesh_obj.data.vertices:
        if vertex.index >= len(vertices):
            continue
        for membership in vertex.groups:
            if membership.group == group.index and membership.weight >= min_weight:
                out.append((vertices[vertex.index], float(membership.weight)))
                break
    return out


def weighted_center(points: list[tuple[Vector, float]]) -> Vector | None:
    if not points:
        return None
    total = sum(weight for _point, weight in points)
    if total <= 1.0e-8:
        return sum((point for point, _weight in points), Vector()) / len(points)
    center = Vector((0.0, 0.0, 0.0))
    for point, weight in points:
        center += point * weight
    return center / total


def endpoint_fallback(armature, role: str, side: str) -> Vector | None:
    if role == "palm":
        return pose_bone_tail_world(armature, f"Wrist_{side}")
    prefix = TIP_BONES[role]
    return pose_bone_tail_world(armature, f"{prefix}_{side}")


def proxy_center_from_surface(
    armature,
    mesh_obj,
    vertices: list[Vector],
    role: str,
    side: str,
    min_weight: float,
    tip_projection_quantile: float,
) -> tuple[Vector | None, str]:
    if role == "palm":
        group_name = f"Fingers1_{side}"
        points = weighted_group_points(mesh_obj, vertices, group_name, min_weight)
        center = weighted_center(points)
        if center is not None:
            return center, f"weighted-group:{group_name}"
        return endpoint_fallback(armature, role, side), "fallback:Wrist.tail"

    prefix = TIP_BONES[role]
    bone_name = f"{prefix}_{side}"
    points = weighted_group_points(mesh_obj, vertices, bone_name, min_weight)
    head = pose_bone_head_world(armature, bone_name)
    tail = pose_bone_tail_world(armature, bone_name)
    if points and head is not None and tail is not None:
        axis = tail - head
        denom = axis.length_squared
        projected = []
        if denom > 1.0e-10:
            for point, weight in points:
                t = float((point - head).dot(axis) / denom)
                projected.append((t, point, weight))
        if projected:
            ordered_t = sorted(row[0] for row in projected)
            index = max(0, min(len(ordered_t) - 1, int(round((len(ordered_t) - 1) * tip_projection_quantile))))
            cutoff = ordered_t[index]
            selected = [(point, weight) for t, point, weight in projected if t >= cutoff]
            if len(selected) < min(8, len(points)):
                keep = max(8, int(round(len(projected) * 0.22)))
                selected = [(point, weight) for _t, point, weight in sorted(projected, key=lambda row: row[0])[-keep:]]
            center = weighted_center(selected)
            if center is not None:
                return center, f"weighted-tip:{bone_name}"
        center = weighted_center(points)
        if center is not None:
            return center, f"weighted-group:{bone_name}"
    return endpoint_fallback(armature, role, side), f"fallback:{bone_name}.tail"


def create_sphere(name: str, radius: float, material, collection, segments: int, rings: int):
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=segments,
        ring_count=rings,
        radius=radius,
        location=(0.0, 0.0, 0.0),
    )
    obj = bpy.context.object
    obj.name = name
    obj.data.name = name + "_Mesh"
    obj.data.materials.append(material)
    obj.hide_viewport = False
    obj.hide_render = True
    obj.show_name = True
    obj.display_type = "TEXTURED"
    if hasattr(obj, "show_in_front"):
        obj.show_in_front = True
    unlink_from_other_collections(obj, collection)
    return obj


def set_linear_interpolation(action) -> None:
    fcurves = []
    if getattr(action, "fcurves", None):
        fcurves.extend(action.fcurves)
    if getattr(action, "is_action_layered", False):
        for layer in action.layers:
            for strip in getattr(layer, "strips", []):
                for channelbag in getattr(strip, "channelbags", []):
                    fcurves.extend(channelbag.fcurves)
    for fcurve in fcurves:
        for key in fcurve.keyframe_points:
            key.interpolation = "LINEAR"


def main() -> None:
    args = parse_args()
    bpy.ops.wm.open_mainfile(filepath=str(args.input.resolve()))
    armature = bpy.data.objects.get(args.armature)
    mesh_obj = bpy.data.objects.get(args.mesh)
    if armature is None or armature.type != "ARMATURE":
        raise RuntimeError(f"Armature {args.armature!r} not found.")
    if mesh_obj is None or mesh_obj.type != "MESH":
        raise RuntimeError(f"Mesh {args.mesh!r} not found.")

    action = choose_action(armature, args.action)
    if args.new_action:
        action = action.copy()
        action.name = args.new_action
        armature.animation_data.action = action

    if not args.no_overwrite:
        remove_existing(args.prefix, args.collection)
    collection = ensure_collection(args.collection)
    mat_left = make_material("MGF_HandSphere_Left_Blue", (0.1, 0.45, 1.0, 1.0))
    mat_right = make_material("MGF_HandSphere_Right_Amber", (1.0, 0.42, 0.08, 1.0))

    roles = ["palm", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
    specs = []
    for side, material in (("L", mat_left), ("R", mat_right)):
        for role in roles:
            radius = args.palm_radius if role == "palm" else args.tip_radius
            name = f"{args.prefix}_{side}_{role}"
            sphere = create_sphere(name, radius, material, collection, args.segments, args.rings)
            sphere["mgf_proxy_role"] = role
            sphere["mgf_proxy_side"] = side
            sphere["mgf_radius"] = float(radius)
            sphere["mgf_source_action"] = action.name
            specs.append({"side": side, "role": role, "object": sphere})

    frames = action_frame_range(action, bpy.context.scene, args.frame_start, args.frame_end, args.frame_step)
    samples = {spec["object"].name: [] for spec in specs}
    sources = {spec["object"].name: set() for spec in specs}
    missing = []
    for index, frame in enumerate(frames, start=1):
        set_frame(frame)
        vertices = evaluated_vertices_world(mesh_obj)
        for spec in specs:
            obj = spec["object"]
            center, source = proxy_center_from_surface(
                armature,
                mesh_obj,
                vertices,
                spec["role"],
                spec["side"],
                args.min_weight,
                args.tip_projection_quantile,
            )
            if center is None:
                missing.append({"frame": frame, "object": obj.name})
                continue
            obj.location = center
            obj.keyframe_insert(data_path="location", frame=frame)
            samples[obj.name].append([float(center.x), float(center.y), float(center.z)])
            sources[obj.name].add(source)
        if index == 1 or index == len(frames) or index % 50 == 0:
            print(f"[hand-spheres] sampled {index}/{len(frames)} frame={frame}")

    for spec in specs:
        obj = spec["object"]
        if obj.animation_data and obj.animation_data.action:
            set_linear_interpolation(obj.animation_data.action)

    set_frame(frames[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output.resolve()))

    report = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "armature": armature.name,
        "mesh": mesh_obj.name,
        "action": action.name,
        "collection": args.collection,
        "frames": frames,
        "sphere_count": len(specs),
        "hide_render": True,
        "objects": [
            {
                "name": spec["object"].name,
                "side": spec["side"],
                "role": spec["role"],
                "radius": float(spec["object"]["mgf_radius"]),
                "hide_render": bool(spec["object"].hide_render),
                "sources": sorted(sources[spec["object"].name]),
                "sample_count": len(samples[spec["object"].name]),
                "first_sample": samples[spec["object"].name][0] if samples[spec["object"].name] else None,
            }
            for spec in specs
        ],
        "missing": missing[:100],
        "missing_count": len(missing),
    }
    report_path = args.output.with_suffix(".hand-spheres-report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "report": str(report_path.resolve())}, indent=2))


if __name__ == "__main__":
    main()
