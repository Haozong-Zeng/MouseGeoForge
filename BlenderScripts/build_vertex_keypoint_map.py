#!/usr/bin/env python3
"""Build a topology-specific vertex-to-keypoint visibility support map.

Run with Blender:

    blender source.blend --background --python build_vertex_keypoint_map.py -- \
      --output vertex_keypoint_map.pkl \
      --report vertex_keypoint_map.audit.json \
      --debug-blend vertex_keypoint_map_debug.blend

The output pickle remains compatible with the legacy renderer:
`dict[int, str]`, mapping mesh vertex indices to armature bone names.
"""

from __future__ import annotations

import argparse
import bmesh
import bpy
import hashlib
import json
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

from mathutils import Matrix, Vector


HAND_LANDMARKS = [
    f"{finger}Finger{segment}_{side}"
    for side in ("L", "R")
    for finger in ("Index", "Middle", "Ring", "Pinky")
    for segment in (1, 2, 3)
] + ["Fingers1_L", "Fingers1_R"]

ADDITIONAL_LANDMARKS = ["Wrist_L", "Wrist_R", "HeadEnd_M"]
DEFAULT_LANDMARKS = HAND_LANDMARKS + ADDITIONAL_LANDMARKS

# Wrist bone heads coincide with the elbows. HeadEnd_M starts just behind the
# nose surface, so a short offset along the rest bone reaches the visible tip.
LANDMARK_ANCHOR_FRACTIONS = {"HeadEnd_M": 0.27}
LANDMARK_SELECTION_MODES = {
    "Wrist_L": "joint_patch",
    "Wrist_R": "joint_patch",
    "HeadEnd_M": "surface_cap",
}


@dataclass(frozen=True)
class VertexCandidate:
    vertex_id: int
    score: float
    axial: float
    radial: float
    distance: float
    angle: float
    weight: float
    plane_crossing: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", default="MS_Mouse")
    parser.add_argument("--armature", default="DeformationSystem")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--debug-blend", default="")
    parser.add_argument("--landmarks", default="")
    parser.add_argument("--vertices-per-keypoint", type=int, default=16)
    parser.add_argument("--angular-sectors", type=int, default=12)
    parser.add_argument("--min-group-weight", type=float, default=0.02)
    parser.add_argument("--marker-radius", type=float, default=0.008)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def resolve_path(text: str) -> Path:
    return Path(bpy.path.abspath(text)).resolve()


def parse_landmarks(text: str) -> list[str]:
    if not text.strip():
        return DEFAULT_LANDMARKS
    return [part.strip() for part in text.split(",") if part.strip()]


def mesh_topology_signature(mesh_obj: bpy.types.Object) -> dict[str, object]:
    digest = hashlib.sha256()
    position_digest = hashlib.sha256()
    mesh = mesh_obj.data
    digest.update(f"v={len(mesh.vertices)};e={len(mesh.edges)};p={len(mesh.polygons)};".encode("ascii"))
    for vertex in mesh.vertices:
        position_digest.update(
            ",".join(f"{float(value):.8g}" for value in vertex.co).encode("ascii")
        )
        position_digest.update(b";")
    for polygon in mesh.polygons:
        digest.update(",".join(str(vertex_id) for vertex_id in polygon.vertices).encode("ascii"))
        digest.update(b";")
    return {
        "mesh": mesh_obj.name,
        "vertex_count": len(mesh.vertices),
        "edge_count": len(mesh.edges),
        "polygon_count": len(mesh.polygons),
        "polygon_index_sha256": digest.hexdigest(),
        "rest_vertex_position_sha256": position_digest.hexdigest(),
    }


def vertex_group_weights(mesh_obj: bpy.types.Object) -> list[dict[str, float]]:
    group_names = {group.index: group.name for group in mesh_obj.vertex_groups}
    result: list[dict[str, float]] = []
    for vertex in mesh_obj.data.vertices:
        result.append(
            {
                group_names[item.group]: float(item.weight)
                for item in vertex.groups
                if item.group in group_names
            }
        )
    return result


def orthogonal_basis(axis: Vector) -> tuple[Vector, Vector]:
    reference = Vector((1.0, 0.0, 0.0))
    if abs(axis.dot(reference)) > 0.85:
        reference = Vector((0.0, 1.0, 0.0))
    first = axis.cross(reference).normalized()
    second = axis.cross(first).normalized()
    return first, second


def landmark_geometry(
    mesh_obj: bpy.types.Object,
    armature_obj: bpy.types.Object,
    bone: bpy.types.Bone,
    anchor_fraction: float = 0.0,
) -> tuple[Vector, Vector, float]:
    mesh_from_world = mesh_obj.matrix_world.inverted()
    world_from_armature = armature_obj.matrix_world
    anchor = bone.head_local.lerp(bone.tail_local, anchor_fraction)
    point = mesh_from_world @ (world_from_armature @ anchor)
    tail = mesh_from_world @ (world_from_armature @ bone.tail_local)
    axis = tail - point
    length = axis.length
    if length <= 1e-8:
        raise ValueError(f"Bone {bone.name!r} has zero rest length")
    return point, axis.normalized(), length


def related_group_names(bone: bpy.types.Bone, selection_mode: str) -> set[str]:
    names = {bone.name}
    if selection_mode in {"joint_ring", "joint_patch"} and bone.parent is not None:
        names.add(bone.parent.name)
    if selection_mode == "joint_ring":
        names.update(child.name for child in bone.children)
    return names


def candidates_for_landmark(
    mesh_obj: bpy.types.Object,
    armature_obj: bpy.types.Object,
    bone: bpy.types.Bone,
    all_weights: list[dict[str, float]],
    min_group_weight: float,
) -> tuple[list[VertexCandidate], dict[str, object]]:
    anchor_fraction = LANDMARK_ANCHOR_FRACTIONS.get(bone.name, 0.0)
    selection_mode = LANDMARK_SELECTION_MODES.get(bone.name, "joint_ring")
    point, axis, bone_length = landmark_geometry(
        mesh_obj,
        armature_obj,
        bone,
        anchor_fraction=anchor_fraction,
    )
    basis_u, basis_v = orthogonal_basis(axis)
    group_names = related_group_names(bone, selection_mode)
    raw: list[tuple[int, float, float, float, float, float, float]] = []

    for vertex in mesh_obj.data.vertices:
        weight = max((all_weights[vertex.index].get(name, 0.0) for name in group_names), default=0.0)
        if weight < min_group_weight:
            continue
        offset = vertex.co - point
        signed_axial = offset.dot(axis)
        radial_vector = offset - signed_axial * axis
        radial = radial_vector.length
        if radial <= 1e-8:
            angle = 0.0
        else:
            angle = math.atan2(radial_vector.dot(basis_v), radial_vector.dot(basis_u))
        raw.append(
            (
                vertex.index,
                float(signed_axial),
                abs(signed_axial),
                radial,
                offset.length,
                angle,
                weight,
            )
        )

    if not raw:
        raise ValueError(f"No weighted mesh vertices found near {bone.name!r}")

    nearest_distances = sorted(item[4] for item in raw)
    distance_rank = min(
        len(nearest_distances) - 1,
        15 if selection_mode == "joint_patch" else 31 if selection_mode == "surface_cap" else 95,
    )
    distance_scale = max(nearest_distances[distance_rank], bone_length * 0.25, 1e-6)
    axial_values = sorted(item[2] for item in raw if item[4] <= distance_scale)
    axial_scale = max(
        axial_values[min(len(axial_values) - 1, 47)] if axial_values else bone_length * 0.2,
        bone_length * 0.08,
        1e-6,
    )

    signed_by_vertex = {item[0]: item[1] for item in raw}
    distance_by_vertex = {item[0]: item[4] for item in raw}
    plane_crossing_vertices: set[int] = set()
    if selection_mode == "joint_ring":
        for edge in mesh_obj.data.edges:
            first, second = edge.vertices
            if first not in signed_by_vertex or second not in signed_by_vertex:
                continue
            if signed_by_vertex[first] * signed_by_vertex[second] > 0:
                continue
            if min(distance_by_vertex[first], distance_by_vertex[second]) > distance_scale * 1.35:
                continue
            plane_crossing_vertices.update((first, second))

    candidates = []
    distance_limit = distance_scale * (1.15 if selection_mode == "joint_patch" else 1.35)
    for vertex_id, _signed_axial, axial, radial, distance, angle, weight in raw:
        if distance > distance_limit:
            continue
        if selection_mode in {"surface_cap", "joint_patch"}:
            score = distance / distance_scale - 0.05 * weight
        else:
            score = axial / axial_scale + 0.20 * distance / distance_scale - 0.05 * weight
        candidates.append(
            VertexCandidate(
                vertex_id=vertex_id,
                score=float(score),
                axial=float(axial),
                radial=float(radial),
                distance=float(distance),
                angle=float(angle),
                weight=float(weight),
                plane_crossing=vertex_id in plane_crossing_vertices,
            )
        )
    candidates.sort(
        key=lambda item: (
            not item.plane_crossing,
            item.score,
            item.distance,
            item.vertex_id,
        )
    )
    return candidates, {
        "bone_head_mesh_local": [float(value) for value in point],
        "bone_axis_mesh_local": [float(value) for value in axis],
        "bone_length_mesh_local": float(bone_length),
        "anchor_fraction": float(anchor_fraction),
        "selection_mode": selection_mode,
        "related_vertex_groups": sorted(group_names),
        "weighted_candidate_count": len(raw),
        "plane_crossing_candidate_count": len(plane_crossing_vertices),
        "distance_scale": float(distance_scale),
        "axial_scale": float(axial_scale),
    }


def angularly_balanced_order(
    candidates: list[VertexCandidate],
    angular_sectors: int,
    vertices_per_keypoint: int,
) -> list[VertexCandidate]:
    def balanced(items: list[VertexCandidate]) -> list[VertexCandidate]:
        buckets: list[list[VertexCandidate]] = [[] for _ in range(angular_sectors)]
        for item in items:
            normalized = (item.angle + math.pi) / (2.0 * math.pi)
            sector = min(angular_sectors - 1, int(normalized * angular_sectors))
            buckets[sector].append(item)
        for bucket in buckets:
            bucket.sort(key=lambda item: (item.score, item.distance, item.vertex_id))

        result = []
        depth = 0
        while any(depth < len(bucket) for bucket in buckets):
            for bucket in buckets:
                if depth < len(bucket):
                    result.append(bucket[depth])
            depth += 1
        return result

    crossing = balanced([item for item in candidates if item.plane_crossing])
    fallback = balanced([item for item in candidates if not item.plane_crossing])
    ordered = crossing + fallback
    if len(ordered) < vertices_per_keypoint:
        raise ValueError(
            f"Only {len(ordered)} candidates available; requested {vertices_per_keypoint}"
        )
    return ordered


def assign_unique_vertices(
    ordered_by_landmark: dict[str, list[VertexCandidate]],
    vertices_per_keypoint: int,
) -> tuple[dict[int, str], dict[str, list[VertexCandidate]], dict[str, int]]:
    vertex_map: dict[int, str] = {}
    selected: dict[str, list[VertexCandidate]] = {}
    conflicts: dict[str, int] = {}
    for landmark, ordered in ordered_by_landmark.items():
        chosen = []
        skipped = 0
        for item in ordered:
            if item.vertex_id in vertex_map:
                skipped += 1
                continue
            vertex_map[item.vertex_id] = landmark
            chosen.append(item)
            if len(chosen) >= vertices_per_keypoint:
                break
        if len(chosen) < vertices_per_keypoint:
            raise ValueError(
                f"Only {len(chosen)} unique vertices available for {landmark}; "
                f"requested {vertices_per_keypoint}"
            )
        selected[landmark] = chosen
        conflicts[landmark] = skipped
    return vertex_map, selected, conflicts


def selection_audit(
    selected: list[VertexCandidate],
    angular_sectors: int,
) -> dict[str, object]:
    occupied = set()
    for item in selected:
        normalized = (item.angle + math.pi) / (2.0 * math.pi)
        occupied.add(min(angular_sectors - 1, int(normalized * angular_sectors)))

    def stats(values: list[float]) -> dict[str, float]:
        values = sorted(values)
        middle = len(values) // 2
        median = (
            values[middle]
            if len(values) % 2
            else 0.5 * (values[middle - 1] + values[middle])
        )
        return {
            "min": float(values[0]),
            "median": float(median),
            "max": float(values[-1]),
        }

    return {
        "selected_vertex_ids": [item.vertex_id for item in selected],
        "selected_count": len(selected),
        "selected_plane_crossing_count": sum(item.plane_crossing for item in selected),
        "occupied_angular_sectors": len(occupied),
        "angular_sector_count": angular_sectors,
        "axial_abs": stats([item.axial for item in selected]),
        "radial": stats([item.radial for item in selected]),
        "distance": stats([item.distance for item in selected]),
        "weight": stats([item.weight for item in selected]),
    }


def color_for_index(index: int, count: int) -> tuple[float, float, float, float]:
    hue = index / max(1, count)
    h = hue * 6.0
    section = int(h) % 6
    fraction = h - int(h)
    values = [
        (1.0, fraction, 0.0),
        (1.0 - fraction, 1.0, 0.0),
        (0.0, 1.0, fraction),
        (0.0, 1.0 - fraction, 1.0),
        (fraction, 0.0, 1.0),
        (1.0, 0.0, 1.0 - fraction),
    ]
    red, green, blue = values[section]
    return red, green, blue, 1.0


def create_debug_markers(
    mesh_obj: bpy.types.Object,
    armature_obj: bpy.types.Object,
    selected: dict[str, list[VertexCandidate]],
    marker_radius: float,
) -> None:
    collection = bpy.data.collections.get("MGF_VertexKeypointMap_Debug")
    if collection is None:
        collection = bpy.data.collections.new("MGF_VertexKeypointMap_Debug")
        bpy.context.scene.collection.children.link(collection)

    armature_obj.data.pose_position = "REST"
    for obj in bpy.context.scene.objects:
        if obj not in {mesh_obj, armature_obj} and obj.name not in collection.objects:
            obj.hide_viewport = True
            obj.hide_render = True

    for index, (landmark, items) in enumerate(selected.items()):
        material = bpy.data.materials.new(f"VKP_MAT_{landmark}")
        material.diffuse_color = color_for_index(index, len(selected))

        debug_mesh = bpy.data.meshes.new(f"VKP_MESH_{landmark}")
        debug_obj = bpy.data.objects.new(f"VKP_{landmark}", debug_mesh)
        debug_obj.matrix_world = mesh_obj.matrix_world.copy()
        collection.objects.link(debug_obj)
        bm = bmesh.new()
        for item in items:
            offset = mesh_obj.data.vertices[item.vertex_id].co
            bmesh.ops.create_icosphere(
                bm,
                subdivisions=1,
                radius=marker_radius,
                matrix=Matrix.Translation(offset),
                calc_uvs=False,
            )
        bm.to_mesh(debug_mesh)
        bm.free()
        debug_obj.data.materials.append(material)

        bone = armature_obj.data.bones[landmark]
        point, _, _ = landmark_geometry(
            mesh_obj,
            armature_obj,
            bone,
            anchor_fraction=LANDMARK_ANCHOR_FRACTIONS.get(landmark, 0.0),
        )
        center = bpy.data.objects.new(f"VKP_CENTER_{landmark}", None)
        center.empty_display_type = "SPHERE"
        center.empty_display_size = marker_radius * 2.0
        center.color = material.diffuse_color
        center.location = mesh_obj.matrix_world @ point
        collection.objects.link(center)


def main() -> None:
    args = parse_args()
    if args.vertices_per_keypoint < 4:
        raise ValueError("--vertices-per-keypoint must be at least 4")
    if args.angular_sectors < 4:
        raise ValueError("--angular-sectors must be at least 4")

    mesh_obj = bpy.data.objects.get(args.mesh)
    armature_obj = bpy.data.objects.get(args.armature)
    if mesh_obj is None or mesh_obj.type != "MESH":
        raise KeyError(f"Mesh object {args.mesh!r} not found")
    if armature_obj is None or armature_obj.type != "ARMATURE":
        raise KeyError(f"Armature object {args.armature!r} not found")

    landmarks = parse_landmarks(args.landmarks)
    missing = [name for name in landmarks if name not in armature_obj.data.bones]
    if missing:
        raise KeyError(f"Missing landmark bones: {missing}")

    all_weights = vertex_group_weights(mesh_obj)
    ordered_by_landmark: dict[str, list[VertexCandidate]] = {}
    geometry_audit: dict[str, dict[str, object]] = {}
    for landmark in landmarks:
        candidates, geometry = candidates_for_landmark(
            mesh_obj,
            armature_obj,
            armature_obj.data.bones[landmark],
            all_weights,
            args.min_group_weight,
        )
        ordered_by_landmark[landmark] = angularly_balanced_order(
            candidates,
            args.angular_sectors,
            args.vertices_per_keypoint,
        )
        geometry_audit[landmark] = geometry

    vertex_map, selected, conflicts = assign_unique_vertices(
        ordered_by_landmark,
        args.vertices_per_keypoint,
    )

    output_path = resolve_path(args.output)
    report_path = resolve_path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(vertex_map, handle, protocol=4)

    report = {
        "format": "mousegeoforge_vertex_keypoint_map_v1",
        "source_blend": bpy.data.filepath,
        "mesh": mesh_obj.name,
        "armature": armature_obj.name,
        "topology": mesh_topology_signature(mesh_obj),
        "landmarks": landmarks,
        "vertices_per_keypoint": args.vertices_per_keypoint,
        "angular_sectors": args.angular_sectors,
        "min_group_weight": args.min_group_weight,
        "mapped_vertex_count": len(vertex_map),
        "semantic_landmarks": {
            "Wrist_L": "left_elbow",
            "Wrist_R": "right_elbow",
            "HeadEnd_M": "nose_tip",
        },
        "vertex_to_keypoint": {str(key): value for key, value in sorted(vertex_map.items())},
        "per_landmark": {},
    }
    for landmark in landmarks:
        report["per_landmark"][landmark] = {
            **geometry_audit[landmark],
            **selection_audit(selected[landmark], args.angular_sectors),
            "conflicting_vertices_skipped": conflicts[landmark],
        }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    if args.debug_blend:
        create_debug_markers(
            mesh_obj,
            armature_obj,
            selected,
            args.marker_radius,
        )
        debug_path = resolve_path(args.debug_blend)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(debug_path))

    print(f"vertex-keypoint map: {output_path}")
    print(f"audit report: {report_path}")
    print(f"landmarks={len(landmarks)} mapped_vertices={len(vertex_map)}")


if __name__ == "__main__":
    main()
