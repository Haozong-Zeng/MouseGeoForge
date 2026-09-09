"""Measure hand bones and weighted hand mesh in a generated Blender project."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


FINGERS = ("Index", "Middle", "Ring", "Pinky")


def blender_args() -> list[str]:
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--weight-threshold", type=float, default=0.1)
    return parser.parse_args(blender_args())


def world_point(armature: bpy.types.Object, point: Vector) -> Vector:
    return armature.matrix_world @ point


def distance(first: Vector, second: Vector) -> float:
    return float((first - second).length)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    blend = position - lower
    return ordered[lower] * (1.0 - blend) + ordered[upper] * blend


def mesh_region(
    mesh_obj: bpy.types.Object,
    group_names: set[str],
    weight_threshold: float,
) -> dict[str, object]:
    selected_indices = []
    group_indices = {
        group.index for group in mesh_obj.vertex_groups if group.name in group_names
    }
    for vertex in mesh_obj.data.vertices:
        if any(
            item.group in group_indices and item.weight >= weight_threshold
            for item in vertex.groups
        ):
            selected_indices.append(vertex.index)

    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = mesh_obj.evaluated_get(depsgraph)
    evaluated_mesh = evaluated.to_mesh()
    try:
        points = [
            evaluated.matrix_world @ evaluated_mesh.vertices[index].co
            for index in selected_indices
        ]
    finally:
        evaluated.to_mesh_clear()
    if not points:
        raise RuntimeError(f"No vertices selected on {mesh_obj.name}")

    center = sum(points, Vector()) / len(points)
    radii = [(point - center).length for point in points]
    axes = []
    for axis in range(3):
        coordinates = [point[axis] for point in points]
        axes.append(
            {
                "min": min(coordinates),
                "max": max(coordinates),
                "extent": max(coordinates) - min(coordinates),
                "p05_p95_extent": (
                    percentile(coordinates, 0.95) - percentile(coordinates, 0.05)
                ),
            }
        )
    return {
        "vertex_count": len(points),
        "center_world": list(center),
        "radius_p95": percentile(radii, 0.95),
        "radius_max": max(radii),
        "world_axis_extents": axes,
    }


def main() -> None:
    args = parse_args()
    scene = bpy.context.scene
    scene.frame_set(args.frame)
    bpy.context.view_layer.update()

    armature = bpy.data.objects["DeformationSystem"]
    mesh_obj = bpy.data.objects["MS_Mouse"]
    report: dict[str, object] = {
        "blend": bpy.data.filepath,
        "frame": args.frame,
        "weight_threshold": args.weight_threshold,
        "sides": {},
    }
    for side in ("L", "R"):
        palm = armature.pose.bones[f"Fingers1_{side}"]
        palm_head = world_point(armature, palm.head)
        palm_tail = world_point(armature, palm.tail)
        finger_rows = {}
        roots = {}
        tips = {}
        for finger in FINGERS:
            bones = [
                armature.pose.bones[f"{finger}Finger{index}_{side}"]
                for index in (1, 2, 3)
            ]
            endpoints = [
                world_point(armature, bones[0].head),
                *[world_point(armature, bone.tail) for bone in bones],
            ]
            lengths = [
                distance(endpoints[index], endpoints[index + 1])
                for index in range(3)
            ]
            roots[finger] = endpoints[0]
            tips[finger] = endpoints[-1]
            finger_rows[finger] = {
                "segment_lengths": lengths,
                "total_length": sum(lengths),
                "root_from_palm_head": distance(endpoints[0], palm_head),
                "root_from_palm_tail": distance(endpoints[0], palm_tail),
                "tip_from_palm_head": distance(endpoints[-1], palm_head),
                "tip_from_palm_tail": distance(endpoints[-1], palm_tail),
            }

        group_names = {f"Fingers1_{side}"}
        group_names.update(
            f"{finger}Finger{index}_{side}"
            for finger in FINGERS
            for index in (1, 2, 3)
        )
        report["sides"][side] = {
            "palm_bone_length": distance(palm_head, palm_tail),
            "finger_root_width_index_to_pinky": distance(
                roots["Index"], roots["Pinky"]
            ),
            "finger_tip_width_index_to_pinky": distance(
                tips["Index"], tips["Pinky"]
            ),
            "fingers": finger_rows,
            "weighted_mesh": mesh_region(
                mesh_obj,
                group_names,
                args.weight_threshold,
            ),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    print("MGF_HAND_SCALE=" + json.dumps(report, ensure_ascii=True))


if __name__ == "__main__":
    main()
