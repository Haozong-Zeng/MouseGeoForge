#!/usr/bin/env python3
"""Rigidly separate dual-mouse instances on frames with evaluated mesh overlap.

The whole instance is translated through its placement parent, preserving the
already solved spine, paws, tail, fur, and internal joint geometry.  Every
integer dataset frame receives a placement key, so discrete poses do not gain
inter-frame interpolation semantics.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree


MESHES = ("LP_Black_Mesh", "LP_Brown_Mesh")
ARMATURES = ("LP_Black_Armature", "LP_Brown_Armature")
PLACEMENTS = ("LP_Black_Placement", "LP_Brown_Placement")


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def evaluated_bvh(obj: bpy.types.Object, depsgraph) -> BVHTree:
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)
    try:
        matrix = evaluated.matrix_world
        vertices = [matrix @ vertex.co for vertex in mesh.vertices]
        polygons = [tuple(polygon.vertices) for polygon in mesh.polygons]
        return BVHTree.FromPolygons(vertices, polygons, all_triangles=False)
    finally:
        evaluated.to_mesh_clear()


def overlap_count(depsgraph) -> int:
    first = evaluated_bvh(bpy.data.objects[MESHES[0]], depsgraph)
    second = evaluated_bvh(bpy.data.objects[MESHES[1]], depsgraph)
    return len(first.overlap(second))


def root_xy(name: str) -> Vector:
    armature = bpy.data.objects[name]
    world = armature.matrix_world @ armature.pose.bones["Root_M"].head
    return Vector((world.x, world.y))


def local_delta(placement: bpy.types.Object, world_xy: Vector) -> Vector:
    world = Vector((world_xy.x, world_xy.y, 0.0))
    if placement.parent is None:
        return world
    return placement.parent.matrix_world.inverted().to_3x3() @ world


def set_pair_shift(
    placements: tuple[bpy.types.Object, bpy.types.Object],
    bases: tuple[Vector, Vector],
    direction: Vector,
    amount_mm: float,
) -> None:
    first_delta = local_delta(placements[0], direction * amount_mm)
    second_delta = local_delta(placements[1], -direction * amount_mm)
    placements[0].location = bases[0] + first_delta
    placements[1].location = bases[1] + second_delta
    bpy.context.view_layer.update()


def unit_or_x(value: Vector) -> Vector:
    if value.length < 1.0e-8:
        return Vector((1.0, 0.0))
    return value.normalized()


def rotated(vector: Vector, angle: float) -> Vector:
    cosine, sine = math.cos(angle), math.sin(angle)
    return Vector((
        cosine * vector.x - sine * vector.y,
        sine * vector.x + cosine * vector.y,
    ))


def find_smallest_clear_shift(
    placements: tuple[bpy.types.Object, bpy.types.Object],
    bases: tuple[Vector, Vector],
    depsgraph,
    preferred: Vector,
    *,
    maximum_mm: float,
    padding_mm: float,
) -> tuple[Vector, float, int]:
    # Prefer the source pair-separation direction; nearby alternatives handle
    # hooked tails and end-to-side contact without changing either pose.
    angles = [0.0]
    for step in range(1, 7):
        angle = step * math.pi / 12.0
        angles.extend((angle, -angle))
    candidates = []
    evaluations = 0
    for angle in angles:
        direction = rotated(preferred, angle)
        first_clear = None
        for amount in range(1, int(math.ceil(maximum_mm)) + 1):
            set_pair_shift(placements, bases, direction, float(amount))
            evaluations += 1
            if overlap_count(depsgraph) == 0:
                first_clear = float(amount)
                break
        if first_clear is None:
            continue
        low = max(0.0, first_clear - 1.0)
        high = first_clear
        for _ in range(5):
            middle = 0.5 * (low + high)
            set_pair_shift(placements, bases, direction, middle)
            evaluations += 1
            if overlap_count(depsgraph) == 0:
                high = middle
            else:
                low = middle
        amount = min(maximum_mm, high + padding_mm)
        set_pair_shift(placements, bases, direction, amount)
        evaluations += 1
        if overlap_count(depsgraph) == 0:
            # Small angular penalty keeps the correction interpretable when
            # several directions have essentially equal translations.
            score = amount + 0.08 * abs(math.degrees(angle))
            candidates.append((score, amount, direction.copy()))
    if not candidates:
        set_pair_shift(placements, bases, preferred, 0.0)
        raise RuntimeError("Could not rigidly separate evaluated meshes")
    _, amount, direction = min(candidates, key=lambda item: item[0])
    return direction, amount, evaluations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--maximum-per-mouse-shift-mm", type=float, default=40.0)
    parser.add_argument("--padding-mm", type=float, default=1.5)
    args = parser.parse_args(args_after_dash())

    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    placements = tuple(bpy.data.objects[name] for name in PLACEMENTS)
    bases = tuple(obj.location.copy() for obj in placements)
    rows = []
    corrected_frames = []
    total_evaluations = 0

    for frame in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(frame)
        set_pair_shift(placements, bases, Vector((1.0, 0.0)), 0.0)
        before = overlap_count(depsgraph)
        direction = Vector((0.0, 0.0))
        amount = 0.0
        if before:
            preferred = unit_or_x(root_xy(ARMATURES[0]) - root_xy(ARMATURES[1]))
            direction, amount, evaluations = find_smallest_clear_shift(
                placements,
                bases,
                depsgraph,
                preferred,
                maximum_mm=args.maximum_per_mouse_shift_mm,
                padding_mm=args.padding_mm,
            )
            total_evaluations += evaluations
            set_pair_shift(placements, bases, direction, amount)
            corrected_frames.append(frame)
        after = overlap_count(depsgraph)
        if after:
            raise RuntimeError(f"Frame {frame} still has {after} overlap pairs")
        for placement in placements:
            placement.keyframe_insert(data_path="location", frame=frame)
        rows.append({
            "frame": frame,
            "overlap_pairs_before": before,
            "overlap_pairs_after": after,
            "per_mouse_shift_mm": amount,
            "world_direction_first_mouse": [direction.x, direction.y],
        })

    scene["MGF_evaluated_mesh_separation"] = json.dumps({
        "policy": "symmetric rigid per-instance translations on discrete frames",
        "corrected_frames": corrected_frames,
        "padding_mm": args.padding_mm,
        "maximum_per_mouse_shift_mm": max(row["per_mouse_shift_mm"] for row in rows),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output), check_existing=False)
    report = {
        "schema": "mousegeoforge.dual_mouse_evaluated_mesh_separation.v1",
        "output_blend": str(args.output),
        "frame_range": [scene.frame_start, scene.frame_end],
        "corrected_frames": corrected_frames,
        "corrected_frame_count": len(corrected_frames),
        "maximum_per_mouse_shift_mm": max(row["per_mouse_shift_mm"] for row in rows),
        "padding_mm": args.padding_mm,
        "candidate_overlap_evaluations": total_evaluations,
        "rows": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
