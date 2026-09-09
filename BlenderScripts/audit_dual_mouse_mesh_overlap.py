#!/usr/bin/env python3
"""Audit evaluated black/brown mouse meshes for triangle intersections."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils.bvhtree import BVHTree


MESHES = ("LP_Black_Mesh", "LP_Brown_Mesh")


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def evaluated_bvh(obj: bpy.types.Object, depsgraph) -> tuple[BVHTree, int, int]:
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)
    try:
        matrix = evaluated.matrix_world
        vertices = [matrix @ vertex.co for vertex in mesh.vertices]
        polygons = [tuple(polygon.vertices) for polygon in mesh.polygons]
        return BVHTree.FromPolygons(vertices, polygons, all_triangles=False), len(vertices), len(polygons)
    finally:
        evaluated.to_mesh_clear()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(args_after_dash())
    for obj in bpy.data.objects:
        if obj.type == "CURVES":
            obj.hide_viewport = True
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    rows = []
    maximum_pairs = 0
    intersecting_frames = []
    for frame in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        first, vertex_count, polygon_count = evaluated_bvh(bpy.data.objects[MESHES[0]], depsgraph)
        second, _, _ = evaluated_bvh(bpy.data.objects[MESHES[1]], depsgraph)
        overlaps = first.overlap(second)
        overlap_count = len(overlaps)
        maximum_pairs = max(maximum_pairs, overlap_count)
        if overlap_count:
            intersecting_frames.append(frame)
        rows.append({"frame": frame, "triangle_overlap_pairs": overlap_count})
    report = {
        "schema": "mousegeoforge.dual_mouse_mesh_overlap_audit.v1",
        "blend": bpy.data.filepath,
        "frame_range": [scene.frame_start, scene.frame_end],
        "meshes": list(MESHES),
        "vertices_per_mesh": vertex_count,
        "polygons_per_mesh": polygon_count,
        "intersecting_frames": intersecting_frames,
        "intersecting_frame_count": len(intersecting_frames),
        "maximum_triangle_overlap_pairs": maximum_pairs,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
