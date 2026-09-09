#!/usr/bin/env python3
"""Export canonical and posed mouse mesh topology for a CSE-like pilot."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import bpy
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mesh", default="MS_Mouse")
    parser.add_argument("--frame", type=int, required=True)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def disable_hair_evaluation() -> int:
    count = 0
    for obj in bpy.data.objects:
        if obj.type != "CURVES":
            continue
        count += 1
        obj.hide_viewport = True
        obj.hide_render = True
        try:
            obj.hide_set(True)
        except RuntimeError:
            pass
        for modifier in obj.modifiers:
            modifier.show_viewport = False
            modifier.show_render = False
    bpy.context.view_layer.update()
    return count


def triangulated_faces(mesh: bpy.types.Mesh) -> np.ndarray:
    triangles: list[tuple[int, int, int]] = []
    for polygon in mesh.polygons:
        vertices = list(polygon.vertices)
        for index in range(1, len(vertices) - 1):
            triangles.append((vertices[0], vertices[index], vertices[index + 1]))
    return np.asarray(triangles, dtype=np.int32)


def main() -> None:
    args = parse_args()
    output = Path(bpy.path.abspath(args.output)).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    mesh_obj = bpy.data.objects.get(args.mesh)
    if mesh_obj is None or mesh_obj.type != "MESH":
        raise KeyError(f"Mesh object {args.mesh!r} not found")

    hair_count = disable_hair_evaluation()
    canonical_vertices = np.asarray(
        [tuple(vertex.co) for vertex in mesh_obj.data.vertices],
        dtype=np.float32,
    )
    edges = np.asarray(
        [tuple(edge.vertices) for edge in mesh_obj.data.edges],
        dtype=np.int32,
    )
    faces = triangulated_faces(mesh_obj.data)

    bpy.context.scene.frame_set(args.frame)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = mesh_obj.evaluated_get(depsgraph)
    posed_mesh = evaluated.to_mesh()
    try:
        if len(posed_mesh.vertices) != len(canonical_vertices):
            raise ValueError(
                "Evaluated topology changed: "
                f"canonical={len(canonical_vertices)} posed={len(posed_mesh.vertices)}"
            )
        world = evaluated.matrix_world
        posed_world_vertices = np.asarray(
            [tuple(world @ vertex.co) for vertex in posed_mesh.vertices],
            dtype=np.float32,
        )
    finally:
        evaluated.to_mesh_clear()

    with output.open("wb") as handle:
        np.savez_compressed(
            handle,
            canonical_vertices=canonical_vertices,
            posed_world_vertices=posed_world_vertices,
            edges=edges,
            faces=faces,
            frame=np.asarray(args.frame, dtype=np.int32),
            mesh_name=np.asarray(mesh_obj.name),
        )
    metadata = {
        "format": "mousegeoforge_mouse_cse_pilot_mesh_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_blend": str(Path(bpy.data.filepath).resolve()),
        "mesh": mesh_obj.name,
        "frame": args.frame,
        "vertex_count": int(len(canonical_vertices)),
        "edge_count": int(len(edges)),
        "triangle_count": int(len(faces)),
        "hair_objects_disabled": hair_count,
        "npz": str(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print("mouse_cse_mesh_export_complete", json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
