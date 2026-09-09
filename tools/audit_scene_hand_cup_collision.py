#!/usr/bin/env python3
"""Audit scene-driven hand animation against cup meshes without an armature action."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-step", type=int, default=5)
    parser.add_argument("--armature", default="DeformationSystem")
    parser.add_argument("--mesh", default="MS_Mouse")
    parser.add_argument("--obstacle", action="append", default=[])
    parser.add_argument("--include-wrist", action="store_true")
    return parser.parse_args(args_after_dash())


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    blender_scripts_dir = repo_root / "BlenderScripts"
    if str(blender_scripts_dir) not in sys.path:
        sys.path.insert(0, str(blender_scripts_dir))
    import frontlimb_quality_check as quality

    scene = bpy.context.scene
    armature = bpy.data.objects[args.armature]
    mesh = bpy.data.objects[args.mesh]
    selected, groups = quality.collect_paw_vertex_indices(
        mesh,
        include_wrist=args.include_wrist,
        min_weight=0.001,
    )
    obstacle_names = args.obstacle or [
        name
        for name in quality.DEFAULT_OBSTACLES
        if bpy.data.objects.get(name) is not None
    ]
    obstacles, missing = quality.obstacle_records(obstacle_names)
    frames = list(
        range(scene.frame_start, scene.frame_end + 1, max(args.sample_step, 1))
    )
    if frames[-1] != scene.frame_end:
        frames.append(scene.frame_end)

    rows = []
    for frame in frames:
        quality.set_frame(frame)
        vertices, polygons = quality.evaluated_mesh_world(mesh)
        sides = {}
        for side in ("L", "R"):
            sides[side] = quality.paw_clearance_for_side(
                vertices,
                polygons,
                selected[side],
                side,
                obstacles,
                polygon_mode="any",
            )
        rows.append({"frame": frame, "sides": sides})

    summary = {}
    for side in ("L", "R"):
        side_rows = [row["sides"][side] for row in rows]
        overlap_rows = [
            (rows[index]["frame"], item["overlap_count"])
            for index, item in enumerate(side_rows)
            if item["overlap_count"] > 0
        ]
        summary[side] = {
            "sampled_frame_count": len(side_rows),
            "overlap_frame_count": len(overlap_rows),
            "max_overlap_count": max(
                (count for _frame, count in overlap_rows), default=0
            ),
            "overlap_frames": [
                {"frame": frame, "count": count}
                for frame, count in sorted(
                    overlap_rows, key=lambda item: item[1], reverse=True
                )[:20]
            ],
            "min_surface_distance": min(
                (
                    item["min_distance"]
                    for item in side_rows
                    if item["min_distance"] is not None
                ),
                default=None,
            ),
        }

    report = {
        "blend_file": bpy.data.filepath,
        "frame_range": [scene.frame_start, scene.frame_end],
        "sample_step": args.sample_step,
        "obstacles": obstacle_names,
        "missing_obstacles": missing,
        "paw_vertex_groups": groups,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
