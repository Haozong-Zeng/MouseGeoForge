#!/usr/bin/env python3
"""Render a small Eevee contact-sheet source set for flexible-tail QA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", default="1,100,200")
    parser.add_argument("--top-camera", default="HomecageCam_Top")
    parser.add_argument("--side-camera", default="HomecageCam_SideNegY")
    parser.add_argument(
        "--side-only",
        action="store_true",
        help="Render only the selected side camera at the middle requested frame.",
    )
    parser.add_argument(
        "--all-side-frames",
        action="store_true",
        help="Render the selected side camera at every requested frame.",
    )
    parser.add_argument(
        "--hide-curves",
        action="store_true",
        help="Hide CURVES hair objects for fast silhouette/ground-contact review.",
    )
    args = parser.parse_args(args_after_dash())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 640
    scene.render.resolution_y = 360
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    if args.hide_curves:
        for obj in bpy.data.objects:
            if obj.type == "CURVES":
                obj.hide_render = True
    frames = [int(token) for token in args.frames.split(",")]
    jobs = [] if args.side_only else [(args.top_camera, frame) for frame in frames]
    if args.all_side_frames:
        jobs.extend((args.side_camera, frame) for frame in frames)
    else:
        jobs.append((args.side_camera, frames[len(frames) // 2]))
    for camera_name, frame in jobs:
        scene.camera = bpy.data.objects[camera_name]
        scene.frame_set(frame)
        scene.render.filepath = str(args.output_dir / f"{camera_name}_frame{frame:03d}.png")
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
