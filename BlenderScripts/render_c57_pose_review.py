"""Render lightweight multi-frame C57 appearance and deformation reviews.

This script is intended for Blender background mode. It never saves the input
blend. Dense fur can be deterministically subsampled for fast canonical-space
and skinning checks while preserving every fur object's surface attachment.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


MOUSE_MESH = "MS_Mouse.001"
RENDER_OBJECTS = {
    MOUSE_MESH,
    "BaseFur.001",
    "EarFur.001",
    "EyesFur.001",
    "LegFur.001",
    "Mouth.001",
    "TailFur.001",
    "Whisker.001",
}
FUR_OBJECTS = {
    "BaseFur.001",
    "EarFur.001",
    "EyesFur.001",
    "LegFur.001",
    "TailFur.001",
}


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", default="1,31,81,181")
    parser.add_argument("--views", default="top,side")
    parser.add_argument("--curve-stride", type=int, default=20)
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--lens", type=float, default=55.0)
    parser.add_argument("--engine", choices=("cycles", "workbench"), default="cycles")
    return parser.parse_args(argv)


def downsample_curves(stride: int) -> dict[str, tuple[int, int]]:
    if stride <= 1:
        return {}
    counts: dict[str, tuple[int, int]] = {}
    for name in FUR_OBJECTS:
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != "CURVES":
            continue
        before = len(obj.data.curves)
        remove = [index for index in range(before) if index % stride]
        obj.data.remove_curves(indices=remove)
        counts[name] = (before, len(obj.data.curves))
    return counts


def hide_environment() -> None:
    for obj in bpy.context.scene.objects:
        if obj.type in {"CAMERA", "LIGHT"}:
            continue
        obj.hide_render = obj.name not in RENDER_OBJECTS


def evaluated_bbox_center(obj: bpy.types.Object) -> Vector:
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    points = [evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box]
    return sum(points, Vector()) / len(points)


def get_review_camera(name: str) -> bpy.types.Object:
    camera = bpy.data.objects.get(name)
    if camera is None:
        data = bpy.data.cameras.new(name)
        camera = bpy.data.objects.new(name, data)
        bpy.context.scene.collection.objects.link(camera)
    return camera


def look_at(camera: bpy.types.Object, target: Vector) -> None:
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()


def place_camera(camera: bpy.types.Object, view: str, center: Vector, lens: float) -> None:
    camera.data.type = "PERSP"
    camera.data.lens = lens
    camera.data.sensor_width = 36.0
    if view == "top":
        camera.location = (center.x, center.y, center.z + 350.0)
    elif view == "side":
        camera.location = (center.x, center.y - 330.0, center.z + 8.0)
    else:
        raise ValueError(f"Unknown view: {view}")
    look_at(camera, center)


def configure_render(args: argparse.Namespace) -> None:
    scene = bpy.context.scene
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    if args.engine == "cycles":
        scene.render.engine = "CYCLES"
        scene.cycles.device = "GPU"
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = True
    else:
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.display.shading.light = "STUDIO"
        scene.display.shading.color_type = "OBJECT"
        scene.display.shading.show_shadows = True
        scene.display.shading.show_cavity = True
        scene.display.shading.show_specular_highlight = True
        colours = {
            MOUSE_MESH: (0.12, 0.12, 0.12, 1.0),
            "BaseFur.001": (0.28, 0.18, 0.10, 1.0),
            "EarFur.001": (0.95, 0.35, 0.08, 1.0),
            "EyesFur.001": (0.55, 0.10, 0.75, 1.0),
            "LegFur.001": (0.05, 0.65, 0.85, 1.0),
            "TailFur.001": (0.15, 0.75, 0.25, 1.0),
            "Mouth.001": (0.80, 0.20, 0.25, 1.0),
            "Whisker.001": (0.90, 0.90, 0.90, 1.0),
        }
        for name, colour in colours.items():
            obj = bpy.data.objects.get(name)
            if obj is not None:
                obj.color = colour


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = [int(value) for value in args.frames.split(",") if value]
    views = [value.strip() for value in args.views.split(",") if value.strip()]
    mouse = bpy.data.objects.get(MOUSE_MESH)
    if mouse is None:
        raise RuntimeError(f"Missing {MOUSE_MESH}")

    counts = downsample_curves(args.curve_stride)
    hide_environment()
    configure_render(args)
    camera = get_review_camera("MGF_C57PoseReview_Camera")
    scene = bpy.context.scene

    for frame in frames:
        scene.frame_set(frame)
        center = evaluated_bbox_center(mouse)
        for view in views:
            place_camera(camera, view, center, args.lens)
            scene.camera = camera
            scene.render.filepath = str(args.output_dir / f"frame_{frame:04d}_{view}.png")
            bpy.ops.render.render(write_still=True)

    print({"output": str(args.output_dir), "frames": frames, "views": views, "curve_counts": counts})


if __name__ == "__main__":
    main()
