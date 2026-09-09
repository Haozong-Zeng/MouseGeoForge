"""Render lightweight material/coverage checks for the BlackFur WashFace file."""

from __future__ import annotations

import argparse
import os
import sys

import bpy


def script_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default="1,45,90,135,181")
    parser.add_argument("--cameras", default="CameraE")
    parser.add_argument(
        "--output",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(bpy.data.filepath))),
            "output",
            "blackfur_washface",
            "previews",
        ),
    )
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--preview-density", type=int, default=2)
    return parser.parse_args(argv)


def make_visible() -> None:
    for obj in bpy.data.objects:
        if obj.type == "CURVES" and ("Fur" in obj.name or obj.name.startswith("Whisker")):
            obj.hide_viewport = False
            obj.hide_render = False
            obj.hide_set(False)


def simplify_hair_density(max_duplicates: int) -> None:
    for obj in bpy.data.objects:
        if obj.type != "CURVES":
            continue
        for modifier in obj.modifiers:
            if modifier.name == "Duplicate Hair Curves":
                current = modifier.get("Input_2")
                if current is not None:
                    modifier["Input_2"] = min(int(current), int(max_duplicates))


def main() -> None:
    args = script_args()
    frames = [int(value) for value in args.frames.split(",") if value.strip()]
    cameras = [value.strip() for value in args.cameras.split(",") if value.strip()]
    os.makedirs(args.output, exist_ok=True)

    make_visible()
    simplify_hair_density(args.preview_density)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.render.resolution_x = args.width
    scene.render.resolution_y = round(args.width * 2 / 3)
    scene.render.resolution_percentage = 100

    for camera_name in cameras:
        camera = bpy.data.objects.get(camera_name)
        if not camera or camera.type != "CAMERA":
            raise RuntimeError(f"Camera not found: {camera_name}")
        scene.camera = camera
        for frame in frames:
            scene.frame_set(frame)
            scene.render.filepath = os.path.join(
                args.output,
                f"{camera_name}_frame_{frame:04d}.png",
            )
            bpy.ops.render.render(write_still=True)
            print(f"MGF_RENDERED={scene.render.filepath}")


if __name__ == "__main__":
    main()
