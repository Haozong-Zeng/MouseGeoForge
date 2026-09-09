"""Render fast mesh-only elbow comparison frames from a V19 branch."""

from __future__ import annotations

import argparse
import os
import sys

import bpy


def args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", default="79,87,98")
    parser.add_argument("--camera", default="Camera.002")
    parser.add_argument("--width", type=int, default=640)
    return parser.parse_args(argv)


def main() -> None:
    options = args()
    os.makedirs(options.output, exist_ok=True)
    scene = bpy.context.scene
    camera = bpy.data.objects.get(options.camera)
    if not camera or camera.type != "CAMERA":
        raise RuntimeError(f"Camera not found: {options.camera}")
    scene.camera = camera
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.resolution_x = options.width
    scene.render.resolution_y = round(options.width * 890 / 1372)
    scene.render.resolution_percentage = 100
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = "WORLD"

    for obj in bpy.data.objects:
        if obj.type == "CURVES":
            obj.hide_render = True

    label = os.path.splitext(os.path.basename(bpy.data.filepath))[0]
    for frame in [int(item) for item in options.frames.split(",")]:
        scene.frame_set(frame)
        scene.render.filepath = os.path.join(
            options.output,
            f"{label}_frame_{frame:04d}.png",
        )
        bpy.ops.render.render(write_still=True)
        print("MGF_RENDERED=" + scene.render.filepath)


if __name__ == "__main__":
    main()
