"""Render low-cost preview frames from a generated Anipose Blender project."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view


def blender_args() -> list[str]:
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera", default="Camera.001")
    parser.add_argument("--frame", type=int, action="append", required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--focus-bone")
    parser.add_argument("--focus-size", type=float, default=0.16)
    parser.add_argument("--hide-curves", action="store_true")
    return parser.parse_args(blender_args())


def main() -> None:
    args = parse_args()
    camera = bpy.data.objects.get(args.camera)
    if camera is None or camera.type != "CAMERA":
        raise KeyError(f"Camera not found: {args.camera}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.camera = camera
    scene.use_nodes = False
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "SINGLE"
    scene.display.shading.single_color = (0.72, 0.72, 0.72)
    scene.display.shading.background_type = "VIEWPORT"
    scene.display.shading.background_color = (0.055, 0.065, 0.08)
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    if args.hide_curves:
        for obj in bpy.data.objects:
            if obj.type == "CURVES":
                obj.hide_render = True
                obj.hide_viewport = True
                for modifier in obj.modifiers:
                    modifier.show_render = False
                    modifier.show_viewport = False
    for frame in args.frame:
        scene.frame_set(frame)
        if args.focus_bone:
            armature = bpy.data.objects["DeformationSystem"]
            bone = armature.pose.bones[args.focus_bone]
            focus = armature.matrix_world @ bone.tail
            projected = world_to_camera_view(scene, camera, focus)
            size = max(0.02, args.focus_size)
            scene.render.use_border = True
            scene.render.use_crop_to_border = True
            scene.render.border_min_x = max(0.0, projected.x - size)
            scene.render.border_max_x = min(1.0, projected.x + size)
            scene.render.border_min_y = max(0.0, projected.y - size)
            scene.render.border_max_y = min(1.0, projected.y + size)
        scene.render.filepath = str(args.output_dir / f"{args.camera}_frame_{frame:06d}.png")
        bpy.ops.render.render(write_still=True)
        print(f"MGF_PREVIEW={scene.render.filepath}")


if __name__ == "__main__":
    main()
