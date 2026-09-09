#!/usr/bin/env python3
"""Render fast hair-free top/side pose review frames from a Homecage blend."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector


CAMERAS = {
    "top": "HomecageCam_Top",
    "side_neg_y": "HomecageCam_SideNegY",
}


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", default="1,15,29,43,57")
    parser.add_argument("--views", default="top,side_neg_y")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--closeup", action="store_true")
    args = parser.parse_args(args_after_dash())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frames = [int(value) for value in args.frames.split(",")]
    views = [value.strip() for value in args.views.split(",") if value.strip()]
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = "WORLD"
    for obj in bpy.data.objects:
        if obj.type == "CURVES":
            obj.hide_render = True
    view_layer = scene.view_layers.get("MGF_Capture_Clean") or scene.view_layers[0]
    rendered = []
    for frame in frames:
        scene.frame_set(frame)
        if args.closeup:
            roots = []
            for name in ("LP_Black_Armature", "LP_Brown_Armature"):
                armature = bpy.data.objects[name]
                roots.append(armature.matrix_world @ armature.pose.bones["Root_M"].head)
            center = sum(roots, Vector()) / len(roots)
        for view in views:
            camera = bpy.data.objects[CAMERAS[view]]
            if args.closeup:
                camera.data.type = "ORTHO"
                camera.data.ortho_scale = 90.0 if view == "top" else 70.0
                if view == "top":
                    camera.location = Vector((center.x, center.y, center.z + 300.0))
                    camera.rotation_euler = (0.0, 0.0, 0.0)
                else:
                    camera.location = Vector((center.x, center.y - 400.0, center.z + 18.0))
                    camera.rotation_euler = (Vector((center.x, center.y, center.z + 3.0)) - camera.location).to_track_quat("-Z", "Y").to_euler()
            scene.camera = camera
            path = output / f"{view}_frame{frame:03d}.png"
            scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True, layer=view_layer.name)
            rendered.append(str(path))
    manifest = {
        "blend": bpy.data.filepath,
        "frames": frames,
        "views": views,
        "resolution": [args.width, args.height],
        "fur": "hidden for pose QA",
        "closeup": args.closeup,
        "renders": rendered,
    }
    (output / "render_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
