#!/usr/bin/env python3
"""Render palm-roll candidates around the evaluated wrist Y axis."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Quaternion, Vector


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--side", choices=("L", "R"), required=True)
    parser.add_argument("--offset", type=float, action="append", required=True)
    parser.add_argument("--camera", default="Camera.002")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    return parser.parse_args(args_after_dash())


def configure_render(resolution: int) -> None:
    scene = bpy.context.scene
    scene.use_nodes = False
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    for engine in ("BLENDER_WORKBENCH", "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            scene.render.engine = engine
            break
        except TypeError:
            continue
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "Standard"
    if hasattr(scene, "display") and hasattr(scene.display, "shading"):
        shading = scene.display.shading
        shading.light = "STUDIO"
        shading.color_type = "OBJECT"
        shading.show_cavity = True
        shading.show_shadows = True


def bone_tail_world(armature, bone) -> Vector:
    return armature.matrix_world @ (
        bone.matrix @ Vector((0.0, bone.bone.length, 0.0))
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.frame_set(args.frame)
    bpy.context.view_layer.update()
    configure_render(args.resolution)
    scene.camera = bpy.data.objects[args.camera]

    armature = bpy.data.objects["DeformationSystem"]
    wrist = armature.pose.bones[f"Wrist_{args.side}"]
    controller = bpy.data.objects[f"CTRL_Hand_{args.side}"]
    head = armature.matrix_world @ wrist.matrix.translation
    tail = bone_tail_world(armature, wrist)
    axis = (tail - head).normalized()
    original = controller.matrix_world.to_quaternion().copy()

    for offset in args.offset:
        controller.rotation_mode = "QUATERNION"
        controller.rotation_quaternion = (
            Quaternion(axis, math.radians(offset)) @ original
        ).normalized()
        controller.keyframe_insert("rotation_quaternion", frame=args.frame)
        bpy.context.view_layer.update()
        label = f"{offset:+06.1f}".replace("+", "p").replace("-", "m")
        output = args.output_dir / (
            f"frame_{args.frame:04d}_{args.side}_roll_{label}_{args.camera}.png"
        )
        scene.render.filepath = str(output)
        bpy.ops.render.render(write_still=True)
        print(output)


if __name__ == "__main__":
    main()
