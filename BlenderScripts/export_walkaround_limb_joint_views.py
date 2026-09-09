#!/usr/bin/env python3
"""Render clean, mouse-centric WalkAround views and export projected limb joints."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector


JOINT_BONES = {
    "left_shoulder": ("Scapula_L", "head"),
    "left_elbow": ("Wrist_L", "head"),
    "left_wrist": ("Fingers1_L", "head"),
    "right_shoulder": ("Scapula_R", "head"),
    "right_elbow": ("Wrist_R", "head"),
    "right_wrist": ("Fingers1_R", "head"),
    "left_hip": ("Hip_L", "head"),
    "left_knee": ("Knee_L", "head"),
    "left_ankle": ("Ankle_L", "head"),
    "right_hip": ("Hip_R", "head"),
    "right_knee": ("Knee_R", "head"),
    "right_ankle": ("Ankle_R", "head"),
}


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--armature", default="DeformationSystem")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=91)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--oblique-step", type=int, default=10)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=540)
    return parser.parse_args(args_after_dash())


def configure_scene(width: int, height: int) -> None:
    scene = bpy.context.scene
    scene.use_nodes = False
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"

    shading = scene.display.shading
    shading.light = "STUDIO"
    shading.color_type = "OBJECT"
    shading.show_cavity = True
    shading.cavity_type = "BOTH"
    shading.show_shadows = True
    shading.show_specular_highlight = True
    shading.background_type = "VIEWPORT"
    shading.background_color = (0.93, 0.93, 0.93)


def isolate_animated_mouse(armature: bpy.types.Object) -> bpy.types.Object:
    mouse = bpy.data.objects.get("MS_Mouse")
    if mouse is None:
        raise RuntimeError("Expected animated mouse mesh 'MS_Mouse' was not found")

    for obj in bpy.data.objects:
        keep = obj == mouse or obj == armature or obj.type == "CAMERA"
        if obj.type in {"MESH", "CURVES", "SURFACE", "META", "FONT"}:
            obj.hide_render = not keep
            try:
                obj.hide_set(not keep)
            except RuntimeError:
                pass
    mouse.hide_render = False
    mouse.hide_viewport = False
    mouse.color = (0.34, 0.20, 0.12, 1.0)
    return mouse


def make_camera() -> bpy.types.Object:
    camera_data = bpy.data.cameras.new("WalkAroundJointAuditCameraData")
    camera_data.lens = 35.0
    camera_data.sensor_width = 36.0
    camera_data.clip_start = 0.1
    camera_data.clip_end = 2000.0
    camera = bpy.data.objects.new("WalkAroundJointAuditCamera", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    return camera


def pose_point(armature: bpy.types.Object, bone_name: str, endpoint: str = "head") -> Vector:
    bone = armature.pose.bones.get(bone_name)
    if bone is None:
        raise KeyError(f"Missing pose bone: {bone_name}")
    return armature.matrix_world @ getattr(bone, endpoint)


def body_basis(armature: bpy.types.Object) -> tuple[Vector, Vector, Vector]:
    root = pose_point(armature, "Root_M")
    head = pose_point(armature, "Head1_M")
    target = (root + head) * 0.5
    forward = head - root
    forward.z = 0.0
    if forward.length < 1e-6:
        forward = Vector((0.0, 1.0, 0.0))
    else:
        forward.normalize()
    side = Vector((forward.y, -forward.x, 0.0)).normalized()
    return target, forward, side


def set_camera_axes(camera: bpy.types.Object, location: Vector, target: Vector, up_hint: Vector) -> None:
    view = (target - location).normalized()
    right = view.cross(up_hint)
    if right.length < 1e-6:
        right = Vector((1.0, 0.0, 0.0))
    else:
        right.normalize()
    up = right.cross(view).normalized()
    camera.matrix_world = Matrix(
        (
            (right.x, up.x, -view.x, location.x),
            (right.y, up.y, -view.y, location.y),
            (right.z, up.z, -view.z, location.z),
            (0.0, 0.0, 0.0, 1.0),
        )
    )


def position_camera(camera: bpy.types.Object, view_name: str, armature: bpy.types.Object) -> None:
    target, forward, side = body_basis(armature)
    target.z += 1.5
    if view_name == "top":
        camera.data.lens = 35.0
        location = target + Vector((0.0, 0.0, 150.0))
        set_camera_axes(camera, location, target, forward)
    elif view_name == "left_oblique":
        camera.data.lens = 48.0
        location = target - side * 135.0 - forward * 20.0 + Vector((0.0, 0.0, 85.0))
        set_camera_axes(camera, location, target, Vector((0.0, 0.0, 1.0)))
    elif view_name == "right_oblique":
        camera.data.lens = 48.0
        location = target + side * 135.0 - forward * 20.0 + Vector((0.0, 0.0, 85.0))
        set_camera_axes(camera, location, target, Vector((0.0, 0.0, 1.0)))
    else:
        raise ValueError(f"Unknown view: {view_name}")


def joint_payload(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    armature: bpy.types.Object,
    width: int,
    height: int,
) -> dict:
    joints = {}
    for joint_name, (bone_name, endpoint) in JOINT_BONES.items():
        world = pose_point(armature, bone_name, endpoint)
        ndc = world_to_camera_view(scene, camera, world)
        joints[joint_name] = {
            "bone": bone_name,
            "endpoint": endpoint,
            "world": [float(value) for value in world],
            "pixel": [float(ndc.x * width), float((1.0 - ndc.y) * height)],
            "camera_depth": float(ndc.z),
            "in_frame": bool(0.0 <= ndc.x <= 1.0 and 0.0 <= ndc.y <= 1.0 and ndc.z > 0.0),
        }
    return joints


def render_view(
    output_dir: Path,
    frame: int,
    view_name: str,
    camera: bpy.types.Object,
    armature: bpy.types.Object,
    width: int,
    height: int,
) -> dict:
    scene = bpy.context.scene
    position_camera(camera, view_name, armature)
    bpy.context.view_layer.update()
    image_path = output_dir / "clean" / view_name / f"frame_{frame:04d}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(image_path)
    bpy.ops.render.render(write_still=True)
    print(f"rendered={image_path}")
    return {
        "frame": frame,
        "view": view_name,
        "image": str(image_path),
        "camera_matrix": [[float(value) for value in row] for row in camera.matrix_world],
        "joints": joint_payload(scene, camera, armature, width, height),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    armature = bpy.data.objects.get(args.armature)
    if armature is None or armature.type != "ARMATURE":
        raise RuntimeError(f"Armature {args.armature!r} was not found")

    configure_scene(args.width, args.height)
    isolate_animated_mouse(armature)
    camera = make_camera()
    records = []
    top_frames = list(range(args.start, args.end + 1, args.step))
    oblique_frames = list(range(args.start, args.end + 1, args.oblique_step))
    if oblique_frames[-1] != args.end:
        oblique_frames.append(args.end)

    for frame in top_frames:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        records.append(render_view(output_dir, frame, "top", camera, armature, args.width, args.height))
    for frame in oblique_frames:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        for view_name in ("left_oblique", "right_oblique"):
            records.append(render_view(output_dir, frame, view_name, camera, armature, args.width, args.height))

    metadata = {
        "blend_file": bpy.data.filepath,
        "armature": armature.name,
        "action": armature.animation_data.action.name
        if armature.animation_data and armature.animation_data.action
        else None,
        "frame_range": [args.start, args.end],
        "top_step": args.step,
        "oblique_step": args.oblique_step,
        "resolution": [args.width, args.height],
        "joint_definition": {
            name: {"bone": bone, "endpoint": endpoint}
            for name, (bone, endpoint) in JOINT_BONES.items()
        },
        "records": records,
    }
    metadata_path = output_dir / "joint_projections.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"metadata_written={metadata_path}")


if __name__ == "__main__":
    main()
