#!/usr/bin/env python3
"""Render the animated mouse mesh alone from a fixed millimeter-scale diagnostic view."""

from pathlib import Path

import bpy
from mathutils import Vector


OUTPUT_DIR = Path(
    r"\\wsl.localhost\Ubuntu\home\haozong\projects\MouseGeoForge\dataset\Cylinder\review\sceneonly_moseq_pose_only"
)
MOUSE_NAME = "MS_Mouse.001"


def point_camera(camera: bpy.types.Object, target: Vector) -> None:
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    mouse = bpy.data.objects[MOUSE_NAME]
    for obj in bpy.data.objects:
        if obj.type in {"MESH", "CURVES", "CURVE", "LIGHT"}:
            obj.hide_render = obj != mouse
    mouse.hide_render = False
    mouse.hide_viewport = False

    camera_data = bpy.data.cameras.new("MGF_PoseOnlyReviewCamera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 150.0
    camera_data.clip_start = 1.0
    camera_data.clip_end = 2000.0
    camera = bpy.data.objects.new("MGF_PoseOnlyReviewCamera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (190.0, -230.0, 450.0)
    point_camera(camera, Vector((0.0, 0.0, 365.0)))
    scene.camera = camera

    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.render.resolution_x = 640
    scene.render.resolution_y = 640
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.world.color = (0.9, 0.9, 0.9)
    scene.render.use_compositing = False
    scene.render.use_sequencer = False

    for frame in (1, 100, 200):
        scene.frame_set(frame)
        output = OUTPUT_DIR / f"pose_only_frame_{frame:04d}.png"
        scene.render.filepath = str(output)
        bpy.ops.render.render(write_still=True)
        print(f"POSE_ONLY_RENDER {output}", flush=True)


if __name__ == "__main__":
    main()
