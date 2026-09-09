#!/usr/bin/env python3
"""Render rest-pose closeups from a vertex-keypoint-map debug Blender file."""

from __future__ import annotations

import argparse
import bpy
import math
import sys
from pathlib import Path

from mathutils import Vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mesh", default="MS_Mouse")
    parser.add_argument("--resolution", type=int, default=720)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def look_at(camera: bpy.types.Object, target: Vector) -> None:
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()


def marker_local_bounds(mesh_obj: bpy.types.Object, predicate) -> tuple[Vector, Vector]:
    points = []
    for obj in bpy.data.objects:
        if obj.type != "MESH" or not obj.name.startswith("VKP_") or not predicate(obj):
            continue
        for vertex in obj.data.vertices:
            points.append(mesh_obj.matrix_world.inverted() @ (obj.matrix_world @ vertex.co))
    if not points:
        raise ValueError("No debug marker vertices found for requested region")
    minimum = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
    maximum = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
    return minimum, maximum


def configure_scene(scene: bpy.types.Scene, resolution: int) -> None:
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = "BOTH"
    scene.display.shading.background_type = "VIEWPORT"
    scene.display.shading.background_color = (0.035, 0.035, 0.035)


def main() -> None:
    args = parse_args()
    output_dir = Path(bpy.path.abspath(args.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    configure_scene(scene, args.resolution)

    mesh_obj = bpy.data.objects.get(args.mesh)
    if mesh_obj is None or mesh_obj.type != "MESH":
        raise KeyError(f"Mesh {args.mesh!r} not found")
    mesh_obj.color = (0.35, 0.35, 0.35, 1.0)

    armature = bpy.data.objects.get("DeformationSystem")
    if armature is not None:
        armature.hide_render = True

    camera_data = bpy.data.cameras.new("VKP_DebugCamera")
    camera_data.type = "ORTHO"
    camera = bpy.data.objects.new("VKP_DebugCamera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera

    marker_objects = [
        obj
        for obj in bpy.data.objects
        if obj.type == "MESH" and obj.name.startswith("VKP_")
    ]

    def render_region(name: str, predicate, views: dict[str, Vector], padding: float) -> None:
        for marker in marker_objects:
            marker.hide_render = not predicate(marker)
        minimum, maximum = marker_local_bounds(mesh_obj, predicate)
        center_local = 0.5 * (minimum + maximum)
        span_local = maximum - minimum
        center_world = mesh_obj.matrix_world @ center_local
        world_scale = sum(mesh_obj.matrix_world.to_scale()) / 3.0
        camera.data.ortho_scale = max(span_local.x, span_local.y, span_local.z) * world_scale * padding
        distance_local = max(span_local.length * 2.5, 1.0)

        for view_name, direction_local in views.items():
            camera_local = center_local + direction_local * distance_local
            camera.location = mesh_obj.matrix_world @ camera_local
            look_at(camera, center_world)
            scene.render.filepath = str(output_dir / f"{name}_{view_name}.png")
            bpy.ops.render.render(write_still=True)
            print(f"rendered {scene.render.filepath}")

    hand_views = {
        "surface": Vector((0.0, -1.0, 0.0)),
        "oblique": Vector((0.65, -1.0, -0.35)).normalized(),
    }
    for side in ("L", "R"):
        render_region(
            f"hand_{side}",
            lambda obj, side=side: "Finger" in obj.name and obj.name.endswith(f"_{side}"),
            hand_views,
            1.35,
        )
        render_region(
            f"elbow_{side}",
            lambda obj, side=side: obj.name == f"VKP_Wrist_{side}",
            {
                "surface": Vector((0.0, -1.0, 0.0)),
                "oblique": Vector((0.7, -1.0, 0.25)).normalized(),
            },
            4.0,
        )

    render_region(
        "nose",
        lambda obj: obj.name == "VKP_HeadEnd_M",
        {
            "front": Vector((0.0, 0.0, 1.0)),
            "profile": Vector((0.0, -1.0, 0.0)),
        },
        4.0,
    )


if __name__ == "__main__":
    main()
