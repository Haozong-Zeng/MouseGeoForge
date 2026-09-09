#!/usr/bin/env python3
"""Render lit close-up views of the animated mouse while retaining SceneOnly glass."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import bpy
from mathutils import Vector


CAMERAS = ("Camera.014", "Camera.015", "Camera.017", "Camera.018")
MOUSE = "MS_Mouse.001"


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=100)
    parser.add_argument("--resolution", type=int, default=960)
    parser.add_argument("--samples", type=int, default=64)
    return parser.parse_args(argv)


def configure_optix(scene: bpy.types.Scene) -> list[str]:
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = "OPTIX"
    preferences.get_devices()
    selected = []
    for device in preferences.devices:
        device.use = device.type == "OPTIX" and "RTX PRO 6000" in device.name.upper()
        if device.use:
            selected.append(device.name)
    if not selected:
        raise RuntimeError("RTX PRO 6000 OptiX device not found")
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    return selected


def point_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def add_area_light(name: str, location: tuple[float, float, float], energy: float, size: float, target: Vector) -> None:
    data = bpy.data.lights.new(name, "AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = location
    point_at(obj, target)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.frame_set(args.frame)
    bpy.context.view_layer.update()
    evaluated = bpy.data.objects[MOUSE].evaluated_get(bpy.context.evaluated_depsgraph_get())
    corners = [evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box]
    mouse_min = Vector(tuple(min(point[axis] for point in corners) for axis in range(3)))
    mouse_max = Vector(tuple(max(point[axis] for point in corners) for axis in range(3)))
    target = (mouse_min + mouse_max) * 0.5

    devices = configure_optix(scene)
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "16"
    scene.render.film_transparent = False
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.render.use_persistent_data = True
    scene.cycles.samples = args.samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.01
    scene.cycles.adaptive_min_samples = 16
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 16
    scene.cycles.diffuse_bounces = 4
    scene.cycles.glossy_bounces = 8
    scene.cycles.transmission_bounces = 12
    scene.cycles.transparent_max_bounces = 24
    scene.cycles.volume_bounces = 2
    scene.cycles.sample_clamp_indirect = 12.0
    try:
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
    except (TypeError, ValueError):
        pass
    try:
        scene.view_settings.view_transform = "AgX"
    except TypeError:
        pass
    scene.view_settings.exposure = 1.0
    scene.view_settings.gamma = 1.0

    add_area_light("MGF_FurReview_Top", (0.0, 0.0, 610.0), 420000.0, 260.0, target)
    add_area_light("MGF_FurReview_Key", (0.0, -260.0, 485.0), 260000.0, 180.0, target)
    add_area_light("MGF_FurReview_Rim", (260.0, 170.0, 470.0), 180000.0, 150.0, target)

    records = []
    for index, camera_name in enumerate(CAMERAS):
        camera = bpy.data.objects[camera_name]
        camera.data.lens = 72.0
        camera.data.dof.use_dof = False
        camera.data.shift_x = 0.0
        camera.data.shift_y = 0.0
        point_at(camera, target)
        scene.camera = camera
        output = output_dir / f"frame_{args.frame:04d}_fur_closeup_{index:02d}_{camera_name.replace('.', '_')}.png"
        scene.render.filepath = str(output)
        started = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        records.append(
            {
                "camera": camera_name,
                "output": str(output),
                "seconds": time.perf_counter() - started,
                "lens_mm": camera.data.lens,
                "temporary_review_lighting": True,
            }
        )
        print(f"FUR_CLOSEUP {json.dumps(records[-1])}", flush=True)

    manifest = {
        "format": "sceneonly_cycles_fur_closeup_v1",
        "blend": bpy.data.filepath,
        "frame": args.frame,
        "resolution": [args.resolution, args.resolution],
        "samples": args.samples,
        "devices": devices,
        "mouse_bounds": [list(mouse_min), list(mouse_max)],
        "note": "Review-only lens and soft fill lighting; source blend was not saved or modified.",
        "renders": records,
    }
    (output_dir / "fur_closeup_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
