#!/usr/bin/env python3
"""Render one SceneOnly animation frame from the bottom and six ring cameras."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import bpy


DEFAULT_CAMERAS = tuple(f"Camera.{index:03d}" for index in range(12, 20))
FUR_OBJECTS = (
    "BaseFur.001",
    "EarFur.001",
    "EyesFur.001",
    "LegFur.001",
    "Mouth.001",
    "TailFur.001",
    "Whisker.001",
    "MGF_AnimatedMouse_GuardShort",
    "MGF_AnimatedMouse_GuardLong",
)


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=100)
    parser.add_argument("--resolution", type=int, default=720)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--adaptive-threshold", type=float, default=0.015)
    parser.add_argument("--exposure", type=float, default=0.7)
    parser.add_argument("--cameras", nargs="*", default=list(DEFAULT_CAMERAS))
    return parser.parse_args(argv)


def configure_optix(scene: bpy.types.Scene) -> dict:
    preferences = bpy.context.preferences.addons["cycles"].preferences
    selected_backend = None
    backend_errors = {}
    for backend in ("OPTIX", "CUDA"):
        try:
            preferences.compute_device_type = backend
            preferences.get_devices()
            selected_backend = backend
            break
        except Exception as error:
            backend_errors[backend] = str(error)
    if selected_backend is None:
        raise RuntimeError(f"Could not initialize an NVIDIA Cycles backend: {backend_errors}")

    selected_devices = []
    for device in preferences.devices:
        use = device.type == selected_backend and "RTX PRO 6000" in device.name.upper()
        device.use = use
        if use:
            selected_devices.append({"name": device.name, "type": device.type})
    if not selected_devices:
        available = [{"name": device.name, "type": device.type} for device in preferences.devices]
        raise RuntimeError(f"RTX PRO 6000 {selected_backend} device unavailable: {available}")

    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    return {"backend": selected_backend, "devices": selected_devices}


def set_if_present(target, attribute: str, value) -> None:
    if hasattr(target, attribute):
        try:
            setattr(target, attribute, value)
        except (TypeError, ValueError):
            pass


def configure_cycles(scene: bpy.types.Scene, args: argparse.Namespace) -> dict:
    device = configure_optix(scene)
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 15
    scene.render.film_transparent = False
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.render.use_persistent_data = True

    cycles = scene.cycles
    cycles.samples = args.samples
    cycles.use_adaptive_sampling = True
    cycles.adaptive_threshold = args.adaptive_threshold
    cycles.adaptive_min_samples = min(16, args.samples)
    cycles.use_denoising = True
    set_if_present(cycles, "denoiser", "OPENIMAGEDENOISE")
    set_if_present(cycles, "denoising_input_passes", "RGB_ALBEDO_NORMAL")
    set_if_present(cycles, "denoising_prefilter", "ACCURATE")
    cycles.max_bounces = 16
    cycles.diffuse_bounces = 4
    cycles.glossy_bounces = 8
    cycles.transmission_bounces = 12
    cycles.transparent_max_bounces = 24
    cycles.volume_bounces = 2
    cycles.sample_clamp_indirect = 12.0
    set_if_present(cycles, "caustics_reflective", True)
    set_if_present(cycles, "caustics_refractive", True)
    set_if_present(cycles, "use_light_tree", True)

    try:
        scene.view_settings.view_transform = "AgX"
    except TypeError:
        pass
    for look in ("AgX - Medium High Contrast", "Medium High Contrast"):
        try:
            scene.view_settings.look = look
            break
        except TypeError:
            continue
    scene.view_settings.exposure = args.exposure
    scene.view_settings.gamma = 1.0
    return device


def enable_fur() -> list[dict]:
    result = []
    for name in FUR_OBJECTS:
        obj = bpy.data.objects.get(name)
        if obj is None:
            raise KeyError(f"Missing fur object {name}")
        obj.hide_render = False
        result.append(
            {
                "name": name,
                "type": obj.type,
                "materials": [material.name for material in obj.data.materials if material],
            }
        )
    return result


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.frame_set(args.frame)
    bpy.context.view_layer.update()
    device = configure_cycles(scene, args)
    fur = enable_fur()

    camera_records = []
    total_started = time.perf_counter()
    for index, camera_name in enumerate(args.cameras):
        camera = bpy.data.objects.get(camera_name)
        if camera is None or camera.type != "CAMERA":
            raise KeyError(f"Missing camera {camera_name}")
        camera.hide_render = False
        scene.camera = camera
        output = output_dir / f"frame_{args.frame:04d}_view_{index:02d}_{camera_name.replace('.', '_')}.png"
        scene.render.filepath = str(output)
        started = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        elapsed = time.perf_counter() - started
        record = {
            "view_index": index,
            "camera": camera_name,
            "output": str(output),
            "seconds": elapsed,
            "bytes": output.stat().st_size,
        }
        camera_records.append(record)
        print(f"HQ_RENDER {json.dumps(record)}", flush=True)

    manifest = {
        "format": "sceneonly_cycles_multiview_hq_v1",
        "blend": bpy.data.filepath,
        "frame": args.frame,
        "resolution": [args.resolution, args.resolution],
        "samples": args.samples,
        "adaptive_threshold": args.adaptive_threshold,
        "exposure": args.exposure,
        "color_mode": scene.render.image_settings.color_mode,
        "color_depth": scene.render.image_settings.color_depth,
        "engine": scene.render.engine,
        "path_tracing": {
            "max_bounces": scene.cycles.max_bounces,
            "diffuse_bounces": scene.cycles.diffuse_bounces,
            "glossy_bounces": scene.cycles.glossy_bounces,
            "transmission_bounces": scene.cycles.transmission_bounces,
            "transparent_max_bounces": scene.cycles.transparent_max_bounces,
            "volume_bounces": scene.cycles.volume_bounces,
        },
        "device": device,
        "fur": fur,
        "cameras": camera_records,
        "total_seconds": time.perf_counter() - total_started,
    }
    (output_dir / "render_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
