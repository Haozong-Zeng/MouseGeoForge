#!/usr/bin/env python3
"""Render a few Cycles review frames from the SceneOnly pose-composite project."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import bpy


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[1, 100, 200])
    parser.add_argument("--cameras", nargs="+", default=["Camera.009", "Camera.013"])
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--engine", choices=("CYCLES", "BLENDER_EEVEE"), default="CYCLES")
    return parser.parse_args(argv)


def configure_gpu(scene: bpy.types.Scene) -> list[str]:
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = "CUDA"
    preferences.get_devices()
    selected = []
    for device in preferences.devices:
        device.use = device.type == "CUDA" and "RTX PRO 6000" in device.name
        if device.use:
            selected.append(device.name)
    if not selected:
        raise RuntimeError(f"RTX PRO 6000 unavailable: {[device.name for device in preferences.devices]}")
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    return selected


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    if args.engine == "CYCLES":
        devices = configure_gpu(scene)
    else:
        scene.render.engine = args.engine
        devices = [args.engine]
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    if args.engine == "CYCLES":
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = True
    scene.render.use_persistent_data = True

    records = []
    for frame in args.frames:
        scene.frame_set(frame)
        for camera_name in args.cameras:
            camera = bpy.data.objects.get(camera_name)
            if camera is None or camera.type != "CAMERA":
                raise KeyError(f"Missing camera {camera_name}")
            scene.camera = camera
            output = output_dir / f"frame_{frame:04d}_{camera_name.replace('.', '_')}.png"
            scene.render.filepath = str(output)
            started = time.perf_counter()
            bpy.ops.render.render(write_still=True)
            elapsed = time.perf_counter() - started
            records.append({"frame": frame, "camera": camera_name, "seconds": elapsed, "file": str(output)})
            print(f"REVIEW_RENDER frame={frame} camera={camera_name} seconds={elapsed:.2f}", flush=True)

    manifest = {
        "blend": bpy.data.filepath,
        "frames": args.frames,
        "cameras": args.cameras,
        "resolution": args.resolution,
        "samples": args.samples,
        "devices": devices,
        "renders": records,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
