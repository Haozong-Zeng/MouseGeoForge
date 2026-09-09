#!/usr/bin/env python3
"""Render the MARS homecage template from its configured multiview cameras."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy


CAMERAS = (
    ("top", "HomecageCam_Top", "MGF_Capture_Clean"),
    ("side_neg_x", "HomecageCam_SideNegX", "MGF_Capture_Clean"),
    ("side_neg_y", "HomecageCam_SideNegY", "MGF_Capture_Clean"),
    ("side_pos_y", "HomecageCam_SidePosY", "MGF_Capture_Clean"),
)


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--engine",
        choices=("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES", "BLENDER_WORKBENCH"),
        default="BLENDER_EEVEE",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Optional square override retained for quick legacy previews.",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--views", default="all")
    parser.add_argument("--frame", type=int, default=1)
    return parser.parse_args(args_after_dash())


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    if not scene.frame_start <= args.frame <= scene.frame_end:
        raise ValueError(
            f"Frame {args.frame} is outside [{scene.frame_start}, {scene.frame_end}]"
        )
    scene.frame_set(args.frame)
    engine = args.engine
    engine_items = {
        item.identifier
        for item in scene.bl_rna.properties["render"].fixed_type.properties["engine"].enum_items
    }
    if engine == "BLENDER_EEVEE_NEXT" and engine not in engine_items:
        engine = "BLENDER_EEVEE"
    scene.render.engine = engine
    width = args.resolution if args.resolution is not None else args.width
    height = args.resolution if args.resolution is not None else args.height
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    if engine == "CYCLES":
        preferences = bpy.context.preferences.addons["cycles"].preferences
        for backend in ("OPTIX", "CUDA"):
            try:
                preferences.compute_device_type = backend
                preferences.get_devices()
                break
            except Exception:
                continue
        gpu_devices = [
            device
            for device in preferences.devices
            if device.type in {"OPTIX", "CUDA"}
        ]
        for device in preferences.devices:
            device.use = device in gpu_devices
        if not gpu_devices:
            raise RuntimeError("No CUDA/OptiX device available for Cycles preview")
        scene.cycles.device = "GPU"
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = True
    scene.view_settings.look = "Medium Low Contrast"

    requested = {name.strip() for name in args.views.split(",") if name.strip()}
    selected = CAMERAS if "all" in requested else tuple(
        record for record in CAMERAS if record[0] in requested
    )
    if not selected:
        raise ValueError(f"No known views selected from {args.views!r}")

    rendered = []
    for label, camera_name, view_layer_name in selected:
        camera = bpy.data.objects.get(camera_name)
        view_layer = scene.view_layers.get(view_layer_name)
        if camera is None:
            raise KeyError(camera_name)
        if view_layer is None:
            raise KeyError(view_layer_name)
        scene.camera = camera
        output = output_dir / f"{label}.png"
        scene.render.filepath = str(output)
        bpy.ops.render.render(write_still=True, layer=view_layer.name)
        rendered.append(
            {
                "label": label,
                "camera": camera.name,
                "view_layer": view_layer.name,
                "path": str(output),
            }
        )

    manifest_path = output_dir / "render_manifest.json"
    previous_renders = {}
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous_renders = {
            record["label"]: record for record in previous.get("renders", [])
        }
    for label, camera_name, view_layer_name in CAMERAS:
        existing_path = output_dir / f"{label}.png"
        if existing_path.is_file() and label not in previous_renders:
            previous_renders[label] = {
                "label": label,
                "camera": camera_name,
                "view_layer": view_layer_name,
                "path": str(existing_path),
            }
    previous_renders.update({record["label"]: record for record in rendered})
    label_order = [record[0] for record in CAMERAS]
    merged_renders = [
        previous_renders[label]
        for label in label_order
        if label in previous_renders
    ]
    manifest = {
        "blend": bpy.data.filepath,
        "engine": engine,
        "frame": args.frame,
        "resolution": [width, height],
        "samples": args.samples,
        "material_profile": scene.get("MGF_homecage_material_profile"),
        "renders": merged_renders,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
