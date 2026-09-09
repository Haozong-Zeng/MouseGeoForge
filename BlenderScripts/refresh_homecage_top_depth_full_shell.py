#!/usr/bin/env python3
"""Regenerate top-view depth while retaining the complete homecage shell."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import bpy
import numpy as np


ROOT = Path(r"\\wsl.localhost\Ubuntu\home\haozong\projects\MouseGeoForge")
PROJECT_NAME = "Homecage-BlackFur-11_01_22_12wk_m0-Core23-Sample10-Frames200-399"
EXPECTED_BLEND = ROOT / "dataset" / "Homecage" / f"{PROJECT_NAME}.blend"
DEFAULT_DATASET = ROOT / "dataset" / "BlenderPyMouse" / PROJECT_NAME
RENDERER_PATH = ROOT / "BlenderScripts" / "blender_water_reaching_rendering.py"
CAMERA_NAME = "HomecageCam_Top"
SOURCE_SHELL = "MGF_MARS_HomecageShell"
OPAQUE_MATERIAL = "MGF_DepthProxy_Opaque"
VIEW_ID = 0


def load_renderer():
    spec = importlib.util.spec_from_file_location("mgf_blendedmvs_renderer", RENDERER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET))
    parser.add_argument("--diagnostic-root", default="")
    parser.add_argument("--frame-start", type=int, default=1)
    parser.add_argument("--frame-end", type=int, default=200)
    parser.add_argument(
        "--engine",
        choices=["CYCLES", "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"],
        default="BLENDER_EEVEE_NEXT",
    )
    parser.add_argument("--cycles-samples", type=int, default=1)
    parser.add_argument("--commit", action=argparse.BooleanOptionalAction, default=False)
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return parser.parse_args(argv)


def safe_remove_temp(path: Path, parent: Path) -> None:
    resolved = path.resolve()
    parent_resolved = parent.resolve()
    if (
        parent_resolved not in resolved.parents
        or not resolved.name.startswith("_top_full_shell_depth_tmp_")
    ):
        raise ValueError(f"Refuse to remove unexpected temp path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if Path(bpy.data.filepath).resolve() != EXPECTED_BLEND.resolve():
        raise ValueError(f"Expected {EXPECTED_BLEND}, opened {bpy.data.filepath}")
    if args.frame_start < 1 or args.frame_end < args.frame_start:
        raise ValueError("Invalid frame range")
    if args.commit and args.diagnostic_root:
        raise ValueError("Use either --commit or --diagnostic-root, not both")
    if not args.commit and not args.diagnostic_root:
        raise ValueError("Diagnostic mode requires --diagnostic-root")

    renderer = load_renderer()
    scene = bpy.context.scene
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    renderer.configure_render_engine(
        scene,
        SimpleNamespace(engine=args.engine, cycles_samples=args.cycles_samples),
    )

    dataset_root = Path(args.dataset_root).resolve()
    output_root = dataset_root if args.commit else Path(args.diagnostic_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    temp_root = output_root / (
        f"_top_full_shell_depth_tmp_{args.frame_start:06d}_{args.frame_end:06d}"
    )
    safe_remove_temp(temp_root, output_root)
    temp_root.mkdir(parents=True)

    camera = bpy.data.objects.get(CAMERA_NAME)
    shell = bpy.data.objects.get(SOURCE_SHELL)
    if camera is None or camera.type != "CAMERA":
        raise KeyError(f"Missing top camera {CAMERA_NAME!r}")
    if shell is None or shell.type != "MESH":
        raise KeyError(f"Missing shell mesh {SOURCE_SHELL!r}")
    opaque_material = bpy.data.materials.get(OPAQUE_MATERIAL)
    if opaque_material is None:
        raise KeyError(f"Missing depth material {OPAQUE_MATERIAL!r}")
    proxies = [
        obj for obj in bpy.data.objects
        if bool(obj.get("mgf_depth_proxy", False))
    ]
    hair_objects = [obj for obj in bpy.data.objects if obj.type == "CURVES"]
    controlled = [shell, *proxies, *hair_objects]
    object_states = {obj.name: obj.hide_render for obj in controlled}
    old_filepath = scene.render.filepath
    old_file_format = scene.render.image_settings.file_format
    old_use_compositing = getattr(scene.render, "use_compositing", None)
    old_camera = scene.camera
    shell_materials = list(shell.data.materials)
    stats = []

    try:
        shell.data.materials.clear()
        shell.data.materials.append(opaque_material)
        shell.hide_render = False
        for proxy in proxies:
            proxy.hide_render = True
        for hair in hair_objects:
            hair.hide_render = True
        scene.camera = camera
        for frame in range(args.frame_start, args.frame_end + 1):
            scene.frame_set(frame)
            stem = f"{VIEW_ID:08d}"
            render_tmp = temp_root / f"frame_{frame:06d}" / stem
            renderer.ensure_clean_dir(render_tmp, overwrite=True)
            depth, _depth_exr = renderer.render_depth_pass(scene, camera, render_tmp)

            frame_root = output_root / f"frame_{frame:06d}"
            depth_dir = frame_root / "rendered_depth_maps"
            cams_dir = frame_root / "cams"
            depth_dir.mkdir(parents=True, exist_ok=True)
            cams_dir.mkdir(parents=True, exist_ok=True)
            renderer.write_pfm(depth_dir / f"{stem}.pfm", depth)
            renderer.write_depth_preview(depth_dir / f"{stem}_preview.png", depth)
            renderer.write_cam_file(
                cams_dir / f"{stem}_cam.txt",
                renderer.camera_extrinsic_matrix(camera),
                renderer.camera_intrinsic_matrix(scene, camera),
                depth,
                128,
            )

            valid = depth[np.isfinite(depth) & (depth > 1e-8)]
            stats.append(
                {
                    "frame": frame,
                    "valid_pixel_fraction": float(valid.size / depth.size),
                    "depth_min_mm": float(valid.min()) if valid.size else None,
                    "depth_max_mm": float(valid.max()) if valid.size else None,
                }
            )
            print(
                f"top_full_shell_depth frame={frame} valid={valid.size}/{depth.size}",
                flush=True,
            )
    finally:
        shell.data.materials.clear()
        for material in shell_materials:
            shell.data.materials.append(material)
        for name, hide_render in object_states.items():
            obj = bpy.data.objects.get(name)
            if obj is not None:
                obj.hide_render = hide_render
        scene.render.filepath = old_filepath
        scene.render.image_settings.file_format = old_file_format
        if old_use_compositing is not None:
            scene.render.use_compositing = old_use_compositing
        scene.camera = old_camera
        safe_remove_temp(temp_root, output_root)

    summary = {
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "commit" if args.commit else "diagnostic",
        "frame_range": [args.frame_start, args.frame_end],
        "view_id": VIEW_ID,
        "camera": CAMERA_NAME,
        "resolution": [1280, 720],
        "depth_render_engine": scene.render.engine,
        "top_depth_wall_policy": "full_shell_retained",
        "depth_excludes_hair_curves": True,
        "pose_or_animation_edits": False,
        "stats": stats,
    }
    status_name = (
        "top_full_shell_depth_status.json"
        if not args.commit
        else f"top_full_shell_depth_status_{args.frame_start:06d}_{args.frame_end:06d}.json"
    )
    write_json_atomic(output_root / status_name, summary)
    print("TOP_FULL_SHELL_DEPTH_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
