#!/usr/bin/env python3
"""Render the black-fur Orbbec sequence with fixed intrinsics on CUDA."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import blender_water_reaching_rendering as render_helpers  # noqa: E402
import convert_cylinder_renders_to_blenderpymouse_core23 as core23  # noqa: E402


LEGACY_VERTEX_IDS = (46, 13, 5017, 446, 93, 131, 116, 148, 9522, 9619)


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--camera", default="CameraOrbbec")
    parser.add_argument("--mesh", default="MS_Mouse.001")
    parser.add_argument("--frame-start", type=int, default=1)
    parser.add_argument("--frame-end", type=int, default=273)
    parser.add_argument("--sensor-width", type=float, default=33.0)
    parser.add_argument("--lens", type=float, default=30.98)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--cuda-device-substring", default="RTX PRO 6000")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-fixed-blend", type=Path)
    parser.add_argument("--progress-every", type=int, default=1)
    return parser.parse_args(args_after_dash())


def camera_intrinsic(scene: bpy.types.Scene, camera_obj: bpy.types.Object) -> np.ndarray:
    return render_helpers.camera_intrinsic_matrix(scene, camera_obj)


def camera_extrinsic(camera_obj: bpy.types.Object) -> np.ndarray:
    return core23.camera_extrinsic(camera_obj)


def configure_cuda(scene: bpy.types.Scene, device_substring: str) -> list[str]:
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = "CUDA"
    preferences.get_devices()
    selected = []
    for device in preferences.devices:
        use = device.type == "CUDA" and device_substring.lower() in device.name.lower()
        device.use = use
        if use:
            selected.append(f"{device.type}:{device.name}")
    if not selected:
        available = [f"{device.type}:{device.name}" for device in preferences.devices]
        raise RuntimeError(f"Requested CUDA device {device_substring!r} not found; available={available}")
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.render.use_persistent_data = True
    return selected


def configure_fixed_camera(
    scene: bpy.types.Scene,
    camera_obj: bpy.types.Object,
    sensor_width: float,
    lens: float,
) -> None:
    camera_obj.data.animation_data_clear()
    camera_obj.data.type = "PERSP"
    camera_obj.data.sensor_fit = "HORIZONTAL"
    camera_obj.data.sensor_width = sensor_width
    camera_obj.data.lens = lens
    camera_obj.data.shift_x = 0.0
    camera_obj.data.shift_y = 0.0
    scene.camera = camera_obj
    scene.frame_end = max(scene.frame_end, 273)


def configure_compositor(scene: bpy.types.Scene, frame_dir: Path) -> None:
    render_helpers.setup_depth_output(scene, frame_dir)
    tree = render_helpers.get_compositor_tree(scene)
    render_layers = tree.nodes.get("MGF_RenderLayers")
    rgb_output = tree.nodes.get("MGF_RGBOutput")
    if rgb_output is None:
        rgb_output = tree.nodes.new(type="CompositorNodeOutputFile")
        rgb_output.name = "MGF_RGBOutput"
    rgb_output.mute = False
    if hasattr(rgb_output, "base_path"):
        rgb_output.base_path = str(frame_dir)
        rgb_output.file_slots[0].path = "normal_image"
        rgb_output.format.file_format = "PNG"
        rgb_output.format.color_mode = "RGBA"
        rgb_input = rgb_output.inputs[0]
    else:
        rgb_output.directory = str(frame_dir)
        rgb_output.file_name = "normal_image"
        rgb_output.file_output_items.clear()
        rgb_item = rgb_output.file_output_items.new(socket_type="RGBA", name="Image")
        rgb_item.override_node_format = True
        rgb_item.format.file_format = "PNG"
        rgb_item.format.color_mode = "RGBA"
        rgb_item.format.color_depth = "8"
        rgb_input = rgb_output.inputs["Image"]
    for node in tree.nodes:
        if getattr(node, "bl_idname", "") == "CompositorNodeOutputFile" and node.name not in {
            "MGF_DepthOutput",
            "MGF_RGBOutput",
        }:
            node.mute = True
    for link in list(tree.links):
        if link.to_node == rgb_output:
            tree.links.remove(link)
    tree.links.new(render_layers.outputs["Image"], rgb_input)
    if hasattr(scene.render, "use_compositing"):
        scene.render.use_compositing = True


def legacy_keypoints(
    scene: bpy.types.Scene,
    camera_obj: bpy.types.Object,
    mesh_obj: bpy.types.Object,
) -> np.ndarray:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = mesh_obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        coordinates = []
        for vertex_id in LEGACY_VERTEX_IDS:
            world = evaluated.matrix_world @ mesh.vertices[vertex_id].co
            ndc = world_to_camera_view(scene, camera_obj, world)
            coordinates.append((float(ndc.x), float(1.0 - ndc.y)))
        return np.asarray(coordinates, dtype=np.float64)
    finally:
        evaluated.to_mesh_clear()


def frame_complete(frame_dir: Path) -> bool:
    required = ("normal_image.png", "depth_image.exr", "depth_image.png", "keypoint_coords.npy", "camera.json")
    return all((frame_dir / name).is_file() and (frame_dir / name).stat().st_size > 0 for name in required)


def clear_partial_outputs(frame_dir: Path) -> None:
    for pattern in ("normal_image*", "depth_image*", "depth_*.exr", "keypoint_coords.npy", "camera.json"):
        for path in frame_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def render_frame(
    scene: bpy.types.Scene,
    camera_obj: bpy.types.Object,
    mesh_obj: bpy.types.Object,
    frame: int,
    frame_dir: Path,
) -> float:
    frame_start = time.perf_counter()
    scene.frame_set(frame)
    bpy.context.view_layer.update()
    frame_dir.mkdir(parents=True, exist_ok=True)
    clear_partial_outputs(frame_dir)

    keypoints = legacy_keypoints(scene, camera_obj, mesh_obj)
    np.save(frame_dir / "keypoint_coords.npy", keypoints)
    intrinsic = camera_intrinsic(scene, camera_obj)
    extrinsic = camera_extrinsic(camera_obj)
    (frame_dir / "camera.json").write_text(
        json.dumps(
            {
                "frame": frame,
                "camera": camera_obj.name,
                "lens_mm": float(camera_obj.data.lens),
                "sensor_width_mm": float(camera_obj.data.sensor_width),
                "sensor_height_mm": float(camera_obj.data.sensor_height),
                "sensor_fit": camera_obj.data.sensor_fit,
                "intrinsic": intrinsic.tolist(),
                "extrinsic": extrinsic.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    configure_compositor(scene, frame_dir)
    scene.render.filepath = str(frame_dir / "normal_image.png")
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    bpy.ops.render.render(write_still=False)

    rgb_candidates = sorted(frame_dir.glob("normal_image*.png"), key=lambda path: path.stat().st_mtime)
    if not rgb_candidates:
        rgb_exr_candidates = sorted(frame_dir.glob("normal_image*.exr"), key=lambda path: path.stat().st_mtime)
        if rgb_exr_candidates:
            rgb_exr = rgb_exr_candidates[-1]
            rgb_image = bpy.data.images.load(str(rgb_exr), check_existing=False)
            try:
                rgb_image.save_render(filepath=str(frame_dir / "normal_image.png"), scene=scene)
            finally:
                bpy.data.images.remove(rgb_image)
            rgb_exr.unlink()
            rgb_candidates = [frame_dir / "normal_image.png"]
    if not rgb_candidates:
        raise FileNotFoundError(f"No RGB PNG produced for frame {frame}")
    rgb_path = frame_dir / "normal_image.png"
    source_rgb_path = rgb_candidates[-1]
    if source_rgb_path != rgb_path:
        source_rgb_path.replace(rgb_path)
    for extra in frame_dir.glob("normal_image*.png"):
        if extra != rgb_path:
            extra.unlink()

    exr_candidates = sorted(frame_dir.glob("*.exr"), key=lambda path: path.stat().st_mtime)
    if not exr_candidates:
        raise FileNotFoundError(f"No depth EXR produced for frame {frame}")
    depth_path = frame_dir / "depth_image.exr"
    source_depth_path = exr_candidates[-1]
    if source_depth_path != depth_path:
        source_depth_path.replace(depth_path)
    for extra in frame_dir.glob("*.exr"):
        if extra != depth_path:
            extra.unlink()

    depth = render_helpers.sanitize_depth(render_helpers.read_exr(str(depth_path)), camera_obj)
    render_helpers.write_depth_preview(frame_dir / "depth_image.png", depth)
    return time.perf_counter() - frame_start


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    camera_obj = bpy.data.objects.get(args.camera)
    mesh_obj = bpy.data.objects.get(args.mesh)
    if camera_obj is None or camera_obj.type != "CAMERA":
        raise KeyError(args.camera)
    if mesh_obj is None or mesh_obj.type != "MESH":
        raise KeyError(args.mesh)

    configure_fixed_camera(scene, camera_obj, args.sensor_width, args.lens)
    devices = configure_cuda(scene, args.cuda_device_substring)
    if args.samples is not None:
        scene.cycles.samples = args.samples
    if args.save_fixed_blend:
        fixed_blend = args.save_fixed_blend.resolve()
        fixed_blend.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(fixed_blend), check_existing=False)

    width, height = render_helpers.get_render_size(scene)
    manifest = {
        "format": "mousegeoforge_openfield_orbbec_fixed_camera_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_blend": str(Path(bpy.data.filepath).resolve()),
        "frames": [args.frame_start, args.frame_end],
        "camera": camera_obj.name,
        "mesh": mesh_obj.name,
        "resolution": [width, height],
        "lens_mm": float(camera_obj.data.lens),
        "sensor_width_mm": float(camera_obj.data.sensor_width),
        "sensor_fit": camera_obj.data.sensor_fit,
        "intrinsic": camera_intrinsic(scene, camera_obj).tolist(),
        "extrinsic": camera_extrinsic(camera_obj).tolist(),
        "intrinsic_is_fixed": True,
        "camera_data_animation_removed": True,
        "render_engine": scene.render.engine,
        "cycles_device": scene.cycles.device,
        "cycles_samples": int(scene.cycles.samples),
        "cycles_use_denoising": bool(scene.cycles.use_denoising),
        "compute_devices": devices,
        "legacy_vertex_ids": list(LEGACY_VERTEX_IDS),
    }

    timings = []
    skipped = 0
    total_start = time.perf_counter()
    for frame in range(args.frame_start, args.frame_end + 1):
        frame_dir = output_root / f"{frame:012d}"
        if args.resume and frame_complete(frame_dir):
            skipped += 1
            continue
        elapsed = render_frame(scene, camera_obj, mesh_obj, frame, frame_dir)
        timings.append({"frame": frame, "seconds": elapsed})
        if frame == args.frame_start or frame == args.frame_end or frame % max(1, args.progress_every) == 0:
            print(f"fixed_camera_render frame={frame}/{args.frame_end} seconds={elapsed:.3f}", flush=True)

    total_elapsed = time.perf_counter() - total_start
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["rendered_frame_count"] = len(timings)
    manifest["skipped_frame_count"] = skipped
    manifest["total_elapsed_seconds"] = total_elapsed
    manifest["timings"] = timings
    if timings:
        values = np.asarray([item["seconds"] for item in timings])
        manifest["timing_summary_seconds"] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    (output_root / "render_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"fixed_camera_render_complete output={output_root} elapsed={total_elapsed:.3f}s", flush=True)


if __name__ == "__main__":
    main()
