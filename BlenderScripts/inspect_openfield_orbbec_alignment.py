#!/usr/bin/env python3
"""Compare legacy normalized top-view keypoints with Blender bone-head projection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import convert_cylinder_renders_to_blenderpymouse_core23 as core23  # noqa: E402


LEGACY_BONES = (
    "HeadEnd_M",
    "Head1_M",
    "Ear01_L",
    "Ear01_R",
    "Neck3_M",
    "Spine3_M",
    "Spine1_M",
    "Tail0_M",
    "Tail6_M",
    "Tail11_M",
)


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", default="1,92,183,273")
    parser.add_argument("--camera", default="CameraOrbbec")
    parser.add_argument("--armature", default="DeformationSystem.001")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    return parser.parse_args(args_after_dash())


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    output = Path(args.output).resolve()
    if args.frames.strip().lower() == "all":
        frames = sorted(int(path.name) for path in source_root.iterdir() if path.is_dir() and path.name.isdigit())
    else:
        frames = [int(value) for value in args.frames.split(",") if value.strip()]
    camera = bpy.data.objects[args.camera]
    armature = bpy.data.objects[args.armature]
    intrinsic = core23.camera_intrinsic(camera, args.width, args.height)
    current_z = float(camera.matrix_world.translation.z)
    candidates = np.unique(np.concatenate((np.linspace(current_z - 150, current_z + 150, 301), [current_z])))
    candidate_errors = []
    per_frame = []
    normalized_camera_points = []
    legacy_pixel_points = []

    for frame in frames:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        world = np.asarray(
            [armature.matrix_world @ armature.pose.bones[name].head for name in LEGACY_BONES],
            dtype=np.float64,
        )
        legacy = np.load(source_root / f"{frame:012d}" / "keypoint_coords.npy").astype(np.float64)
        inspect_depth = len(frames) <= 10 or frame in {frames[0], frames[len(frames) // 2], frames[-1]}
        source_depth = (
            core23.read_exr(source_root / f"{frame:012d}" / "depth_image.exr")
            if inspect_depth
            else None
        )
        legacy_px = legacy * np.asarray((args.width, args.height), dtype=np.float64)
        current_extrinsic = core23.camera_extrinsic(camera)
        camera_xyz = (current_extrinsic[:3, :3] @ world.T).T + current_extrinsic[:3, 3]
        frame_normalized = camera_xyz[:, :2] / camera_xyz[:, 2:3]
        normalized_camera_points.append(frame_normalized)
        legacy_pixel_points.append(legacy_px)
        frame_fit_x = np.column_stack((frame_normalized[:, 0], np.ones(len(frame_normalized))))
        frame_fit_y = np.column_stack((frame_normalized[:, 1], np.ones(len(frame_normalized))))
        frame_fx, frame_cx = np.linalg.lstsq(frame_fit_x, legacy_px[:, 0], rcond=None)[0]
        frame_fy, frame_cy = np.linalg.lstsq(frame_fit_y, legacy_px[:, 1], rcond=None)[0]
        frame_fitted_uv = np.column_stack(
            (frame_fx * frame_normalized[:, 0] + frame_cx, frame_fy * frame_normalized[:, 1] + frame_cy)
        )
        frame_fitted_errors = np.linalg.norm(frame_fitted_uv - legacy_px, axis=1)
        frame_errors = []
        for candidate_z in candidates:
            extrinsic = core23.camera_extrinsic(camera, float(candidate_z))
            _, uv = core23.project_points(world, extrinsic, intrinsic)
            errors = np.linalg.norm(uv - legacy_px, axis=1)
            frame_errors.append(errors)
        candidate_errors.append(np.stack(frame_errors))
        current_index = int(np.argmin(np.abs(candidates - current_z)))
        per_frame.append(
            {
                "frame": frame,
                "camera_lens_mm_at_frame": float(camera.data.lens),
                "camera_sensor_width_mm_at_frame": float(camera.data.sensor_width),
                "camera_sensor_height_mm_at_frame": float(camera.data.sensor_height),
                "camera_sensor_fit_at_frame": camera.data.sensor_fit,
                "fitted_intrinsic_at_frame": [
                    [float(frame_fx), 0.0, float(frame_cx)],
                    [0.0, float(frame_fy), float(frame_cy)],
                    [0.0, 0.0, 1.0],
                ],
                "fitted_intrinsic_mean_error_px": float(frame_fitted_errors.mean()),
                "fitted_intrinsic_max_error_px": float(frame_fitted_errors.max()),
                "legacy_points": legacy_px.tolist(),
                "current_z_mean_error_px": float(np.mean(frame_errors[current_index])),
                "current_z_max_error_px": float(np.max(frame_errors[current_index])),
                "depth_percentiles": (
                    np.percentile(source_depth[np.isfinite(source_depth) & (source_depth > 0)], [1, 5, 50, 95, 99]).tolist()
                    if source_depth is not None
                    else None
                ),
                "depth_floor_samples": (
                    {
                        "bottom_left": float(source_depth[int(args.height * 0.8), int(args.width * 0.2)]),
                        "bottom_right": float(source_depth[int(args.height * 0.8), int(args.width * 0.8)]),
                        "center": float(source_depth[args.height // 2, args.width // 2]),
                    }
                    if source_depth is not None
                    else None
                ),
            }
        )

    errors = np.stack(candidate_errors)
    mean_by_candidate = errors.mean(axis=(0, 2))
    best_index = int(np.argmin(mean_by_candidate))
    best_z = float(candidates[best_index])
    best_errors = errors[:, best_index, :]
    current_index = int(np.argmin(np.abs(candidates - current_z)))
    current_errors = errors[:, current_index, :]
    normalized = np.concatenate(normalized_camera_points)
    legacy_pixels = np.concatenate(legacy_pixel_points)
    fit_x = np.column_stack((normalized[:, 0], np.ones(len(normalized))))
    fit_y = np.column_stack((normalized[:, 1], np.ones(len(normalized))))
    fitted_fx, fitted_cx = np.linalg.lstsq(fit_x, legacy_pixels[:, 0], rcond=None)[0]
    fitted_fy, fitted_cy = np.linalg.lstsq(fit_y, legacy_pixels[:, 1], rcond=None)[0]
    fitted_uv = np.column_stack(
        (fitted_fx * normalized[:, 0] + fitted_cx, fitted_fy * normalized[:, 1] + fitted_cy)
    )
    fitted_errors = np.linalg.norm(fitted_uv - legacy_pixels, axis=1)
    horizontal_focal = float(camera.data.lens * args.width / camera.data.sensor_width)
    horizontal_uv = normalized * horizontal_focal + np.asarray((args.width / 2, args.height / 2))
    horizontal_errors = np.linalg.norm(horizontal_uv - legacy_pixels, axis=1)
    payload = {
        "blend": bpy.data.filepath,
        "source_root": str(source_root),
        "camera": camera.name,
        "camera_lens_mm": float(camera.data.lens),
        "camera_sensor_fit": camera.data.sensor_fit,
        "camera_sensor_width_mm": float(camera.data.sensor_width),
        "camera_sensor_height_mm": float(camera.data.sensor_height),
        "current_intrinsic": intrinsic.tolist(),
        "armature": armature.name,
        "frames": frames,
        "legacy_bones": list(LEGACY_BONES),
        "current_camera_world_z": current_z,
        "best_camera_world_z": best_z,
        "current_mean_error_px": float(current_errors.mean()),
        "current_median_error_px": float(np.median(current_errors)),
        "current_max_error_px": float(current_errors.max()),
        "best_mean_error_px": float(best_errors.mean()),
        "best_median_error_px": float(np.median(best_errors)),
        "best_max_error_px": float(best_errors.max()),
        "fitted_intrinsic_at_current_z": [
            [float(fitted_fx), 0.0, float(fitted_cx)],
            [0.0, float(fitted_fy), float(fitted_cy)],
            [0.0, 0.0, 1.0],
        ],
        "fitted_lens_mm_from_fx": float(fitted_fx * camera.data.sensor_width / args.width),
        "fitted_lens_mm_from_fy": float(fitted_fy * camera.data.sensor_width / args.height),
        "fitted_intrinsic_mean_error_px": float(fitted_errors.mean()),
        "fitted_intrinsic_median_error_px": float(np.median(fitted_errors)),
        "fitted_intrinsic_max_error_px": float(fitted_errors.max()),
        "physical_horizontal_intrinsic": [
            [horizontal_focal, 0.0, args.width / 2],
            [0.0, horizontal_focal, args.height / 2],
            [0.0, 0.0, 1.0],
        ],
        "physical_horizontal_mean_error_px": float(horizontal_errors.mean()),
        "physical_horizontal_median_error_px": float(np.median(horizontal_errors)),
        "physical_horizontal_max_error_px": float(horizontal_errors.max()),
        "per_frame": per_frame,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
