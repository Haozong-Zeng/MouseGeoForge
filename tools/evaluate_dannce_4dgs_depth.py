#!/usr/bin/env python3
"""Check rendered mouse depth against projected DANNCE 3D joints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--visualizations", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[27, 72])
    parser.add_argument("--window-radius", type=int, default=2)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    visualizations = args.visualizations.resolve()
    output = args.output or visualizations / "depth_keypoint_check.json"
    manifest = json.loads((dataset / "pilot_manifest.json").read_text(encoding="utf-8"))
    mat = loadmat(args.annotation, squeeze_me=True, struct_as_record=False)
    label_entry = list(np.asarray(mat["labelData"], dtype=object).reshape(-1))[0]
    label_frames = np.asarray(label_entry.data_frame, dtype=np.int64)
    label_xyz = np.asarray(label_entry.data_3d, dtype=np.float64).reshape(-1, 22, 3)
    center = np.asarray(manifest["world_center_mm"], dtype=np.float64)
    scale = float(manifest["world_scale_mm"])
    focal = float(manifest["virtual_focal_px"])
    size = int(manifest["resolution"][0])
    radius = args.window_radius
    records = []
    all_errors = []
    for frame in args.frames:
        points_world = label_xyz[int(np.flatnonzero(label_frames == frame)[0])]
        valid_joint = np.isfinite(points_world).all(axis=1) & (
            np.linalg.norm(points_world, axis=1) > 1e-5
        )
        points = (points_world - center) / scale
        for camera_index, camera in enumerate(manifest["cameras"], start=1):
            rotation = np.asarray(camera["rotation_world_to_camera"], dtype=np.float64)
            translation = np.asarray(
                camera["translation_world_to_camera_normalized"], dtype=np.float64
            )
            camera_xyz = points @ rotation.T + translation
            uv = np.stack(
                (
                    focal * camera_xyz[:, 0] / camera_xyz[:, 2] + size * 0.5,
                    focal * camera_xyz[:, 1] / camera_xyz[:, 2] + size * 0.5,
                ),
                axis=-1,
            )
            stem = f"frame{frame:04d}_cam{camera_index:02d}"
            depth = np.load(visualizations / "selected" / f"{stem}_depth.npy")
            alpha = (
                cv2.imread(
                    str(visualizations / "selected" / f"{stem}_alpha.png"),
                    cv2.IMREAD_GRAYSCALE,
                ).astype(np.float32)
                / 255.0
            )
            errors = []
            for joint in np.flatnonzero(valid_joint):
                x, y = np.rint(uv[joint]).astype(int)
                if not (radius <= x < size - radius and radius <= y < size - radius):
                    continue
                if alpha[y, x] <= 0.5:
                    continue
                depth_window = depth[
                    y - radius : y + radius + 1, x - radius : x + radius + 1
                ]
                alpha_window = alpha[
                    y - radius : y + radius + 1, x - radius : x + radius + 1
                ]
                values = depth_window[(alpha_window > 0.5) & (depth_window > 0)]
                if len(values):
                    errors.append((float(np.median(values)) - camera_xyz[joint, 2]) * scale)
            all_errors.extend(errors)
            records.append(
                {
                    "frame": frame,
                    "camera": camera_index,
                    "sample_count": len(errors),
                    "median_absolute_error_mm": float(np.median(np.abs(errors))),
                    "median_bias_mm": float(np.median(errors)),
                }
            )
    absolute = np.abs(all_errors)
    result = {
        "comparison": "rendered first-surface depth versus internal/possibly occluded DANNCE joints",
        "warning": "This is a sanity check, not a surface-depth ground-truth metric.",
        "frames": args.frames,
        "sample_count": len(all_errors),
        "median_absolute_error_mm": float(np.median(absolute)),
        "mean_absolute_error_mm": float(np.mean(absolute)),
        "p90_absolute_error_mm": float(np.percentile(absolute, 90)),
        "median_bias_mm": float(np.median(all_errors)),
        "per_frame_camera": records,
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
