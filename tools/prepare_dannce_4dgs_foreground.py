#!/usr/bin/env python3
"""Create a mouse-only 4D-GS dataset and visual-hull initialization."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.io import loadmat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("dataset/DANNCE-4DGS-Pilot-mm1-f0000-0079"),
    )
    parser.add_argument(
        "--masks",
        type=Path,
        default=Path("output/DANNCE-4DGS-Pilot-mm1-f0000-0079/sam2_masks_v1"),
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        default=Path("dataset/DANNCE/raw/markerless_mouse_1/label3d_dannce.mat"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/DANNCE-4DGS-Pilot-mm1-f0000-0079-ForegroundV1"),
    )
    parser.add_argument("--voxel-mm", type=float, default=2.0)
    parser.add_argument("--bounds-padding-mm", type=float, default=12.0)
    parser.add_argument("--minimum-views", type=int, default=4)
    parser.add_argument("--max-points", type=int, default=80000)
    parser.add_argument("--seed", type=int, default=20260907)
    return parser.parse_args()


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    vertices = np.empty(
        len(points),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["nx"] = vertices["ny"] = vertices["nz"] = 0
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)


def project(points: np.ndarray, camera: dict, focal: float, size: int):
    rotation = np.asarray(camera["rotation_world_to_camera"], dtype=np.float64)
    translation = np.asarray(
        camera["translation_world_to_camera_normalized"], dtype=np.float64
    )
    camera_xyz = points @ rotation.T + translation
    uv = np.empty((len(points), 2), dtype=np.float64)
    uv[:, 0] = focal * camera_xyz[:, 0] / camera_xyz[:, 2] + size * 0.5
    uv[:, 1] = focal * camera_xyz[:, 1] / camera_xyz[:, 2] + size * 0.5
    x = np.rint(uv[:, 0]).astype(np.int32)
    y = np.rint(uv[:, 1]).astype(np.int32)
    valid = (
        (camera_xyz[:, 2] > 0)
        & (x >= 0)
        & (x < size)
        & (y >= 0)
        & (y < size)
    )
    return x, y, valid


def silhouette_votes(
    points: np.ndarray,
    masks: list[np.ndarray],
    cameras: list[dict],
    focal: float,
    size: int,
) -> np.ndarray:
    votes = np.zeros(len(points), dtype=np.uint8)
    for mask, camera in zip(masks, cameras, strict=True):
        x, y, valid = project(points, camera, focal, size)
        inside = np.zeros(len(points), dtype=bool)
        inside[valid] = mask[y[valid], x[valid]] > 0
        votes += inside
    return votes


def sample_colors(
    points: np.ndarray,
    frame_images: list[list[np.ndarray]],
    frame_masks: list[list[np.ndarray]],
    cameras: list[dict],
    focal: float,
    size: int,
) -> np.ndarray:
    color_sum = np.zeros((len(points), 3), dtype=np.float64)
    color_count = np.zeros(len(points), dtype=np.int32)
    for images, masks in zip(frame_images, frame_masks, strict=True):
        for image, mask, camera in zip(images, masks, cameras, strict=True):
            x, y, valid = project(points, camera, focal, size)
            selector = np.zeros(len(points), dtype=bool)
            selector[valid] = mask[y[valid], x[valid]] > 0
            color_sum[selector] += image[y[selector], x[selector], ::-1]
            color_count[selector] += 1
    colors = np.full((len(points), 3), 90.0, dtype=np.float64)
    valid = color_count > 0
    colors[valid] = color_sum[valid] / color_count[valid, None]
    return np.uint8(np.clip(colors, 0, 255))


def write_point_preview(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    panels = []
    for first, second, title in ((0, 1, "XY top"), (0, 2, "XZ side"), (1, 2, "YZ side")):
        panel = np.full((512, 512, 3), 255, dtype=np.uint8)
        low = points[:, [first, second]].min(axis=0)
        high = points[:, [first, second]].max(axis=0)
        span = np.maximum(high - low, 1e-6)
        xy = (points[:, [first, second]] - low) / span
        xy = np.rint(xy * 439 + 36).astype(np.int32)
        xy[:, 1] = 511 - xy[:, 1]
        for pixel, color in zip(xy, colors, strict=True):
            cv2.circle(panel, tuple(pixel), 1, tuple(int(v) for v in color[::-1]), -1)
        cv2.rectangle(panel, (0, 0), (150, 29), (15, 15, 15), -1)
        cv2.putText(panel, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(panel)
    cv2.imwrite(str(path), np.concatenate(panels, axis=1))


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    mask_source = args.masks.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / "pilot_manifest.json").read_text(encoding="utf-8"))
    mask_manifest = json.loads(
        (mask_source / "mask_manifest.json").read_text(encoding="utf-8")
    )
    frame_count = int(manifest["frame_count"])
    camera_count = int(manifest["camera_count"])
    size = int(manifest["resolution"][0])
    focal = float(manifest["virtual_focal_px"])
    center = np.asarray(manifest["world_center_mm"], dtype=np.float64)
    world_scale = float(manifest["world_scale_mm"])
    prompt_frames = [int(value) for value in mask_manifest["prompt_frames_zero_based"]]

    # Build the mouse-only image sequence. Hard foreground masks are deliberate:
    # they provide an unambiguous white-background target and a crisp alpha edge.
    for camera_number in range(1, camera_count + 1):
        camera_output = output / f"cam{camera_number:02d}"
        camera_output.mkdir(parents=True, exist_ok=True)
        dataset_mask_output = output / "masks" / f"cam{camera_number:02d}"
        dataset_mask_output.mkdir(parents=True, exist_ok=True)
        for frame in range(frame_count):
            image = cv2.imread(
                str(source / f"cam{camera_number:02d}" / f"frame_{frame + 1:05d}.jpg")
            )
            mask = cv2.imread(
                str(
                    mask_source
                    / "masks"
                    / f"cam{camera_number:02d}"
                    / f"frame_{frame + 1:05d}.png"
                ),
                cv2.IMREAD_GRAYSCALE,
            )
            foreground = np.full_like(image, 255)
            foreground[mask > 0] = image[mask > 0]
            cv2.imwrite(
                str(camera_output / f"frame_{frame + 1:05d}.jpg"),
                foreground,
                [cv2.IMWRITE_JPEG_QUALITY, 97],
            )
            cv2.imwrite(
                str(dataset_mask_output / f"frame_{frame + 1:05d}.png"), mask
            )

    shutil.copytree(source / "sparse_", output / "sparse_", dirs_exist_ok=True)
    shutil.copy2(source / "poses_bounds_multipleview.npy", output)

    mat = loadmat(args.annotation, squeeze_me=True, struct_as_record=False)
    label_entry = list(np.asarray(mat["labelData"], dtype=object).reshape(-1))[0]
    label_frames = np.asarray(label_entry.data_frame, dtype=np.int64)
    label_xyz = np.asarray(label_entry.data_3d, dtype=np.float64).reshape(-1, 22, 3)
    selected_xyz = []
    for frame in prompt_frames:
        index = int(np.flatnonzero(label_frames == frame)[0])
        points_world = label_xyz[index]
        valid = np.isfinite(points_world).all(axis=1) & (
            np.linalg.norm(points_world, axis=1) > 1e-5
        )
        selected_xyz.append((points_world[valid] - center) / world_scale)
    all_labels = np.concatenate(selected_xyz, axis=0)
    padding = args.bounds_padding_mm / world_scale
    lower = all_labels.min(axis=0) - padding
    upper = all_labels.max(axis=0) + padding
    step = args.voxel_mm / world_scale
    axes = [np.arange(lower[i], upper[i] + 0.5 * step, step) for i in range(3)]
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    candidates = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3).astype(np.float32)

    frame_masks: list[list[np.ndarray]] = []
    frame_images: list[list[np.ndarray]] = []
    occupancy = np.zeros(len(candidates), dtype=bool)
    per_frame_occupied = {}
    for frame in prompt_frames:
        masks = [
            cv2.imread(
                str(
                    mask_source
                    / "masks"
                    / f"cam{camera_number:02d}"
                    / f"frame_{frame + 1:05d}.png"
                ),
                cv2.IMREAD_GRAYSCALE,
            )
            for camera_number in range(1, camera_count + 1)
        ]
        images = [
            cv2.imread(
                str(source / f"cam{camera_number:02d}" / f"frame_{frame + 1:05d}.jpg")
            )
            for camera_number in range(1, camera_count + 1)
        ]
        votes = silhouette_votes(candidates, masks, manifest["cameras"], focal, size)
        occupied = votes >= args.minimum_views
        occupancy |= occupied
        per_frame_occupied[str(frame)] = int(np.count_nonzero(occupied))
        frame_masks.append(masks)
        frame_images.append(images)

    points = candidates[occupancy]
    rng = np.random.default_rng(args.seed)
    if len(points) > args.max_points:
        points = points[rng.choice(len(points), args.max_points, replace=False)]
    # Break up the voxel lattice while keeping samples within half a voxel.
    points = points + rng.uniform(-0.45 * step, 0.45 * step, points.shape).astype(np.float32)
    colors = sample_colors(
        points,
        frame_images,
        frame_masks,
        manifest["cameras"],
        focal,
        size,
    )
    write_ply(output / "points3D_multipleview.ply", points, colors)
    write_point_preview(output / "visual_hull_orthographic_preview.jpg", points, colors)

    foreground_manifest = {
        **manifest,
        "format": "dannce_calibrated_4dgs_mouse_foreground_v1",
        "source_dataset": str(source),
        "mask_source": str(mask_source),
        "mask_method": "SAM 2.1 small, boxes and positive points projected from DANNCE 3D labels",
        "background": "white",
        "depth_semantics": "mouse-only after foreground 4D-GS training",
        "visual_hull": {
            "prompt_frames_zero_based": prompt_frames,
            "voxel_mm": args.voxel_mm,
            "minimum_silhouette_views": args.minimum_views,
            "candidate_count": len(candidates),
            "occupied_per_frame": per_frame_occupied,
            "union_count_before_sampling": int(np.count_nonzero(occupancy)),
            "initial_point_count": len(points),
            "bounds_normalized": [lower.tolist(), upper.tolist()],
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "pilot_manifest.json").write_text(
        json.dumps(foreground_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(foreground_manifest["visual_hull"], indent=2))


if __name__ == "__main__":
    main()
