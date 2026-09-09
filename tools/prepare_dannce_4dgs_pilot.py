#!/usr/bin/env python3
"""Prepare a calibrated, distortion-free DANNCE clip for official 4D-GS."""

from __future__ import annotations

import argparse
import json
import struct
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.io import loadmat
from scipy.spatial.transform import Rotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session-root",
        type=Path,
        default=Path("dataset/DANNCE/raw/markerless_mouse_1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/DANNCE-4DGS-Pilot-mm1-f0000-0079"),
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--focal", type=float, default=800.0)
    parser.add_argument("--world-scale-mm", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=20260907)
    return parser.parse_args()


def distortion_map(
    size: int,
    focal: float,
    virtual_rotation: np.ndarray,
    source_rotation: np.ndarray,
    source_intrinsic: np.ndarray,
    radial: np.ndarray,
    tangential: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_pixel, y_pixel = np.meshgrid(
        np.arange(size, dtype=np.float64), np.arange(size, dtype=np.float64)
    )
    virtual_rays = np.stack(
        (
            (x_pixel - size * 0.5) / focal,
            (y_pixel - size * 0.5) / focal,
            np.ones_like(x_pixel),
        ),
        axis=-1,
    )
    source_rays = virtual_rays @ virtual_rotation @ source_rotation.T
    x = source_rays[..., 0] / source_rays[..., 2]
    y = source_rays[..., 1] / source_rays[..., 2]
    radius2 = x * x + y * y
    k1, k2, k3 = radial
    p1, p2 = tangential
    scale = 1.0 + k1 * radius2 + k2 * radius2**2 + k3 * radius2**3
    xd = x * scale + 2.0 * p1 * x * y + p2 * (radius2 + 2.0 * x * x)
    yd = y * scale + p1 * (radius2 + 2.0 * y * y) + 2.0 * p2 * x * y
    map_x = (
        source_intrinsic[0, 0] * xd
        + source_intrinsic[0, 1] * yd
        + source_intrinsic[0, 2]
    )
    map_y = (
        source_intrinsic[1, 0] * xd
        + source_intrinsic[1, 1] * yd
        + source_intrinsic[1, 2]
    )
    return map_x.astype(np.float32), map_y.astype(np.float32)


def look_at_rotation(
    camera_center: np.ndarray,
    target: np.ndarray,
    source_rotation: np.ndarray,
) -> np.ndarray:
    z_axis = target - camera_center
    z_axis /= np.linalg.norm(z_axis)
    x_axis = source_rotation[0] - np.dot(source_rotation[0], z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    if np.dot(y_axis, source_rotation[1]) < 0:
        x_axis *= -1
        y_axis *= -1
    rotation = np.stack((x_axis, y_axis, z_axis))
    if np.linalg.det(rotation) < 0.999:
        raise ValueError("Invalid virtual camera rotation")
    return rotation


def write_colmap_cameras(path: Path, size: int, focal: float) -> None:
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<iiQQ", 1, 1, size, size))  # PINHOLE
        handle.write(struct.pack("<dddd", focal, focal, size * 0.5, size * 0.5))


def write_colmap_images(path: Path, cameras: list[dict]) -> None:
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(cameras)))
        for camera in cameras:
            quaternion_xyzw = Rotation.from_matrix(camera["rotation"]).as_quat()
            qx, qy, qz, qw = quaternion_xyzw
            values = (
                camera["view_id"] + 1,
                qw,
                qx,
                qy,
                qz,
                *camera["translation_normalized"],
                1,
            )
            handle.write(struct.pack("<idddddddi", *values))
            handle.write(f"frame{camera['view_id'] + 1:02d}.jpg".encode("utf-8"))
            handle.write(b"\x00")
            handle.write(struct.pack("<Q", 0))


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


def initial_points(
    label_xyz: np.ndarray,
    center_world: np.ndarray,
    world_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    finite_labels = label_xyz[np.isfinite(label_xyz).all(axis=-1)].reshape(-1, 3)
    mouse_points = np.repeat(finite_labels, 350, axis=0)
    mouse_points += rng.normal(0.0, 5.0, size=mouse_points.shape)

    arena_center = np.asarray([57.0, 58.0, 0.0])
    count_floor = 12_000
    angle = rng.uniform(0, 2 * np.pi, count_floor)
    radius = 82.0 * np.sqrt(rng.uniform(0, 1, count_floor))
    floor = np.stack(
        (
            arena_center[0] + radius * np.cos(angle),
            arena_center[1] + radius * np.sin(angle),
            rng.normal(0.0, 1.0, count_floor),
        ),
        axis=-1,
    )

    count_wall = 8_000
    wall_angle = rng.uniform(0, 2 * np.pi, count_wall)
    wall_radius = rng.normal(84.0, 1.5, count_wall)
    wall = np.stack(
        (
            arena_center[0] + wall_radius * np.cos(wall_angle),
            arena_center[1] + wall_radius * np.sin(wall_angle),
            rng.uniform(0, 145.0, count_wall),
        ),
        axis=-1,
    )
    world = np.concatenate((mouse_points, floor, wall), axis=0)
    return ((world - center_world) / world_scale).astype(np.float32)


def sample_point_colors(
    points: np.ndarray,
    cameras: list[dict],
    first_images: list[np.ndarray],
    size: int,
    focal: float,
) -> np.ndarray:
    color_sum = np.zeros((len(points), 3), dtype=np.float64)
    color_count = np.zeros(len(points), dtype=np.int32)
    for camera, image in zip(cameras, first_images, strict=True):
        camera_xyz = points @ camera["rotation"].T + camera["translation_normalized"]
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
        color_sum[valid] += image[y[valid], x[valid], ::-1]
        color_count[valid] += 1
    colors = np.full((len(points), 3), 127.0)
    valid = color_count > 0
    colors[valid] = color_sum[valid] / color_count[valid, None]
    return np.clip(colors, 0, 255).astype(np.uint8)


def main() -> None:
    args = parse_args()
    if args.frames < 2:
        raise ValueError("At least two frames are required")
    args.output.mkdir(parents=True, exist_ok=True)
    mat = loadmat(
        args.session_root / "label3d_dannce.mat",
        squeeze_me=True,
        struct_as_record=False,
    )
    parameters = list(np.asarray(mat["params"], dtype=object).reshape(-1))
    camera_names = [str(value) for value in np.asarray(mat["camnames"]).reshape(-1)]
    com = np.asarray(mat["com"].com3d, dtype=np.float64)
    stop_frame = args.start_frame + args.frames
    center_world = np.median(com[args.start_frame:stop_frame], axis=0)
    target_world = np.mean(com[args.start_frame:stop_frame], axis=0)

    cameras = []
    for view_id, (name, parameter) in enumerate(
        zip(camera_names, parameters, strict=True)
    ):
        source_rotation = np.asarray(parameter.r, dtype=np.float64).T
        source_translation = np.asarray(parameter.t, dtype=np.float64).reshape(3)
        camera_center = -source_rotation.T @ source_translation
        rotation = look_at_rotation(camera_center, target_world, source_rotation)
        translation_mm = -rotation @ camera_center
        translation_normalized = (
            rotation @ center_world + translation_mm
        ) / args.world_scale_mm
        intrinsic = np.asarray(parameter.K, dtype=np.float64).T
        map_x, map_y = distortion_map(
            args.size,
            args.focal,
            rotation,
            source_rotation,
            intrinsic,
            np.asarray(parameter.RDistort, dtype=np.float64),
            np.asarray(parameter.TDistort, dtype=np.float64),
        )
        cameras.append(
            {
                "view_id": view_id,
                "name": name,
                "rotation": rotation,
                "translation_normalized": translation_normalized,
                "camera_center_normalized": (
                    (camera_center - center_world) / args.world_scale_mm
                ),
                "map_x": map_x,
                "map_y": map_y,
            }
        )

    first_images = []
    for camera in cameras:
        output_camera = args.output / f"cam{camera['view_id'] + 1:02d}"
        output_camera.mkdir(parents=True, exist_ok=True)
        source_video = (
            args.session_root
            / "videos"
            / camera["name"]
            / f"{(args.start_frame // 3000) * 3000}.mp4"
        )
        segment_start = int(source_video.stem)
        capture = cv2.VideoCapture(str(source_video))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open {source_video}")
        try:
            for output_index, source_frame in enumerate(
                range(args.start_frame, stop_frame), start=1
            ):
                local_frame = source_frame - segment_start
                capture.set(cv2.CAP_PROP_POS_FRAMES, local_frame)
                ok, image = capture.read()
                if not ok:
                    raise RuntimeError(f"Could not read {source_video}:{local_frame}")
                virtual = cv2.remap(
                    image,
                    camera["map_x"],
                    camera["map_y"],
                    interpolation=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_CONSTANT,
                )
                if output_index == 1:
                    first_images.append(virtual.copy())
                path = output_camera / f"frame_{output_index:05d}.jpg"
                if not cv2.imwrite(
                    str(path), virtual, [cv2.IMWRITE_JPEG_QUALITY, 95]
                ):
                    raise RuntimeError(f"Could not write {path}")
        finally:
            capture.release()

    sparse = args.output / "sparse_"
    sparse.mkdir(exist_ok=True)
    write_colmap_cameras(sparse / "cameras.bin", args.size, args.focal)
    write_colmap_images(sparse / "images.bin", cameras)

    label_entries = list(np.asarray(mat["labelData"], dtype=object).reshape(-1))
    label_frames = np.asarray(label_entries[0].data_frame, dtype=np.int64)
    label_xyz = np.asarray(label_entries[0].data_3d, dtype=np.float64).reshape(-1, 22, 3)
    selected_labels = label_xyz[
        (label_frames >= args.start_frame) & (label_frames < stop_frame)
    ]
    rng = np.random.default_rng(args.seed)
    points = initial_points(selected_labels, center_world, args.world_scale_mm, rng)
    colors = sample_point_colors(points, cameras, first_images, args.size, args.focal)
    write_ply(args.output / "points3D_multipleview.ply", points, colors)

    poses_bounds = []
    for camera in cameras:
        c2w = np.eye(4)
        c2w[:3, :3] = camera["rotation"].T
        c2w[:3, 3] = camera["camera_center_normalized"]
        stored = np.column_stack(
            (-c2w[:3, 1], c2w[:3, 0], c2w[:3, 2], c2w[:3, 3])
        )
        stored = np.column_stack(
            (stored, np.asarray([args.size, args.size, args.focal]))
        )
        poses_bounds.append(np.concatenate((stored.reshape(-1), [0.1, 6.0])))
    np.save(args.output / "poses_bounds_multipleview.npy", np.stack(poses_bounds))

    manifest = {
        "format": "dannce_calibrated_4dgs_pilot_v1",
        "data_origin": "real_multiview_video",
        "source_session": str(args.session_root.resolve()),
        "source_frames_zero_based": [args.start_frame, stop_frame - 1],
        "frame_count": args.frames,
        "camera_count": len(cameras),
        "resolution": [args.size, args.size],
        "virtual_focal_px": args.focal,
        "world_center_mm": center_world.tolist(),
        "world_scale_mm": args.world_scale_mm,
        "initial_point_count": len(points),
        "contains_hand_labeled_frames": label_frames[
            (label_frames >= args.start_frame) & (label_frames < stop_frame)
        ].tolist(),
        "camera_model": "centered pinhole virtual views from exact DANNCE distortion model",
        "cameras": [
            {
                "view_id": camera["view_id"],
                "name": camera["name"],
                "rotation_world_to_camera": camera["rotation"].tolist(),
                "translation_world_to_camera_normalized": camera[
                    "translation_normalized"
                ].tolist(),
                "camera_center_normalized": camera[
                    "camera_center_normalized"
                ].tolist(),
            }
            for camera in cameras
        ],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output / "pilot_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
