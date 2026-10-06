#!/usr/bin/env python3
"""Small-data VGGT-Omega fine-tuning entrypoint for BlendedMVS-style folders.

RGB images are required model inputs. Camera and depth labels are independently
optional per view, so camera-only, depth-only, and mixed-supervision scenes can
share one training root:

    scene/
      blended_images/00000000.jpg
      rendered_depth_maps/00000000.pfm  # optional
      cams/00000000_cam.txt             # optional
      cams/pair.txt                     # optional

Camera files are treated as OpenCV camera-from-world matrices, which matches
the BlendedMVS convention verified by the local depth reprojection checks.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import math
import random
import sys
import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.pose_enc import encoding_to_camera, extri_intri_to_pose_encoding
from training.geometry_coupled_losses import (
    multiview_keypoint_geometry_loss,
    official_depth_loss,
    official_point_loss,
    relative_rig_camera_loss,
    relative_rig_frame_errors,
    shared_rig_temporal_loss,
    token_matching_loss,
)
from training.keypoint_head_losses import keypoint_head_loss
from training.instance_pose_losses import (
    InstancePoseLossWeights,
    instance_targets_from_batch,
    instance_pose_loss,
)
from training.tail_curve_losses import TailCurveLossConfig


REAL_PALM_KEYPOINTS = ("r_mid_A", "l_mid_A")
REAL_SCALE_ANCHORS = ("point1", "point2", "point3", "spout_tip_A")
REAL_SPARSE_KEYPOINTS = (*REAL_PALM_KEYPOINTS, *REAL_SCALE_ANCHORS)
REAL_MOUSE_KEYPOINTS = (
    "r1_tip_A",
    "r1_mid_A",
    "r1_joint_A",
    "r2_tip_A",
    "r2_mid_A",
    "r2_joint_A",
    "r3_tip_A",
    "r3_mid_A",
    "r3_joint_A",
    "r4_tip_A",
    "r4_mid_A",
    "r4_joint_A",
    "r_mid_A",
    "l1_tip_A",
    "l1_mid_A",
    "l1_joint_A",
    "l2_tip_A",
    "l2_mid_A",
    "l2_joint_A",
    "l3_tip_A",
    "l3_mid_A",
    "l3_joint_A",
    "l4_tip_A",
    "l4_mid_A",
    "l4_joint_A",
    "l_mid_A",
    "snout_A",
    "mouth_tip_A",
)
REAL_GEOMETRY_KEYPOINTS = (*REAL_MOUSE_KEYPOINTS, *REAL_SCALE_ANCHORS)
REAL_TO_BLENDER_KEYPOINT = {
    **{
        f"{side_short}{digit_index}_{part}_A": (
            f"{digit_name}Finger{part_index}_{side_long}"
        )
        for side_short, side_long in (("r", "R"), ("l", "L"))
        for digit_index, digit_name in enumerate(
            ("Pinky", "Ring", "Middle", "Index"),
            start=1,
        )
        for part, part_index in (("joint", 1), ("mid", 2), ("tip", 3))
    },
    "r_mid_A": "Fingers1_R",
    "l_mid_A": "Fingers1_L",
    "snout_A": "HeadEnd_M",
}


@dataclass(frozen=True)
class BlendedFrame:
    scene: str
    view_id: int
    image_path: Path
    masked_image_path: Path | None
    depth_path: Path | None
    cam_path: Path | None
    body_mask_path: Path | None
    semantic_mask_path: Path | None
    extrinsic: np.ndarray | None
    intrinsic: np.ndarray | None


@dataclass(frozen=True)
class BlendedScene:
    name: str
    root: Path
    frames: dict[int, BlendedFrame]
    pair_map: dict[int, list[int]]


@dataclass(frozen=True)
class RealPoseTrial:
    name: str
    root: Path
    videos: dict[str, Path]
    extrinsics: dict[str, np.ndarray]
    intrinsics: dict[str, np.ndarray]
    frame_count: int


@dataclass(frozen=True)
class RealPalmCamera:
    name: str
    matrix: np.ndarray
    distortions: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray


@dataclass(frozen=True)
class RealPalmSample:
    trial: str
    video_path: Path
    camera: str
    frame: int
    image_width: int
    image_height: int
    xy: np.ndarray
    target_depth: np.ndarray
    valid: np.ndarray
    is_palm: np.ndarray
    point_weights: np.ndarray
    scale_mm: float


def read_pfm(path: Path, flip_y: bool = True) -> np.ndarray:
    with path.open("rb") as f:
        header = f.readline().decode("ascii").rstrip()
        if header not in {"PF", "Pf"}:
            raise ValueError(f"Not a PFM file: {path}")
        dims_line = f.readline().decode("ascii").strip()
        while dims_line.startswith("#"):
            dims_line = f.readline().decode("ascii").strip()
        width, height = map(int, dims_line.split())
        scale = float(f.readline().decode("ascii").strip())
        endian = "<" if scale < 0 else ">"
        channels = 3 if header == "PF" else 1
        data = np.fromfile(f, endian + "f", width * height * channels)
    shape = (height, width, channels) if channels == 3 else (height, width)
    arr = np.reshape(data, shape)
    if flip_y:
        arr = np.flipud(arr)
    if channels == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.float32, copy=False)


def read_cam(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    ext_start = lines.index("extrinsic") + 1
    extrinsic = np.array([[float(v) for v in lines[ext_start + i].split()] for i in range(4)], dtype=np.float32)
    int_start = lines.index("intrinsic") + 1
    intrinsic = np.array([[float(v) for v in lines[int_start + i].split()] for i in range(3)], dtype=np.float32)
    return extrinsic, intrinsic


def read_real_calibration(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    extrinsics: dict[str, np.ndarray] = {}
    intrinsics: dict[str, np.ndarray] = {}
    idx = 0
    while f"cam_{idx}" in data:
        item = data[f"cam_{idx}"]
        name = str(item["name"])
        rotation_vec = np.asarray(item["rotation"], dtype=np.float32)
        rotation, _ = cv2.Rodrigues(rotation_vec)
        translation = np.asarray(item["translation"], dtype=np.float32)
        extrinsic = np.eye(4, dtype=np.float32)
        extrinsic[:3, :3] = rotation.astype(np.float32)
        extrinsic[:3, 3] = translation
        extrinsics[name] = extrinsic
        intrinsics[name] = np.asarray(item["matrix"], dtype=np.float32)
        idx += 1
    if not extrinsics:
        raise ValueError(f"No cam_* entries found in {path}")
    return extrinsics, intrinsics


def video_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def read_video_rgb(path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, bgr = cap.read()
        if not ok or bgr is None:
            raise RuntimeError(f"Could not read frame {frame_idx} from {path}")
    finally:
        cap.release()
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_pair_file(path: Path) -> dict[int, list[int]]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        return {}
    pair_map: dict[int, list[int]] = {}
    count = int(lines[0])
    idx = 1
    for _ in range(count):
        ref = int(lines[idx])
        idx += 1
        parts = lines[idx].split()
        idx += 1
        src_count = int(parts[0])
        pair_map[ref] = [int(parts[1 + 2 * i]) for i in range(src_count)]
    return pair_map


def parse_name_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def find_optional_view_file(scene_root: Path, dir_names: list[str], view_id: int) -> Path | None:
    stems = [f"{view_id:08d}", str(view_id)]
    suffixes = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
    for dir_name in dir_names:
        directory = scene_root / dir_name
        if not directory.exists():
            continue
        for stem in stems:
            for suffix in suffixes:
                path = directory / f"{stem}{suffix}"
                if path.exists():
                    return path
    return None


def discover_scene_roots(root: Path) -> list[Path]:
    if (root / "blended_images").exists():
        return [root]
    scenes = [path for path in sorted(root.iterdir()) if (path / "blended_images").exists()]
    if scenes:
        return scenes
    scenes = sorted(images_dir.parent for images_dir in root.rglob("blended_images"))
    if not scenes:
        raise ValueError(f"No scenes with blended_images found under {root}")
    return scenes


def find_view_file(directory: Path, view_id: int, suffixes: tuple[str, ...]) -> Path | None:
    stems = (f"{view_id:08d}", str(view_id))
    for stem in stems:
        for suffix in suffixes:
            path = directory / f"{stem}{suffix}"
            if path.exists():
                return path
    return None


def frame_has_camera(frame: BlendedFrame) -> bool:
    return frame.extrinsic is not None and frame.intrinsic is not None


def frame_has_supervision(frame: BlendedFrame) -> bool:
    return frame_has_camera(frame) or frame.depth_path is not None


def load_scene(scene_root: Path) -> BlendedScene:
    frames: dict[int, BlendedFrame] = {}
    body_mask_dirs = parse_name_list(getattr(load_scene, "body_mask_dirs", "body_masks,mouse_masks,masks"))
    semantic_mask_dirs = parse_name_list(getattr(load_scene, "semantic_mask_dirs", "semantic_masks,part_masks"))
    image_dir = scene_root / "blended_images"
    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and not path.stem.endswith("_masked")
    )
    for image_path in image_paths:
        try:
            view_id = int(image_path.stem)
        except ValueError:
            continue

        masked_image_path = find_view_file(
            image_dir,
            view_id,
            ("_masked.jpg", "_masked.jpeg", "_masked.png"),
        )
        depth_path = find_view_file(scene_root / "rendered_depth_maps", view_id, (".pfm",))
        cam_path = find_view_file(scene_root / "cams", view_id, ("_cam.txt",))
        extrinsic = None
        intrinsic = None
        if cam_path is not None:
            try:
                extrinsic, intrinsic = read_cam(cam_path)
            except (OSError, ValueError) as exc:
                print(f"warning: invalid camera label ignored: {cam_path} ({exc})")
                cam_path = None

        frames[view_id] = BlendedFrame(
            scene=scene_root.name,
            view_id=view_id,
            image_path=image_path,
            masked_image_path=masked_image_path,
            depth_path=depth_path,
            cam_path=cam_path,
            body_mask_path=find_optional_view_file(scene_root, body_mask_dirs, view_id),
            semantic_mask_path=find_optional_view_file(scene_root, semantic_mask_dirs, view_id),
            extrinsic=extrinsic,
            intrinsic=intrinsic,
        )
    pair_path = scene_root / "cams" / "pair.txt"
    return BlendedScene(
        name=scene_root.name,
        root=scene_root,
        frames=frames,
        pair_map=read_pair_file(pair_path) if pair_path.exists() else {},
    )


def resize_depth_nearest(depth: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    resized = Image.fromarray(depth).resize(output_size, Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.float32).copy()


def read_binary_mask(path: Path | None, output_size: tuple[int, int]) -> np.ndarray:
    width, height = output_size
    if path is None or not path.exists():
        return np.zeros((height, width), dtype=bool)
    with Image.open(path) as img:
        img = img.convert("L").resize(output_size, Image.Resampling.NEAREST)
        arr = np.asarray(img, dtype=np.uint8)
    return arr > 0


def read_semantic_mask(path: Path | None, output_size: tuple[int, int]) -> np.ndarray:
    width, height = output_size
    if path is None or not path.exists():
        return np.zeros((height, width), dtype=np.int64)
    with Image.open(path) as img:
        img = img.convert("L").resize(output_size, Image.Resampling.NEAREST)
        arr = np.asarray(img, dtype=np.uint8)
    return arr.astype(np.int64, copy=False)


def canonicalize_extrinsics(extrinsics: np.ndarray, enabled: bool) -> np.ndarray:
    """Put the first selected camera at identity to remove arbitrary scene gauge."""
    return canonicalize_available_extrinsics(
        extrinsics,
        np.ones(len(extrinsics), dtype=bool),
        enabled,
    )


def canonicalize_available_extrinsics(
    extrinsics: np.ndarray,
    available: np.ndarray,
    enabled: bool,
) -> np.ndarray:
    """Canonicalize only labeled cameras; unlabeled placeholder entries stay identity."""
    out = extrinsics[:, :3, :4].astype(np.float32, copy=True)
    valid_indices = np.flatnonzero(available)
    if not enabled or valid_indices.size == 0:
        return out

    ref_idx = int(valid_indices[0])
    ref = np.eye(4, dtype=np.float32)
    ref[:3, :4] = extrinsics[ref_idx, :3, :4]
    world_from_ref = np.linalg.inv(ref).astype(np.float32)
    for idx in valid_indices:
        extrinsic = extrinsics[idx]
        e4 = np.eye(4, dtype=np.float32)
        e4[:3, :4] = extrinsic[:3, :4]
        out[idx] = (e4 @ world_from_ref)[:3, :4]
    return out


def choose_view_ids(
    frames: dict[int, BlendedFrame],
    pair_map: dict[int, list[int]],
    num_views: int,
    rng: random.Random,
    top_srcs: int,
) -> list[int]:
    valid_ids = sorted(frames)
    refs = [
        ref
        for ref, srcs in pair_map.items()
        if ref in frames and len([src for src in srcs if src in frames and src != ref]) >= num_views - 1
    ]
    if refs:
        ref = rng.choice(refs)
        srcs = [src for src in pair_map[ref] if src in frames and src != ref]
        src_pool = srcs[: max(num_views - 1, top_srcs)]
        chosen_srcs = rng.sample(src_pool, num_views - 1) if len(src_pool) > num_views - 1 else src_pool[: num_views - 1]
        chosen = [ref] + chosen_srcs
    else:
        if len(valid_ids) < num_views:
            raise ValueError(f"Scene has only {len(valid_ids)} valid views, need {num_views}")
        chosen = rng.sample(valid_ids, num_views)

    camera_ids = [view_id for view_id in chosen if frame_has_camera(frames[view_id])]
    if not camera_ids:
        scene_camera_ids = [view_id for view_id in valid_ids if frame_has_camera(frames[view_id])]
        if scene_camera_ids:
            replacement = rng.choice(scene_camera_ids)
            chosen[-1] = replacement
            chosen = list(dict.fromkeys(chosen))
            for candidate in valid_ids:
                if len(chosen) >= num_views:
                    break
                if candidate not in chosen:
                    chosen.append(candidate)
            camera_ids = [replacement]
    if camera_ids:
        ref_camera = camera_ids[0]
        chosen = [ref_camera] + [view_id for view_id in chosen if view_id != ref_camera]
    if not any(frame_has_supervision(frames[view_id]) for view_id in chosen):
        supervised_ids = [view_id for view_id in valid_ids if frame_has_supervision(frames[view_id])]
        if supervised_ids:
            chosen[-1] = rng.choice(supervised_ids)
    return chosen


def discover_keypoint_names(scenes: list[BlendedScene]) -> list[str]:
    """Build a deterministic union for mixed partial-label keypoint schemas."""
    sequences: list[list[str]] = []
    npz_datasets: set[Path] = set()
    for scene in scenes:
        dataset_root = scene.root.parent
        if dataset_root in npz_datasets:
            continue
        path = scene.root / "keypoints.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as arrays:
            names = [str(value) for value in arrays["names"]]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate keypoint names in {path}")
        sequences.append(names)
        npz_datasets.add(dataset_root)

    # Old pseudo-synthetic exports predate keypoints.npz but retain valid 2D
    # labels in each per-view annotation JSON. Include one schema per dataset.
    annotation_datasets: set[Path] = set()
    for scene in scenes:
        dataset_root = scene.root.parent
        if dataset_root in npz_datasets or dataset_root in annotation_datasets:
            continue
        for view_id in sorted(scene.frames):
            path = scene.root / "annotations" / f"{view_id:08d}.json"
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            keypoints = payload.get("keypoints")
            if not isinstance(keypoints, list) or not keypoints:
                continue
            names = [str(item["name"]) for item in keypoints if "name" in item]
            if names and len(names) == len(set(names)):
                sequences.append(names)
                annotation_datasets.add(dataset_root)
                break

    union: list[str] = []
    seen: set[str] = set()
    for names in sequences:
        for name in names:
            if name not in seen:
                union.append(name)
                seen.add(name)
    return union


def scene_instance_count(scene: BlendedScene) -> int:
    """Return the number of labeled instances without assuming an NPZ version."""
    path = scene.root / "keypoints.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as arrays:
            if "instance_ids" in arrays:
                return int(len(arrays["instance_ids"]))
            joint_world = arrays["joint_world_xyz"]
            return int(joint_world.shape[0]) if joint_world.ndim == 3 else 1
    if any(frame.body_mask_path is not None for frame in scene.frames.values()):
        return 1
    for view_id in sorted(scene.frames):
        annotation = scene.root / "annotations" / f"{view_id:08d}.json"
        if not annotation.is_file():
            continue
        try:
            payload = json.loads(annotation.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload.get("keypoints"), list) and payload["keypoints"]:
            return 1
    return 0


def discover_max_instances(scenes: list[BlendedScene]) -> int:
    """Find a fixed padding width so M=1 and M>1 scenes collate together."""
    declared_counts: dict[Path, int | None] = {}
    maximum = 0
    for scene in scenes:
        dataset_root = scene.root.parent
        if dataset_root not in declared_counts:
            metadata_path = dataset_root / "metadata.json"
            declared_count: int | None = None
            if metadata_path.is_file():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    value = int(metadata.get("instance_count", 0) or 0)
                    declared_count = value if value > 0 else None
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    declared_count = None
            declared_counts[dataset_root] = declared_count
        count = declared_counts[dataset_root]
        if count is None:
            count = scene_instance_count(scene)
        maximum = max(maximum, count)
    return max(1, maximum)


def transform_world_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    return (transform[:3, :3] @ points.T).T + transform[:3, 3]


def load_selected_keypoints(
    scene: BlendedScene,
    canonical_names: list[str],
    view_ids: list[int],
    original_sizes: list[tuple[int, int]],
    output_size: tuple[int, int],
    original_extrinsics: np.ndarray,
    camera_available: np.ndarray,
    canonicalize_camera: bool,
    scale: float,
) -> dict[str, np.ndarray]:
    keypoint_count = len(canonical_names)
    view_count = len(view_ids)
    empty = {
        "available": np.asarray(False),
        "valid": np.zeros(keypoint_count, dtype=bool),
        "joint_world_xyz": np.zeros((keypoint_count, 3), dtype=np.float32),
        "surface_world_xyz": np.zeros((keypoint_count, 3), dtype=np.float32),
        "uv": np.zeros((view_count, keypoint_count, 2), dtype=np.float32),
        "in_frame": np.zeros((view_count, keypoint_count), dtype=bool),
        "surface_visible": np.zeros((view_count, keypoint_count), dtype=bool),
    }
    path = scene.root / "keypoints.npz"
    if not canonical_names or not path.is_file():
        return empty

    with np.load(path, allow_pickle=False) as arrays:
        source_names = [str(value) for value in arrays["names"]]
        source_view_ids = [int(value) for value in arrays["view_ids"]]
        source_joint_world = arrays["joint_world_xyz"].astype(np.float32)
        source_surface_world = arrays["surface_world_xyz"].astype(np.float32)
        source_uv = arrays["uv"].astype(np.float32)
        source_in_frame = arrays["in_frame"].astype(bool)
        source_visible = arrays["surface_visible"].astype(bool)

    source_name_index = {name: index for index, name in enumerate(source_names)}
    source_view_index = {view_id: index for index, view_id in enumerate(source_view_ids)}
    valid = np.asarray([name in source_name_index for name in canonical_names], dtype=bool)
    if not valid.any():
        return empty

    name_indices = np.asarray(
        [source_name_index.get(name, 0) for name in canonical_names],
        dtype=np.int64,
    )
    joint_world = source_joint_world[name_indices].copy()
    surface_world = source_surface_world[name_indices].copy()
    joint_world[~valid] = 0.0
    surface_world[~valid] = 0.0

    if canonicalize_camera:
        valid_cameras = np.flatnonzero(camera_available)
        if valid_cameras.size:
            reference_transform = original_extrinsics[int(valid_cameras[0])]
            joint_world = transform_world_points(joint_world, reference_transform)
            surface_world = transform_world_points(surface_world, reference_transform)
    if scale != 1.0:
        joint_world /= scale
        surface_world /= scale

    output_width, output_height = output_size
    uv = np.zeros((view_count, keypoint_count, 2), dtype=np.float32)
    in_frame = np.zeros((view_count, keypoint_count), dtype=bool)
    visible = np.zeros((view_count, keypoint_count), dtype=bool)
    for selected_index, (view_id, original_size) in enumerate(zip(view_ids, original_sizes)):
        source_index = source_view_index.get(view_id)
        if source_index is None:
            continue
        original_width, original_height = original_size
        uv[selected_index] = source_uv[source_index, name_indices]
        uv[selected_index, :, 0] *= output_width / float(original_width)
        uv[selected_index, :, 1] *= output_height / float(original_height)
        in_frame[selected_index] = source_in_frame[source_index, name_indices] & valid
        visible[selected_index] = source_visible[source_index, name_indices] & valid

    return {
        "available": np.asarray(True),
        "valid": valid,
        "joint_world_xyz": joint_world,
        "surface_world_xyz": surface_world,
        "uv": uv,
        "in_frame": in_frame,
        "surface_visible": visible,
    }


def _normalized_box_from_mask(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    y, x = np.nonzero(mask)
    if not x.size:
        return np.zeros(4, dtype=np.float32)
    x0 = float(x.min()) / width
    y0 = float(y.min()) / height
    x1 = float(x.max() + 1) / width
    y1 = float(y.max() + 1) / height
    return np.asarray(
        [(x0 + x1) * 0.5, (y0 + y1) * 0.5, x1 - x0, y1 - y0],
        dtype=np.float32,
    )


def _weak_box_from_keypoints(points: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if not valid.any():
        return np.zeros(4, dtype=np.float32)
    selected = points[valid]
    lower = np.maximum(selected.min(axis=0) - 0.05, 0.0)
    upper = np.minimum(selected.max(axis=0) + 0.05, 1.0)
    size = np.maximum(upper - lower, 1e-4)
    center = 0.5 * (lower + upper)
    return np.asarray([center[0], center[1], size[0], size[1]], dtype=np.float32)


def load_selected_instances(
    scene: BlendedScene,
    canonical_names: list[str],
    view_ids: list[int],
    original_sizes: list[tuple[int, int]],
    output_size: tuple[int, int],
    original_extrinsics: np.ndarray,
    camera_available: np.ndarray,
    canonicalize_camera: bool,
    scale: float,
    max_instances: int,
) -> dict[str, np.ndarray]:
    """Load legacy M=1 and native M>1 NPZ files into one padded contract."""
    keypoint_count = len(canonical_names)
    view_count = len(view_ids)
    output_width, output_height = output_size
    result = {
        "available": np.asarray(False),
        "legacy_available": np.asarray(False),
        "actual_instance_count": np.asarray(0, dtype=np.int64),
        "instance_valid": np.zeros((view_count, max_instances), dtype=bool),
        "instance_ids": np.full((view_count, max_instances), -1, dtype=np.int64),
        "boxes_cxcywh": np.zeros((view_count, max_instances, 4), dtype=np.float32),
        "box_valid": np.zeros((view_count, max_instances), dtype=bool),
        "masks": np.zeros(
            (view_count, max_instances, output_height, output_width), dtype=bool
        ),
        "mask_valid": np.zeros((view_count, max_instances), dtype=bool),
        "keypoint_xy": np.zeros(
            (view_count, max_instances, keypoint_count, 2), dtype=np.float32
        ),
        "keypoint_valid": np.zeros(
            (view_count, max_instances, keypoint_count), dtype=bool
        ),
        "keypoint_in_frame": np.zeros((view_count, max_instances, keypoint_count), dtype=bool),
        "keypoint_label_valid": np.zeros((view_count, max_instances, keypoint_count), dtype=bool),
        "keypoint_occlusion_cause": np.full((view_count, max_instances, keypoint_count), -1, dtype=np.int8),
        "keypoint_visible": np.zeros(
            (view_count, max_instances, keypoint_count), dtype=bool
        ),
        "keypoint_visibility_valid": np.zeros(
            (view_count, max_instances, keypoint_count), dtype=bool
        ),
        "keypoint_world_xyz": np.zeros(
            (max_instances, keypoint_count, 3), dtype=np.float32
        ),
        "surface_world_xyz": np.zeros(
            (max_instances, keypoint_count, 3), dtype=np.float32
        ),
        "name_valid": np.zeros((max_instances, keypoint_count), dtype=bool),
        "world_valid": np.zeros((max_instances, keypoint_count), dtype=bool),
        "surface_valid": np.zeros((max_instances, keypoint_count), dtype=bool),
    }
    path = scene.root / "keypoints.npz"
    source_view_ids: list[int] = []
    source_instance_count = 0
    valid_names = np.zeros(keypoint_count, dtype=bool)
    source_uv = source_in_frame = source_visible = None
    source_boxes = source_box_valid = None
    if canonical_names and path.is_file():
        with np.load(path, allow_pickle=False) as arrays:
            from training.keypoint_contract import read_validity, read_occlusion
            availability = read_validity(arrays, path)
            source_occlusion_cause = read_occlusion(arrays, path)
            source_names = [str(value) for value in arrays["names"]]
            source_view_ids = [int(value) for value in arrays["view_ids"]]
            source_joint_world = arrays["joint_world_xyz"].astype(np.float32)
            source_surface_world = arrays["surface_world_xyz"].astype(np.float32)
            source_uv = arrays["uv"].astype(np.float32)
            source_in_frame = arrays["in_frame"].astype(bool)
            source_visible = arrays["surface_visible"].astype(bool)
            source_world_valid = availability['world_valid']
            source_visibility_valid = availability['visibility_valid']
            source_label_valid = availability['xy_label_valid']
            source_surface_valid = availability['surface_valid']
            source_ids = (
                arrays["instance_ids"].astype(np.int64)
                if "instance_ids" in arrays
                else None
            )
            if "boxes_cxcywh" in arrays and "box_valid" in arrays:
                source_boxes = arrays["boxes_cxcywh"].astype(np.float32)
                source_box_valid = arrays["box_valid"].astype(bool)
                if source_boxes.ndim == 2:
                    source_boxes = source_boxes[:, None]
                    source_box_valid = source_box_valid[:, None]
        if source_joint_world.ndim == 2:
            source_joint_world = source_joint_world[None]
            source_surface_world = source_surface_world[None]
            source_uv = source_uv[:, None]
            source_in_frame = source_in_frame[:, None]
            source_visible = source_visible[:, None]
            source_world_valid = source_world_valid[None]
            source_visibility_valid = source_visibility_valid[:, None]
            source_label_valid = source_label_valid[:, None]
            source_surface_valid = source_surface_valid[None]
            source_occlusion_cause = source_occlusion_cause[:, None]
        if source_joint_world.ndim != 3 or source_uv.ndim != 4:
            raise ValueError(
                f"Unsupported keypoint instance layout in {path}: "
                f"world={source_joint_world.shape}, uv={source_uv.shape}"
            )
        source_instance_count = int(source_joint_world.shape[0])
        if source_instance_count > max_instances:
            raise ValueError(
                f"Scene {scene.root} has {source_instance_count} instances, "
                f"padding contract only allows {max_instances}"
            )
        source_name_index = {name: index for index, name in enumerate(source_names)}
        valid_names = np.asarray(
            [name in source_name_index for name in canonical_names], dtype=bool
        )
        name_indices = np.asarray(
            [source_name_index.get(name, 0) for name in canonical_names], dtype=np.int64
        )
        joint_world = source_joint_world[:, name_indices].copy()
        surface_world = source_surface_world[:, name_indices].copy()
        world_valid = (
            np.take(source_world_valid, name_indices, axis=1)
            & valid_names[None]
            & np.isfinite(joint_world).all(axis=-1)
        )
        joint_world[~np.isfinite(joint_world)] = 0.0
        surface_world[~np.isfinite(surface_world)] = 0.0
        joint_world[:, ~valid_names] = 0.0
        surface_world[:, ~valid_names] = 0.0
        if canonicalize_camera:
            valid_cameras = np.flatnonzero(camera_available)
            if valid_cameras.size:
                reference_transform = original_extrinsics[int(valid_cameras[0])]
                joint_world = transform_world_points(
                    joint_world.reshape(-1, 3), reference_transform
                ).reshape(joint_world.shape)
                surface_world = transform_world_points(
                    surface_world.reshape(-1, 3), reference_transform
                ).reshape(surface_world.shape)
        if scale != 1.0:
            joint_world /= scale
            surface_world /= scale
        result["keypoint_world_xyz"][:source_instance_count] = joint_world
        result["surface_world_xyz"][:source_instance_count] = surface_world
        result["name_valid"][:source_instance_count] = valid_names
        result["world_valid"][:source_instance_count] = world_valid
        result["surface_valid"][:source_instance_count] = (
            np.take(source_surface_valid, name_indices, axis=1) & valid_names[None]
        )
        ids = (
            source_ids
            if source_ids is not None
            else np.arange(source_instance_count, dtype=np.int64)
        )
        result["instance_ids"][:, :source_instance_count] = ids[None]

        source_name_indices = name_indices
        source_view_index = {view_id: index for index, view_id in enumerate(source_view_ids)}
        for selected_index, (view_id, original_size) in enumerate(
            zip(view_ids, original_sizes, strict=True)
        ):
            source_index = source_view_index.get(view_id)
            if source_index is None:
                continue
            original_width, original_height = original_size
            uv = np.take(
                source_uv[source_index], source_name_indices, axis=1
            ).copy()
            uv[..., 0] *= output_width / float(original_width)
            uv[..., 1] *= output_height / float(original_height)
            in_frame = (
                np.take(source_in_frame[source_index], source_name_indices, axis=1)
                & valid_names[None]
            )
            label_valid = np.take(source_label_valid[source_index], source_name_indices, axis=1) & valid_names[None]
            cause = np.take(source_occlusion_cause[source_index], source_name_indices, axis=1).copy()
            cause[~in_frame] = -1
            result['keypoint_occlusion_cause'][selected_index, :source_instance_count] = cause
            uv[~np.isfinite(uv)] = 0.0
            visible = (
                np.take(source_visible[source_index], source_name_indices, axis=1)
                & in_frame
            )
            visibility_valid = (
                np.take(
                    source_visibility_valid[source_index],
                    source_name_indices,
                    axis=1,
                )
                & in_frame
            )
            result["keypoint_xy"][selected_index, :source_instance_count] = uv
            result["keypoint_xy"][selected_index, :source_instance_count, :, 0] /= float(output_width)
            result["keypoint_xy"][selected_index, :source_instance_count, :, 1] /= float(output_height)
            result["keypoint_in_frame"][selected_index, :source_instance_count] = in_frame
            result["keypoint_label_valid"][selected_index, :source_instance_count] = label_valid
            result["keypoint_valid"][selected_index, :source_instance_count] = in_frame & label_valid
            result["keypoint_visible"][selected_index, :source_instance_count] = visible
            result["keypoint_visibility_valid"][selected_index, :source_instance_count] = (
                visibility_valid
            )

    if canonical_names and not path.is_file():
        canonical_index = {name: index for index, name in enumerate(canonical_names)}
        found_legacy = False
        for selected_index, (view_id, original_size) in enumerate(
            zip(view_ids, original_sizes, strict=True)
        ):
            annotation_path = (
                scene.root / "annotations" / f"{view_id:08d}.json"
            )
            if not annotation_path.is_file():
                continue
            try:
                payload = json.loads(annotation_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            keypoints = payload.get("keypoints")
            if not isinstance(keypoints, list):
                continue
            original_width, original_height = original_size
            for item in keypoints:
                name = str(item.get("name", ""))
                keypoint_index = canonical_index.get(name)
                if keypoint_index is None:
                    continue
                result["name_valid"][0, keypoint_index] = True
                from training.keypoint_contract import legacy_point
                xy, in_frame, label_valid, visible, known = legacy_point(item, original_width, original_height)
                result['keypoint_in_frame'][selected_index, 0, keypoint_index] = in_frame
                result['keypoint_label_valid'][selected_index, 0, keypoint_index] = label_valid
                if not (in_frame and label_valid):
                    continue
                result["keypoint_xy"][selected_index, 0, keypoint_index] = (
                    xy[0] / original_width,
                    xy[1] / original_height,
                )
                result["keypoint_valid"][selected_index, 0, keypoint_index] = True
                result["keypoint_visible"][selected_index, 0, keypoint_index] = visible
                result["keypoint_visibility_valid"][
                    selected_index, 0, keypoint_index
                ] = known
                found_legacy = True
        if found_legacy:
            source_instance_count = 1
            result["instance_ids"][:, 0] = 0

    if source_instance_count == 0 and any(
        scene.frames[view_id].body_mask_path is not None for view_id in view_ids
    ):
        source_instance_count = 1
        result["instance_ids"][:, 0] = 0

    for selected_index, view_id in enumerate(view_ids):
        frame = scene.frames[view_id]
        for instance_index in range(source_instance_count):
            explicit = (
                scene.root
                / "masks"
                / f"{view_id:08d}_instance{instance_index:02d}.png"
            )
            mask_path = (
                explicit
                if explicit.is_file()
                else frame.body_mask_path
                if source_instance_count == 1
                else None
            )
            mask = read_binary_mask(mask_path, output_size)
            result["masks"][selected_index, instance_index] = mask
            has_mask = bool(mask.any())
            has_points = bool(
                result["keypoint_valid"][selected_index, instance_index].any()
            )
            result["mask_valid"][selected_index, instance_index] = has_mask
            result["instance_valid"][selected_index, instance_index] = (
                has_mask or has_points
            )
            if has_mask:
                result["boxes_cxcywh"][selected_index, instance_index] = (
                    _normalized_box_from_mask(mask)
                )
                result["box_valid"][selected_index, instance_index] = True
            elif has_points:
                result["boxes_cxcywh"][selected_index, instance_index] = (
                    _weak_box_from_keypoints(
                        result["keypoint_xy"][selected_index, instance_index],
                        result["keypoint_valid"][selected_index, instance_index],
                    )
                )

            # Explicit normalized detection labels remain authoritative when
            # segmentation is absent (for example, sparse YOLO pose datasets).
            if source_boxes is not None and view_id in source_view_ids:
                source_index = source_view_ids.index(view_id)
                expected = (len(source_view_ids), source_instance_count, 4)
                if source_boxes.shape != expected or source_box_valid.shape != expected[:2]:
                    raise ValueError(f"Invalid explicit box shapes in {path}")
                box = source_boxes[source_index, instance_index]
                if source_box_valid[source_index, instance_index]:
                    if not (np.isfinite(box).all() and (box >= 0).all()
                            and (box <= 1).all() and (box[2:] > 0).all()):
                        raise ValueError(f"Invalid normalized box in {path}")
                    result["boxes_cxcywh"][selected_index, instance_index] = box
                    result["box_valid"][selected_index, instance_index] = True
                    result["instance_valid"][selected_index, instance_index] = True

    result["actual_instance_count"] = np.asarray(source_instance_count, dtype=np.int64)
    result["available"] = np.asarray(result["instance_valid"].any())
    result["legacy_available"] = np.asarray(
        source_instance_count == 1 and result["instance_valid"].any()
    )
    return result


class BlendedMVSDataset(Dataset):
    def __init__(
        self,
        scenes: list[BlendedScene],
        num_views: int,
        image_width: int,
        image_height: int,
        samples_per_scene: int,
        seed: int,
        use_masked_images: bool,
        pfm_flip_y: bool,
        canonicalize_camera: bool,
        scale_mode: str,
        top_srcs: int,
        load_keypoints: bool,
        normalization_scale_override: float | None = None,
    ) -> None:
        if not scenes:
            raise ValueError("BlendedMVSDataset needs at least one scene")
        self.scenes = scenes
        self.num_views = num_views
        self.image_width = image_width
        self.image_height = image_height
        self.output_size = (image_width, image_height)
        self.samples_per_scene = samples_per_scene
        self.seed = seed
        self.use_masked_images = use_masked_images
        self.pfm_flip_y = pfm_flip_y
        self.canonicalize_camera = canonicalize_camera
        self.scale_mode = scale_mode
        if normalization_scale_override is not None and (
            not np.isfinite(normalization_scale_override) or normalization_scale_override <= 0
        ):
            raise ValueError('normalization_scale_override must be finite and positive')
        self.normalization_scale_override = normalization_scale_override
        self.top_srcs = top_srcs
        self.load_keypoints = load_keypoints
        self.keypoint_names = discover_keypoint_names(scenes) if load_keypoints else []
        self.max_instances = discover_max_instances(scenes) if load_keypoints else 1
        self.samples = [
            (scene_idx, repeat_idx)
            for scene_idx in range(len(scenes))
            for repeat_idx in range(max(1, samples_per_scene))
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        scene_idx, repeat_idx = self.samples[idx]
        scene = self.scenes[scene_idx]
        rng = random.Random(self.seed + 1_000_003 * idx + 9_176 * repeat_idx)
        view_ids = choose_view_ids(scene.frames, scene.pair_map, self.num_views, rng, self.top_srcs)

        images, depths, masks, body_masks, semantic_masks = [], [], [], [], []
        extrinsics, intrinsics, camera_available, depth_available = [], [], [], []
        original_sizes = []
        for view_id in view_ids:
            frame = scene.frames[view_id]
            image_path = (
                frame.masked_image_path
                if self.use_masked_images and frame.masked_image_path is not None
                else frame.image_path
            )
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                width, height = img.size
                original_sizes.append((width, height))
                img = img.resize(self.output_size, Image.Resampling.BICUBIC)
                image = np.asarray(img, dtype=np.float32) / 255.0

            has_depth = frame.depth_path is not None
            if has_depth:
                depth = read_pfm(frame.depth_path, flip_y=self.pfm_flip_y)
                depth = resize_depth_nearest(depth, self.output_size)
                mask = np.isfinite(depth) & (depth > 1e-6)
            else:
                depth = np.zeros((self.image_height, self.image_width), dtype=np.float32)
                mask = np.zeros_like(depth, dtype=bool)
            body_mask = read_binary_mask(frame.body_mask_path, self.output_size)
            semantic_mask = read_semantic_mask(frame.semantic_mask_path, self.output_size)

            has_camera = frame_has_camera(frame)
            if has_camera:
                intrinsic = frame.intrinsic.astype(np.float32, copy=True)
                intrinsic[0, 0] *= self.image_width / float(width)
                intrinsic[0, 2] *= self.image_width / float(width)
                intrinsic[1, 1] *= self.image_height / float(height)
                intrinsic[1, 2] *= self.image_height / float(height)
                extrinsic = frame.extrinsic.astype(np.float32, copy=True)
            else:
                intrinsic = np.eye(3, dtype=np.float32)
                extrinsic = np.eye(4, dtype=np.float32)

            images.append(torch.from_numpy(image).permute(2, 0, 1))
            depths.append(depth)
            masks.append(mask)
            body_masks.append(body_mask)
            semantic_masks.append(semantic_mask)
            extrinsics.append(extrinsic)
            intrinsics.append(torch.from_numpy(intrinsic))
            camera_available.append(has_camera)
            depth_available.append(has_depth)

        depths_np = np.stack(depths, axis=0).astype(np.float32)
        masks_np = np.stack(masks, axis=0)
        camera_available_np = np.asarray(camera_available, dtype=bool)
        depth_available_np = np.asarray(depth_available, dtype=bool)
        original_extrinsics_np = np.stack(extrinsics, axis=0).astype(np.float32)
        extrinsics_np = canonicalize_available_extrinsics(
            original_extrinsics_np,
            camera_available_np,
            self.canonicalize_camera,
        )
        scale = 1.0
        scale_source = "raw"
        if self.normalization_scale_override is not None:
            scale = float(self.normalization_scale_override)
            scale_source = 'fixed_scene_scale'
        elif self.scale_mode == "mean_point_distance":
            scale = mean_world_point_distance(
                depths_np,
                masks_np,
                extrinsics_np,
                torch.stack(intrinsics).numpy(),
            )
            scale_source = "mean_point_distance"
        if self.normalization_scale_override is None and self.scale_mode in {"median_depth", "auto"}:
            valid_depths = depths_np[masks_np]
            if valid_depths.size:
                scale = max(float(np.median(valid_depths)), 1e-6)
                scale_source = "median_depth"
        if self.normalization_scale_override is None and (self.scale_mode == "median_baseline" or (self.scale_mode == "auto" and scale_source == "raw")):
            valid_extrinsics = extrinsics_np[camera_available_np]
            if len(valid_extrinsics) >= 2:
                scale = median_camera_baseline(valid_extrinsics)
                scale_source = "median_baseline"
        if scale_source != "raw":
            depths_np = depths_np / scale
            extrinsics_np[camera_available_np, :3, 3] /= scale

        instances = load_selected_instances(
            scene,
            self.keypoint_names,
            view_ids,
            original_sizes,
            self.output_size,
            original_extrinsics_np,
            camera_available_np,
            self.canonicalize_camera,
            scale,
            self.max_instances,
        )

        # Legacy single-keypoint heads retain their old tensors for M=1 scenes.
        # Native instance-pose training always consumes the padded fields below.
        legacy_available = bool(instances["legacy_available"])
        legacy_valid = (
            instances["world_valid"][0]
            if legacy_available
            else np.zeros(len(self.keypoint_names), dtype=bool)
        )
        legacy_joint = (
            instances["keypoint_world_xyz"][0]
            if legacy_available
            else np.zeros((len(self.keypoint_names), 3), dtype=np.float32)
        )
        legacy_surface = (
            instances["surface_world_xyz"][0]
            if legacy_available
            else np.zeros((len(self.keypoint_names), 3), dtype=np.float32)
        )
        legacy_uv = instances["keypoint_xy"][:, 0].copy()
        legacy_uv[..., 0] *= self.image_width
        legacy_uv[..., 1] *= self.image_height
        legacy_in_frame = instances["keypoint_valid"][:, 0]
        legacy_visible = instances["keypoint_visible"][:, 0]

        return {
            "scene": scene.name,
            "view_ids": ",".join(str(v) for v in view_ids),
            "scale_source": scale_source,
            "scale": torch.tensor(scale, dtype=torch.float32),
            "images": torch.stack(images).float(),
            "depths": torch.from_numpy(depths_np).float(),
            "depth_masks": torch.from_numpy(masks_np).bool(),
            "depth_available": torch.from_numpy(depth_available_np).bool(),
            "body_masks": torch.from_numpy(np.stack(body_masks, axis=0)).bool(),
            "semantic_masks": torch.from_numpy(np.stack(semantic_masks, axis=0)).long(),
            "extrinsics": torch.from_numpy(extrinsics_np).float(),
            "intrinsics": torch.stack(intrinsics).float(),
            "camera_available": torch.from_numpy(camera_available_np).bool(),
            "keypoints_available": torch.tensor(legacy_available, dtype=torch.bool),
            "keypoint_valid": torch.from_numpy(legacy_valid).bool(),
            "keypoint_joint_world_xyz": torch.from_numpy(legacy_joint).float(),
            "keypoint_surface_world_xyz": torch.from_numpy(legacy_surface).float(),
            "keypoint_uv": torch.from_numpy(legacy_uv).float(),
            "keypoint_in_frame": torch.from_numpy(legacy_in_frame).bool(),
            "keypoint_surface_visible": torch.from_numpy(legacy_visible).bool(),
            "instance_supervision_available": torch.from_numpy(instances["available"]).bool(),
            "actual_instance_count": torch.from_numpy(instances["actual_instance_count"]).long(),
            "instance_valid": torch.from_numpy(instances["instance_valid"]).bool(),
            "instance_ids": torch.from_numpy(instances["instance_ids"]).long(),
            "boxes_cxcywh": torch.from_numpy(instances["boxes_cxcywh"]).float(),
            "box_valid": torch.from_numpy(instances["box_valid"]).bool(),
            "masks": torch.from_numpy(instances["masks"]).bool(),
            "mask_valid": torch.from_numpy(instances["mask_valid"]).bool(),
            "instance_keypoint_xy": torch.from_numpy(instances["keypoint_xy"]).float(),
            "instance_keypoint_valid": torch.from_numpy(instances["keypoint_valid"]).bool(),
            "instance_keypoint_in_frame": torch.from_numpy(instances["keypoint_in_frame"]).bool(),
            "instance_keypoint_label_valid": torch.from_numpy(instances["keypoint_label_valid"]).bool(),
            "instance_keypoint_occlusion_cause": torch.from_numpy(instances["keypoint_occlusion_cause"]).long(),
            "instance_keypoint_visible": torch.from_numpy(instances["keypoint_visible"]).bool(),
            "instance_keypoint_visibility_valid": torch.from_numpy(
                instances["keypoint_visibility_valid"]
            ).bool(),
            "instance_keypoint_world_xyz": torch.from_numpy(
                instances["keypoint_world_xyz"]
            ).float(),
            "surface_world_xyz_instances": torch.from_numpy(
                instances["surface_world_xyz"]
            ).float(),
            "instance_world_valid": torch.from_numpy(instances["world_valid"]).bool(),
            "instance_surface_valid": torch.from_numpy(instances["surface_valid"]).bool(),
        }


def camera_centers_from_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    centers = []
    for extrinsic in extrinsics:
        rotation = extrinsic[:3, :3]
        translation = extrinsic[:3, 3]
        centers.append(-rotation.T @ translation)
    return np.stack(centers, axis=0).astype(np.float32)


def mean_world_point_distance(
    depths: np.ndarray,
    masks: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    stride: int = 4,
) -> float:
    """Official VGGT scene unit: mean world-point distance from camera zero."""

    distances = []
    stride = max(int(stride), 1)
    for view_index in range(len(depths)):
        depth = depths[view_index, ::stride, ::stride]
        valid = masks[view_index, ::stride, ::stride] & np.isfinite(depth) & (depth > 1e-6)
        if not valid.any():
            continue
        y, x = np.meshgrid(
            np.arange(0, depths.shape[-2], stride, dtype=np.float32)[: depth.shape[0]],
            np.arange(0, depths.shape[-1], stride, dtype=np.float32)[: depth.shape[1]],
            indexing="ij",
        )
        intrinsic = intrinsics[view_index]
        camera = np.stack(
            [
                (x - intrinsic[0, 2]) / max(float(intrinsic[0, 0]), 1e-6) * depth,
                (y - intrinsic[1, 2]) / max(float(intrinsic[1, 1]), 1e-6) * depth,
                depth,
            ],
            axis=-1,
        )
        rotation = extrinsics[view_index, :3, :3]
        translation = extrinsics[view_index, :3, 3]
        world = (camera - translation[None, None, :]) @ rotation
        distances.append(np.linalg.norm(world[valid], axis=-1))
    if not distances:
        return 1.0
    return max(float(np.concatenate(distances).mean()), 1e-6)


def median_camera_baseline(extrinsics: np.ndarray) -> float:
    centers = camera_centers_from_extrinsics(extrinsics)
    distances = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            distances.append(float(np.linalg.norm(centers[i] - centers[j])))
    if not distances:
        return 1.0
    return max(float(np.median(distances)), 1e-6)


def discover_real_trials(roots_text: str, cameras: list[str]) -> list[RealPoseTrial]:
    trials: list[RealPoseTrial] = []
    for root_text in parse_name_list(roots_text):
        root = Path(root_text).resolve()
        if not root.exists():
            print(f"warning: real validation root does not exist: {root}")
            continue
        calibration = root / "calibration" / "calibration.toml"
        if not calibration.exists():
            print(f"warning: real validation calibration missing: {calibration}")
            continue
        extrinsics, intrinsics = read_real_calibration(calibration)
        videos = {cam: root / "videos-raw" / f"pawfilt_cam{cam}.mp4" for cam in cameras}
        missing = [str(path) for path in videos.values() if not path.exists()]
        if missing:
            print(f"warning: real validation videos missing for {root.name}: {missing[:3]}")
            continue
        missing_cal = [cam for cam in cameras if cam not in extrinsics or cam not in intrinsics]
        if missing_cal:
            print(f"warning: real validation calibration missing cameras for {root.name}: {missing_cal}")
            continue
        frame_count = min(video_frame_count(path) for path in videos.values())
        trials.append(
            RealPoseTrial(
                name=root.name,
                root=root,
                videos=videos,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                frame_count=frame_count,
            )
        )
    return trials


def choose_real_frames(frame_count: int, frames_text: str, start: int, step: int, count: int) -> list[int]:
    if frames_text:
        frames = [int(part) for part in parse_name_list(frames_text)]
    else:
        frames = [start + idx * max(1, step) for idx in range(max(1, count))]
    return [frame for frame in frames if 0 <= frame < frame_count]


def discover_real_palm_roots(roots_text: str, cameras: list[str]) -> list[Path]:
    roots: list[Path] = []
    for item in parse_name_list(roots_text):
        candidate = Path(item).expanduser().resolve()
        candidates = [candidate]
        if candidate.is_dir() and "_Trial_" not in candidate.name:
            candidates = sorted(path for path in candidate.iterdir() if path.is_dir() and "_Trial_" in path.name)
        for root in candidates:
            required = [root / "calibration" / "calibration.toml"]
            required.extend(root / "videos-raw" / f"pawfilt_cam{cam}.mp4" for cam in cameras)
            required.extend(root / "pose-2d" / f"pawfilt_cam{cam}.h5" for cam in cameras)
            if all(path.exists() for path in required):
                roots.append(root)
            elif "_Trial_" in root.name:
                missing = [str(path) for path in required if not path.exists()]
                print(f"warning: real palm trial skipped {root.name}: missing={missing[:3]}")
    return list(dict.fromkeys(roots))


def read_real_palm_calibration(path: Path) -> dict[str, RealPalmCamera]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    cameras: dict[str, RealPalmCamera] = {}
    idx = 0
    while f"cam_{idx}" in data:
        item = data[f"cam_{idx}"]
        name = str(item["name"])
        rotation = cv2.Rodrigues(np.asarray(item["rotation"], dtype=np.float64))[0]
        cameras[name] = RealPalmCamera(
            name=name,
            matrix=np.asarray(item["matrix"], dtype=np.float64),
            distortions=np.asarray(item.get("distortions", []), dtype=np.float64),
            rotation=rotation,
            translation=np.asarray(item["translation"], dtype=np.float64),
        )
        idx += 1
    return cameras


def read_dlc_points(path: Path, keypoints: tuple[str, ...]) -> dict[str, np.ndarray]:
    frame = pd.read_hdf(path)
    result = {}
    for keypoint in keypoints:
        values = []
        for coord in ("x", "y", "likelihood"):
            matches = [
                column
                for column in frame.columns
                if keypoint in tuple(str(part) for part in (column if isinstance(column, tuple) else (column,)))
                and coord in tuple(str(part) for part in (column if isinstance(column, tuple) else (column,)))
            ]
            if not matches:
                values = []
                break
            values.append(frame[matches[0]].to_numpy(dtype=np.float64))
        if values:
            result[keypoint] = np.stack(values, axis=-1)
        else:
            result[keypoint] = np.full((len(frame), 3), np.nan, dtype=np.float64)
            result[keypoint][:, 2] = 0.0
    return result


def triangulate_camera_pair(
    first_points: np.ndarray,
    second_points: np.ndarray,
    first: RealPalmCamera,
    second: RealPalmCamera,
) -> np.ndarray:
    result = np.full((len(first_points), 3), np.nan, dtype=np.float64)
    valid = np.isfinite(first_points).all(axis=1) & np.isfinite(second_points).all(axis=1)
    if not valid.any():
        return result
    first_undistorted = cv2.undistortPoints(
        first_points[valid, None, :], first.matrix, first.distortions
    ).reshape(-1, 2)
    second_undistorted = cv2.undistortPoints(
        second_points[valid, None, :], second.matrix, second.distortions
    ).reshape(-1, 2)
    first_projection = np.concatenate([first.rotation, first.translation[:, None]], axis=1)
    second_projection = np.concatenate([second.rotation, second.translation[:, None]], axis=1)
    homogeneous = cv2.triangulatePoints(
        first_projection,
        second_projection,
        first_undistorted.T,
        second_undistorted.T,
    ).T
    finite_w = np.abs(homogeneous[:, 3]) > 1e-9
    points = np.full((len(homogeneous), 3), np.nan, dtype=np.float64)
    points[finite_w] = homogeneous[finite_w, :3] / homogeneous[finite_w, 3:4]
    result[valid] = points
    return result


class RealPalmDepthDataset(Dataset):
    def __init__(
        self,
        roots: list[Path],
        cameras: list[str],
        image_width: int,
        image_height: int,
        frame_start: int,
        frame_step: int,
        frame_count: int,
        likelihood_threshold: float,
        palm_weight: float,
        anchor_weight: float,
    ) -> None:
        if len(cameras) != 3:
            raise ValueError("Real palm leave-one-camera-out targets require exactly three cameras")
        self.image_width = image_width
        self.image_height = image_height
        self.output_size = (image_width, image_height)
        self.samples: list[RealPalmSample] = []
        for root in roots:
            self._add_trial(
                root,
                cameras,
                frame_start,
                frame_step,
                frame_count,
                likelihood_threshold,
                palm_weight,
                anchor_weight,
            )
        if not self.samples:
            raise ValueError("No valid real palm sparse-depth samples were generated")

    def _add_trial(
        self,
        root: Path,
        cameras: list[str],
        frame_start: int,
        frame_step: int,
        frame_count: int,
        threshold: float,
        palm_weight: float,
        anchor_weight: float,
    ) -> None:
        calibration = read_real_palm_calibration(root / "calibration" / "calibration.toml")
        if any(cam not in calibration for cam in cameras):
            return
        videos = {cam: root / "videos-raw" / f"pawfilt_cam{cam}.mp4" for cam in cameras}
        points = {
            cam: read_dlc_points(root / "pose-2d" / f"pawfilt_cam{cam}.h5", REAL_SPARSE_KEYPOINTS)
            for cam in cameras
        }
        total_frames = min(
            *[video_frame_count(path) for path in videos.values()],
            *[len(points[cam][REAL_SPARSE_KEYPOINTS[0]]) for cam in cameras],
        )
        frames = list(range(frame_start, total_frames, max(1, frame_step)))
        if frame_count > 0:
            frames = frames[:frame_count]
        if not frames:
            return
        frame_array = np.asarray(frames, dtype=np.int64)
        labels_by_camera: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        all_anchor_depths = []

        for camera in cameras:
            other = [name for name in cameras if name != camera]
            xy = np.full((len(frames), len(REAL_SPARSE_KEYPOINTS), 2), np.nan, dtype=np.float64)
            target = np.full((len(frames), len(REAL_SPARSE_KEYPOINTS)), np.nan, dtype=np.float64)
            valid = np.zeros((len(frames), len(REAL_SPARSE_KEYPOINTS)), dtype=bool)
            for point_idx, keypoint in enumerate(REAL_SPARSE_KEYPOINTS):
                current_values = points[camera][keypoint][frame_array]
                first_values = points[other[0]][keypoint][frame_array]
                second_values = points[other[1]][keypoint][frame_array]
                pair_valid = (
                    (first_values[:, 2] >= threshold)
                    & (second_values[:, 2] >= threshold)
                    & np.isfinite(first_values[:, :2]).all(axis=1)
                    & np.isfinite(second_values[:, :2]).all(axis=1)
                )
                pair_first = first_values[:, :2].copy()
                pair_second = second_values[:, :2].copy()
                pair_first[~pair_valid] = np.nan
                pair_second[~pair_valid] = np.nan
                world = triangulate_camera_pair(
                    pair_first,
                    pair_second,
                    calibration[other[0]],
                    calibration[other[1]],
                )
                camera_points = (calibration[camera].rotation @ world.T).T + calibration[camera].translation
                current_valid = (
                    (current_values[:, 2] >= threshold)
                    & np.isfinite(current_values[:, :2]).all(axis=1)
                    & np.isfinite(camera_points).all(axis=1)
                    & (camera_points[:, 2] > 1e-6)
                )
                xy[:, point_idx] = current_values[:, :2]
                target[:, point_idx] = camera_points[:, 2]
                valid[:, point_idx] = current_valid
                if keypoint in REAL_SCALE_ANCHORS:
                    all_anchor_depths.extend(camera_points[current_valid, 2].tolist())
            labels_by_camera[camera] = (xy, target, valid)

        finite_anchor_depths = np.asarray(all_anchor_depths, dtype=np.float64)
        finite_anchor_depths = finite_anchor_depths[np.isfinite(finite_anchor_depths) & (finite_anchor_depths > 0)]
        if not finite_anchor_depths.size:
            print(f"warning: real palm trial skipped {root.name}: no valid static anchor depths")
            return
        scale_mm = float(np.median(finite_anchor_depths))

        for camera in cameras:
            cap = cv2.VideoCapture(str(videos[camera]))
            try:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            finally:
                cap.release()
            xy, target, valid = labels_by_camera[camera]
            for frame_pos, frame_idx in enumerate(frames):
                if not valid[frame_pos, : len(REAL_PALM_KEYPOINTS)].any():
                    continue
                is_palm = np.asarray(
                    [keypoint in REAL_PALM_KEYPOINTS for keypoint in REAL_SPARSE_KEYPOINTS], dtype=bool
                )
                weights = np.where(is_palm, palm_weight, anchor_weight).astype(np.float32)
                self.samples.append(
                    RealPalmSample(
                        trial=root.name,
                        video_path=videos[camera],
                        camera=camera,
                        frame=frame_idx,
                        image_width=width,
                        image_height=height,
                        xy=xy[frame_pos].astype(np.float32),
                        target_depth=(target[frame_pos] / scale_mm).astype(np.float32),
                        valid=valid[frame_pos],
                        is_palm=is_palm,
                        point_weights=weights,
                        scale_mm=scale_mm,
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[idx]
        rgb = read_video_rgb(sample.video_path, sample.frame)
        image = np.asarray(
            Image.fromarray(rgb).resize(self.output_size, Image.Resampling.BICUBIC),
            dtype=np.float32,
        ) / 255.0
        xy = sample.xy.copy()
        xy[:, 0] *= self.image_width / float(sample.image_width)
        xy[:, 1] *= self.image_height / float(sample.image_height)
        return {
            "scene": sample.trial,
            "frame": torch.tensor(sample.frame, dtype=torch.long),
            "view_ids": sample.camera,
            "scale": torch.tensor(sample.scale_mm, dtype=torch.float32),
            "scale_source": "median_static_anchor_depth",
            "images": torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float(),
            "sparse_xy": torch.from_numpy(xy).unsqueeze(0).float(),
            "sparse_depths": torch.from_numpy(sample.target_depth).unsqueeze(0).float(),
            "sparse_valid": torch.from_numpy(sample.valid).unsqueeze(0).bool(),
            "sparse_is_palm": torch.from_numpy(sample.is_palm).unsqueeze(0).bool(),
            "sparse_weights": torch.from_numpy(sample.point_weights).unsqueeze(0).float(),
        }


class RealPoseDataset(Dataset):
    def __init__(
        self,
        trials: list[RealPoseTrial],
        cameras: list[str],
        image_width: int,
        image_height: int,
        frames_text: str,
        frame_start: int,
        frame_step: int,
        frame_count: int,
        canonicalize_camera: bool,
        scale_mode: str,
    ) -> None:
        self.trials = trials
        self.cameras = cameras
        self.image_width = image_width
        self.image_height = image_height
        self.output_size = (image_width, image_height)
        self.canonicalize_camera = canonicalize_camera
        self.scale_mode = scale_mode
        self.samples: list[tuple[int, int]] = []
        frames_by_trial: list[list[int]] = []
        for trial_idx, trial in enumerate(trials):
            frames_by_trial.append(
                choose_real_frames(trial.frame_count, frames_text, frame_start, frame_step, frame_count)
            )
        max_frames = max((len(frames) for frames in frames_by_trial), default=0)
        for frame_position in range(max_frames):
            for trial_idx, frames in enumerate(frames_by_trial):
                if frame_position < len(frames):
                    self.samples.append((trial_idx, frames[frame_position]))
        if not self.samples:
            raise ValueError("No real validation frames selected")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        trial_idx, frame_idx = self.samples[idx]
        trial = self.trials[trial_idx]
        images, intrinsics, extrinsics = [], [], []
        for cam in self.cameras:
            rgb = read_video_rgb(trial.videos[cam], frame_idx)
            height, width = rgb.shape[:2]
            image = np.asarray(
                Image.fromarray(rgb).resize(self.output_size, Image.Resampling.BICUBIC),
                dtype=np.float32,
            ) / 255.0
            intrinsic = trial.intrinsics[cam].astype(np.float32, copy=True)
            intrinsic[0, 0] *= self.image_width / float(width)
            intrinsic[0, 2] *= self.image_width / float(width)
            intrinsic[1, 1] *= self.image_height / float(height)
            intrinsic[1, 2] *= self.image_height / float(height)
            images.append(torch.from_numpy(image).permute(2, 0, 1))
            intrinsics.append(torch.from_numpy(intrinsic))
            extrinsics.append(trial.extrinsics[cam])

        extrinsics_np = canonicalize_extrinsics(np.stack(extrinsics, axis=0), self.canonicalize_camera)
        scale = 1.0
        if self.scale_mode == "median_baseline":
            scale = median_camera_baseline(extrinsics_np)
            extrinsics_np[:, :3, 3] /= scale
        return {
            "scene": trial.name,
            "frame": torch.tensor(frame_idx, dtype=torch.long),
            "view_ids": ",".join(self.cameras),
            "scale": torch.tensor(scale, dtype=torch.float32),
            "scale_source": self.scale_mode,
            "images": torch.stack(images).float(),
            "extrinsics": torch.from_numpy(extrinsics_np).float(),
            "intrinsics": torch.stack(intrinsics).float(),
        }


class RealMultiViewGeometryDataset(Dataset):
    """Synchronized real views with Anipose 3D and DLC 2D supervision."""

    def __init__(
        self,
        trials: list[RealPoseTrial],
        cameras: list[str],
        image_width: int,
        image_height: int,
        frame_start: int,
        frame_step: int,
        frame_count: int,
        likelihood_threshold: float,
        keypoints: tuple[str, ...],
        palm_weight: float,
        finger_weight: float,
        face_weight: float,
        anchor_weight: float,
        canonicalize_camera: bool,
        scale_mode: str,
        scale_by_trial: dict[str, float] | None = None,
    ) -> None:
        if len(cameras) < 2:
            raise ValueError("Real multiview geometry supervision requires at least two cameras")
        self.trials = trials
        self.cameras = cameras
        self.image_width = image_width
        self.image_height = image_height
        self.output_size = (image_width, image_height)
        self.likelihood_threshold = float(likelihood_threshold)
        self.keypoints = keypoints
        self.canonicalize_camera = canonicalize_camera
        self.scale_mode = scale_mode
        self.scale_by_trial = scale_by_trial or {}
        self.pose3d: dict[str, pd.DataFrame] = {}
        self.points2d: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        self.calibrations: dict[str, dict[str, RealPalmCamera]] = {}
        self.samples: list[tuple[int, int]] = []
        self.point_weights = np.asarray(
            [
                (
                    palm_weight
                    if name in REAL_PALM_KEYPOINTS
                    else anchor_weight
                    if name in REAL_SCALE_ANCHORS
                    else face_weight
                    if name in {"snout_A", "mouth_tip_A"}
                    else finger_weight
                )
                for name in keypoints
            ],
            dtype=np.float32,
        )
        self.point_is_palm = np.asarray(
            [name in REAL_PALM_KEYPOINTS for name in keypoints],
            dtype=bool,
        )
        self.point_is_anchor = np.asarray(
            [name in REAL_SCALE_ANCHORS for name in keypoints],
            dtype=bool,
        )
        self.point_is_face = np.asarray(
            [name in {"snout_A", "mouth_tip_A"} for name in keypoints],
            dtype=bool,
        )
        self.point_is_finger = ~(
            self.point_is_palm | self.point_is_anchor | self.point_is_face
        )

        frames_by_trial: list[list[int]] = []
        valid_trials: list[RealPoseTrial] = []
        for trial in trials:
            pose3d_path = trial.root / "pose-3d" / "pawfilt.csv"
            pose2d_paths = {
                cam: trial.root / "pose-2d" / f"pawfilt_cam{cam}.h5"
                for cam in cameras
            }
            if not pose3d_path.exists() or any(not path.exists() for path in pose2d_paths.values()):
                print(f"warning: real geometry labels missing for {trial.name}")
                continue
            pose3d = pd.read_csv(pose3d_path)
            if "fnum" in pose3d:
                pose3d = pose3d.set_index("fnum", drop=False)
            else:
                pose3d.index = np.arange(len(pose3d), dtype=np.int64)
            points2d = {
                cam: read_dlc_points(path, keypoints)
                for cam, path in pose2d_paths.items()
            }
            calibration = read_real_palm_calibration(
                trial.root / "calibration" / "calibration.toml"
            )
            label_frame_count = min(
                len(pose3d),
                *[len(points2d[cam][keypoints[0]]) for cam in cameras],
            )
            usable_count = min(trial.frame_count, label_frame_count)
            frames = list(range(frame_start, usable_count, max(1, frame_step)))
            if frame_count > 0:
                frames = frames[:frame_count]
            filtered_frames = []
            for frame_idx in frames:
                palms_valid = any(
                    sum(
                        bool(
                            np.isfinite(points2d[cam][name][frame_idx, :2]).all()
                            and points2d[cam][name][frame_idx, 2]
                            >= self.likelihood_threshold
                        )
                        for cam in cameras
                    )
                    >= 2
                    for name in REAL_PALM_KEYPOINTS
                )
                if palms_valid:
                    filtered_frames.append(frame_idx)
            if not filtered_frames:
                print(f"warning: no valid real geometry frames for {trial.name}")
                continue
            valid_index = len(valid_trials)
            valid_trials.append(trial)
            self.pose3d[trial.name] = pose3d
            self.points2d[trial.name] = points2d
            self.calibrations[trial.name] = calibration
            frames_by_trial.append(filtered_frames)

        self.trials = valid_trials
        max_frames = max((len(frames) for frames in frames_by_trial), default=0)
        for frame_position in range(max_frames):
            for trial_idx, frames in enumerate(frames_by_trial):
                if frame_position < len(frames):
                    self.samples.append((trial_idx, frames[frame_position]))
        if not self.samples:
            raise ValueError("No synchronized real multiview geometry samples were generated")

    def _world_point_valid(self, row: pd.Series, name: str) -> bool:
        xyz = np.asarray(
            [row.get(f"{name}_{axis}", np.nan) for axis in ("x", "y", "z")],
            dtype=np.float64,
        )
        ncams = float(row.get(f"{name}_ncams", 0.0))
        score = float(row.get(f"{name}_score", 0.0))
        return bool(
            np.isfinite(xyz).all()
            and ncams >= 2
            and score >= self.likelihood_threshold
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        trial_idx, frame_idx = self.samples[idx]
        trial = self.trials[trial_idx]
        images, intrinsics, raw_extrinsics = [], [], []
        xy_by_view, source_xy_by_view, likelihood_by_view, valid_by_view = [], [], [], []
        points2d = self.points2d[trial.name]

        for cam in self.cameras:
            rgb = read_video_rgb(trial.videos[cam], frame_idx)
            source_height, source_width = rgb.shape[:2]
            image = np.asarray(
                Image.fromarray(rgb).resize(self.output_size, Image.Resampling.BICUBIC),
                dtype=np.float32,
            ) / 255.0
            intrinsic = trial.intrinsics[cam].astype(np.float32, copy=True)
            intrinsic[0, 0] *= self.image_width / float(source_width)
            intrinsic[0, 2] *= self.image_width / float(source_width)
            intrinsic[1, 1] *= self.image_height / float(source_height)
            intrinsic[1, 2] *= self.image_height / float(source_height)
            xy = np.stack(
                [points2d[cam][name][frame_idx, :2] for name in self.keypoints],
                axis=0,
            ).astype(np.float32)
            likelihood = np.asarray(
                [points2d[cam][name][frame_idx, 2] for name in self.keypoints],
                dtype=np.float32,
            )
            source_xy = xy.copy()
            xy[:, 0] *= self.image_width / float(source_width)
            xy[:, 1] *= self.image_height / float(source_height)
            view_valid = (
                np.isfinite(xy).all(axis=1)
                & np.isfinite(likelihood)
                & (likelihood >= self.likelihood_threshold)
                & (xy[:, 0] >= 0)
                & (xy[:, 0] < self.image_width)
                & (xy[:, 1] >= 0)
                & (xy[:, 1] < self.image_height)
            )
            xy[~view_valid] = 0.0
            images.append(torch.from_numpy(image).permute(2, 0, 1))
            intrinsics.append(torch.from_numpy(intrinsic))
            raw_extrinsics.append(trial.extrinsics[cam])
            xy_by_view.append(xy)
            source_xy_by_view.append(source_xy)
            likelihood_by_view.append(likelihood)
            valid_by_view.append(view_valid)

        raw_extrinsics_np = np.stack(raw_extrinsics, axis=0).astype(np.float32)
        extrinsics_np = canonicalize_extrinsics(raw_extrinsics_np, self.canonicalize_camera)
        scale = 1.0
        if self.scale_mode == "median_baseline":
            scale = median_camera_baseline(extrinsics_np)
            extrinsics_np[:, :3, 3] /= scale
        elif self.scale_mode == "vggt_rig":
            if trial.name not in self.scale_by_trial:
                raise KeyError(f"Missing VGGT rig scale for real trial {trial.name}")
            scale = max(float(self.scale_by_trial[trial.name]), 1e-6)
            extrinsics_np[:, :3, 3] /= scale

        world = np.zeros((len(self.keypoints), 3), dtype=np.float32)
        world_valid = np.zeros(len(self.keypoints), dtype=bool)
        reference_extrinsic = raw_extrinsics_np[0]
        source_xy_np = np.stack(source_xy_by_view, axis=0)
        likelihood_np = np.stack(likelihood_by_view, axis=0)
        calibration = self.calibrations[trial.name]
        for point_idx, _ in enumerate(self.keypoints):
            valid_views = np.flatnonzero(
                np.isfinite(source_xy_np[:, point_idx]).all(axis=1)
                & np.isfinite(likelihood_np[:, point_idx])
                & (likelihood_np[:, point_idx] >= self.likelihood_threshold)
            )
            if valid_views.size < 2:
                continue
            pair_points = []
            for first_pos, first_idx in enumerate(valid_views):
                for second_idx in valid_views[first_pos + 1 :]:
                    first_cam = self.cameras[int(first_idx)]
                    second_cam = self.cameras[int(second_idx)]
                    pair_world = triangulate_camera_pair(
                        source_xy_np[first_idx, point_idx][None, :],
                        source_xy_np[second_idx, point_idx][None, :],
                        calibration[first_cam],
                        calibration[second_cam],
                    )[0]
                    if np.isfinite(pair_world).all():
                        pair_points.append(pair_world)
            if not pair_points:
                continue
            raw_world = np.median(np.stack(pair_points, axis=0), axis=0).astype(np.float32)
            if self.canonicalize_camera:
                raw_world = transform_world_points(raw_world[None, :], reference_extrinsic)[0]
            if self.scale_mode in {"median_baseline", "vggt_rig"}:
                raw_world = raw_world / scale
            world[point_idx] = raw_world
            world_valid[point_idx] = True

        view_valid_np = np.stack(valid_by_view, axis=0)
        rig_likelihood_values = likelihood_np[view_valid_np]
        rig_frame_confidence = (
            float(np.median(rig_likelihood_values))
            if rig_likelihood_values.size
            else 0.0
        )
        rig_frame_coverage = float(view_valid_np.mean())
        view_valid_np &= world_valid[None, :]
        weights = np.broadcast_to(
            self.point_weights[None, :],
            view_valid_np.shape,
        ).copy()
        return {
            "scene": trial.name,
            "frame": torch.tensor(frame_idx, dtype=torch.long),
            "view_ids": ",".join(self.cameras),
            "scale": torch.tensor(scale, dtype=torch.float32),
            "scale_source": self.scale_mode,
            "images": torch.stack(images).float(),
            "extrinsics": torch.from_numpy(extrinsics_np).float(),
            "intrinsics": torch.stack(intrinsics).float(),
            "geometry_world": torch.from_numpy(world).float(),
            "geometry_valid": torch.from_numpy(world_valid).bool(),
            "geometry_xy": torch.from_numpy(np.stack(xy_by_view, axis=0)).float(),
            "geometry_view_valid": torch.from_numpy(view_valid_np).bool(),
            "geometry_weights": torch.from_numpy(weights).float(),
            "geometry_is_palm": torch.from_numpy(self.point_is_palm).bool(),
            "geometry_is_finger": torch.from_numpy(self.point_is_finger).bool(),
            "geometry_is_face": torch.from_numpy(self.point_is_face).bool(),
            "geometry_is_anchor": torch.from_numpy(self.point_is_anchor).bool(),
            "rig_frame_confidence": torch.tensor(
                rig_frame_confidence,
                dtype=torch.float32,
            ),
            "rig_frame_coverage": torch.tensor(
                rig_frame_coverage,
                dtype=torch.float32,
            ),
        }


class SameTrialFrameBatchSampler(Sampler[list[int]]):
    """Yield small multi-frame batches that never cross trial boundaries."""

    def __init__(
        self,
        dataset: RealMultiViewGeometryDataset,
        frames_per_batch: int,
        *,
        shuffle: bool,
    ) -> None:
        self.frames_per_batch = max(2, int(frames_per_batch))
        self.shuffle = bool(shuffle)
        self.groups: dict[int, list[int]] = {}
        for sample_idx, (trial_idx, _) in enumerate(dataset.samples):
            self.groups.setdefault(trial_idx, []).append(sample_idx)

    def __iter__(self):
        batches = []
        for indices in self.groups.values():
            trial_indices = list(indices)
            if self.shuffle:
                random.shuffle(trial_indices)
            for start in range(0, len(trial_indices), self.frames_per_batch):
                batch = trial_indices[start : start + self.frames_per_batch]
                if len(batch) >= 2:
                    batches.append(batch)
        if self.shuffle:
            random.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        count = 0
        for indices in self.groups.values():
            full_batches, remainder = divmod(len(indices), self.frames_per_batch)
            count += full_batches + int(remainder >= 2)
        return count


def split_scenes(scenes: list[BlendedScene], val_scenes: int) -> tuple[list[BlendedScene], list[BlendedScene]]:
    if val_scenes <= 0 or len(scenes) == 1:
        return scenes, []
    val_count = min(val_scenes, len(scenes) - 1)
    return scenes[:-val_count], scenes[-val_count:]


def training_image_shape(args: argparse.Namespace) -> tuple[int, int]:
    width = args.image_width if args.image_width > 0 else args.image_size
    height = args.image_height if args.image_height > 0 else args.image_size
    return width, height


def make_loader(args: argparse.Namespace, split: str) -> DataLoader:
    load_scene.body_mask_dirs = args.body_mask_dirs
    load_scene.semantic_mask_dirs = args.semantic_mask_dirs
    roots = discover_scene_roots(Path(args.blendedmvs_root))
    if args.scene_path_includes:
        include_patterns = parse_name_list(args.scene_path_includes)
        roots = [
            root
            for root in roots
            if any(
                fnmatch.fnmatch(root.as_posix(), pattern)
                for pattern in include_patterns
            )
        ]
    if args.scene_path_excludes:
        exclude_patterns = parse_name_list(args.scene_path_excludes)
        roots = [
            root
            for root in roots
            if not any(
                fnmatch.fnmatch(root.as_posix(), pattern)
                for pattern in exclude_patterns
            )
        ]
    if args.max_scenes:
        roots = roots[: args.max_scenes]
    scenes = [load_scene(root) for root in roots]
    scenes = [
        scene
        for scene in scenes
        if len(scene.frames) >= args.num_views
        and any(frame_has_supervision(frame) for frame in scene.frames.values())
    ]
    train_scenes, val_scenes = split_scenes(scenes, args.val_scenes)
    chosen = train_scenes if split == "train" else val_scenes
    if not chosen:
        raise ValueError(f"No {split} scenes available after split")
    image_width, image_height = training_image_shape(args)
    dataset = BlendedMVSDataset(
        scenes=chosen,
        num_views=args.num_views,
        image_width=image_width,
        image_height=image_height,
        samples_per_scene=args.samples_per_scene if split == "train" else 1,
        seed=args.seed + (0 if split == "train" else 10_000),
        use_masked_images=args.use_masked_images,
        pfm_flip_y=args.pfm_flip_y,
        canonicalize_camera=args.canonicalize_camera,
        scale_mode=args.scale_mode,
        top_srcs=args.top_srcs,
        load_keypoints=args.load_keypoints,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(split == "train"),
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def make_real_val_loader(args: argparse.Namespace) -> DataLoader | None:
    if not args.real_val_roots:
        return None
    cameras = parse_name_list(args.real_val_cameras)
    trials = discover_real_trials(args.real_val_roots, cameras)
    if not trials:
        print("real_val_loader_skipped reason=no_valid_trials")
        return None
    image_width, image_height = training_image_shape(args)
    dataset = RealPoseDataset(
        trials=trials,
        cameras=cameras,
        image_width=image_width,
        image_height=image_height,
        frames_text=args.real_val_frames,
        frame_start=args.real_val_frame_start,
        frame_step=args.real_val_frame_step,
        frame_count=args.real_val_frame_count,
        canonicalize_camera=args.canonicalize_camera,
        scale_mode=args.real_val_scale_mode,
    )
    print(
        "real_val_dataset_ok",
        f"trials={len(trials)}",
        f"samples={len(dataset)}",
        f"cameras={','.join(cameras)}",
        f"scale_mode={args.real_val_scale_mode}",
    )
    return DataLoader(
        dataset,
        batch_size=args.real_val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def make_real_camera_train_loader(args: argparse.Namespace) -> DataLoader | None:
    if not args.real_camera_train_roots:
        return None
    cameras = parse_name_list(args.real_camera_train_cameras)
    trials = discover_real_trials(args.real_camera_train_roots, cameras)
    if not trials:
        print("real_camera_train_loader_skipped reason=no_valid_trials")
        return None
    image_width, image_height = training_image_shape(args)
    dataset = RealPoseDataset(
        trials=trials,
        cameras=cameras,
        image_width=image_width,
        image_height=image_height,
        frames_text=args.real_camera_train_frames,
        frame_start=args.real_camera_train_frame_start,
        frame_step=args.real_camera_train_frame_step,
        frame_count=args.real_camera_train_frame_count,
        canonicalize_camera=args.canonicalize_camera,
        scale_mode=args.real_camera_train_scale_mode,
    )
    print(
        "real_camera_train_dataset_ok",
        f"trials={len(trials)}",
        f"samples={len(dataset)}",
        f"cameras={','.join(cameras)}",
        f"frame_step={args.real_camera_train_frame_step}",
        f"scale_mode={args.real_camera_train_scale_mode}",
    )
    return DataLoader(
        dataset,
        batch_size=args.real_camera_train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def make_real_palm_loader(args: argparse.Namespace, split: str) -> DataLoader | None:
    roots_text = args.real_train_roots if split == "train" else args.real_depth_val_roots
    if not roots_text:
        return None
    cameras = parse_name_list(args.real_depth_cameras)
    roots = discover_real_palm_roots(roots_text, cameras)
    if not roots:
        print(f"real_palm_{split}_loader_skipped reason=no_valid_trials")
        return None
    frame_step = args.real_train_frame_step if split == "train" else args.real_depth_val_frame_step
    frame_count = args.real_train_frame_count if split == "train" else args.real_depth_val_frame_count
    image_width, image_height = training_image_shape(args)
    dataset = RealPalmDepthDataset(
        roots=roots,
        cameras=cameras,
        image_width=image_width,
        image_height=image_height,
        frame_start=args.real_depth_frame_start,
        frame_step=frame_step,
        frame_count=frame_count,
        likelihood_threshold=args.real_depth_likelihood_threshold,
        palm_weight=args.real_palm_point_weight,
        anchor_weight=args.real_anchor_point_weight,
    )
    print(
        f"real_palm_{split}_dataset_ok",
        f"trials={len(roots)}",
        f"samples={len(dataset)}",
        f"cameras={','.join(cameras)}",
        f"frame_step={frame_step}",
    )
    return DataLoader(
        dataset,
        batch_size=args.real_sparse_batch_size,
        shuffle=(split == "train"),
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def make_real_geometry_loader(args: argparse.Namespace, split: str) -> DataLoader | None:
    roots_text = (
        args.real_geometry_train_roots
        if split == "train"
        else args.real_geometry_val_roots
    )
    if not roots_text:
        return None
    cameras = parse_name_list(args.real_geometry_cameras)
    trials = discover_real_trials(roots_text, cameras)
    if not trials:
        print(f"real_geometry_{split}_loader_skipped reason=no_valid_trials")
        return None
    image_width, image_height = training_image_shape(args)
    frame_step = (
        args.real_geometry_train_frame_step
        if split == "train"
        else args.real_geometry_val_frame_step
    )
    frame_count = (
        args.real_geometry_train_frame_count
        if split == "train"
        else args.real_geometry_val_frame_count
    )
    keypoints = tuple(
        parse_name_list(args.real_geometry_keypoints)
        if args.real_geometry_keypoints
        else REAL_GEOMETRY_KEYPOINTS
    )
    scale_by_trial: dict[str, float] = {}
    if args.real_geometry_scale_mode == "vggt_rig":
        if not args.real_geometry_scale_map:
            raise ValueError("--real-geometry-scale-map is required for vggt_rig scale mode")
        scale_payload = json.loads(Path(args.real_geometry_scale_map).read_text(encoding="utf-8"))
        scale_entries = scale_payload.get("trials", scale_payload)
        if isinstance(scale_entries, list):
            scale_by_trial = {
                str(entry["trial"]): float(entry["target_scale"])
                for entry in scale_entries
            }
        else:
            scale_by_trial = {
                str(name): float(
                    entry["target_scale"] if isinstance(entry, dict) else entry
                )
                for name, entry in scale_entries.items()
            }
    dataset = RealMultiViewGeometryDataset(
        trials=trials,
        cameras=cameras,
        image_width=image_width,
        image_height=image_height,
        frame_start=args.real_geometry_frame_start,
        frame_step=frame_step,
        frame_count=frame_count,
        likelihood_threshold=args.real_geometry_likelihood_threshold,
        keypoints=keypoints,
        palm_weight=args.real_geometry_palm_weight,
        finger_weight=args.real_geometry_finger_weight,
        face_weight=args.real_geometry_face_weight,
        anchor_weight=args.real_geometry_anchor_weight,
        canonicalize_camera=args.canonicalize_camera,
        scale_mode=args.real_geometry_scale_mode,
        scale_by_trial=scale_by_trial,
    )
    print(
        f"real_geometry_{split}_dataset_ok",
        f"trials={len(dataset.trials)}",
        f"samples={len(dataset)}",
        f"cameras={','.join(cameras)}",
        f"keypoints={len(keypoints)}",
        f"frame_step={frame_step}",
        f"image_shape={image_width}x{image_height}",
    )
    temporal_frames = (
        max(1, int(getattr(args, "real_geometry_temporal_frames", 1)))
        if split == "train"
        else 1
    )
    if temporal_frames > 1:
        return DataLoader(
            dataset,
            batch_sampler=SameTrialFrameBatchSampler(
                dataset,
                temporal_frames,
                shuffle=True,
            ),
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return DataLoader(
        dataset,
        batch_size=args.real_geometry_batch_size,
        shuffle=(split == "train"),
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def set_requires_grad(module: torch.nn.Module, value: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(value)


def configure_trainable(model: VGGTOmega, args: argparse.Namespace) -> list[str]:
    set_requires_grad(model, False)
    if args.train_mode == "full":
        set_requires_grad(model, True)
    else:
        if args.train_mode in {
            "light_heads",
            "camera_branch",
            "geometry_adapters",
        } and model.camera_head is not None:
            set_requires_grad(model.camera_head.token_norm, True)
            set_requires_grad(model.camera_head.trunk_norm, True)
            set_requires_grad(model.camera_head.camera_branch, True)
        if args.train_mode in {
            "light_heads",
            "depth_projection",
            "geometry_adapters",
        } and model.dense_head is not None:
            set_requires_grad(model.dense_head.proj, True)
            if args.train_mode != "geometry_adapters":
                set_requires_grad(model.dense_head.proj_conf, True)
        if args.train_mode in {"heads", "camera"} and model.camera_head is not None:
            set_requires_grad(model.camera_head, True)
        if args.train_mode in {"heads", "dense"} and model.dense_head is not None:
            set_requires_grad(model.dense_head, True)
        if args.train_mode in {
            "keypoint_head",
            "keypoint_last_block",
        } and model.keypoint_head is not None:
            set_requires_grad(model.keypoint_head, True)
        if args.train_mode in {
            "instance_pose_head",
            "instance_pose_last_block",
        } and model.instance_pose_head is not None:
            set_requires_grad(model.instance_pose_head, True)
        if args.train_mode in {
            "last_blocks",
            "geometry_adapters",
            "keypoint_last_block",
            "instance_pose_last_block",
        }:
            if args.train_mode == "last_blocks":
                if model.camera_head is not None:
                    set_requires_grad(model.camera_head, True)
                if model.dense_head is not None:
                    set_requires_grad(model.dense_head, True)
            n = min(args.unfreeze_last_blocks, model.aggregator.depth)
            for block in model.aggregator.frame_blocks[-n:]:
                set_requires_grad(block, True)
            for block in model.aggregator.inter_frame_blocks[-n:]:
                set_requires_grad(block, True)
        if args.train_camera_register_tokens:
            model.aggregator.camera_token.requires_grad_(True)
            model.aggregator.register_token.requires_grad_(True)
    if args.freeze_confidence_projection and model.dense_head is not None:
        set_requires_grad(model.dense_head.proj_conf, False)
    return [name for name, param in model.named_parameters() if param.requires_grad]


def aggregator_is_frozen(model: VGGTOmega) -> bool:
    return not any(param.requires_grad for name, param in model.named_parameters() if name.startswith("aggregator."))


def forward_for_training(model: VGGTOmega, images: torch.Tensor, frozen_aggregator: bool) -> dict[str, torch.Tensor]:
    amp_enabled = images.is_cuda
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    grad_context = torch.no_grad() if frozen_aggregator else torch.enable_grad()
    with grad_context:
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            aggregated_tokens_list, patch_token_start = model.aggregator(images)
    if frozen_aggregator:
        aggregated_tokens_list = [
            value.detach() if value is not None else None
            for value in aggregated_tokens_list
        ]

    final_tokens = aggregated_tokens_list[-1]
    if final_tokens is None:
        raise ValueError("Aggregator did not cache its final layer")
    predictions: dict[str, torch.Tensor] = {
        "patch_tokens": final_tokens[:, :, patch_token_start:].contiguous(),
    }
    with torch.autocast(device_type="cuda", enabled=False):
        if model.camera_head is not None:
            predictions["pose_enc"] = model.camera_head(
                aggregated_tokens_list,
                patch_token_start=patch_token_start,
            )
        if model.dense_head is not None:
            depth, depth_conf = model.dense_head(
                aggregated_tokens_list,
                images=images,
                patch_token_start=patch_token_start,
            )
            predictions["depth"] = depth
            predictions["depth_conf"] = depth_conf
        if model.keypoint_head is not None:
            heatmaps, visibility, pixel_sigma = model.keypoint_head(
                aggregated_tokens_list,
                images=images,
                patch_token_start=patch_token_start,
            )
            predictions["keypoint_heatmap_logits"] = heatmaps
            predictions["keypoint_visibility_logits"] = visibility
            predictions["keypoint_pixel_sigma"] = pixel_sigma
        if model.instance_pose_head is not None:
            predictions.update(
                model.instance_pose_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_token_start=patch_token_start,
                )
            )
    return predictions


def camera_loss(
    pred_pose: torch.Tensor,
    target_pose: torch.Tensor,
    available: torch.Tensor | None = None,
    loss_type: str = "smooth_l1",
) -> dict[str, torch.Tensor]:
    if available is not None:
        available = available.bool()
        if not available.any():
            zero = pred_pose.mean() * 0.0
            return {"loss_cam_t": zero, "loss_cam_q": zero, "loss_cam_fov": zero}
        pred_pose = pred_pose[available]
        target_pose = target_pose[available]
    pred_t, target_t = pred_pose[..., :3], target_pose[..., :3]
    pred_q = F.normalize(pred_pose[..., 3:7], dim=-1)
    target_q = F.normalize(target_pose[..., 3:7], dim=-1)
    pred_fov, target_fov = pred_pose[..., 7:], target_pose[..., 7:]
    if loss_type == "l1":
        translation_loss = F.l1_loss(pred_t, target_t)
        fov_loss = F.l1_loss(pred_fov, target_fov)
    elif loss_type == "smooth_l1":
        translation_loss = F.smooth_l1_loss(pred_t, target_t)
        fov_loss = F.smooth_l1_loss(pred_fov, target_fov)
    else:
        raise ValueError(f"Unknown camera loss type: {loss_type}")
    return {
        "loss_cam_t": translation_loss,
        "loss_cam_q": torch.minimum((pred_q - target_q).abs(), (pred_q + target_q).abs()).mean(),
        "loss_cam_fov": fov_loss,
    }


def weighted_camera_loss(
    pred_pose: torch.Tensor,
    batch: dict[str, Any],
    translation_weight: float,
    rotation_weight: float,
    fov_weight: float,
    loss_type: str = "smooth_l1",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target_pose = extri_intri_to_pose_encoding(
        batch["extrinsics"],
        batch["intrinsics"],
        batch["images"].shape[-2:],
    )
    losses = camera_loss(pred_pose, target_pose, loss_type=loss_type)
    total = (
        translation_weight * losses["loss_cam_t"]
        + rotation_weight * losses["loss_cam_q"]
        + fov_weight * losses["loss_cam_fov"]
    )
    return total, losses


def parse_label_weight_map(text: str) -> dict[int, float]:
    weights: dict[int, float] = {}
    for item in parse_name_list(text):
        if ":" not in item:
            raise ValueError(f"Expected LABEL:WEIGHT in --semantic-label-weights, got {item!r}")
        label_text, weight_text = item.split(":", 1)
        weights[int(label_text)] = float(weight_text)
    return weights


def log_depth_residual(pred_depth: torch.Tensor, target_depth: torch.Tensor) -> torch.Tensor:
    if pred_depth.ndim == 5:
        pred_depth = pred_depth[..., 0]
    return torch.log(pred_depth.clamp_min(1e-6)) - torch.log(target_depth.clamp_min(1e-6))


def masked_smooth_l1(residual: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if valid.sum() < 16:
        return residual.mean() * 0.0
    return F.smooth_l1_loss(residual[valid], torch.zeros_like(residual[valid]))


def weighted_masked_smooth_l1(
    residual: torch.Tensor,
    valid: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if valid.sum() < 16:
        return residual.mean() * 0.0
    loss = F.smooth_l1_loss(residual, torch.zeros_like(residual), reduction="none")
    if weights is None:
        return loss[valid].mean()
    weights = weights.to(dtype=loss.dtype)
    weighted = loss[valid] * weights[valid].clamp_min(0.0)
    denom = weights[valid].clamp_min(0.0).sum().clamp_min(1e-6)
    return weighted.sum() / denom


def semantic_pixel_weights(semantic_mask: torch.Tensor, weight_map: dict[int, float]) -> torch.Tensor | None:
    if not weight_map:
        return None
    weights = torch.ones_like(semantic_mask, dtype=torch.float32)
    for label, weight in weight_map.items():
        weights = torch.where(semantic_mask == label, torch.full_like(weights, float(weight)), weights)
    return weights


def geometry_pixel_weights(
    valid: torch.Tensor,
    body_mask: torch.Tensor,
    semantic_mask: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor | None:
    weights = semantic_pixel_weights(
        semantic_mask,
        parse_label_weight_map(args.semantic_label_weights),
    )
    if not body_mask.any() or args.body_depth_balance <= 0:
        return weights
    base = torch.ones_like(valid, dtype=torch.float32) if weights is None else weights.float()
    positive = valid & body_mask
    negative = valid & ~body_mask
    positive_count = positive.sum(dim=(-2, -1), keepdim=True).clamp_min(1)
    negative_count = negative.sum(dim=(-2, -1), keepdim=True).clamp_min(1)
    valid_count = valid.sum(dim=(-2, -1), keepdim=True).clamp_min(1)
    positive_fraction = float(np.clip(args.body_depth_fraction, 0.0, 1.0))
    balanced = torch.where(
        positive,
        valid_count.float() * positive_fraction / positive_count.float(),
        valid_count.float() * (1.0 - positive_fraction) / negative_count.float(),
    )
    blend = float(np.clip(args.body_depth_balance, 0.0, 1.0))
    return base * ((1.0 - blend) + blend * balanced)


def region_balanced_loss(
    residual: torch.Tensor,
    valid: torch.Tensor,
    region_mask: torch.Tensor,
    positive_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positive = valid & region_mask
    negative = valid & ~region_mask
    pos_loss = masked_smooth_l1(residual, positive)
    neg_loss = masked_smooth_l1(residual, negative)
    if positive.sum() >= 16 and negative.sum() >= 16:
        fraction = float(np.clip(positive_fraction, 0.0, 1.0))
        return fraction * pos_loss + (1.0 - fraction) * neg_loss, pos_loss, neg_loss
    if positive.sum() >= 16:
        return pos_loss, pos_loss, neg_loss
    if negative.sum() >= 16:
        return neg_loss, pos_loss, neg_loss
    zero = residual.mean() * 0.0
    return zero, zero, zero


def gradient_loss_one_scale(residual: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    losses = []
    valid_x = valid[..., :, 1:] & valid[..., :, :-1]
    if valid_x.sum() >= 16:
        dx = residual[..., :, 1:] - residual[..., :, :-1]
        losses.append(dx[valid_x].abs().mean())
    valid_y = valid[..., 1:, :] & valid[..., :-1, :]
    if valid_y.sum() >= 16:
        dy = residual[..., 1:, :] - residual[..., :-1, :]
        losses.append(dy[valid_y].abs().mean())
    if not losses:
        return residual.mean() * 0.0
    return torch.stack(losses).mean()


def depth_gradient_loss(
    pred_depth: torch.Tensor,
    target_depth: torch.Tensor,
    valid: torch.Tensor,
    scales: list[int],
) -> torch.Tensor:
    if pred_depth.ndim == 5:
        pred_depth = pred_depth[..., 0]
    losses = []
    for scale in scales:
        if scale <= 1:
            pred_s, target_s, valid_s = pred_depth, target_depth, valid
        else:
            pred_s = F.avg_pool2d(pred_depth.flatten(0, 1).unsqueeze(1), scale, stride=scale).squeeze(1)
            target_s = F.avg_pool2d(target_depth.flatten(0, 1).unsqueeze(1), scale, stride=scale).squeeze(1)
            valid_s = (
                F.avg_pool2d(valid.float().flatten(0, 1).unsqueeze(1), scale, stride=scale).squeeze(1) > 0.99
            )
            pred_s = pred_s.reshape(*pred_depth.shape[:2], *pred_s.shape[-2:])
            target_s = target_s.reshape(*target_depth.shape[:2], *target_s.shape[-2:])
            valid_s = valid_s.reshape(*valid.shape[:2], *valid_s.shape[-2:])
        residual = log_depth_residual(pred_s, target_s)
        residual = torch.where(valid_s, residual, torch.zeros_like(residual))
        losses.append(gradient_loss_one_scale(residual, valid_s))
    return torch.stack(losses).mean() if losses else pred_depth.mean() * 0.0


def depth_loss(
    pred_depth: torch.Tensor,
    target_depth: torch.Tensor,
    mask: torch.Tensor,
    body_mask: torch.Tensor,
    semantic_mask: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    if pred_depth.ndim == 5:
        pred_depth = pred_depth[..., 0]
    valid = mask & torch.isfinite(target_depth) & (target_depth > 1e-6)
    residual = log_depth_residual(pred_depth, target_depth)
    residual = torch.where(valid, residual, torch.zeros_like(residual))
    label_weights = parse_label_weight_map(args.semantic_label_weights)
    pixel_weights = semantic_pixel_weights(semantic_mask, label_weights)
    all_loss = weighted_masked_smooth_l1(residual, valid, pixel_weights)
    body_loss, body_only, scene_only = region_balanced_loss(
        residual,
        valid,
        body_mask,
        positive_fraction=args.body_depth_fraction,
    )
    if body_mask.any() and args.body_depth_balance > 0:
        balance = float(np.clip(args.body_depth_balance, 0.0, 1.0))
        base_loss = balance * body_loss + (1.0 - balance) * all_loss
    else:
        base_loss = all_loss

    logs = {
        "loss_depth": base_loss,
        "loss_depth_all": all_loss,
        "loss_depth_body": body_only,
        "loss_depth_scene": scene_only,
    }
    if args.depth_gradient_weight > 0:
        scales = [int(part) for part in parse_name_list(args.depth_gradient_scales)]
        loss_grad = depth_gradient_loss(pred_depth, target_depth, valid, scales)
        logs["loss_depth_grad"] = loss_grad
        logs["loss_depth"] = logs["loss_depth"] + args.depth_gradient_weight * loss_grad
    return logs


def sparse_depth_loss(
    pred_depth: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if pred_depth.ndim == 5:
        pred_depth = pred_depth[..., 0]
    batch_size, views, height, width = pred_depth.shape
    xy = batch["sparse_xy"]
    target = batch["sparse_depths"]
    label_valid = (
        batch["sparse_valid"]
        & torch.isfinite(target)
        & (target > 1e-6)
        & torch.isfinite(xy).all(dim=-1)
    )
    safe_xy = torch.where(label_valid[..., None], xy, torch.zeros_like(xy))
    safe_target = torch.where(label_valid, target, torch.ones_like(target))
    point_count = xy.shape[-2]
    grid = safe_xy.clone()
    grid[..., 0] = 2.0 * grid[..., 0] / max(width - 1, 1) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / max(height - 1, 1) - 1.0
    sampled = F.grid_sample(
        pred_depth.reshape(batch_size * views, 1, height, width),
        grid.reshape(batch_size * views, point_count, 1, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).reshape(batch_size, views, point_count)
    valid = (
        label_valid
        & torch.isfinite(sampled)
        & (sampled > 1e-6)
        & (grid[..., 0].abs() <= 1.0)
        & (grid[..., 1].abs() <= 1.0)
    )
    residual = torch.log(sampled.clamp_min(1e-6)) - torch.log(safe_target.clamp_min(1e-6))
    residual = torch.where(valid, residual, torch.zeros_like(residual))
    per_point = F.smooth_l1_loss(residual, torch.zeros_like(residual), reduction="none")
    weights = batch["sparse_weights"].to(dtype=per_point.dtype).clamp_min(0.0)
    weighted_valid = weights * valid.to(dtype=weights.dtype)
    denom = weighted_valid.sum().clamp_min(1e-6)
    loss = (per_point * weighted_valid).sum() / denom

    scale_mm = batch["scale"].to(dtype=sampled.dtype)[:, None, None]
    error_mm = (sampled - safe_target).abs() * scale_mm
    palm_valid = valid & batch["sparse_is_palm"]
    anchor_valid = valid & ~batch["sparse_is_palm"]

    def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.any():
            return values[mask].mean()
        return values.mean() * 0.0

    return {
        "loss_sparse_depth": loss,
        "sparse_depth_mae_mm": masked_mean(error_mm, valid),
        "sparse_palm_mae_mm": masked_mean(error_mm, palm_valid),
        "sparse_anchor_mae_mm": masked_mean(error_mm, anchor_valid),
        "sparse_valid_points": valid.sum().to(dtype=sampled.dtype),
        "sparse_palm_points": palm_valid.sum().to(dtype=sampled.dtype),
    }


def sample_depth_at_keypoints(
    depth: torch.Tensor,
    xy: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if depth.ndim == 5:
        depth = depth[..., 0]
    batch_size, views, height, width = depth.shape
    point_count = xy.shape[-2]
    grid = xy.clone()
    grid[..., 0] = 2.0 * grid[..., 0] / max(width - 1, 1) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / max(height - 1, 1) - 1.0
    sampled = F.grid_sample(
        depth.reshape(batch_size * views, 1, height, width),
        grid.reshape(batch_size * views, point_count, 1, 2),
        mode=mode,
        padding_mode="zeros",
        align_corners=True,
    ).reshape(batch_size, views, point_count)
    return sampled, grid


def synthetic_keypoint_sparse_depth_loss(
    pred_depth: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if pred_depth.ndim == 5:
        pred_depth = pred_depth[..., 0]
    xy = batch["keypoint_uv"]
    available = batch["keypoints_available"][:, None, None]
    label_valid = (
        available
        & batch["keypoint_valid"][:, None, :]
        & batch["keypoint_in_frame"]
        & batch["keypoint_surface_visible"]
        & batch["depth_available"][:, :, None]
        & torch.isfinite(xy).all(dim=-1)
    )
    safe_xy = torch.where(label_valid[..., None], xy, torch.zeros_like(xy))
    pred_sampled, grid = sample_depth_at_keypoints(pred_depth, safe_xy, mode="bilinear")
    target_sampled, _ = sample_depth_at_keypoints(batch["depths"], safe_xy, mode="nearest")
    target_mask, _ = sample_depth_at_keypoints(batch["depth_masks"].float(), safe_xy, mode="nearest")
    valid = (
        label_valid
        & torch.isfinite(pred_sampled)
        & torch.isfinite(target_sampled)
        & (pred_sampled > 1e-6)
        & (target_sampled > 1e-6)
        & (target_mask > 0.5)
        & (grid[..., 0].abs() <= 1.0)
        & (grid[..., 1].abs() <= 1.0)
    )
    safe_target = torch.where(valid, target_sampled, torch.ones_like(target_sampled))
    residual = torch.log(pred_sampled.clamp_min(1e-6)) - torch.log(safe_target.clamp_min(1e-6))
    per_point = F.smooth_l1_loss(residual, torch.zeros_like(residual), reduction="none")
    denominator = valid.sum().clamp_min(1).to(dtype=per_point.dtype)
    loss = (per_point * valid.to(dtype=per_point.dtype)).sum() / denominator

    original_scale = batch["scale"].to(dtype=pred_sampled.dtype)[:, None, None]
    absolute_error = (pred_sampled - safe_target).abs() * original_scale
    mae = (
        absolute_error[valid].mean()
        if valid.any()
        else absolute_error.mean() * 0.0
    )
    return {
        "loss_keypoint_sparse_depth": loss,
        "keypoint_sparse_depth_mae": mae,
        "keypoint_sparse_depth_points": valid.sum().to(dtype=pred_sampled.dtype),
    }


def keypoint_gaussian_weight_map(
    xy: torch.Tensor,
    valid_points: torch.Tensor,
    height: int,
    width: int,
    sigma: float,
) -> torch.Tensor:
    if xy.shape[-2] == 0 or not valid_points.any():
        return xy.new_zeros((*xy.shape[:2], height, width))
    sigma = max(float(sigma), 1e-3)
    x_coordinates = torch.arange(width, device=xy.device, dtype=xy.dtype)
    y_coordinates = torch.arange(height, device=xy.device, dtype=xy.dtype)
    x_distance = (x_coordinates[None, None, None, :] - xy[..., 0, None]) / sigma
    y_distance = (y_coordinates[None, None, None, :] - xy[..., 1, None]) / sigma
    valid_float = valid_points.to(dtype=xy.dtype)
    gaussian_x = torch.exp(-0.5 * x_distance.square()) * valid_float[..., None]
    gaussian_y = torch.exp(-0.5 * y_distance.square()) * valid_float[..., None]
    weights = torch.einsum("bvkh,bvkw->bvhw", gaussian_y, gaussian_x)
    return weights.clamp(max=1.0)


def keypoint_region_depth_loss(
    pred_depth: torch.Tensor,
    batch: dict[str, torch.Tensor],
    sigma: float,
) -> dict[str, torch.Tensor]:
    if pred_depth.ndim == 5:
        pred_depth = pred_depth[..., 0]
    height, width = pred_depth.shape[-2:]
    point_valid = (
        batch["keypoints_available"][:, None, None]
        & batch["keypoint_valid"][:, None, :]
        & batch["keypoint_in_frame"]
        & batch["keypoint_surface_visible"]
        & batch["depth_available"][:, :, None]
        & torch.isfinite(batch["keypoint_uv"]).all(dim=-1)
    )
    weights = keypoint_gaussian_weight_map(
        batch["keypoint_uv"],
        point_valid,
        height,
        width,
        sigma,
    )
    dense_valid = (
        batch["depth_masks"]
        & batch["depth_available"][..., None, None]
        & torch.isfinite(batch["depths"])
        & (batch["depths"] > 1e-6)
    )
    residual = log_depth_residual(pred_depth, batch["depths"])
    residual = torch.where(dense_valid, residual, torch.zeros_like(residual))
    loss = weighted_masked_smooth_l1(residual, dense_valid, weights)
    return {
        "loss_keypoint_region_depth": loss,
        "keypoint_region_weight_mass": weights[dense_valid].sum(),
        "keypoint_region_points": point_valid.sum().to(dtype=pred_depth.dtype),
    }


def keypoint_reprojection_loss(
    pred_pose: torch.Tensor,
    batch: dict[str, torch.Tensor],
    image_size_hw: tuple[int, int],
) -> dict[str, torch.Tensor]:
    height, width = image_size_hw
    pred_extrinsics, pred_intrinsics = encoding_to_camera(
        pred_pose,
        image_size_hw,
        build_intrinsics=True,
    )
    world = batch["keypoint_joint_world_xyz"]
    pred_camera = torch.einsum(
        "bvij,bkj->bvki",
        pred_extrinsics[..., :3],
        world,
    ) + pred_extrinsics[..., 3][:, :, None, :]
    homogeneous = torch.einsum("bvij,bvkj->bvki", pred_intrinsics, pred_camera)
    pred_uv = homogeneous[..., :2] / homogeneous[..., 2:3].clamp_min(1e-6)
    target_uv = batch["keypoint_uv"]
    valid = (
        batch["keypoints_available"][:, None, None]
        & batch["keypoint_valid"][:, None, :]
        & batch["keypoint_in_frame"]
        & batch["camera_available"][:, :, None]
        & torch.isfinite(world).all(dim=-1)[:, None, :]
        & torch.isfinite(target_uv).all(dim=-1)
        & torch.isfinite(pred_uv).all(dim=-1)
        & (pred_camera[..., 2] > 1e-4)
    )
    normalizer = pred_uv.new_tensor([max(width, 1), max(height, 1)])
    residual = (pred_uv - target_uv) / normalizer
    per_coordinate = F.smooth_l1_loss(
        residual,
        torch.zeros_like(residual),
        beta=0.01,
        reduction="none",
    )
    per_point = per_coordinate.sum(dim=-1)
    denominator = valid.sum().clamp_min(1).to(dtype=per_point.dtype)
    loss = (per_point * valid.to(dtype=per_point.dtype)).sum() / denominator
    pixel_error = torch.linalg.vector_norm(pred_uv - target_uv, dim=-1)
    mean_pixel_error = (
        pixel_error[valid].mean()
        if valid.any()
        else pixel_error.mean() * 0.0
    )
    return {
        "loss_keypoint_reprojection": loss,
        "keypoint_reprojection_error_px": mean_pixel_error,
        "keypoint_reprojection_points": valid.sum().to(dtype=pred_pose.dtype),
    }


def state_dict_from_payload(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    return payload


def load_state_file(model: VGGTOmega, path: str | Path, label: str, strict: bool) -> Any:
    payload = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict_from_payload(payload), strict=strict)
    if missing or unexpected:
        print(f"{label}_loaded_with_mismatch missing={len(missing)} unexpected={len(unexpected)}")
    print(f"loaded_{label}={path}")
    return payload


def load_checkpoint_chain(
    model: VGGTOmega,
    path: str | Path,
    *,
    stack: tuple[Path, ...] = (),
) -> Any:
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = (PROJECT_ROOT / checkpoint_path).resolve()
    else:
        checkpoint_path = checkpoint_path.resolve()
    if checkpoint_path in stack:
        chain = " -> ".join(str(item) for item in (*stack, checkpoint_path))
        raise ValueError(f"Adapter checkpoint cycle detected: {chain}")

    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and payload.get("save_full") is False:
        base_checkpoint = payload.get("base_checkpoint")
        if not base_checkpoint:
            raise ValueError(f"Adapter checkpoint {checkpoint_path} does not record base_checkpoint")
        load_checkpoint_chain(model, base_checkpoint, stack=(*stack, checkpoint_path))
        missing, unexpected = model.load_state_dict(payload["model"], strict=False)
        if unexpected:
            print(f"adapter_loaded_with_mismatch missing={len(missing)} unexpected={len(unexpected)}")
        print(f"loaded_adapter_checkpoint={checkpoint_path}")
        return payload

    missing, unexpected = model.load_state_dict(state_dict_from_payload(payload), strict=False)
    if missing or unexpected:
        print(f"checkpoint_loaded_with_mismatch missing={len(missing)} unexpected={len(unexpected)}")
    print(f"loaded_checkpoint={checkpoint_path}")
    return payload


def load_model(
    args: argparse.Namespace,
    device: torch.device,
    keypoint_names: list[str] | None = None,
) -> VGGTOmega:
    keypoint_names = keypoint_names or []
    model = VGGTOmega(
        enable_camera=args.forward_camera_head,
        enable_depth=args.forward_depth_head,
        enable_keypoints=args.enable_keypoint_head,
        enable_multi_instance_pose=args.enable_multi_instance_pose,
        num_instance_queries=args.num_instance_queries,
        num_keypoints=len(keypoint_names),
    ).to(device)
    if args.checkpoint:
        load_checkpoint_chain(model, args.checkpoint)
    if args.adapter_checkpoint:
        load_state_file(model, args.adapter_checkpoint, "adapter_checkpoint", strict=False)
    return model


def selected_keypoint_mask(
    names: list[str],
    subset: str,
    device: torch.device,
) -> torch.Tensor:
    subset = subset.strip()
    if not subset or subset.lower() == "all":
        return torch.ones(len(names), dtype=torch.bool, device=device)
    if subset.lower() == "coarse":
        coarse_tokens = (
            "Fingers1_",
            "Wrist_",
            "Elbow",
            "Scapula",
            "Shoulder",
            "HeadEnd",
            "Snout",
            "Nose",
        )
        return torch.tensor(
            [any(token in name for token in coarse_tokens) for name in names],
            dtype=torch.bool,
            device=device,
        )
    requested = set(parse_name_list(subset))
    unknown = sorted(requested.difference(names))
    if unknown:
        raise ValueError(f"Unknown --keypoint-head-subset names: {unknown}")
    return torch.tensor(
        [name in requested for name in names],
        dtype=torch.bool,
        device=device,
    )


def tail_curve_config_from_args(args: argparse.Namespace) -> TailCurveLossConfig:
    """Translate CLI options into the trainer-independent curve configuration."""
    return TailCurveLossConfig(
        weight=args.keypoint_head_tail_curve_weight,
        curve_2d_weight=args.keypoint_head_tail_curve_2d_weight,
        curve_3d_weight=args.keypoint_head_tail_curve_3d_weight,
        partial_2d_weight=args.keypoint_head_tail_partial_2d_weight,
        base_anchor_weight=args.keypoint_head_tail_base_anchor_weight,
        tip_anchor_weight=args.keypoint_head_tail_tip_anchor_weight,
        length_weight=args.keypoint_head_tail_length_weight,
        curvature_weight=args.keypoint_head_tail_curvature_weight,
        interior_point_weight=args.keypoint_head_tail_interior_point_weight,
        samples=args.keypoint_head_tail_curve_samples,
        smoothness=args.keypoint_head_tail_smoothness,
        min_points=args.keypoint_head_tail_min_points,
        distance=args.keypoint_head_tail_curve_distance,
        soft_dtw_gamma=args.keypoint_head_tail_soft_dtw_gamma,
        partial_temperature=args.keypoint_head_tail_partial_temperature,
        robust_views=args.keypoint_head_tail_robust_views,
        robust_delta=args.keypoint_head_tail_robust_delta,
        detach_fit_weights=args.keypoint_head_tail_detach_fit_weights,
    )


def synthetic_keypoint_head_loss(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    names = list(args.resolved_keypoint_names)
    point_valid = batch["keypoint_valid"] & selected_keypoint_mask(
        names,
        args.keypoint_head_subset,
        batch["images"].device,
    )[None, :]
    view_valid = batch["keypoint_in_frame"] & point_valid[:, None, :]
    camera_available = batch["camera_available"]
    return keypoint_head_loss(
        predictions,
        batch,
        target_xy=batch["keypoint_uv"],
        target_world=batch["keypoint_joint_world_xyz"],
        point_valid=point_valid,
        view_valid=view_valid,
        visibility_target=batch["keypoint_surface_visible"],
        camera_available=camera_available,
        heatmap_sigma=args.keypoint_head_heatmap_sigma,
        heatmap_weight=args.keypoint_head_heatmap_weight,
        visibility_weight=args.keypoint_head_visibility_weight,
        center3d_weight=args.keypoint_head_center3d_weight,
        predicted_rig_weight=args.keypoint_head_predicted_rig_weight,
        fixed_rig_weight=args.keypoint_head_fixed_rig_weight,
        keypoint_names=names,
        surface_visible=batch["keypoint_surface_visible"],
        tail_curve_config=tail_curve_config_from_args(args),
    )


def remap_real_keypoint_targets(
    batch: dict[str, Any],
    real_names: list[str],
    canonical_names: list[str],
) -> dict[str, torch.Tensor]:
    images = batch["images"]
    batch_size, views = images.shape[:2]
    keypoint_count = len(canonical_names)
    target_xy = images.new_zeros((batch_size, views, keypoint_count, 2))
    target_world = images.new_zeros((batch_size, keypoint_count, 3))
    point_valid = torch.zeros(
        (batch_size, keypoint_count),
        dtype=torch.bool,
        device=images.device,
    )
    view_valid = torch.zeros(
        (batch_size, views, keypoint_count),
        dtype=torch.bool,
        device=images.device,
    )
    canonical_index = {name: index for index, name in enumerate(canonical_names)}
    for real_index, real_name in enumerate(real_names):
        canonical_name = REAL_TO_BLENDER_KEYPOINT.get(real_name)
        canonical_idx = canonical_index.get(canonical_name) if canonical_name else None
        if canonical_idx is None:
            continue
        target_xy[:, :, canonical_idx] = batch["geometry_xy"][:, :, real_index]
        target_world[:, canonical_idx] = batch["geometry_world"][:, real_index]
        point_valid[:, canonical_idx] = batch["geometry_valid"][:, real_index]
        view_valid[:, :, canonical_idx] = batch["geometry_view_valid"][:, :, real_index]
    return {
        "target_xy": target_xy,
        "target_world": target_world,
        "point_valid": point_valid,
        "view_valid": view_valid,
    }


def real_keypoint_head_loss(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    canonical_names = list(args.resolved_keypoint_names)
    remapped = remap_real_keypoint_targets(
        batch,
        list(args.resolved_real_geometry_keypoint_names),
        canonical_names,
    )
    remapped["point_valid"] &= selected_keypoint_mask(
        canonical_names,
        args.keypoint_head_subset,
        batch["images"].device,
    )[None, :]
    remapped["view_valid"] &= remapped["point_valid"][:, None, :]
    camera_available = torch.ones(
        batch["images"].shape[:2],
        dtype=torch.bool,
        device=batch["images"].device,
    )
    return keypoint_head_loss(
        predictions,
        batch,
        target_xy=remapped["target_xy"],
        target_world=remapped["target_world"],
        point_valid=remapped["point_valid"],
        view_valid=remapped["view_valid"],
        visibility_target=remapped["view_valid"],
        camera_available=camera_available,
        heatmap_sigma=args.keypoint_head_heatmap_sigma,
        heatmap_weight=args.keypoint_head_heatmap_weight,
        visibility_weight=args.keypoint_head_visibility_weight,
        center3d_weight=args.keypoint_head_center3d_weight,
        predicted_rig_weight=args.keypoint_head_predicted_rig_weight,
        fixed_rig_weight=args.keypoint_head_fixed_rig_weight,
        keypoint_names=canonical_names,
        surface_visible=remapped["view_valid"],
        tail_curve_config=tail_curve_config_from_args(args),
    )


def make_optimizer(model: VGGTOmega, trainable_names: list[str], args: argparse.Namespace) -> torch.optim.Optimizer:
    trainable = dict(model.named_parameters())
    head_params, agg_params = [], []
    for name in trainable_names:
        param = trainable[name]
        if name.startswith("aggregator."):
            agg_params.append(param)
        else:
            head_params.append(param)
    groups = []
    if head_params:
        groups.append({"params": head_params, "lr": args.lr})
    if agg_params:
        groups.append({"params": agg_params, "lr": args.lr * args.aggregator_lr_mult})
    return torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    args: argparse.Namespace,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = max(int(total_steps), 1)
    warmup_steps = max(int(round(total_steps * args.warmup_ratio)), 1)

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return max(float(step + 1) / warmup_steps, 1e-8)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def compute_loss(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    images = batch["images"]
    logs: dict[str, torch.Tensor] = {}
    total = images.mean() * 0.0
    camera_available = batch.get(
        "camera_available",
        torch.ones(batch["extrinsics"].shape[:2], dtype=torch.bool, device=images.device),
    )
    depth_available = batch.get(
        "depth_available",
        torch.ones(batch["depths"].shape[:2], dtype=torch.bool, device=images.device),
    )
    active_loss_terms = 0
    logs["camera_supervised_views"] = camera_available.sum().to(dtype=images.dtype)
    logs["depth_supervised_views"] = depth_available.sum().to(dtype=images.dtype)
    keypoints_available = batch.get(
        "keypoints_available",
        torch.zeros(images.shape[0], dtype=torch.bool, device=images.device),
    )
    instance_supervision_available = batch.get(
        "instance_supervision_available",
        batch["instance_valid"].flatten(1).any(dim=1)
        if "instance_valid" in batch
        else keypoints_available,
    )
    logs["keypoint_supervised_scenes"] = keypoints_available.sum().to(dtype=images.dtype)
    logs["instance_supervised_scenes"] = instance_supervision_available.sum().to(
        dtype=images.dtype
    )
    if "keypoint_in_frame" in batch:
        logs["keypoint_in_frame_points"] = batch["keypoint_in_frame"].sum().to(dtype=images.dtype)
        logs["keypoint_visible_points"] = batch["keypoint_surface_visible"].sum().to(dtype=images.dtype)

    if "pose_enc" in predictions and args.camera_weight > 0 and camera_available.any():
        target_pose = extri_intri_to_pose_encoding(
            batch["extrinsics"],
            batch["intrinsics"],
            images.shape[-2:],
        )
        cam_losses = camera_loss(
            predictions["pose_enc"],
            target_pose,
            camera_available,
            loss_type=args.camera_loss_type,
        )
        logs.update(cam_losses)
        total = total + args.camera_weight * (
            args.translation_weight * cam_losses["loss_cam_t"]
            + args.rotation_weight * cam_losses["loss_cam_q"]
            + args.fov_weight * cam_losses["loss_cam_fov"]
        )
        active_loss_terms += 1
    if (
        "pose_enc" in predictions
        and args.relative_rig_weight > 0
        and (camera_available.sum(dim=1) >= 2).any()
    ):
        relative_logs = relative_rig_camera_loss(
            predictions["pose_enc"],
            batch,
            rotation_weight=args.relative_rig_rotation_weight,
            center_weight=args.relative_rig_center_weight,
            baseline_weight=args.relative_rig_baseline_weight,
            focal_weight=args.relative_rig_focal_weight,
        )
        logs.update(relative_logs)
        total = total + args.relative_rig_weight * relative_logs["loss_relative_rig"]
        active_loss_terms += 1
    if "depth" in predictions and args.depth_weight > 0 and depth_available.any():
        depth_mask = batch["depth_masks"] & depth_available[..., None, None]
        if args.depth_loss_mode == "official":
            pixel_weights = geometry_pixel_weights(
                depth_mask,
                batch["body_masks"],
                batch["semantic_masks"],
                args,
            )
            depth_logs = official_depth_loss(
                predictions["depth"],
                batch["depths"],
                depth_mask,
                pixel_weights=pixel_weights,
                gradient_weight=args.depth_gradient_weight,
                confidence=(
                    predictions.get("depth_conf")
                    if args.use_confidence_loss
                    else None
                ),
                confidence_alpha=args.confidence_alpha,
            )
            depth_logs["loss_depth"] = depth_logs["loss_depth_official"]
        else:
            depth_logs = depth_loss(
                predictions["depth"],
                batch["depths"],
                depth_mask,
                batch["body_masks"],
                batch["semantic_masks"],
                args,
            )
        logs.update(depth_logs)
        total = total + args.depth_weight * depth_logs["loss_depth"]
        active_loss_terms += 1
    if (
        "depth" in predictions
        and "pose_enc" in predictions
        and args.point_weight > 0
        and depth_available.any()
        and camera_available.any()
    ):
        point_valid = (
            batch["depth_masks"]
            & depth_available[..., None, None]
            & camera_available[..., None, None]
        )
        point_logs = official_point_loss(
            predictions["depth"],
            predictions["pose_enc"],
            batch["depths"],
            batch["extrinsics"],
            batch["intrinsics"],
            point_valid,
            images.shape[-2:],
            stride=args.point_loss_stride,
            gradient_weight=args.point_gradient_weight,
            confidence=(
                predictions.get("depth_conf")
                if args.use_confidence_loss
                else None
            ),
            confidence_alpha=args.confidence_alpha,
        )
        logs.update(point_logs)
        total = total + args.point_weight * point_logs["loss_point"]
        active_loss_terms += 1
    if (
        "patch_tokens" in predictions
        and args.match_weight > 0
        and depth_available.any()
        and camera_available.any()
    ):
        match_logs = token_matching_loss(
            predictions["patch_tokens"],
            batch,
            patch_size=16,
            max_pairs=args.match_max_pairs,
            depth_tolerance=args.match_depth_tolerance,
            border=args.match_border,
            temperature=args.match_temperature,
        )
        logs.update(match_logs)
        if match_logs["loss_match"].requires_grad:
            total = total + args.match_weight * match_logs["loss_match"]
            active_loss_terms += 1
    if "depth" in predictions and keypoints_available.any():
        sparse_logs = synthetic_keypoint_sparse_depth_loss(predictions["depth"], batch)
        logs.update(sparse_logs)
        if args.keypoint_sparse_depth_weight > 0:
            total = total + args.keypoint_sparse_depth_weight * sparse_logs["loss_keypoint_sparse_depth"]
            active_loss_terms += 1
        if args.keypoint_region_weight > 0:
            region_logs = keypoint_region_depth_loss(
                predictions["depth"],
                batch,
                args.keypoint_region_sigma,
            )
            logs.update(region_logs)
            total = total + args.keypoint_region_weight * region_logs["loss_keypoint_region_depth"]
            active_loss_terms += 1
    if "pose_enc" in predictions and keypoints_available.any():
        reprojection_logs = keypoint_reprojection_loss(
            predictions["pose_enc"],
            batch,
            images.shape[-2:],
        )
        logs.update(reprojection_logs)
        if args.keypoint_reprojection_weight > 0:
            total = total + (
                args.keypoint_reprojection_weight
                * reprojection_logs["loss_keypoint_reprojection"]
            )
            active_loss_terms += 1
    if (
        "keypoint_heatmap_logits" in predictions
        and "pose_enc" in predictions
        and keypoints_available.any()
        and args.keypoint_head_weight > 0
    ):
        head_loss, head_logs = synthetic_keypoint_head_loss(
            predictions,
            batch,
            args,
        )
        logs.update(head_logs)
        total = total + args.keypoint_head_weight * head_loss
        active_loss_terms += 1
    if (
        "instance_objectness_logits" in predictions
        and instance_supervision_available.any()
        and args.instance_pose_weight > 0
    ):
        instance_targets = instance_targets_from_batch(batch)
        instance_total, instance_logs = instance_pose_loss(
            predictions,
            instance_targets,
            InstancePoseLossWeights(
                objectness=args.instance_objectness_weight,
                box_l1=args.instance_box_l1_weight,
                box_giou=args.instance_box_giou_weight,
                mask_bce=args.instance_mask_bce_weight,
                mask_dice=args.instance_mask_dice_weight,
                keypoint_l1=args.instance_keypoint_l1_weight,
                keypoint_visibility=args.instance_keypoint_visibility_weight,
                local_heatmap=args.instance_local_heatmap_weight,
                reid=args.instance_reid_weight,
                keypoint_3d=args.instance_keypoint_3d_weight,
                reprojection=args.instance_reprojection_weight,
                no_object=args.instance_no_object_weight,
            ),
            image_size=images.shape[-2:],
            world_scale=args.instance_world_scale,
        )
        logs.update(instance_logs)
        total = total + args.instance_pose_weight * instance_total
        active_loss_terms += 1
    if "depth_conf" in predictions:
        logs["depth_conf_mean"] = predictions["depth_conf"].mean()
        logs["depth_conf_floor_fraction"] = (
            predictions["depth_conf"] < 1.01
        ).float().mean()
    logs["active_loss_terms"] = torch.tensor(active_loss_terms, dtype=images.dtype, device=images.device)
    logs["loss"] = total
    return total, logs


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


@torch.no_grad()
def evaluate(
    model: VGGTOmega,
    loader: DataLoader | None,
    device: torch.device,
    args: argparse.Namespace,
    frozen_aggregator: bool,
    max_batches: int,
) -> dict[str, float]:
    if loader is None:
        return {}
    was_training = model.training
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        batch = to_device(batch, device)
        predictions = forward_for_training(model, batch["images"], frozen_aggregator)
        _, logs = compute_loss(predictions, batch, args)
        for key, value in logs.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        count += 1
    if was_training:
        model.train()
    if count == 0:
        return {}
    return {f"val_{key}": value / count for key, value in totals.items()}


@torch.no_grad()
def evaluate_real_pose(
    model: VGGTOmega,
    loader: DataLoader | None,
    device: torch.device,
    frozen_aggregator: bool,
    max_batches: int,
) -> dict[str, float]:
    if loader is None:
        return {}
    was_training = model.training
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        batch = to_device(batch, device)
        predictions = forward_for_training(model, batch["images"], frozen_aggregator)
        if "pose_enc" not in predictions:
            continue
        target_pose = extri_intri_to_pose_encoding(batch["extrinsics"], batch["intrinsics"], batch["images"].shape[-2:])
        logs = camera_loss(predictions["pose_enc"], target_pose)
        combined = logs["loss_cam_t"] + logs["loss_cam_q"] + logs["loss_cam_fov"]
        logs["loss_camera"] = combined
        for key, value in logs.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        count += 1
    if was_training:
        model.train()
    if count == 0:
        return {}
    return {f"real_val_{key}": value / count for key, value in totals.items()}


@torch.no_grad()
def evaluate_real_palm_depth(
    model: VGGTOmega,
    loader: DataLoader | None,
    device: torch.device,
    frozen_aggregator: bool,
    max_batches: int,
) -> dict[str, float]:
    if loader is None:
        return {}
    was_training = model.training
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        batch = to_device(batch, device)
        predictions = forward_for_training(model, batch["images"], frozen_aggregator)
        if "depth" not in predictions:
            continue
        logs = sparse_depth_loss(predictions["depth"], batch)
        for key, value in logs.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        count += 1
    if was_training:
        model.train()
    if count == 0:
        return {}
    return {f"real_depth_val_{key}": value / count for key, value in totals.items()}


def compute_real_geometry_loss(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, Any],
    args: argparse.Namespace,
    *,
    use_temporal_filter: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    total = batch["images"].mean() * 0.0
    logs: dict[str, torch.Tensor] = {}
    rig_frame_mask = None
    if (
        use_temporal_filter
        and "rig_frame_confidence" in batch
        and "rig_frame_coverage" in batch
    ):
        rig_frame_mask = (
            batch["rig_frame_confidence"]
            >= float(getattr(args, "real_geometry_rig_min_confidence", 0.8))
        ) & (
            batch["rig_frame_coverage"]
            >= float(getattr(args, "real_geometry_rig_min_coverage", 0.35))
        )
        logs["rig_frame_confidence_mean"] = batch["rig_frame_confidence"].mean()
        logs["rig_frame_coverage_mean"] = batch["rig_frame_coverage"].mean()
        if "depth_conf" in predictions:
            model_confidence = predictions["depth_conf"].float()
            model_confidence = model_confidence.reshape(
                model_confidence.shape[0],
                -1,
            ).mean(dim=-1)
            logs["rig_frame_model_confidence_mean"] = model_confidence.mean()
            keep_fraction = float(
                getattr(args, "real_geometry_rig_keep_fraction", 1.0)
            )
            candidate_indices = torch.nonzero(
                rig_frame_mask,
                as_tuple=False,
            ).flatten()
            if keep_fraction < 1.0 and candidate_indices.numel() > 2:
                keep_count = max(
                    2,
                    min(
                        candidate_indices.numel(),
                        int(round(candidate_indices.numel() * keep_fraction)),
                    ),
                )
                top_candidates = candidate_indices[
                    torch.topk(
                        model_confidence[candidate_indices],
                        k=keep_count,
                    ).indices
                ]
                model_mask = torch.zeros_like(rig_frame_mask)
                model_mask[top_candidates] = True
                rig_frame_mask = rig_frame_mask & model_mask
        if "pose_enc" in predictions:
            rig_rotation_error, rig_center_error = relative_rig_frame_errors(
                predictions["pose_enc"],
                batch,
            )
            logs["rig_frame_rotation_error_deg_mean"] = rig_rotation_error.mean()
            logs["rig_frame_center_error_mean"] = rig_center_error.mean()
            target_consistent = (
                rig_rotation_error
                <= float(
                    getattr(
                        args,
                        "real_geometry_rig_max_rotation_error_deg",
                        30.0,
                    )
                )
            ) & (
                rig_center_error
                <= float(
                    getattr(
                        args,
                        "real_geometry_rig_max_center_error",
                        0.4,
                    )
                )
            )
            logs["rig_frame_target_outlier_count"] = (
                rig_frame_mask & ~target_consistent
            ).sum().to(dtype=total.dtype)
            rig_frame_mask = rig_frame_mask & target_consistent
        logs["rig_frame_high_confidence_count"] = rig_frame_mask.sum().to(
            dtype=total.dtype
        )
        logs["rig_frame_low_confidence_count"] = (~rig_frame_mask).sum().to(
            dtype=total.dtype
        )
    if (
        args.real_geometry_weight > 0
        and "depth" in predictions
        and "pose_enc" in predictions
    ):
        geometry_logs = multiview_keypoint_geometry_loss(
            predictions["depth"],
            predictions["pose_enc"],
            batch,
            predicted_rig_weight=args.real_geometry_predicted_rig_weight,
            fixed_rig_weight=args.real_geometry_fixed_rig_weight,
            pairwise_weight=args.real_geometry_pairwise_weight,
            reprojection_weight=args.real_geometry_reprojection_weight,
        )
        logs.update(geometry_logs)
        total = total + args.real_geometry_weight * geometry_logs["loss_real_geometry"]
    if args.real_geometry_camera_weight > 0 and "pose_enc" in predictions:
        camera_total, camera_logs = weighted_camera_loss(
            predictions["pose_enc"],
            batch,
            args.real_camera_translation_weight,
            args.real_camera_rotation_weight,
            args.real_camera_fov_weight,
            loss_type=args.camera_loss_type,
        )
        logs.update(
            {
                f"real_geometry_camera_{key}": value
                for key, value in camera_logs.items()
            }
        )
        logs["loss_real_geometry_camera"] = camera_total
        total = total + args.real_geometry_camera_weight * camera_total
    if (
        args.real_geometry_relative_rig_weight > 0
        and "pose_enc" in predictions
    ):
        relative_logs = relative_rig_camera_loss(
            predictions["pose_enc"],
            batch,
            rotation_weight=args.relative_rig_rotation_weight,
            center_weight=args.relative_rig_center_weight,
            baseline_weight=args.relative_rig_baseline_weight,
            focal_weight=args.relative_rig_focal_weight,
            frame_mask=rig_frame_mask,
        )
        logs.update(
            {
                f"real_geometry_{key}": value
                for key, value in relative_logs.items()
            }
        )
        total = (
            total
            + args.real_geometry_relative_rig_weight
            * relative_logs["loss_relative_rig"]
        )
    if (
        use_temporal_filter
        and getattr(args, "real_geometry_shared_rig_weight", 0.0) > 0
        and "pose_enc" in predictions
    ):
        temporal_logs = shared_rig_temporal_loss(
            predictions["pose_enc"],
            batch,
            frame_mask=rig_frame_mask,
            outlier_rotation_deg=getattr(
                args,
                "real_geometry_shared_rig_outlier_rotation_deg",
                15.0,
            ),
            outlier_center_error=getattr(
                args,
                "real_geometry_shared_rig_outlier_center",
                0.35,
            ),
            rotation_weight=getattr(
                args,
                "real_geometry_shared_rig_rotation_weight",
                1.0,
            ),
            center_weight=getattr(
                args,
                "real_geometry_shared_rig_center_weight",
                1.0,
            ),
            baseline_weight=getattr(
                args,
                "real_geometry_shared_rig_baseline_weight",
                1.0,
            ),
            focal_weight=getattr(
                args,
                "real_geometry_shared_rig_focal_weight",
                0.1,
            ),
        )
        logs.update(temporal_logs)
        total = (
            total
            + float(args.real_geometry_shared_rig_weight)
            * temporal_logs["loss_shared_rig_temporal"]
        )
    if (
        "keypoint_heatmap_logits" in predictions
        and args.real_keypoint_head_weight > 0
    ):
        head_total, head_logs = real_keypoint_head_loss(
            predictions,
            batch,
            args,
        )
        logs.update(
            {
                f"real_{key}": value
                for key, value in head_logs.items()
            }
        )
        total = total + args.real_keypoint_head_weight * head_total
        scale = batch["scale"].to(dtype=head_total.dtype).mean()
        logs["real_keypoint_head_fixed_mae_mm"] = (
            head_logs["keypoint_head_fixed_mae"] * scale
        )
        logs["real_keypoint_head_predicted_mae_mm"] = (
            head_logs["keypoint_head_predicted_mae"] * scale
        )
    logs["loss_real_geometry_total"] = total
    return total, logs


@torch.no_grad()
def evaluate_real_geometry(
    model: VGGTOmega,
    loader: DataLoader | None,
    device: torch.device,
    args: argparse.Namespace,
    frozen_aggregator: bool,
    max_batches: int,
) -> dict[str, float]:
    if loader is None:
        return {}
    was_training = model.training
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        batch = to_device(batch, device)
        predictions = forward_for_training(model, batch["images"], frozen_aggregator)
        _, logs = compute_real_geometry_loss(predictions, batch, args)
        for key, value in logs.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        count += 1
    if was_training:
        model.train()
    if count == 0:
        return {}
    return {
        f"real_geometry_val_{key}": value / count
        for key, value in totals.items()
    }


def normalized_real_geometry_score(
    row: dict[str, Any],
    baseline: dict[str, float],
) -> float | None:
    metric_keys = (
        "real_geometry_val_geometry_predicted_mae_mm",
        "real_geometry_val_geometry_fixed_mae_mm",
        "real_geometry_val_geometry_palm_predicted_mae_mm",
        "real_geometry_val_geometry_palm_fixed_mae_mm",
        "real_geometry_val_geometry_pairwise_spread_mm",
        "real_geometry_val_geometry_reprojection_error_px",
    )
    ratios = []
    for key in metric_keys:
        current = row.get(key)
        reference = baseline.get(key)
        if current is None or reference is None:
            continue
        if not np.isfinite(current) or not np.isfinite(reference) or reference <= 1e-12:
            continue
        ratios.append(float(current) / float(reference))
    if not ratios:
        return None
    return float(np.mean(ratios))


def append_log(path: Path | None, row: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        existing_rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    new_keys = [key for key in row.keys() if key not in fieldnames]
    if not new_keys:
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)
        return

    fieldnames.extend(new_keys)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerow(row)


def save_checkpoint_to(
    output: Path,
    args: argparse.Namespace,
    model: VGGTOmega,
    trainable_names: list[str],
    step: int,
    epoch: int | None = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    full_state = model.state_dict()
    if args.save_full:
        state_dict = full_state
    else:
        trainable_set = set(trainable_names)
        state_dict = {name: tensor.detach().cpu() for name, tensor in full_state.items() if name in trainable_set}
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "model": state_dict,
            "trainable_param_names": trainable_names,
            "save_full": args.save_full,
            "base_checkpoint": args.checkpoint,
            "args": vars(args),
        },
        output,
    )
    print(f"saved={output}")


def save_checkpoint(args: argparse.Namespace, model: VGGTOmega, trainable_names: list[str], step: int, epoch: int | None = None) -> None:
    if not args.output:
        return
    save_checkpoint_to(Path(args.output), args, model, trainable_names, step, epoch=epoch)


def save_epoch_checkpoint(args: argparse.Namespace, model: VGGTOmega, trainable_names: list[str], step: int, epoch: int) -> None:
    if not args.checkpoint_dir:
        return
    output_base = Path(args.output)
    stem = output_base.stem if output_base.name else "vggt_omega_adapter"
    checkpoint_dir = Path(args.checkpoint_dir)
    path = checkpoint_dir / f"{stem}_epoch{epoch + 1:03d}_step{step:06d}.pt"
    save_checkpoint_to(path, args, model, trainable_names, step, epoch=epoch + 1)


def save_step_checkpoint(args: argparse.Namespace, model: VGGTOmega, trainable_names: list[str], step: int, epoch: int) -> None:
    if not args.checkpoint_dir:
        return
    output_base = Path(args.output)
    stem = output_base.stem if output_base.name else "vggt_omega_adapter"
    path = Path(args.checkpoint_dir) / f"{stem}_step{step:06d}.pt"
    save_checkpoint_to(path, args, model, trainable_names, step, epoch=epoch)


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    image_width, image_height = training_image_shape(args)
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Training image dimensions must be positive")
    if image_width % 16 or image_height % 16:
        raise ValueError(
            f"Training image dimensions must be divisible by patch size 16, got "
            f"{image_width}x{image_height}"
        )

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    train_loader = make_loader(args, "train")
    val_loader = None
    if args.val_scenes > 0:
        try:
            val_loader = make_loader(args, "val")
        except ValueError as exc:
            print(f"val_loader_skipped reason={exc}")
    real_val_loader = make_real_val_loader(args)
    real_camera_train_loader = make_real_camera_train_loader(args)
    real_train_loader = make_real_palm_loader(args, "train")
    real_depth_val_loader = make_real_palm_loader(args, "val")
    real_geometry_train_loader = make_real_geometry_loader(args, "train")
    real_geometry_val_loader = make_real_geometry_loader(args, "val")
    args.resolved_keypoint_names = list(train_loader.dataset.keypoint_names)
    if (
        args.enable_keypoint_head or args.enable_multi_instance_pose
    ) and not args.resolved_keypoint_names:
        raise ValueError(
            "A keypoint or instance-pose head requires at least one keypoints.npz file "
            "in the training data"
        )
    real_geometry_dataset = (
        real_geometry_train_loader.dataset
        if real_geometry_train_loader is not None
        else real_geometry_val_loader.dataset
        if real_geometry_val_loader is not None
        else None
    )
    args.resolved_real_geometry_keypoint_names = (
        list(real_geometry_dataset.keypoints)
        if real_geometry_dataset is not None
        else []
    )

    first = next(iter(train_loader))
    scene_groups = Counter(scene.root.parent.name for scene in train_loader.dataset.scenes)
    all_frames = [
        frame
        for scene in train_loader.dataset.scenes
        for frame in scene.frames.values()
    ]
    labeled_cameras = sum(frame_has_camera(frame) for frame in all_frames)
    labeled_depths = sum(frame.depth_path is not None for frame in all_frames)
    print(
        "dataset_ok",
        f"train_scenes={len(train_loader.dataset.scenes)}",
        "scene_groups="
        + ",".join(f"{name}:{count}" for name, count in sorted(scene_groups.items())),
        f"samples={len(train_loader.dataset)}",
        f"views={len(all_frames)}",
        f"camera_labels={labeled_cameras}",
        f"depth_labels={labeled_depths}",
        f"images={tuple(first['images'].shape)}",
        f"depths={tuple(first['depths'].shape)}",
        f"camera_supervised_views={int(first['camera_available'].sum())}",
        f"depth_supervised_views={int(first['depth_available'].sum())}",
        f"valid_depth_px={int(((first['depths'] > 1e-6) & first['depth_masks']).sum())}",
        f"body_mask_px={int(first['body_masks'].sum())}",
        f"semantic_labels={sorted(int(v) for v in torch.unique(first['semantic_masks']).tolist())[:12]}",
        f"keypoint_names={len(train_loader.dataset.keypoint_names)}",
        f"keypoint_labeled={bool(first['keypoints_available'][0])}",
        f"keypoint_visible={int(first['keypoint_surface_visible'].sum())}",
        f"scale={float(first['scale'][0]):.6g}",
        f"scale_source={first['scale_source'][0]}",
        f"scene={first['scene'][0]}",
        f"view_ids={first['view_ids'][0]}",
    )
    if args.dry_run == "data":
        return

    model = load_model(args, device, args.resolved_keypoint_names)
    trainable_names = configure_trainable(model, args)
    trainable_params = sum(param.numel() for _, param in model.named_parameters() if param.requires_grad)
    frozen_aggregator = aggregator_is_frozen(model)
    print(
        f"train_mode={args.train_mode}",
        f"trainable_params={trainable_params:,}",
        f"trainable_tensors={len(trainable_names)}",
        f"frozen_aggregator={frozen_aggregator}",
        f"scale_mode={args.scale_mode}",
        f"canonicalize_camera={args.canonicalize_camera}",
    )
    if args.dry_run == "model":
        model.train()
        batch = to_device(first, device)
        predictions = forward_for_training(model, batch["images"], frozen_aggregator)
        loss, logs = compute_loss(predictions, batch, args)
        if real_camera_train_loader is not None and args.real_camera_train_weight > 0:
            real_camera_batch = to_device(next(iter(real_camera_train_loader)), device)
            real_camera_predictions = forward_for_training(
                model,
                real_camera_batch["images"],
                frozen_aggregator,
            )
            real_camera_loss_value, real_camera_logs = weighted_camera_loss(
                real_camera_predictions["pose_enc"],
                real_camera_batch,
                args.real_camera_translation_weight,
                args.real_camera_rotation_weight,
                args.real_camera_fov_weight,
            )
            loss = loss + args.real_camera_train_weight * real_camera_loss_value
            logs.update(
                {f"real_camera_train_{key}": value for key, value in real_camera_logs.items()}
            )
        if real_train_loader is not None:
            real_batch = to_device(next(iter(real_train_loader)), device)
            real_predictions = forward_for_training(model, real_batch["images"], frozen_aggregator)
            real_logs = sparse_depth_loss(real_predictions["depth"], real_batch)
            loss = loss + args.real_sparse_weight * real_logs["loss_sparse_depth"]
            logs.update({f"real_train_{key}": value for key, value in real_logs.items()})
        if (
            real_geometry_train_loader is not None
            and (
                args.real_geometry_weight > 0
                or args.real_geometry_camera_weight > 0
                or args.real_keypoint_head_weight > 0
            )
        ):
            geometry_batch = to_device(next(iter(real_geometry_train_loader)), device)
            geometry_predictions = forward_for_training(
                model,
                geometry_batch["images"],
                frozen_aggregator,
            )
            geometry_loss, geometry_logs = compute_real_geometry_loss(
                geometry_predictions,
                geometry_batch,
                args,
            )
            loss = loss + geometry_loss
            logs.update(
                {
                    f"real_geometry_train_{key}": value
                    for key, value in geometry_logs.items()
                }
            )
        if not loss.requires_grad:
            raise ValueError(
                "The selected train mode has no trainable prediction branch for the labels in the first batch"
            )
        loss.backward()
        print("model_dry_run_ok", " ".join(f"{k}={float(v.detach().cpu()):.4f}" for k, v in logs.items()))
        return

    optimizer = make_optimizer(model, trainable_names, args)
    total_steps = (
        args.max_steps
        if args.max_steps > 0
        else args.epochs * len(train_loader)
    )
    scheduler = (
        make_scheduler(optimizer, total_steps, args)
        if args.lr_schedule == "warmup_cosine"
        else None
    )
    model.train()
    log_path = Path(args.log_csv) if args.log_csv else None
    baseline = evaluate_real_pose(model, real_val_loader, device, frozen_aggregator, args.real_val_batches)
    baseline.update(
        evaluate_real_palm_depth(
            model,
            real_depth_val_loader,
            device,
            frozen_aggregator,
            args.real_depth_val_batches,
        )
    )
    baseline.update(
        evaluate_real_geometry(
            model,
            real_geometry_val_loader,
            device,
            args,
            frozen_aggregator,
            args.real_geometry_val_batches,
        )
    )
    best_real_palm_mae = baseline.get("real_depth_val_sparse_palm_mae_mm", float("inf"))
    baseline_real_camera = baseline.get("real_val_loss_camera", float("inf"))
    baseline_real_palm = baseline.get("real_depth_val_sparse_palm_mae_mm", float("inf"))
    best_real_joint_score = float("inf")
    best_real_geometry_score = float("inf")
    best_real_keypoint_head_mae = float("inf")
    best_real_relative_rig_loss = baseline.get(
        "real_geometry_val_real_geometry_loss_relative_rig",
        float("inf"),
    )
    if baseline:
        row = {"step": -1, "epoch": -1, "scene": "real_val_baseline", "view_ids": args.real_val_cameras, "scale": 0.0}
        row.update(baseline)
        append_log(log_path, row)
        print(" ".join(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}" for key, value in row.items()), flush=True)

    step = 0
    real_camera_train_iterator = (
        iter(real_camera_train_loader) if real_camera_train_loader is not None else None
    )
    real_train_iterator = iter(real_train_loader) if real_train_loader is not None else None
    real_geometry_train_iterator = (
        iter(real_geometry_train_loader)
        if real_geometry_train_loader is not None
        else None
    )
    for epoch in range(args.epochs):
        for batch in train_loader:
            batch = to_device(batch, device)
            predictions = forward_for_training(model, batch["images"], frozen_aggregator)
            loss, logs = compute_loss(predictions, batch, args)
            if (
                real_camera_train_loader is not None
                and args.real_camera_train_weight > 0
                and step % max(1, args.real_camera_train_every) == 0
            ):
                try:
                    real_camera_batch = next(real_camera_train_iterator)
                except StopIteration:
                    real_camera_train_iterator = iter(real_camera_train_loader)
                    real_camera_batch = next(real_camera_train_iterator)
                real_camera_batch = to_device(real_camera_batch, device)
                real_camera_predictions = forward_for_training(
                    model,
                    real_camera_batch["images"],
                    frozen_aggregator,
                )
                real_camera_loss_value, real_camera_logs = weighted_camera_loss(
                    real_camera_predictions["pose_enc"],
                    real_camera_batch,
                    args.real_camera_translation_weight,
                    args.real_camera_rotation_weight,
                    args.real_camera_fov_weight,
                    loss_type=args.camera_loss_type,
                )
                loss = loss + args.real_camera_train_weight * real_camera_loss_value
                logs.update(
                    {
                        f"real_camera_train_{key}": value
                        for key, value in real_camera_logs.items()
                    }
                )
                logs["real_camera_train_loss_camera"] = real_camera_loss_value
                logs["loss"] = loss
            if (
                real_train_loader is not None
                and args.real_sparse_weight > 0
                and step % max(1, args.real_sparse_every) == 0
            ):
                try:
                    real_batch = next(real_train_iterator)
                except StopIteration:
                    real_train_iterator = iter(real_train_loader)
                    real_batch = next(real_train_iterator)
                real_batch = to_device(real_batch, device)
                real_predictions = forward_for_training(model, real_batch["images"], frozen_aggregator)
                real_logs = sparse_depth_loss(real_predictions["depth"], real_batch)
                loss = loss + args.real_sparse_weight * real_logs["loss_sparse_depth"]
                logs.update({f"real_train_{key}": value for key, value in real_logs.items()})
                logs["loss"] = loss
            if (
                real_geometry_train_loader is not None
                and (
                    args.real_geometry_weight > 0
                    or args.real_geometry_camera_weight > 0
                    or args.real_geometry_relative_rig_weight > 0
                    or getattr(args, "real_geometry_shared_rig_weight", 0.0) > 0
                    or args.real_keypoint_head_weight > 0
                )
                and step % max(1, args.real_geometry_every) == 0
            ):
                try:
                    geometry_batch = next(real_geometry_train_iterator)
                except StopIteration:
                    real_geometry_train_iterator = iter(real_geometry_train_loader)
                    geometry_batch = next(real_geometry_train_iterator)
                geometry_batch = to_device(geometry_batch, device)
                geometry_predictions = forward_for_training(
                    model,
                    geometry_batch["images"],
                    frozen_aggregator,
                )
                geometry_loss, geometry_logs = compute_real_geometry_loss(
                    geometry_predictions,
                    geometry_batch,
                    args,
                    use_temporal_filter=True,
                )
                loss = loss + geometry_loss
                logs.update(
                    {
                        f"real_geometry_train_{key}": value
                        for key, value in geometry_logs.items()
                    }
                )
                logs["loss"] = loss
            if not loss.requires_grad:
                print(
                    "batch_skipped",
                    f"scene={batch['scene'][0]}",
                    "reason=no_trainable_branch_for_available_labels",
                    f"camera_views={int(batch['camera_available'].sum())}",
                    f"depth_views={int(batch['depth_available'].sum())}",
                    flush=True,
                )
                continue

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), args.grad_clip
                )
                if not torch.isfinite(grad_norm):
                    bad_gradients = [
                        name
                        for name, param in model.named_parameters()
                        if param.requires_grad
                        and param.grad is not None
                        and not torch.isfinite(param.grad).all()
                    ]
                    raise FloatingPointError(
                        f"Non-finite gradient norm at step {step}; tensors={bad_gradients[:12]}"
                    )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            row: dict[str, Any] = {
                "step": step,
                "epoch": epoch,
                "scene": batch["scene"][0],
                "view_ids": batch["view_ids"][0],
                "scale": float(batch["scale"][0].detach().cpu()),
                "scale_source": batch["scale_source"][0],
                "lr": optimizer.param_groups[0]["lr"],
                "aggregator_lr": (
                    optimizer.param_groups[-1]["lr"]
                    if len(optimizer.param_groups) > 1
                    else optimizer.param_groups[0]["lr"]
                ),
            }
            row.update({key: float(value.detach().cpu()) for key, value in logs.items()})
            if args.val_every and step % args.val_every == 0:
                row.update(evaluate(model, val_loader, device, args, frozen_aggregator, args.val_batches))
                row.update(evaluate_real_pose(model, real_val_loader, device, frozen_aggregator, args.real_val_batches))
                row.update(
                    evaluate_real_palm_depth(
                        model,
                        real_depth_val_loader,
                        device,
                        frozen_aggregator,
                        args.real_depth_val_batches,
                    )
                )
                row.update(
                    evaluate_real_geometry(
                        model,
                        real_geometry_val_loader,
                        device,
                        args,
                        frozen_aggregator,
                        args.real_geometry_val_batches,
                    )
                )
                current_real_palm_mae = row.get("real_depth_val_sparse_palm_mae_mm")
                if current_real_palm_mae is not None and current_real_palm_mae < best_real_palm_mae:
                    best_real_palm_mae = current_real_palm_mae
                    if args.checkpoint_dir:
                        best_path = Path(args.checkpoint_dir) / f"{Path(args.output).stem}_best_real_palm.pt"
                        save_checkpoint_to(best_path, args, model, trainable_names, step, epoch=epoch)
                current_real_camera = row.get("real_val_loss_camera")
                if (
                    current_real_camera is not None
                    and current_real_palm_mae is not None
                    and np.isfinite(baseline_real_camera)
                    and np.isfinite(baseline_real_palm)
                ):
                    joint_score = 0.5 * (
                        current_real_camera / max(baseline_real_camera, 1e-12)
                        + current_real_palm_mae / max(baseline_real_palm, 1e-12)
                    )
                    row["real_joint_normalized_score"] = joint_score
                    if joint_score < best_real_joint_score:
                        best_real_joint_score = joint_score
                        if args.checkpoint_dir:
                            joint_path = (
                                Path(args.checkpoint_dir)
                                / f"{Path(args.output).stem}_best_real_joint.pt"
                            )
                            save_checkpoint_to(
                                joint_path,
                                args,
                                model,
                                trainable_names,
                                step,
                                epoch=epoch,
                            )
                geometry_score = normalized_real_geometry_score(row, baseline)
                if geometry_score is not None:
                    row["real_geometry_normalized_score"] = geometry_score
                    if geometry_score < best_real_geometry_score:
                        best_real_geometry_score = geometry_score
                        if args.checkpoint_dir:
                            geometry_path = (
                                Path(args.checkpoint_dir)
                                / f"{Path(args.output).stem}_best_real_geometry.pt"
                            )
                            save_checkpoint_to(
                                geometry_path,
                                args,
                                model,
                                trainable_names,
                                step,
                                epoch=epoch,
                            )
                keypoint_head_mae = row.get(
                    "real_geometry_val_real_keypoint_head_predicted_mae_mm"
                )
                if (
                    keypoint_head_mae is not None
                    and keypoint_head_mae < best_real_keypoint_head_mae
                ):
                    best_real_keypoint_head_mae = keypoint_head_mae
                    if args.checkpoint_dir:
                        keypoint_path = (
                            Path(args.checkpoint_dir)
                            / f"{Path(args.output).stem}_best_real_keypoint_head.pt"
                        )
                        save_checkpoint_to(
                            keypoint_path,
                            args,
                            model,
                            trainable_names,
                            step,
                            epoch=epoch,
                        )
                relative_rig_loss = row.get(
                    "real_geometry_val_real_geometry_loss_relative_rig"
                )
                if (
                    relative_rig_loss is not None
                    and relative_rig_loss < best_real_relative_rig_loss
                ):
                    best_real_relative_rig_loss = relative_rig_loss
                    if args.checkpoint_dir:
                        relative_rig_path = (
                            Path(args.checkpoint_dir)
                            / f"{Path(args.output).stem}_best_real_relative_rig.pt"
                        )
                        save_checkpoint_to(
                            relative_rig_path,
                            args,
                            model,
                            trainable_names,
                            step,
                            epoch=epoch,
                        )
            append_log(log_path, row)

            if step % args.log_every == 0:
                print(" ".join(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}" for key, value in row.items()), flush=True)
            step += 1
            if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
                save_step_checkpoint(args, model, trainable_names, step, epoch)
            if args.max_steps and step >= args.max_steps:
                save_epoch_checkpoint(args, model, trainable_names, step, epoch)
                save_checkpoint(args, model, trainable_names, step, epoch=epoch + 1)
                return
        save_epoch_checkpoint(args, model, trainable_names, step, epoch)
    save_checkpoint(args, model, trainable_names, step, epoch=args.epochs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blendedmvs-root", default=str(PROJECT_ROOT / "dataset" / "blendedmvs_sample"))
    parser.add_argument("--checkpoint", default=str(REPO_ROOT / "vggt_omega_1b_512.pt"))
    parser.add_argument("--adapter-checkpoint", default="")
    parser.add_argument("--output", default=str(REPO_ROOT / "checkpoints" / "vggt_omega_blendedmvs_light_heads_smoke.pt"))
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Also save a lightweight checkpoint every N optimizer steps; 0 disables step checkpoints.",
    )
    parser.add_argument("--log-csv", default=str(REPO_ROOT / "checkpoints" / "vggt_omega_blendedmvs_light_heads_smoke_log.csv"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--image-width",
        type=int,
        default=0,
        help="Optional rectangular input width; 0 falls back to --image-size.",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=0,
        help="Optional rectangular input height; 0 falls back to --image-size.",
    )
    parser.add_argument("--num-views", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--samples-per-scene", type=int, default=8)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--val-scenes", type=int, default=1)
    parser.add_argument("--val-batches", type=int, default=2)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument(
        "--real-val-roots",
        default="",
        help="Comma-separated real trial directories with videos-raw and calibration/calibration.toml.",
    )
    parser.add_argument("--real-val-cameras", default="A,B,C")
    parser.add_argument("--real-val-frames", default="")
    parser.add_argument("--real-val-frame-start", type=int, default=0)
    parser.add_argument("--real-val-frame-step", type=int, default=20)
    parser.add_argument("--real-val-frame-count", type=int, default=32)
    parser.add_argument("--real-val-batches", type=int, default=4, help="Maximum batches; 0 evaluates all batches.")
    parser.add_argument("--real-val-batch-size", type=int, default=1)
    parser.add_argument("--real-val-scale-mode", choices=["raw", "median_baseline"], default="median_baseline")
    parser.add_argument("--real-camera-train-roots", default="")
    parser.add_argument("--real-camera-train-cameras", default="A,B,C")
    parser.add_argument("--real-camera-train-frames", default="")
    parser.add_argument("--real-camera-train-frame-start", type=int, default=0)
    parser.add_argument("--real-camera-train-frame-step", type=int, default=20)
    parser.add_argument("--real-camera-train-frame-count", type=int, default=32)
    parser.add_argument("--real-camera-train-batch-size", type=int, default=1)
    parser.add_argument("--real-camera-train-every", type=int, default=2)
    parser.add_argument("--real-camera-train-weight", type=float, default=0.0)
    parser.add_argument("--real-camera-translation-weight", type=float, default=1.0)
    parser.add_argument("--real-camera-rotation-weight", type=float, default=1.0)
    parser.add_argument("--real-camera-fov-weight", type=float, default=1.0)
    parser.add_argument(
        "--real-camera-train-scale-mode",
        choices=["raw", "median_baseline"],
        default="median_baseline",
    )
    parser.add_argument(
        "--real-train-roots",
        default="",
        help="Comma-separated real trial directories used for sparse palm-depth training.",
    )
    parser.add_argument(
        "--real-depth-val-roots",
        default="",
        help="Comma-separated held-out real trials used for sparse palm-depth validation.",
    )
    parser.add_argument("--real-depth-cameras", default="A,B,C")
    parser.add_argument("--real-depth-frame-start", type=int, default=0)
    parser.add_argument("--real-train-frame-step", type=int, default=5)
    parser.add_argument("--real-train-frame-count", type=int, default=0)
    parser.add_argument("--real-depth-val-frame-step", type=int, default=20)
    parser.add_argument("--real-depth-val-frame-count", type=int, default=0)
    parser.add_argument("--real-depth-likelihood-threshold", type=float, default=0.7)
    parser.add_argument("--real-sparse-batch-size", type=int, default=4)
    parser.add_argument("--real-sparse-every", type=int, default=1)
    parser.add_argument("--real-sparse-weight", type=float, default=1.0)
    parser.add_argument("--real-palm-point-weight", type=float, default=4.0)
    parser.add_argument("--real-anchor-point-weight", type=float, default=1.0)
    parser.add_argument(
        "--real-depth-val-batches",
        type=int,
        default=0,
        help="Maximum held-out sparse-depth batches; 0 evaluates all batches.",
    )
    parser.add_argument("--real-geometry-train-roots", default="")
    parser.add_argument("--real-geometry-val-roots", default="")
    parser.add_argument("--real-geometry-cameras", default="A,B,C")
    parser.add_argument(
        "--real-geometry-keypoints",
        default="",
        help="Optional comma-separated keypoint subset; empty uses hand, face, and rig anchors.",
    )
    parser.add_argument("--real-geometry-frame-start", type=int, default=0)
    parser.add_argument("--real-geometry-train-frame-step", type=int, default=5)
    parser.add_argument("--real-geometry-val-frame-step", type=int, default=20)
    parser.add_argument("--real-geometry-train-frame-count", type=int, default=0)
    parser.add_argument("--real-geometry-val-frame-count", type=int, default=0)
    parser.add_argument("--real-geometry-val-batches", type=int, default=0)
    parser.add_argument("--real-geometry-batch-size", type=int, default=1)
    parser.add_argument("--real-geometry-every", type=int, default=2)
    parser.add_argument("--real-geometry-likelihood-threshold", type=float, default=0.7)
    parser.add_argument(
        "--real-geometry-scale-mode",
        choices=["raw", "median_baseline", "vggt_rig"],
        default="median_baseline",
    )
    parser.add_argument("--real-geometry-scale-map", default="")
    parser.add_argument("--real-geometry-weight", type=float, default=0.0)
    parser.add_argument("--real-geometry-predicted-rig-weight", type=float, default=1.0)
    parser.add_argument("--real-geometry-fixed-rig-weight", type=float, default=1.0)
    parser.add_argument("--real-geometry-pairwise-weight", type=float, default=0.5)
    parser.add_argument("--real-geometry-reprojection-weight", type=float, default=0.1)
    parser.add_argument("--real-geometry-camera-weight", type=float, default=0.0)
    parser.add_argument(
        "--real-geometry-relative-rig-weight",
        type=float,
        default=0.0,
        help=(
            "Gauge- and scale-invariant supervision on pairwise camera rotations "
            "and the normalized camera-center triangle for real geometry batches."
        ),
    )
    parser.add_argument(
        "--real-geometry-temporal-frames",
        type=int,
        default=1,
        help="Frames from one real trial grouped into each shared-rig training batch.",
    )
    parser.add_argument(
        "--real-geometry-rig-min-confidence",
        type=float,
        default=0.8,
        help="Minimum median DLC likelihood for a frame to supervise the rig.",
    )
    parser.add_argument(
        "--real-geometry-rig-min-coverage",
        type=float,
        default=0.35,
        help="Minimum valid view-keypoint fraction for a frame to supervise the rig.",
    )
    parser.add_argument(
        "--real-geometry-rig-keep-fraction",
        type=float,
        default=1.0,
        help=(
            "Fraction of DLC-qualified frames retained by frozen VGGT dense-head "
            "confidence inside each same-trial temporal batch."
        ),
    )
    parser.add_argument(
        "--real-geometry-rig-max-rotation-error-deg",
        type=float,
        default=30.0,
        help="Reject predicted per-frame rigs farther from the fixed trial rig.",
    )
    parser.add_argument(
        "--real-geometry-rig-max-center-error",
        type=float,
        default=0.4,
        help="Reject normalized camera-center outliers before rig supervision.",
    )
    parser.add_argument("--real-geometry-shared-rig-weight", type=float, default=0.0)
    parser.add_argument(
        "--real-geometry-shared-rig-outlier-rotation-deg",
        type=float,
        default=15.0,
    )
    parser.add_argument(
        "--real-geometry-shared-rig-outlier-center",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--real-geometry-shared-rig-rotation-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--real-geometry-shared-rig-center-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--real-geometry-shared-rig-baseline-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--real-geometry-shared-rig-focal-weight",
        type=float,
        default=0.1,
    )
    parser.add_argument("--real-geometry-palm-weight", type=float, default=4.0)
    parser.add_argument("--real-geometry-finger-weight", type=float, default=2.0)
    parser.add_argument("--real-geometry-face-weight", type=float, default=1.5)
    parser.add_argument("--real-geometry-anchor-weight", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--aggregator-lr-mult", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument(
        "--lr-schedule",
        choices=["constant", "warmup_cosine"],
        default="constant",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--min-lr-ratio", type=float, default=0.01)
    parser.add_argument("--camera-weight", type=float, default=1.0)
    parser.add_argument(
        "--relative-rig-weight",
        type=float,
        default=0.0,
        help=(
            "Gauge- and scale-invariant relative-rig loss for synthetic camera labels."
        ),
    )
    parser.add_argument("--relative-rig-rotation-weight", type=float, default=1.0)
    parser.add_argument("--relative-rig-center-weight", type=float, default=1.0)
    parser.add_argument("--relative-rig-baseline-weight", type=float, default=0.5)
    parser.add_argument("--relative-rig-focal-weight", type=float, default=0.25)
    parser.add_argument(
        "--camera-loss-type",
        choices=["l1", "smooth_l1"],
        default="smooth_l1",
    )
    parser.add_argument("--translation-weight", type=float, default=1.0)
    parser.add_argument("--rotation-weight", type=float, default=1.0)
    parser.add_argument("--fov-weight", type=float, default=1.0)
    parser.add_argument("--depth-weight", type=float, default=0.2)
    parser.add_argument(
        "--depth-loss-mode",
        choices=["log_huber", "official"],
        default="log_huber",
    )
    parser.add_argument("--point-weight", type=float, default=0.0)
    parser.add_argument("--point-loss-stride", type=int, default=1)
    parser.add_argument("--point-gradient-weight", type=float, default=1.0)
    parser.add_argument("--match-weight", type=float, default=0.0)
    parser.add_argument("--match-max-pairs", type=int, default=256)
    parser.add_argument("--match-depth-tolerance", type=float, default=0.01)
    parser.add_argument("--match-border", type=int, default=4)
    parser.add_argument("--match-temperature", type=float, default=0.1)
    parser.add_argument(
        "--use-confidence-loss",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--confidence-alpha", type=float, default=0.0)
    parser.add_argument(
        "--body-mask-dirs",
        default="body_masks,mouse_masks,masks",
        help="Comma-separated optional per-view binary body mask directories under each scene.",
    )
    parser.add_argument(
        "--semantic-mask-dirs",
        default="semantic_masks,part_masks",
        help="Comma-separated optional per-view uint8 semantic/part mask directories under each scene.",
    )
    parser.add_argument(
        "--body-depth-balance",
        type=float,
        default=0.0,
        help=(
            "Blend global depth loss with body-vs-scene balanced loss. "
            "0 keeps legacy all-valid-pixel averaging; 1 uses only balanced body/scene loss."
        ),
    )
    parser.add_argument(
        "--body-depth-fraction",
        type=float,
        default=0.6,
        help="Fraction of balanced depth loss assigned to body pixels when body masks exist.",
    )
    parser.add_argument(
        "--semantic-label-weights",
        default="",
        help="Optional LABEL:WEIGHT comma list for semantic masks, e.g. '1:1,2:3,3:6'. Label 0 defaults to 1.",
    )
    parser.add_argument("--depth-gradient-weight", type=float, default=0.0)
    parser.add_argument("--depth-gradient-scales", default="1,2,4")
    parser.add_argument(
        "--load-keypoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load optional keypoints.npz files when present; missing files remain valid unlabeled samples.",
    )
    parser.add_argument(
        "--keypoint-region-weight",
        type=float,
        default=0.0,
        help="Weight of Gaussian keypoint-neighborhood log-depth loss inside the dense depth objective.",
    )
    parser.add_argument("--keypoint-region-sigma", type=float, default=6.0)
    parser.add_argument(
        "--keypoint-sparse-depth-weight",
        type=float,
        default=0.0,
        help="Top-level weight for visible keypoint surface depth sampled from the GT depth map.",
    )
    parser.add_argument(
        "--keypoint-reprojection-weight",
        type=float,
        default=0.0,
        help="Top-level weight for GT-world-keypoint reprojection through the predicted cameras.",
    )
    parser.add_argument(
        "--enable-keypoint-head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Attach a per-view heatmap head and fuse its Gaussian rays into 3D keypoints.",
    )
    parser.add_argument(
        "--enable-multi-instance-pose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Attach the query-based multi-mouse detection, mask, ReID, and "
            "per-instance keypoint head."
        ),
    )
    parser.add_argument("--num-instance-queries", type=int, default=6)
    parser.add_argument("--instance-pose-weight", type=float, default=0.0)
    parser.add_argument("--instance-objectness-weight", type=float, default=1.0)
    parser.add_argument("--instance-box-l1-weight", type=float, default=5.0)
    parser.add_argument("--instance-box-giou-weight", type=float, default=2.0)
    parser.add_argument("--instance-mask-bce-weight", type=float, default=2.0)
    parser.add_argument("--instance-mask-dice-weight", type=float, default=2.0)
    parser.add_argument("--instance-keypoint-l1-weight", type=float, default=5.0)
    parser.add_argument("--instance-keypoint-visibility-weight", type=float, default=0.5)
    parser.add_argument("--instance-local-heatmap-weight", type=float, default=1.0)
    parser.add_argument("--instance-reid-weight", type=float, default=0.25)
    parser.add_argument(
        "--instance-keypoint-3d-weight",
        type=float,
        default=1.0,
        help=(
            "Per-instance 3D loss after differentiable calibrated multiview "
            "triangulation; missing world labels are ignored."
        ),
    )
    parser.add_argument(
        "--instance-reprojection-weight",
        type=float,
        default=1.0,
        help=(
            "Per-instance multiview reprojection loss after Hungarian/identity "
            "association."
        ),
    )
    parser.add_argument(
        "--instance-world-scale",
        type=float,
        default=1.0,
        help=(
            "Coordinate scale used to normalize instance 3D residuals. The "
            "default matches --scale-mode auto/median_baseline; use roughly "
            "100 for raw millimeter coordinates."
        ),
    )
    parser.add_argument("--instance-no-object-weight", type=float, default=0.1)
    parser.add_argument(
        "--forward-camera-head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the camera head. Disable for an isolated instance-pose experiment.",
    )
    parser.add_argument(
        "--forward-depth-head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the dense depth head. Disable for a faster isolated keypoint-head experiment.",
    )
    parser.add_argument(
        "--keypoint-head-subset",
        default="all",
        help=(
            "Keypoint channels used by head losses: 'all', 'coarse', or a comma-separated "
            "list of canonical Blender keypoint names."
        ),
    )
    parser.add_argument("--keypoint-head-weight", type=float, default=0.0)
    parser.add_argument("--real-keypoint-head-weight", type=float, default=0.0)
    parser.add_argument("--keypoint-head-heatmap-weight", type=float, default=1.0)
    parser.add_argument("--keypoint-head-visibility-weight", type=float, default=0.1)
    parser.add_argument("--keypoint-head-center3d-weight", type=float, default=1.0)
    parser.add_argument("--keypoint-head-fixed-rig-weight", type=float, default=1.0)
    parser.add_argument("--keypoint-head-predicted-rig-weight", type=float, default=0.25)
    parser.add_argument(
        "--keypoint-head-heatmap-sigma",
        type=float,
        default=6.0,
        help="Target 2D Gaussian sigma in full-resolution input pixels.",
    )
    parser.add_argument(
        "--keypoint-head-tail-curve-weight",
        type=float,
        default=0.0,
        help="Top-level tail curve loss weight. Zero preserves the legacy point-only objective.",
    )
    parser.add_argument("--keypoint-head-tail-curve-2d-weight", type=float, default=1.0)
    parser.add_argument("--keypoint-head-tail-curve-3d-weight", type=float, default=1.0)
    parser.add_argument("--keypoint-head-tail-partial-2d-weight", type=float, default=0.25)
    parser.add_argument("--keypoint-head-tail-base-anchor-weight", type=float, default=0.5)
    parser.add_argument("--keypoint-head-tail-tip-anchor-weight", type=float, default=0.25)
    parser.add_argument("--keypoint-head-tail-length-weight", type=float, default=0.1)
    parser.add_argument("--keypoint-head-tail-curvature-weight", type=float, default=0.05)
    parser.add_argument(
        "--keypoint-head-tail-interior-point-weight",
        type=float,
        default=1.0,
        help="Multiplier on legacy heatmap/3D point losses for tail_q1..tail_q5.",
    )
    parser.add_argument("--keypoint-head-tail-curve-samples", type=int, default=32)
    parser.add_argument("--keypoint-head-tail-smoothness", type=float, default=1e-2)
    parser.add_argument("--keypoint-head-tail-min-points", type=int, default=4)
    parser.add_argument(
        "--keypoint-head-tail-curve-distance",
        choices=["ordered", "chamfer", "soft_dtw"],
        default="ordered",
        help="Curve correspondence ablation. ordered is fastest; soft_dtw is reparameterization tolerant.",
    )
    parser.add_argument("--keypoint-head-tail-soft-dtw-gamma", type=float, default=0.05)
    parser.add_argument("--keypoint-head-tail-partial-temperature", type=float, default=0.01)
    parser.add_argument(
        "--keypoint-head-tail-robust-views",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Downweight a view whose tail curve residual is inconsistent with the other views.",
    )
    parser.add_argument("--keypoint-head-tail-robust-delta", type=float, default=2.5)
    parser.add_argument(
        "--keypoint-head-tail-detach-fit-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prevent visibility uncertainty from reducing loss by changing spline fit weights.",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-srcs", type=int, default=8)
    parser.add_argument(
        "--train-mode",
        choices=[
            "light_heads",
            "camera_branch",
            "depth_projection",
            "geometry_adapters",
            "heads",
            "camera",
            "dense",
            "keypoint_head",
            "keypoint_last_block",
            "instance_pose_head",
            "instance_pose_last_block",
            "last_blocks",
            "full",
        ],
        default="light_heads",
    )
    parser.add_argument("--unfreeze-last-blocks", type=int, default=1)
    parser.add_argument(
        "--freeze-confidence-projection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep confidence output channels frozen while adapting depth/point predictions.",
    )
    parser.add_argument("--train-camera-register-tokens", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--canonicalize-camera", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pfm-flip-y", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-masked-images", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--scene-path-includes",
        default="",
        help="Comma-separated fnmatch patterns; only matching scene paths are loaded.",
    )
    parser.add_argument(
        "--scene-path-excludes",
        default="",
        help="Comma-separated fnmatch patterns removed after scene discovery.",
    )
    parser.add_argument(
        "--scale-mode",
        choices=[
            "raw",
            "median_depth",
            "median_baseline",
            "mean_point_distance",
            "auto",
        ],
        default="auto",
        help=(
            "Scene scale normalization. 'auto' uses median valid depth when present, "
            "otherwise the median labeled-camera baseline."
        ),
    )
    parser.add_argument("--save-full", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", choices=["none", "data", "model"], default="none")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
