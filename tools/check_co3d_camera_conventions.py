#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import html
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


S_P3D_TO_CV = np.diag([-1.0, -1.0, 1.0]).astype(np.float64)


@dataclass(frozen=True)
class FrameData:
    sequence_name: str
    frame_number: int
    image_path: Path
    depth_path: Path
    mask_path: Path
    image_size_wh: tuple[int, int]
    depth_scale: float
    viewpoint: dict[str, Any]


@dataclass(frozen=True)
class Candidate:
    name: str
    description: str
    fn: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def read_jgz(path: Path) -> Any:
    with gzip.open(path, "rt") as f:
        return json.load(f)


def read_depth(path: Path, scale_adjustment: float) -> np.ndarray:
    with Image.open(path) as depth_pil:
        raw = np.array(depth_pil, dtype=np.uint16)
    depth = np.frombuffer(raw.tobytes(), dtype=np.float16).reshape(raw.shape)
    return depth.astype(np.float64) * float(scale_adjustment)


def read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as mask_pil:
        return np.asarray(mask_pil) > 0


def intrinsic_from_co3d(viewpoint: dict[str, Any], image_size_wh: tuple[int, int]) -> np.ndarray:
    width, height = image_size_wh
    min_side = float(min(width, height))
    focal = viewpoint["focal_length"]
    pp = viewpoint["principal_point"]
    fx = float(focal[0]) * min_side / 2.0
    fy = float(focal[1]) * min_side / 2.0
    cx = width / 2.0 - float(pp[0]) * min_side / 2.0
    cy = height / 2.0 - float(pp[1]) * min_side / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def invert_extrinsic(r: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r_inv = r.T
    t_inv = -r_inv @ t
    return r_inv, t_inv


def make_candidates() -> list[Candidate]:
    def official_p3d_to_opencv(r: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # CO3D/PyTorch3D stores R in right-multiply form:
        # X_cam_p3d(row) = X_world(row) @ R + T.
        # In column-vector form, R_col = R.T. OpenCV camera axes are
        # +X right, +Y down, +Z forward, so flip PyTorch3D camera X/Y.
        return S_P3D_TO_CV @ r.T, S_P3D_TO_CV @ t

    def current(r: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return S_P3D_TO_CV @ r, S_P3D_TO_CV @ t

    def raw_w2c(r: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return r, t

    def raw_c2w_then_current(r: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r_w2c, t_w2c = invert_extrinsic(r, t)
        return S_P3D_TO_CV @ r_w2c, S_P3D_TO_CV @ t_w2c

    def raw_c2w_no_flip(r: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return invert_extrinsic(r, t)

    def inverse_of_current(r: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r_cur, t_cur = current(r, t)
        return invert_extrinsic(r_cur, t_cur)

    def sandwich_w2c(r: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return S_P3D_TO_CV @ r @ S_P3D_TO_CV, S_P3D_TO_CV @ t

    def right_flip_w2c(r: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return r @ S_P3D_TO_CV, t

    return [
        Candidate("official_p3d_to_opencv", "CO3D/PyTorch3D right-multiply R, converted to OpenCV: S @ R.T, S @ T.", official_p3d_to_opencv),
        Candidate("current_s_left_w2c", "Current assumption: OpenCV camera = S @ raw PyTorch3D world-to-camera.", current),
        Candidate("raw_w2c_no_flip", "Treat raw R/T as OpenCV world-to-camera without axis flip.", raw_w2c),
        Candidate("raw_c2w_s_left", "Treat raw R/T as camera-to-world, invert, then apply S on camera axes.", raw_c2w_then_current),
        Candidate("raw_c2w_no_flip", "Treat raw R/T as OpenCV camera-to-world, invert only.", raw_c2w_no_flip),
        Candidate("inverse_of_current", "Invert the current assumption.", inverse_of_current),
        Candidate("sandwich_s_r_s", "Apply S on both camera/world sides.", sandwich_w2c),
        Candidate("right_flip_w2c", "Apply S on the world side only.", right_flip_w2c),
    ]


def camera_from_candidate(frame: FrameData, candidate: Candidate) -> tuple[np.ndarray, np.ndarray]:
    r_raw = np.asarray(frame.viewpoint["R"], dtype=np.float64)
    t_raw = np.asarray(frame.viewpoint["T"], dtype=np.float64)
    r, t = candidate.fn(r_raw, t_raw)
    extrinsic = np.concatenate([r, t[:, None]], axis=1)
    intri = intrinsic_from_co3d(frame.viewpoint, frame.image_size_wh)
    return extrinsic, intri


def load_frames(category_root: Path) -> dict[str, list[FrameData]]:
    root = category_root.parent
    frames = read_jgz(category_root / "frame_annotations.jgz")
    grouped: dict[str, list[FrameData]] = defaultdict(list)
    for anno in frames:
        image_rel = Path(anno["image"]["path"])
        depth_rel = Path(anno["depth"]["path"])
        mask_rel = Path(anno["depth"]["mask_path"])
        image_size = anno["image"]["size"]
        # CO3D stores image.size as [height, width], while PIL uses (width, height).
        image_size_wh = (int(image_size[1]), int(image_size[0]))
        grouped[anno["sequence_name"]].append(
            FrameData(
                sequence_name=anno["sequence_name"],
                frame_number=int(anno["frame_number"]),
                image_path=root / image_rel,
                depth_path=root / depth_rel,
                mask_path=root / mask_rel,
                image_size_wh=image_size_wh,
                depth_scale=float(anno["depth"].get("scale_adjustment", 1.0)),
                viewpoint=anno["viewpoint"],
            )
        )
    for seq_frames in grouped.values():
        seq_frames.sort(key=lambda f: f.frame_number)
    return grouped


def load_sequence_quality(category_root: Path) -> dict[str, dict[str, float]]:
    quality: dict[str, dict[str, float]] = {}
    path = category_root / "sequence_annotations.jgz"
    if not path.exists():
        return quality
    for item in read_jgz(path):
        point_cloud = item.get("point_cloud") or {}
        quality[item["sequence_name"]] = {
            "pointcloud_quality": float(point_cloud.get("quality_score", float("nan"))),
            "viewpoint_quality": float(item.get("viewpoint_quality_score", float("nan"))),
            "point_count": float(point_cloud.get("n_points", float("nan"))),
        }
    return quality


def choose_sequences(
    category_root: Path,
    grouped: dict[str, list[FrameData]],
    sequence_quality: dict[str, dict[str, float]],
    count: int,
    min_frames: int,
    seed: int,
    min_pointcloud_quality: float,
    min_viewpoint_quality: float,
) -> list[str]:
    rng = random.Random(seed)
    eligible = []
    for name, frames in grouped.items():
        if len(frames) < min_frames or not (category_root / name / "pointcloud.ply").exists():
            continue
        q = sequence_quality.get(name, {})
        pcq = q.get("pointcloud_quality", float("nan"))
        vpq = q.get("viewpoint_quality", float("nan"))
        if np.isfinite(pcq) and pcq < min_pointcloud_quality:
            continue
        if np.isfinite(vpq) and vpq < min_viewpoint_quality:
            continue
        eligible.append(name)
    eligible.sort(
        key=lambda name: (
            -nan_to_zero(sequence_quality.get(name, {}).get("pointcloud_quality", 0.0)),
            -nan_to_zero(sequence_quality.get(name, {}).get("viewpoint_quality", 0.0)),
            -len(grouped[name]),
            name,
        )
    )
    if len(eligible) <= count:
        return eligible
    top = eligible[: max(count * 4, count)]
    rng.shuffle(top)
    return sorted(top[:count])


def choose_frames(frames: list[FrameData], count: int, sampling: str) -> list[FrameData]:
    if len(frames) <= count:
        return frames
    if sampling == "spread":
        idxs = np.linspace(0, len(frames) - 1, count).round().astype(int)
    elif sampling == "nearby":
        start = max(0, (len(frames) - count) // 2)
        idxs = np.arange(start, start + count)
    else:
        raise ValueError(f"Unknown frame sampling mode: {sampling}")
    return [frames[int(i)] for i in idxs]


def unproject_depth_sample(
    frame: FrameData,
    extrinsic: np.ndarray,
    intri: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = read_depth(frame.depth_path, frame.depth_scale)
    mask = read_mask(frame.mask_path)
    h, w = depth.shape
    valid_y, valid_x = np.nonzero(mask & np.isfinite(depth) & (depth > 1e-6))
    if len(valid_x) > max_points:
        take = rng.choice(len(valid_x), size=max_points, replace=False)
        valid_x = valid_x[take]
        valid_y = valid_y[take]
    z = depth[valid_y, valid_x]
    x = (valid_x.astype(np.float64) - intri[0, 2]) * z / intri[0, 0]
    y = (valid_y.astype(np.float64) - intri[1, 2]) * z / intri[1, 1]
    cam = np.stack([x, y, z], axis=1)
    r = extrinsic[:, :3]
    t = extrinsic[:, 3]
    world = (r.T @ (cam - t).T).T
    rgb = np.asarray(Image.open(frame.image_path).convert("RGB"))
    colors = rgb[valid_y, valid_x]
    return world, z, colors


def project_points(points_world: np.ndarray, extrinsic: np.ndarray, intri: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cam = (extrinsic[:, :3] @ points_world.T + extrinsic[:, 3:4]).T
    z = cam[:, 2]
    uv = np.full((len(points_world), 2), np.nan, dtype=np.float64)
    positive = z > 1e-6
    uv[positive, 0] = intri[0, 0] * cam[positive, 0] / z[positive] + intri[0, 2]
    uv[positive, 1] = intri[1, 1] * cam[positive, 1] / z[positive] + intri[1, 2]
    return uv, z, positive


def depth_reprojection_metrics(
    src: FrameData,
    dst: FrameData,
    src_cam: tuple[np.ndarray, np.ndarray],
    dst_cam: tuple[np.ndarray, np.ndarray],
    max_points: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    src_extri, src_intri = src_cam
    dst_extri, dst_intri = dst_cam
    points_world, _, _ = unproject_depth_sample(src, src_extri, src_intri, max_points=max_points, rng=rng)
    n = int(len(points_world))
    if n == 0:
        return empty_metric("depth_", n)

    uv, proj_z, positive = project_points(points_world, dst_extri, dst_intri)
    dst_depth = read_depth(dst.depth_path, dst.depth_scale)
    dst_mask = read_mask(dst.mask_path)
    h, w = dst_depth.shape
    x, y, finite = rounded_uv(uv)
    inside = finite & positive & (x >= 0) & (x < w) & (y >= 0) & (y < h)
    hit_mask = np.zeros(n, dtype=bool)
    valid_depth = np.zeros(n, dtype=bool)
    hit_mask[inside] = dst_mask[y[inside], x[inside]]
    target_z = np.full(n, np.nan, dtype=np.float64)
    target_z[inside] = dst_depth[y[inside], x[inside]]
    valid_depth[inside] = hit_mask[inside] & np.isfinite(target_z[inside]) & (target_z[inside] > 1e-6)

    log_err = np.abs(np.log(np.clip(proj_z[valid_depth], 1e-6, None)) - np.log(np.clip(target_z[valid_depth], 1e-6, None)))
    rel_err = np.abs(proj_z[valid_depth] - target_z[valid_depth]) / np.clip(target_z[valid_depth], 1e-6, None)
    return {
        "depth_sample_count": n,
        "depth_positive_z_rate": float(positive.mean()),
        "depth_in_image_rate": float(inside.mean()),
        "depth_mask_hit_rate_all": float(hit_mask.mean()),
        "depth_mask_hit_rate_inside": safe_rate(hit_mask.sum(), inside.sum()),
        "depth_valid_target_rate": float(valid_depth.mean()),
        "depth_median_log_error": finite_median(log_err),
        "depth_median_relative_error": finite_median(rel_err),
        "depth_agree_10pct_rate": float((rel_err < 0.10).mean()) if len(rel_err) else float("nan"),
    }


def prefixed(metrics: dict[str, float], old_prefix: str, new_prefix: str) -> dict[str, float]:
    out = {}
    for key, value in metrics.items():
        if key.startswith(old_prefix):
            out[new_prefix + key[len(old_prefix) :]] = value
        else:
            out[new_prefix + key] = value
    return out


def depth_self_projection_metrics(
    frame: FrameData,
    cam: tuple[np.ndarray, np.ndarray],
    max_points: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    metrics = depth_reprojection_metrics(frame, frame, cam, cam, max_points=max_points, rng=rng)
    return prefixed(metrics, "depth_", "depth_self_")


def empty_metric(prefix: str, n: int) -> dict[str, float]:
    return {
        f"{prefix}sample_count": float(n),
        f"{prefix}positive_z_rate": float("nan"),
        f"{prefix}in_image_rate": float("nan"),
        f"{prefix}mask_hit_rate_all": float("nan"),
        f"{prefix}mask_hit_rate_inside": float("nan"),
    }


def safe_rate(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def finite_median(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else float("nan")


def finite_mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else float("nan")


def rounded_uv(uv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1])
    x = np.full(len(uv), -1, dtype=np.int64)
    y = np.full(len(uv), -1, dtype=np.int64)
    x[finite] = np.rint(uv[finite, 0]).astype(np.int64)
    y[finite] = np.rint(uv[finite, 1]).astype(np.int64)
    return x, y, finite


def projected_silhouette_mask(
    uv: np.ndarray,
    z: np.ndarray,
    mask_shape: tuple[int, int],
    point_radius: int,
    fill_mode: str,
    close_radius: int,
) -> np.ndarray:
    h, w = mask_shape
    x, y, finite = rounded_uv(uv)
    inside = finite & (z > 1e-6) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y[inside], x[inside]] = 255
    if point_radius > 0:
        kernel = 2 * int(point_radius) + 1
        mask = np.asarray(Image.fromarray(mask, mode="L").filter(ImageFilter.MaxFilter(kernel)))
    if fill_mode == "contour":
        if not mask.any():
            return mask > 0
        if close_radius > 0:
            kernel_size = 2 * int(close_radius) + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, contours, contourIdx=-1, color=255, thickness=cv2.FILLED)
        mask = filled
    elif fill_mode == "hull":
        if not inside.any():
            return mask > 0
        coords = np.stack([x[inside], y[inside]], axis=1).astype(np.int32)
        hull = cv2.convexHull(coords)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
    elif fill_mode != "splat":
        raise ValueError(f"Unknown silhouette fill mode: {fill_mode}")
    return mask > 0


def mask_overlap_metrics(target_mask: np.ndarray, projected_mask: np.ndarray) -> dict[str, float]:
    target = target_mask.astype(bool)
    projected = projected_mask.astype(bool)
    intersection = int((target & projected).sum())
    union = int((target | projected).sum())
    target_count = int(target.sum())
    projected_count = int(projected.sum())
    return {
        "ply_silhouette_iou": safe_rate(intersection, union),
        "ply_silhouette_target_coverage_rate": safe_rate(intersection, target_count),
        "ply_silhouette_projection_precision_rate": safe_rate(intersection, projected_count),
        "ply_silhouette_area_ratio": safe_rate(projected_count, target_count),
    }


def label_font(size: int = 18):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def read_ply_vertices(path: Path, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PLY header: {path}")
            header_lines.append(line.decode("ascii").strip())
            if line.strip() == b"end_header":
                break
        fmt = next(line for line in header_lines if line.startswith("format ")).split()[1]
        vertex_line = next(line for line in header_lines if line.startswith("element vertex "))
        vertex_count = int(vertex_line.split()[2])
        props = [line.split()[1:] for line in header_lines if line.startswith("property ")]
        if fmt == "binary_little_endian":
            dtype_fields = []
            for typ, name in props:
                dtype_fields.append((name, ply_dtype(typ)))
            arr = np.fromfile(f, dtype=np.dtype(dtype_fields), count=vertex_count)
        elif fmt == "ascii":
            data = []
            for _ in range(vertex_count):
                data.append(f.readline().decode("ascii").split())
            dtype_fields = [(name, np.float64 if typ in {"float", "double"} else np.uint8) for typ, name in props]
            arr = np.array([tuple(row) for row in data], dtype=np.dtype(dtype_fields))
        else:
            raise ValueError(f"Unsupported PLY format {fmt}: {path}")

    points = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
    if all(name in arr.dtype.names for name in ["red", "green", "blue"]):
        colors = np.stack([arr["red"], arr["green"], arr["blue"]], axis=1).astype(np.uint8)
    else:
        colors = np.full((len(points), 3), [0, 220, 255], dtype=np.uint8)

    if len(points) > max_points:
        rng = np.random.default_rng(seed)
        take = rng.choice(len(points), size=max_points, replace=False)
        points = points[take]
        colors = colors[take]
    return points, colors


def ply_dtype(name: str) -> str:
    mapping = {
        "float": "<f4",
        "float32": "<f4",
        "double": "<f8",
        "uchar": "u1",
        "uint8": "u1",
        "char": "i1",
        "int": "<i4",
        "int32": "<i4",
        "uint": "<u4",
        "uint32": "<u4",
    }
    if name not in mapping:
        raise ValueError(f"Unsupported PLY property type: {name}")
    return mapping[name]


def pointcloud_metrics(
    frame: FrameData,
    candidate_cam: tuple[np.ndarray, np.ndarray],
    points: np.ndarray,
    colors: np.ndarray,
    object_center: np.ndarray,
    visible_rel_threshold: float,
    silhouette_point_radius: int,
    silhouette_fill_mode: str,
    silhouette_close_radius: int,
) -> dict[str, float]:
    extri, intri = candidate_cam
    uv, z, positive = project_points(points, extri, intri)
    mask = read_mask(frame.mask_path)
    depth = read_depth(frame.depth_path, frame.depth_scale)
    image = np.asarray(Image.open(frame.image_path).convert("RGB"))
    h, w = mask.shape
    x, y, finite = rounded_uv(uv)
    inside = finite & positive & (x >= 0) & (x < w) & (y >= 0) & (y < h)
    hit = np.zeros(len(points), dtype=bool)
    hit[inside] = mask[y[inside], x[inside]]
    target_z = np.full(len(points), np.nan, dtype=np.float64)
    target_z[inside] = depth[y[inside], x[inside]]
    depth_valid = inside & hit & np.isfinite(target_z) & (target_z > 1e-6)
    depth_rel = np.full(len(points), np.inf, dtype=np.float64)
    depth_rel[depth_valid] = np.abs(z[depth_valid] - target_z[depth_valid]) / np.clip(target_z[depth_valid], 1e-6, None)
    visible = depth_valid & (depth_rel < visible_rel_threshold)
    if visible.any():
        sampled_rgb = image[y[visible], x[visible]].astype(np.float64)
        ply_rgb = colors[visible].astype(np.float64)
        color_mae = float(np.mean(np.abs(sampled_rgb - ply_rgb)) / 255.0)
        color_median = float(np.median(np.abs(sampled_rgb - ply_rgb)) / 255.0)
    else:
        color_mae = float("nan")
        color_median = float("nan")

    r = extri[:, :3]
    t = extri[:, 3]
    center = -r.T @ t
    forward = r.T @ np.array([0.0, 0.0, 1.0])
    to_object = object_center - center
    look_cos = float(np.dot(forward, to_object) / (np.linalg.norm(forward) * np.linalg.norm(to_object) + 1e-12))
    silhouette = projected_silhouette_mask(
        uv,
        z,
        mask.shape,
        silhouette_point_radius,
        silhouette_fill_mode,
        silhouette_close_radius,
    )
    silhouette_metrics = mask_overlap_metrics(mask, silhouette)

    return {
        "ply_sample_count": float(len(points)),
        "ply_positive_z_rate": float(positive.mean()),
        "ply_in_image_rate": float(inside.mean()),
        "ply_mask_hit_rate_all": float(hit.mean()),
        "ply_mask_hit_rate_inside": safe_rate(hit.sum(), inside.sum()),
        "ply_depth_visible_rate_all": float(visible.mean()),
        "ply_depth_visible_rate_masked": safe_rate(visible.sum(), hit.sum()),
        "ply_visible_color_mae": color_mae,
        "ply_visible_color_median_abs": color_median,
        "ply_camera_look_cos": look_cos,
        "ply_median_positive_z": finite_median(z[positive]),
        **silhouette_metrics,
    }


def draw_projected_points(
    image_path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    extri: np.ndarray,
    intri: np.ndarray,
    title: str,
    max_width: int,
    visible_depth: np.ndarray | None = None,
    visible_mask: np.ndarray | None = None,
    visible_rel_threshold: float = 0.05,
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    arr = np.asarray(image).copy()
    h, w = arr.shape[:2]
    uv, z, positive = project_points(points, extri, intri)
    x, y, finite = rounded_uv(uv)
    inside = finite & positive & (x >= 0) & (x < w) & (y >= 0) & (y < h)
    if visible_depth is not None:
        visible = np.zeros(len(points), dtype=bool)
        target_z = np.full(len(points), np.nan, dtype=np.float64)
        target_z[inside] = visible_depth[y[inside], x[inside]]
        visible[inside] = np.isfinite(target_z[inside]) & (target_z[inside] > 1e-6)
        if visible_mask is not None:
            visible[inside] &= visible_mask[y[inside], x[inside]]
        rel = np.full(len(points), np.inf, dtype=np.float64)
        rel[visible] = np.abs(z[visible] - target_z[visible]) / np.clip(target_z[visible], 1e-6, None)
        inside = inside & visible & (rel < visible_rel_threshold)
    xi = x[inside]
    yi = y[inside]
    ci = colors[inside].astype(np.float32)
    for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
        xx = xi + dx
        yy = yi + dy
        ok = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
        arr[yy[ok], xx[ok]] = (0.35 * arr[yy[ok], xx[ok]] + 0.65 * ci[ok]).astype(np.uint8)

    out = Image.fromarray(arr)
    if out.width > max_width:
        scale = max_width / out.width
        out = out.resize((max_width, int(out.height * scale)), Image.BICUBIC)
    draw = ImageDraw.Draw(out)
    font = label_font(18)
    draw.rectangle([0, 0, out.width, 38], fill=(0, 0, 0))
    draw.text((8, 9), title[:140], fill=(255, 255, 255), font=font)
    return out


def draw_silhouette_comparison(
    frame: FrameData,
    points: np.ndarray,
    extri: np.ndarray,
    intri: np.ndarray,
    title: str,
    max_width: int,
    point_radius: int,
    fill_mode: str,
    close_radius: int,
) -> Image.Image:
    image = Image.open(frame.image_path).convert("RGB")
    arr = np.asarray(image).copy()
    target_mask = read_mask(frame.mask_path)
    uv, z, _ = project_points(points, extri, intri)
    projected_mask = projected_silhouette_mask(uv, z, target_mask.shape, point_radius, fill_mode, close_radius)
    metrics = mask_overlap_metrics(target_mask, projected_mask)

    overlap = target_mask & projected_mask
    target_only = target_mask & ~projected_mask
    projected_only = projected_mask & ~target_mask
    overlay = arr.astype(np.float32)
    colors = [
        (target_only, np.array([0.0, 255.0, 80.0], dtype=np.float32)),
        (projected_only, np.array([255.0, 0.0, 180.0], dtype=np.float32)),
        (overlap, np.array([255.0, 220.0, 0.0], dtype=np.float32)),
    ]
    for mask, color in colors:
        overlay[mask] = 0.45 * overlay[mask] + 0.55 * color
    out = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    if out.width > max_width:
        scale = max_width / out.width
        out = out.resize((max_width, int(out.height * scale)), Image.BICUBIC)
    draw = ImageDraw.Draw(out)
    font = label_font(18)
    subtitle = (
        f"{title} | IoU={metrics['ply_silhouette_iou']:.3f} "
        f"cov={metrics['ply_silhouette_target_coverage_rate']:.3f} "
        f"prec={metrics['ply_silhouette_projection_precision_rate']:.3f}"
    )
    draw.rectangle([0, 0, out.width, 58], fill=(0, 0, 0))
    draw.text((8, 8), subtitle[:150], fill=(255, 255, 255), font=font)
    draw.text((8, 34), "green=target only, magenta=projection only, yellow=overlap", fill=(255, 255, 255), font=font)
    return out


def draw_depth_reprojection(
    src: FrameData,
    dst: FrameData,
    src_cam: tuple[np.ndarray, np.ndarray],
    dst_cam: tuple[np.ndarray, np.ndarray],
    title: str,
    max_points: int,
    max_width: int,
    rng: np.random.Generator,
) -> Image.Image:
    dst_image = Image.open(dst.image_path).convert("RGB")
    arr = np.asarray(dst_image).copy()
    h, w = arr.shape[:2]
    points, _, _ = unproject_depth_sample(src, src_cam[0], src_cam[1], max_points=max_points, rng=rng)
    uv, proj_z, positive = project_points(points, dst_cam[0], dst_cam[1])
    dst_depth = read_depth(dst.depth_path, dst.depth_scale)
    dst_mask = read_mask(dst.mask_path)
    x, y, finite = rounded_uv(uv)
    inside = finite & positive & (x >= 0) & (x < w) & (y >= 0) & (y < h)
    valid = np.zeros(len(points), dtype=bool)
    rel = np.full(len(points), np.inf, dtype=np.float64)
    valid[inside] = dst_mask[y[inside], x[inside]] & (dst_depth[y[inside], x[inside]] > 1e-6)
    rel[valid] = np.abs(proj_z[valid] - dst_depth[y[valid], x[valid]]) / np.clip(dst_depth[y[valid], x[valid]], 1e-6, None)
    good = inside & valid & (rel < 0.10)
    bad = inside & valid & ~good
    no_depth = inside & ~valid
    for mask, color in [(no_depth, np.array([255, 210, 0])), (bad, np.array([255, 40, 40])), (good, np.array([0, 255, 80]))]:
        xi = x[mask]
        yi = y[mask]
        for dx, dy in [(0, 0), (1, 0), (0, 1)]:
            xx = xi + dx
            yy = yi + dy
            ok = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
            arr[yy[ok], xx[ok]] = color
    out = Image.fromarray(arr)
    if out.width > max_width:
        scale = max_width / out.width
        out = out.resize((max_width, int(out.height * scale)), Image.BICUBIC)
    draw = ImageDraw.Draw(out)
    font = label_font(18)
    draw.rectangle([0, 0, out.width, 38], fill=(0, 0, 0))
    draw.text((8, 9), title[:140], fill=(255, 255, 255), font=font)
    return out


def make_contact_sheet(images: list[Image.Image], columns: int = 2, margin: int = 8) -> Image.Image:
    if not images:
        raise ValueError("No images for contact sheet")
    cell_w = max(img.width for img in images)
    cell_h = max(img.height for img in images)
    rows = int(math.ceil(len(images) / columns))
    sheet = Image.new("RGB", (columns * cell_w + (columns + 1) * margin, rows * cell_h + (rows + 1) * margin), (245, 245, 245))
    for idx, img in enumerate(images):
        row, col = divmod(idx, columns)
        x = margin + col * (cell_w + margin)
        y = margin + row * (cell_h + margin)
        sheet.paste(img, (x, y))
    return sheet


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_candidate[row["candidate"]].append(row)
    summary = []
    metric_names = [
        "depth_positive_z_rate",
        "depth_in_image_rate",
        "depth_mask_hit_rate_all",
        "depth_mask_hit_rate_inside",
        "depth_valid_target_rate",
        "depth_median_log_error",
        "depth_median_relative_error",
        "depth_agree_10pct_rate",
        "depth_self_positive_z_rate",
        "depth_self_in_image_rate",
        "depth_self_mask_hit_rate_all",
        "depth_self_mask_hit_rate_inside",
        "depth_self_valid_target_rate",
        "depth_self_median_log_error",
        "depth_self_median_relative_error",
        "depth_self_agree_10pct_rate",
        "ply_positive_z_rate",
        "ply_in_image_rate",
        "ply_mask_hit_rate_all",
        "ply_mask_hit_rate_inside",
        "ply_depth_visible_rate_all",
        "ply_depth_visible_rate_masked",
        "ply_visible_color_mae",
        "ply_visible_color_median_abs",
        "ply_silhouette_iou",
        "ply_silhouette_target_coverage_rate",
        "ply_silhouette_projection_precision_rate",
        "ply_silhouette_area_ratio",
        "ply_camera_look_cos",
    ]
    for candidate, candidate_rows in by_candidate.items():
        out = {"candidate": candidate, "sample_count": len(candidate_rows)}
        for metric in metric_names:
            values = np.asarray([float(row.get(metric, float("nan"))) for row in candidate_rows], dtype=np.float64)
            if metric.endswith("_rate") or metric.endswith("_cos"):
                out[metric] = finite_mean(values)
            else:
                out[metric] = finite_median(values)
        depth_support = nan_to_zero(out["depth_mask_hit_rate_all"])
        ply_support = nan_to_zero(out["ply_mask_hit_rate_all"])
        ply_visible_support = nan_to_zero(out["ply_depth_visible_rate_all"])
        silhouette_iou = nan_to_zero(out["ply_silhouette_iou"])
        silhouette_target = nan_to_zero(out["ply_silhouette_target_coverage_rate"])
        silhouette_precision = nan_to_zero(out["ply_silhouette_projection_precision_rate"])
        depth_quality = depth_support * nan_to_zero(out["depth_agree_10pct_rate"])
        rel_penalty = depth_support * min(nan_to_large(out["depth_median_relative_error"]), 2.0)
        color_penalty = ply_visible_support * min(nan_to_large(out["ply_visible_color_mae"]), 1.0)
        score = (
            5.0 * ply_support
            + 6.0 * ply_visible_support
            + 4.0 * silhouette_iou
            + 1.5 * silhouette_target
            + 1.5 * silhouette_precision
            + 5.0 * depth_support
            + 2.0 * depth_quality
            + 0.8 * nan_to_zero(out["depth_in_image_rate"])
            + 0.6 * nan_to_zero(out["ply_in_image_rate"])
            + 0.8 * nan_to_zero(out["ply_camera_look_cos"])
            - rel_penalty
            - color_penalty
        )
        out["diagnostic_score"] = score
        summary.append(out)
    summary.sort(key=lambda row: row["diagnostic_score"], reverse=True)
    for rank, row in enumerate(summary, start=1):
        row["rank"] = rank
    return summary


def nan_to_zero(value: float) -> float:
    return 0.0 if not np.isfinite(value) else float(value)


def nan_to_large(value: float) -> float:
    return 5.0 if not np.isfinite(value) else float(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_html(path: Path, summary: list[dict[str, Any]], detail_csv: Path, images: list[Path]) -> None:
    headers = [
        "rank",
        "candidate",
        "diagnostic_score",
        "ply_mask_hit_rate_all",
        "ply_silhouette_iou",
        "ply_silhouette_target_coverage_rate",
        "ply_silhouette_projection_precision_rate",
        "ply_depth_visible_rate_all",
        "ply_visible_color_mae",
        "depth_mask_hit_rate_all",
        "depth_in_image_rate",
        "depth_median_relative_error",
        "depth_agree_10pct_rate",
        "depth_self_in_image_rate",
        "depth_self_mask_hit_rate_all",
        "depth_self_median_relative_error",
        "depth_self_agree_10pct_rate",
        "ply_camera_look_cos",
    ]
    rows_html = []
    for row in summary:
        cells = []
        for h in headers:
            value = row.get(h, "")
            if isinstance(value, float):
                value = f"{value:.4g}" if np.isfinite(value) else "nan"
            cells.append(f"<td>{html.escape(str(value))}</td>")
        rows_html.append("<tr>" + "".join(cells) + "</tr>")
    image_html = "\n".join(f'<h3>{html.escape(img.name)}</h3><img src="{html.escape(img.name)}" style="max-width: 100%; border: 1px solid #ccc;">' for img in images)
    doc = f"""<!doctype html>
<meta charset="utf-8">
<title>CO3D camera convention check</title>
<style>
body {{ font-family: sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; margin-bottom: 24px; }}
td, th {{ border: 1px solid #ccc; padding: 4px 8px; }}
th {{ background: #eee; }}
code {{ background: #f3f3f3; padding: 1px 3px; }}
</style>
<h1>CO3D Camera Convention Check</h1>
<p><code>depth_self_*</code> is a same-frame masked-depth round-trip control. It should be near-perfect; if it is not, depth decoding, mask loading, intrinsics, or project/unproject math is broken.</p>
<p><code>depth_*</code> is cross-frame masked-depth reprojection and is the cleaner camera-consistency diagnostic than the sequence PLY overlay.</p>
<p>Depth overlay colors: green = projected source depth agrees within 10%, red = projected to valid mask but depth disagrees, yellow = projected in image without target depth support.</p>
<p>Detailed per-sequence metrics: <code>{html.escape(detail_csv.name)}</code></p>
<table><thead><tr>{''.join(f'<th>{html.escape(h)}</th>' for h in headers)}</tr></thead><tbody>
{''.join(rows_html)}
</tbody></table>
{image_html}
"""
    path.write_text(doc)


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = make_candidates()
    all_rows: list[dict[str, Any]] = []
    image_outputs: list[Path] = []
    rng = np.random.default_rng(args.seed)

    for category_root_str in args.category_root:
        category_root = Path(category_root_str)
        category = category_root.name
        grouped = load_frames(category_root)
        sequence_quality = load_sequence_quality(category_root)
        if args.sequence_name:
            seq_names = [
                name
                for name in args.sequence_name
                if name in grouped and (category_root / name / "pointcloud.ply").exists()
            ]
        else:
            seq_names = choose_sequences(
                category_root,
                grouped,
                sequence_quality,
                args.sequences_per_category,
                args.frames_per_sequence,
                args.seed,
                args.min_pointcloud_quality,
                args.min_viewpoint_quality,
            )
        if not seq_names:
            print(f"No eligible sequences under {category_root}")
            continue
        print(f"{category}: checking sequences {', '.join(seq_names)}")
        for seq_idx, seq_name in enumerate(seq_names):
            q = sequence_quality.get(seq_name, {})
            print(
                f"  {seq_name}: frames={len(grouped[seq_name])}, "
                f"pointcloud_quality={q.get('pointcloud_quality', float('nan')):.4g}, "
                f"viewpoint_quality={q.get('viewpoint_quality', float('nan')):.4g}"
            )
            frames = choose_frames(grouped[seq_name], args.frames_per_sequence, args.frame_sampling)
            ply_path = category_root / seq_name / "pointcloud.ply"
            points, colors = read_ply_vertices(ply_path, max_points=args.max_ply_points, seed=args.seed + seq_idx)
            object_center = np.nanmedian(points, axis=0)

            cameras = {
                cand.name: {frame.frame_number: camera_from_candidate(frame, cand) for frame in frames}
                for cand in candidates
            }
            for cand in candidates:
                for frame in frames:
                    row = {
                        "category": category,
                        "sequence": seq_name,
                        "frame": frame.frame_number,
                        "candidate": cand.name,
                        "description": cand.description,
                        "diagnostic_kind": "ply_projection",
                    }
                    row.update(
                        pointcloud_metrics(
                            frame,
                            cameras[cand.name][frame.frame_number],
                            points,
                            colors,
                            object_center,
                            args.visible_depth_rel_threshold,
                            args.silhouette_point_radius,
                            args.silhouette_fill_mode,
                            args.silhouette_close_radius,
                        )
                    )
                    all_rows.append(row)
                    self_row = {
                        "category": category,
                        "sequence": seq_name,
                        "frame": frame.frame_number,
                        "candidate": cand.name,
                        "description": cand.description,
                        "diagnostic_kind": "depth_self_projection",
                    }
                    self_row.update(
                        depth_self_projection_metrics(
                            frame,
                            cameras[cand.name][frame.frame_number],
                            max_points=args.max_depth_points,
                            rng=rng,
                        )
                    )
                    all_rows.append(self_row)

                for src, dst in zip(frames[:-1], frames[1:]):
                    row = {
                        "category": category,
                        "sequence": seq_name,
                        "src_frame": src.frame_number,
                        "dst_frame": dst.frame_number,
                        "candidate": cand.name,
                        "description": cand.description,
                        "diagnostic_kind": "depth_cross_frame_projection",
                    }
                    row.update(
                        depth_reprojection_metrics(
                            src,
                            dst,
                            cameras[cand.name][src.frame_number],
                            cameras[cand.name][dst.frame_number],
                            max_points=args.max_depth_points,
                            rng=rng,
                        )
                    )
                    all_rows.append(row)

            if seq_idx == 0:
                frame = frames[len(frames) // 2]
                pc_images = []
                pc_visible_images = []
                silhouette_images = []
                depth_images = []
                src, dst = frames[0], frames[-1]
                visible_depth = read_depth(frame.depth_path, frame.depth_scale)
                visible_mask = read_mask(frame.mask_path)
                for cand in candidates:
                    extri, intri = cameras[cand.name][frame.frame_number]
                    pc_images.append(
                        draw_projected_points(
                            frame.image_path,
                            points,
                            colors,
                            extri,
                            intri,
                            f"{category}/{seq_name} frame {frame.frame_number} | {cand.name}",
                            max_width=args.overlay_width,
                        )
                    )
                    pc_visible_images.append(
                        draw_projected_points(
                            frame.image_path,
                            points,
                            colors,
                            extri,
                            intri,
                            f"{category}/{seq_name} frame {frame.frame_number} | visible {cand.name}",
                            max_width=args.overlay_width,
                            visible_depth=visible_depth,
                            visible_mask=visible_mask,
                            visible_rel_threshold=args.visible_depth_rel_threshold,
                        )
                    )
                    silhouette_images.append(
                        draw_silhouette_comparison(
                            frame,
                            points,
                            extri,
                            intri,
                            f"{category}/{seq_name} frame {frame.frame_number} | mask {cand.name}",
                            max_width=args.overlay_width,
                            point_radius=args.silhouette_point_radius,
                            fill_mode=args.silhouette_fill_mode,
                            close_radius=args.silhouette_close_radius,
                        )
                    )
                    depth_images.append(
                        draw_depth_reprojection(
                            src,
                            dst,
                            cameras[cand.name][src.frame_number],
                            cameras[cand.name][dst.frame_number],
                            f"{category}/{seq_name} {src.frame_number}->{dst.frame_number} | {cand.name}",
                            max_points=args.max_depth_points,
                            max_width=args.overlay_width,
                            rng=rng,
                        )
                    )
                pc_sheet = make_contact_sheet(pc_images, columns=2)
                pc_out = output_dir / f"{category}_{seq_name}_pointcloud_projection_candidates.jpg"
                pc_sheet.save(pc_out, quality=92)
                image_outputs.append(pc_out)
                pc_visible_sheet = make_contact_sheet(pc_visible_images, columns=2)
                pc_visible_out = output_dir / f"{category}_{seq_name}_visible_pointcloud_projection_candidates.jpg"
                pc_visible_sheet.save(pc_visible_out, quality=92)
                image_outputs.append(pc_visible_out)
                silhouette_sheet = make_contact_sheet(silhouette_images, columns=2)
                silhouette_out = output_dir / f"{category}_{seq_name}_silhouette_mask_candidates.jpg"
                silhouette_sheet.save(silhouette_out, quality=92)
                image_outputs.append(silhouette_out)
                depth_sheet = make_contact_sheet(depth_images, columns=2)
                depth_out = output_dir / f"{category}_{seq_name}_depth_reprojection_candidates.jpg"
                depth_sheet.save(depth_out, quality=92)
                image_outputs.append(depth_out)

            if args.current_frame_overlays:
                current_name = "current_s_left_w2c"
                current_images = []
                current_visible_images = []
                for frame in frames:
                    frame_metrics = pointcloud_metrics(
                        frame,
                        cameras[current_name][frame.frame_number],
                        points,
                        colors,
                        object_center,
                        args.visible_depth_rel_threshold,
                        args.silhouette_point_radius,
                        args.silhouette_fill_mode,
                        args.silhouette_close_radius,
                    )
                    extri, intri = cameras[current_name][frame.frame_number]
                    current_images.append(
                        draw_projected_points(
                            frame.image_path,
                            points,
                            colors,
                            extri,
                            intri,
                            (
                                f"{category}/{seq_name} frame {frame.frame_number} | {current_name} | "
                                f"ply_hit={frame_metrics['ply_mask_hit_rate_all']:.3f}"
                            ),
                            max_width=args.overlay_width,
                        )
                    )
                    visible_depth = read_depth(frame.depth_path, frame.depth_scale)
                    visible_mask = read_mask(frame.mask_path)
                    current_visible_images.append(
                        draw_projected_points(
                            frame.image_path,
                            points,
                            colors,
                            extri,
                            intri,
                            (
                                f"{category}/{seq_name} frame {frame.frame_number} | visible {current_name} | "
                                f"ply_hit={frame_metrics['ply_mask_hit_rate_all']:.3f}"
                            ),
                            max_width=args.overlay_width,
                            visible_depth=visible_depth,
                            visible_mask=visible_mask,
                            visible_rel_threshold=args.visible_depth_rel_threshold,
                        )
                    )
                current_sheet = make_contact_sheet(current_images, columns=min(3, max(1, len(current_images))))
                current_out = output_dir / f"{category}_{seq_name}_current_pointcloud_frames.jpg"
                current_sheet.save(current_out, quality=92)
                image_outputs.append(current_out)
                current_visible_sheet = make_contact_sheet(current_visible_images, columns=min(3, max(1, len(current_visible_images))))
                current_visible_out = output_dir / f"{category}_{seq_name}_current_visible_pointcloud_frames.jpg"
                current_visible_sheet.save(current_visible_out, quality=92)
                image_outputs.append(current_visible_out)

    summary = aggregate_rows(all_rows)
    detail_csv = output_dir / "per_sequence_candidate_metrics.csv"
    summary_csv = output_dir / "summary_by_candidate.csv"
    write_csv(detail_csv, all_rows)
    write_csv(summary_csv, summary)
    write_html(output_dir / "index.html", summary, detail_csv, image_outputs)
    print(f"wrote {summary_csv}")
    print(f"wrote {detail_csv}")
    print(f"wrote {output_dir / 'index.html'}")
    if summary:
        print("top candidates:")
        for row in summary[: min(5, len(summary))]:
            print(
                f"  #{row['rank']} {row['candidate']}: score={row['diagnostic_score']:.4f}, "
                f"ply_mask_inside={row['ply_mask_hit_rate_inside']:.4f}, "
                f"sil_iou={row['ply_silhouette_iou']:.4f}, "
                f"depth_mask_inside={row['depth_mask_hit_rate_inside']:.4f}, "
                f"depth_rel_err={row['depth_median_relative_error']:.4f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--category-root",
        action="append",
        default=[],
        help="Path to a CO3D category directory containing frame_annotations.jgz.",
    )
    parser.add_argument("--output", default="output/co3d_camera_convention_check")
    parser.add_argument("--sequences-per-category", type=int, default=2)
    parser.add_argument("--frames-per-sequence", type=int, default=4)
    parser.add_argument("--frame-sampling", choices=["nearby", "spread"], default="nearby")
    parser.add_argument("--min-pointcloud-quality", type=float, default=0.0)
    parser.add_argument("--min-viewpoint-quality", type=float, default=0.0)
    parser.add_argument("--current-frame-overlays", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visible-depth-rel-threshold", type=float, default=0.05)
    parser.add_argument(
        "--silhouette-point-radius",
        type=int,
        default=5,
        help="Pixel radius used to rasterize projected PLY points into a silhouette mask.",
    )
    parser.add_argument(
        "--silhouette-fill-mode",
        choices=["splat", "contour", "hull"],
        default="contour",
        help="How to turn projected PLY points into a filled 2D mask.",
    )
    parser.add_argument(
        "--silhouette-close-radius",
        type=int,
        default=15,
        help="Morphological close radius for --silhouette-fill-mode contour.",
    )
    parser.add_argument("--sequence-name", action="append", default=[])
    parser.add_argument("--max-depth-points", type=int, default=8000)
    parser.add_argument("--max-ply-points", type=int, default=50000)
    parser.add_argument("--overlay-width", type=int, default=900)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not args.category_root:
        for path in [
            Path("dataset/co3d_apple/apple"),
            Path("dataset/co3d_toybus/toybus"),
        ]:
            if path.exists():
                args.category_root.append(str(path))
    return args


if __name__ == "__main__":
    run(parse_args())
