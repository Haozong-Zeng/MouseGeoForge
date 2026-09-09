#!/usr/bin/env python3
"""Refine a VGGT calibration into metric scale using manually selected points.

Inputs:
  Either one DATASET_DIR containing:
    frame_XXXXXX/          VGGT frame directories
    selected_points.json   video-level manual annotations from point_selector.py

  Or one or more FRAME_DIRs, each containing:
    calibration.toml      per-frame VGGT calibration
    selected_points.json  manual annotations from point_selector.py

The camera rig is static and the annotated points are stationary across the
video, so annotations from multiple frames are merged into a single bundle
adjustment with shared cameras and shared 3D points: the same point id in
different frames is treated as the same physical point.

The initialization and prior calibration is, in order of precedence:
  --calibration PATH  (e.g. an aggregated prior from aggregate_vggt_priors.py)
  first annotated frame's calibration.toml

Prior strengths can be calibrated from the cross-frame spread of VGGT
predictions via --prior-stats (prior_stats.json from aggregate_vggt_priors.py).

Outputs:
  metric_refinement/calibration_metric.toml
  metric_refinement/points_3d.csv
  metric_refinement/distance_report.csv
  metric_refinement/validation_report.csv
  metric_refinement/reprojection_report.csv
  metric_refinement/intrinsics_report.csv
  metric_refinement/annotation_outliers.csv
  metric_refinement/camera_positions.csv
  metric_refinement/camera_pairwise_distances.csv
  metric_refinement/loo_report.csv            (with --loo)
  metric_refinement/summary.json

The refinement fixes one origin camera to remove gauge freedom, optimizes the
other camera poses and all reconstructable 3D points, and uses entered
distances as metric constraints. Intrinsics and distortion can also be
optimized with priors via explicit command-line flags.

Prior modes (--prior-mode):
  legacy    rvec/tvec difference priors toward the (scaled) prior calibration.
            Reproduces the original behavior of this script.
  absolute  geodesic rotation priors plus camera-center priors. With
            --free-scale the prior rig scale is a free variable so scale error
            in the initial metric estimate is not baked into the prior.
  relative  gauge-invariant priors on pairwise relative rotations and baseline
            vectors. Constrains rig shape (what VGGT predicts well) without
            trusting VGGT's absolute pose in the origin frame.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from scipy.optimize import least_squares

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None


UNIT_TO_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0}
SHARED_INTRINSICS_KEY = "__shared__"
RAYLEIGH_MEDIAN = math.sqrt(2.0 * math.log(2.0))


@dataclass(frozen=True)
class Camera:
    index: int
    name: str
    size: tuple[int, int]
    matrix: np.ndarray
    distortions: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    fisheye: bool = False

    @property
    def rotation(self) -> np.ndarray:
        rotation, _ = cv2.Rodrigues(self.rvec.reshape(3).astype(np.float64))
        return rotation.astype(np.float64)

    @property
    def center(self) -> np.ndarray:
        return -self.rotation.T @ self.tvec.reshape(3)

    @property
    def projection(self) -> np.ndarray:
        return self.matrix @ np.hstack([self.rotation, self.tvec.reshape(3, 1)])


@dataclass(frozen=True)
class Observation:
    point_id: str
    camera_name: str
    xy: np.ndarray
    sigma_px: float
    frame: str = ""


@dataclass(frozen=True)
class DistanceConstraint:
    id: str
    point_a: str
    point_b: str
    target_mm: float
    sigma_mm: float
    use_for_refinement: bool
    unit: str
    value: float


@dataclass(frozen=True)
class RigidTemplate:
    id: str
    sigma_mm: float
    points: dict[str, np.ndarray]


@dataclass
class PriorConfig:
    mode: str
    free_scale: bool
    rotation_sigma_deg: dict[str, float]
    translation_sigma_mm: dict[str, float]
    pair_rotation_sigma_deg: dict[str, float]
    pair_baseline_sigma_mm: dict[str, float]
    focal_sigma_percent: dict[str, float]
    principal_sigma_px: dict[str, float]
    distortion_sigma: float


@dataclass
class Layout:
    camera_slices: dict[str, slice] = field(default_factory=dict)
    intrinsic_slices: dict[str, slice] = field(default_factory=dict)
    distortion_slices: dict[str, slice] = field(default_factory=dict)
    point_slices: dict[str, slice] = field(default_factory=dict)
    template_slices: dict[str, slice] = field(default_factory=dict)
    scale_index: int | None = None
    size: int = 0


def read_toml(path: Path) -> dict[str, Any]:
    if tomllib is None:
        raise RuntimeError("Python 3.11+ tomllib is required.")
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_calibration(path: Path) -> list[Camera]:
    data = read_toml(path)
    cameras: list[Camera] = []
    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        match = None
        if str(key).startswith("cam_"):
            try:
                match = int(str(key).split("_", 1)[1])
            except ValueError:
                match = None
        if match is None:
            continue
        name = str(value.get("name") or key)
        size_value = value.get("size") or [0, 0]
        camera = Camera(
            index=match,
            name=name,
            size=(int(size_value[0]), int(size_value[1])),
            matrix=np.asarray(value["matrix"], dtype=np.float64).reshape(3, 3),
            distortions=np.asarray(value.get("distortions") or [0, 0, 0, 0, 0], dtype=np.float64).reshape(-1),
            rvec=np.asarray(value["rotation"], dtype=np.float64).reshape(3),
            tvec=np.asarray(value["translation"], dtype=np.float64).reshape(3),
            fisheye=bool(value.get("fisheye", False)),
        )
        cameras.append(camera)
    cameras.sort(key=lambda cam: cam.index)
    if not cameras:
        raise ValueError(f"No cameras found in {path}")
    return cameras


def parse_annotation_file(
    path: Path,
    cameras: Sequence[Camera],
    default_reprojection_sigma_px: float,
    default_distance_sigma_mm: float,
    default_template_sigma_mm: float,
    frame_tag: str,
) -> tuple[list[Observation], list[DistanceConstraint], list[RigidTemplate]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    camera_by_name = {cam.name: cam for cam in cameras}
    selected_camera_sizes: dict[str, tuple[int, int, int, int]] = {}
    for cam in data.get("cameras", []):
        if not isinstance(cam, dict):
            continue
        name = str(cam.get("name") or "")
        image_size = cam.get("image_size") or [None, None]
        source_size = cam.get("source_size") or [None, None]
        if name:
            selected_camera_sizes[name] = (
                int(image_size[0] or 0),
                int(image_size[1] or 0),
                int(source_size[0] or 0),
                int(source_size[1] or 0),
            )

    observations: list[Observation] = []
    for point in data.get("points", []):
        if not isinstance(point, dict):
            continue
        point_id = str(point.get("id") or "").strip()
        if not point_id:
            continue
        try:
            point_sigma_px = float(point.get("sigma_px", default_reprojection_sigma_px))
        except (TypeError, ValueError):
            point_sigma_px = default_reprojection_sigma_px
        if point_sigma_px <= 0:
            point_sigma_px = default_reprojection_sigma_px
        per_camera = point.get("observations") or {}
        if not isinstance(per_camera, dict):
            continue
        for cam_name, obs in per_camera.items():
            if cam_name not in camera_by_name or not isinstance(obs, dict):
                continue
            try:
                if "source_x" in obs and "source_y" in obs:
                    xy = np.array([float(obs["source_x"]), float(obs["source_y"])], dtype=np.float64)
                else:
                    image_w, image_h, source_w, source_h = selected_camera_sizes.get(cam_name, (0, 0, 0, 0))
                    if image_w <= 0 or image_h <= 0:
                        raise ValueError("missing image/source mapping")
                    xy = np.array(
                        [
                            float(obs["x"]) * source_w / image_w,
                            float(obs["y"]) * source_h / image_h,
                        ],
                        dtype=np.float64,
                    )
                try:
                    sigma_px = float(obs.get("sigma_px", point_sigma_px))
                except (AttributeError, TypeError, ValueError):
                    sigma_px = point_sigma_px
                if sigma_px <= 0:
                    sigma_px = point_sigma_px
            except (KeyError, TypeError, ValueError):
                continue
            observations.append(
                Observation(
                    point_id=point_id,
                    camera_name=str(cam_name),
                    xy=xy,
                    sigma_px=sigma_px,
                    frame=frame_tag,
                )
            )

    distances: list[DistanceConstraint] = []
    for idx, item in enumerate(data.get("distances", []), start=1):
        if not isinstance(item, dict):
            continue
        point_a = str(item.get("point_a") or "").strip()
        point_b = str(item.get("point_b") or "").strip()
        if not point_a or not point_b or point_a == point_b:
            continue
        unit = str(item.get("unit") or "mm").lower()
        sigma_unit = str(item.get("sigma_unit") or unit).lower()
        try:
            value = float(item.get("value"))
            target_mm = float(item.get("distance_mm", value * UNIT_TO_MM.get(unit, 1.0)))
        except (TypeError, ValueError):
            continue
        try:
            if "sigma_mm" in item:
                sigma_mm = float(item["sigma_mm"])
            else:
                sigma_value = float(item.get("sigma_value", default_distance_sigma_mm / UNIT_TO_MM.get(sigma_unit, 1.0)))
                sigma_mm = sigma_value * UNIT_TO_MM.get(sigma_unit, 1.0)
        except (TypeError, ValueError):
            sigma_mm = default_distance_sigma_mm
        if sigma_mm <= 0:
            sigma_mm = default_distance_sigma_mm
        if value <= 0 or target_mm <= 0:
            continue
        use_for_refinement = bool(item.get("use_for_refinement", not bool(item.get("holdout", False))))
        distances.append(
            DistanceConstraint(
                id=str(item.get("id") or f"D{idx:03d}"),
                point_a=point_a,
                point_b=point_b,
                target_mm=target_mm,
                sigma_mm=sigma_mm,
                use_for_refinement=use_for_refinement,
                unit=unit,
                value=value,
            )
        )

    templates: list[RigidTemplate] = []
    for idx, item in enumerate(data.get("templates", []), start=1):
        if not isinstance(item, dict):
            continue
        raw_points = item.get("points") or {}
        if not isinstance(raw_points, dict):
            continue
        template_points: dict[str, np.ndarray] = {}
        for pid, xyz in raw_points.items():
            try:
                template_points[str(pid)] = np.asarray(xyz, dtype=np.float64).reshape(3)
            except (TypeError, ValueError):
                continue
        if len(template_points) < 3:
            continue
        try:
            sigma_mm = float(item.get("sigma_mm", default_template_sigma_mm))
        except (TypeError, ValueError):
            sigma_mm = default_template_sigma_mm
        if sigma_mm <= 0:
            sigma_mm = default_template_sigma_mm
        templates.append(
            RigidTemplate(
                id=str(item.get("id") or f"T{idx:03d}"),
                sigma_mm=sigma_mm,
                points=template_points,
            )
        )

    return observations, distances, templates


def merge_annotations(
    annotation_sets: Sequence[tuple[list[Observation], list[DistanceConstraint], list[RigidTemplate]]],
) -> tuple[list[str], list[Observation], list[DistanceConstraint], list[RigidTemplate]]:
    observations: list[Observation] = []
    for obs_list, _, _ in annotation_sets:
        observations.extend(obs_list)

    cameras_per_point: dict[str, set[str]] = {}
    order: list[str] = []
    for obs in observations:
        if obs.point_id not in cameras_per_point:
            cameras_per_point[obs.point_id] = set()
            order.append(obs.point_id)
        cameras_per_point[obs.point_id].add(obs.camera_name)
    point_ids = [pid for pid in order if len(cameras_per_point[pid]) >= 2]
    valid = set(point_ids)
    observations = [obs for obs in observations if obs.point_id in valid]
    if not point_ids:
        raise ValueError("No points with at least two calibrated-camera observations")

    distances: list[DistanceConstraint] = []
    seen_pairs: dict[frozenset[str], DistanceConstraint] = {}
    for _, distance_list, _ in annotation_sets:
        for dist in distance_list:
            if dist.point_a not in valid or dist.point_b not in valid:
                continue
            pair = frozenset((dist.point_a, dist.point_b))
            existing = seen_pairs.get(pair)
            if existing is None:
                seen_pairs[pair] = dist
                distances.append(dist)
            elif abs(existing.target_mm - dist.target_mm) > 1e-6:
                print(
                    f"Warning: duplicate distance {dist.point_a}-{dist.point_b} with conflicting targets "
                    f"({existing.target_mm} vs {dist.target_mm} mm); keeping the first."
                )

    templates: list[RigidTemplate] = []
    seen_templates: set[str] = set()
    for _, _, template_list in annotation_sets:
        for template in template_list:
            matched = {pid: xyz for pid, xyz in template.points.items() if pid in valid}
            if len(matched) < 3 or template.id in seen_templates:
                continue
            seen_templates.add(template.id)
            templates.append(RigidTemplate(id=template.id, sigma_mm=template.sigma_mm, points=matched))

    return point_ids, observations, distances, templates


def undistort_to_normalized(camera: Camera, xy: np.ndarray) -> np.ndarray:
    points = np.asarray(xy, dtype=np.float64).reshape(1, 1, 2)
    dist = np.asarray(camera.distortions, dtype=np.float64).reshape(-1)
    if camera.fisheye:
        fisheye_dist = np.zeros(4, dtype=np.float64)
        fisheye_dist[: min(4, dist.size)] = dist[:4]
        normalized = cv2.fisheye.undistortPoints(points, camera.matrix, fisheye_dist.reshape(4, 1))
    else:
        normalized = cv2.undistortPoints(points, camera.matrix, dist)
    return normalized.reshape(2).astype(np.float64)


def triangulate_point(
    observations: Sequence[Observation],
    cameras_by_name: dict[str, Camera],
    method: str = "pixel",
) -> np.ndarray:
    rows = []
    for obs in observations:
        camera = cameras_by_name[obs.camera_name]
        if method == "normalized":
            x, y = undistort_to_normalized(camera, obs.xy)
            projection = np.hstack([camera.rotation, camera.tvec.reshape(3, 1)])
        else:
            x, y = obs.xy
            projection = camera.projection
        rows.append(x * projection[2, :] - projection[0, :])
        rows.append(y * projection[2, :] - projection[1, :])
    system = np.asarray(rows, dtype=np.float64)
    _, _, vt = np.linalg.svd(system)
    homogeneous = vt[-1]
    if abs(float(homogeneous[3])) <= 1e-12:
        raise np.linalg.LinAlgError("homogeneous coordinate is near zero")
    return homogeneous[:3] / homogeneous[3]


def template_pseudo_distances(templates: Sequence[RigidTemplate]) -> list[DistanceConstraint]:
    pseudo: list[DistanceConstraint] = []
    for template in templates:
        ids = sorted(template.points)
        for i, pid_a in enumerate(ids):
            for pid_b in ids[i + 1 :]:
                target = float(np.linalg.norm(template.points[pid_a] - template.points[pid_b]))
                if target <= 0:
                    continue
                pseudo.append(
                    DistanceConstraint(
                        id=f"{template.id}:{pid_a}-{pid_b}",
                        point_a=pid_a,
                        point_b=pid_b,
                        target_mm=target,
                        sigma_mm=template.sigma_mm,
                        use_for_refinement=True,
                        unit="mm",
                        value=target,
                    )
                )
    return pseudo


def estimate_metric_scale(points_raw: dict[str, np.ndarray], distances: Sequence[DistanceConstraint]) -> float:
    raw_lengths = []
    target_lengths = []
    weights = []
    for dist in distances:
        if dist.point_a not in points_raw or dist.point_b not in points_raw:
            continue
        raw_length = float(np.linalg.norm(points_raw[dist.point_a] - points_raw[dist.point_b]))
        if raw_length > 1e-12:
            raw_lengths.append(raw_length)
            target_lengths.append(dist.target_mm)
            weights.append(1.0 / max(dist.sigma_mm, 1e-9) ** 2)
    if not raw_lengths:
        return 1.0
    raw = np.asarray(raw_lengths, dtype=np.float64)
    target = np.asarray(target_lengths, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    denom = float(np.sum(weight * raw * raw))
    return float(np.sum(weight * raw * target) / denom) if denom > 1e-12 else 1.0


def intrinsic_vector(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(
        [matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]],
        dtype=np.float64,
    )


def distortion_vector(camera: Camera) -> np.ndarray:
    values = np.asarray(camera.distortions, dtype=np.float64).reshape(-1)
    if camera.fisheye:
        out = np.zeros(4, dtype=np.float64)
        out[: min(4, values.size)] = values[:4]
        return out
    if values.size >= 4:
        return values.copy()
    out = np.zeros(5, dtype=np.float64)
    out[: values.size] = values
    return out


def distortion_free_count(mode: str, total: int) -> int:
    if mode == "k1":
        return min(1, total)
    if mode == "k1k2":
        return min(2, total)
    return total


def rigid_align(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    cross = (source - source_mean).T @ (target - target_mean)
    u, _, vt = np.linalg.svd(cross)
    rotation = vt.T @ np.diag([1.0, 1.0, float(np.linalg.det(vt.T @ u.T))]) @ u.T
    translation = target_mean - rotation @ source_mean
    return rotation, translation


def project(
    camera: Camera,
    rvec: np.ndarray,
    tvec: np.ndarray,
    point: np.ndarray,
    matrix: np.ndarray | None = None,
    distortions: np.ndarray | None = None,
) -> np.ndarray:
    camera_matrix = np.asarray(matrix if matrix is not None else camera.matrix, dtype=np.float64).reshape(3, 3)
    dist = np.asarray(distortions if distortions is not None else camera.distortions, dtype=np.float64).reshape(-1)
    xyz = np.asarray(point, dtype=np.float64).reshape(1, 1, 3)
    if camera.fisheye:
        fisheye_dist = np.zeros(4, dtype=np.float64)
        fisheye_dist[: min(4, dist.size)] = dist[:4]
        image_points, _ = cv2.fisheye.projectPoints(
            xyz,
            rvec.reshape(3).astype(np.float64),
            tvec.reshape(3).astype(np.float64),
            camera_matrix,
            fisheye_dist.reshape(4, 1),
        )
    else:
        image_points, _ = cv2.projectPoints(
            xyz,
            rvec.reshape(3).astype(np.float64),
            tvec.reshape(3).astype(np.float64),
            camera_matrix,
            dist,
        )
    return image_points.reshape(-1, 2)[0].astype(np.float64)


def camera_from_params(
    camera: Camera,
    rvec: np.ndarray,
    tvec: np.ndarray,
    matrix: np.ndarray | None = None,
    distortions: np.ndarray | None = None,
) -> Camera:
    return Camera(
        index=camera.index,
        name=camera.name,
        size=camera.size,
        matrix=np.asarray(matrix if matrix is not None else camera.matrix, dtype=np.float64).reshape(3, 3),
        distortions=np.asarray(distortions if distortions is not None else camera.distortions, dtype=np.float64).reshape(-1),
        rvec=rvec.reshape(3),
        tvec=tvec.reshape(3),
        fisheye=camera.fisheye,
    )


def rotation_delta_deg(a: np.ndarray, b: np.ndarray) -> float:
    ra, _ = cv2.Rodrigues(a.reshape(3))
    rb, _ = cv2.Rodrigues(b.reshape(3))
    value = np.clip((np.trace(ra.T @ rb) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(value)))


def geodesic_residual(rotation_ref: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    delta, _ = cv2.Rodrigues(rotation_ref.T @ rotation)
    return delta.reshape(3)


def pair_key(name_a: str, name_b: str) -> str:
    return f"{name_a}-{name_b}"


def resolve_prior_config(
    args: argparse.Namespace,
    cameras: Sequence[Camera],
    prior_stats: dict[str, Any] | None,
    scale_to_mm: float,
) -> PriorConfig:
    names = [camera.name for camera in cameras]
    stats_cameras = (prior_stats or {}).get("cameras", {})
    stats_pairs = (prior_stats or {}).get("pairs", {})
    sigma_scale = max(args.prior_sigma_scale, 1e-6)

    rotation_sigma_deg: dict[str, float] = {}
    translation_sigma_mm: dict[str, float] = {}
    focal_sigma_percent: dict[str, float] = {}
    principal_sigma_px: dict[str, float] = {}
    for name in names:
        cam_stats = stats_cameras.get(name, {})
        if args.rotation_prior_deg is not None:
            rotation_sigma_deg[name] = args.rotation_prior_deg
        elif "rotation_sigma_deg" in cam_stats:
            rotation_sigma_deg[name] = max(float(cam_stats["rotation_sigma_deg"]), 0.05)
        else:
            rotation_sigma_deg[name] = 3.0
        if args.translation_prior_mm is not None:
            translation_sigma_mm[name] = args.translation_prior_mm
        elif "center_sigma" in cam_stats:
            translation_sigma_mm[name] = max(float(cam_stats["center_sigma"]) * scale_to_mm, 0.1)
        else:
            translation_sigma_mm[name] = 25.0
        if args.intrinsics_prior_percent is not None:
            focal_sigma_percent[name] = args.intrinsics_prior_percent
        elif "fx_sigma_percent" in cam_stats:
            focal_sigma_percent[name] = max(
                0.5,
                0.5 * (float(cam_stats["fx_sigma_percent"]) + float(cam_stats.get("fy_sigma_percent", cam_stats["fx_sigma_percent"]))),
            )
        else:
            focal_sigma_percent[name] = 10.0
        if args.principal_point_prior_px is not None:
            principal_sigma_px[name] = args.principal_point_prior_px
        elif "principal_point_sigma_px" in cam_stats:
            principal_sigma_px[name] = max(float(cam_stats["principal_point_sigma_px"]), 1.0)
        else:
            principal_sigma_px[name] = 50.0
        rotation_sigma_deg[name] *= sigma_scale
        translation_sigma_mm[name] *= sigma_scale

    pair_rotation_sigma_deg: dict[str, float] = {}
    pair_baseline_sigma_mm: dict[str, float] = {}
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            key = pair_key(name_a, name_b)
            pair_stats = stats_pairs.get(key) or stats_pairs.get(pair_key(name_b, name_a)) or {}
            if args.rotation_prior_deg is None and "relative_rotation_sigma_deg" in pair_stats:
                pair_rotation_sigma_deg[key] = max(float(pair_stats["relative_rotation_sigma_deg"]), 0.05) * sigma_scale
            else:
                pair_rotation_sigma_deg[key] = math.hypot(rotation_sigma_deg[name_a], rotation_sigma_deg[name_b])
            if args.translation_prior_mm is None and "baseline_vector_sigma" in pair_stats:
                pair_baseline_sigma_mm[key] = max(float(pair_stats["baseline_vector_sigma"]) * scale_to_mm, 0.1) * sigma_scale
            else:
                pair_baseline_sigma_mm[key] = math.hypot(translation_sigma_mm[name_a], translation_sigma_mm[name_b])

    return PriorConfig(
        mode=args.prior_mode,
        free_scale=bool(args.free_scale),
        rotation_sigma_deg=rotation_sigma_deg,
        translation_sigma_mm=translation_sigma_mm,
        pair_rotation_sigma_deg=pair_rotation_sigma_deg,
        pair_baseline_sigma_mm=pair_baseline_sigma_mm,
        focal_sigma_percent=focal_sigma_percent,
        principal_sigma_px=principal_sigma_px,
        distortion_sigma=args.distortion_prior if args.distortion_prior is not None else 0.05,
    )


def pack_variables(
    cameras: Sequence[Camera],
    point_ids: Sequence[str],
    templates: Sequence[RigidTemplate],
    init_rvecs: dict[str, np.ndarray],
    init_tvecs: dict[str, np.ndarray],
    init_matrices: dict[str, np.ndarray],
    init_distortions: dict[str, np.ndarray],
    init_points: dict[str, np.ndarray],
    template_init_poses: dict[str, tuple[np.ndarray, np.ndarray]],
    origin: str,
    refine_intrinsics: bool,
    intrinsics_mode: str,
    refine_distortion: bool,
    distortion_mode: str,
    free_scale: bool,
) -> tuple[np.ndarray, Layout]:
    values: list[float] = []
    layout = Layout()
    offset = 0
    for camera in cameras:
        if camera.name != origin:
            block = np.concatenate([init_rvecs[camera.name], init_tvecs[camera.name]])
            values.extend(block.tolist())
            layout.camera_slices[camera.name] = slice(offset, offset + 6)
            offset += 6
        if refine_intrinsics and intrinsics_mode != "f_shared":
            if intrinsics_mode == "fxfy_pp":
                block = intrinsic_vector(init_matrices[camera.name])
            elif intrinsics_mode == "fxfy":
                block = intrinsic_vector(init_matrices[camera.name])[:2]
            else:  # "f": one multiplicative gain on fx and fy
                block = np.asarray([1.0], dtype=np.float64)
            values.extend(block.tolist())
            layout.intrinsic_slices[camera.name] = slice(offset, offset + block.size)
            offset += block.size
        if refine_distortion:
            total = init_distortions[camera.name].reshape(-1).size
            count = distortion_free_count(distortion_mode, total)
            block = init_distortions[camera.name].reshape(-1)[:count]
            values.extend(block.tolist())
            layout.distortion_slices[camera.name] = slice(offset, offset + count)
            offset += count
    if refine_intrinsics and intrinsics_mode == "f_shared":
        values.append(1.0)
        layout.intrinsic_slices[SHARED_INTRINSICS_KEY] = slice(offset, offset + 1)
        offset += 1
    for point_id in point_ids:
        values.extend(init_points[point_id].tolist())
        layout.point_slices[point_id] = slice(offset, offset + 3)
        offset += 3
    for template in templates:
        rvec, tvec = template_init_poses[template.id]
        values.extend(rvec.reshape(3).tolist())
        values.extend(tvec.reshape(3).tolist())
        layout.template_slices[template.id] = slice(offset, offset + 6)
        offset += 6
    if free_scale:
        values.append(0.0)
        layout.scale_index = offset
        offset += 1
    layout.size = offset
    return np.asarray(values, dtype=np.float64), layout


def unpack_variables(
    x: np.ndarray,
    cameras: Sequence[Camera],
    point_ids: Sequence[str],
    layout: Layout,
    init_rvecs: dict[str, np.ndarray],
    init_tvecs: dict[str, np.ndarray],
    init_matrices: dict[str, np.ndarray],
    init_distortions: dict[str, np.ndarray],
    intrinsics_mode: str,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, tuple[np.ndarray, np.ndarray]],
    float,
]:
    rvecs = {name: value.copy() for name, value in init_rvecs.items()}
    tvecs = {name: value.copy() for name, value in init_tvecs.items()}
    matrices = {name: value.copy() for name, value in init_matrices.items()}
    distortions = {name: value.copy() for name, value in init_distortions.items()}
    shared_slice = layout.intrinsic_slices.get(SHARED_INTRINSICS_KEY)
    shared_gain = float(x[shared_slice][0]) if shared_slice is not None else None
    for camera in cameras:
        sl = layout.camera_slices.get(camera.name)
        if sl is not None:
            block = x[sl]
            rvecs[camera.name] = block[:3]
            tvecs[camera.name] = block[3:6]
        if shared_gain is not None:
            matrix = init_matrices[camera.name].copy()
            matrix[0, 0] *= shared_gain
            matrix[1, 1] *= shared_gain
            matrices[camera.name] = matrix
        else:
            intrinsic_sl = layout.intrinsic_slices.get(camera.name)
            if intrinsic_sl is not None:
                block = x[intrinsic_sl]
                matrix = init_matrices[camera.name].copy()
                if intrinsics_mode == "fxfy_pp":
                    matrix[0, 0] = float(block[0])
                    matrix[1, 1] = float(block[1])
                    matrix[0, 2] = float(block[2])
                    matrix[1, 2] = float(block[3])
                elif intrinsics_mode == "fxfy":
                    matrix[0, 0] = float(block[0])
                    matrix[1, 1] = float(block[1])
                else:
                    matrix[0, 0] *= float(block[0])
                    matrix[1, 1] *= float(block[0])
                matrices[camera.name] = matrix
        distortion_sl = layout.distortion_slices.get(camera.name)
        if distortion_sl is not None:
            block = x[distortion_sl]
            full = init_distortions[camera.name].copy()
            full[: block.size] = block
            distortions[camera.name] = full
    points = {point_id: x[layout.point_slices[point_id]] for point_id in point_ids}
    template_poses = {
        template_id: (x[sl][:3], x[sl][3:6]) for template_id, sl in layout.template_slices.items()
    }
    log_scale = float(x[layout.scale_index]) if layout.scale_index is not None else 0.0
    return rvecs, tvecs, matrices, distortions, points, template_poses, log_scale


def build_residual_function(
    cameras: Sequence[Camera],
    observations: Sequence[Observation],
    distances: Sequence[DistanceConstraint],
    templates: Sequence[RigidTemplate],
    point_ids: Sequence[str],
    layout: Layout,
    init_rvecs: dict[str, np.ndarray],
    init_tvecs: dict[str, np.ndarray],
    init_matrices: dict[str, np.ndarray],
    init_distortions: dict[str, np.ndarray],
    prior: PriorConfig,
    intrinsics_mode: str,
    origin: str,
):
    cameras_by_name = {camera.name: camera for camera in cameras}
    names = [camera.name for camera in cameras]
    init_rotations = {name: cv2.Rodrigues(init_rvecs[name].reshape(3))[0] for name in names}
    init_centers = {name: -init_rotations[name].T @ init_tvecs[name].reshape(3) for name in names}
    rot_sigma_rad = {name: math.radians(prior.rotation_sigma_deg[name]) for name in names}
    pair_rot_sigma_rad = {key: math.radians(value) for key, value in prior.pair_rotation_sigma_deg.items()}

    def residuals(x: np.ndarray) -> np.ndarray:
        rvecs, tvecs, matrices, distortions, points, template_poses, log_scale = unpack_variables(
            x,
            cameras,
            point_ids,
            layout,
            init_rvecs,
            init_tvecs,
            init_matrices,
            init_distortions,
            intrinsics_mode,
        )
        prior_scale = math.exp(log_scale) if prior.free_scale else 1.0
        out: list[float] = []
        for obs in observations:
            if obs.point_id not in points:
                continue
            camera = cameras_by_name[obs.camera_name]
            try:
                matrix = matrices[camera.name]
                if matrix[0, 0] <= 1.0 or matrix[1, 1] <= 1.0:
                    raise FloatingPointError("invalid focal length")
                pred = project(
                    camera,
                    rvecs[camera.name],
                    tvecs[camera.name],
                    points[obs.point_id],
                    matrix=matrix,
                    distortions=distortions[camera.name],
                )
            except (FloatingPointError, cv2.error):
                pred = np.array([1e6, 1e6], dtype=np.float64)
            if not np.all(np.isfinite(pred)):
                pred = np.array([1e6, 1e6], dtype=np.float64)
            out.extend(((pred - obs.xy) / max(obs.sigma_px, 1e-9)).tolist())

        for dist in distances:
            if dist.point_a not in points or dist.point_b not in points:
                continue
            length = float(np.linalg.norm(points[dist.point_a] - points[dist.point_b]))
            out.append((length - dist.target_mm) / max(dist.sigma_mm, 1e-9))

        for template in templates:
            rvec_t, tvec_t = template_poses[template.id]
            rotation_t, _ = cv2.Rodrigues(rvec_t.reshape(3))
            sigma = max(template.sigma_mm, 1e-9)
            for pid, xyz in template.points.items():
                if pid not in points:
                    continue
                predicted = rotation_t @ xyz + tvec_t
                out.extend(((predicted - points[pid]) / sigma).tolist())

        if prior.mode == "legacy":
            for camera in cameras:
                if camera.name not in layout.camera_slices:
                    continue
                name = camera.name
                out.extend(((rvecs[name] - init_rvecs[name]) / rot_sigma_rad[name]).tolist())
                out.extend(((tvecs[name] - prior_scale * init_tvecs[name]) / prior.translation_sigma_mm[name]).tolist())
        elif prior.mode == "absolute":
            for camera in cameras:
                if camera.name not in layout.camera_slices:
                    continue
                name = camera.name
                rotation, _ = cv2.Rodrigues(rvecs[name].reshape(3))
                out.extend((geodesic_residual(init_rotations[name], rotation) / rot_sigma_rad[name]).tolist())
                center = -rotation.T @ tvecs[name].reshape(3)
                out.extend(((center - prior_scale * init_centers[name]) / prior.translation_sigma_mm[name]).tolist())
        else:  # relative
            rotations = {name: cv2.Rodrigues(rvecs[name].reshape(3))[0] for name in names}
            centers = {name: -rotations[name].T @ tvecs[name].reshape(3) for name in names}
            for i, name_a in enumerate(names):
                for name_b in names[i + 1 :]:
                    key = pair_key(name_a, name_b)
                    relative = rotations[name_a] @ rotations[name_b].T
                    relative_init = init_rotations[name_a] @ init_rotations[name_b].T
                    out.extend((geodesic_residual(relative_init, relative) / pair_rot_sigma_rad[key]).tolist())
                    baseline = rotations[name_a] @ (centers[name_b] - centers[name_a])
                    baseline_init = init_rotations[name_a] @ (init_centers[name_b] - init_centers[name_a])
                    out.extend(((baseline - prior_scale * baseline_init) / prior.pair_baseline_sigma_mm[key]).tolist())

        shared_slice = layout.intrinsic_slices.get(SHARED_INTRINSICS_KEY)
        if shared_slice is not None:
            gain = float(x[shared_slice][0])
            mean_percent = float(np.mean([prior.focal_sigma_percent[name] for name in names]))
            out.append((gain - 1.0) / max(mean_percent / 100.0, 1e-9))
        else:
            for camera in cameras:
                sl = layout.intrinsic_slices.get(camera.name)
                if sl is None:
                    continue
                name = camera.name
                init_vec = intrinsic_vector(init_matrices[name])
                current_vec = intrinsic_vector(matrices[name])
                focal_fraction = max(prior.focal_sigma_percent[name], 1e-9) / 100.0
                out.append((current_vec[0] - init_vec[0]) / max(abs(init_vec[0]) * focal_fraction, 1e-9))
                out.append((current_vec[1] - init_vec[1]) / max(abs(init_vec[1]) * focal_fraction, 1e-9))
                if intrinsics_mode == "fxfy_pp":
                    out.append((current_vec[2] - init_vec[2]) / max(prior.principal_sigma_px[name], 1e-9))
                    out.append((current_vec[3] - init_vec[3]) / max(prior.principal_sigma_px[name], 1e-9))
        for camera in cameras:
            sl = layout.distortion_slices.get(camera.name)
            if sl is None:
                continue
            name = camera.name
            free = x[sl]
            init_free = init_distortions[name].reshape(-1)[: free.size]
            out.extend(((free - init_free) / max(prior.distortion_sigma, 1e-12)).tolist())
        return np.asarray(out, dtype=np.float64)

    return residuals


@dataclass
class Solution:
    rvecs: dict[str, np.ndarray]
    tvecs: dict[str, np.ndarray]
    matrices: dict[str, np.ndarray]
    distortions: dict[str, np.ndarray]
    points: dict[str, np.ndarray]
    template_poses: dict[str, tuple[np.ndarray, np.ndarray]]
    log_scale: float
    observations: list[Observation]
    dropped_observations: list[Observation]
    scale_to_mm: float
    scale_source: str
    init_rvecs: dict[str, np.ndarray]
    init_tvecs: dict[str, np.ndarray]
    init_matrices: dict[str, np.ndarray]
    init_distortions: dict[str, np.ndarray]
    init_points: dict[str, np.ndarray]
    prior: PriorConfig
    layout: Layout
    result: Any
    initial_cost: float
    sigma_refit_factors: list[float]
    covariance: np.ndarray | None


def solve_once(
    cameras: Sequence[Camera],
    point_ids: Sequence[str],
    observations: Sequence[Observation],
    fit_distances: Sequence[DistanceConstraint],
    templates: Sequence[RigidTemplate],
    points_raw: dict[str, np.ndarray],
    prior_stats: dict[str, Any] | None,
    origin: str,
    args: argparse.Namespace,
    compute_covariance: bool = False,
) -> Solution:
    point_ids = list(point_ids)
    observations = list(observations)
    fit_distances = list(fit_distances)

    scale_distances = fit_distances if fit_distances else template_pseudo_distances(templates)
    scale_to_mm = estimate_metric_scale(points_raw, scale_distances)
    if fit_distances:
        scale_source = "fit_distances"
    elif templates:
        scale_source = "templates"
    else:
        scale_source = "none"

    init_points = {point_id: points_raw[point_id] * scale_to_mm for point_id in point_ids}
    init_rvecs = {camera.name: camera.rvec.copy() for camera in cameras}
    init_tvecs = {camera.name: camera.tvec.copy() * scale_to_mm for camera in cameras}
    init_matrices = {camera.name: camera.matrix.copy() for camera in cameras}
    init_distortions = {camera.name: distortion_vector(camera) for camera in cameras}

    template_init_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for template in templates:
        matched = [pid for pid in template.points if pid in init_points]
        source = np.asarray([template.points[pid] for pid in matched])
        target = np.asarray([init_points[pid] for pid in matched])
        rotation, translation = rigid_align(source, target)
        rvec, _ = cv2.Rodrigues(rotation)
        template_init_poses[template.id] = (rvec.reshape(3), translation.reshape(3))

    prior = resolve_prior_config(args, cameras, prior_stats, scale_to_mm)

    def build_and_solve(
        current_observations: Sequence[Observation],
        x_start: np.ndarray | None,
    ) -> tuple[Any, np.ndarray, Layout, float]:
        x0, layout = pack_variables(
            cameras,
            point_ids,
            templates,
            init_rvecs,
            init_tvecs,
            init_matrices,
            init_distortions,
            init_points,
            template_init_poses,
            origin=origin,
            refine_intrinsics=args.refine_intrinsics,
            intrinsics_mode=args.intrinsics_mode,
            refine_distortion=args.refine_distortion,
            distortion_mode=args.distortion_mode,
            free_scale=args.free_scale,
        )
        if x_start is not None and x_start.size == x0.size:
            x0 = x_start
        residual_func = build_residual_function(
            cameras=cameras,
            observations=current_observations,
            distances=fit_distances,
            templates=templates,
            point_ids=point_ids,
            layout=layout,
            init_rvecs=init_rvecs,
            init_tvecs=init_tvecs,
            init_matrices=init_matrices,
            init_distortions=init_distortions,
            prior=prior,
            intrinsics_mode=args.intrinsics_mode,
            origin=origin,
        )
        initial_cost = float(0.5 * np.sum(residual_func(x0) ** 2))
        result = least_squares(
            residual_func,
            x0,
            loss=args.loss,
            f_scale=args.loss_f_scale,
            max_nfev=args.max_nfev,
            verbose=2 if args.verbose else 0,
        )
        return result, result.x, layout, initial_cost

    current_observations = list(observations)
    result, x, layout, initial_cost = build_and_solve(current_observations, None)

    sigma_refit_factors: list[float] = []
    for _ in range(max(args.sigma_refit_iters, 0)):
        refined = unpack_variables(
            x, cameras, point_ids, layout, init_rvecs, init_tvecs, init_matrices, init_distortions, args.intrinsics_mode
        )
        rvecs, tvecs, matrices, distortions, points = refined[:5]
        normalized = []
        for obs in current_observations:
            pred = project(
                next(c for c in cameras if c.name == obs.camera_name),
                rvecs[obs.camera_name],
                tvecs[obs.camera_name],
                points[obs.point_id],
                matrix=matrices[obs.camera_name],
                distortions=distortions[obs.camera_name],
            )
            normalized.append(float(np.linalg.norm(pred - obs.xy)) / max(obs.sigma_px, 1e-9))
        factor = float(np.clip(np.median(normalized) / RAYLEIGH_MEDIAN, 0.25, 4.0)) if normalized else 1.0
        sigma_refit_factors.append(factor)
        current_observations = [
            replace(obs, sigma_px=max(obs.sigma_px * factor, 0.05)) for obs in current_observations
        ]
        result, x, layout, _ = build_and_solve(current_observations, x)

    dropped_observations: list[Observation] = []
    if args.drop_outlier_obs_z > 0:
        refined = unpack_variables(
            x, cameras, point_ids, layout, init_rvecs, init_tvecs, init_matrices, init_distortions, args.intrinsics_mode
        )
        rvecs, tvecs, matrices, distortions, points = refined[:5]
        scored: list[tuple[float, Observation]] = []
        for obs in current_observations:
            pred = project(
                next(c for c in cameras if c.name == obs.camera_name),
                rvecs[obs.camera_name],
                tvecs[obs.camera_name],
                points[obs.point_id],
                matrix=matrices[obs.camera_name],
                distortions=distortions[obs.camera_name],
            )
            scored.append((float(np.linalg.norm(pred - obs.xy)) / max(obs.sigma_px, 1e-9), obs))
        cameras_per_point: dict[str, set[str]] = {}
        for _, obs in scored:
            cameras_per_point.setdefault(obs.point_id, set()).add(obs.camera_name)
        keep: list[Observation] = []
        for score, obs in sorted(scored, key=lambda item: -item[0]):
            if score > args.drop_outlier_obs_z and len(cameras_per_point[obs.point_id]) > 2:
                dropped_observations.append(obs)
                cameras_per_point[obs.point_id].discard(obs.camera_name)
            else:
                keep.append(obs)
        if dropped_observations:
            current_observations = keep
            result, x, layout, _ = build_and_solve(current_observations, x)

    refined_rvecs, refined_tvecs, refined_matrices, refined_distortions, refined_points, template_poses, log_scale = unpack_variables(
        x, cameras, point_ids, layout, init_rvecs, init_tvecs, init_matrices, init_distortions, args.intrinsics_mode
    )

    covariance = None
    if compute_covariance:
        try:
            jac = np.asarray(result.jac)
            covariance = np.linalg.pinv(jac.T @ jac)
        except (np.linalg.LinAlgError, ValueError):
            covariance = None

    return Solution(
        rvecs=refined_rvecs,
        tvecs=refined_tvecs,
        matrices=refined_matrices,
        distortions=refined_distortions,
        points=refined_points,
        template_poses=template_poses,
        log_scale=log_scale,
        observations=list(current_observations),
        dropped_observations=dropped_observations,
        scale_to_mm=scale_to_mm,
        scale_source=scale_source,
        init_rvecs=init_rvecs,
        init_tvecs=init_tvecs,
        init_matrices=init_matrices,
        init_distortions=init_distortions,
        init_points=init_points,
        prior=prior,
        layout=layout,
        result=result,
        initial_cost=initial_cost,
        sigma_refit_factors=sigma_refit_factors,
        covariance=covariance,
    )


def point_sigma_from_covariance(covariance: np.ndarray | None, layout: Layout, point_id: str) -> np.ndarray | None:
    if covariance is None:
        return None
    sl = layout.point_slices.get(point_id)
    if sl is None:
        return None
    block = covariance[sl, sl]
    diag = np.clip(np.diag(block), 0.0, None)
    return np.sqrt(diag)


def distance_sigma_from_covariance(
    covariance: np.ndarray | None,
    layout: Layout,
    points: dict[str, np.ndarray],
    point_a: str,
    point_b: str,
) -> float | None:
    if covariance is None:
        return None
    sl_a = layout.point_slices.get(point_a)
    sl_b = layout.point_slices.get(point_b)
    if sl_a is None or sl_b is None:
        return None
    delta = points[point_a] - points[point_b]
    length = float(np.linalg.norm(delta))
    if length <= 1e-12:
        return None
    unit = delta / length
    cov_aa = covariance[sl_a, sl_a]
    cov_bb = covariance[sl_b, sl_b]
    cov_ab = covariance[sl_a, sl_b]
    variance = float(unit @ cov_aa @ unit + unit @ cov_bb @ unit - 2.0 * unit @ cov_ab @ unit)
    return math.sqrt(max(variance, 0.0))


def reprojection_rows(
    cameras: Sequence[Camera],
    observations: Sequence[Observation],
    points: dict[str, np.ndarray],
    rvecs: dict[str, np.ndarray],
    tvecs: dict[str, np.ndarray],
    matrices: dict[str, np.ndarray],
    distortions: dict[str, np.ndarray],
    phase: str,
    outlier_z_threshold: float,
) -> list[dict[str, Any]]:
    cameras_by_name = {cam.name: cam for cam in cameras}
    rows = []
    for obs in observations:
        if obs.point_id not in points:
            continue
        pred = project(
            cameras_by_name[obs.camera_name],
            rvecs[obs.camera_name],
            tvecs[obs.camera_name],
            points[obs.point_id],
            matrix=matrices[obs.camera_name],
            distortions=distortions[obs.camera_name],
        )
        error = float(np.linalg.norm(pred - obs.xy))
        normalized = error / max(obs.sigma_px, 1e-9)
        rows.append(
            {
                "phase": phase,
                "frame": obs.frame,
                "point_id": obs.point_id,
                "camera": obs.camera_name,
                "sigma_px": obs.sigma_px,
                "observed_x": obs.xy[0],
                "observed_y": obs.xy[1],
                "projected_x": pred[0],
                "projected_y": pred[1],
                "error_px": error,
                "normalized_error": normalized,
                "outlier": normalized > outlier_z_threshold,
            }
        )
    return rows


def distance_rows(
    points: dict[str, np.ndarray],
    distances: Sequence[DistanceConstraint],
    phase: str,
    outlier_z_threshold: float,
    covariance: np.ndarray | None = None,
    layout: Layout | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for dist in distances:
        if dist.point_a not in points or dist.point_b not in points:
            continue
        reconstructed = float(np.linalg.norm(points[dist.point_a] - points[dist.point_b]))
        residual = reconstructed - dist.target_mm
        normalized = residual / max(dist.sigma_mm, 1e-9)
        predicted_sigma = (
            distance_sigma_from_covariance(covariance, layout, points, dist.point_a, dist.point_b)
            if covariance is not None and layout is not None
            else None
        )
        rows.append(
            {
                "phase": phase,
                "distance_id": dist.id,
                "point_a": dist.point_a,
                "point_b": dist.point_b,
                "role": "fit" if dist.use_for_refinement else "holdout",
                "used_in_refinement": dist.use_for_refinement,
                "target_mm": dist.target_mm,
                "sigma_mm": dist.sigma_mm,
                "reconstructed_mm": reconstructed,
                "residual_mm": residual,
                "abs_residual_mm": abs(residual),
                "normalized_residual": normalized,
                "predicted_sigma_mm": predicted_sigma,
                "outlier": abs(normalized) > outlier_z_threshold,
            }
        )
    return rows


def finite_summary(values: Sequence[float]) -> dict[str, float | None]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "median": None, "rmse": None, "max": None}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "rmse": float(np.sqrt(np.mean(arr**2))),
        "max": float(np.max(arr)),
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names: list[str] = list(fieldnames or [])
    for row in rows:
        for key in row:
            if key not in names:
                names.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def toml_array(values: Any) -> str:
    return json.dumps(np.asarray(values, dtype=np.float64).tolist(), allow_nan=False)


def intrinsics_rows(
    cameras: Sequence[Camera],
    init_matrices: dict[str, np.ndarray],
    refined_matrices: dict[str, np.ndarray],
    init_distortions: dict[str, np.ndarray],
    refined_distortions: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows = []
    for camera in cameras:
        init_k = init_matrices[camera.name]
        refined_k = refined_matrices[camera.name]
        init_d = init_distortions[camera.name]
        refined_d = refined_distortions[camera.name]
        fx0, fy0, cx0, cy0 = init_k[0, 0], init_k[1, 1], init_k[0, 2], init_k[1, 2]
        fx1, fy1, cx1, cy1 = refined_k[0, 0], refined_k[1, 1], refined_k[0, 2], refined_k[1, 2]
        rows.append(
            {
                "camera": camera.name,
                "initial_fx": fx0,
                "refined_fx": fx1,
                "delta_fx": fx1 - fx0,
                "delta_fx_percent": (fx1 - fx0) / fx0 * 100.0 if abs(fx0) > 1e-12 else None,
                "initial_fy": fy0,
                "refined_fy": fy1,
                "delta_fy": fy1 - fy0,
                "delta_fy_percent": (fy1 - fy0) / fy0 * 100.0 if abs(fy0) > 1e-12 else None,
                "initial_cx": cx0,
                "refined_cx": cx1,
                "delta_cx_px": cx1 - cx0,
                "initial_cy": cy0,
                "refined_cy": cy1,
                "delta_cy_px": cy1 - cy0,
                "initial_distortions": json.dumps(init_d.tolist()),
                "refined_distortions": json.dumps(refined_d.tolist()),
                "distortion_delta_l2": float(np.linalg.norm(refined_d - init_d)),
            }
        )
    return rows


def write_calibration(
    path: Path,
    cameras: Sequence[Camera],
    rvecs: dict[str, np.ndarray],
    tvecs: dict[str, np.ndarray],
    matrices: dict[str, np.ndarray],
    distortions: dict[str, np.ndarray],
) -> None:
    lines = [
        "# Generated by vggt_calibration_pipeline/refine_metric_calibration.py",
        "# Translations are metric millimeters from manual point/distance refinement.",
        "",
    ]
    for camera in sorted(cameras, key=lambda cam: cam.index):
        lines.extend(
            [
                f"[cam_{camera.index}]",
                f"name = {json.dumps(camera.name)}",
                f"size = {json.dumps([int(camera.size[0]), int(camera.size[1])])}",
                f"matrix = {toml_array(matrices[camera.name])}",
                f"distortions = {toml_array(distortions[camera.name])}",
                f"rotation = {toml_array(rvecs[camera.name])}",
                f"translation = {toml_array(tvecs[camera.name])}",
            ]
        )
        if camera.fisheye:
            lines.append("fisheye = true")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_leave_one_out(
    cameras: Sequence[Camera],
    point_ids: Sequence[str],
    observations: Sequence[Observation],
    fit_distances: Sequence[DistanceConstraint],
    templates: Sequence[RigidTemplate],
    points_raw: dict[str, np.ndarray],
    prior_stats: dict[str, Any] | None,
    origin: str,
    args: argparse.Namespace,
    full_points: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for held_out in fit_distances:
        fold_distances = [dist for dist in fit_distances if dist.id != held_out.id]
        try:
            fold = solve_once(
                cameras,
                point_ids,
                observations,
                fold_distances,
                templates,
                points_raw,
                prior_stats,
                origin,
                args,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            rows.append({"distance_id": held_out.id, "error": str(error)})
            continue
        reconstructed = float(np.linalg.norm(fold.points[held_out.point_a] - fold.points[held_out.point_b]))
        fit_reconstructed = float(np.linalg.norm(full_points[held_out.point_a] - full_points[held_out.point_b]))
        rows.append(
            {
                "distance_id": held_out.id,
                "point_a": held_out.point_a,
                "point_b": held_out.point_b,
                "target_mm": held_out.target_mm,
                "loo_reconstructed_mm": reconstructed,
                "loo_error_mm": reconstructed - held_out.target_mm,
                "loo_abs_error_mm": abs(reconstructed - held_out.target_mm),
                "fit_reconstructed_mm": fit_reconstructed,
                "fit_error_mm": fit_reconstructed - held_out.target_mm,
                "optimizer_success": bool(fold.result.success),
            }
        )
    return rows


def frame_number_from_name(name: str) -> int | None:
    match = re.fullmatch(r"frame_(\d+)", name)
    return int(match.group(1)) if match else None


def frame_sort_key(path: Path) -> tuple[int, str]:
    number = frame_number_from_name(path.stem if path.suffix == ".json" else path.name)
    return (number if number is not None else -1, path.name)


def resolve_points_frame_dir(points_path: Path, dataset_dir: Path) -> Path:
    if points_path.name == "selected_points.json":
        if (points_path.parent / "images.npy").is_file():
            return points_path.parent
        try:
            data = json.loads(points_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        frame_dir_value = data.get("frame_dir") if isinstance(data, dict) else None
        if frame_dir_value:
            frame_dir = Path(str(frame_dir_value)).expanduser().resolve()
            if frame_dir.is_dir():
                return frame_dir
        frame_name = data.get("frame_name") if isinstance(data, dict) else None
        if frame_name:
            frame_dir = dataset_dir / str(frame_name)
            if frame_dir.is_dir():
                return frame_dir
        raise ValueError(
            f"{points_path} is a dataset-level annotation but does not record a valid frame_dir/frame_name. "
            "Open it with point_selector.py and save again."
        )
    frame_number = frame_number_from_name(points_path.stem)
    if frame_number is not None:
        frame_dir = dataset_dir / points_path.stem
        if frame_dir.is_dir():
            return frame_dir
    try:
        data = json.loads(points_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = {}
    frame_dir_value = data.get("frame_dir") if isinstance(data, dict) else None
    if frame_dir_value:
        frame_dir = Path(str(frame_dir_value)).expanduser().resolve()
        if frame_dir.is_dir():
            return frame_dir
    raise ValueError(f"Could not determine frame directory for annotation file: {points_path}")


def default_points_path_for_frame(frame_dir: Path) -> Path:
    root_points = frame_dir.parent / "selected_points.json"
    if root_points.is_file():
        return root_points
    root_points = frame_dir.parent / f"{frame_dir.name}.json"
    if root_points.is_file():
        return root_points
    return frame_dir / "selected_points.json"


def resolve_refinement_inputs(args: argparse.Namespace) -> tuple[list[Path], list[Path], Path, Path]:
    input_paths = [frame_dir.expanduser().resolve() for frame_dir in args.frame_dirs]
    explicit_points = args.points.expanduser().resolve() if args.points is not None else None

    if len(input_paths) == 1 and input_paths[0].is_dir() and not (input_paths[0] / "images.npy").is_file():
        dataset_dir = input_paths[0]
        if explicit_points is not None:
            points_paths = [explicit_points]
        else:
            root_points = dataset_dir / "selected_points.json"
            points_paths = [root_points] if root_points.is_file() else []
            if not points_paths:
                points_paths = sorted(
                    [
                        path
                        for path in dataset_dir.glob("frame_*.json")
                        if path.is_file() and frame_number_from_name(path.stem) is not None
                    ],
                    key=frame_sort_key,
                )
            if not points_paths:
                points_paths = sorted(
                    [path for path in dataset_dir.glob("frame_*/selected_points.json") if path.is_file()],
                    key=lambda path: frame_sort_key(path.parent),
                )
        if not points_paths:
            raise FileNotFoundError(
                f"No annotation JSON files found in {dataset_dir}. "
                "Expected DATASET_DIR/selected_points.json from point_selector.py "
                "or fallback DATASET_DIR/frame_XXXXXX.json / old FRAME_DIR/selected_points.json files. "
                f"Searched: {dataset_dir / 'selected_points.json'}, {dataset_dir / 'frame_*.json'}, "
                f"and {dataset_dir / 'frame_*' / 'selected_points.json'}. "
                f"Create one with: python vggt_calibration_pipeline/point_selector.py {dataset_dir}"
            )
        frame_dirs = [resolve_points_frame_dir(points_path, dataset_dir) for points_path in points_paths]
        calibration_path = args.calibration or frame_dirs[0] / "calibration.toml"
        output_dir = args.output_dir or dataset_dir / "metric_refinement"
        return frame_dirs, points_paths, calibration_path, output_dir

    if explicit_points is not None and len(input_paths) > 1:
        raise ValueError("--points can only be used with a single FRAME_DIR or DATASET_DIR")
    frame_dirs = input_paths
    points_paths = [explicit_points if explicit_points is not None else default_points_path_for_frame(frame_dir) for frame_dir in frame_dirs]
    calibration_path = args.calibration or frame_dirs[0] / "calibration.toml"
    output_dir = args.output_dir or frame_dirs[0] / "metric_refinement"
    return frame_dirs, points_paths, calibration_path, output_dir


def optimize_refinement(args: argparse.Namespace) -> dict[str, Any]:
    frame_dirs, point_path_values, calibration_path, output_dir = resolve_refinement_inputs(args)

    cameras = load_calibration(calibration_path)
    cameras_by_name = {cam.name: cam for cam in cameras}
    origin = args.origin or cameras[0].name
    if origin not in cameras_by_name:
        raise ValueError(f"Origin camera {origin!r} not found. Available: {sorted(cameras_by_name)}")

    prior_stats = None
    if args.prior_stats is not None:
        prior_stats = json.loads(args.prior_stats.expanduser().resolve().read_text(encoding="utf-8"))

    annotation_sets = []
    points_paths = []
    for frame_dir, points_path in zip(frame_dirs, point_path_values, strict=True):
        points_paths.append(str(points_path))
        annotation_sets.append(
            parse_annotation_file(
                points_path,
                cameras,
                default_reprojection_sigma_px=args.reprojection_sigma_px,
                default_distance_sigma_mm=args.distance_sigma_mm,
                default_template_sigma_mm=args.template_sigma_mm,
                frame_tag=frame_dir.name,
            )
        )
    point_ids, observations, distances, templates = merge_annotations(annotation_sets)

    fit_distances = [dist for dist in distances if dist.use_for_refinement]
    holdout_distances = [dist for dist in distances if not dist.use_for_refinement]
    obs_by_point: dict[str, list[Observation]] = {point_id: [] for point_id in point_ids}
    for obs in observations:
        obs_by_point.setdefault(obs.point_id, []).append(obs)

    points_raw = {}
    for point_id, point_observations in obs_by_point.items():
        if len({obs.camera_name for obs in point_observations}) < 2:
            continue
        points_raw[point_id] = triangulate_point(point_observations, cameras_by_name, method=args.triangulation)
    point_ids = [point_id for point_id in point_ids if point_id in points_raw]
    observations = [obs for obs in observations if obs.point_id in points_raw]
    if len(point_ids) < 2:
        raise ValueError("Need at least two reconstructable points")
    templates = [
        template
        for template in templates
        if len([pid for pid in template.points if pid in points_raw]) >= 3
    ]

    solution = solve_once(
        cameras,
        point_ids,
        observations,
        fit_distances,
        templates,
        points_raw,
        prior_stats,
        origin,
        args,
        compute_covariance=True,
    )
    result = solution.result
    layout = solution.layout

    output_dir.mkdir(parents=True, exist_ok=True)
    write_calibration(
        output_dir / "calibration_metric.toml",
        cameras,
        solution.rvecs,
        solution.tvecs,
        solution.matrices,
        solution.distortions,
    )

    initial_reproj = reprojection_rows(
        cameras,
        observations,
        solution.init_points,
        solution.init_rvecs,
        solution.init_tvecs,
        solution.init_matrices,
        solution.init_distortions,
        "initial",
        args.outlier_z_threshold,
    )
    refined_reproj = reprojection_rows(
        cameras,
        solution.observations,
        solution.points,
        solution.rvecs,
        solution.tvecs,
        solution.matrices,
        solution.distortions,
        "refined",
        args.outlier_z_threshold,
    )
    write_csv(output_dir / "reprojection_report.csv", initial_reproj + refined_reproj)

    intrinsics_report = intrinsics_rows(
        cameras, solution.init_matrices, solution.matrices, solution.init_distortions, solution.distortions
    )
    write_csv(output_dir / "intrinsics_report.csv", intrinsics_report)

    initial_distances = distance_rows(solution.init_points, distances, "initial", args.outlier_z_threshold)
    refined_distances = distance_rows(
        solution.points,
        distances,
        "refined",
        args.outlier_z_threshold,
        covariance=solution.covariance,
        layout=layout,
    )
    write_csv(output_dir / "distance_report.csv", initial_distances + refined_distances)
    write_csv(
        output_dir / "validation_report.csv",
        [row for row in refined_distances if not row["used_in_refinement"]],
        fieldnames=[
            "phase",
            "distance_id",
            "point_a",
            "point_b",
            "role",
            "used_in_refinement",
            "target_mm",
            "sigma_mm",
            "reconstructed_mm",
            "residual_mm",
            "abs_residual_mm",
            "normalized_residual",
            "predicted_sigma_mm",
            "outlier",
        ],
    )

    outlier_rows: list[dict[str, Any]] = []
    for row in refined_distances:
        if row.get("outlier"):
            outlier_rows.append(
                {
                    "type": "distance",
                    "id": row["distance_id"],
                    "role": row["role"],
                    "camera": "",
                    "point_id": f"{row['point_a']}-{row['point_b']}",
                    "residual": row["residual_mm"],
                    "sigma": row["sigma_mm"],
                    "normalized_residual": row["normalized_residual"],
                    "threshold": args.outlier_z_threshold,
                }
            )
    for row in refined_reproj:
        if row.get("outlier"):
            outlier_rows.append(
                {
                    "type": "reprojection",
                    "id": row["point_id"],
                    "role": "fit",
                    "camera": row["camera"],
                    "point_id": row["point_id"],
                    "residual": row["error_px"],
                    "sigma": row["sigma_px"],
                    "normalized_residual": row["normalized_error"],
                    "threshold": args.outlier_z_threshold,
                }
            )
    for obs in solution.dropped_observations:
        outlier_rows.append(
            {
                "type": "dropped_observation",
                "id": obs.point_id,
                "role": "fit",
                "camera": obs.camera_name,
                "point_id": obs.point_id,
                "residual": None,
                "sigma": obs.sigma_px,
                "normalized_residual": None,
                "threshold": args.drop_outlier_obs_z,
            }
        )
    write_csv(
        output_dir / "annotation_outliers.csv",
        outlier_rows,
        fieldnames=[
            "type",
            "id",
            "role",
            "camera",
            "point_id",
            "residual",
            "sigma",
            "normalized_residual",
            "threshold",
        ],
    )

    point_rows = []
    for point_id in point_ids:
        sigma = point_sigma_from_covariance(solution.covariance, layout, point_id)
        point_rows.append(
            {
                "point_id": point_id,
                "x_mm": solution.points[point_id][0],
                "y_mm": solution.points[point_id][1],
                "z_mm": solution.points[point_id][2],
                "initial_x_mm": solution.init_points[point_id][0],
                "initial_y_mm": solution.init_points[point_id][1],
                "initial_z_mm": solution.init_points[point_id][2],
                "shift_mm": float(np.linalg.norm(solution.points[point_id] - solution.init_points[point_id])),
                "num_observations": len(obs_by_point.get(point_id, [])),
                "sigma_x_mm": sigma[0] if sigma is not None else None,
                "sigma_y_mm": sigma[1] if sigma is not None else None,
                "sigma_z_mm": sigma[2] if sigma is not None else None,
                "sigma_norm_mm": float(np.linalg.norm(sigma)) if sigma is not None else None,
            }
        )
    write_csv(output_dir / "points_3d.csv", point_rows)

    initial_cameras = {
        cam.name: camera_from_params(
            cam, solution.init_rvecs[cam.name], solution.init_tvecs[cam.name], solution.init_matrices[cam.name], solution.init_distortions[cam.name]
        )
        for cam in cameras
    }
    refined_cameras = {
        cam.name: camera_from_params(
            cam,
            solution.rvecs[cam.name],
            solution.tvecs[cam.name],
            solution.matrices[cam.name],
            solution.distortions[cam.name],
        )
        for cam in cameras
    }
    camera_rows = []
    for cam in cameras:
        init_center = initial_cameras[cam.name].center
        refined_center = refined_cameras[cam.name].center
        camera_rows.append(
            {
                "camera": cam.name,
                "fixed_origin": cam.name == origin,
                "initial_center_x_mm": init_center[0],
                "initial_center_y_mm": init_center[1],
                "initial_center_z_mm": init_center[2],
                "refined_center_x_mm": refined_center[0],
                "refined_center_y_mm": refined_center[1],
                "refined_center_z_mm": refined_center[2],
                "center_shift_mm": float(np.linalg.norm(refined_center - init_center)),
                "rotation_delta_deg": rotation_delta_deg(solution.init_rvecs[cam.name], solution.rvecs[cam.name]),
                "initial_tvec_x_mm": solution.init_tvecs[cam.name][0],
                "initial_tvec_y_mm": solution.init_tvecs[cam.name][1],
                "initial_tvec_z_mm": solution.init_tvecs[cam.name][2],
                "refined_tvec_x_mm": solution.tvecs[cam.name][0],
                "refined_tvec_y_mm": solution.tvecs[cam.name][1],
                "refined_tvec_z_mm": solution.tvecs[cam.name][2],
            }
        )
    write_csv(output_dir / "camera_positions.csv", camera_rows)

    pair_rows = []
    for i, cam_a in enumerate(cameras):
        for cam_b in cameras[i + 1 :]:
            initial_distance = float(np.linalg.norm(initial_cameras[cam_a.name].center - initial_cameras[cam_b.name].center))
            refined_distance = float(np.linalg.norm(refined_cameras[cam_a.name].center - refined_cameras[cam_b.name].center))
            pair_rows.append(
                {
                    "camera_a": cam_a.name,
                    "camera_b": cam_b.name,
                    "initial_baseline_mm": initial_distance,
                    "refined_baseline_mm": refined_distance,
                    "delta_mm": refined_distance - initial_distance,
                    "delta_percent": (refined_distance - initial_distance) / initial_distance * 100.0 if initial_distance else None,
                }
            )
    write_csv(output_dir / "camera_pairwise_distances.csv", pair_rows)

    loo_rows: list[dict[str, Any]] = []
    loo_summary: dict[str, Any] | None = None
    if args.loo and len(fit_distances) >= 2:
        loo_rows = run_leave_one_out(
            cameras,
            point_ids,
            observations,
            fit_distances,
            templates,
            points_raw,
            prior_stats,
            origin,
            args,
            solution.points,
        )
        write_csv(output_dir / "loo_report.csv", loo_rows)
        loo_errors = [row["loo_error_mm"] for row in loo_rows if "loo_error_mm" in row]
        loo_summary = finite_summary(loo_errors)
        loo_summary["mean_abs"] = float(np.mean(np.abs(loo_errors))) if loo_errors else None
        loo_summary["median_abs"] = float(np.median(np.abs(loo_errors))) if loo_errors else None
        loo_summary["count"] = len(loo_errors)

    initial_reproj_summary = finite_summary([row["error_px"] for row in initial_reproj])
    refined_reproj_summary = finite_summary([row["error_px"] for row in refined_reproj])
    initial_distance_summary = finite_summary([row["residual_mm"] for row in initial_distances])
    refined_distance_summary = finite_summary([row["residual_mm"] for row in refined_distances])
    refined_fit_distance_summary = finite_summary(
        [row["residual_mm"] for row in refined_distances if row["used_in_refinement"]]
    )
    refined_holdout_distance_summary = finite_summary(
        [row["residual_mm"] for row in refined_distances if not row["used_in_refinement"]]
    )
    refined_reprojection_outliers = sum(1 for row in refined_reproj if row.get("outlier"))
    refined_distance_outliers = sum(1 for row in refined_distances if row.get("outlier"))
    fx_pct_summary = finite_summary([row["delta_fx_percent"] for row in intrinsics_report if row["delta_fx_percent"] is not None])
    fy_pct_summary = finite_summary([row["delta_fy_percent"] for row in intrinsics_report if row["delta_fy_percent"] is not None])
    principal_shift_summary = finite_summary(
        [
            float(math.hypot(row["delta_cx_px"], row["delta_cy_px"]))
            for row in intrinsics_report
        ]
    )
    distortion_delta_summary = finite_summary([row["distortion_delta_l2"] for row in intrinsics_report])

    summary = {
        "frame_dirs": [str(frame_dir) for frame_dir in frame_dirs],
        "calibration": str(calibration_path),
        "points_json": points_paths,
        "prior_stats": str(args.prior_stats) if args.prior_stats else None,
        "output_dir": str(output_dir),
        "origin_camera": origin,
        "prior_mode": args.prior_mode,
        "free_scale": bool(args.free_scale),
        "prior_scale": float(math.exp(solution.log_scale)) if args.free_scale else 1.0,
        "prior_sigma_scale": args.prior_sigma_scale,
        "triangulation": args.triangulation,
        "intrinsics_mode": args.intrinsics_mode if args.refine_intrinsics else None,
        "distortion_mode": args.distortion_mode if args.refine_distortion else None,
        "sigma_refit_factors": solution.sigma_refit_factors,
        "num_dropped_observations": len(solution.dropped_observations),
        "num_cameras": len(cameras),
        "num_points": len(point_ids),
        "num_observations": len(observations),
        "num_distances": len(distances),
        "num_fit_distances": len(fit_distances),
        "num_holdout_distances": len(holdout_distances),
        "num_templates": len(templates),
        "refine_intrinsics": bool(args.refine_intrinsics),
        "refine_distortion": bool(args.refine_distortion),
        "scale_to_mm": solution.scale_to_mm,
        "scale_source": solution.scale_source,
        "prior_sigmas": {
            "rotation_deg": solution.prior.rotation_sigma_deg,
            "translation_mm": solution.prior.translation_sigma_mm,
            "pair_rotation_deg": solution.prior.pair_rotation_sigma_deg,
            "pair_baseline_mm": solution.prior.pair_baseline_sigma_mm,
            "focal_percent": solution.prior.focal_sigma_percent,
            "principal_px": solution.prior.principal_sigma_px,
        },
        "initial_cost": solution.initial_cost,
        "final_cost": float(result.cost),
        "optimizer_success": bool(result.success),
        "optimizer_message": result.message,
        "optimizer_nfev": int(result.nfev),
        "reprojection_px": {
            "initial": initial_reproj_summary,
            "refined": refined_reproj_summary,
        },
        "distance_residual_mm": {
            "initial": initial_distance_summary,
            "refined": refined_distance_summary,
            "refined_fit": refined_fit_distance_summary,
            "refined_holdout": refined_holdout_distance_summary,
        },
        "loo_mm": loo_summary,
        "outliers": {
            "threshold_z": args.outlier_z_threshold,
            "refined_reprojection_count": refined_reprojection_outliers,
            "refined_distance_count": refined_distance_outliers,
        },
        "intrinsics_delta": {
            "fx_percent": fx_pct_summary,
            "fy_percent": fy_pct_summary,
            "principal_shift_px": principal_shift_summary,
            "distortion_l2": distortion_delta_summary,
        },
        "outputs": {
            "calibration_metric": str(output_dir / "calibration_metric.toml"),
            "points_3d": str(output_dir / "points_3d.csv"),
            "distance_report": str(output_dir / "distance_report.csv"),
            "validation_report": str(output_dir / "validation_report.csv"),
            "reprojection_report": str(output_dir / "reprojection_report.csv"),
            "intrinsics_report": str(output_dir / "intrinsics_report.csv"),
            "annotation_outliers": str(output_dir / "annotation_outliers.csv"),
            "camera_positions": str(output_dir / "camera_positions.csv"),
            "camera_pairwise_distances": str(output_dir / "camera_pairwise_distances.csv"),
            "loo_report": str(output_dir / "loo_report.csv") if loo_rows else None,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("frame_dirs", type=Path, nargs="+", help="One DATASET_DIR with selected_points.json, or one or more annotated frame directories")
    parser.add_argument("--calibration", type=Path, help="Prior/init calibration. Default: first FRAME_DIR/calibration.toml. Use vggt_prior/calibration_prior.toml from aggregate_vggt_priors.py for the aggregated prior.")
    parser.add_argument("--prior-stats", type=Path, help="prior_stats.json from aggregate_vggt_priors.py; supplies empirical per-camera/per-pair prior sigmas")
    parser.add_argument("--points", type=Path, help="Explicit annotation JSON. Default: DATASET_DIR/selected_points.json or FRAME_DIR/selected_points.json")
    parser.add_argument("--output-dir", type=Path, help="Default: DATASET_DIR/metric_refinement for dataset input, else first FRAME_DIR/metric_refinement")
    parser.add_argument("--origin", help="Fixed origin camera name. Default: first camera in TOML.")
    parser.add_argument("--prior-mode", default="legacy", choices=["legacy", "absolute", "relative"], help="Camera pose prior parametrization (see module docstring)")
    parser.add_argument("--free-scale", action="store_true", help="Let the prior rig scale float instead of freezing the initial metric scale into the prior")
    parser.add_argument("--prior-sigma-scale", type=float, default=1.0, help="Multiply all pose prior sigmas (>1 weakens the prior)")
    parser.add_argument("--triangulation", default="pixel", choices=["pixel", "normalized"], help="DLT initialization: raw pixel coords (legacy) or undistorted normalized coords")
    parser.add_argument("--reprojection-sigma-px", type=float, default=2.0)
    parser.add_argument("--distance-sigma-mm", type=float, default=1.0)
    parser.add_argument("--template-sigma-mm", type=float, default=0.5)
    parser.add_argument("--rotation-prior-deg", type=float, default=None, help="Default: prior-stats value if given, else 3.0")
    parser.add_argument("--translation-prior-mm", type=float, default=None, help="Default: prior-stats value if given, else 25.0")
    parser.add_argument("--refine-intrinsics", action="store_true", help="Optimize intrinsics with priors (see --intrinsics-mode).")
    parser.add_argument("--intrinsics-mode", default="fxfy_pp", choices=["fxfy_pp", "fxfy", "f", "f_shared"], help="fxfy_pp: fx,fy,cx,cy per camera; fxfy: fx,fy per camera; f: one focal gain per camera; f_shared: one global focal gain")
    parser.add_argument("--refine-distortion", action="store_true", help="Optimize distortion coefficients with priors (see --distortion-mode).")
    parser.add_argument("--distortion-mode", default="full", choices=["k1", "k1k2", "full"], help="How many distortion coefficients to free")
    parser.add_argument("--intrinsics-prior-percent", type=float, default=None, help="1-sigma focal prior as percent of initial fx/fy. Default: prior-stats value if given, else 10.0")
    parser.add_argument("--principal-point-prior-px", type=float, default=None, help="1-sigma prior for cx/cy movement. Default: prior-stats value if given, else 50.0")
    parser.add_argument("--distortion-prior", type=float, default=0.05, help="1-sigma prior for each free distortion coefficient.")
    parser.add_argument("--sigma-refit-iters", type=int, default=0, help="IRLS iterations: rescale click sigmas from robust residual statistics and refit")
    parser.add_argument("--drop-outlier-obs-z", type=float, default=0.0, help="After fitting, drop observations with normalized residual above this and refit (0 = off)")
    parser.add_argument("--loo", action="store_true", help="Run leave-one-distance-out cross-validation over fit distances")
    parser.add_argument("--loss", default="soft_l1", choices=["linear", "soft_l1", "huber", "cauchy", "arctan"])
    parser.add_argument("--loss-f-scale", type=float, default=1.0)
    parser.add_argument("--outlier-z-threshold", type=float, default=3.0)
    parser.add_argument("--max-nfev", type=int, default=5000)
    parser.add_argument("--verbose", action="store_true")
    return parser


def parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def main() -> int:
    args = parse_args()
    summary = optimize_refinement(args)
    printed = {
        "output_dir": summary["output_dir"],
        "prior_mode": summary["prior_mode"],
        "free_scale": summary["free_scale"],
        "prior_scale": summary["prior_scale"],
        "scale_to_mm": summary["scale_to_mm"],
        "scale_source": summary["scale_source"],
        "num_points": summary["num_points"],
        "num_observations": summary["num_observations"],
        "num_distances": summary["num_distances"],
        "num_fit_distances": summary["num_fit_distances"],
        "num_holdout_distances": summary["num_holdout_distances"],
        "refine_intrinsics": summary["refine_intrinsics"],
        "intrinsics_mode": summary["intrinsics_mode"],
        "refine_distortion": summary["refine_distortion"],
        "distortion_mode": summary["distortion_mode"],
        "reprojection_rmse_initial_px": summary["reprojection_px"]["initial"]["rmse"],
        "reprojection_rmse_refined_px": summary["reprojection_px"]["refined"]["rmse"],
        "distance_rmse_initial_mm": summary["distance_residual_mm"]["initial"]["rmse"],
        "distance_rmse_refined_mm": summary["distance_residual_mm"]["refined"]["rmse"],
        "fit_distance_rmse_refined_mm": summary["distance_residual_mm"]["refined_fit"]["rmse"],
        "holdout_distance_rmse_refined_mm": summary["distance_residual_mm"]["refined_holdout"]["rmse"],
        "refined_distance_outliers": summary["outliers"]["refined_distance_count"],
        "refined_reprojection_outliers": summary["outliers"]["refined_reprojection_count"],
        "intrinsics_delta": summary["intrinsics_delta"],
    }
    if summary.get("loo_mm"):
        printed["loo_rmse_mm"] = summary["loo_mm"]["rmse"]
        printed["loo_mean_abs_mm"] = summary["loo_mm"]["mean_abs"]
        printed["loo_max_abs_mm"] = summary["loo_mm"]["max"]
    print(json.dumps(printed, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
