#!/usr/bin/env python3
"""Browser GUI for selecting shared calibration points in a VGGT video dataset.

The VGGT pipeline outputs usually keep the displayed frame set in
``images.npy`` rather than source PNG files. This tool serves those arrays to a
local browser, records point observations for any subset of cameras, and saves
the video-level annotations as ``DATASET_DIR/selected_points.json`` by default.

Example:
    python vggt_calibration_pipeline/point_selector.py \
        /media/tony/backup_inte/vggt_paper/data_set/videos_sz3y
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
import time
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import cv2
from PIL import Image

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback.
    tomllib = None


DEFAULT_DATASET_DIR = Path("/media/tony/backup_inte/vggt_paper/data_set/videos_sz3y")
DEFAULT_OUTPUT_NAME = "selected_points.json"
UNIT_TO_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0}
DEFAULT_REPROJECTION_SIGMA_PX = 2.0
DEFAULT_DISTANCE_SIGMA_MM = 1.0


@dataclass(frozen=True)
class CameraView:
    index: int
    name: str
    width: int
    height: int
    source_width: int
    source_height: int


@dataclass(frozen=True)
class CameraCalibration:
    name: str
    size: tuple[int, int]
    matrix: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray

    @property
    def projection(self) -> np.ndarray:
        return self.matrix @ np.hstack([self.rotation, self.translation.reshape(3, 1)])


class PointSelectorState:
    def __init__(self, dataset_dir: Path, frame_dir: Path, output_path: Path | None):
        self.dataset_dir = dataset_dir
        self.frame_dir = frame_dir
        self.output_path = output_path
        self.frame_dirs = self._discover_frame_dirs(dataset_dir, frame_dir)
        self.current_frame_option_index = self._current_frame_option_index(frame_dir)
        self.images = self._load_images(frame_dir / "images.npy")
        self.cameras = self._camera_views()
        self.frame_images_cache: dict[int, np.ndarray] = {self.current_frame_option_index: self.images}
        self.image_png_cache: dict[tuple[int, int], bytes] = {}

    @staticmethod
    def _frame_number(path: Path) -> int | None:
        match = re.fullmatch(r"frame_(\d+)", path.name)
        return int(match.group(1)) if match else None

    @classmethod
    def _discover_frame_dirs(cls, dataset_dir: Path, frame_dir: Path) -> list[Path]:
        siblings = [
            path
            for path in dataset_dir.glob("frame_*")
            if path.is_dir() and (path / "images.npy").is_file() and cls._frame_number(path) is not None
        ]
        if not siblings:
            return [frame_dir]
        siblings = sorted(siblings, key=lambda path: (cls._frame_number(path) or -1, path.name))
        if frame_dir not in siblings:
            siblings.append(frame_dir)
            siblings = sorted(siblings, key=lambda path: (cls._frame_number(path) or -1, path.name))
        return siblings

    def _current_frame_option_index(self, frame_dir: Path) -> int:
        try:
            return self.frame_dirs.index(frame_dir)
        except ValueError:
            return 0

    def frame_dir_for_option(self, frame_option_index: int) -> Path:
        if frame_option_index < 0 or frame_option_index >= len(self.frame_dirs):
            raise IndexError(frame_option_index)
        return self.frame_dirs[frame_option_index]

    def output_path_for_frame_option(self, frame_option_index: int) -> Path:
        if self.output_path is not None:
            return self.output_path
        return self.dataset_dir / DEFAULT_OUTPUT_NAME

    def frame_option_index_from_payload(self, payload: dict[str, Any]) -> int:
        try:
            frame_option_index = int(payload.get("display_frame_index", self.current_frame_option_index))
        except (TypeError, ValueError):
            frame_option_index = self.current_frame_option_index
        if frame_option_index < 0 or frame_option_index >= len(self.frame_dirs):
            raise ValueError(f"Invalid display_frame_index {frame_option_index}")
        return frame_option_index

    @staticmethod
    def _load_images(path: Path) -> np.ndarray:
        if not path.is_file():
            raise FileNotFoundError(f"Missing images.npy: {path}")
        images = np.load(path)
        if images.ndim == 5 and images.shape[0] == 1:
            images = images[0]
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected images.npy shape [F, 3, H, W], got {images.shape}")
        return images.astype(np.float32, copy=False)

    def _camera_views(self) -> list[CameraView]:
        num_cameras = int(self.images.shape[0])
        height = int(self.images.shape[2])
        width = int(self.images.shape[3])
        calibration = _read_calibration_minimal(self.frame_dir / "calibration.toml")

        views: list[CameraView] = []
        for idx in range(num_cameras):
            fallback_name = chr(ord("A") + idx)
            cam = calibration.get(idx, {})
            name = str(cam.get("name") or fallback_name)
            size = cam.get("size")
            if (
                isinstance(size, (list, tuple))
                and len(size) >= 2
                and int(size[0]) > 0
                and int(size[1]) > 0
            ):
                source_width, source_height = int(size[0]), int(size[1])
            else:
                source_width, source_height = width, height
            views.append(
                CameraView(
                    index=idx,
                    name=name,
                    width=width,
                    height=height,
                    source_width=source_width,
                    source_height=source_height,
                )
            )
        return views

    def _images_for_frame_option(self, frame_option_index: int) -> np.ndarray:
        if frame_option_index < 0 or frame_option_index >= len(self.frame_dirs):
            raise IndexError(frame_option_index)
        if frame_option_index not in self.frame_images_cache:
            images = self._load_images(self.frame_dirs[frame_option_index] / "images.npy")
            if images.shape != self.images.shape:
                raise ValueError(
                    f"Frame {self.frame_dirs[frame_option_index].name} has images.npy shape {images.shape}; "
                    f"expected {self.images.shape}"
                )
            self.frame_images_cache[frame_option_index] = images
        return self.frame_images_cache[frame_option_index]

    def image_png(self, index: int, frame_option_index: int | None = None) -> bytes:
        frame_option_index = self.current_frame_option_index if frame_option_index is None else frame_option_index
        if index < 0 or index >= len(self.cameras):
            raise IndexError(index)
        cache_key = (frame_option_index, index)
        if cache_key not in self.image_png_cache:
            images = self._images_for_frame_option(frame_option_index)
            image = np.transpose(images[index], (1, 2, 0))
            image_u8 = (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
            buffer = BytesIO()
            Image.fromarray(image_u8, mode="RGB").save(buffer, format="PNG")
            self.image_png_cache[cache_key] = buffer.getvalue()
        return self.image_png_cache[cache_key]

    def metadata(self) -> dict[str, Any]:
        output_path = self.output_path_for_frame_option(self.current_frame_option_index)
        points = {}
        if output_path.is_file():
            try:
                points = json.loads(output_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                points = {"load_error": f"Could not parse {output_path}"}

        return {
            "dataset_dir": str(self.dataset_dir),
            "frame_dir": str(self.frame_dir),
            "output_path": str(output_path),
            "schema_version": 3,
            "annotation_scope": "video",
            "coordinate_origin": "top_left",
            "image_space": "model",
            "display_frame_index": self.current_frame_option_index,
            "display_frames": [
                {
                    "index": idx,
                    "name": path.name,
                    "frame_number": self._frame_number(path),
                    "frame_dir": str(path),
                    "output_path": str(self.output_path_for_frame_option(idx)),
                    "has_existing_points": self.output_path_for_frame_option(idx).is_file(),
                }
                for idx, path in enumerate(self.frame_dirs)
            ],
            "cameras": [
                {
                    "index": cam.index,
                    "name": cam.name,
                    "width": cam.width,
                    "height": cam.height,
                    "source_width": cam.source_width,
                    "source_height": cam.source_height,
                    "image_url": f"/api/image/{self.current_frame_option_index}/{cam.index}.png",
                }
                for cam in self.cameras
            ],
            "existing_points": points,
        }

    def save_points(self, payload: dict[str, Any]) -> dict[str, Any]:
        frame_option_index = self.frame_option_index_from_payload(payload)
        frame_dir = self.frame_dir_for_option(frame_option_index)
        output_path = self.output_path_for_frame_option(frame_option_index)
        points = payload.get("points")
        if not isinstance(points, list):
            raise ValueError("Payload must contain a points list")

        camera_by_name = {cam.name: cam for cam in self.cameras}
        normalized_points = []
        warnings: list[str] = []

        for point in points:
            if not isinstance(point, dict):
                continue
            point_id = str(point.get("id") or "").strip()
            if not point_id:
                continue
            label = str(point.get("label") or point_id).strip()
            notes = str(point.get("notes") or "")
            try:
                sigma_px = float(point.get("sigma_px", DEFAULT_REPROJECTION_SIGMA_PX))
            except (TypeError, ValueError):
                warnings.append(f"{point_id} has an invalid click uncertainty; using {DEFAULT_REPROJECTION_SIGMA_PX:g} px")
                sigma_px = DEFAULT_REPROJECTION_SIGMA_PX
            if sigma_px <= 0:
                warnings.append(f"{point_id} click uncertainty must be positive; using {DEFAULT_REPROJECTION_SIGMA_PX:g} px")
                sigma_px = DEFAULT_REPROJECTION_SIGMA_PX
            observations_in = point.get("observations") or {}
            if not isinstance(observations_in, dict):
                observations_in = {}

            observations: dict[str, dict[str, float]] = {}
            for cam_name, obs in observations_in.items():
                if cam_name not in camera_by_name or not isinstance(obs, dict):
                    continue
                cam = camera_by_name[cam_name]
                try:
                    x = float(obs["x"])
                    y = float(obs["y"])
                except (KeyError, TypeError, ValueError):
                    continue
                x = max(0.0, min(float(cam.width - 1), x))
                y = max(0.0, min(float(cam.height - 1), y))
                source_x = x * cam.source_width / cam.width
                source_y = y * cam.source_height / cam.height
                observations[cam_name] = {
                    "x": x,
                    "y": y,
                    "source_x": source_x,
                    "source_y": source_y,
                }

            if len(observations) < 2:
                warnings.append(f"{point_id} has fewer than 2 camera observations")

            normalized_points.append(
                {
                    "id": point_id,
                    "label": label,
                    "notes": notes,
                    "sigma_px": sigma_px,
                    "observations": observations,
                }
            )

        point_ids = {point["id"] for point in normalized_points}
        normalized_distances = []
        distances = payload.get("distances") or []
        if not isinstance(distances, list):
            raise ValueError("Payload distances must be a list")
        for idx, distance in enumerate(distances, start=1):
            if not isinstance(distance, dict):
                continue
            distance_id = str(distance.get("id") or f"D{idx:03d}").strip()
            point_a = str(distance.get("point_a") or "").strip()
            point_b = str(distance.get("point_b") or "").strip()
            unit = str(distance.get("unit") or "mm").strip().lower()
            sigma_unit = str(distance.get("sigma_unit") or unit).strip().lower()
            notes = str(distance.get("notes") or "")
            try:
                value = float(distance.get("value"))
            except (TypeError, ValueError):
                warnings.append(f"{distance_id} has an invalid distance value")
                continue
            try:
                if "sigma_mm" in distance:
                    sigma_mm = float(distance["sigma_mm"])
                    sigma_value = sigma_mm / UNIT_TO_MM.get(sigma_unit, UNIT_TO_MM.get(unit, 1.0))
                else:
                    sigma_value = float(distance.get("sigma_value", DEFAULT_DISTANCE_SIGMA_MM / UNIT_TO_MM.get(sigma_unit, 1.0)))
                    sigma_mm = sigma_value * UNIT_TO_MM.get(sigma_unit, UNIT_TO_MM.get(unit, 1.0))
            except (TypeError, ValueError):
                warnings.append(f"{distance_id} has an invalid distance uncertainty; using {DEFAULT_DISTANCE_SIGMA_MM:g} mm")
                sigma_unit = "mm"
                sigma_value = DEFAULT_DISTANCE_SIGMA_MM
                sigma_mm = DEFAULT_DISTANCE_SIGMA_MM
            if not distance_id:
                continue
            if point_a not in point_ids or point_b not in point_ids:
                warnings.append(f"{distance_id} references a missing point")
                continue
            if point_a == point_b:
                warnings.append(f"{distance_id} uses the same point twice")
                continue
            if unit not in UNIT_TO_MM:
                warnings.append(f"{distance_id} has unsupported unit {unit!r}")
                continue
            if sigma_unit not in UNIT_TO_MM:
                warnings.append(f"{distance_id} has unsupported uncertainty unit {sigma_unit!r}")
                continue
            if value <= 0:
                warnings.append(f"{distance_id} must be greater than zero")
                continue
            if sigma_mm <= 0:
                warnings.append(f"{distance_id} uncertainty must be greater than zero; using {DEFAULT_DISTANCE_SIGMA_MM:g} mm")
                sigma_unit = "mm"
                sigma_value = DEFAULT_DISTANCE_SIGMA_MM
                sigma_mm = DEFAULT_DISTANCE_SIGMA_MM
            use_for_refinement = bool(distance.get("use_for_refinement", not bool(distance.get("holdout", False))))
            normalized_distances.append(
                {
                    "id": distance_id,
                    "point_a": point_a,
                    "point_b": point_b,
                    "value": value,
                    "unit": unit,
                    "distance_mm": value * UNIT_TO_MM[unit],
                    "sigma_value": sigma_value,
                    "sigma_unit": sigma_unit,
                    "sigma_mm": sigma_mm,
                    "use_for_refinement": use_for_refinement,
                    "holdout": not use_for_refinement,
                    "notes": notes,
                }
            )

        document = {
            "schema_version": 3,
            "annotation_scope": "video",
            "dataset_dir": str(self.dataset_dir),
            "frame_dir": str(frame_dir),
            "frame_name": frame_dir.name,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "coordinate_origin": "top_left",
            "image_space": {
                "name": "model",
                "description": "x/y are pixels in images.npy after VGGT preprocessing",
            },
            "cameras": [
                {
                    "index": cam.index,
                    "name": cam.name,
                    "image_size": [cam.width, cam.height],
                    "source_size": [cam.source_width, cam.source_height],
                    "source_mapping": {
                        "source_x": f"x * {cam.source_width} / {cam.width}",
                        "source_y": f"y * {cam.source_height} / {cam.height}",
                    },
                }
                for cam in self.cameras
            ],
            "distance_units": {
                "stored_metric": "distance_mm",
                "stored_uncertainty": "sigma_mm",
                "accepted_units": sorted(UNIT_TO_MM),
            },
            "points": normalized_points,
            "distances": normalized_distances,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp_path.replace(output_path)

        return {
            "ok": True,
            "output_path": str(output_path),
            "frame_dir": str(frame_dir),
            "frame_name": frame_dir.name,
            "point_count": len(normalized_points),
            "distance_count": len(normalized_distances),
            "warnings": warnings,
        }

    def reconstruct(self, payload: dict[str, Any]) -> dict[str, Any]:
        frame_option_index = self.frame_option_index_from_payload(payload)
        frame_dir = self.frame_dir_for_option(frame_option_index)
        calibrations = _read_calibration_full(frame_dir / "calibration.toml")
        points_in = payload.get("points") or []
        distances_in = payload.get("distances") or []
        if not isinstance(points_in, list):
            raise ValueError("Payload must contain a points list")
        if not isinstance(distances_in, list):
            raise ValueError("Payload distances must be a list")

        camera_by_name = {cam.name: cam for cam in self.cameras}
        raw_points: dict[str, np.ndarray] = {}
        point_records: list[dict[str, Any]] = []
        warnings: list[str] = []

        for point in points_in:
            if not isinstance(point, dict):
                continue
            point_id = str(point.get("id") or "").strip()
            if not point_id:
                continue
            observations = point.get("observations") or {}
            if not isinstance(observations, dict):
                continue

            rays: list[tuple[str, float, float, np.ndarray]] = []
            used_observations: dict[str, dict[str, float]] = {}
            for cam_name, obs in observations.items():
                if not isinstance(obs, dict) or cam_name not in calibrations:
                    continue
                view = camera_by_name.get(cam_name)
                if view is None:
                    continue
                try:
                    if "source_x" in obs and "source_y" in obs:
                        source_x = float(obs["source_x"])
                        source_y = float(obs["source_y"])
                    else:
                        x = float(obs["x"])
                        y = float(obs["y"])
                        source_x = x * view.source_width / view.width
                        source_y = y * view.source_height / view.height
                except (KeyError, TypeError, ValueError):
                    continue
                rays.append((cam_name, source_x, source_y, calibrations[cam_name].projection))
                used_observations[cam_name] = {"source_x": source_x, "source_y": source_y}

            if len(rays) < 2:
                warnings.append(f"{point_id} needs observations in at least 2 calibrated cameras")
                continue

            try:
                xyz = _triangulate_point(rays)
            except np.linalg.LinAlgError as exc:
                warnings.append(f"{point_id} failed triangulation: {exc}")
                continue
            if not np.all(np.isfinite(xyz)):
                warnings.append(f"{point_id} triangulated to non-finite coordinates")
                continue

            reproj_errors = []
            for cam_name, source_x, source_y, projection in rays:
                proj = projection @ np.array([xyz[0], xyz[1], xyz[2], 1.0], dtype=np.float64)
                if abs(float(proj[2])) <= 1e-12:
                    continue
                pred = proj[:2] / proj[2]
                reproj_errors.append(float(np.linalg.norm(pred - np.array([source_x, source_y], dtype=np.float64))))

            raw_points[point_id] = xyz
            point_records.append(
                {
                    "id": point_id,
                    "label": str(point.get("label") or point_id),
                    "raw_xyz": xyz.tolist(),
                    "observations": used_observations,
                    "num_observations": len(used_observations),
                    "reprojection_rmse_px": float(np.sqrt(np.mean(np.square(reproj_errors)))) if reproj_errors else None,
                }
            )

        distances = _normalize_distances_for_reconstruction(distances_in)
        scale_pairs = []
        edges: list[dict[str, Any]] = []
        for distance in distances:
            point_a = distance["point_a"]
            point_b = distance["point_b"]
            if point_a not in raw_points or point_b not in raw_points:
                warnings.append(f"{distance['id']} references a point that could not be reconstructed")
                continue
            raw_length = float(np.linalg.norm(raw_points[point_a] - raw_points[point_b]))
            target_mm = float(distance["distance_mm"])
            if raw_length <= 1e-12:
                warnings.append(f"{distance['id']} has zero reconstructed length")
                continue
            if distance["use_for_refinement"]:
                sigma_mm = max(float(distance["sigma_mm"]), 1e-9)
                scale_pairs.append((raw_length, target_mm, sigma_mm))
            edges.append(
                {
                    "id": distance["id"],
                    "point_a": point_a,
                    "point_b": point_b,
                    "target_mm": target_mm,
                    "sigma_mm": float(distance["sigma_mm"]),
                    "use_for_refinement": bool(distance["use_for_refinement"]),
                    "role": "fit" if distance["use_for_refinement"] else "holdout",
                    "raw_length": raw_length,
                }
            )

        if scale_pairs:
            raw_lengths = np.array([item[0] for item in scale_pairs], dtype=np.float64)
            target_lengths = np.array([item[1] for item in scale_pairs], dtype=np.float64)
            weights = 1.0 / np.square(np.array([item[2] for item in scale_pairs], dtype=np.float64))
            denom = float(np.sum(weights * raw_lengths * raw_lengths))
            scale_to_mm = float(np.sum(weights * raw_lengths * target_lengths) / denom) if denom > 1e-12 else 1.0
            coordinate_unit = "mm"
        else:
            scale_to_mm = 1.0
            coordinate_unit = "vggt_raw"
            warnings.append("No fit distance constraints; 3D coordinates are in raw VGGT scale")

        if point_records:
            coords = np.array([np.asarray(record["raw_xyz"], dtype=np.float64) * scale_to_mm for record in point_records])
            center = np.mean(coords, axis=0)
        else:
            center = np.zeros(3, dtype=np.float64)

        for record in point_records:
            xyz_mm = np.asarray(record["raw_xyz"], dtype=np.float64) * scale_to_mm
            record["xyz"] = (xyz_mm - center).tolist()
            del record["raw_xyz"]

        for edge in edges:
            reconstructed_mm = edge["raw_length"] * scale_to_mm
            edge["reconstructed_mm"] = reconstructed_mm
            edge["residual_mm"] = reconstructed_mm - edge["target_mm"]
            edge["normalized_residual"] = edge["residual_mm"] / max(float(edge["sigma_mm"]), 1e-9)

        return {
            "ok": True,
            "coordinate_unit": coordinate_unit,
            "scale_to_mm": scale_to_mm,
            "center_subtracted": center.tolist(),
            "points": point_records,
            "edges": edges,
            "warnings": warnings,
        }


def _read_calibration_minimal(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        return {}
    if tomllib is not None:
        try:
            with path.open("rb") as handle:
                data = tomllib.load(handle)
            out: dict[int, dict[str, Any]] = {}
            for key, value in data.items():
                match = re.fullmatch(r"cam_(\d+)", str(key))
                if match and isinstance(value, dict):
                    out[int(match.group(1))] = {
                        "name": value.get("name"),
                        "size": value.get("size"),
                    }
            return out
        except Exception:
            pass

    out = {}
    current: int | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        table = re.match(r"\s*\[cam_(\d+)\]\s*$", line)
        if table:
            current = int(table.group(1))
            out.setdefault(current, {})
            continue
        if current is None:
            continue
        name = re.match(r'\s*name\s*=\s*"([^"]+)"\s*$', line)
        if name:
            out[current]["name"] = name.group(1)
            continue
        size = re.match(r"\s*size\s*=\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]\s*$", line)
        if size:
            out[current]["size"] = [int(size.group(1)), int(size.group(2))]
    return out


def _read_calibration_full(path: Path) -> dict[str, CameraCalibration]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing calibration.toml: {path}")
    if tomllib is None:
        raise RuntimeError("Python 3.11+ is required for calibration parsing, or install toml.")
    with path.open("rb") as handle:
        data = tomllib.load(handle)

    cameras: dict[str, CameraCalibration] = {}
    for table_key, table_value in data.items():
        if not isinstance(table_value, dict) or not str(table_key).startswith("cam_"):
            continue
        name = str(table_value.get("name") or table_key)
        matrix = np.asarray(table_value.get("matrix"), dtype=np.float64).reshape(3, 3)
        rotation_vec = np.asarray(table_value.get("rotation"), dtype=np.float64).reshape(3)
        rotation, _ = cv2.Rodrigues(rotation_vec)
        translation = np.asarray(table_value.get("translation"), dtype=np.float64).reshape(3)
        size_value = table_value.get("size") or [0, 0]
        size = (int(size_value[0]), int(size_value[1]))
        cameras[name] = CameraCalibration(
            name=name,
            size=size,
            matrix=matrix,
            rotation=rotation.astype(np.float64),
            translation=translation.astype(np.float64),
        )
    if not cameras:
        raise ValueError(f"No camera calibrations found in {path}")
    return cameras


def _triangulate_point(rays: list[tuple[str, float, float, np.ndarray]]) -> np.ndarray:
    rows = []
    for _, x, y, projection in rays:
        rows.append(x * projection[2, :] - projection[0, :])
        rows.append(y * projection[2, :] - projection[1, :])
    system = np.asarray(rows, dtype=np.float64)
    _, _, vt = np.linalg.svd(system)
    homogeneous = vt[-1]
    if abs(float(homogeneous[3])) <= 1e-12:
        raise np.linalg.LinAlgError("homogeneous coordinate is near zero")
    return (homogeneous[:3] / homogeneous[3]).astype(np.float64)


def _normalize_distances_for_reconstruction(distances: list[Any]) -> list[dict[str, Any]]:
    normalized = []
    for idx, distance in enumerate(distances, start=1):
        if not isinstance(distance, dict):
            continue
        distance_id = str(distance.get("id") or f"D{idx:03d}").strip()
        point_a = str(distance.get("point_a") or "").strip()
        point_b = str(distance.get("point_b") or "").strip()
        unit = str(distance.get("unit") or "mm").strip().lower()
        sigma_unit = str(distance.get("sigma_unit") or unit).strip().lower()
        try:
            value = float(distance.get("value"))
        except (TypeError, ValueError):
            continue
        if not distance_id or not point_a or not point_b or point_a == point_b or value <= 0:
            continue
        if unit not in UNIT_TO_MM or sigma_unit not in UNIT_TO_MM:
            continue
        if "distance_mm" in distance:
            try:
                distance_mm = float(distance["distance_mm"])
            except (TypeError, ValueError):
                distance_mm = value * UNIT_TO_MM[unit]
        else:
            distance_mm = value * UNIT_TO_MM[unit]
        if distance_mm <= 0:
            continue
        try:
            if "sigma_mm" in distance:
                sigma_mm = float(distance["sigma_mm"])
            else:
                sigma_value = float(distance.get("sigma_value", DEFAULT_DISTANCE_SIGMA_MM / UNIT_TO_MM[sigma_unit]))
                sigma_mm = sigma_value * UNIT_TO_MM[sigma_unit]
        except (TypeError, ValueError):
            sigma_mm = DEFAULT_DISTANCE_SIGMA_MM
        if sigma_mm <= 0:
            sigma_mm = DEFAULT_DISTANCE_SIGMA_MM
        use_for_refinement = bool(distance.get("use_for_refinement", not bool(distance.get("holdout", False))))
        normalized.append(
            {
                "id": distance_id,
                "point_a": point_a,
                "point_b": point_b,
                "distance_mm": distance_mm,
                "sigma_mm": sigma_mm,
                "use_for_refinement": use_for_refinement,
            }
        )
    return normalized


def make_handler(state: PointSelectorState) -> type[BaseHTTPRequestHandler]:
    class PointSelectorHandler(BaseHTTPRequestHandler):
        server_version = "VGGTPointSelector/1.0"

        def log_message(self, fmt: str, *args: object) -> None:
            sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    self._send_text(HTML_PAGE, "text/html; charset=utf-8")
                    return
                if parsed.path == "/app.js":
                    self._send_text(APP_JS, "application/javascript; charset=utf-8")
                    return
                if parsed.path == "/style.css":
                    self._send_text(STYLE_CSS, "text/css; charset=utf-8")
                    return
                if parsed.path == "/api/frame":
                    self._send_json(state.metadata())
                    return
                match = re.fullmatch(r"/api/image/(\d+)/(\d+)\.png", parsed.path)
                if match:
                    data = state.image_png(int(match.group(2)), int(match.group(1)))
                    self._send_bytes(data, "image/png")
                    return
                match = re.fullmatch(r"/api/image/(\d+)\.png", parsed.path)
                if match:
                    data = state.image_png(int(match.group(1)))
                    self._send_bytes(data, "image/png")
                    return
                self._send_error(404, "Not found")
            except Exception as exc:
                self._send_error(500, str(exc))

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path not in {"/api/points", "/api/reconstruction"}:
                    self._send_error(404, "Not found")
                    return
                length = int(self.headers.get("Content-Length") or "0")
                body = self.rfile.read(length)
                payload = json.loads(body.decode("utf-8"))
                if parsed.path == "/api/reconstruction":
                    result = state.reconstruct(payload)
                else:
                    result = state.save_points(payload)
                self._send_json(result)
            except Exception as exc:
                self._send_error(400, str(exc))

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            data = json.dumps(payload, allow_nan=False).encode("utf-8")
            self._send_bytes(data, "application/json; charset=utf-8", status)

        def _send_text(self, text: str, content_type: str, status: int = 200) -> None:
            self._send_bytes(text.encode("utf-8"), content_type, status)

        def _send_bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type or mimetypes.types_map.get(".bin", "application/octet-stream"))
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _send_error(self, status: int, message: str) -> None:
            self._send_json({"ok": False, "error": message}, status)

    return PointSelectorHandler


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VGGT Point Selector</title>
  <link rel="stylesheet" href="/style.css">
</head>
<body>
  <header class="topbar">
    <div class="titleblock">
      <h1>VGGT Point Selector</h1>
      <div id="framePath" class="pathline"></div>
    </div>
    <div class="frameScrubber" title="Change displayed frame while keeping the same annotations">
      <button id="framePrevBtn" type="button" title="Previous frame">-</button>
      <label class="frameSliderWrap">
        <span>Display frame</span>
        <input id="frameSlider" type="range" min="0" max="0" step="1" value="0">
      </label>
      <button id="frameNextBtn" type="button" title="Next frame">+</button>
      <div id="frameReadout" class="frameReadout">frame_000000</div>
    </div>
    <div class="toolbar">
      <button id="newPointBtn" title="Create a new point ID">New</button>
      <button id="deletePointBtn" title="Delete the selected point">Delete</button>
      <button id="labelsToggleBtn" class="toggle" title="Show labels for all points">Labels</button>
      <button id="render3dBtn" title="Render metric 3D reconstruction">3D</button>
      <button id="saveBtn" class="primary" title="Save selected points">Save</button>
    </div>
  </header>

  <main class="workspace">
    <aside class="sidebar">
      <section class="panel">
        <div class="panelHead">
          <h2>Points</h2>
          <span id="pointCount" class="counter">0</span>
        </div>
        <div id="pointList" class="pointList"></div>
      </section>

      <section class="panel detail">
        <div class="panelHead">
          <h2>Selected</h2>
          <span id="selectedStatus" class="statusPill">None</span>
        </div>
        <label class="field">
          <span>ID</span>
          <input id="pointIdInput" type="text" autocomplete="off">
        </label>
        <label class="field">
          <span>Label</span>
          <input id="pointLabelInput" type="text" autocomplete="off">
        </label>
        <label class="field">
          <span>Notes</span>
          <textarea id="pointNotesInput" rows="3"></textarea>
        </label>
        <label class="field">
          <span>Click uncertainty px</span>
          <input id="pointSigmaInput" type="number" min="0.1" step="0.1" autocomplete="off">
        </label>
        <div id="obsEditor" class="obsEditor"></div>
      </section>

      <section class="panel detail">
        <div class="panelHead">
          <h2>Distances</h2>
          <span id="distanceCount" class="counter">0</span>
        </div>
        <div class="distanceForm">
          <label class="field compact">
            <span>Point A</span>
            <select id="distancePointA"></select>
          </label>
          <label class="field compact">
            <span>Point B</span>
            <select id="distancePointB"></select>
          </label>
          <div class="distanceValueRow">
            <label class="field compact">
              <span>Distance</span>
              <input id="distanceValueInput" type="number" min="0" step="0.001" autocomplete="off">
            </label>
            <label class="field compact">
              <span>Unit</span>
              <select id="distanceUnitSelect">
                <option value="mm">mm</option>
                <option value="cm">cm</option>
                <option value="m">m</option>
              </select>
            </label>
          </div>
          <div class="distanceValueRow">
            <label class="field compact">
              <span>Uncertainty +/-</span>
              <input id="distanceSigmaInput" type="number" min="0" step="0.001" autocomplete="off">
            </label>
            <label class="field compact">
              <span>Unit</span>
              <select id="distanceSigmaUnitSelect">
                <option value="mm">mm</option>
                <option value="cm">cm</option>
                <option value="m">m</option>
              </select>
            </label>
          </div>
          <label class="checkField">
            <input id="distanceUseForRefinementInput" type="checkbox" checked>
            <span>Use for refinement</span>
          </label>
          <label class="field compact">
            <span>Notes</span>
            <textarea id="distanceNotesInput" rows="2"></textarea>
          </label>
          <div class="distanceButtons">
            <button id="saveDistanceBtn" type="button">Add Distance</button>
            <button id="clearDistanceBtn" type="button">Clear</button>
          </div>
        </div>
        <div id="distanceList" class="distanceList"></div>
      </section>

      <section class="panel">
        <div class="panelHead">
          <h2>Save</h2>
          <span id="saveStatus" class="statusPill">Clean</span>
        </div>
        <div id="outputPath" class="smallText"></div>
        <div id="warnings" class="warnings"></div>
      </section>
    </aside>

    <section id="imageGrid" class="imageGrid"></section>
  </main>

  <script src="/app.js"></script>
</body>
</html>
"""


STYLE_CSS = r"""
:root {
  color-scheme: dark;
  --bg: #111315;
  --surface: #191d21;
  --surface-2: #222830;
  --line: #333c46;
  --text: #eceff1;
  --muted: #a8b0b8;
  --accent: #49a6ff;
  --accent-2: #f2c14e;
  --danger: #ff6b6b;
  --ok: #59c28f;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 14px/1.35 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.topbar {
  min-height: 64px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 16px;
  padding: 10px 16px;
  border-bottom: 1px solid var(--line);
  background: #15191d;
}

h1, h2 {
  margin: 0;
  font-weight: 650;
}

h1 {
  font-size: 18px;
}

h2 {
  font-size: 13px;
  color: var(--muted);
  text-transform: uppercase;
}

.titleblock {
  min-width: 0;
}

.pathline {
  margin-top: 3px;
  color: var(--muted);
  font-size: 12px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 72vw;
}

.toolbar {
  display: flex;
  align-items: center;
  gap: 8px;
  flex: 0 0 auto;
}

.frameScrubber {
  display: grid;
  grid-template-columns: auto minmax(180px, 360px) auto minmax(112px, auto);
  align-items: center;
  gap: 8px;
  flex: 1 1 420px;
  min-width: 280px;
  max-width: 620px;
}

.frameScrubber button {
  width: 30px;
  height: 30px;
  padding: 0;
  font-weight: 700;
}

.frameSliderWrap {
  display: grid;
  gap: 3px;
}

.frameSliderWrap span {
  color: var(--muted);
  font-size: 11px;
  text-transform: uppercase;
}

.frameSliderWrap input {
  padding: 0;
}

.frameReadout {
  color: var(--muted);
  font-size: 12px;
  white-space: nowrap;
}

button {
  border: 1px solid var(--line);
  background: var(--surface-2);
  color: var(--text);
  padding: 8px 12px;
  border-radius: 6px;
  cursor: pointer;
  font: inherit;
}

button:hover {
  border-color: var(--accent);
}

button.primary {
  background: var(--accent);
  border-color: var(--accent);
  color: #07131f;
  font-weight: 650;
}

button.toggle.active {
  border-color: var(--accent-2);
  color: #17120a;
  background: var(--accent-2);
}

.workspace {
  display: grid;
  grid-template-columns: minmax(280px, 340px) minmax(0, 1fr);
  height: calc(100vh - 65px);
  min-height: 520px;
}

.sidebar {
  border-right: 1px solid var(--line);
  background: var(--surface);
  overflow: auto;
  padding: 12px;
}

.panel {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #171b1f;
  margin-bottom: 12px;
  overflow: hidden;
}

.panelHead {
  height: 38px;
  padding: 0 10px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid var(--line);
  background: #1d2228;
}

.counter, .statusPill {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 28px;
  height: 22px;
  border-radius: 999px;
  padding: 0 8px;
  background: var(--surface-2);
  color: var(--muted);
  font-size: 12px;
}

.statusPill.dirty {
  color: #17120a;
  background: var(--accent-2);
}

.statusPill.good {
  color: #06140d;
  background: var(--ok);
}

.pointList {
  max-height: 260px;
  overflow: auto;
}

.pointRow {
  display: grid;
  grid-template-columns: minmax(64px, 1fr) auto;
  gap: 8px;
  align-items: center;
  padding: 8px 10px;
  border-bottom: 1px solid #252b31;
  cursor: pointer;
}

.pointRow:last-child {
  border-bottom: none;
}

.pointRow:hover {
  background: #20262c;
}

.pointRow.active {
  background: #17324a;
}

.pointMain {
  min-width: 0;
}

.pointId {
  font-weight: 650;
}

.pointSub {
  color: var(--muted);
  font-size: 12px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.camDots {
  display: flex;
  gap: 5px;
}

.camDot {
  width: 24px;
  height: 22px;
  border-radius: 6px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: #2b3138;
  color: #7f8993;
  font-size: 12px;
  font-weight: 650;
}

.camDot.on {
  color: #111315;
  background: var(--accent-2);
}

.detail {
  padding-bottom: 8px;
}

.field {
  display: grid;
  gap: 5px;
  padding: 8px 10px 0;
}

.field span {
  color: var(--muted);
  font-size: 12px;
}

input, textarea, select {
  width: 100%;
  background: #111519;
  border: 1px solid var(--line);
  border-radius: 6px;
  color: var(--text);
  padding: 8px;
  font: inherit;
}

textarea {
  resize: vertical;
}

select {
  appearance: none;
}

.field.compact {
  padding-top: 6px;
}

.distanceForm {
  padding: 2px 0 8px;
}

.distanceValueRow {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 84px;
  gap: 6px;
}

.distanceButtons {
  display: flex;
  gap: 8px;
  padding: 8px 10px 0;
}

.distanceButtons button {
  flex: 1;
}

.checkField {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 10px 0;
  color: var(--muted);
  font-size: 12px;
}

.checkField input {
  width: auto;
}

.distanceList {
  border-top: 1px solid var(--line);
  max-height: 180px;
  overflow: auto;
}

.distanceRow {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 8px;
  align-items: center;
  padding: 8px 10px;
  border-bottom: 1px solid #252b31;
  cursor: pointer;
}

.distanceRow:last-child {
  border-bottom: none;
}

.distanceRow:hover {
  background: #20262c;
}

.distanceRow.active {
  background: #17324a;
}

.distanceTitle {
  font-weight: 650;
}

.distanceSub {
  color: var(--muted);
  font-size: 12px;
}

.distanceDelete {
  height: 30px;
  padding: 0 8px;
  color: var(--danger);
}

.obsEditor {
  padding: 8px 10px 0;
}

.obsRow {
  display: grid;
  grid-template-columns: 28px 1fr 1fr auto;
  gap: 6px;
  align-items: center;
  margin-bottom: 6px;
}

.obsRow input {
  padding: 6px;
}

.obsRow button {
  height: 31px;
  padding: 0 8px;
  color: var(--danger);
}

.smallText {
  padding: 10px;
  color: var(--muted);
  font-size: 12px;
  word-break: break-all;
}

.warnings {
  padding: 0 10px 10px;
  color: var(--accent-2);
  font-size: 12px;
}

.imageGrid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(410px, 1fr));
  gap: 12px;
  padding: 12px;
  overflow: auto;
  align-content: start;
}

.renderPanel {
  grid-column: 1 / -1;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #101316;
  overflow: hidden;
}

.renderPanel.hidden {
  display: none;
}

.renderHead {
  min-height: 36px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 6px 10px;
  background: #1d2228;
  border-bottom: 1px solid var(--line);
}

.renderTitle {
  font-weight: 650;
}

.renderMeta {
  color: var(--muted);
  font-size: 12px;
}

.renderBody {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(260px, 340px);
  min-height: 420px;
}

.renderCanvas {
  width: 100%;
  height: 460px;
  display: block;
  background: #050607;
  cursor: grab;
}

.renderCanvas.dragging {
  cursor: grabbing;
}

.renderStats {
  border-left: 1px solid var(--line);
  padding: 10px;
  overflow: auto;
  max-height: 460px;
}

.renderStatsTable {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}

.renderStatsTable th,
.renderStatsTable td {
  border-bottom: 1px solid #252b31;
  padding: 5px 4px;
  text-align: left;
}

.renderStatsTable th {
  color: var(--muted);
  font-weight: 650;
}

.imagePanel {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #101316;
  overflow: hidden;
  min-width: 0;
}

.imageHead {
  min-height: 36px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  padding: 6px 10px;
  background: #1d2228;
  border-bottom: 1px solid var(--line);
}

.cameraName {
  font-weight: 650;
}

.cameraMeta {
  color: var(--muted);
  font-size: 12px;
}

.imageControls {
  display: grid;
  gap: 5px;
  justify-items: end;
  flex: 0 0 auto;
}

.zoomControls {
  display: flex;
  align-items: center;
  gap: 5px;
}

.zoomControls button {
  width: 30px;
  height: 28px;
  padding: 0;
  font-weight: 700;
}

.zoomReadout {
  min-width: 48px;
  color: var(--muted);
  font-size: 12px;
  text-align: right;
}

.contrastControl {
  display: grid;
  grid-template-columns: auto minmax(90px, 120px) 42px;
  align-items: center;
  gap: 6px;
  color: var(--muted);
  font-size: 12px;
}

.contrastControl input {
  padding: 0;
}

.contrastReadout {
  text-align: right;
}

.imageStage {
  position: relative;
  width: 100%;
  aspect-ratio: var(--image-aspect);
  background: #050607;
  user-select: none;
  overflow: hidden;
  touch-action: none;
  cursor: crosshair;
}

.imageStage.panning {
  cursor: grabbing;
}

.imageStage.zoomed {
  cursor: grab;
}

.imageStage.zoomed.panning {
  cursor: grabbing;
}

.imageLayer {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  transform-origin: 0 0;
  transform: translate(var(--pan-x, 0px), var(--pan-y, 0px)) scale(var(--zoom, 1));
}

.cameraImage {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: fill;
  pointer-events: none;
  filter: contrast(var(--contrast, 1));
}

.overlay {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  overflow: visible;
}

.markerCircle {
  fill: rgba(242, 193, 78, 0.88);
  stroke: #111315;
  stroke-width: 2.5;
}

.markerCircle.current {
  fill: rgba(73, 166, 255, 0.95);
}

.markerLabel {
  fill: #ffffff;
  stroke: #000000;
  stroke-width: 3;
  paint-order: stroke;
  font-size: 14px;
  font-weight: 750;
  pointer-events: none;
}

@media (max-width: 900px) {
  .workspace {
    grid-template-columns: 1fr;
    height: auto;
  }

  .sidebar {
    border-right: none;
    border-bottom: 1px solid var(--line);
  }

  .imageGrid {
    grid-template-columns: 1fr;
  }

  .renderBody {
    grid-template-columns: 1fr;
  }

  .renderStats {
    border-left: none;
    border-top: 1px solid var(--line);
  }
}
"""


APP_JS = r"""
const state = {
  frame: null,
  cameras: [],
  points: [],
  distances: [],
  selectedId: null,
  selectedDistanceId: null,
  displayFrames: [],
  displayFrameIndex: 0,
  dirty: false,
  viewByCamera: {},
  showAllLabels: false,
  reconstruction: null,
  renderView: { rotX: -0.5, rotY: 0.75, zoom: 1, panX: 0, panY: 0 },
};

const UNIT_TO_MM = { mm: 1, cm: 10, m: 1000 };
const DEFAULT_REPROJECTION_SIGMA_PX = 2;
const DEFAULT_DISTANCE_SIGMA_MM = 1;

const els = {
  framePath: document.getElementById("framePath"),
  outputPath: document.getElementById("outputPath"),
  imageGrid: document.getElementById("imageGrid"),
  pointList: document.getElementById("pointList"),
  pointCount: document.getElementById("pointCount"),
  distanceList: document.getElementById("distanceList"),
  distanceCount: document.getElementById("distanceCount"),
  selectedStatus: document.getElementById("selectedStatus"),
  saveStatus: document.getElementById("saveStatus"),
  warnings: document.getElementById("warnings"),
  frameSlider: document.getElementById("frameSlider"),
  framePrevBtn: document.getElementById("framePrevBtn"),
  frameNextBtn: document.getElementById("frameNextBtn"),
  frameReadout: document.getElementById("frameReadout"),
  pointIdInput: document.getElementById("pointIdInput"),
  pointLabelInput: document.getElementById("pointLabelInput"),
  pointNotesInput: document.getElementById("pointNotesInput"),
  pointSigmaInput: document.getElementById("pointSigmaInput"),
  obsEditor: document.getElementById("obsEditor"),
  distancePointA: document.getElementById("distancePointA"),
  distancePointB: document.getElementById("distancePointB"),
  distanceValueInput: document.getElementById("distanceValueInput"),
  distanceUnitSelect: document.getElementById("distanceUnitSelect"),
  distanceSigmaInput: document.getElementById("distanceSigmaInput"),
  distanceSigmaUnitSelect: document.getElementById("distanceSigmaUnitSelect"),
  distanceUseForRefinementInput: document.getElementById("distanceUseForRefinementInput"),
  distanceNotesInput: document.getElementById("distanceNotesInput"),
  saveDistanceBtn: document.getElementById("saveDistanceBtn"),
  clearDistanceBtn: document.getElementById("clearDistanceBtn"),
  newPointBtn: document.getElementById("newPointBtn"),
  deletePointBtn: document.getElementById("deletePointBtn"),
  labelsToggleBtn: document.getElementById("labelsToggleBtn"),
  render3dBtn: document.getElementById("render3dBtn"),
  saveBtn: document.getElementById("saveBtn"),
};

function cleanPoint(point) {
  const sigmaPx = Number(point.sigma_px);
  return {
    id: String(point.id || "").trim(),
    label: String(point.label || point.id || "").trim(),
    notes: String(point.notes || ""),
    sigma_px: Number.isFinite(sigmaPx) && sigmaPx > 0 ? sigmaPx : DEFAULT_REPROJECTION_SIGMA_PX,
    observations: point.observations && typeof point.observations === "object" ? point.observations : {},
  };
}

function cleanDistance(distance, index = 0) {
  const unit = UNIT_TO_MM[String(distance.unit || "mm").toLowerCase()] ? String(distance.unit || "mm").toLowerCase() : "mm";
  const sigmaUnit = UNIT_TO_MM[String(distance.sigma_unit || unit).toLowerCase()] ? String(distance.sigma_unit || unit).toLowerCase() : unit;
  const value = Number(distance.value);
  let sigmaValue = Number(distance.sigma_value);
  if (!Number.isFinite(sigmaValue) && Number.isFinite(Number(distance.sigma_mm))) {
    sigmaValue = Number(distance.sigma_mm) / UNIT_TO_MM[sigmaUnit];
  }
  return {
    id: String(distance.id || `D${String(index + 1).padStart(3, "0")}`).trim(),
    point_a: String(distance.point_a || "").trim(),
    point_b: String(distance.point_b || "").trim(),
    value: Number.isFinite(value) ? value : "",
    unit,
    sigma_value: Number.isFinite(sigmaValue) && sigmaValue > 0 ? sigmaValue : DEFAULT_DISTANCE_SIGMA_MM / UNIT_TO_MM[sigmaUnit],
    sigma_unit: sigmaUnit,
    use_for_refinement: Boolean(distance.use_for_refinement ?? !distance.holdout),
    notes: String(distance.notes || ""),
  };
}

function loadExisting(existing) {
  if (!existing || !Array.isArray(existing.points)) {
    state.points = [];
    state.distances = [];
    return;
  }
  state.points = existing.points.map(cleanPoint).filter((p) => p.id);
  state.distances = Array.isArray(existing.distances)
    ? existing.distances.map(cleanDistance).filter((d) => d.id)
    : [];
}

function nextPointId() {
  let maxNum = 0;
  for (const point of state.points) {
    const match = /^P(\d+)$/i.exec(point.id);
    if (match) maxNum = Math.max(maxNum, Number(match[1]));
  }
  return `P${String(maxNum + 1).padStart(3, "0")}`;
}

function nextDistanceId() {
  let maxNum = 0;
  for (const distance of state.distances) {
    const match = /^D(\d+)$/i.exec(distance.id);
    if (match) maxNum = Math.max(maxNum, Number(match[1]));
  }
  return `D${String(maxNum + 1).padStart(3, "0")}`;
}

function selectedPoint() {
  return state.points.find((point) => point.id === state.selectedId) || null;
}

function selectedDistance() {
  return state.distances.find((distance) => distance.id === state.selectedDistanceId) || null;
}

function setDirty(value) {
  state.dirty = value;
  els.saveStatus.textContent = value ? "Dirty" : "Clean";
  els.saveStatus.classList.toggle("dirty", value);
  els.saveStatus.classList.toggle("good", !value);
}

function selectPoint(id) {
  state.selectedId = id;
  renderAll();
}

function createPoint() {
  const id = nextPointId();
  state.points.push({ id, label: id, notes: "", sigma_px: DEFAULT_REPROJECTION_SIGMA_PX, observations: {} });
  state.selectedId = id;
  setDirty(true);
  renderAll();
}

function deleteSelectedPoint() {
  if (!state.selectedId) return;
  const deletedId = state.selectedId;
  state.points = state.points.filter((point) => point.id !== state.selectedId);
  state.distances = state.distances.filter((distance) => distance.point_a !== deletedId && distance.point_b !== deletedId);
  state.selectedId = state.points[0]?.id || null;
  if (selectedDistance() === null) {
    state.selectedDistanceId = null;
  }
  setDirty(true);
  renderAll();
}

function currentOrNewPoint() {
  let point = selectedPoint();
  if (!point) {
    createPoint();
    point = selectedPoint();
  }
  return point;
}

function addObservation(camera, x, y) {
  const point = currentOrNewPoint();
  point.observations[camera.name] = {
    x: clamp(x, 0, camera.width - 1),
    y: clamp(y, 0, camera.height - 1),
  };
  setDirty(true);
  renderAll();
}

function removeObservation(cameraName) {
  const point = selectedPoint();
  if (!point) return;
  delete point.observations[cameraName];
  setDirty(true);
  renderAll();
}

function clamp(value, minValue, maxValue) {
  return Math.max(minValue, Math.min(maxValue, value));
}

function cameraByName(name) {
  return state.cameras.find((camera) => camera.name === name);
}

function sourceXY(camera, obs) {
  return {
    x: obs.x * camera.source_width / camera.width,
    y: obs.y * camera.source_height / camera.height,
  };
}

function displayFrame() {
  return state.displayFrames[state.displayFrameIndex] || null;
}

function cameraImageUrl(camera) {
  return `/api/image/${state.displayFrameIndex}/${camera.index}.png`;
}

function updateDisplayFrameUi(updateImages = false) {
  const frame = displayFrame();
  const maxIndex = Math.max(0, state.displayFrames.length - 1);
  els.frameSlider.min = "0";
  els.frameSlider.max = String(maxIndex);
  els.frameSlider.step = "1";
  els.frameSlider.value = String(state.displayFrameIndex);
  els.frameSlider.disabled = state.displayFrames.length <= 1;
  els.framePrevBtn.disabled = state.displayFrameIndex <= 0;
  els.frameNextBtn.disabled = state.displayFrameIndex >= maxIndex;
  els.frameReadout.textContent = frame
    ? `${frame.name} (${state.displayFrameIndex + 1}/${state.displayFrames.length})`
    : "No display frames";
  els.frameReadout.title = frame?.frame_dir || "";
  if (frame?.output_path) {
    els.outputPath.textContent = frame.output_path;
  }
  if (!updateImages) return;
  for (const camera of state.cameras) {
    const stage = stageForCamera(camera);
    const img = stage?.querySelector(".cameraImage");
    if (img) img.src = cameraImageUrl(camera);
  }
}

function setDisplayFrame(index) {
  const maxIndex = Math.max(0, state.displayFrames.length - 1);
  const nextIndex = Math.round(clamp(Number(index), 0, maxIndex));
  if (nextIndex === state.displayFrameIndex) {
    updateDisplayFrameUi(false);
    return;
  }
  state.displayFrameIndex = nextIndex;
  updateDisplayFrameUi(true);
}

function distanceMm(distance) {
  const value = Number(distance.value);
  return Number.isFinite(value) ? value * UNIT_TO_MM[distance.unit] : NaN;
}

function distanceSigmaMm(distance) {
  const value = Number(distance.sigma_value);
  return Number.isFinite(value) ? value * UNIT_TO_MM[distance.sigma_unit] : NaN;
}

function pointLabelById(pointId) {
  const point = state.points.find((p) => p.id === pointId);
  return point ? (point.label || point.id) : pointId;
}

function resetDistanceFormFields() {
  state.selectedDistanceId = null;
  els.distanceValueInput.value = "";
  els.distanceUnitSelect.value = "mm";
  els.distanceSigmaInput.value = String(DEFAULT_DISTANCE_SIGMA_MM);
  els.distanceSigmaUnitSelect.value = "mm";
  els.distanceUseForRefinementInput.checked = true;
  els.distanceNotesInput.value = "";
  if (state.points.length >= 2) {
    els.distancePointA.value = state.points[0].id;
    els.distancePointB.value = state.points[1].id;
  }
  els.saveDistanceBtn.textContent = "Add Distance";
}

function clearDistanceForm() {
  resetDistanceFormFields();
  renderDistances();
}

function populateDistanceForm(distance) {
  state.selectedDistanceId = distance.id;
  els.distancePointA.value = distance.point_a;
  els.distancePointB.value = distance.point_b;
  els.distanceValueInput.value = distance.value;
  els.distanceUnitSelect.value = distance.unit;
  els.distanceSigmaInput.value = distance.sigma_value;
  els.distanceSigmaUnitSelect.value = distance.sigma_unit;
  els.distanceUseForRefinementInput.checked = Boolean(distance.use_for_refinement);
  els.distanceNotesInput.value = distance.notes || "";
  els.saveDistanceBtn.textContent = "Update Distance";
  renderDistances();
}

function saveDistanceFromForm() {
  const pointA = els.distancePointA.value;
  const pointB = els.distancePointB.value;
  const value = Number(els.distanceValueInput.value);
  const unit = els.distanceUnitSelect.value;
  const sigmaValue = Number(els.distanceSigmaInput.value);
  const sigmaUnit = els.distanceSigmaUnitSelect.value;
  if (!pointA || !pointB || pointA === pointB || !Number.isFinite(value) || value <= 0 || !Number.isFinite(sigmaValue) || sigmaValue <= 0) {
    renderWarnings(["Distance needs two different points, a positive value, and a positive uncertainty"]);
    return;
  }
  const editingDistanceId = state.selectedDistanceId;
  const payload = {
    id: editingDistanceId || nextDistanceId(),
    point_a: pointA,
    point_b: pointB,
    value,
    unit,
    sigma_value: sigmaValue,
    sigma_unit: sigmaUnit,
    use_for_refinement: els.distanceUseForRefinementInput.checked,
    notes: els.distanceNotesInput.value,
  };
  const existing = state.distances.findIndex((distance) => distance.id === payload.id);
  if (existing >= 0) {
    state.distances[existing] = payload;
    state.selectedDistanceId = payload.id;
  } else {
    state.distances.push(payload);
    resetDistanceFormFields();
  }
  setDirty(true);
  renderAll();
}

function deleteDistance(distanceId) {
  state.distances = state.distances.filter((distance) => distance.id !== distanceId);
  if (state.selectedDistanceId === distanceId) {
    state.selectedDistanceId = null;
  }
  setDirty(true);
  renderAll();
}

function observationCount(point) {
  return Object.keys(point.observations || {}).length;
}

function ensureView(camera) {
  if (!state.viewByCamera[camera.name]) {
    state.viewByCamera[camera.name] = { zoom: 1, panX: 0, panY: 0, contrast: 1 };
  } else if (!Number.isFinite(state.viewByCamera[camera.name].contrast)) {
    state.viewByCamera[camera.name].contrast = 1;
  }
  return state.viewByCamera[camera.name];
}

function stageForCamera(camera) {
  return document.querySelector(`.imageStage[data-camera="${cssEscape(camera.name)}"]`);
}

function clampView(camera, view) {
  const stage = stageForCamera(camera);
  if (!stage || view.zoom <= 1) {
    view.zoom = Math.max(1, view.zoom);
    view.panX = 0;
    view.panY = 0;
    return;
  }
  const rect = stage.getBoundingClientRect();
  const minPanX = rect.width * (1 - view.zoom);
  const minPanY = rect.height * (1 - view.zoom);
  view.panX = clamp(view.panX, minPanX, 0);
  view.panY = clamp(view.panY, minPanY, 0);
}

function applyView(camera) {
  const stage = stageForCamera(camera);
  if (!stage) return;
  const view = ensureView(camera);
  clampView(camera, view);
  const layer = stage.querySelector(".imageLayer");
  const image = stage.querySelector(".cameraImage");
  const readout = document.querySelector(`.zoomReadout[data-camera="${cssEscape(camera.name)}"]`);
  const contrastReadout = document.querySelector(`.contrastReadout[data-camera="${cssEscape(camera.name)}"]`);
  const contrastInput = document.querySelector(`input[data-contrast-camera="${cssEscape(camera.name)}"]`);
  layer.style.setProperty("--zoom", String(view.zoom));
  layer.style.setProperty("--pan-x", `${view.panX}px`);
  layer.style.setProperty("--pan-y", `${view.panY}px`);
  image.style.setProperty("--contrast", String(view.contrast));
  stage.classList.toggle("zoomed", view.zoom > 1.001);
  if (readout) readout.textContent = `${Math.round(view.zoom * 100)}%`;
  if (contrastReadout) contrastReadout.textContent = `${Math.round(view.contrast * 100)}%`;
  if (contrastInput && Number(contrastInput.value) !== view.contrast) {
    contrastInput.value = String(view.contrast);
  }
}

function zoomCameraAt(camera, clientX, clientY, factor) {
  const stage = stageForCamera(camera);
  if (!stage) return;
  const rect = stage.getBoundingClientRect();
  const view = ensureView(camera);
  const oldZoom = view.zoom;
  const newZoom = clamp(oldZoom * factor, 1, 12);
  const cssX = clientX - rect.left;
  const cssY = clientY - rect.top;
  const imageCssX = (cssX - view.panX) / oldZoom;
  const imageCssY = (cssY - view.panY) / oldZoom;
  view.zoom = newZoom;
  view.panX = cssX - imageCssX * newZoom;
  view.panY = cssY - imageCssY * newZoom;
  applyView(camera);
}

function resetCameraView(camera) {
  const view = ensureView(camera);
  view.zoom = 1;
  view.panX = 0;
  view.panY = 0;
  applyView(camera);
}

function screenToImageCoords(camera, clientX, clientY) {
  const stage = stageForCamera(camera);
  const view = ensureView(camera);
  const rect = stage.getBoundingClientRect();
  const cssX = clientX - rect.left;
  const cssY = clientY - rect.top;
  const x = ((cssX - view.panX) / view.zoom) * camera.width / rect.width;
  const y = ((cssY - view.panY) / view.zoom) * camera.height / rect.height;
  return {
    x: clamp(x, 0, camera.width - 1),
    y: clamp(y, 0, camera.height - 1),
  };
}

function bindStagePointer(stage, camera) {
  let drag = null;

  stage.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    const view = ensureView(camera);
    drag = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      panX: view.panX,
      panY: view.panY,
      moved: false,
    };
    stage.setPointerCapture(event.pointerId);
  });

  stage.addEventListener("pointermove", (event) => {
    if (!drag || drag.pointerId !== event.pointerId) return;
    const dx = event.clientX - drag.startX;
    const dy = event.clientY - drag.startY;
    if (Math.hypot(dx, dy) < 3 && !drag.moved) return;
    drag.moved = true;
    stage.classList.add("panning");
    const view = ensureView(camera);
    view.panX = drag.panX + dx;
    view.panY = drag.panY + dy;
    applyView(camera);
  });

  function finishPointer(event) {
    if (!drag || drag.pointerId !== event.pointerId) return;
    stage.releasePointerCapture(event.pointerId);
    stage.classList.remove("panning");
    if (!drag.moved) {
      const coords = screenToImageCoords(camera, event.clientX, event.clientY);
      addObservation(camera, coords.x, coords.y);
    }
    drag = null;
  }

  stage.addEventListener("pointerup", finishPointer);
  stage.addEventListener("pointercancel", (event) => {
    if (drag && drag.pointerId === event.pointerId) {
      stage.releasePointerCapture(event.pointerId);
      stage.classList.remove("panning");
      drag = null;
    }
  });

  stage.addEventListener("wheel", (event) => {
    event.preventDefault();
    const factor = event.deltaY < 0 ? 1.18 : 1 / 1.18;
    zoomCameraAt(camera, event.clientX, event.clientY, factor);
  }, { passive: false });
}

async function requestReconstruction() {
  const response = await fetch("/api/reconstruction", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      points: state.points,
      distances: state.distances,
      display_frame_index: state.displayFrameIndex,
    }),
  });
  const result = await response.json();
  if (!response.ok || !result.ok) {
    renderWarnings([result.error || "3D reconstruction failed"]);
    return;
  }
  state.reconstruction = result;
  renderImages();
  draw3DScene();
  renderWarnings(result.warnings || []);
}

function renderReconstructionPanel() {
  const panel = document.createElement("article");
  panel.className = `renderPanel${state.reconstruction ? "" : " hidden"}`;
  const recon = state.reconstruction;
  const edgeRows = recon?.edges?.length
    ? recon.edges.map((edge) => `
        <tr>
          <td>${escapeHtml(edge.id)}</td>
          <td>${escapeHtml(edge.point_a)}-${escapeHtml(edge.point_b)}</td>
          <td>${escapeHtml(edge.role || "fit")}</td>
          <td>${Number(edge.target_mm).toFixed(2)}</td>
          <td>${Number(edge.sigma_mm).toFixed(2)}</td>
          <td>${Number(edge.reconstructed_mm).toFixed(2)}</td>
          <td>${Number(edge.residual_mm).toFixed(2)}</td>
          <td>${Number(edge.normalized_residual).toFixed(2)}</td>
        </tr>
      `).join("")
    : `<tr><td colspan="8">No valid distance edges</td></tr>`;
  const warnings = recon?.warnings?.length
    ? `<div class="warnings">${recon.warnings.map((w) => `<div>${escapeHtml(w)}</div>`).join("")}</div>`
    : "";
  panel.innerHTML = `
    <div class="renderHead">
      <div>
        <div class="renderTitle">3D Reconstruction</div>
        <div class="renderMeta">${recon ? `${recon.points.length} points | ${recon.edges.length} distance edges | unit: ${escapeHtml(recon.coordinate_unit)}` : ""}</div>
      </div>
      <div class="toolbar">
        <button type="button" id="rerender3dBtn">Rerender</button>
        <button type="button" id="reset3dBtn">Reset View</button>
      </div>
    </div>
    <div class="renderBody">
      <canvas id="render3dCanvas" class="renderCanvas"></canvas>
      <div class="renderStats">
        ${warnings}
        <div class="smallText">Scale to mm: ${recon ? Number(recon.scale_to_mm).toPrecision(6) : ""}</div>
        <table class="renderStatsTable">
          <thead>
            <tr><th>ID</th><th>Points</th><th>Role</th><th>Target</th><th>+/-</th><th>3D</th><th>Residual</th><th>z</th></tr>
          </thead>
          <tbody>${edgeRows}</tbody>
        </table>
      </div>
    </div>
  `;
  els.imageGrid.appendChild(panel);
  if (!recon) return;
  panel.querySelector("#rerender3dBtn").addEventListener("click", requestReconstruction);
  panel.querySelector("#reset3dBtn").addEventListener("click", () => {
    state.renderView = { rotX: -0.5, rotY: 0.75, zoom: 1, panX: 0, panY: 0 };
    draw3DScene();
  });
  bindRenderCanvas(panel.querySelector("#render3dCanvas"));
}

function renderImages() {
  els.imageGrid.innerHTML = "";
  renderReconstructionPanel();
  for (const camera of state.cameras) {
    ensureView(camera);
    const panel = document.createElement("article");
    panel.className = "imagePanel";
    panel.innerHTML = `
      <div class="imageHead">
        <div>
          <div class="cameraName">Camera ${escapeHtml(camera.name)}</div>
          <div class="cameraMeta">${camera.width}x${camera.height} -> ${camera.source_width}x${camera.source_height}</div>
        </div>
        <div class="imageControls">
          <div class="zoomControls">
            <button type="button" data-zoom="out" title="Zoom out">-</button>
            <button type="button" data-zoom="in" title="Zoom in">+</button>
            <button type="button" data-zoom="reset" title="Reset zoom">1:1</button>
            <span class="zoomReadout" data-camera="${escapeHtml(camera.name)}">100%</span>
          </div>
          <label class="contrastControl" title="Contrast">
            <span>C</span>
            <input type="range" min="0.4" max="3" step="0.05" value="1" data-contrast-camera="${escapeHtml(camera.name)}">
            <span class="contrastReadout" data-camera="${escapeHtml(camera.name)}">100%</span>
          </label>
        </div>
      </div>
      <div class="imageStage" data-camera="${escapeHtml(camera.name)}" style="--image-aspect: ${camera.width} / ${camera.height}">
        <div class="imageLayer">
          <img class="cameraImage" alt="Camera ${escapeHtml(camera.name)}" src="${cameraImageUrl(camera)}">
          <svg class="overlay" viewBox="0 0 ${camera.width} ${camera.height}" preserveAspectRatio="none"></svg>
        </div>
      </div>
    `;
    const stage = panel.querySelector(".imageStage");
    bindStagePointer(stage, camera);
    panel.querySelector('[data-zoom="out"]').addEventListener("click", () => {
      const rect = stage.getBoundingClientRect();
      zoomCameraAt(camera, rect.left + rect.width / 2, rect.top + rect.height / 2, 1 / 1.35);
    });
    panel.querySelector('[data-zoom="in"]').addEventListener("click", () => {
      const rect = stage.getBoundingClientRect();
      zoomCameraAt(camera, rect.left + rect.width / 2, rect.top + rect.height / 2, 1.35);
    });
    panel.querySelector('[data-zoom="reset"]').addEventListener("click", () => resetCameraView(camera));
    panel.querySelector(`[data-contrast-camera="${cssEscape(camera.name)}"]`).addEventListener("input", (event) => {
      const view = ensureView(camera);
      view.contrast = clamp(Number(event.target.value), 0.4, 3);
      applyView(camera);
    });
    els.imageGrid.appendChild(panel);
    applyView(camera);
  }
  updateDisplayFrameUi(false);
  renderMarkers();
  draw3DScene();
}

function bindRenderCanvas(canvas) {
  if (!canvas) return;
  let drag = null;
  canvas.addEventListener("pointerdown", (event) => {
    drag = {
      pointerId: event.pointerId,
      x: event.clientX,
      y: event.clientY,
      rotX: state.renderView.rotX,
      rotY: state.renderView.rotY,
    };
    canvas.classList.add("dragging");
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!drag || drag.pointerId !== event.pointerId) return;
    const dx = event.clientX - drag.x;
    const dy = event.clientY - drag.y;
    state.renderView.rotY = drag.rotY + dx * 0.01;
    state.renderView.rotX = clamp(drag.rotX + dy * 0.01, -Math.PI / 2, Math.PI / 2);
    draw3DScene();
  });
  function finish(event) {
    if (!drag || drag.pointerId !== event.pointerId) return;
    canvas.releasePointerCapture(event.pointerId);
    canvas.classList.remove("dragging");
    drag = null;
  }
  canvas.addEventListener("pointerup", finish);
  canvas.addEventListener("pointercancel", finish);
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    state.renderView.zoom = clamp(state.renderView.zoom * (event.deltaY < 0 ? 1.15 : 1 / 1.15), 0.2, 8);
    draw3DScene();
  }, { passive: false });
}

function rotate3D(xyz) {
  const [x, y, z] = xyz;
  const cy = Math.cos(state.renderView.rotY);
  const sy = Math.sin(state.renderView.rotY);
  const cx = Math.cos(state.renderView.rotX);
  const sx = Math.sin(state.renderView.rotX);
  const x1 = x * cy + z * sy;
  const z1 = -x * sy + z * cy;
  const y1 = y * cx - z1 * sx;
  const z2 = y * sx + z1 * cx;
  return [x1, y1, z2];
}

function draw3DScene() {
  const recon = state.reconstruction;
  const canvas = document.getElementById("render3dCanvas");
  if (!recon || !canvas) return;
  const rect = canvas.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const width = rect.width;
  const height = rect.height;
  ctx.clearRect(0, 0, width, height);

  const points = recon.points || [];
  if (!points.length) {
    ctx.fillStyle = "#a8b0b8";
    ctx.fillText("No reconstructable points", 16, 24);
    return;
  }

  const maxAbs = Math.max(
    1e-6,
    ...points.flatMap((point) => point.xyz.map((value) => Math.abs(Number(value))))
  );
  const scale = Math.min(width, height) * 0.38 * state.renderView.zoom / maxAbs;
  const projected = new Map();
  for (const point of points) {
    const [rx, ry, rz] = rotate3D(point.xyz.map(Number));
    projected.set(point.id, {
      x: width / 2 + state.renderView.panX + rx * scale,
      y: height / 2 + state.renderView.panY - ry * scale,
      z: rz,
      point,
    });
  }

  ctx.lineWidth = 1;
  ctx.strokeStyle = "#333c46";
  ctx.beginPath();
  ctx.moveTo(16, height - 24);
  ctx.lineTo(width - 16, height - 24);
  ctx.stroke();
  ctx.fillStyle = "#a8b0b8";
  ctx.font = "12px system-ui, sans-serif";
  ctx.fillText(`unit: ${recon.coordinate_unit}`, 16, height - 32);

  const edgeDraws = (recon.edges || [])
    .map((edge) => ({ edge, a: projected.get(edge.point_a), b: projected.get(edge.point_b) }))
    .filter((item) => item.a && item.b)
    .sort((u, v) => ((u.a.z + u.b.z) - (v.a.z + v.b.z)));
  for (const item of edgeDraws) {
    const residual = Math.abs(Number(item.edge.normalized_residual || 0));
    ctx.strokeStyle = residual > 3 ? "#ff6b6b" : residual > 2 ? "#f2c14e" : "#59c28f";
    ctx.lineWidth = item.edge.use_for_refinement === false ? 1.5 : 2.5;
    ctx.setLineDash(item.edge.use_for_refinement === false ? [6, 4] : []);
    ctx.beginPath();
    ctx.moveTo(item.a.x, item.a.y);
    ctx.lineTo(item.b.x, item.b.y);
    ctx.stroke();
  }
  ctx.setLineDash([]);

  const pointDraws = [...projected.values()].sort((a, b) => a.z - b.z);
  for (const item of pointDraws) {
    ctx.beginPath();
    ctx.fillStyle = "#49a6ff";
    ctx.strokeStyle = "#050607";
    ctx.lineWidth = 2;
    ctx.arc(item.x, item.y, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#eceff1";
    ctx.font = "12px system-ui, sans-serif";
    ctx.fillText(item.point.label || item.point.id, item.x + 8, item.y - 8);
  }
}

function renderMarkers() {
  for (const camera of state.cameras) {
    const stage = stageForCamera(camera);
    if (!stage) continue;
    const svg = stage.querySelector(".overlay");
    svg.innerHTML = "";

    for (const point of state.points) {
      const obs = point.observations?.[camera.name];
      if (!obs) continue;
      const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
      group.setAttribute("data-point-id", point.id);
      group.style.cursor = "pointer";
      group.addEventListener("pointerdown", (event) => event.stopPropagation());
      group.addEventListener("pointerup", (event) => event.stopPropagation());
      group.addEventListener("click", (event) => {
        event.stopPropagation();
        selectPoint(point.id);
      });

      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("cx", obs.x);
      circle.setAttribute("cy", obs.y);
      circle.setAttribute("r", point.id === state.selectedId ? "7" : "5");
      circle.setAttribute("class", `markerCircle${point.id === state.selectedId ? " current" : ""}`);

      group.appendChild(circle);
      if (state.showAllLabels || point.id === state.selectedId) {
        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.setAttribute("x", obs.x + 8);
        label.setAttribute("y", obs.y - 8);
        label.setAttribute("class", "markerLabel");
        label.textContent = point.label || point.id;
        group.appendChild(label);
      }
      svg.appendChild(group);
    }
  }
}

function renderPointList() {
  els.pointCount.textContent = String(state.points.length);
  els.pointList.innerHTML = "";
  for (const point of state.points) {
    const row = document.createElement("div");
    row.className = `pointRow${point.id === state.selectedId ? " active" : ""}`;
    row.addEventListener("click", () => selectPoint(point.id));

    const main = document.createElement("div");
    main.className = "pointMain";
    const count = observationCount(point);
    main.innerHTML = `
      <div class="pointId">${escapeHtml(point.id)}</div>
      <div class="pointSub">${escapeHtml(point.label || point.id)} - ${count} camera${count === 1 ? "" : "s"}</div>
    `;

    const dots = document.createElement("div");
    dots.className = "camDots";
    for (const camera of state.cameras) {
      const dot = document.createElement("span");
      dot.className = `camDot${point.observations?.[camera.name] ? " on" : ""}`;
      dot.textContent = camera.name;
      dots.appendChild(dot);
    }

    row.appendChild(main);
    row.appendChild(dots);
    els.pointList.appendChild(row);
  }
}

function renderDistanceOptions() {
  const currentA = els.distancePointA.value;
  const currentB = els.distancePointB.value;
  const optionHtml = state.points
    .map((point) => `<option value="${escapeHtml(point.id)}">${escapeHtml(point.id)} - ${escapeHtml(point.label || point.id)}</option>`)
    .join("");
  els.distancePointA.innerHTML = optionHtml;
  els.distancePointB.innerHTML = optionHtml;

  if (state.points.some((point) => point.id === currentA)) {
    els.distancePointA.value = currentA;
  } else if (state.points[0]) {
    els.distancePointA.value = state.points[0].id;
  }

  if (state.points.some((point) => point.id === currentB)) {
    els.distancePointB.value = currentB;
  } else if (state.points[1]) {
    els.distancePointB.value = state.points[1].id;
  } else if (state.points[0]) {
    els.distancePointB.value = state.points[0].id;
  }

  const enabled = state.points.length >= 2;
  els.distancePointA.disabled = !enabled;
  els.distancePointB.disabled = !enabled;
  els.distanceValueInput.disabled = !enabled;
  els.distanceUnitSelect.disabled = !enabled;
  els.distanceSigmaInput.disabled = !enabled;
  els.distanceSigmaUnitSelect.disabled = !enabled;
  els.distanceUseForRefinementInput.disabled = !enabled;
  els.distanceNotesInput.disabled = !enabled;
  els.saveDistanceBtn.disabled = !enabled;
}

function renderDistances() {
  renderDistanceOptions();
  els.distanceCount.textContent = String(state.distances.length);
  els.distanceList.innerHTML = "";
  for (const distance of state.distances) {
    const row = document.createElement("div");
    row.className = `distanceRow${distance.id === state.selectedDistanceId ? " active" : ""}`;
    row.addEventListener("click", () => populateDistanceForm(distance));
    const mm = distanceMm(distance);
    const sigma = distanceSigmaMm(distance);
    const role = distance.use_for_refinement ? "fit" : "holdout";
    row.innerHTML = `
      <div>
        <div class="distanceTitle">${escapeHtml(distance.id)}: ${escapeHtml(distance.point_a)} to ${escapeHtml(distance.point_b)}</div>
        <div class="distanceSub">${escapeHtml(pointLabelById(distance.point_a))} - ${escapeHtml(pointLabelById(distance.point_b))} | ${Number(distance.value).toFixed(3)} ${escapeHtml(distance.unit)} (${Number.isFinite(mm) ? mm.toFixed(3) : "?"} mm) +/- ${Number.isFinite(sigma) ? sigma.toFixed(3) : "?"} mm | ${role}</div>
      </div>
      <button type="button" class="distanceDelete" title="Delete distance">X</button>
    `;
    row.querySelector(".distanceDelete").addEventListener("click", (event) => {
      event.stopPropagation();
      deleteDistance(distance.id);
    });
    els.distanceList.appendChild(row);
  }
  if (!selectedDistance()) {
    els.saveDistanceBtn.textContent = "Add Distance";
  }
}

function renderSelected() {
  const point = selectedPoint();
  els.selectedStatus.textContent = point ? `${observationCount(point)} obs` : "None";
  els.pointIdInput.value = point?.id || "";
  els.pointLabelInput.value = point?.label || "";
  els.pointNotesInput.value = point?.notes || "";
  els.pointSigmaInput.value = point ? String(point.sigma_px || DEFAULT_REPROJECTION_SIGMA_PX) : "";
  els.pointIdInput.disabled = !point;
  els.pointLabelInput.disabled = !point;
  els.pointNotesInput.disabled = !point;
  els.pointSigmaInput.disabled = !point;

  els.obsEditor.innerHTML = "";
  if (!point) return;

  for (const camera of state.cameras) {
    const obs = point.observations?.[camera.name];
    const row = document.createElement("div");
    row.className = "obsRow";
    row.innerHTML = `
      <div class="camDot ${obs ? "on" : ""}">${escapeHtml(camera.name)}</div>
      <input type="number" step="0.01" min="0" max="${camera.width - 1}" value="${obs ? Number(obs.x).toFixed(2) : ""}" placeholder="x">
      <input type="number" step="0.01" min="0" max="${camera.height - 1}" value="${obs ? Number(obs.y).toFixed(2) : ""}" placeholder="y">
      <button title="Remove this camera observation">X</button>
    `;
    const [xInput, yInput] = row.querySelectorAll("input");
    function updateObs() {
      const x = Number(xInput.value);
      const y = Number(yInput.value);
      if (!Number.isFinite(x) || !Number.isFinite(y)) return;
      point.observations[camera.name] = {
        x: clamp(x, 0, camera.width - 1),
        y: clamp(y, 0, camera.height - 1),
      };
      setDirty(true);
      renderAll();
    }
    xInput.addEventListener("change", updateObs);
    yInput.addEventListener("change", updateObs);
    row.querySelector("button").addEventListener("click", () => removeObservation(camera.name));
    els.obsEditor.appendChild(row);

    if (obs) {
      const mapped = sourceXY(camera, obs);
      const small = document.createElement("div");
      small.className = "smallText";
      small.textContent = `${camera.name} source: ${mapped.x.toFixed(1)}, ${mapped.y.toFixed(1)}`;
      els.obsEditor.appendChild(small);
    }
  }
}

function renderWarnings(extra = []) {
  const warnings = [];
  const pointIds = new Set(state.points.map((point) => point.id));
  for (const point of state.points) {
    if (observationCount(point) < 2) {
      warnings.push(`${point.id}: fewer than 2 cameras`);
    }
    if (!Number.isFinite(Number(point.sigma_px)) || Number(point.sigma_px) <= 0) {
      warnings.push(`${point.id}: invalid click uncertainty`);
    }
  }
  for (const distance of state.distances) {
    if (!pointIds.has(distance.point_a) || !pointIds.has(distance.point_b)) {
      warnings.push(`${distance.id}: references a missing point`);
    } else if (distance.point_a === distance.point_b) {
      warnings.push(`${distance.id}: uses the same point twice`);
    } else if (!Number.isFinite(Number(distance.value)) || Number(distance.value) <= 0) {
      warnings.push(`${distance.id}: invalid distance`);
    } else if (!Number.isFinite(Number(distance.sigma_value)) || Number(distance.sigma_value) <= 0) {
      warnings.push(`${distance.id}: invalid distance uncertainty`);
    }
  }
  warnings.push(...extra);
  const uniqueWarnings = [...new Set(warnings)];
  els.warnings.innerHTML = uniqueWarnings.map((w) => `<div>${escapeHtml(w)}</div>`).join("");
}

function renderAll() {
  renderPointList();
  renderDistances();
  renderSelected();
  renderMarkers();
  renderWarnings();
}

function bindInputs() {
  els.newPointBtn.addEventListener("click", createPoint);
  els.deletePointBtn.addEventListener("click", deleteSelectedPoint);
  els.labelsToggleBtn.addEventListener("click", () => {
    state.showAllLabels = !state.showAllLabels;
    els.labelsToggleBtn.classList.toggle("active", state.showAllLabels);
    els.labelsToggleBtn.title = state.showAllLabels
      ? "Show only the selected point label"
      : "Show labels for all points";
    renderMarkers();
  });
  els.render3dBtn.addEventListener("click", requestReconstruction);
  els.saveDistanceBtn.addEventListener("click", saveDistanceFromForm);
  els.clearDistanceBtn.addEventListener("click", clearDistanceForm);
  els.saveBtn.addEventListener("click", savePoints);
  els.frameSlider.addEventListener("input", () => setDisplayFrame(Number(els.frameSlider.value)));
  els.framePrevBtn.addEventListener("click", () => setDisplayFrame(state.displayFrameIndex - 1));
  els.frameNextBtn.addEventListener("click", () => setDisplayFrame(state.displayFrameIndex + 1));

  els.pointIdInput.addEventListener("change", () => {
    const point = selectedPoint();
    if (!point) return;
    const nextId = els.pointIdInput.value.trim();
    if (!nextId || state.points.some((other) => other !== point && other.id === nextId)) {
      els.pointIdInput.value = point.id;
      return;
    }
    const oldId = point.id;
    point.id = nextId;
    for (const distance of state.distances) {
      if (distance.point_a === oldId) distance.point_a = nextId;
      if (distance.point_b === oldId) distance.point_b = nextId;
    }
    state.selectedId = nextId;
    if (!point.label) point.label = nextId;
    setDirty(true);
    renderAll();
  });

  els.pointLabelInput.addEventListener("input", () => {
    const point = selectedPoint();
    if (!point) return;
    point.label = els.pointLabelInput.value;
    setDirty(true);
    renderPointList();
    renderMarkers();
  });

  els.pointNotesInput.addEventListener("input", () => {
    const point = selectedPoint();
    if (!point) return;
    point.notes = els.pointNotesInput.value;
    setDirty(true);
  });

  els.pointSigmaInput.addEventListener("change", () => {
    const point = selectedPoint();
    if (!point) return;
    const sigmaPx = Number(els.pointSigmaInput.value);
    if (!Number.isFinite(sigmaPx) || sigmaPx <= 0) {
      els.pointSigmaInput.value = String(point.sigma_px || DEFAULT_REPROJECTION_SIGMA_PX);
      return;
    }
    point.sigma_px = sigmaPx;
    setDirty(true);
    renderWarnings();
  });

  window.addEventListener("beforeunload", (event) => {
    if (!state.dirty) return;
    event.preventDefault();
    event.returnValue = "";
  });

  document.addEventListener("keydown", (event) => {
    const tagName = event.target?.tagName?.toLowerCase();
    const editingText = tagName === "input" || tagName === "textarea" || tagName === "select";
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
      event.preventDefault();
      savePoints();
    }
    if (!editingText && event.key === "[") {
      event.preventDefault();
      setDisplayFrame(state.displayFrameIndex - 1);
    }
    if (!editingText && event.key === "]") {
      event.preventDefault();
      setDisplayFrame(state.displayFrameIndex + 1);
    }
  });

  window.addEventListener("resize", () => {
    for (const camera of state.cameras) {
      applyView(camera);
    }
    draw3DScene();
  });
}

async function savePoints() {
  const response = await fetch("/api/points", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      points: state.points,
      distances: state.distances,
      display_frame_index: state.displayFrameIndex,
    }),
  });
  const result = await response.json();
  if (!response.ok || !result.ok) {
    renderWarnings([result.error || "Save failed"]);
    return;
  }
  if (result.output_path) {
    els.outputPath.textContent = result.output_path;
  }
  setDirty(false);
  renderWarnings(result.warnings || []);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function cssEscape(value) {
  if (window.CSS && CSS.escape) return CSS.escape(value);
  return String(value).replace(/[^a-zA-Z0-9_-]/g, "\\$&");
}

async function init() {
  bindInputs();
  const response = await fetch("/api/frame");
  state.frame = await response.json();
  state.cameras = state.frame.cameras;
  state.displayFrames = Array.isArray(state.frame.display_frames) ? state.frame.display_frames : [];
  state.displayFrameIndex = Number.isFinite(Number(state.frame.display_frame_index)) ? Number(state.frame.display_frame_index) : 0;
  els.framePath.textContent = state.frame.frame_dir;
  els.outputPath.textContent = state.frame.output_path;
  updateDisplayFrameUi(false);
  loadExisting(state.frame.existing_points);
  state.selectedId = state.points[0]?.id || null;
  setDirty(false);
  renderImages();
  renderAll();
}

init().catch((err) => {
  document.body.innerHTML = `<pre style="padding:16px;color:#ff6b6b">${escapeHtml(err.stack || err.message || err)}</pre>`;
});
"""


def _frame_name_from_arg(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith(".json"):
        text = Path(text).stem
    if re.fullmatch(r"\d+", text):
        return f"frame_{int(text):06d}"
    match = re.fullmatch(r"frame_(\d+)", text)
    if match:
        return f"frame_{int(match.group(1)):06d}"
    raise ValueError(f"Invalid frame {value!r}; use a number like 276 or a name like frame_000276")


def _dataset_frames(dataset_dir: Path) -> list[Path]:
    frames = [
        path
        for path in dataset_dir.glob("frame_*")
        if path.is_dir() and (path / "images.npy").is_file() and PointSelectorState._frame_number(path) is not None
    ]
    return sorted(frames, key=lambda path: (PointSelectorState._frame_number(path) or -1, path.name))


def resolve_selector_input(input_path: Path, frame_arg: str | None, output_path: Path | None) -> tuple[Path, Path]:
    input_path = input_path.expanduser().resolve()
    requested_frame = _frame_name_from_arg(frame_arg)
    if (input_path / "images.npy").is_file():
        frame_dir = input_path
        dataset_dir = input_path.parent if PointSelectorState._frame_number(input_path) is not None else input_path
        if requested_frame is not None and requested_frame != frame_dir.name:
            candidate = dataset_dir / requested_frame
            if not (candidate / "images.npy").is_file():
                raise FileNotFoundError(f"Requested frame not found: {candidate}")
            frame_dir = candidate
        return dataset_dir, frame_dir

    dataset_dir = input_path
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    frames = _dataset_frames(dataset_dir)
    if not frames:
        raise FileNotFoundError(f"No frame_*/images.npy directories found in {dataset_dir}")

    if requested_frame is None and output_path is not None:
        requested_frame = _frame_name_from_arg(output_path.name)
    if requested_frame is not None:
        frame_dir = dataset_dir / requested_frame
        if not (frame_dir / "images.npy").is_file():
            raise FileNotFoundError(f"Requested frame not found: {frame_dir}")
        return dataset_dir, frame_dir

    return dataset_dir, frames[0]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Video dataset directory containing frame_* folders. A frame_* directory is also accepted.",
    )
    parser.add_argument("--frame", help="Initial frame to display when a dataset directory is passed, e.g. 276 or frame_000276.")
    parser.add_argument("--output", type=Path, help=f"Explicit output JSON. Default: DATASET_DIR/{DEFAULT_OUTPUT_NAME}.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser automatically.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = args.output.expanduser().resolve() if args.output else None
    dataset_dir, frame_dir = resolve_selector_input(args.dataset_dir, args.frame, output_path)
    state = PointSelectorState(dataset_dir, frame_dir, output_path)

    handler = make_handler(state)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{server.server_port}/"
    print(f"Serving VGGT point selector at {url}")
    print(f"Dataset: {dataset_dir}")
    print(f"Frame:   {frame_dir}")
    print(f"Output:  {output_path if output_path is not None else dataset_dir / DEFAULT_OUTPUT_NAME}")
    if not args.no_browser:
        time.sleep(0.2)
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
