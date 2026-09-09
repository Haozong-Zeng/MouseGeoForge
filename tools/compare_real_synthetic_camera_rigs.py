#!/usr/bin/env python3
"""Compare normalized A/B/C camera geometry between real and Blender datasets."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


CAMERAS = ("A", "B", "C")
SYNTHETIC_CAMERA_ALIASES = {
    "CameraA": "A",
    "CameraB": "B",
    "CameraC": "C",
    "Camera.003": "A",
    "Camera.001": "B",
    "Camera.002": "C",
}


@dataclass
class Rig:
    name: str
    extrinsics: dict[str, np.ndarray]
    intrinsics: dict[str, np.ndarray]
    sizes: dict[str, tuple[int, int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_cam(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    ext_start = lines.index("extrinsic") + 1
    int_start = lines.index("intrinsic") + 1
    extrinsic = np.asarray(
        [[float(value) for value in lines[ext_start + row].split()] for row in range(4)],
        dtype=np.float64,
    )
    intrinsic = np.asarray(
        [[float(value) for value in lines[int_start + row].split()] for row in range(3)],
        dtype=np.float64,
    )
    return extrinsic, intrinsic


def load_real_rig(root: Path) -> Rig:
    data = tomllib.loads((root / "calibration" / "calibration.toml").read_text())
    extrinsics: dict[str, np.ndarray] = {}
    intrinsics: dict[str, np.ndarray] = {}
    sizes: dict[str, tuple[int, int]] = {}
    index = 0
    while f"cam_{index}" in data:
        item = data[f"cam_{index}"]
        name = str(item["name"])
        if name in CAMERAS:
            rotation = cv2.Rodrigues(np.asarray(item["rotation"], dtype=np.float64))[0]
            extrinsic = np.eye(4, dtype=np.float64)
            extrinsic[:3, :3] = rotation
            extrinsic[:3, 3] = np.asarray(item["translation"], dtype=np.float64)
            extrinsics[name] = extrinsic
            intrinsics[name] = np.asarray(item["matrix"], dtype=np.float64)
            sizes[name] = tuple(int(value) for value in item["size"])
        index += 1
    if set(extrinsics) != set(CAMERAS):
        raise ValueError(f"Missing A/B/C calibration in {root}")
    return Rig(root.name, extrinsics, intrinsics, sizes)


def load_synthetic_rig(dataset: Path) -> tuple[Rig, dict[str, float]]:
    scenes = sorted(dataset.glob("frame_*"))
    if not scenes:
        raise ValueError(f"No frame scenes under {dataset}")
    scene = scenes[0]
    extrinsics: dict[str, np.ndarray] = {}
    intrinsics: dict[str, np.ndarray] = {}
    sizes: dict[str, tuple[int, int]] = {}
    view_for_camera: dict[str, int] = {}
    for annotation_path in sorted((scene / "annotations").glob("*.json")):
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        alias = SYNTHETIC_CAMERA_ALIASES.get(str(annotation["camera"]))
        if alias not in CAMERAS:
            continue
        view_id = int(annotation["view_id"])
        extrinsic, intrinsic = read_cam(scene / "cams" / f"{view_id:08d}_cam.txt")
        image_path = scene / "blended_images" / f"{view_id:08d}.jpg"
        with Image.open(image_path) as image:
            sizes[alias] = image.size
        extrinsics[alias] = extrinsic
        intrinsics[alias] = intrinsic
        view_for_camera[alias] = view_id
    if set(extrinsics) != set(CAMERAS):
        raise ValueError(f"Could not map A/B/C cameras in {dataset}")

    max_rotation_deg = 0.0
    max_center_fraction = 0.0
    reference_centers = camera_centers(extrinsics)
    reference_scale = median_baseline(reference_centers)
    for later_scene in scenes[1:]:
        for camera in CAMERAS:
            view_id = view_for_camera[camera]
            current, _ = read_cam(later_scene / "cams" / f"{view_id:08d}_cam.txt")
            max_rotation_deg = max(
                max_rotation_deg,
                rotation_distance_deg(extrinsics[camera][:3, :3], current[:3, :3]),
            )
            current_center = -current[:3, :3].T @ current[:3, 3]
            max_center_fraction = max(
                max_center_fraction,
                float(np.linalg.norm(current_center - reference_centers[camera]) / reference_scale),
            )
    stability = {
        "max_rotation_change_deg": max_rotation_deg,
        "max_center_change_over_median_baseline": max_center_fraction,
    }
    return Rig(dataset.name, extrinsics, intrinsics, sizes), stability


def camera_centers(extrinsics: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        camera: -extrinsics[camera][:3, :3].T @ extrinsics[camera][:3, 3]
        for camera in CAMERAS
    }


def median_baseline(centers: dict[str, np.ndarray]) -> float:
    distances = [
        np.linalg.norm(centers[first] - centers[second])
        for first, second in itertools.combinations(CAMERAS, 2)
    ]
    return max(float(np.median(distances)), 1e-12)


def canonical_centers(rig: Rig) -> dict[str, np.ndarray]:
    reference = rig.extrinsics["A"]
    centers = camera_centers(rig.extrinsics)
    rotation = reference[:3, :3]
    origin = centers["A"]
    canonical = {
        camera: rotation @ (centers[camera] - origin)
        for camera in CAMERAS
    }
    scale = median_baseline(canonical)
    return {camera: center / scale for camera, center in canonical.items()}


def relative_rotations(rig: Rig) -> dict[str, np.ndarray]:
    world_from_a = np.linalg.inv(rig.extrinsics["A"])
    return {
        camera: (rig.extrinsics[camera] @ world_from_a)[:3, :3]
        for camera in CAMERAS
    }


def normalized_intrinsics(rig: Rig) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for camera in CAMERAS:
        width, height = rig.sizes[camera]
        intrinsic = rig.intrinsics[camera]
        output[camera] = np.asarray(
            [
                intrinsic[0, 0] / width,
                intrinsic[1, 1] / height,
                intrinsic[0, 2] / width,
                intrinsic[1, 2] / height,
                math.degrees(2.0 * math.atan(width / (2.0 * intrinsic[0, 0]))),
                math.degrees(2.0 * math.atan(height / (2.0 * intrinsic[1, 1]))),
            ],
            dtype=np.float64,
        )
    return output


def baseline_ratios(rig: Rig) -> dict[str, float]:
    centers = camera_centers(rig.extrinsics)
    scale = median_baseline(centers)
    return {
        first + second: float(np.linalg.norm(centers[first] - centers[second]) / scale)
        for first, second in itertools.combinations(CAMERAS, 2)
    }


def rotation_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first @ second.T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def compare_rigs(first: Rig, second: Rig) -> dict[str, float]:
    first_centers = canonical_centers(first)
    second_centers = canonical_centers(second)
    center_rmse = math.sqrt(
        np.mean(
            [
                np.sum((first_centers[camera] - second_centers[camera]) ** 2)
                for camera in CAMERAS
            ]
        )
    )
    first_rotations = relative_rotations(first)
    second_rotations = relative_rotations(second)
    rotation_mean = float(
        np.mean(
            [
                rotation_distance_deg(first_rotations[camera], second_rotations[camera])
                for camera in ("B", "C")
            ]
        )
    )
    first_baselines = baseline_ratios(first)
    second_baselines = baseline_ratios(second)
    baseline_mape = 100.0 * float(
        np.mean(
            [
                abs(first_baselines[pair] - second_baselines[pair])
                / max(second_baselines[pair], 1e-12)
                for pair in sorted(first_baselines)
            ]
        )
    )
    first_intrinsics = normalized_intrinsics(first)
    second_intrinsics = normalized_intrinsics(second)
    focal_mape = 100.0 * float(
        np.mean(
            [
                abs(first_intrinsics[camera][axis] - second_intrinsics[camera][axis])
                / max(abs(second_intrinsics[camera][axis]), 1e-12)
                for camera in CAMERAS
                for axis in (0, 1)
            ]
        )
    )
    principal_distance = float(
        np.mean(
            [
                np.linalg.norm(
                    first_intrinsics[camera][2:4] - second_intrinsics[camera][2:4]
                )
                for camera in CAMERAS
            ]
        )
    )
    return {
        "center_rmse_normalized": center_rmse,
        "relative_rotation_mean_deg": rotation_mean,
        "baseline_ratio_mape_percent": baseline_mape,
        "normalized_focal_mape_percent": focal_mape,
        "principal_point_distance_normalized": principal_distance,
    }


def rig_summary(rig: Rig) -> dict:
    intrinsics = normalized_intrinsics(rig)
    centers = canonical_centers(rig)
    rotations = relative_rotations(rig)
    return {
        "name": rig.name,
        "baseline_ratios": baseline_ratios(rig),
        "canonical_centers": {
            camera: centers[camera].tolist() for camera in CAMERAS
        },
        "relative_rotation_deg": {
            camera: rotation_distance_deg(rotations[camera], np.eye(3))
            for camera in ("B", "C")
        },
        "normalized_intrinsics": {
            camera: {
                "fx_over_width": float(intrinsics[camera][0]),
                "fy_over_height": float(intrinsics[camera][1]),
                "cx_over_width": float(intrinsics[camera][2]),
                "cy_over_height": float(intrinsics[camera][3]),
                "horizontal_fov_deg": float(intrinsics[camera][4]),
                "vertical_fov_deg": float(intrinsics[camera][5]),
            }
            for camera in CAMERAS
        },
    }


def main() -> None:
    args = parse_args()
    real_rigs = [load_real_rig(root.resolve()) for root in args.real_roots]
    synthetic_entries = []
    for dataset in sorted(args.synthetic_root.resolve().iterdir()):
        if not dataset.is_dir() or not (dataset / "metadata.json").is_file():
            continue
        rig, stability = load_synthetic_rig(dataset)
        synthetic_entries.append((rig, stability))

    rows = []
    synthetic_reports = []
    for synthetic, stability in synthetic_entries:
        comparisons = []
        for real in real_rigs:
            metrics = compare_rigs(synthetic, real)
            rows.append(
                {
                    "synthetic_dataset": synthetic.name,
                    "real_trial": real.name,
                    **metrics,
                }
            )
            comparisons.append(metrics)
        synthetic_reports.append(
            {
                "dataset": synthetic.name,
                "stability": stability,
                "mean_vs_real": {
                    key: float(np.mean([item[key] for item in comparisons]))
                    for key in comparisons[0]
                },
                "rig": rig_summary(synthetic),
            }
        )

    real_pair_metrics = [
        compare_rigs(first, second)
        for first, second in itertools.combinations(real_rigs, 2)
    ]
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "camera_aliases": SYNTHETIC_CAMERA_ALIASES,
        "real_trial_count": len(real_rigs),
        "synthetic_dataset_count": len(synthetic_entries),
        "real_rig_variation_mean": {
            key: float(np.mean([item[key] for item in real_pair_metrics]))
            for key in real_pair_metrics[0]
        },
        "real_rigs": [rig_summary(rig) for rig in real_rigs],
        "synthetic_rigs": synthetic_reports,
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "camera_rig_comparison.json"
    csv_path = output_dir / "camera_rig_pairwise_metrics.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
