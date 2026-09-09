#!/usr/bin/env python3
"""Fit one robust full-video VGGT-to-Anipose Sim(3) and compose it with Blender."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vggt-pose", type=Path, required=True)
    parser.add_argument("--anipose-pose", type=Path, required=True)
    parser.add_argument("--blender-alignment", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--min-cameras", type=float, default=2.0)
    parser.add_argument("--mad-multiplier", type=float, default=3.5)
    return parser.parse_args()


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    covariance = dst_centered.T @ src_centered / len(src)
    u, singular_values, vt = np.linalg.svd(covariance)
    sign = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1, -1] = -1.0
    rotation = u @ sign @ vt
    variance = np.mean(np.sum(src_centered**2, axis=1))
    scale = float(np.trace(np.diag(singular_values) @ sign) / max(variance, 1e-12))
    translation = dst_mean - scale * rotation @ src_mean
    return scale, rotation, translation


def keypoint_names(columns: list[str]) -> list[str]:
    names = []
    for column in columns:
        if not column.endswith("_x"):
            continue
        name = column[:-2]
        if name.startswith(("M_", "center_")):
            continue
        if all(f"{name}_{axis}" in columns for axis in "xyz"):
            names.append(name)
    return names


def collect_pairs(
    vggt: pd.DataFrame,
    anipose: pd.DataFrame,
    score_threshold: float,
    min_cameras: float,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, str]]]:
    common_frames = min(len(vggt), len(anipose))
    names = sorted(set(keypoint_names(list(vggt.columns))) & set(keypoint_names(list(anipose.columns))))
    src, dst, labels = [], [], []
    for frame in range(common_frames):
        for name in names:
            source = vggt.loc[frame, [f"{name}_{axis}" for axis in "xyz"]].to_numpy(float)
            target = anipose.loc[frame, [f"{name}_{axis}" for axis in "xyz"]].to_numpy(float)
            if not np.isfinite(source).all() or not np.isfinite(target).all():
                continue
            score_values = []
            camera_values = []
            for table in (vggt, anipose):
                score_column = f"{name}_score"
                camera_column = f"{name}_ncams"
                if score_column in table:
                    score_values.append(float(table.loc[frame, score_column]))
                if camera_column in table:
                    camera_values.append(float(table.loc[frame, camera_column]))
            if score_values and (not np.isfinite(score_values).all() or min(score_values) < score_threshold):
                continue
            if camera_values and (not np.isfinite(camera_values).all() or min(camera_values) < min_cameras):
                continue
            src.append(source)
            dst.append(target)
            labels.append((frame, name))
    if len(src) < 3:
        raise ValueError("Not enough valid shared 3D observations for Sim(3)")
    return np.asarray(src), np.asarray(dst), labels


def robust_similarity(
    src: np.ndarray,
    dst: np.ndarray,
    mad_multiplier: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask = np.ones(len(src), dtype=bool)
    for _ in range(10):
        scale, rotation, translation = umeyama(src[mask], dst[mask])
        residual = np.linalg.norm(scale * (src @ rotation.T) + translation - dst, axis=1)
        median = float(np.median(residual[mask]))
        mad = float(np.median(np.abs(residual[mask] - median)))
        threshold = median + mad_multiplier * max(1.4826 * mad, 1e-6)
        next_mask = residual <= threshold
        if next_mask.sum() < 3 or np.array_equal(next_mask, mask):
            break
        mask = next_mask
    scale, rotation, translation = umeyama(src[mask], dst[mask])
    residual = np.linalg.norm(scale * (src @ rotation.T) + translation - dst, axis=1)
    return scale, rotation, translation, mask, residual


def restore_calibration_world(anipose_pose: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    coordinate_rotation = np.asarray(
        [
            [float(anipose_pose[f"M_{row}{col}"].mean()) for col in range(3)]
            for row in range(3)
        ],
        dtype=np.float64,
    )
    center_columns = (
        [f"center_{axis}" for axis in "xyz"]
        if all(f"center_{axis}" in anipose_pose for axis in "xyz")
        else [f"center_{axis}" for axis in range(3)]
    )
    coordinate_center = np.asarray(
        [float(anipose_pose[column].mean()) for column in center_columns],
        dtype=np.float64,
    )
    adjusted_to_calibration_rotation = np.linalg.inv(coordinate_rotation)
    restored = anipose_pose.copy()
    for name in keypoint_names(list(anipose_pose.columns)):
        columns = [f"{name}_{axis}" for axis in "xyz"]
        points = anipose_pose[columns].to_numpy(dtype=np.float64)
        valid = np.isfinite(points).all(axis=1)
        points[valid] = (
            adjusted_to_calibration_rotation @ (points[valid] + coordinate_center).T
        ).T
        restored.loc[:, columns] = points
    return restored, {
        "formula": "X_calibration = inv(M) @ (X_anipose_adjusted + center)",
        "M": coordinate_rotation.tolist(),
        "center": coordinate_center.tolist(),
        "determinant_M": float(np.linalg.det(coordinate_rotation)),
        "orthogonality_error_M": float(
            np.linalg.norm(coordinate_rotation @ coordinate_rotation.T - np.eye(3))
        ),
    }


def calibration_extrinsics(path: Path) -> dict[str, list[list[float]]]:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    result = {}
    for value in raw.values():
        if not isinstance(value, dict) or "name" not in value:
            continue
        name = str(value["name"])
        if name not in {"A", "B", "C"}:
            continue
        rotation, _ = cv2.Rodrigues(np.asarray(value["rotation"], dtype=np.float64))
        translation = np.asarray(value["translation"], dtype=np.float64)
        result[name] = np.column_stack([rotation, translation]).tolist()
    if set(result) != {"A", "B", "C"}:
        raise KeyError(f"Calibration lacks A/B/C extrinsics: {path}")
    return result


def main() -> None:
    args = parse_args()
    vggt = pd.read_csv(args.vggt_pose)
    anipose_adjusted = pd.read_csv(args.anipose_pose)
    anipose, anipose_coordinate_frame = restore_calibration_world(anipose_adjusted)
    src, dst, labels = collect_pairs(
        vggt, anipose, args.score_threshold, args.min_cameras
    )
    scale_va, rotation_va, translation_va, inliers, residual = robust_similarity(
        src, dst, args.mad_multiplier
    )
    blender = json.loads(args.blender_alignment.read_text(encoding="utf-8"))
    scale_vb = float(blender["scale"])
    rotation_vb = np.asarray(blender["rotation"], dtype=np.float64)
    translation_vb = np.asarray(blender["translation"], dtype=np.float64)

    scale_ab = scale_vb / scale_va
    rotation_ab = rotation_vb @ rotation_va.T
    translation_ab = translation_vb - scale_ab * rotation_ab @ translation_va
    inlier_residual = residual[inliers]
    per_keypoint = {}
    labels_array = np.asarray(labels, dtype=object)
    for name in sorted(set(labels_array[:, 1])):
        values = residual[(labels_array[:, 1] == name) & inliers]
        if values.size:
            per_keypoint[str(name)] = {
                "count": int(values.size),
                "median": float(np.median(values)),
                "p95": float(np.quantile(values, 0.95)),
            }
    report = {
        "coordinate_formula": "X_calibration = scale * rotation @ X_vggt + translation",
        "vggt_to_calibration": {
            "scale": scale_va,
            "rotation": rotation_va.tolist(),
            "translation": translation_va.tolist(),
        },
        "source_to_blender": {
            "scale": scale_ab,
            "rotation": rotation_ab.tolist(),
            "translation": translation_ab.tolist(),
        },
        "calibration_extrinsics": calibration_extrinsics(args.calibration),
        "anipose_coordinate_frame": anipose_coordinate_frame,
        "fit": {
            "candidate_count": int(len(src)),
            "inlier_count": int(inliers.sum()),
            "inlier_fraction": float(inliers.mean()),
            "rmse_calibration_units": float(np.sqrt(np.mean(inlier_residual**2))),
            "median_calibration_units": float(np.median(inlier_residual)),
            "p95_calibration_units": float(np.quantile(inlier_residual, 0.95)),
            "mad_multiplier": args.mad_multiplier,
            "per_keypoint": per_keypoint,
        },
        "inputs": {
            "vggt_pose": str(args.vggt_pose.resolve()),
            "anipose_pose": str(args.anipose_pose.resolve()),
            "blender_alignment": str(args.blender_alignment.resolve()),
            "calibration": str(args.calibration.resolve()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("vggt_to_calibration", "source_to_blender", "fit")}, indent=2))


if __name__ == "__main__":
    main()
