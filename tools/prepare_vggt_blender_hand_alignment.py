#!/usr/bin/env python3
"""Align VGGT/DLC 3D keypoints to a Blender rig and derive hand descriptors."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


CV_FROM_BLENDER_CAMERA = np.diag([1.0, -1.0, -1.0])
SIDES = ("l", "r")
FINGERS = ("1", "2", "3", "4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rig-inspection", type=Path, required=True)
    parser.add_argument("--fixed-extrinsics", type=Path, required=True)
    parser.add_argument("--pose-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--min-cameras", type=float, default=2.0)
    return parser.parse_args()


def project_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def camera_center(extrinsic: np.ndarray) -> np.ndarray:
    return -extrinsic[:3, :3].T @ extrinsic[:3, 3]


def rotation_error_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    delta = rotation_a @ rotation_b.T
    cosine = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    u, singular_values, vt = np.linalg.svd(dst_centered.T @ src_centered / len(src))
    sign = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1, -1] = -1.0
    rotation = u @ sign @ vt
    variance = np.mean(np.sum(src_centered**2, axis=1))
    scale = float(np.trace(np.diag(singular_values) @ sign) / max(variance, 1e-12))
    translation = dst_mean - scale * rotation @ src_mean
    return scale, rotation, translation


def blender_camera_extrinsic(camera: dict) -> np.ndarray:
    world_from_camera = np.asarray(camera["matrix_world"], dtype=np.float64)
    rotation = CV_FROM_BLENDER_CAMERA @ world_from_camera[:3, :3].T
    center = world_from_camera[:3, 3]
    translation = -rotation @ center
    return np.column_stack([rotation, translation])


def load_fixed_extrinsics(path: Path) -> dict[str, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    fixed = data.get("fixed_relative_extrinsics")
    if fixed is None:
        fixed = data.get("camera_stabilization", {}).get("fixed_relative_extrinsics")
    if not fixed:
        raise KeyError(f"No fixed_relative_extrinsics in {path}")
    return {
        str(name): np.asarray(value, dtype=np.float64).reshape(3, 4)
        for name, value in fixed.items()
    }


def unique_blender_cameras(rig: dict) -> dict[str, dict]:
    cameras: dict[str, dict] = {}
    centers: list[np.ndarray] = []
    for camera in rig["cameras"]:
        if camera["name"].startswith("pasted__"):
            continue
        center = np.asarray(camera["location"], dtype=np.float64)
        if any(np.linalg.norm(center - existing) < 1e-3 for existing in centers):
            continue
        cameras[camera["name"]] = camera
        centers.append(center)
    return cameras


def mapping_score(
    source_names: list[str],
    source_extrinsics: dict[str, np.ndarray],
    target_names: tuple[str, ...],
    target_cameras: dict[str, dict],
) -> tuple[float, dict]:
    source_centers = np.stack([camera_center(source_extrinsics[name]) for name in source_names])
    target_centers = np.stack(
        [np.asarray(target_cameras[name]["location"], dtype=np.float64) for name in target_names]
    )
    scale, rotation, translation = umeyama(source_centers, target_centers)
    predicted_centers = (scale * (rotation @ source_centers.T)).T + translation
    center_rmse = float(
        np.sqrt(np.mean(np.sum((predicted_centers - target_centers) ** 2, axis=1)))
    )
    rotation_errors = []
    for source_name, target_name in zip(source_names, target_names):
        source_rotation = source_extrinsics[source_name][:3, :3]
        target_rotation = blender_camera_extrinsic(target_cameras[target_name])[:3, :3]
        predicted_rotation = source_rotation @ rotation.T
        rotation_errors.append(rotation_error_deg(predicted_rotation, target_rotation))
    baseline = np.mean(
        [
            np.linalg.norm(target_centers[i] - target_centers[j])
            for i, j in itertools.combinations(range(len(target_centers)), 2)
        ]
    )
    normalized_center = center_rmse / max(float(baseline), 1e-6)
    score = normalized_center + np.mean(rotation_errors) / 45.0
    return score, {
        "scale": scale,
        "rotation": rotation,
        "translation": translation,
        "center_rmse": center_rmse,
        "rotation_errors_deg": rotation_errors,
    }


def find_camera_mapping(
    source_extrinsics: dict[str, np.ndarray],
    target_cameras: dict[str, dict],
) -> tuple[dict[str, str], dict]:
    source_names = sorted(source_extrinsics)
    if len(source_names) != 3:
        raise ValueError(f"Expected three source cameras, found {source_names}")
    if len(target_cameras) < 3:
        raise ValueError("Need at least three unique Blender cameras")
    candidates = []
    for target_names in itertools.permutations(sorted(target_cameras), len(source_names)):
        score, diagnostics = mapping_score(
            source_names,
            source_extrinsics,
            target_names,
            target_cameras,
        )
        candidates.append((score, target_names, diagnostics))
    candidates.sort(key=lambda item: item[0])
    score, target_names, diagnostics = candidates[0]
    runner_up = candidates[1]
    diagnostics["mapping_score"] = score
    diagnostics["runner_up_score"] = runner_up[0]
    diagnostics["runner_up_mapping"] = dict(zip(source_names, runner_up[1]))
    return dict(zip(source_names, target_names)), diagnostics


def orientation_guided_similarity(
    mapping: dict[str, str],
    source_extrinsics: dict[str, np.ndarray],
    target_cameras: dict[str, dict],
) -> tuple[float, np.ndarray, np.ndarray, dict]:
    source_centers = []
    target_centers = []
    world_rotations = []
    for source_name, target_name in mapping.items():
        source = source_extrinsics[source_name]
        target = blender_camera_extrinsic(target_cameras[target_name])
        source_centers.append(camera_center(source))
        target_centers.append(camera_center(target))
        world_rotations.append(target[:3, :3].T @ source[:3, :3])

    source_centers_np = np.stack(source_centers)
    target_centers_np = np.stack(target_centers)
    rotation = project_rotation(np.mean(world_rotations, axis=0))
    source_centered_rotated = (
        rotation @ (source_centers_np - source_centers_np.mean(axis=0)).T
    ).T
    target_centered = target_centers_np - target_centers_np.mean(axis=0)
    scale = float(
        np.sum(source_centered_rotated * target_centered)
        / max(np.sum(source_centered_rotated**2), 1e-12)
    )
    translation = (
        target_centers_np.mean(axis=0) - scale * rotation @ source_centers_np.mean(axis=0)
    )
    predicted_centers = (scale * (rotation @ source_centers_np.T)).T + translation
    center_errors = np.linalg.norm(predicted_centers - target_centers_np, axis=1)
    rotation_errors = []
    for source_name, target_name in mapping.items():
        source_rotation = source_extrinsics[source_name][:3, :3]
        target_rotation = blender_camera_extrinsic(target_cameras[target_name])[:3, :3]
        rotation_errors.append(
            rotation_error_deg(source_rotation @ rotation.T, target_rotation)
        )
    diagnostics = {
        "center_errors": center_errors.tolist(),
        "center_rmse": float(np.sqrt(np.mean(center_errors**2))),
        "rotation_errors_deg": rotation_errors,
        "rotation_mean_deg": float(np.mean(rotation_errors)),
        "rotation_max_deg": float(np.max(rotation_errors)),
        "determinant": float(np.linalg.det(rotation)),
    }
    return scale, rotation, translation, diagnostics


def keypoint_names(columns: list[str]) -> list[str]:
    names = []
    column_set = set(columns)
    for column in columns:
        if not column.endswith("_x"):
            continue
        name = column[:-2]
        if f"{name}_y" in column_set and f"{name}_z" in column_set:
            names.append(name)
    return names


def transform_keypoints(
    source: pd.DataFrame,
    names: list[str],
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> pd.DataFrame:
    output = source.copy()
    for name in names:
        points = source[[f"{name}_x", f"{name}_y", f"{name}_z"]].to_numpy(
            dtype=np.float64
        )
        transformed = (scale * (rotation @ points.T)).T + translation
        output.loc[:, [f"{name}_x", f"{name}_y", f"{name}_z"]] = transformed
    return output


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-9:
        return np.full(3, np.nan)
    return vector / norm


def point(row: pd.Series, name: str) -> np.ndarray:
    return np.asarray([row[f"{name}_x"], row[f"{name}_y"], row[f"{name}_z"]], dtype=float)


def point_is_valid(
    row: pd.Series,
    name: str,
    score_threshold: float,
    min_cameras: float,
) -> bool:
    xyz = point(row, name)
    if not np.isfinite(xyz).all():
        return False
    score_col = f"{name}_score"
    ncams_col = f"{name}_ncams"
    return (
        (score_col not in row or float(row[score_col]) >= score_threshold)
        and (ncams_col not in row or float(row[ncams_col]) >= min_cameras)
    )


def hand_frame(row: pd.Series, side: str) -> tuple[np.ndarray, np.ndarray] | None:
    palm_center = point(row, f"{side}_mid_A")
    joints = np.stack([point(row, f"{side}{finger}_joint_A") for finger in FINGERS])
    across = normalize(joints[-1] - joints[0])
    forward = joints.mean(axis=0) - palm_center
    forward = normalize(forward - across * np.dot(forward, across))
    normal = normalize(np.cross(across, forward))
    if not np.isfinite(np.concatenate([palm_center, across, forward, normal])).all():
        return None
    return palm_center, np.column_stack([across, forward, normal])


def signed_angle(y: float, x: float) -> float:
    return float(np.degrees(np.arctan2(y, x)))


def build_hand_descriptors(
    aligned: pd.DataFrame,
    score_threshold: float,
    min_cameras: float,
) -> pd.DataFrame:
    rows = []
    previous_frames: dict[str, np.ndarray] = {}
    for frame_idx, row in aligned.iterrows():
        frame_number = int(row["fnum"]) if "fnum" in row else int(frame_idx)
        for side in SIDES:
            required = [f"{side}_mid_A"]
            for finger in FINGERS:
                required.extend(
                    [
                        f"{side}{finger}_joint_A",
                        f"{side}{finger}_mid_A",
                        f"{side}{finger}_tip_A",
                    ]
                )
            valid = all(
                point_is_valid(row, name, score_threshold, min_cameras)
                for name in required
            )
            frame = hand_frame(row, side) if valid else None
            if frame is None:
                rows.append({"frame": frame_number, "side": side, "valid": False})
                continue
            palm_center, palm_rotation = frame
            if side in previous_frames:
                previous = previous_frames[side]
                if np.dot(previous[:, 2], palm_rotation[:, 2]) < 0:
                    palm_rotation[:, 0] *= -1.0
                    palm_rotation[:, 2] *= -1.0
            previous_frames[side] = palm_rotation
            base = {
                "frame": frame_number,
                "side": side,
                "valid": True,
                "palm_x": palm_center[0],
                "palm_y": palm_center[1],
                "palm_z": palm_center[2],
            }
            for axis_idx, axis_name in enumerate(("across", "forward", "normal")):
                for component_idx, component_name in enumerate(("x", "y", "z")):
                    base[f"{axis_name}_{component_name}"] = palm_rotation[
                        component_idx, axis_idx
                    ]
            for finger in FINGERS:
                joint = point(row, f"{side}{finger}_joint_A")
                middle = point(row, f"{side}{finger}_mid_A")
                tip = point(row, f"{side}{finger}_tip_A")
                proximal = normalize(middle - joint)
                distal = normalize(tip - middle)
                local_proximal = palm_rotation.T @ proximal
                spread = signed_angle(local_proximal[0], local_proximal[1])
                base_flex = signed_angle(
                    local_proximal[2],
                    np.hypot(local_proximal[0], local_proximal[1]),
                )
                bend_axis = palm_rotation[:, 0]
                pip_flex = signed_angle(
                    float(np.dot(np.cross(proximal, distal), bend_axis)),
                    float(np.dot(proximal, distal)),
                )
                prefix = f"finger_{finger}"
                base[f"{prefix}_spread_deg"] = spread
                base[f"{prefix}_base_flex_deg"] = base_flex
                base[f"{prefix}_pip_flex_deg"] = pip_flex
                base[f"{prefix}_proximal_length"] = float(np.linalg.norm(middle - joint))
                base[f"{prefix}_distal_length"] = float(np.linalg.norm(tip - middle))
            rows.append(base)
    return pd.DataFrame(rows)


def static_landmark_summary(frame: pd.DataFrame, names: list[str]) -> dict:
    output = {}
    for name in names:
        if not all(f"{name}_{axis}" in frame for axis in "xyz"):
            continue
        points = frame[[f"{name}_x", f"{name}_y", f"{name}_z"]].to_numpy(float)
        median = np.nanmedian(points, axis=0)
        distances = np.linalg.norm(points - median, axis=1)
        output[name] = {
            "median": median.tolist(),
            "median_radius": float(np.nanmedian(distances)),
            "p95_radius": float(np.nanpercentile(distances, 95)),
            "valid_frames": int(np.isfinite(points).all(axis=1).sum()),
        }
    return output


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    rig = json.loads(args.rig_inspection.read_text(encoding="utf-8"))
    source_extrinsics = load_fixed_extrinsics(args.fixed_extrinsics)
    target_cameras = unique_blender_cameras(rig)
    mapping, mapping_diagnostics = find_camera_mapping(source_extrinsics, target_cameras)
    scale, rotation, translation, alignment_diagnostics = orientation_guided_similarity(
        mapping,
        source_extrinsics,
        target_cameras,
    )

    source = pd.read_csv(args.pose_csv)
    names = keypoint_names(list(source.columns))
    aligned = transform_keypoints(source, names, scale, rotation, translation)
    descriptors = build_hand_descriptors(
        aligned,
        score_threshold=args.score_threshold,
        min_cameras=args.min_cameras,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    aligned_path = args.output_dir / "aligned_keypoints_blender.csv"
    descriptor_path = args.output_dir / "hand_descriptors.csv"
    report_path = args.output_dir / "alignment.json"
    aligned.to_csv(aligned_path, index=False)
    descriptors.to_csv(descriptor_path, index=False)
    report = {
        "source_pose_csv": str(args.pose_csv.resolve()),
        "source_fixed_extrinsics": str(args.fixed_extrinsics.resolve()),
        "rig_inspection": str(args.rig_inspection.resolve()),
        "coordinate_formula": "X_blender = scale * rotation @ X_vggt + translation",
        "camera_mapping": mapping,
        "scale": scale,
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "mapping_diagnostics": mapping_diagnostics,
        "alignment_diagnostics": alignment_diagnostics,
        "keypoint_count": len(names),
        "frame_count": len(aligned),
        "valid_descriptor_rows": int(descriptors.get("valid", pd.Series(dtype=bool)).sum()),
        "static_landmarks_blender": static_landmark_summary(
            aligned,
            ["spout_tip_A", "point1", "point2", "point3"],
        ),
        "outputs": {
            "aligned_keypoints": str(aligned_path.resolve()),
            "hand_descriptors": str(descriptor_path.resolve()),
        },
    }
    report_path.write_text(json.dumps(json_ready(report), indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "camera_mapping": mapping,
                "center_rmse": alignment_diagnostics["center_rmse"],
                "rotation_mean_deg": alignment_diagnostics["rotation_mean_deg"],
                "frames": len(aligned),
                "valid_descriptor_rows": report["valid_descriptor_rows"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
