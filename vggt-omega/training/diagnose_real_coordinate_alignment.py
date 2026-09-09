"""Diagnose per-trial real-data gauge normalization for VGGT-Omega training."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.finetune_blendedmvs import (
    REAL_GEOMETRY_KEYPOINTS,
    RealMultiViewGeometryDataset,
    discover_real_trials,
)
from training.geometry_coupled_losses import project_sparse_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cameras", default="A,B,C")
    parser.add_argument("--frame-step", type=int, default=20)
    parser.add_argument("--frame-count", type=int, default=0)
    parser.add_argument("--image-width", type=int, default=624)
    parser.add_argument("--image-height", type=int, default=416)
    parser.add_argument("--likelihood-threshold", type=float, default=0.7)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cameras = [item.strip() for item in args.cameras.split(",") if item.strip()]
    roots = sorted(
        path
        for path in args.root.iterdir()
        if path.is_dir()
        and (path / "pose-3d" / "pawfilt.csv").is_file()
        and (path / "calibration" / "calibration.toml").is_file()
    )
    trials = discover_real_trials(",".join(str(path) for path in roots), cameras)
    dataset = RealMultiViewGeometryDataset(
        trials=trials,
        cameras=cameras,
        image_width=args.image_width,
        image_height=args.image_height,
        frame_start=0,
        frame_step=args.frame_step,
        frame_count=args.frame_count,
        likelihood_threshold=args.likelihood_threshold,
        keypoints=REAL_GEOMETRY_KEYPOINTS,
        palm_weight=4.0,
        finger_weight=2.0,
        face_weight=1.5,
        anchor_weight=1.0,
        canonicalize_camera=True,
        scale_mode="median_baseline",
    )

    accumulators: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for sample_index in range(len(dataset)):
        sample = dataset[sample_index]
        trial = str(sample["scene"])
        valid = sample["geometry_valid"]
        point_norms = torch.linalg.vector_norm(
            sample["geometry_world"][valid],
            dim=-1,
        )
        if point_norms.numel():
            accumulators[trial]["normalized_point_norm"].extend(
                point_norms.tolist()
            )

        first_extrinsic = sample["extrinsics"][0]
        accumulators[trial]["first_rotation_error"].append(
            float(torch.linalg.matrix_norm(first_extrinsic[:3, :3] - torch.eye(3)))
        )
        accumulators[trial]["first_translation_error"].append(
            float(torch.linalg.vector_norm(first_extrinsic[:3, 3]))
        )

        projected_uv, projected_depth = project_sparse_points(
            sample["geometry_world"][None],
            sample["extrinsics"][None],
            sample["intrinsics"][None],
        )
        projected_uv = projected_uv[0]
        projected_depth = projected_depth[0]
        reprojection_valid = (
            sample["geometry_view_valid"]
            & torch.isfinite(projected_uv).all(dim=-1)
            & (projected_depth > 1e-6)
        )
        error = torch.linalg.vector_norm(
            projected_uv - sample["geometry_xy"],
            dim=-1,
        )
        if reprojection_valid.any():
            accumulators[trial]["reprojection_error_px"].extend(
                error[reprojection_valid].tolist()
            )
        accumulators[trial]["baseline_scale"].append(float(sample["scale"]))

    rows = []
    trial_lookup = {trial.name: trial for trial in dataset.trials}
    for trial in sorted(accumulators):
        values = accumulators[trial]
        normalized_norms = np.asarray(values["normalized_point_norm"])
        baseline_scale = float(np.median(values["baseline_scale"]))
        ratio = float(normalized_norms.mean())
        pose3d = dataset.pose3d[trial]
        rotation_columns = [f"M_{row}{column}" for row in range(3) for column in range(3)]
        center_columns = (
            [f"center_{axis}" for axis in "xyz"]
            if all(f"center_{axis}" in pose3d for axis in "xyz")
            else [f"center_{index}" for index in range(3)]
        )
        if all(column in pose3d for column in rotation_columns):
            anipose_rotation = np.asarray(
                [
                    [float(pose3d[f"M_{row}{column}"].mean()) for column in range(3)]
                    for row in range(3)
                ]
            )
        else:
            anipose_rotation = np.full((3, 3), np.nan)
        if all(column in pose3d for column in center_columns):
            anipose_center = np.asarray(
                [float(pose3d[column].mean()) for column in center_columns]
            )
        else:
            anipose_center = np.full(3, np.nan)
        rows.append(
            {
                "trial": trial,
                "root": str(trial_lookup[trial].root),
                "samples": len(values["baseline_scale"]),
                "baseline_scale": baseline_scale,
                "sparse_mean_point_scale": baseline_scale * ratio,
                "sparse_mean_point_to_baseline_ratio": ratio,
                "first_rotation_error_max": max(values["first_rotation_error"]),
                "first_translation_error_max": max(values["first_translation_error"]),
                "label_reprojection_mean_px": float(
                    np.mean(values["reprojection_error_px"])
                ),
                "label_reprojection_median_px": float(
                    np.median(values["reprojection_error_px"])
                ),
                "anipose_adjustment_rotation": anipose_rotation.tolist(),
                "anipose_adjustment_center": anipose_center.tolist(),
            }
        )

    ratios = np.asarray(
        [row["sparse_mean_point_to_baseline_ratio"] for row in rows]
    )
    centers = np.asarray([row["anipose_adjustment_center"] for row in rows])
    rotations = np.asarray(
        [row["anipose_adjustment_rotation"] for row in rows]
    )
    finite_rotation_indices = np.flatnonzero(np.isfinite(rotations).all(axis=(1, 2)))
    reference_rotation = (
        rotations[int(finite_rotation_indices[0])]
        if finite_rotation_indices.size
        else np.eye(3)
    )
    relative_angles = []
    for rotation in rotations:
        if not np.isfinite(rotation).all():
            continue
        relative = rotation @ reference_rotation.T
        cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
        relative_angles.append(float(np.degrees(np.arccos(cosine))))
    summary = {
        "trials": len(rows),
        "samples": len(dataset),
        "ratio_min": float(ratios.min()),
        "ratio_median": float(np.median(ratios)),
        "ratio_max": float(ratios.max()),
        "ratio_cv": float(ratios.std() / ratios.mean()),
        "anipose_center_span": (
            np.nanmax(centers, axis=0) - np.nanmin(centers, axis=0)
        ).tolist(),
        "anipose_rotation_difference_max_deg": (
            max(relative_angles) if relative_angles else None
        ),
        "note": (
            "sparse_mean_point_scale approximates the official dense mean-point "
            "normalization using triangulated keypoints."
        ),
    }
    payload = {"summary": summary, "trials": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
