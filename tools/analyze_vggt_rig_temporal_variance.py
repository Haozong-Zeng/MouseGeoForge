#!/usr/bin/env python3
"""Compare per-trial temporal relative-rig variance from cached VGGT outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        nargs=3,
        action="append",
        metavar=("LABEL", "PERFRAME_ROOT", "COLOR"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def project_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(matrix)
    result = u @ vh
    if np.linalg.det(result) < 0:
        u[:, -1] *= -1
        result = u @ vh
    return result


def rotation_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    delta = first @ second.T
    cosine = np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def frame_rig_features(camera_csv: Path, score_csv: Path) -> pd.DataFrame:
    cameras = pd.read_csv(camera_csv)
    scores = pd.read_csv(score_csv)[["frame", "model_score"]]
    records = []
    for frame, group in cameras.groupby("frame", sort=True):
        by_cam = {str(row.cam): row for row in group.itertuples(index=False)}
        if not {"A", "B", "C"}.issubset(by_cam):
            continue
        rotations = {}
        centers = {}
        focals = {}
        for cam in ("A", "B", "C"):
            row = by_cam[cam]
            rotation = np.asarray(
                [
                    [row.relative_to_camA_r00, row.relative_to_camA_r01, row.relative_to_camA_r02],
                    [row.relative_to_camA_r10, row.relative_to_camA_r11, row.relative_to_camA_r12],
                    [row.relative_to_camA_r20, row.relative_to_camA_r21, row.relative_to_camA_r22],
                ],
                dtype=np.float64,
            )
            translation = np.asarray(
                [
                    row.relative_to_camA_t0,
                    row.relative_to_camA_t1,
                    row.relative_to_camA_t2,
                ],
                dtype=np.float64,
            )
            rotations[cam] = project_rotation(rotation)
            centers[cam] = -(rotations[cam].T @ translation)
            focals[cam] = 0.5 * (float(row.fx) + float(row.fy))
        pair_distances = np.asarray(
            [
                np.linalg.norm(centers["A"] - centers["B"]),
                np.linalg.norm(centers["A"] - centers["C"]),
                np.linalg.norm(centers["B"] - centers["C"]),
            ]
        )
        scale = max(float(pair_distances.mean()), 1e-8)
        records.append(
            {
                "frame": int(frame),
                "rotation_B": rotations["B"],
                "rotation_C": rotations["C"],
                "center_B": centers["B"] / scale,
                "center_C": centers["C"] / scale,
                "baseline_AB": pair_distances[0] / scale,
                "baseline_AC": pair_distances[1] / scale,
                "baseline_BC": pair_distances[2] / scale,
                "focal_A": focals["A"],
                "focal_B": focals["B"],
                "focal_C": focals["C"],
            }
        )
    frame_table = pd.DataFrame(records)
    return frame_table.merge(scores, on="frame", how="left")


def subset_metrics(frames: pd.DataFrame, subset: str) -> dict[str, float | int]:
    if len(frames) < 2:
        return {
            "subset": subset,
            "frames": len(frames),
            "rotation_jitter_median_deg": np.nan,
            "rotation_jitter_p90_deg": np.nan,
            "center_jitter_median": np.nan,
            "baseline_ratio_std": np.nan,
            "focal_cv": np.nan,
        }
    central_rotations = {}
    central_centers = {}
    for cam in ("B", "C"):
        central_rotations[cam] = project_rotation(
            np.stack(frames[f"rotation_{cam}"].to_list()).mean(axis=0)
        )
        central_centers[cam] = np.median(
            np.stack(frames[f"center_{cam}"].to_list()),
            axis=0,
        )
    rotation_jitter = []
    center_jitter = []
    for row in frames.itertuples(index=False):
        rotation_jitter.extend(
            [
                rotation_angle_deg(row.rotation_B, central_rotations["B"]),
                rotation_angle_deg(row.rotation_C, central_rotations["C"]),
            ]
        )
        center_jitter.extend(
            [
                np.linalg.norm(row.center_B - central_centers["B"]),
                np.linalg.norm(row.center_C - central_centers["C"]),
            ]
        )
    baseline_ratio_std = np.mean(
        [
            frames[column].std(ddof=0)
            for column in ("baseline_AB", "baseline_AC", "baseline_BC")
        ]
    )
    focal_cv = np.mean(
        [
            frames[column].std(ddof=0) / max(frames[column].mean(), 1e-8)
            for column in ("focal_A", "focal_B", "focal_C")
        ]
    )
    return {
        "subset": subset,
        "frames": len(frames),
        "rotation_jitter_median_deg": float(np.median(rotation_jitter)),
        "rotation_jitter_p90_deg": float(np.percentile(rotation_jitter, 90)),
        "center_jitter_median": float(np.median(center_jitter)),
        "baseline_ratio_std": float(baseline_ratio_std),
        "focal_cv": float(focal_cv),
    }


def analyze_model(label: str, root: Path) -> pd.DataFrame:
    rows = []
    for trial_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        camera_csv = trial_dir / "camera_extrinsics.csv"
        score_csv = trial_dir / "frame_model_scores.csv"
        if not camera_csv.exists() or not score_csv.exists():
            continue
        frames = frame_rig_features(camera_csv, score_csv)
        if frames.empty:
            continue
        score_threshold = frames["model_score"].median()
        subsets = {
            "all": frames,
            "high_confidence": frames[frames["model_score"] >= score_threshold],
            "low_confidence": frames[frames["model_score"] < score_threshold],
        }
        for subset, subset_frame in subsets.items():
            rows.append(
                {
                    "model": label,
                    "trial": trial_dir.name,
                    **subset_metrics(subset_frame, subset),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    colors = {}
    tables = []
    for label, root, color in args.model:
        colors[label] = color
        tables.append(analyze_model(label, Path(root).resolve()))
    metrics = pd.concat(tables, ignore_index=True)
    metrics.to_csv(args.output_dir / "trial_temporal_rig_metrics.csv", index=False)

    aggregate = (
        metrics.groupby(["model", "subset"], as_index=False)
        .agg(
            trials=("trial", "nunique"),
            rotation_jitter_median_deg=("rotation_jitter_median_deg", "median"),
            rotation_jitter_p90_deg=("rotation_jitter_p90_deg", "median"),
            center_jitter_median=("center_jitter_median", "median"),
            baseline_ratio_std=("baseline_ratio_std", "median"),
            focal_cv=("focal_cv", "median"),
        )
    )
    aggregate.to_csv(args.output_dir / "aggregate_temporal_rig_metrics.csv", index=False)
    (args.output_dir / "temporal_rig_summary.json").write_text(
        json.dumps(aggregate.to_dict(orient="records"), indent=2) + "\n",
        encoding="utf-8",
    )

    all_metrics = aggregate[aggregate["subset"] == "all"]
    metric_columns = [
        ("rotation_jitter_median_deg", "Rotation jitter (deg)"),
        ("center_jitter_median", "Center jitter"),
        ("baseline_ratio_std", "Baseline-ratio std"),
        ("focal_cv", "Focal coefficient of variation"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, (column, title) in zip(axes.flat, metric_columns):
        labels = all_metrics["model"].tolist()
        values = all_metrics[column].to_numpy()
        axis.bar(
            np.arange(len(labels)),
            values,
            color=[colors[label] for label in labels],
        )
        axis.set_title(title)
        axis.set_xticks(np.arange(len(labels)))
        axis.set_xticklabels(labels, rotation=15, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Full-video relative-rig temporal stability")
    fig.savefig(args.output_dir / "temporal_rig_variance_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
