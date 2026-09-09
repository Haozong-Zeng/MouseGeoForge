#!/usr/bin/env python3
"""Evaluate and visualize a pose-matched real/Blender pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--likelihood-threshold", type=float, default=0.5)
    return parser.parse_args()


def point_group(name: str) -> str:
    if name in {"l_mid_A", "r_mid_A"}:
        return "palm"
    if name.startswith(("l", "r")):
        return "finger"
    if name in {"snout_A", "mouth_tip_A"}:
        return "face"
    return "rig"


def draw_points(image: np.ndarray, rows: pd.DataFrame, synthetic: bool) -> np.ndarray:
    result = image.copy()
    x_name = "x_synthetic" if synthetic else "x_undistorted"
    y_name = "y_synthetic" if synthetic else "y_undistorted"
    for row in rows.itertuples():
        name = row.bodypart
        color = (80, 210, 255) if name.startswith("r") else (255, 120, 70) if name.startswith("l") else (80, 255, 100)
        point = (int(round(getattr(row, x_name))), int(round(getattr(row, y_name))))
        cv2.circle(result, point, 3, color, -1, lineType=cv2.LINE_AA)
    return result


def label(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 34), (20, 20, 20), -1)
    cv2.putText(result, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def main() -> None:
    args = parse_args()
    pilot_dir = args.pilot_dir.resolve()
    real = pd.read_csv(pilot_dir / "dlc_keypoints.csv")
    synthetic = pd.read_csv(pilot_dir / "blender_keypoint_projection.csv")
    merged = real.merge(
        synthetic,
        on=["camera", "source_frame", "blender_frame", "bodypart"],
        how="inner",
    )
    merged = merged[
        (merged["likelihood"] >= args.likelihood_threshold)
        & (merged["camera_depth"] > 0)
        & np.isfinite(merged[["x_undistorted", "y_undistorted", "x_synthetic", "y_synthetic"]]).all(axis=1)
    ].copy()
    merged["dx"] = merged["x_synthetic"] - merged["x_undistorted"]
    merged["dy"] = merged["y_synthetic"] - merged["y_undistorted"]
    merged["pixel_error"] = np.hypot(merged["dx"], merged["dy"])
    merged["group"] = merged["bodypart"].map(point_group)
    merged.to_csv(pilot_dir / "reprojection_rows.csv", index=False)

    summaries = []
    for keys, group in merged.groupby(["camera", "group"], sort=True):
        summaries.append(
            {
                "camera": keys[0],
                "group": keys[1],
                "count": int(len(group)),
                "mean_px": float(group["pixel_error"].mean()),
                "median_px": float(group["pixel_error"].median()),
                "p95_px": float(group["pixel_error"].quantile(0.95)),
            }
        )
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(pilot_dir / "reprojection_summary.csv", index=False)

    comparison_dir = pilot_dir / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    for (camera, source_frame), rows in merged.groupby(["camera", "source_frame"]):
        filename = f"frame_{int(source_frame):06d}.png"
        real_path = pilot_dir / "real_undistorted" / f"cam{camera}" / filename
        synthetic_path = pilot_dir / "synthetic" / f"cam{camera}" / filename
        if not real_path.exists() or not synthetic_path.exists():
            continue
        real_image = cv2.imread(str(real_path), cv2.IMREAD_COLOR)
        synthetic_image = cv2.imread(str(synthetic_path), cv2.IMREAD_COLOR)
        if real_image is None or synthetic_image is None:
            continue
        synthetic_image = cv2.resize(synthetic_image, (real_image.shape[1], real_image.shape[0]))
        real_overlay = draw_points(real_image, rows, synthetic=False)
        synthetic_overlay = draw_points(synthetic_image, rows, synthetic=True)
        frame_error = float(rows["pixel_error"].median())
        combined = np.hstack(
            [
                label(real_overlay, f"Real undistorted cam{camera} frame {source_frame}"),
                label(synthetic_overlay, f"V19 pose-matched | median reproj {frame_error:.1f}px"),
            ]
        )
        cv2.imwrite(str(comparison_dir / f"cam{camera}_{filename}"), combined)

    report = {
        "pilot_dir": str(pilot_dir),
        "likelihood_threshold": args.likelihood_threshold,
        "matched_rows": int(len(merged)),
        "overall_mean_px": float(merged["pixel_error"].mean()),
        "overall_median_px": float(merged["pixel_error"].median()),
        "overall_p95_px": float(merged["pixel_error"].quantile(0.95)),
        "summary": summaries,
        "interpretation": (
            "This reprojection audit measures consistency between DLC observations and the "
            "aligned VGGT/Anipose 3D marker package. Model-surface mismatch must be inspected "
            "separately in the side-by-side images."
        ),
    }
    (pilot_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
