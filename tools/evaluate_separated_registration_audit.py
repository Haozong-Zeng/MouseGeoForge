#!/usr/bin/env python3
"""Separate camera/3D projection error from actual Blender bone registration error."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


KEYS = ["camera", "source_frame", "blender_frame", "bodypart"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--likelihood-threshold", type=float, default=0.5)
    return parser.parse_args()


def metric_summary(frame: pd.DataFrame, metric: str, groups: list[str]) -> list[dict]:
    rows = []
    for keys, values in frame.groupby(groups, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = {name: value for name, value in zip(groups, keys)}
        row.update(
            {
                "count": int(len(values)),
                "mean_px": float(values[metric].mean()),
                "median_px": float(values[metric].median()),
                "p95_px": float(values[metric].quantile(0.95)),
            }
        )
        rows.append(row)
    return rows


def draw_cross(image: np.ndarray, point: tuple[int, int], color: tuple[int, int, int], size: int) -> None:
    x, y = point
    cv2.line(image, (x - size, y - size), (x + size, y + size), color, 2, cv2.LINE_AA)
    cv2.line(image, (x - size, y + size), (x + size, y - size), color, 2, cv2.LINE_AA)


def make_overlays(audit_dir: Path, merged: pd.DataFrame) -> None:
    output_dir = audit_dir / "separated_overlays"
    output_dir.mkdir(parents=True, exist_ok=True)
    for (camera, frame), rows in merged.groupby(["camera", "source_frame"], sort=True):
        filename = f"frame_{int(frame):06d}.png"
        path = audit_dir / "real_undistorted" / f"cam{camera}" / filename
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        for row in rows.itertuples():
            real = (int(round(row.x_undistorted)), int(round(row.y_undistorted)))
            target = (int(round(row.x_synthetic)), int(round(row.y_synthetic)))
            model = (int(round(row.x_model)), int(round(row.y_model)))
            cv2.circle(image, real, 5, (80, 255, 80), 2, cv2.LINE_AA)
            draw_cross(image, target, (255, 220, 60), 5)
            cv2.drawMarker(
                image,
                model,
                (255, 80, 220),
                markerType=cv2.MARKER_TRIANGLE_UP,
                markerSize=12,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
            cv2.line(image, target, model, (200, 120, 200), 1, cv2.LINE_AA)
        cv2.rectangle(image, (0, 0), (image.shape[1], 58), (18, 18, 18), -1)
        cv2.putText(
            image,
            f"cam{camera} frame {frame} | green=real DLC  cyan=aligned 3D  magenta=model bone",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        median_model = float(rows["model_real_px"].median())
        median_target = float(rows["target_real_px"].median())
        cv2.putText(
            image,
            f"median target-real={median_target:.1f}px, model-real={median_model:.1f}px",
            (12, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(output_dir / f"cam{camera}_{filename}"), image)


def main() -> None:
    args = parse_args()
    audit_dir = args.audit_dir.resolve()
    real = pd.read_csv(audit_dir / "dlc_keypoints.csv")
    target = pd.read_csv(audit_dir / "blender_keypoint_projection.csv")
    model = pd.read_csv(audit_dir / "model_bone_projection.csv")
    static = pd.read_csv(audit_dir / "static_rig_projection.csv")
    manifest = json.loads((audit_dir / "render_manifest.json").read_text(encoding="utf-8"))

    merged = real.merge(target, on=KEYS, how="inner").merge(model, on=KEYS, how="inner")
    merged = merged[
        (merged["likelihood"] >= args.likelihood_threshold)
        & (merged["camera_depth_x"] > 0)
        & (merged["camera_depth_y"] > 0)
    ].copy()
    merged["target_real_px"] = np.hypot(
        merged["x_synthetic"] - merged["x_undistorted"],
        merged["y_synthetic"] - merged["y_undistorted"],
    )
    merged["model_real_px"] = np.hypot(
        merged["x_model"] - merged["x_undistorted"],
        merged["y_model"] - merged["y_undistorted"],
    )
    merged["model_target_px"] = np.hypot(
        merged["x_model"] - merged["x_synthetic"],
        merged["y_model"] - merged["y_synthetic"],
    )
    merged.to_csv(audit_dir / "separated_registration_rows.csv", index=False)

    static_merged = real.merge(static, on=KEYS, how="inner")
    static_merged = static_merged[static_merged["likelihood"] >= args.likelihood_threshold].copy()
    static_merged["static_real_px"] = np.hypot(
        static_merged["x_static"] - static_merged["x_undistorted"],
        static_merged["y_static"] - static_merged["y_undistorted"],
    )
    static_merged.to_csv(audit_dir / "static_registration_rows.csv", index=False)

    camera_pose = manifest.get("camera_pose_comparison", [])
    report = {
        "audit_dir": str(audit_dir),
        "matched_model_rows": int(len(merged)),
        "target_projection": {
            "median_px": float(merged["target_real_px"].median()),
            "p95_px": float(merged["target_real_px"].quantile(0.95)),
        },
        "actual_model_bones": {
            "median_real_px": float(merged["model_real_px"].median()),
            "p95_real_px": float(merged["model_real_px"].quantile(0.95)),
            "median_target_px": float(merged["model_target_px"].median()),
            "p95_target_px": float(merged["model_target_px"].quantile(0.95)),
        },
        "by_camera_category_target_real": metric_summary(
            merged, "target_real_px", ["camera", "category"]
        ),
        "by_camera_category_model_real": metric_summary(
            merged, "model_real_px", ["camera", "category"]
        ),
        "by_camera_category_model_target": metric_summary(
            merged, "model_target_px", ["camera", "category"]
        ),
        "static_spout_proxy": {
            "count": int(len(static_merged)),
            "median_real_px": float(static_merged["static_real_px"].median()),
            "p95_real_px": float(static_merged["static_real_px"].quantile(0.95)),
            "warning": "Part_WaterPipe vertex 124 is a nearest modeled tip proxy, not a manually certified correspondence.",
        },
        "temporary_vs_builtin_cameras": camera_pose,
    }
    (audit_dir / "separated_registration_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    make_overlays(audit_dir, merged)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
