#!/usr/bin/env python3
"""Build clean source and pose-matched interaction sheets for visual QA."""

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
    parser.add_argument("--camera", default="B")
    parser.add_argument("--frames", default="62,112,162")
    parser.add_argument("--likelihood-threshold", type=float, default=0.5)
    parser.add_argument("--padding", type=int, default=55)
    return parser.parse_args()


def interaction_keypoint(name: str) -> bool:
    return name.startswith(("l", "r")) or name in {
        "snout_A",
        "mouth_tip_A",
        "spout_tip_A",
    }


def labelled(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 34), (16, 16, 16), -1)
    cv2.putText(
        result,
        text,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return result


def fit_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((height, width, 3), 12, dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def square_bounds(
    points: pd.DataFrame,
    image_width: int,
    image_height: int,
    padding: int,
    minimum_size: int = 300,
) -> tuple[int, int, int, int]:
    min_x = float(points["x_undistorted"].min()) - padding
    max_x = float(points["x_undistorted"].max()) + padding
    min_y = float(points["y_undistorted"].min()) - padding
    max_y = float(points["y_undistorted"].max()) + padding
    size = max(max_x - min_x, max_y - min_y, minimum_size)
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    x0 = int(round(center_x - size / 2.0))
    y0 = int(round(center_y - size / 2.0))
    x0 = min(max(0, x0), max(0, image_width - int(round(size))))
    y0 = min(max(0, y0), max(0, image_height - int(round(size))))
    x1 = min(image_width, x0 + int(round(size)))
    y1 = min(image_height, y0 + int(round(size)))
    return x0, y0, x1, y1


def main() -> None:
    args = parse_args()
    pilot_dir = args.pilot_dir.resolve()
    camera = args.camera.upper()
    frames = sorted({int(value.strip()) for value in args.frames.split(",") if value.strip()})
    labels = pd.read_csv(pilot_dir / "dlc_keypoints.csv")
    selected = labels[
        (labels["camera"] == camera)
        & labels["source_frame"].isin(frames)
        & (labels["likelihood"] >= args.likelihood_threshold)
        & labels["bodypart"].map(interaction_keypoint)
    ].copy()
    if selected.empty:
        raise ValueError("No valid interaction keypoints for requested camera/frames")

    first_path = pilot_dir / "real_undistorted" / f"cam{camera}" / f"frame_{frames[0]:06d}.png"
    first_image = cv2.imread(str(first_path), cv2.IMREAD_COLOR)
    if first_image is None:
        raise FileNotFoundError(first_path)
    image_height, image_width = first_image.shape[:2]
    x0 = max(0, int(np.floor(selected["x_undistorted"].min())) - args.padding)
    y0 = max(0, int(np.floor(selected["y_undistorted"].min())) - args.padding)
    x1 = min(image_width, int(np.ceil(selected["x_undistorted"].max())) + args.padding)
    y1 = min(image_height, int(np.ceil(selected["y_undistorted"].max())) + args.padding)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Invalid interaction crop")

    source_panels = []
    comparison_rows = []
    active_rows = []
    active_crops = {}
    output_dir = pilot_dir / "sample_sheets"
    crop_dir = output_dir / "interaction_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        filename = f"frame_{frame:06d}.png"
        real_path = pilot_dir / "real_undistorted" / f"cam{camera}" / filename
        synthetic_path = pilot_dir / "synthetic" / f"cam{camera}" / filename
        real = cv2.imread(str(real_path), cv2.IMREAD_COLOR)
        synthetic = cv2.imread(str(synthetic_path), cv2.IMREAD_COLOR)
        if real is None or synthetic is None:
            raise FileNotFoundError(f"Missing pair for cam{camera} frame {frame}")
        synthetic = cv2.resize(synthetic, (image_width, image_height), interpolation=cv2.INTER_CUBIC)
        real_crop = real[y0:y1, x0:x1]
        synthetic_crop = synthetic[y0:y1, x0:x1]
        cv2.imwrite(str(crop_dir / f"real_cam{camera}_{filename}"), real_crop)
        cv2.imwrite(str(crop_dir / f"synthetic_cam{camera}_{filename}"), synthetic_crop)

        frame_points = selected[selected["source_frame"] == frame]
        mouth_rows = frame_points[frame_points["bodypart"] == "mouth_tip_A"]
        palm_rows = frame_points[frame_points["bodypart"].isin(["l_mid_A", "r_mid_A"])]
        if mouth_rows.empty or palm_rows.empty:
            raise ValueError(f"Missing mouth/palm points for frame {frame}")
        mouth_xy = mouth_rows[["x_undistorted", "y_undistorted"]].iloc[0].to_numpy(float)
        palm_xy = palm_rows[["x_undistorted", "y_undistorted"]].to_numpy(float)
        active_palm = palm_rows.iloc[int(np.argmin(np.linalg.norm(palm_xy - mouth_xy, axis=1)))]
        active_side = str(active_palm["bodypart"])[0].lower()
        active_points = frame_points[
            frame_points["bodypart"].str.startswith(active_side)
            | frame_points["bodypart"].isin(["snout_A", "mouth_tip_A", "spout_tip_A"])
        ]
        ax0, ay0, ax1, ay1 = square_bounds(
            active_points, image_width, image_height, args.padding
        )
        active_real = real[ay0:ay1, ax0:ax1]
        active_synthetic = synthetic[ay0:ay1, ax0:ax1]
        cv2.imwrite(str(crop_dir / f"real_active_{active_side}_cam{camera}_{filename}"), active_real)
        cv2.imwrite(
            str(crop_dir / f"synthetic_active_{active_side}_cam{camera}_{filename}"),
            active_synthetic,
        )
        active_crops[str(frame)] = {
            "side": active_side,
            "xyxy": [ax0, ay0, ax1, ay1],
        }
        active_rows.append(
            np.hstack(
                [
                    labelled(
                        fit_panel(active_real, 520, 420),
                        f"Real active hand ({active_side.upper()}) | frame {frame}",
                    ),
                    labelled(
                        fit_panel(active_synthetic, 520, 420),
                        f"Blender active hand ({active_side.upper()})",
                    ),
                ]
            )
        )

        source_panels.append(
            labelled(fit_panel(real, 600, 400), f"Source photo | cam{camera} frame {frame}")
        )
        full_real = labelled(fit_panel(real, 480, 320), f"Real source | frame {frame}")
        full_synthetic = labelled(fit_panel(synthetic, 480, 320), "Blender V19")
        crop_real = labelled(fit_panel(real_crop, 480, 320), "Real interaction crop")
        crop_synthetic = labelled(fit_panel(synthetic_crop, 480, 320), "Blender interaction crop")
        comparison_rows.append(np.hstack([full_real, full_synthetic, crop_real, crop_synthetic]))

    source_sheet = np.hstack(source_panels)
    comparison_sheet = np.vstack(comparison_rows)
    active_sheet = np.vstack(active_rows)
    source_sheet_path = output_dir / f"source_photos_cam{camera}.png"
    comparison_sheet_path = output_dir / f"local_interaction_comparison_cam{camera}.png"
    active_sheet_path = output_dir / f"active_hand_face_comparison_cam{camera}.png"
    cv2.imwrite(str(source_sheet_path), source_sheet)
    cv2.imwrite(str(comparison_sheet_path), comparison_sheet)
    cv2.imwrite(str(active_sheet_path), active_sheet)
    report = {
        "camera": camera,
        "source_frames": frames,
        "source_photo_type": "undistorted with original calibrated K",
        "interaction_keypoints": "hands, snout, mouth, spout",
        "shared_crop_xyxy": [x0, y0, x1, y1],
        "source_sheet": str(source_sheet_path),
        "comparison_sheet": str(comparison_sheet_path),
        "active_hand_face_sheet": str(active_sheet_path),
        "active_crops": active_crops,
    }
    (output_dir / "sample_sheet_manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
