#!/usr/bin/env python3
"""Extract calibrated real frames and DLC labels for a pose-matched Blender pilot."""

from __future__ import annotations

import argparse
import csv
import json
import tomllib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


CAMERAS = ("A", "B", "C")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, default=62)
    parser.add_argument("--frame-end", type=int, default=162)
    parser.add_argument("--frame-step", type=int, default=25)
    parser.add_argument(
        "--frames",
        help="Optional comma-separated source frame indices; overrides start/end/step.",
    )
    parser.add_argument("--likelihood-threshold", type=float, default=0.5)
    return parser.parse_args()


def selected_frames(args: argparse.Namespace) -> list[int]:
    if args.frames:
        frames = sorted({int(value.strip()) for value in args.frames.split(",") if value.strip()})
    else:
        frames = list(range(args.frame_start, args.frame_end + 1, args.frame_step))
        if frames[-1] != args.frame_end:
            frames.append(args.frame_end)
    if not frames or frames[0] < 0:
        raise ValueError("At least one non-negative frame is required")
    return frames


def load_calibration(path: Path) -> dict[str, dict]:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    cameras = {}
    for value in raw.values():
        if not isinstance(value, dict) or "name" not in value:
            continue
        name = str(value["name"])
        if name not in CAMERAS:
            continue
        cameras[name] = {
            "size": [int(v) for v in value["size"]],
            "matrix": np.asarray(value["matrix"], dtype=np.float64),
            "distortions": np.asarray(value["distortions"], dtype=np.float64),
        }
    missing = sorted(set(CAMERAS) - set(cameras))
    if missing:
        raise KeyError(f"Missing calibration cameras: {missing}")
    return cameras


def read_frames(video_path: Path, wanted: list[int]) -> dict[int, np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frames: dict[int, np.ndarray] = {}
    wanted_set = set(wanted)
    last = max(wanted)
    index = 0
    while index <= last:
        ok, image = capture.read()
        if not ok:
            break
        if index in wanted_set:
            frames[index] = image
        index += 1
    capture.release()
    missing = sorted(wanted_set - set(frames))
    if missing:
        raise IndexError(f"Video {video_path} is missing requested frames: {missing}")
    return frames


def dlc_table(path: Path) -> pd.DataFrame:
    frame = pd.read_hdf(path)
    if not isinstance(frame.columns, pd.MultiIndex):
        raise ValueError(f"Expected DLC MultiIndex columns in {path}")
    scorer = frame.columns.get_level_values("scorer")[0]
    return frame[scorer]


def draw_points(image: np.ndarray, rows: list[dict], threshold: float) -> np.ndarray:
    result = image.copy()
    for row in rows:
        if row["likelihood"] < threshold:
            continue
        point = (int(round(row["x_undistorted"])), int(round(row["y_undistorted"])))
        side = row["bodypart"][:1].lower()
        color = (80, 210, 255) if side == "r" else (255, 120, 70) if side == "l" else (80, 255, 100)
        cv2.circle(result, point, 3, color, -1, lineType=cv2.LINE_AA)
    return result


def main() -> None:
    args = parse_args()
    trial_dir = args.trial_dir.resolve()
    output_dir = args.output_dir.resolve()
    frames = selected_frames(args)
    calibration_path = trial_dir / "calibration" / "calibration.toml"
    calibration = load_calibration(calibration_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    pair_rows: list[dict] = []
    for camera in CAMERAS:
        video_path = trial_dir / "videos-raw" / f"pawfilt_cam{camera}.mp4"
        pose_path = trial_dir / "pose-2d" / f"pawfilt_cam{camera}.h5"
        images = read_frames(video_path, frames)
        poses = dlc_table(pose_path)
        intrinsic = calibration[camera]["matrix"]
        distortion = calibration[camera]["distortions"]
        distorted_dir = output_dir / "real_distorted" / f"cam{camera}"
        undistorted_dir = output_dir / "real_undistorted" / f"cam{camera}"
        overlay_dir = output_dir / "real_undistorted_overlay" / f"cam{camera}"
        for directory in (distorted_dir, undistorted_dir, overlay_dir):
            directory.mkdir(parents=True, exist_ok=True)

        bodyparts = list(dict.fromkeys(poses.columns.get_level_values("bodyparts")))
        for frame_index in frames:
            image = images[frame_index]
            height, width = image.shape[:2]
            expected_size = tuple(calibration[camera]["size"])
            if (width, height) != expected_size:
                raise ValueError(
                    f"cam{camera} frame size {(width, height)} differs from calibration {expected_size}"
                )
            undistorted = cv2.undistort(image, intrinsic, distortion, None, intrinsic)
            frame_rows: list[dict] = []
            for bodypart in bodyparts:
                x = float(poses.loc[frame_index, (bodypart, "x")])
                y = float(poses.loc[frame_index, (bodypart, "y")])
                likelihood = float(poses.loc[frame_index, (bodypart, "likelihood")])
                point = np.asarray([[[x, y]]], dtype=np.float64)
                undistorted_point = cv2.undistortPoints(
                    point, intrinsic, distortion, P=intrinsic
                )[0, 0]
                row = {
                    "camera": camera,
                    "source_frame": frame_index,
                    "blender_frame": frame_index + 1,
                    "bodypart": bodypart,
                    "x": x,
                    "y": y,
                    "x_undistorted": float(undistorted_point[0]),
                    "y_undistorted": float(undistorted_point[1]),
                    "likelihood": likelihood,
                }
                frame_rows.append(row)
                all_rows.append(row)

            filename = f"frame_{frame_index:06d}.png"
            cv2.imwrite(str(distorted_dir / filename), image)
            cv2.imwrite(str(undistorted_dir / filename), undistorted)
            cv2.imwrite(
                str(overlay_dir / filename),
                draw_points(undistorted, frame_rows, args.likelihood_threshold),
            )
            pair_rows.append(
                {
                    "camera": camera,
                    "source_frame": frame_index,
                    "blender_frame": frame_index + 1,
                    "real_distorted": str((distorted_dir / filename).resolve()),
                    "real_undistorted": str((undistorted_dir / filename).resolve()),
                    "synthetic": str(
                        (output_dir / "synthetic" / f"cam{camera}" / filename).resolve()
                    ),
                }
            )

    label_path = output_dir / "dlc_keypoints.csv"
    with label_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    pair_path = output_dir / "pairs.csv"
    with pair_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)

    manifest = {
        "trial_dir": str(trial_dir),
        "calibration": str(calibration_path.resolve()),
        "source_frames": frames,
        "blender_frame_offset": 1,
        "camera_names": list(CAMERAS),
        "likelihood_threshold": args.likelihood_threshold,
        "pair_count": len(pair_rows),
        "keypoint_row_count": len(all_rows),
    }
    (output_dir / "real_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
