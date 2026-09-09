#!/usr/bin/env python3
"""Summarize a Keypoint-MoSeq 3D recording and optionally match it to an HDF5 source."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


PERCENTILES = (0, 1, 5, 25, 50, 75, 95, 99, 100)

# The official v2 open-field file contains the same ordering without tail_tip.
# This sample's geometry strongly indicates that tail_tip was inserted at index 4.
INFERRED_BODY_PARTS_15 = [
    "spine1",
    "spine2",
    "spine3",
    "tail_base",
    "tail_tip",
    "forehead",
    "left_ear",
    "right_ear",
    "nose",
    "left_hindpaw_base",
    "left_hindpaw_tip",
    "right_hindpaw_base",
    "right_hindpaw_tip",
    "left_forepaw",
    "right_forepaw",
]

INFERRED_EDGES_15 = [
    (8, 5),
    (5, 6),
    (5, 7),
    (5, 0),
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 13),
    (0, 14),
    (2, 9),
    (9, 10),
    (2, 11),
    (11, 12),
]


def percentile_dict(values: np.ndarray, axis=None) -> dict[str, object]:
    result = np.percentile(values, PERCENTILES, axis=axis)
    return {
        str(percentile): np.asarray(result[index]).tolist()
        for index, percentile in enumerate(PERCENTILES)
    }


def digest_sample(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def match_hdf5(values: np.ndarray, hdf5_path: Path | None) -> dict[str, object] | None:
    if hdf5_path is None:
        return None

    import h5py

    matches = []
    datasets = []
    target_digest = digest_sample(values)
    with h5py.File(hdf5_path, "r") as handle:
        file_attrs = {key: str(value) for key, value in handle.attrs.items()}

        def inspect(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            item = {
                "name": name,
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "attrs": {key: str(value) for key, value in obj.attrs.items()},
            }
            if obj.ndim == 1 and obj.size <= 100:
                item["values"] = [
                    value.decode("utf-8") if isinstance(value, bytes) else str(value)
                    for value in obj[...]
                ]
            datasets.append(item)
            if tuple(obj.shape) != tuple(values.shape):
                return
            candidate = obj[...]
            if digest_sample(candidate) == target_digest or np.array_equal(candidate, values):
                matches.append(name)

        handle.visititems(inspect)

    return {
        "path": str(hdf5_path),
        "file_attrs": file_attrs,
        "dataset_count": len(datasets),
        "datasets": datasets,
        "exact_matches": matches,
    }


def analyze(values: np.ndarray, fps: float) -> dict[str, object]:
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError(f"Expected (frames, keypoints, 3), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Input contains NaN or infinite coordinates")

    frames, keypoints, dimensions = values.shape
    center = np.median(values, axis=1)
    centered = values - center[:, None, :]

    delta = values[:, :, None, :] - values[:, None, :, :]
    distances = np.linalg.norm(delta, axis=-1)
    upper = np.triu_indices(keypoints, 1)
    pairwise = distances[:, upper[0], upper[1]]

    xy_distances = np.linalg.norm(delta[..., :2], axis=-1)
    pairwise_xy = xy_distances[:, upper[0], upper[1]]
    center_step = np.linalg.norm(np.diff(center, axis=0), axis=1)
    center_step_xy = np.linalg.norm(np.diff(center[:, :2], axis=0), axis=1)

    pair_median = np.median(pairwise, axis=0)
    pair_mad = np.median(np.abs(pairwise - pair_median), axis=0)
    pair_relative_mad = pair_mad / np.maximum(pair_median, 1e-9)
    stable_order = np.lexsort((-pair_median, pair_relative_mad))
    stable_pairs = []
    for index in stable_order:
        if pair_median[index] < 10:
            continue
        stable_pairs.append(
            {
                "pair": [int(upper[0][index]), int(upper[1][index])],
                "median": float(pair_median[index]),
                "mad": float(pair_mad[index]),
                "relative_mad": float(pair_relative_mad[index]),
            }
        )
        if len(stable_pairs) == 20:
            break

    center_p01 = np.percentile(center, 1, axis=0)
    center_p99 = np.percentile(center, 99, axis=0)
    center_p05 = np.percentile(center, 5, axis=0)
    center_p95 = np.percentile(center, 95, axis=0)
    coordinate_p01 = np.percentile(values, 1, axis=(0, 1))
    coordinate_p99 = np.percentile(values, 99, axis=(0, 1))

    report = {
        "shape": [frames, keypoints, dimensions],
        "dtype": str(values.dtype),
        "all_finite": True,
        "timing": {
            "assumed_fps": fps,
            "duration_seconds": frames / fps,
            "duration_minutes": frames / fps / 60,
        },
        "coordinate_range": {
            "min": values.min(axis=(0, 1)).tolist(),
            "max": values.max(axis=(0, 1)).tolist(),
            "span": np.ptp(values, axis=(0, 1)).tolist(),
            "p01_p99_span": (coordinate_p99 - coordinate_p01).tolist(),
        },
        "robust_center": {
            "definition": "coordinate-wise median of all keypoints in each frame",
            "min": center.min(axis=0).tolist(),
            "max": center.max(axis=0).tolist(),
            "span": np.ptp(center, axis=0).tolist(),
            "p01_p99_span": (center_p99 - center_p01).tolist(),
            "p05_p95_span": (center_p95 - center_p05).tolist(),
        },
        "center_step_per_frame_3d": percentile_dict(center_step),
        "center_step_per_frame_xy": percentile_dict(center_step_xy),
        "centered_axis_span": percentile_dict(np.ptp(centered, axis=1), axis=0),
        "max_pairwise_distance_3d": percentile_dict(pairwise.max(axis=1)),
        "max_pairwise_distance_xy": percentile_dict(pairwise_xy.max(axis=1)),
        "max_xy_radius_about_center": percentile_dict(
            np.linalg.norm(centered[..., :2], axis=-1).max(axis=1)
        ),
        "stable_pairs": stable_pairs,
    }
    if keypoints == 15:
        report["inferred_bodyparts"] = INFERRED_BODY_PARTS_15
        report["inferred_bodyparts_note"] = (
            "High-confidence topology inference; names are not embedded in the NPY. "
            "The official v2 3D file uses the same order with tail_tip omitted."
        )
    return report


def plot_summary(values: np.ndarray, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    center = np.median(values, axis=1)
    arena_center = np.median(center[:, :2], axis=0)
    centered_path = center[:, :2] - arena_center
    representative_frames = [0, len(values) // 3, 2 * len(values) // 3]

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

    axes[0].plot(centered_path[:, 0], centered_path[:, 1], color="#176B87", lw=0.35)
    axes[0].add_patch(Circle((0, 0), 85, fill=False, color="#B42318", lw=2, label="current 170 diameter"))
    axes[0].add_patch(Circle((0, 0), 180, fill=False, color="#E88D20", lw=1.5, label="360 diameter"))
    axes[0].add_patch(Rectangle((-150, -150), 300, 300, fill=False, color="#177245", lw=1.5, label="paper 300 x 300 arena"))
    axes[0].set_title("Robust-center XY path")
    axes[0].set_xlabel("X (data units)")
    axes[0].set_ylabel("Y (data units)")
    axes[0].axis("equal")
    axes[0].legend(fontsize=8, loc="upper right")

    colors = ["#176B87", "#E88D20", "#8A3FFC"]
    for frame, color in zip(representative_frames, colors):
        pose = values[frame] - np.median(values[frame], axis=0)
        for start, end in INFERRED_EDGES_15:
            axes[1].plot(pose[[start, end], 0], pose[[start, end], 1], color=color, alpha=0.65, lw=1)
        axes[1].scatter(pose[:, 0], pose[:, 1], s=16, color=color, alpha=0.8, label=f"frame {frame}")
    axes[1].set_title("Centered poses, XY")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")
    axes[1].axis("equal")
    axes[1].legend(fontsize=8)

    frame = 0
    pose = values[frame] - np.median(values[frame], axis=0)
    for start, end in INFERRED_EDGES_15:
        axes[2].plot(pose[[start, end], 0], pose[[start, end], 2], color="#555555", lw=1)
    axes[2].scatter(pose[:, 0], pose[:, 2], s=24, color="#176B87")
    for index, (x_coord, z_coord) in enumerate(pose[:, [0, 2]]):
        axes[2].annotate(str(index), (x_coord, z_coord), xytext=(3, 3), textcoords="offset points", fontsize=8)
    axes[2].set_title("Frame 0, XZ with indices")
    axes[2].set_xlabel("X")
    axes[2].set_ylabel("Z")
    axes[2].axis("equal")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--source-h5", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = np.load(args.input)
    report = {
        "input": str(args.input),
        "analysis": analyze(values, args.fps),
        "source_hdf5": match_hdf5(values, args.source_h5),
    }
    payload = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    if args.plot:
        plot_summary(values, args.plot)
    print(payload)


if __name__ == "__main__":
    main()
