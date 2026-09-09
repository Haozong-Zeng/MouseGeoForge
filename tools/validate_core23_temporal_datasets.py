#!/usr/bin/env python3
"""Validate MouseGeoForge Core23 temporal sequence manifests and payloads."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np

from core23_temporal_format import CORE23, FORMAT


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    return parser.parse_args()


def resolve(root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def validate_manifest(manifest: Path, project_root: Path) -> dict:
    records = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"Empty manifest: {manifest}")
    sequence_ids: set[str] = set()
    split_counts: Counter[str] = Counter()
    condition_counts: Counter[str] = Counter()
    total_frames = total_rgb_frames = total_valid_points = 0
    max_speed_mm_s = 0.0
    speed_samples: list[np.ndarray] = []

    for record in records:
        sequence_id = record["sequence_id"]
        if sequence_id in sequence_ids:
            raise ValueError(f"Duplicate sequence_id {sequence_id} in {manifest}")
        sequence_ids.add(sequence_id)
        split_counts[record.get("split", "unset")] += 1
        condition_counts[record.get("condition", "unspecified")] += 1
        path = resolve(project_root, record["sequence_npz"])
        if not path.is_file():
            raise FileNotFoundError(path)

        with np.load(path, allow_pickle=False) as data:
            names = tuple(str(value) for value in data["names"].tolist())
            if str(data["format"].item()) != FORMAT or names != CORE23:
                raise ValueError(f"Format/schema mismatch: {path}")
            xyz = data["joint_world_xyz"]
            valid = data["world_valid"]
            frames = data["frame_indices"]
            timestamps = data["timestamps_seconds"]
            instances = int(record["instance_count"])
            expected_xyz = (int(record["frame_count"]), instances, len(CORE23), 3)
            if xyz.shape != expected_xyz or valid.shape != expected_xyz[:-1]:
                raise ValueError(
                    f"Pose shape mismatch {path}: {xyz.shape}/{valid.shape}, expected {expected_xyz}"
                )
            if frames.shape != (expected_xyz[0],) or timestamps.shape != frames.shape:
                raise ValueError(f"Timeline shape mismatch: {path}")
            if not np.array_equal(frames, np.arange(expected_xyz[0])):
                raise ValueError(f"Non-contiguous frame indices: {path}")
            if not np.all(np.diff(timestamps) > 0):
                raise ValueError(f"Non-monotonic timestamps: {path}")
            if expected_xyz[0] > 1:
                expected_dt = 1.0 / float(record["fps"])
                if not np.allclose(np.diff(timestamps), expected_dt, atol=1e-10):
                    raise ValueError(f"Timestamp/FPS mismatch: {path}")
            if not np.isfinite(xyz).all() or np.any(valid & ~np.isfinite(xyz).all(axis=-1)):
                raise ValueError(f"Non-finite converted coordinates: {path}")
            if np.any(valid & np.all(xyz == 0, axis=-1)):
                raise ValueError(f"A valid point was replaced by the missing-value zero sentinel: {path}")
            if bool(data["fixed_camera"].item()) != bool(record["fixed_camera"]):
                raise ValueError(f"fixed_camera mismatch: {path}")

            if bool(record.get("camera_available", False)):
                for key, shape in (
                    ("projection_matrices", (record["camera_count"], 3, 4)),
                    ("intrinsics", (record["camera_count"], 3, 3)),
                    ("extrinsics", (record["camera_count"], 4, 4)),
                ):
                    if data[key].shape != shape or not np.isfinite(data[key]).all():
                        raise ValueError(f"Invalid static camera array {key}: {path}")
                # A fixed rig is deliberately represented without a leading T axis.
                if data["projection_matrices"].ndim != 3:
                    raise ValueError(f"Camera parameters were expanded over time: {path}")

            rgb_count = int(record.get("rgb_frame_count", 0))
            if rgb_count:
                if data["rgb_pose_indices"].shape != (rgb_count,):
                    raise ValueError(f"RGB/pose index shape mismatch: {path}")
                expected_box = (rgb_count, record["camera_count"], instances)
                if data["box_valid"].shape != expected_box:
                    raise ValueError(f"Box validity shape mismatch: {path}")
                if data["boxes_cxcywh"].shape != (*expected_box, 4):
                    raise ValueError(f"Box shape mismatch: {path}")
                if np.any(data["box_valid"] & np.any(
                    (data["boxes_cxcywh"] < 0) | (data["boxes_cxcywh"] > 1), axis=-1
                )):
                    raise ValueError(f"Normalized box outside [0,1]: {path}")

            mask_path = record.get("mask_rle")
            if mask_path:
                with gzip.open(resolve(project_root, mask_path), "rt", encoding="utf-8") as stream:
                    mask = json.load(stream)
                if mask["sequence_id"] != sequence_id:
                    raise ValueError(f"Mask bundle sequence mismatch: {path}")
                if len(mask["pose_frame_indices"]) != rgb_count:
                    raise ValueError(f"Mask bundle length mismatch: {path}")

            # Keep continuity diagnostics memory-bounded by sampling every 100th step.
            if xyz.shape[0] > 1:
                both = valid[1:] & valid[:-1]
                delta = np.linalg.norm(np.diff(xyz, axis=0), axis=-1) * float(record["fps"])
                sampled = delta[::100][both[::100]]
                if sampled.size:
                    speed_samples.append(sampled.astype(np.float32))
                    max_speed_mm_s = max(max_speed_mm_s, float(delta[both].max()))

            total_frames += expected_xyz[0]
            total_rgb_frames += rgb_count
            total_valid_points += int(valid.sum())

    speeds = np.concatenate(speed_samples) if speed_samples else np.empty(0)
    return {
        "manifest": manifest.as_posix(),
        "sequence_count": len(records),
        "frame_count": total_frames,
        "rgb_frame_count": total_rgb_frames,
        "valid_point_count": total_valid_points,
        "split_counts": dict(sorted(split_counts.items())),
        "condition_counts": dict(sorted(condition_counts.items())),
        "sampled_speed_mm_s_p50_p95_p99": (
            np.percentile(speeds, (50, 95, 99)).round(3).tolist() if speeds.size else []
        ),
        "max_speed_mm_s": round(max_speed_mm_s, 3),
    }


def main() -> None:
    args = parse_args()
    reports = [validate_manifest(path, args.project_root) for path in args.manifests]
    print(json.dumps(reports, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
