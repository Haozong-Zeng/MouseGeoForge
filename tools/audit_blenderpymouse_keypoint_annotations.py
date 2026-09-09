#!/usr/bin/env python3
"""Audit BlenderPyMouse keypoint files and independently reproject their 3D points."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-keypoints", type=int, default=29)
    parser.add_argument("--projection-tolerance", type=float, default=1e-3)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def reproject(
    world: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    camera = np.einsum("vij,kj->vki", extrinsics[:, :3, :3], world)
    camera += extrinsics[:, None, :3, 3]
    homogeneous = np.einsum("vij,vkj->vki", intrinsics, camera)
    return homogeneous[..., :2] / homogeneous[..., 2:3]


def audit_dataset(
    dataset_root: Path,
    expected_keypoints: int,
    projection_tolerance: float,
) -> dict:
    frame_roots = sorted(path for path in dataset_root.glob("frame_*") if path.is_dir())
    errors = []
    keypoint_visible = defaultdict(int)
    keypoint_in_frame = defaultdict(int)
    keypoint_instances = defaultdict(int)
    scene_count = 0
    view_count = 0
    visible_count = 0
    in_frame_count = 0
    instance_count = 0
    max_projection_error = 0.0

    for frame_root in frame_roots:
        npz_path = frame_root / "keypoints.npz"
        json_path = frame_root / "keypoints.json"
        scene_meta_path = frame_root / "scene_meta.json"
        missing = [str(path.name) for path in (npz_path, json_path, scene_meta_path) if not path.is_file()]
        if missing:
            errors.append({"scene": frame_root.name, "error": "missing files", "files": missing})
            continue

        scene_count += 1
        scene_meta = load_json(scene_meta_path)
        with np.load(npz_path, allow_pickle=False) as arrays:
            names = [str(value) for value in arrays["names"]]
            world = arrays["joint_world_xyz"].astype(np.float64)
            extrinsics = arrays["extrinsics"].astype(np.float64)
            intrinsics = arrays["intrinsics"].astype(np.float64)
            stored_uv = arrays["uv"].astype(np.float64)
            in_frame = arrays["in_frame"].astype(bool)
            visible = arrays["surface_visible"].astype(bool)

        if len(names) != expected_keypoints:
            errors.append(
                {
                    "scene": frame_root.name,
                    "error": "unexpected keypoint count",
                    "actual": len(names),
                    "expected": expected_keypoints,
                }
            )
        if len(set(names)) != len(names):
            errors.append({"scene": frame_root.name, "error": "duplicate keypoint names"})
        if not np.isfinite(world).all():
            errors.append({"scene": frame_root.name, "error": "non-finite world coordinates"})

        independent_uv = reproject(world, extrinsics, intrinsics)
        finite = np.isfinite(stored_uv).all(axis=-1) & np.isfinite(independent_uv).all(axis=-1)
        if finite.any():
            projection_error = float(np.abs(independent_uv[finite] - stored_uv[finite]).max())
            max_projection_error = max(max_projection_error, projection_error)
            if projection_error > projection_tolerance:
                errors.append(
                    {
                        "scene": frame_root.name,
                        "error": "reprojection mismatch",
                        "maximum_pixel_error": projection_error,
                    }
                )

        annotations = sorted((frame_root / "annotations").glob("*.json"))
        if len(annotations) != len(scene_meta.get("views", [])):
            errors.append(
                {
                    "scene": frame_root.name,
                    "error": "annotation/view count mismatch",
                    "annotations": len(annotations),
                    "views": len(scene_meta.get("views", [])),
                }
            )
        for annotation_path in annotations:
            annotation = load_json(annotation_path)
            keypoints = annotation.get("keypoints", [])
            if len(keypoints) != len(names):
                errors.append(
                    {
                        "scene": frame_root.name,
                        "view": annotation_path.stem,
                        "error": "annotation keypoint count mismatch",
                        "actual": len(keypoints),
                        "expected": len(names),
                    }
                )

        view_count += visible.shape[0]
        visible_count += int(visible.sum())
        in_frame_count += int(in_frame.sum())
        instance_count += int(visible.size)
        for keypoint_index, name in enumerate(names):
            keypoint_visible[name] += int(visible[:, keypoint_index].sum())
            keypoint_in_frame[name] += int(in_frame[:, keypoint_index].sum())
            keypoint_instances[name] += int(visible.shape[0])

    per_keypoint = {}
    for name in sorted(keypoint_instances):
        total = keypoint_instances[name]
        per_keypoint[name] = {
            "instances": total,
            "in_frame": keypoint_in_frame[name],
            "visible": keypoint_visible[name],
            "in_frame_rate": keypoint_in_frame[name] / total if total else 0.0,
            "visible_rate": keypoint_visible[name] / total if total else 0.0,
        }
    return {
        "dataset": dataset_root.name,
        "expected_scene_count": len(frame_roots),
        "audited_scene_count": scene_count,
        "view_count": view_count,
        "keypoint_instances": instance_count,
        "in_frame_instances": in_frame_count,
        "visible_instances": visible_count,
        "in_frame_rate": in_frame_count / instance_count if instance_count else 0.0,
        "visible_rate": visible_count / instance_count if instance_count else 0.0,
        "maximum_independent_reprojection_error_px": max_projection_error,
        "error_count": len(errors),
        "errors": errors,
        "per_keypoint": per_keypoint,
    }


def main() -> None:
    args = parse_args()
    datasets_root = args.datasets_root.resolve()
    results = [
        audit_dataset(dataset_root, args.expected_keypoints, args.projection_tolerance)
        for dataset_root in sorted(path for path in datasets_root.iterdir() if path.is_dir())
    ]
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets_root": str(datasets_root),
        "expected_keypoints": args.expected_keypoints,
        "projection_tolerance_px": args.projection_tolerance,
        "dataset_count": len(results),
        "scene_count": sum(item["audited_scene_count"] for item in results),
        "view_count": sum(item["view_count"] for item in results),
        "keypoint_instances": sum(item["keypoint_instances"] for item in results),
        "visible_instances": sum(item["visible_instances"] for item in results),
        "error_count": sum(item["error_count"] for item in results),
        "maximum_independent_reprojection_error_px": max(
            (item["maximum_independent_reprojection_error_px"] for item in results),
            default=0.0,
        ),
        "datasets": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        "keypoint_audit_complete",
        f"datasets={payload['dataset_count']}",
        f"scenes={payload['scene_count']}",
        f"views={payload['view_count']}",
        f"instances={payload['keypoint_instances']}",
        f"visible={payload['visible_instances']}",
        f"errors={payload['error_count']}",
        f"max_reprojection_error_px={payload['maximum_independent_reprojection_error_px']:.6g}",
    )
    if payload["error_count"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
