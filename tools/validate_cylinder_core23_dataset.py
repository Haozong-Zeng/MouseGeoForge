#!/usr/bin/env python3
"""Validate converted Cylinder Core-23 datasets and Camera.008 recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_pfm(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        if handle.readline().strip() != b"Pf":
            raise ValueError(f"Expected grayscale PFM: {path}")
        width, height = (int(value) for value in handle.readline().split())
        scale = float(handle.readline())
        endian = "<" if scale < 0 else ">"
        image = np.fromfile(handle, dtype=endian + "f4", count=width * height).reshape(height, width)
    return np.flipud(image)


def read_cam_extrinsic(path: Path) -> np.ndarray:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    start = lines.index("extrinsic") + 1
    return np.asarray([[float(value) for value in lines[start + row].split()] for row in range(4)])


def project(point: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    camera = extrinsic[:3, :3] @ point + extrinsic[:3, 3]
    homogeneous = intrinsic @ camera
    return homogeneous[:2] / homogeneous[2]


def validate_dataset(root: Path) -> dict:
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    errors = []
    view_count = 0
    visible_count = 0
    total_keypoints = 0
    all_nonzero = []
    reprojection_errors = []
    cam_file_errors = []
    npz_errors = []
    rotation_errors = []
    rotation_determinants = []
    camera8 = {
        "view_count": 0,
        "depth_available_count": 0,
        "pfm_nonzero": [],
        "translation_z": [],
        "visible_count": 0,
        "keypoint_count": 0,
        "support_min_depth_residual": [],
    }

    for frame in metadata["frame_numbers"]:
        scene_root = root / f"frame_{frame:06d}"
        keypoint_npz = np.load(scene_root / "keypoints.npz", allow_pickle=False)
        for view_id in range(8):
            stem = f"{view_id:08d}"
            annotation = json.loads((scene_root / "annotations" / f"{stem}.json").read_text(encoding="utf-8"))
            extrinsic = np.asarray(annotation["extrinsic"], dtype=np.float64)
            intrinsic = np.asarray(annotation["intrinsic"], dtype=np.float64)
            pfm = read_pfm(scene_root / "rendered_depth_maps" / f"{stem}.pfm")
            nonzero = int(np.count_nonzero(np.isfinite(pfm) & (pfm > 0)))
            all_nonzero.append(nonzero)
            view_count += 1

            rotation = extrinsic[:3, :3]
            rotation_errors.append(float(np.max(np.abs(rotation @ rotation.T - np.eye(3)))))
            rotation_determinants.append(float(np.linalg.det(rotation)))
            cam_extrinsic = read_cam_extrinsic(scene_root / "cams" / f"{stem}_cam.txt")
            cam_file_errors.append(float(np.max(np.abs(cam_extrinsic - extrinsic))))
            npz_errors.append(float(np.max(np.abs(keypoint_npz["extrinsics"][view_id] - extrinsic))))

            if bool(annotation["depth_available"]) != (nonzero > 0):
                errors.append(f"depth availability mismatch: frame={frame} view={view_id}")

            for keypoint in annotation["keypoints"]:
                total_keypoints += 1
                visible_count += int(bool(keypoint["visible"]))
                expected_uv = project(
                    np.asarray(keypoint["joint_world_xyz"], dtype=np.float64),
                    extrinsic,
                    intrinsic,
                )
                reprojection_errors.append(float(np.max(np.abs(expected_uv - np.asarray(keypoint["uv"])))))

            if view_id == 7:
                camera8["view_count"] += 1
                camera8["depth_available_count"] += int(bool(annotation["depth_available"]))
                camera8["pfm_nonzero"].append(nonzero)
                camera8["translation_z"].append(float(extrinsic[2, 3]))
                camera8["keypoint_count"] += len(annotation["keypoints"])
                camera8["visible_count"] += sum(int(bool(item["visible"])) for item in annotation["keypoints"])
                camera8["support_min_depth_residual"].extend(
                    float(item["support_min_depth_residual"])
                    for item in annotation["keypoints"]
                    if item["support_min_depth_residual"] is not None
                )

    residuals = np.asarray(camera8.pop("support_min_depth_residual"), dtype=np.float64)
    camera8_nonzero = camera8.pop("pfm_nonzero")
    camera8_z = camera8.pop("translation_z")
    camera8.update(
        {
            "depth_available_all": camera8["depth_available_count"] == camera8["view_count"],
            "pfm_nonzero_min": min(camera8_nonzero),
            "pfm_nonzero_max": max(camera8_nonzero),
            "extrinsic_translation_z_min": min(camera8_z),
            "extrinsic_translation_z_max": max(camera8_z),
            "visible_rate": camera8["visible_count"] / camera8["keypoint_count"],
            "support_residual_median": float(np.median(residuals)),
            "support_residual_p95": float(np.percentile(residuals, 95)),
        }
    )

    if metadata.get("camera_world_z_overrides", {}).get("Camera.008") != 650.0:
        errors.append("metadata Camera.008 Z override is not 650")
    if metadata.get("depth_invalid_cameras"):
        errors.append(f"unexpected invalid depth cameras: {metadata['depth_invalid_cameras']}")

    return {
        "dataset": root.name,
        "scene_count": metadata["scene_count"],
        "view_count": view_count,
        "total_keypoint_instances": total_keypoints,
        "visible_keypoint_instances": visible_count,
        "visibility_rate": visible_count / total_keypoints,
        "pfm_nonzero_min": min(all_nonzero),
        "pfm_nonzero_max": max(all_nonzero),
        "max_reprojection_error_px": max(reprojection_errors),
        "max_cam_file_extrinsic_error": max(cam_file_errors),
        "max_npz_extrinsic_error": max(npz_errors),
        "max_rotation_orthogonality_error": max(rotation_errors),
        "rotation_determinant_range": [min(rotation_determinants), max(rotation_determinants)],
        "camera8": camera8,
        "errors": errors,
    }


def main() -> None:
    args = parse_args()
    payload = {"datasets": [validate_dataset(root.resolve()) for root in args.dataset]}
    payload["valid"] = all(not item["errors"] for item in payload["datasets"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
