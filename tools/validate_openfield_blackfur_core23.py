#!/usr/bin/env python3
"""Validate the Openfield BlackFur CranialWindow Core-23 dataset."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_pfm(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        if handle.readline().strip() != b"Pf":
            raise ValueError(f"Expected grayscale PFM: {path}")
        width, height = (int(value) for value in handle.readline().split())
        scale = float(handle.readline())
        endian = "<" if scale < 0 else ">"
        data = np.fromfile(handle, dtype=endian + "f4", count=width * height).reshape(height, width)
    return np.flipud(data)


def read_cam_extrinsic(path: Path) -> np.ndarray:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    start = lines.index("extrinsic") + 1
    return np.asarray([[float(value) for value in lines[start + row].split()] for row in range(4)])


def project(point: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    camera = extrinsic[:3, :3] @ point + extrinsic[:3, 3]
    homogeneous = intrinsic @ camera
    return homogeneous[:2] / homogeneous[2]


def main() -> None:
    args = parse_args()
    root = args.dataset.resolve()
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    errors = []
    total_keypoints = 0
    visible_keypoints = 0
    in_frame_keypoints = 0
    reprojection_errors = []
    legacy_mean_errors = []
    legacy_max_errors = []
    mask_pixels = []
    mask_bbox_areas = []
    pfm_nonzero = []
    cam_file_errors = []
    npz_errors = []
    rotation_errors = []
    rotation_determinants = []
    intrinsics = []
    per_keypoint = defaultdict(lambda: {"instances": 0, "visible": 0, "in_frame": 0})

    for frame in metadata["frame_numbers"]:
        scene_root = root / f"frame_{frame:06d}"
        annotation_path = scene_root / "annotations" / "00000000.json"
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        npz = np.load(scene_root / "keypoints.npz", allow_pickle=False)
        intrinsic = np.asarray(annotation["intrinsic"], dtype=np.float64)
        extrinsic = np.asarray(annotation["extrinsic"], dtype=np.float64)
        intrinsics.append(intrinsic)

        pfm = read_pfm(scene_root / "rendered_depth_maps" / "00000000.pfm")
        nonzero = int(np.count_nonzero(np.isfinite(pfm) & (pfm > 0)))
        pfm_nonzero.append(nonzero)
        if not annotation["depth_available"] or nonzero == 0:
            errors.append(f"missing valid depth at frame {frame}")

        mask = np.asarray(Image.open(scene_root / "masks" / "00000000.png").convert("L")) > 0
        count = int(mask.sum())
        mask_pixels.append(count)
        ys, xs = np.where(mask)
        if not len(xs):
            errors.append(f"empty mask at frame {frame}")
            mask_bbox_areas.append(0)
        else:
            mask_bbox_areas.append(int((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)))

        rotation = extrinsic[:3, :3]
        rotation_errors.append(float(np.max(np.abs(rotation @ rotation.T - np.eye(3)))))
        rotation_determinants.append(float(np.linalg.det(rotation)))
        cam_extrinsic = read_cam_extrinsic(scene_root / "cams" / "00000000_cam.txt")
        cam_file_errors.append(float(np.max(np.abs(cam_extrinsic - extrinsic))))
        npz_errors.append(float(np.max(np.abs(npz["extrinsics"][0] - extrinsic))))

        legacy_mean_errors.append(float(annotation["legacy_projection_error_mean_px"]))
        legacy_max_errors.append(float(annotation["legacy_projection_error_max_px"]))
        for keypoint in annotation["keypoints"]:
            total_keypoints += 1
            visible = int(bool(keypoint["visible"]))
            in_frame = int(bool(keypoint["in_frame"]))
            visible_keypoints += visible
            in_frame_keypoints += in_frame
            summary = per_keypoint[keypoint["name"]]
            summary["instances"] += 1
            summary["visible"] += visible
            summary["in_frame"] += in_frame
            expected_uv = project(
                np.asarray(keypoint["joint_world_xyz"], dtype=np.float64),
                extrinsic,
                intrinsic,
            )
            reprojection_errors.append(float(np.max(np.abs(expected_uv - np.asarray(keypoint["uv"])))))

        pair_lines = (scene_root / "cams" / "pair.txt").read_text(encoding="utf-8").splitlines()
        if pair_lines != ["1", "0", "0"]:
            errors.append(f"unexpected single-view pair file at frame {frame}: {pair_lines}")

    intrinsics = np.stack(intrinsics)
    for summary in per_keypoint.values():
        summary["visible_rate"] = summary["visible"] / summary["instances"]
        summary["in_frame_rate"] = summary["in_frame"] / summary["instances"]

    payload = {
        "dataset": root.name,
        "scene_count": metadata["scene_count"],
        "view_count": metadata["scene_count"],
        "camera": metadata["cameras"][0],
        "total_keypoint_instances": total_keypoints,
        "in_frame_keypoint_instances": in_frame_keypoints,
        "visible_keypoint_instances": visible_keypoints,
        "visibility_rate": visible_keypoints / total_keypoints,
        "max_independent_reprojection_error_px": max(reprojection_errors),
        "legacy_projection_mean_error_px": float(np.mean(legacy_mean_errors)),
        "legacy_projection_median_error_px": float(np.median(legacy_mean_errors)),
        "legacy_projection_max_error_px": max(legacy_max_errors),
        "intrinsic_ranges": {
            "fx": [float(intrinsics[:, 0, 0].min()), float(intrinsics[:, 0, 0].max())],
            "fy": [float(intrinsics[:, 1, 1].min()), float(intrinsics[:, 1, 1].max())],
            "cx": [float(intrinsics[:, 0, 2].min()), float(intrinsics[:, 0, 2].max())],
            "cy": [float(intrinsics[:, 1, 2].min()), float(intrinsics[:, 1, 2].max())],
        },
        "mask_pixel_range": [min(mask_pixels), max(mask_pixels)],
        "mask_pixel_median": float(np.median(mask_pixels)),
        "mask_bbox_area_range": [min(mask_bbox_areas), max(mask_bbox_areas)],
        "pfm_nonzero_range": [min(pfm_nonzero), max(pfm_nonzero)],
        "max_cam_file_extrinsic_error": max(cam_file_errors),
        "max_npz_extrinsic_error": max(npz_errors),
        "max_rotation_orthogonality_error": max(rotation_errors),
        "rotation_determinant_range": [min(rotation_determinants), max(rotation_determinants)],
        "per_keypoint": dict(sorted(per_keypoint.items())),
        "errors": errors,
        "valid": not errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "per_keypoint"}, indent=2))


if __name__ == "__main__":
    main()
