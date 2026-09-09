#!/usr/bin/env python3
"""Validate Blender-derived keypoint NPZ files in legacy PseudoSynthetic datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FIELDS = {
    "format",
    "frame",
    "names",
    "semantic_names",
    "view_ids",
    "camera_names",
    "intrinsics",
    "extrinsics",
    "joint_world_xyz",
    "surface_world_xyz",
    "support_vertex_ids",
    "support_world_xyz",
    "joint_camera_xyz",
    "uv",
    "in_frame",
    "surface_visible",
    "support_visible_count",
}


def frame_number(path: Path) -> int:
    return int(path.name.removeprefix("frame_"))


def projected_uv(world: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    camera = (extrinsic[:, :3, :3] @ world.T[None]).transpose(0, 2, 1)
    camera += extrinsic[:, None, :3, 3]
    homogeneous = (intrinsic @ camera.transpose(0, 2, 1)).transpose(0, 2, 1)
    return homogeneous[..., :2] / homogeneous[..., 2:3]


def audit_dataset(root: Path) -> dict:
    scenes = sorted(root.glob("frame_*"), key=frame_number)
    errors: list[str] = []
    max_projection_error = 0.0
    visible_count = 0
    for scene in scenes:
        npz_path = scene / "keypoints.npz"
        json_path = scene / "keypoints.json"
        if not npz_path.is_file() or not json_path.is_file():
            errors.append(f"{scene.name}: missing keypoints.npz or keypoints.json")
            continue
        with np.load(npz_path, allow_pickle=False) as arrays:
            missing = REQUIRED_FIELDS.difference(arrays.files)
            if missing:
                errors.append(f"{scene.name}: missing fields {sorted(missing)}")
                continue
            frame = int(arrays["frame"])
            names = arrays["names"]
            views = arrays["view_ids"]
            world = arrays["joint_world_xyz"]
            uv = arrays["uv"]
            if frame != frame_number(scene):
                errors.append(f"{scene.name}: NPZ frame is {frame}")
            expected = (len(views), len(names))
            shapes = {
                "intrinsics": (len(views), 3, 3),
                "extrinsics": (len(views), 4, 4),
                "joint_world_xyz": (len(names), 3),
                "surface_world_xyz": (len(names), 3),
                "joint_camera_xyz": (*expected, 3),
                "uv": (*expected, 2),
                "in_frame": expected,
                "surface_visible": expected,
            }
            for field, expected_shape in shapes.items():
                if arrays[field].shape != expected_shape:
                    errors.append(
                        f"{scene.name}: {field} shape {arrays[field].shape}, expected {expected_shape}"
                    )
            if not np.isfinite(world).all():
                errors.append(f"{scene.name}: non-finite joint_world_xyz")
            if not np.isfinite(uv).all():
                errors.append(f"{scene.name}: non-finite uv")
            reprojection = projected_uv(world, arrays["extrinsics"], arrays["intrinsics"])
            error = float(np.linalg.norm(reprojection - uv, axis=-1).max())
            max_projection_error = max(max_projection_error, error)
            if error > 1e-2:
                errors.append(f"{scene.name}: maximum reprojection error {error:.6g}px")
            visible_count += int(arrays["surface_visible"].sum())

    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    annotation_metadata = metadata.get("keypoint_annotations", {})
    if annotation_metadata.get("complete_scene_count") != len(scenes):
        errors.append("metadata complete_scene_count does not match scene count")
    return {
        "dataset": root.name,
        "scene_count": len(scenes),
        "npz_count": sum((scene / "keypoints.npz").is_file() for scene in scenes),
        "keypoint_count": int(annotation_metadata.get("keypoint_count", 0)),
        "view_count": int(annotation_metadata.get("complete_view_count", 0)),
        "visible_keypoint_instances": visible_count,
        "max_reprojection_error_px": max_projection_error,
        "errors": errors,
    }


def audit_pair(original: Path, transferred: Path) -> dict:
    errors: list[str] = []
    compared = 0
    fields = (
        "names",
        "view_ids",
        "intrinsics",
        "extrinsics",
        "joint_world_xyz",
        "surface_world_xyz",
        "uv",
        "in_frame",
        "surface_visible",
    )
    for original_scene in sorted(original.glob("frame_*"), key=frame_number):
        transferred_scene = transferred / original_scene.name
        with np.load(original_scene / "keypoints.npz", allow_pickle=False) as left, np.load(
            transferred_scene / "keypoints.npz", allow_pickle=False
        ) as right:
            for field in fields:
                if not np.array_equal(left[field], right[field]):
                    errors.append(f"{original_scene.name}: {field} differs")
            compared += 1
    return {
        "trial": original.name.removesuffix("_PseudoSynthetic"),
        "scene_pairs_compared": compared,
        "geometry_identical": not errors,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "BlenderPyMouse",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=(
            PROJECT_ROOT
            / "dataset"
            / "BlenderPyMouse"
            / "_annotation_audit"
            / "pseudosynthetic_keypoints_audit.json"
        ),
    )
    args = parser.parse_args()
    datasets = sorted(
        path
        for path in args.root.glob("*PseudoSynthetic*")
        if path.is_dir()
    )
    rows = [audit_dataset(path) for path in datasets]
    pairs = []
    for original in datasets:
        if not original.name.endswith("_PseudoSynthetic"):
            continue
        transferred = original.with_name(original.name + "Transfered")
        pairs.append(audit_pair(original, transferred))
    error_count = sum(len(row["errors"]) for row in rows) + sum(
        len(pair["errors"]) for pair in pairs
    )
    payload = {
        "status": "complete" if error_count == 0 else "failed",
        "dataset_count": len(rows),
        "scene_count": sum(row["scene_count"] for row in rows),
        "npz_count": sum(row["npz_count"] for row in rows),
        "error_count": error_count,
        "datasets": rows,
        "original_transferred_pairs": pairs,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    if error_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
