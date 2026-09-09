#!/usr/bin/env python3
"""Validate the completed dual-mouse Homecage BlenderPyMouse dataset."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy
import numpy as np
from PIL import Image


CAMERAS = (
    "HomecageCam_Top",
    "HomecageCam_SideNegX",
    "HomecageCam_SideNegY",
    "HomecageCam_SidePosY",
)


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def pfm_header(path: Path) -> tuple[int, int, int]:
    with path.open("rb") as handle:
        magic = handle.readline().decode("ascii").strip()
        width, height = map(int, handle.readline().decode("ascii").split())
        scale = float(handle.readline().decode("ascii"))
    if magic != "Pf" or scale == 0:
        raise ValueError(f"Invalid PFM header: {path}")
    return width, height, path.stat().st_size


def main() -> None:
    args = args_after_dash()
    if len(args) != 2:
        raise SystemExit("Usage: blender -b scene.blend -P validate_dual_homecage_blenderpymouse.py -- dataset report.json")
    root = Path(args[0]).resolve()
    report_path = Path(args[1]).resolve()
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    counts = {
        "scenes": 0,
        "rgb": 0,
        "depth_pfm": 0,
        "depth_preview": 0,
        "camera_files": 0,
        "annotations": 0,
        "instance_masks": 0,
        "combined_masks": 0,
        "keypoint_npz": 0,
    }
    overlap_pixels = 0
    empty_masks = 0
    render_resolution = metadata.get("render_resolution", [1280, 720])
    expected_shape = (int(render_resolution[1]), int(render_resolution[0]))
    frame_numbers = [int(frame) for frame in metadata.get("frame_numbers", [])]
    if not frame_numbers:
        first_frame, last_frame = metadata.get("frames", [1, 200])
        frame_numbers = list(range(int(first_frame), int(last_frame) + 1))
    expected_scene_count = len(frame_numbers)
    for frame in frame_numbers:
        scene = root / f"frame_{frame:06d}"
        if not scene.is_dir():
            errors.append(f"missing scene {scene.name}")
            continue
        counts["scenes"] += 1
        for view_id, camera_name in enumerate(CAMERAS):
            stem = f"{view_id:08d}"
            files = {
                "rgb": scene / "blended_images" / f"{stem}.jpg",
                "depth_pfm": scene / "rendered_depth_maps" / f"{stem}.pfm",
                "depth_preview": scene / "rendered_depth_maps" / f"{stem}_preview.png",
                "camera_files": scene / "cams" / f"{stem}_cam.txt",
                "annotations": scene / "annotations" / f"{stem}.json",
                "combined_masks": scene / "masks" / f"{stem}.png",
            }
            for key, path in files.items():
                if not path.is_file() or path.stat().st_size <= 0:
                    errors.append(f"missing/empty {path}")
                else:
                    counts[key] += 1
            if files["depth_pfm"].is_file():
                width, height, size = pfm_header(files["depth_pfm"])
                if (height, width) != expected_shape or size < width * height * 4:
                    errors.append(f"bad depth shape/size {files['depth_pfm']}")
            if files["annotations"].is_file():
                annotation = json.loads(files["annotations"].read_text(encoding="utf-8"))
                if annotation.get("camera") != camera_name or len(annotation.get("instances", [])) != 2:
                    errors.append(f"bad dual annotation {files['annotations']}")
            masks = []
            for instance_id in range(2):
                path = scene / "masks" / f"{stem}_instance{instance_id:02d}.png"
                if not path.is_file() or path.stat().st_size <= 0:
                    errors.append(f"missing/empty {path}")
                    continue
                counts["instance_masks"] += 1
                mask = np.asarray(Image.open(path).convert("L")) > 0
                if mask.shape != expected_shape:
                    errors.append(f"bad mask shape {path}: {mask.shape}")
                if not mask.any():
                    empty_masks += 1
                    errors.append(f"empty instance mask {path}")
                masks.append(mask)
            if len(masks) == 2:
                overlap_pixels += int(np.logical_and(masks[0], masks[1]).sum())
        npz_path = scene / "keypoints.npz"
        if not npz_path.is_file() or npz_path.stat().st_size <= 0:
            errors.append(f"missing/empty {npz_path}")
        else:
            counts["keypoint_npz"] += 1
            with np.load(npz_path, allow_pickle=False) as labels:
                expected = {
                    "joint_world_xyz": (2, 23, 3),
                    "uv": (4, 2, 23, 2),
                    "in_frame": (4, 2, 23),
                    "surface_visible": (4, 2, 23),
                    "intrinsics": (4, 3, 3),
                    "extrinsics": (4, 4, 4),
                }
                for key, shape in expected.items():
                    if key not in labels or labels[key].shape != shape:
                        errors.append(f"bad NPZ {key} shape in {npz_path}")

    for camera_name in CAMERAS:
        camera = bpy.data.objects.get(camera_name)
        if camera is None or camera.type != "CAMERA" or camera.data.type != "PERSP":
            errors.append(f"camera is not perspective: {camera_name}")
    source_indices = metadata.get("source_frame_indices", [])
    if len(source_indices) != expected_scene_count or len(set(source_indices)) != expected_scene_count:
        errors.append(
            f"source_frame_indices are not {expected_scene_count} unique source poses"
        )
    if metadata.get("side_depth_wall_policy") != "ignore_only_wall_between_camera_and_mouse":
        errors.append("wrong side depth wall policy")
    if metadata.get("top_view_depth_wall_policy") != "full_shell_retained":
        errors.append("wrong top depth wall policy")
    expected_counts = {
        "scenes": expected_scene_count,
        "rgb": expected_scene_count * len(CAMERAS),
        "depth_pfm": expected_scene_count * len(CAMERAS),
        "depth_preview": expected_scene_count * len(CAMERAS),
        "camera_files": expected_scene_count * len(CAMERAS),
        "annotations": expected_scene_count * len(CAMERAS),
        "instance_masks": expected_scene_count * len(CAMERAS) * 2,
        "combined_masks": expected_scene_count * len(CAMERAS),
        "keypoint_npz": expected_scene_count,
    }
    for key, expected in expected_counts.items():
        if counts[key] != expected:
            errors.append(f"count mismatch {key}: {counts[key]} != {expected}")
    if overlap_pixels:
        errors.append(f"instance masks overlap at {overlap_pixels} pixels")

    payload = {
        "status": "complete" if not errors else "failed",
        "dataset_root": str(root),
        "format": metadata.get("format"),
        "counts": counts,
        "resolution": render_resolution,
        "camera_model": "perspective",
        "side_depth_wall_policy": metadata.get("side_depth_wall_policy"),
        "top_view_depth_wall_policy": metadata.get("top_view_depth_wall_policy"),
        "unique_source_pose_count": len(set(source_indices)),
        "instance_mask_overlap_pixels": overlap_pixels,
        "empty_instance_masks": empty_masks,
        "errors": errors[:100],
        "error_count": len(errors),
    }
    write_tmp = report_path.with_suffix(report_path.suffix + ".tmp")
    write_tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_tmp.replace(report_path)
    print(json.dumps(payload, indent=2), flush=True)
    if errors:
        raise RuntimeError(f"Dataset validation failed with {len(errors)} errors")


if __name__ == "__main__":
    main()
