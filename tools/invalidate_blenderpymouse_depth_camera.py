#!/usr/bin/env python3
"""Invalidate one camera's inconsistent depth while preserving other supervision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--camera", default="Camera.008")
    parser.add_argument("--reason", default="failed_systematic_camera_depth_scale_check")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_pfm_shape(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.readline().strip()
        if header not in {b"Pf", b"PF"}:
            raise ValueError(f"Invalid PFM: {path}")
        width, height = (int(value) for value in handle.readline().split())
    return height, width


def write_zero_pfm(path: Path, shape: tuple[int, int]) -> None:
    with path.open("wb") as handle:
        handle.write(b"Pf\n")
        handle.write(f"{shape[1]} {shape[0]}\n".encode("ascii"))
        handle.write(b"-1.0\n")
        np.zeros(shape, dtype=np.float32).tofile(handle)


def update_keypoints(keypoints: list[dict]) -> None:
    for item in keypoints:
        item["surface_visible"] = False
        item["visible"] = False
        item["occluded_but_annotated"] = bool(item.get("in_frame", False))
        item["support_visible_count"] = 0
        item["support_min_depth_residual"] = None


def update_npz(path: Path, view_id: int) -> None:
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    arrays["surface_visible"][view_id] = False
    arrays["support_visible_count"][view_id] = 0
    available = arrays.get("depth_available", np.ones(len(arrays["view_ids"]), dtype=bool))
    available[view_id] = False
    arrays["depth_available"] = available
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def reset_cam_depth_range(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    for index in range(len(lines) - 1, -1, -1):
        values = lines[index].split()
        if len(values) == 4:
            try:
                depth_num = int(values[2])
            except ValueError:
                continue
            depth_min, depth_max = 0.1, 10.0
            interval = (depth_max - depth_min) / max(depth_num - 1, 1)
            lines[index] = f"{depth_min:.10g} {interval:.10g} {depth_num:d} {depth_max:.10g}"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return
    raise ValueError(f"Could not find depth range in camera file: {path}")


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    scene_roots = sorted(path for path in dataset.glob("frame_*") if path.is_dir())
    for scene_root in scene_roots:
        scene_meta_path = scene_root / "scene_meta.json"
        scene_meta = json.loads(scene_meta_path.read_text(encoding="utf-8"))
        for candidate in scene_meta["views"]:
            candidate_id = int(candidate["view_id"])
            candidate_annotation_path = scene_root / "annotations" / f"{candidate_id:08d}.json"
            candidate_annotation = json.loads(candidate_annotation_path.read_text(encoding="utf-8"))
            for target in (candidate, candidate_annotation):
                target.setdefault("depth_source_exr", str(Path(target["source_directory"]) / "depth_image.exr"))
                if target["camera"] != args.camera:
                    target["depth_available"] = True
                    target["depth_source_validation"] = "passed"
            write_json(candidate_annotation_path, candidate_annotation)
        matches = [view for view in scene_meta["views"] if view["camera"] == args.camera]
        if len(matches) != 1:
            raise ValueError(f"Expected one {args.camera} view in {scene_meta_path}, got {len(matches)}")
        view = matches[0]
        view_id = int(view["view_id"])
        depth_path = scene_root / Path(view["depth_pfm"].replace("\\", "/"))
        write_zero_pfm(depth_path, read_pfm_shape(depth_path))
        reset_cam_depth_range(scene_root / Path(view["cam"].replace("\\", "/")))

        annotation_path = scene_root / "annotations" / f"{view_id:08d}.json"
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        for target in (view, annotation):
            target["depth_available"] = False
            target.setdefault("depth_source_exr", str(Path(target["source_directory"]) / "depth_image.exr"))
            target["depth_source_validation"] = args.reason
            update_keypoints(target["keypoints"])
        write_json(annotation_path, annotation)
        write_json(scene_meta_path, scene_meta)
        update_npz(scene_root / "keypoints.npz", view_id)

        keypoints_path = scene_root / "keypoints.json"
        keypoints = json.loads(keypoints_path.read_text(encoding="utf-8"))
        for keypoint_view in keypoints["views"]:
            if int(keypoint_view["view_id"]) == view_id:
                update_keypoints(keypoint_view["keypoints"])
        write_json(keypoints_path, keypoints)

    metadata_path = dataset / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cameras = set(metadata.get("depth_invalid_cameras", []))
    cameras.add(args.camera)
    metadata["depth_invalid_cameras"] = sorted(cameras)
    metadata["depth_invalid_camera_policy"] = (
        "standard PFM is all-zero; RGB, mask, camera, and keypoints remain available"
    )
    metadata["depth_invalid_camera_reason"] = args.reason
    write_json(metadata_path, metadata)
    print(f"invalidated_camera_depth dataset={dataset} camera={args.camera} scenes={len(scene_roots)}")


if __name__ == "__main__":
    main()
