#!/usr/bin/env python3
"""Validate DANNCE mask/depth additions and exercise the training dataloader."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VGGT_ROOT = PROJECT_ROOT / "vggt-omega"
if str(VGGT_ROOT) not in sys.path:
    sys.path.insert(0, str(VGGT_ROOT))

from training.finetune_blendedmvs import (  # noqa: E402
    BlendedMVSDataset,
    load_scene,
    read_pfm,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    return parser.parse_args()


def validate(root: Path) -> dict:
    root = root.resolve()
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    scenes = [root / name for name in metadata["scene_folders"]]
    expected_views = int(metadata["views_per_scene"])
    expected_size = tuple(metadata["render_resolution"])
    errors: list[str] = []
    valid_counts = []
    mask_counts = []
    outside_counts = []
    depth_minima = []
    depth_maxima = []
    joint_differences = []

    for scene in scenes:
        with np.load(scene / "keypoints.npz", allow_pickle=False) as arrays:
            uv = arrays["uv"][:, 0]
            in_frame = arrays["in_frame"][:, 0]
        for view_id in range(expected_views):
            stem = f"{view_id:08d}"
            image_path = scene / "blended_images" / f"{stem}.jpg"
            mask_path = scene / "masks" / f"{stem}.png"
            depth_path = scene / "rendered_depth_maps" / f"{stem}.pfm"
            annotation_path = scene / "annotations" / f"{stem}.json"
            for path in (image_path, mask_path, depth_path, annotation_path):
                if not path.is_file():
                    errors.append(f"missing:{path}")
            if errors and not all(path.is_file() for path in (image_path, mask_path, depth_path, annotation_path)):
                continue
            with Image.open(image_path) as image:
                image_size = image.size
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            depth = read_pfm(depth_path, flip_y=True)
            if image_size != expected_size or mask.shape[::-1] != expected_size or depth.shape[::-1] != expected_size:
                errors.append(
                    f"size:{scene.name}/view{view_id}:rgb={image_size},mask={mask.shape[::-1]},depth={depth.shape[::-1]}"
                )
            mask_valid = mask > 127
            depth_valid = np.isfinite(depth) & (depth > 0)
            if np.any(depth_valid & (depth >= 10_000.0)):
                errors.append(f"implausible-depth:{scene.name}/view{view_id}")
            outside = depth_valid & ~mask_valid
            mask_counts.append(int(np.count_nonzero(mask_valid)))
            valid_counts.append(int(np.count_nonzero(depth_valid)))
            outside_counts.append(int(np.count_nonzero(outside)))
            if np.any(depth_valid):
                depth_minima.append(float(depth[depth_valid].min()))
                depth_maxima.append(float(depth[depth_valid].max()))
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            if annotation.get("depth_units") != "millimeter" or not annotation.get("depth_available"):
                errors.append(f"annotation-depth-metadata:{scene.name}/view{view_id}")
            points = annotation["keypoints"]
            for keypoint_index in np.flatnonzero(in_frame[view_id]):
                x, y = np.rint(uv[view_id, keypoint_index]).astype(int)
                if 0 <= x < depth.shape[1] and 0 <= y < depth.shape[0] and depth[y, x] > 0:
                    camera_z = float(points[int(keypoint_index)]["camera_depth"])
                    if camera_z > 0:
                        joint_differences.append(float(depth[y, x] - camera_z))

    loaded_scenes = [load_scene(scenes[index]) for index in sorted({0, len(scenes) // 2, len(scenes) - 1})]
    loader_dataset = BlendedMVSDataset(
        scenes=loaded_scenes,
        num_views=expected_views,
        image_width=128,
        image_height=128,
        samples_per_scene=1,
        seed=5706,
        use_masked_images=False,
        pfm_flip_y=True,
        canonicalize_camera=False,
        scale_mode="raw",
        top_srcs=8,
        load_keypoints=True,
    )
    sample = loader_dataset[0]
    if not bool(sample["depth_available"].all()):
        errors.append("dataloader-depth-unavailable")
    if not bool(sample["mask_valid"].all()):
        errors.append("dataloader-mask-unavailable")
    differences = np.asarray(joint_differences, dtype=np.float64)
    payload = {
        "status": "complete" if not errors else "failed",
        "dataset": str(root),
        "scene_count": len(scenes),
        "view_count": len(scenes) * expected_views,
        "resolution": list(expected_size),
        "depth_format": "PFM float32",
        "depth_units": "millimeter",
        "mask_file_count": len(mask_counts),
        "depth_file_count": len(valid_counts),
        "mask_pixel_range": [min(mask_counts), max(mask_counts)],
        "depth_valid_pixel_range": [min(valid_counts), max(valid_counts)],
        "depth_outside_mask_pixels": int(sum(outside_counts)),
        "depth_range_mm": [min(depth_minima), max(depth_maxima)],
        "surface_minus_internal_joint_mm": {
            "sample_count": int(len(differences)),
            "median": float(np.median(differences)),
            "median_absolute": float(np.median(np.abs(differences))),
            "p90_absolute": float(np.percentile(np.abs(differences), 90)),
        },
        "dataloader": {
            "images": list(sample["images"].shape),
            "depths": list(sample["depths"].shape),
            "body_masks": list(sample["body_masks"].shape),
            "instance_masks": list(sample["masks"].shape),
            "max_instances": loader_dataset.max_instances,
            "all_depth_views_available": bool(sample["depth_available"].all()),
            "all_instance_masks_available": bool(sample["mask_valid"].all()),
        },
        "errors": errors,
    }
    (root / "pseudo_depth_validation.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    args = parse_args()
    reports = [validate(path) for path in args.dataset]
    print(json.dumps(reports, indent=2, ensure_ascii=False))
    if any(report["status"] != "complete" for report in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
