#!/usr/bin/env python3
"""Build reference manifests for real-partial and synthetic-full Core23 data.

The builder intentionally does not duplicate source images. Manifest paths are
relative to the MouseGeoForge project root and can therefore mix large datasets
without another multi-gigabyte copy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE23 = (
    "nose",
    "head_center",
    "left_ear",
    "right_ear",
    "neck",
    "spine_center",
    "lumbar_spine",
    "tail_base",
    "tail_q1",
    "tail_q2",
    "tail_q3",
    "tail_q4",
    "tail_q5",
    "tail_tip",
    "left_forepaw_center",
    "right_forepaw_center",
    "left_hindpaw_center",
    "right_hindpaw_center",
    "left_elbow",
    "right_elbow",
    "left_knee",
    "right_knee",
    "abdomen_center",
)
CRIM13_TO_CORE23 = {
    "nose": "nose",
    "right_ear": "right_ear",
    "left_ear": "left_ear",
    "top_of_neck": "neck",
    "right_rear_knee": "right_knee",
    "left_rear_knee": "left_knee",
    "base_of_tail": "tail_base",
}
MICE = ("black_mouse", "white_mouse")


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def stable_real_split(domain: str, session: str) -> str:
    if domain == "OOD":
        return "test"
    bucket = int(hashlib.sha1(session.encode("utf-8")).hexdigest()[:8], 16) % 10
    return "val" if bucket < 2 else "train"


def empty_points() -> tuple[list[list[float]], list[bool]]:
    return [[0.0, 0.0] for _ in CORE23], [False for _ in CORE23]


def weak_box(points: list[list[float]], valid: list[bool], padding: float = 0.05) -> list[float]:
    selected = np.asarray([point for point, keep in zip(points, valid, strict=True) if keep])
    lower = np.maximum(selected.min(axis=0) - padding, 0.0)
    upper = np.minimum(selected.max(axis=0) + padding, 1.0)
    size = np.maximum(upper - lower, 1e-4)
    center = 0.5 * (lower + upper)
    return [float(center[0]), float(center[1]), float(size[0]), float(size[1])]


def build_crim13_records(source_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    dimension_cache: dict[str, tuple[int, int]] = {}
    for domain in ("InD", "OOD"):
        csv_path = source_root / f"labels_{domain}.csv"
        with csv_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            scorer = next(reader)
            bodyparts = next(reader)
            coordinates = next(reader)
            if not (len(scorer) == len(bodyparts) == len(coordinates)):
                raise ValueError(f"Malformed three-row header in {csv_path}")
            columns = list(zip(bodyparts[1:], coordinates[1:], strict=True))
            for row in reader:
                if not row:
                    continue
                image_path = source_root / row[0]
                session = image_path.parent.name
                if session not in dimension_cache:
                    with Image.open(image_path) as image:
                        dimension_cache[session] = image.size
                width, height = dimension_cache[session]
                values = {
                    column: float(value)
                    for column, value in zip(columns, row[1:], strict=True)
                }
                all_xy: list[list[list[float]]] = []
                all_valid: list[list[bool]] = []
                all_boxes: list[list[float]] = []
                for mouse in MICE:
                    xy, valid = empty_points()
                    for source_name, target_name in CRIM13_TO_CORE23.items():
                        target_index = CORE23.index(target_name)
                        xy[target_index] = [
                            values[(f"{mouse}_{source_name}", "x")] / width,
                            values[(f"{mouse}_{source_name}", "y")] / height,
                        ]
                        valid[target_index] = True
                    all_xy.append(xy)
                    all_valid.append(valid)
                    all_boxes.append(weak_box(xy, valid))
                frame_name = image_path.stem
                records.append(
                    {
                        "schema": "mousegeoforge_instance_core23_v1",
                        "sample_id": f"crim13:{domain}:{session}:{frame_name}",
                        "split": stable_real_split(domain, session),
                        "source": "lightningpose_crim13",
                        "domain": domain,
                        "supervision": "real_partial_2d",
                        "identity_scope": f"crim13:{domain}:{session}",
                        "image_size": [width, height],
                        "views": [
                            {
                                "view_id": 0,
                                "camera_name": "top_uncalibrated",
                                "image": relative(image_path),
                                "channels": 1,
                                "intrinsic": None,
                                "extrinsic": None,
                                "depth": None,
                            }
                        ],
                        "instance_ids": [0, 1],
                        "instance_names": list(MICE),
                        "boxes_cxcywh": [all_boxes],
                        "box_valid": [[False, False]],
                        "box_source": "seven_keypoint_hull_with_padding_weak_only",
                        "mask_paths": [[None, None]],
                        "mask_valid": [[False, False]],
                        "keypoint_names": list(CORE23),
                        "keypoint_xy": [[all_xy[0], all_xy[1]]],
                        "keypoint_valid": [[all_valid[0], all_valid[1]]],
                        "keypoint_visible": [[all_valid[0], all_valid[1]]],
                        "keypoint_visibility_valid": [
                            [[False] * len(CORE23), [False] * len(CORE23)]
                        ],
                        "keypoint_world_xyz": None,
                        "world_valid": [[False] * len(CORE23) for _ in MICE],
                    }
                )
    return records


def mask_box(path: Path, width: int, height: int) -> list[float]:
    with Image.open(path) as image:
        bounds = image.getbbox()
    if bounds is None:
        raise ValueError(f"Empty synthetic mouse mask: {path}")
    left, top, right, bottom = bounds
    center_x = (left + right) / (2.0 * width)
    center_y = (top + bottom) / (2.0 * height)
    return [center_x, center_y, (right - left) / width, (bottom - top) / height]


def synthetic_split(frame: int, total: int) -> str:
    train_end = round(total * 0.8)
    val_end = round(total * 0.9)
    if frame <= train_end:
        return "train"
    return "val" if frame <= val_end else "test"


def build_homecage_records(source_root: Path) -> list[dict[str, Any]]:
    metadata = json.loads((source_root / "metadata.json").read_text(encoding="utf-8"))
    width, height = metadata["render_resolution"]
    scene_dirs = sorted(source_root.glob("frame_*"))
    records = []
    for scene_dir in scene_dirs:
        with np.load(scene_dir / "keypoints.npz", allow_pickle=False) as labels:
            names = tuple(str(name) for name in labels["names"].tolist())
            if names != CORE23:
                raise ValueError(f"Core23 mismatch in {scene_dir}: {names}")
            frame = int(labels["frame"])
            view_ids = labels["view_ids"].astype(int).tolist()
            camera_names = labels["camera_names"].astype(str).tolist()
            intrinsics = labels["intrinsics"].astype(float).tolist()
            extrinsics = labels["extrinsics"].astype(float).tolist()
            xy = labels["uv"].astype(float)
            xy[..., 0] /= width
            xy[..., 1] /= height
            in_frame = labels["in_frame"].astype(bool)
            visible = labels["surface_visible"].astype(bool)
            world = labels["joint_world_xyz"].astype(float).tolist()
        views = []
        boxes = []
        masks = []
        keypoint_xy = []
        keypoint_valid = []
        keypoint_visible = []
        visibility_valid = []
        for array_index, (view_id, camera_name) in enumerate(
            zip(view_ids, camera_names, strict=True)
        ):
            stem = f"{view_id:08d}"
            image_path = scene_dir / "blended_images" / f"{stem}.jpg"
            mask_path = scene_dir / "masks" / f"{stem}.png"
            depth_path = scene_dir / "rendered_depth_maps" / f"{stem}.pfm"
            views.append(
                {
                    "view_id": view_id,
                    "camera_name": camera_name,
                    "image": relative(image_path),
                    "channels": 3,
                    "intrinsic": intrinsics[array_index],
                    "extrinsic": extrinsics[array_index],
                    "depth": relative(depth_path),
                    "depth_unit": "millimeter",
                }
            )
            boxes.append([mask_box(mask_path, width, height)])
            masks.append([relative(mask_path)])
            keypoint_xy.append([xy[array_index].tolist()])
            keypoint_valid.append([in_frame[array_index].tolist()])
            keypoint_visible.append([visible[array_index].tolist()])
            visibility_valid.append([in_frame[array_index].tolist()])
        source_indices = metadata.get("source_video_sampling", {}).get(
            "source_frame_indices", []
        )
        source_index = source_indices[frame - 1] if len(source_indices) >= frame else None
        records.append(
            {
                "schema": "mousegeoforge_instance_core23_v1",
                "sample_id": f"homecage_blackfur:{frame:06d}",
                "split": synthetic_split(frame, len(scene_dirs)),
                "source": "homecage_blackfur_blender",
                "supervision": "synthetic_full_multiview_3d",
                "identity_scope": "homecage_blackfur_sequence_001",
                "source_frame_index": source_index,
                "image_size": [width, height],
                "views": views,
                "instance_ids": [0],
                "instance_names": ["black_mouse"],
                "boxes_cxcywh": boxes,
                "box_valid": [[True] for _ in views],
                "box_source": "instance_mask_exact",
                "mask_paths": masks,
                "mask_valid": [[True] for _ in views],
                "keypoint_names": list(CORE23),
                "keypoint_xy": keypoint_xy,
                "keypoint_valid": keypoint_valid,
                "keypoint_visible": keypoint_visible,
                "keypoint_visibility_valid": visibility_valid,
                "keypoint_world_xyz": [world],
                "world_valid": [[True] * len(CORE23)],
                "world_unit": "millimeter",
            }
        )
    return records


def split_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        split: [record for record in records if record["split"] == split]
        for split in ("train", "val", "test")
    }


def write_dataset(
    output: Path,
    records: list[dict[str, Any]],
    *,
    description: str,
    source_roots: list[Path],
) -> dict[str, int]:
    by_split = split_records(records)
    counts = {
        split: write_jsonl(output / f"{split}.jsonl", split_records)
        for split, split_records in by_split.items()
    }
    write_json(
        output / "metadata.json",
        {
            "schema": "mousegeoforge_instance_core23_v1",
            "description": description,
            "project_root": str(PROJECT_ROOT),
            "reference_only": True,
            "source_roots": [relative(path) for path in source_roots],
            "keypoint_names": list(CORE23),
            "counts": counts,
            "source_counts": dict(Counter(record["source"] for record in records)),
            "supervision_counts": dict(
                Counter(record["supervision"] for record in records)
            ),
        },
    )
    return counts


def write_readme(path: Path) -> None:
    content = """# Homecage + Lightning Pose Core23 datasets

These manifests apply the PyMouseLifter principle: real 2D labels define the target domain
and social-pose distribution, while Blender supplies complete geometry and unbiased dense
labels. Images are referenced in place; this directory does not duplicate source media.

Datasets:

- `LightningPose-CRIM13-Core23-Partial2D-v1`: two mice, one uncalibrated top view,
  seven mapped Core23 points per mouse. Boxes are weak point hulls and are explicitly
  marked invalid for box loss. Mask, visibility, camera, depth, and 3D losses are disabled.
- `Homecage-BlackFur-Core23-Full3D-v1`: one black mouse, four calibrated views,
  complete Core23, exact mask-derived boxes, instance masks, metric depth and 3D points.
- `Homecage-LightningPose-Core23-Hybrid-v1`: combined manifests plus smaller pilot
  manifests for fast loader/training checks.

CRIM13 mapping:

`nose -> nose`, `left/right_ear -> left/right_ear`, `top_of_neck -> neck`,
`left/right_rear_knee -> left/right_knee`, `base_of_tail -> tail_base`.

Splits are group-safe: CRIM13 InD sessions are assigned wholly to train or validation;
OOD sessions are test only. The single synthetic sequence uses contiguous 80/10/10 frame
blocks, which is useful for regression checks but not an independent-animal benchmark.
"""
    path.write_text(content, encoding="utf-8")


def build_pilot(by_split: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    limits = {
        "train": {"lightningpose_crim13": 64, "homecage_blackfur_blender": 32},
        "val": {"lightningpose_crim13": 24, "homecage_blackfur_blender": 12},
        "test": {"lightningpose_crim13": 32, "homecage_blackfur_blender": 8},
    }
    result: dict[str, list[dict[str, Any]]] = {}
    for split, records in by_split.items():
        selected = []
        for source, limit in limits[split].items():
            candidates = [record for record in records if record["source"] == source]
            selected.extend(candidates[:limit])
        result[split] = sorted(selected, key=lambda record: record["sample_id"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--crim13-root",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "LightningPose-CRIM13" / "crim13",
    )
    parser.add_argument(
        "--homecage-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "dataset"
            / "BlenderPyMouse"
            / "Homecage-BlackFur-KeypointMoseq-Core23-Sample10-Frames200-399"
        ),
    )
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "dataset")
    args = parser.parse_args()

    real_records = build_crim13_records(args.crim13_root)
    synthetic_records = build_homecage_records(args.homecage_root)
    real_output = args.dataset_root / "LightningPose-CRIM13-Core23-Partial2D-v1"
    synthetic_output = args.dataset_root / "Homecage-BlackFur-Core23-Full3D-v1"
    hybrid_output = args.dataset_root / "Homecage-LightningPose-Core23-Hybrid-v1"
    real_counts = write_dataset(
        real_output,
        real_records,
        description="Lightning Pose CRIM13 two-mouse partial Core23 2D labels",
        source_roots=[args.crim13_root],
    )
    synthetic_counts = write_dataset(
        synthetic_output,
        synthetic_records,
        description="Homecage Blender black-mouse full multiview Core23 supervision",
        source_roots=[args.homecage_root],
    )
    hybrid_records = real_records + synthetic_records
    hybrid_counts = write_dataset(
        hybrid_output,
        hybrid_records,
        description="PyMouseLifter-style real-partial plus synthetic-full Core23 mixture",
        source_roots=[args.crim13_root, args.homecage_root],
    )
    hybrid_by_split = split_records(hybrid_records)
    pilot = build_pilot(hybrid_by_split)
    for split, records in pilot.items():
        write_jsonl(hybrid_output / f"pilot_{split}.jsonl", records)
    write_json(
        hybrid_output / "mixture.json",
        {
            "schema": "mousegeoforge_instance_core23_v1",
            "recommended_batch_source_weights": {
                "homecage_blackfur_blender": 0.6,
                "lightningpose_crim13": 0.4,
            },
            "loss_policy": {
                "synthetic_full_multiview_3d": [
                    "objectness",
                    "box",
                    "mask",
                    "keypoint_2d",
                    "visibility",
                    "reid_across_views",
                    "depth",
                    "keypoint_3d",
                ],
                "real_partial_2d": ["objectness", "partial_keypoint_2d"],
            },
            "pilot_counts": {split: len(records) for split, records in pilot.items()},
        },
    )
    write_json(
        hybrid_output / "keypoint_mapping.json",
        {
            "target": list(CORE23),
            "lightningpose_crim13": CRIM13_TO_CORE23,
        },
    )
    write_readme(hybrid_output / "README.md")
    print(
        json.dumps(
            {
                "real": real_counts,
                "synthetic": synthetic_counts,
                "hybrid": hybrid_counts,
                "pilot": {split: len(records) for split, records in pilot.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
