#!/usr/bin/env python3
"""Create a small CRIM13-conditioned two-mouse Homecage composite dataset.

This is a 2D detection/instance-pose pilot, not metric 3D ground truth. Sparse
real layouts select the closest fully labeled Blender poses; all pixels and
labels then receive the same similarity transform.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE23 = (
    "nose", "head_center", "left_ear", "right_ear", "neck", "spine_center",
    "lumbar_spine", "tail_base", "tail_q1", "tail_q2", "tail_q3", "tail_q4",
    "tail_q5", "tail_tip", "left_forepaw_center", "right_forepaw_center",
    "left_hindpaw_center", "right_hindpaw_center", "left_elbow", "right_elbow",
    "left_knee", "right_knee", "abdomen_center",
)
CONDITIONING_INDICES = tuple(
    CORE23.index(name)
    for name in ("nose", "left_ear", "right_ear", "neck", "left_knee", "right_knee", "tail_base")
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def project_path(path: str) -> Path:
    return PROJECT_ROOT / path


def pose_descriptor(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    scale = np.sqrt(np.square(centered).sum())
    return centered / max(float(scale), 1e-8)


def similarity_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / source.shape[0]
    left, singular, right_t = np.linalg.svd(covariance)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_t
    source_variance = np.square(source_centered).sum() / source.shape[0]
    scale = float(singular.sum() / max(source_variance, 1e-8))
    translation = target_mean - scale * (rotation @ source_mean)
    matrix = np.concatenate((scale * rotation, translation[:, None]), axis=1).astype(np.float32)
    predicted = source @ matrix[:, :2].T + matrix[:, 2]
    residual = float(np.linalg.norm(predicted - target, axis=1).mean())
    return matrix, residual


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:, :2].T + matrix[:, 2]


def exact_box(mask: np.ndarray) -> list[float]:
    height, width = mask.shape
    y, x = np.where(mask)
    if x.size == 0:
        raise ValueError("Composite produced an empty instance mask")
    left, right = x.min(), x.max() + 1
    top, bottom = y.min(), y.max() + 1
    return [
        float((left + right) / (2.0 * width)),
        float((top + bottom) / (2.0 * height)),
        float((right - left) / width),
        float((bottom - top) / height),
    ]


def synthetic_pose(record: dict[str, Any]) -> np.ndarray:
    width, height = record["image_size"]
    points = np.asarray(record["keypoint_xy"][0][0], dtype=np.float32)
    return points * np.asarray([width, height], dtype=np.float32)


def conditioning_target(record: dict[str, Any], instance: int, width: int, height: int) -> np.ndarray:
    points = np.asarray(record["keypoint_xy"][0][instance], dtype=np.float32)
    # Map the real arena into the bedding rectangle of the synthetic homecage.
    points[:, 0] = 0.15 + 0.70 * points[:, 0]
    points[:, 1] = 0.14 + 0.72 * points[:, 1]
    return points * np.asarray([width, height], dtype=np.float32)


def best_pose(
    target: np.ndarray,
    bank: list[dict[str, Any]],
) -> tuple[dict[str, Any], np.ndarray, float]:
    target_conditioning = target[list(CONDITIONING_INDICES)]
    target_descriptor = pose_descriptor(target_conditioning)
    best: tuple[float, dict[str, Any], np.ndarray, float] | None = None
    for record in bank:
        source = synthetic_pose(record)
        source_conditioning = source[list(CONDITIONING_INDICES)]
        descriptor_error = float(
            np.square(pose_descriptor(source_conditioning) - target_descriptor).mean()
        )
        matrix, residual = similarity_transform(source_conditioning, target_conditioning)
        candidate = (descriptor_error, record, matrix, residual)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise ValueError("Empty synthetic pose bank")
    return best[1], best[2], best[3]


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return mask > 0


def build_clean_background(bank: list[dict[str, Any]], output: Path) -> np.ndarray:
    """Use a mask-aware temporal median so moving mice disappear cleanly."""
    sample_count = min(41, len(bank))
    indices = np.linspace(0, len(bank) - 1, sample_count).round().astype(int)
    frames = np.stack(
        [read_rgb(project_path(bank[index]["views"][0]["image"])) for index in indices]
    ).astype(np.float32)
    masks = np.stack(
        [read_mask(project_path(bank[index]["mask_paths"][0][0])) for index in indices]
    )
    frames[masks] = np.nan
    background = np.nanmedian(frames, axis=0)
    fallback = np.nanmedian(frames.reshape(-1, 3), axis=0)
    background = np.where(np.isnan(background), fallback, background).astype(np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), background, [cv2.IMWRITE_JPEG_QUALITY, 96])
    return background


def fit_transform_inside_bedding(
    record: dict[str, Any],
    matrix: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    source_mask = read_mask(project_path(record["mask_paths"][0][0])).astype(np.uint8)
    probe = cv2.warpAffine(
        source_mask,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    ) > 0
    y, x = np.where(probe)
    if x.size == 0:
        return matrix
    # Stay on bedding and away from the right-side tunnel hardware.
    minimum_x, maximum_x = 0.12 * width, 0.83 * width
    minimum_y, maximum_y = 0.10 * height, 0.90 * height
    delta_x = max(minimum_x - x.min(), 0.0) + min(maximum_x - x.max(), 0.0)
    delta_y = max(minimum_y - y.min(), 0.0) + min(maximum_y - y.max(), 0.0)
    fitted = matrix.copy()
    fitted[0, 2] += delta_x
    fitted[1, 2] += delta_y
    return fitted


def warped_mouse(
    record: dict[str, Any],
    matrix: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image = read_rgb(project_path(record["views"][0]["image"]))
    mask = read_mask(project_path(record["mask_paths"][0][0])).astype(np.uint8) * 255
    warped_image = cv2.warpAffine(
        image, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT
    )
    warped_mask = cv2.warpAffine(
        mask, matrix, (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT
    ) > 0
    points = transform_points(synthetic_pose(record), matrix)
    source_valid = np.asarray(record["keypoint_valid"][0][0], dtype=bool)
    in_frame = (
        (points[:, 0] >= 0)
        & (points[:, 0] < width)
        & (points[:, 1] >= 0)
        & (points[:, 1] < height)
    )
    return warped_image, warped_mask, points, source_valid & in_frame


def composite_one(
    real: dict[str, Any],
    bank: list[dict[str, Any]],
    output_root: Path,
    output_index: int,
    background: np.ndarray,
) -> dict[str, Any]:
    width, height = bank[0]["image_size"]
    selected = []
    for instance in range(2):
        target = conditioning_target(real, instance, width, height)
        source, matrix, residual = best_pose(target, bank)
        matrix = fit_transform_inside_bedding(source, matrix, width, height)
        transformed_conditioning = transform_points(
            synthetic_pose(source)[list(CONDITIONING_INDICES)], matrix
        )
        residual = float(
            np.linalg.norm(
                transformed_conditioning - target[list(CONDITIONING_INDICES)], axis=1
            ).mean()
        )
        selected.append((source, matrix, residual, *warped_mouse(source, matrix, width, height)))

    canvas = background.astype(np.float32).copy()
    # Alternate depth order so one identity is not always the occluder.
    order = (0, 1) if output_index % 2 == 0 else (1, 0)
    for instance in order:
        warped_image, warped_mask = selected[instance][3], selected[instance][4]
        alpha = cv2.GaussianBlur(warped_mask.astype(np.float32), (5, 5), 0)[..., None]
        canvas = warped_image.astype(np.float32) * alpha + canvas * (1.0 - alpha)
    image = np.clip(canvas, 0, 255).astype(np.uint8)

    full_masks = [selected[index][4] for index in range(2)]
    visible_masks = [mask.copy() for mask in full_masks]
    lower, upper = order[0], order[1]
    visible_masks[lower] &= ~full_masks[upper]
    output_stem = f"{output_index:06d}"
    image_path = output_root / "images" / f"{output_stem}.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 94])
    mask_paths = []
    for instance, mask in enumerate(visible_masks):
        path = output_root / "masks" / f"{output_stem}_instance{instance}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), mask.astype(np.uint8) * 255)
        mask_paths.append(path)

    keypoint_xy = []
    keypoint_valid = []
    keypoint_visible = []
    boxes = []
    for instance in range(2):
        points = selected[instance][5]
        valid = selected[instance][6]
        visible = valid.copy()
        if instance == lower:
            rounded = np.rint(points).astype(int)
            safe_x = np.clip(rounded[:, 0], 0, width - 1)
            safe_y = np.clip(rounded[:, 1], 0, height - 1)
            visible &= ~full_masks[upper][safe_y, safe_x]
        normalized = points / np.asarray([width, height], dtype=np.float32)
        keypoint_xy.append(normalized.astype(float).tolist())
        keypoint_valid.append(valid.tolist())
        keypoint_visible.append(visible.tolist())
        boxes.append(exact_box(visible_masks[instance]))

    def rel(path: Path) -> str:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()

    return {
        "schema": "mousegeoforge_instance_core23_v1",
        "sample_id": f"homecage_crim13_composite:{output_stem}",
        "split": real["split"],
        "source": "homecage_crim13_pose_conditioned_composite",
        "supervision": "synthetic_full_2d_two_mouse",
        "identity_scope": f"composite:{real['identity_scope']}",
        "conditioning_sample_id": real["sample_id"],
        "conditioning_keypoints": [CORE23[index] for index in CONDITIONING_INDICES],
        "pose_bank_sample_ids": [selected[0][0]["sample_id"], selected[1][0]["sample_id"]],
        "conditioning_residual_px": [selected[0][2], selected[1][2]],
        "instance_overlap_pixels": int(np.logical_and(full_masks[0], full_masks[1]).sum()),
        "occluded_keypoint_count": int(
            sum(sum(valid) - sum(visible) for valid, visible in zip(keypoint_valid, keypoint_visible, strict=True))
        ),
        "appearance_policy": "same_black_c57",
        "image_size": [width, height],
        "views": [{
            "view_id": 0,
            "camera_name": "homecage_top_composite_uncalibrated",
            "image": rel(image_path),
            "channels": 3,
            "intrinsic": None,
            "extrinsic": None,
            "depth": None,
        }],
        "instance_ids": [0, 1],
        "instance_names": ["resident_proxy", "intruder_proxy"],
        "boxes_cxcywh": [[boxes[0], boxes[1]]],
        "box_valid": [[True, True]],
        "box_source": "visible_instance_mask_exact",
        "mask_paths": [[rel(mask_paths[0]), rel(mask_paths[1])]],
        "mask_valid": [[True, True]],
        "keypoint_names": list(CORE23),
        "keypoint_xy": [[keypoint_xy[0], keypoint_xy[1]]],
        "keypoint_valid": [[keypoint_valid[0], keypoint_valid[1]]],
        "keypoint_visible": [[keypoint_visible[0], keypoint_visible[1]]],
        "keypoint_visibility_valid": [[keypoint_valid[0], keypoint_valid[1]]],
        "keypoint_world_xyz": None,
        "world_valid": [[False] * 23, [False] * 23],
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hybrid-root",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "Homecage-LightningPose-Core23-Hybrid-v1",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "dataset"
            / "Homecage-CRIM13-PoseConditioned-Core23-MultiMouse-Pilot-v1"
        ),
    )
    parser.add_argument("--train-count", type=int, default=24)
    parser.add_argument("--val-count", type=int, default=8)
    parser.add_argument("--test-count", type=int, default=8)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    complete_bank = []
    for source_split in ("train", "val", "test"):
        complete_bank.extend(
            record
            for record in load_jsonl(args.hybrid_root / f"{source_split}.jsonl")
            if record["source"] == "homecage_blackfur_blender"
        )
    background = build_clean_background(
        complete_bank,
        args.output_root / "assets" / "clean_homecage_top_background.jpg",
    )

    all_records = []
    output_index = 0
    limits = {"train": args.train_count, "val": args.val_count, "test": args.test_count}
    for split in ("train", "val", "test"):
        hybrid = load_jsonl(args.hybrid_root / f"{split}.jsonl")
        real = [record for record in hybrid if record["source"] == "lightningpose_crim13"]
        bank = [record for record in hybrid if record["source"] == "homecage_blackfur_blender"]
        if not bank:
            # Keep visual assets disjoint when possible, but fall back to the complete pose bank.
            bank = []
            for source_split in ("train", "val", "test"):
                bank.extend(
                    record
                    for record in load_jsonl(args.hybrid_root / f"{source_split}.jsonl")
                    if record["source"] == "homecage_blackfur_blender"
                )
        chosen_indices = np.linspace(
            0,
            len(real) - 1,
            min(limits[split], len(real)),
        ).round().astype(int)
        chosen = [real[index] for index in chosen_indices]
        split_records = []
        for real_record in chosen:
            split_records.append(
                composite_one(
                    real_record,
                    bank,
                    args.output_root,
                    output_index,
                    background,
                )
            )
            output_index += 1
        write_jsonl(args.output_root / f"{split}.jsonl", split_records)
        write_jsonl(
            args.output_root / f"{split}_contact.jsonl",
            [record for record in split_records if record["instance_overlap_pixels"] > 0],
        )
        all_records.extend(split_records)

    residuals = np.asarray(
        [value for record in all_records for value in record["conditioning_residual_px"]]
    )
    metadata = {
        "schema": "mousegeoforge_instance_core23_v1",
        "description": "CRIM13-conditioned two-C57 Homecage 2D composite pilot",
        "counts": {split: limits[split] for split in limits},
        "total": len(all_records),
        "contact_counts": {
            split: sum(
                record["split"] == split and record["instance_overlap_pixels"] > 0
                for record in all_records
            )
            for split in ("train", "val", "test")
        },
        "occlusion_samples": sum(record["occluded_keypoint_count"] > 0 for record in all_records),
        "occluded_keypoints": sum(record["occluded_keypoint_count"] for record in all_records),
        "conditioning_residual_px_mean": float(residuals.mean()),
        "conditioning_residual_px_p95": float(np.percentile(residuals, 95)),
        "valid_losses": ["objectness", "box", "mask", "keypoint_2d", "visibility"],
        "invalid_losses": ["camera", "depth", "keypoint_3d", "metric_geometry"],
        "limitations": [
            "2D similarity composition is not physical 3D retargeting",
            "mask-aware temporal-median background may smooth small bedding details",
            "both instances reuse one C57 appearance family",
        ],
    }
    (args.output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_root / "README.md").write_text(
        """# CRIM13-conditioned Homecage multi-mouse pilot

This pilot follows the PyMouseLifter data-synthesis idea without claiming false 3D labels.
Each CRIM13 two-mouse layout selects two closest Core23 Blender poses. The mouse RGB,
mask, and all 23 points receive one shared similarity transform before occlusion-aware
composition in the Homecage top view.

Use only objectness, box, mask, 2D Core23 and visibility losses. Camera, depth and 3D fields
are intentionally unavailable. This dataset is suitable for bootstrapping the instance head;
the next production dataset should retarget two armatures and render their physical contact
inside Blender.
""",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
