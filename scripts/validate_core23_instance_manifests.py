#!/usr/bin/env python3
"""Validate Core23 instance JSONL manifests and render pilot overlays."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE23 = (
    "nose", "head_center", "left_ear", "right_ear", "neck", "spine_center",
    "lumbar_spine", "tail_base", "tail_q1", "tail_q2", "tail_q3", "tail_q4",
    "tail_q5", "tail_tip", "left_forepaw_center", "right_forepaw_center",
    "left_hindpaw_center", "right_hindpaw_center", "left_elbow", "right_elbow",
    "left_knee", "right_knee", "abdomen_center",
)
COLORS = ((255, 65, 65), (65, 190, 255), (255, 190, 45), (175, 85, 255))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def shape(value: Any) -> tuple[int, ...]:
    result = []
    while isinstance(value, list):
        result.append(len(value))
        value = value[0] if value else None
    return tuple(result)


def validate_record(record: dict[str, Any]) -> None:
    if record["schema"] != "mousegeoforge_instance_core23_v1":
        raise ValueError(f"Bad schema in {record['sample_id']}")
    if tuple(record["keypoint_names"]) != CORE23:
        raise ValueError(f"Core23 mismatch in {record['sample_id']}")
    views = len(record["views"])
    instances = len(record["instance_ids"])
    expected = {
        "boxes_cxcywh": (views, instances, 4),
        "box_valid": (views, instances),
        "mask_paths": (views, instances),
        "mask_valid": (views, instances),
        "keypoint_xy": (views, instances, 23, 2),
        "keypoint_valid": (views, instances, 23),
        "keypoint_visible": (views, instances, 23),
        "keypoint_visibility_valid": (views, instances, 23),
        "world_valid": (instances, 23),
    }
    for key, expected_shape in expected.items():
        actual = shape(record[key])
        if actual != expected_shape:
            raise ValueError(
                f"{record['sample_id']} {key}: {actual}, expected {expected_shape}"
            )
    width, height = record["image_size"]
    for view_index, view in enumerate(record["views"]):
        image = PROJECT_ROOT / view["image"]
        if not image.is_file():
            raise FileNotFoundError(image)
        for instance_index in range(instances):
            mask_path = record["mask_paths"][view_index][instance_index]
            mask_valid = record["mask_valid"][view_index][instance_index]
            if mask_valid and (not mask_path or not (PROJECT_ROOT / mask_path).is_file()):
                raise FileNotFoundError(mask_path)
            box = record["boxes_cxcywh"][view_index][instance_index]
            if any(not 0.0 <= value <= 1.0 for value in box):
                raise ValueError(f"Out-of-range box in {record['sample_id']}: {box}")
            for point, valid in zip(
                record["keypoint_xy"][view_index][instance_index],
                record["keypoint_valid"][view_index][instance_index],
                strict=True,
            ):
                if valid and not (0.0 <= point[0] <= 1.0 and 0.0 <= point[1] <= 1.0):
                    raise ValueError(
                        f"Out-of-frame valid keypoint in {record['sample_id']}: {point}"
                    )
        if view["depth"] is not None and not (PROJECT_ROOT / view["depth"]).is_file():
            raise FileNotFoundError(view["depth"])
        if view["intrinsic"] is not None and shape(view["intrinsic"]) != (3, 3):
            raise ValueError(f"Bad intrinsic in {record['sample_id']}")
        if view["extrinsic"] is not None and shape(view["extrinsic"]) != (4, 4):
            raise ValueError(f"Bad extrinsic in {record['sample_id']}")
    if width <= 0 or height <= 0:
        raise ValueError(f"Bad image size in {record['sample_id']}")
    if record["source"] == "lightningpose_crim13":
        counts = [
            sum(valid)
            for valid in record["keypoint_valid"][0]
        ]
        if views != 1 or instances != 2 or counts != [7, 7]:
            raise ValueError(f"Unexpected CRIM13 contract in {record['sample_id']}: {counts}")
        if any(record["box_valid"][0]) or any(record["mask_valid"][0]):
            raise ValueError(f"CRIM13 weak boxes/masks marked strong in {record['sample_id']}")
    elif record["source"] == "homecage_blackfur_blender":
        if views != 4 or instances != 1:
            raise ValueError(f"Unexpected Homecage contract in {record['sample_id']}")
        if not all(entry[0] for entry in record["box_valid"]):
            raise ValueError(f"Missing synthetic box supervision in {record['sample_id']}")
        if not all(entry[0] for entry in record["mask_valid"]):
            raise ValueError(f"Missing synthetic mask supervision in {record['sample_id']}")


def validate_splits(dataset_root: Path) -> dict[str, Any]:
    sample_ids: set[str] = set()
    identity_splits: dict[str, set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()
    supervision: Counter[str] = Counter()
    for split in ("train", "val", "test"):
        for record in load_jsonl(dataset_root / f"{split}.jsonl"):
            validate_record(record)
            sample_id = record["sample_id"]
            if sample_id in sample_ids:
                raise ValueError(f"Duplicate sample ID: {sample_id}")
            sample_ids.add(sample_id)
            identity_splits[record["identity_scope"]].add(split)
            counts[split] += 1
            supervision[record["supervision"]] += 1
    leaking = {
        scope: sorted(splits)
        for scope, splits in identity_splits.items()
        if len(splits) > 1 and not scope.startswith("homecage_blackfur_sequence")
    }
    if leaking:
        raise ValueError(f"Identity/session leakage across splits: {leaking}")
    return {
        "counts": dict(counts),
        "supervision": dict(supervision),
        "unique_samples": len(sample_ids),
        "identity_scopes": len(identity_splits),
        "real_session_leakage": 0,
    }


def draw_record(record: dict[str, Any], view_index: int = 0) -> Image.Image:
    width, height = record["image_size"]
    image = Image.open(PROJECT_ROOT / record["views"][view_index]["image"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    for instance_index in range(len(record["instance_ids"])):
        color = COLORS[instance_index % len(COLORS)]
        box = record["boxes_cxcywh"][view_index][instance_index]
        center_x, center_y, box_w, box_h = box
        rectangle = (
            (center_x - box_w / 2) * width,
            (center_y - box_h / 2) * height,
            (center_x + box_w / 2) * width,
            (center_y + box_h / 2) * height,
        )
        draw.rectangle(rectangle, outline=color, width=3)
        for point, valid in zip(
            record["keypoint_xy"][view_index][instance_index],
            record["keypoint_valid"][view_index][instance_index],
            strict=True,
        ):
            if valid:
                x, y = point[0] * width, point[1] * height
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline=(0, 0, 0))
    draw.rectangle((0, 0, width, 28), fill=(0, 0, 0))
    draw.text((8, 7), record["sample_id"], fill=(255, 255, 255))
    return image


def contact_sheet(records: list[dict[str, Any]], output: Path) -> None:
    thumbnails = []
    thumb_w, thumb_h = 480, 300
    for record in records:
        image = draw_record(record)
        image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (thumb_w, thumb_h), (32, 32, 32))
        canvas.paste(image, ((thumb_w - image.width) // 2, (thumb_h - image.height) // 2))
        thumbnails.append(canvas)
    columns = 3
    rows = (len(thumbnails) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_w, rows * thumb_h), (20, 20, 20))
    for index, thumbnail in enumerate(thumbnails):
        sheet.paste(thumbnail, ((index % columns) * thumb_w, (index // columns) * thumb_h))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "Homecage-LightningPose-Core23-Hybrid-v1",
    )
    args = parser.parse_args()
    report = validate_splits(args.dataset_root)
    pilot_path = args.dataset_root / "pilot_train.jsonl"
    pilot_train = load_jsonl(pilot_path if pilot_path.is_file() else args.dataset_root / "train.jsonl")
    real = [record for record in pilot_train if record["source"] == "lightningpose_crim13"][:6]
    synthetic = [
        record for record in pilot_train if record["source"] == "homecage_blackfur_blender"
    ][:6]
    review_images = []
    if real:
        contact_sheet(real, args.dataset_root / "review" / "pilot_real_overlay.jpg")
        review_images.append("review/pilot_real_overlay.jpg")
    if synthetic:
        contact_sheet(synthetic, args.dataset_root / "review" / "pilot_synthetic_overlay.jpg")
        review_images.append("review/pilot_synthetic_overlay.jpg")
    if not review_images:
        cross_split = []
        for split in ("train", "val", "test"):
            cross_split.extend(load_jsonl(args.dataset_root / f"{split}.jsonl")[:2])
        contact_sheet(
            cross_split,
            args.dataset_root / "review" / "pilot_composite_overlay.jpg",
        )
        review_images.append("review/pilot_composite_overlay.jpg")
    report["review_images"] = review_images
    (args.dataset_root / "validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
