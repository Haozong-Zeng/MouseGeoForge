#!/usr/bin/env python3
"""Audit every BlenderPyMouse dataset through the padded instance dataloader."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VGGT_ROOT = PROJECT_ROOT / "vggt-omega"
if str(VGGT_ROOT) not in sys.path:
    sys.path.insert(0, str(VGGT_ROOT))

from training.finetune_blendedmvs import (  # noqa: E402
    BlendedMVSDataset,
    discover_scene_roots,
    load_scene,
)
from training.instance_pose_losses import instance_targets_from_batch  # noqa: E402


def dataset_for_scenes(scenes, num_views: int) -> BlendedMVSDataset:
    return BlendedMVSDataset(
        scenes=scenes,
        num_views=num_views,
        image_width=128,
        image_height=128,
        samples_per_scene=1,
        seed=5706,
        use_masked_images=False,
        pfm_flip_y=False,
        canonicalize_camera=False,
        scale_mode="raw",
        top_srcs=8,
        load_keypoints=True,
    )


def shape(value) -> list[int] | None:
    return list(value.shape) if torch.is_tensor(value) else None


def audit_one(root: Path) -> dict:
    scene_roots = discover_scene_roots(root)
    scene = load_scene(scene_roots[0])
    num_views = min(2, len(scene.frames))
    dataset = dataset_for_scenes([scene], num_views)
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
    targets = instance_targets_from_batch(batch)
    expected_prefix = [1, num_views, dataset.max_instances]
    checks = {
        "instance_valid": shape(batch["instance_valid"]),
        "instance_ids": shape(batch["instance_ids"]),
        "boxes_cxcywh": shape(batch["boxes_cxcywh"]),
        "masks": shape(batch["masks"]),
        "instance_keypoint_xy": shape(batch["instance_keypoint_xy"]),
        "instance_keypoint_valid": shape(batch["instance_keypoint_valid"]),
        "instance_keypoint_world_xyz": shape(batch["instance_keypoint_world_xyz"]),
    }
    if checks["instance_valid"] != expected_prefix:
        raise AssertionError((root, checks["instance_valid"], expected_prefix))
    if checks["instance_ids"] != expected_prefix:
        raise AssertionError((root, checks["instance_ids"], expected_prefix))
    if checks["boxes_cxcywh"] != [*expected_prefix, 4]:
        raise AssertionError((root, checks["boxes_cxcywh"]))
    if checks["masks"] != [*expected_prefix, 128, 128]:
        raise AssertionError((root, checks["masks"]))
    if not torch.equal(targets["instance_valid"], batch["instance_valid"]):
        raise AssertionError(f"Native instance targets were not preserved for {root}")
    return {
        "dataset": root.name,
        "scene": scene.root.name,
        "views_loaded": num_views,
        "canonical_keypoints": len(dataset.keypoint_names),
        "max_instances": dataset.max_instances,
        "actual_instance_count": int(batch["actual_instance_count"][0]),
        "supervised_instance_views": int(batch["instance_valid"].sum()),
        "valid_instance_masks": int(batch["mask_valid"].sum()),
        "legacy_single_keypoint_compatible": bool(batch["keypoints_available"][0]),
        "shapes": checks,
    }


def audit_mixed_batch(root: Path) -> dict:
    single_root = (
        root
        / "Homecage-BlackFur-KeypointMoseq-Core23-Sample10-Frames200-399"
        / "frame_000001"
    )
    dual_root = (
        root
        / "Homecage-BlackBrownFur-LightningPose-CRIM13-DiversePose400-DepthHeightV1"
        / "frame_000001"
    )
    scenes = [load_scene(single_root), load_scene(dual_root)]
    dataset = dataset_for_scenes(scenes, num_views=4)
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
    targets = instance_targets_from_batch(batch)
    expected = {
        "images": [2, 4, 3, 128, 128],
        "instance_valid": [2, 4, 2],
        "masks": [2, 4, 2, 128, 128],
        "keypoint_xy": [2, 4, 2, 23, 2],
        "keypoint_world_xyz": [2, 2, 23, 3],
    }
    actual = {
        "images": shape(batch["images"]),
        "instance_valid": shape(targets["instance_valid"]),
        "masks": shape(targets["masks"]),
        "keypoint_xy": shape(targets["keypoint_xy"]),
        "keypoint_world_xyz": shape(targets["keypoint_world_xyz"]),
    }
    if actual != expected:
        raise AssertionError({"actual": actual, "expected": expected})
    per_sample_instances = batch["instance_valid"].any(dim=1).sum(dim=1).tolist()
    if per_sample_instances != [1, 2]:
        raise AssertionError(f"Wrong mixed M=1/M=2 padding: {per_sample_instances}")
    if batch["instance_valid"][0, :, 1].any():
        raise AssertionError("Padded second instance of the single-mouse sample is valid")
    if not batch["instance_valid"][1].all():
        raise AssertionError("A real dual-mouse instance was dropped")
    return {
        "datasets": [single_root.parent.name, dual_root.parent.name],
        "max_instances": dataset.max_instances,
        "per_sample_instances": per_sample_instances,
        "shapes": actual,
        "single_padding_valid": True,
        "native_target_passthrough": True,
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
            / "multi_instance_dataloader_audit.json"
        ),
    )
    args = parser.parse_args()
    roots = [
        path
        for path in sorted(args.root.iterdir())
        if path.is_dir() and not path.name.startswith("_")
    ]
    results = []
    errors = []
    for root in roots:
        try:
            results.append(audit_one(root))
        except Exception as error:  # keep auditing the remaining datasets
            errors.append({"dataset": root.name, "error": repr(error)})
    mixed = None
    try:
        mixed = audit_mixed_batch(args.root)
    except Exception as error:
        errors.append({"dataset": "mixed_single_dual_batch", "error": repr(error)})
    payload = {
        "status": "complete" if not errors else "failed",
        "root": str(args.root.resolve()),
        "dataset_count": len(results),
        "labeled_dataset_count": sum(row["canonical_keypoints"] > 0 for row in results),
        "unlabeled_dataset_count": sum(row["canonical_keypoints"] == 0 for row in results),
        "datasets": results,
        "mixed_single_dual_batch": mixed,
        "errors": errors,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
