#!/usr/bin/env python3
"""Numerical smoke test for BlenderPyMouse keypoint loading and losses."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_SCRIPT = PROJECT_ROOT / "vggt-omega" / "training" / "finetune_blendedmvs.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "BlenderPyMouse",
    )
    parser.add_argument("--image-width", type=int, default=624)
    parser.add_argument("--image-height", type=int, default=416)
    return parser.parse_args()


def load_training_module():
    spec = importlib.util.spec_from_file_location("mgf_keypoint_training_test", TRAINING_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(TRAINING_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_training_args(training, args: argparse.Namespace):
    original_argv = sys.argv
    try:
        sys.argv = [
            "test_vggt_keypoint_supervision",
            "--blendedmvs-root",
            str(args.datasets_root),
            "--image-width",
            str(args.image_width),
            "--image-height",
            str(args.image_height),
            "--num-views",
            "3",
            "--batch-size",
            "1",
            "--num-workers",
            "0",
            "--samples-per-scene",
            "1",
            "--max-scenes",
            "2",
            "--val-scenes",
            "0",
            "--camera-weight",
            "0",
            "--depth-weight",
            "0",
            "--keypoint-region-weight",
            "0.1",
            "--keypoint-sparse-depth-weight",
            "0.5",
            "--keypoint-reprojection-weight",
            "0.1",
        ]
        return training.parse_args()
    finally:
        sys.argv = original_argv


def main() -> None:
    args = parse_args()
    training = load_training_module()
    training_args = make_training_args(training, args)
    batch = next(iter(training.make_loader(training_args, "train")))
    if not bool(batch["keypoints_available"].all()):
        raise AssertionError("Expected keypoint labels in the smoke-test batch")
    if batch["keypoint_joint_world_xyz"].shape[-2] != 29:
        raise AssertionError(batch["keypoint_joint_world_xyz"].shape)

    world = batch["keypoint_joint_world_xyz"]
    extrinsics = batch["extrinsics"]
    intrinsics = batch["intrinsics"]
    camera = torch.einsum(
        "bvij,bkj->bvki",
        extrinsics[..., :3],
        world,
    ) + extrinsics[..., 3][:, :, None, :]
    homogeneous = torch.einsum("bvij,bvkj->bvki", intrinsics, camera)
    projected_uv = homogeneous[..., :2] / homogeneous[..., 2:3]
    valid = batch["keypoint_in_frame"]
    maximum_error = (projected_uv - batch["keypoint_uv"]).abs()[valid].max()
    if not torch.isfinite(maximum_error) or maximum_error > 1e-3:
        raise AssertionError(f"Canonical keypoint reprojection error: {float(maximum_error)}")

    target_pose = training.extri_intri_to_pose_encoding(
        extrinsics,
        intrinsics,
        batch["images"].shape[-2:],
    )
    pred_pose = target_pose.detach().clone()
    pred_pose[..., 0] += 0.01
    pred_pose[..., 7] += 0.01
    pred_pose.requires_grad_(True)

    pred_depth = batch["depths"].detach().clone().clamp_min(1e-3) * 1.1
    pred_depth = pred_depth.unsqueeze(-1).requires_grad_(True)
    predictions = {"pose_enc": pred_pose, "depth": pred_depth}
    loss, logs = training.compute_loss(predictions, batch, training_args)
    if not torch.isfinite(loss):
        raise AssertionError(f"Non-finite combined keypoint loss: {float(loss)}")
    loss.backward()
    if pred_depth.grad is None or not torch.isfinite(pred_depth.grad).all() or pred_depth.grad.abs().sum() == 0:
        raise AssertionError("Keypoint depth losses did not produce finite non-zero gradients")
    if pred_pose.grad is None or not torch.isfinite(pred_pose.grad).all() or pred_pose.grad.abs().sum() == 0:
        raise AssertionError("Keypoint reprojection loss did not produce finite non-zero gradients")

    print(
        "vggt_keypoint_supervision_test_ok",
        f"shape={tuple(batch['images'].shape)}",
        f"keypoints={batch['keypoint_joint_world_xyz'].shape[-2]}",
        f"visible={int(batch['keypoint_surface_visible'].sum())}",
        f"canonical_reprojection_max_error_px={float(maximum_error):.9g}",
        f"loss_region={float(logs['loss_keypoint_region_depth'].detach()):.6g}",
        f"loss_sparse={float(logs['loss_keypoint_sparse_depth'].detach()):.6g}",
        f"loss_reprojection={float(logs['loss_keypoint_reprojection'].detach()):.6g}",
    )


if __name__ == "__main__":
    main()
