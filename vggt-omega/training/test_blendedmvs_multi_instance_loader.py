from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from training.finetune_blendedmvs import BlendedMVSDataset, load_scene
from training.instance_pose_losses import instance_targets_from_batch


NAMES = np.asarray(["nose", "neck", "tail_base"])


def test_explicit_yolo_boxes_without_segmentation(tmp_path: Path) -> None:
    root = tmp_path / "yolo"
    write_scene(root, 1)
    with np.load(root / "keypoints.npz") as source:
        payload = {key: source[key].copy() for key in source.files}
    payload["boxes_cxcywh"] = np.asarray([[[.4, .5, .6, .7]], [[.5, .4, .3, .2]]], np.float32)
    payload["box_valid"] = np.ones((2, 1), bool)
    payload["in_frame"][:] = False  # Box-only annotations still identify an instance.
    np.savez_compressed(root / "keypoints.npz", **payload)
    scene = load_scene(root)
    for view_id, frame in scene.frames.items():
        scene.frames[view_id] = replace(frame, body_mask_path=None)
    dataset = BlendedMVSDataset(scenes=[scene], num_views=2, image_width=96,
        image_height=64, samples_per_scene=1, seed=1, use_masked_images=False,
        pfm_flip_y=False, canonicalize_camera=False, scale_mode="raw", top_srcs=2,
        load_keypoints=True)
    sample = dataset[0]
    order = [int(v) for v in sample["view_ids"].split(",")]
    assert np.allclose(sample["boxes_cxcywh"].numpy(), payload["boxes_cxcywh"][order])
    assert sample["box_valid"].all() and sample["instance_valid"].all()
    assert not sample["mask_valid"].any()


def write_scene(root: Path, instances: int) -> None:
    image_dir = root / "blended_images"
    mask_dir = root / "masks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir()
    for view_id in range(2):
        image = np.zeros((64, 96, 3), dtype=np.uint8)
        image[..., view_id] = 80
        Image.fromarray(image).save(image_dir / f"{view_id:08d}.jpg")
        combined = np.zeros((64, 96), dtype=np.uint8)
        for instance_id in range(instances):
            mask = np.zeros_like(combined)
            left = 8 + 36 * instance_id
            mask[16:48, left : left + 24] = 255
            combined |= mask
            if instances > 1:
                Image.fromarray(mask).save(
                    mask_dir / f"{view_id:08d}_instance{instance_id:02d}.png"
                )
        Image.fromarray(combined).save(mask_dir / f"{view_id:08d}.png")

    joint = np.zeros((instances, 3, 3), dtype=np.float32)
    surface = np.zeros_like(joint)
    uv = np.zeros((2, instances, 3, 2), dtype=np.float32)
    in_frame = np.ones((2, instances, 3), dtype=bool)
    visible = np.ones_like(in_frame)
    for instance_id in range(instances):
        joint[instance_id, :, 0] = instance_id * 10 + np.arange(3)
        surface[instance_id] = joint[instance_id]
        uv[:, instance_id, :, 0] = 18 + 36 * instance_id
        uv[:, instance_id, :, 1] = np.asarray([20, 30, 40])
    payload = {
        "format": np.asarray("test_dual" if instances > 1 else "test_single"),
        "names": NAMES,
        "view_ids": np.arange(2, dtype=np.int32),
        "joint_world_xyz": joint,
        "surface_world_xyz": surface,
        "uv": uv,
        "in_frame": in_frame,
        "surface_visible": visible,
    }
    if instances > 1:
        payload["instance_ids"] = np.arange(instances, dtype=np.int32)
    else:
        for key in (
            "joint_world_xyz",
            "surface_world_xyz",
        ):
            payload[key] = payload[key][0]
        for key in ("uv", "in_frame", "surface_visible"):
            payload[key] = payload[key][:, 0]
    np.savez_compressed(root / "keypoints.npz", **payload)


def test_single_and_dual_scenes_collate_with_padded_instance_axis(tmp_path: Path) -> None:
    single_root = tmp_path / "single"
    dual_root = tmp_path / "dual"
    write_scene(single_root, 1)
    write_scene(dual_root, 2)
    scenes = [load_scene(single_root), load_scene(dual_root)]
    dataset = BlendedMVSDataset(
        scenes=scenes,
        num_views=2,
        image_width=96,
        image_height=64,
        samples_per_scene=1,
        seed=1,
        use_masked_images=False,
        pfm_flip_y=False,
        canonicalize_camera=False,
        scale_mode="raw",
        top_srcs=2,
        load_keypoints=True,
    )
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
    targets = instance_targets_from_batch(batch)

    assert dataset.max_instances == 2
    assert batch["actual_instance_count"].tolist() == [1, 2]
    assert batch["instance_valid"].shape == (2, 2, 2)
    assert batch["masks"].shape == (2, 2, 2, 64, 96)
    assert batch["instance_keypoint_xy"].shape == (2, 2, 2, 3, 2)
    assert batch["instance_keypoint_world_xyz"].shape == (2, 2, 3, 3)
    assert batch["instance_valid"][0, :, 0].all()
    assert not batch["instance_valid"][0, :, 1].any()
    assert batch["instance_valid"][1].all()
    assert torch.equal(targets["keypoint_xy"], batch["instance_keypoint_xy"])
    assert torch.equal(targets["keypoint_valid"], batch["instance_keypoint_valid"])
    assert targets["keypoint_world_xyz"].shape == (2, 2, 3, 3)


def test_single_scene_keeps_legacy_and_native_contracts(tmp_path: Path) -> None:
    root = tmp_path / "single"
    write_scene(root, 1)
    dataset = BlendedMVSDataset(
        scenes=[load_scene(root)],
        num_views=2,
        image_width=96,
        image_height=64,
        samples_per_scene=1,
        seed=1,
        use_masked_images=False,
        pfm_flip_y=False,
        canonicalize_camera=False,
        scale_mode="raw",
        top_srcs=2,
        load_keypoints=True,
    )
    sample = dataset[0]
    assert bool(sample["keypoints_available"])
    assert bool(sample["instance_supervision_available"])
    assert sample["keypoint_uv"].shape == (2, 3, 2)
    assert sample["instance_keypoint_xy"].shape == (2, 1, 3, 2)
    expected = sample["keypoint_uv"].clone()
    expected[..., 0] /= 96.0
    expected[..., 1] /= 64.0
    assert torch.allclose(sample["instance_keypoint_xy"][:, 0], expected)


def test_explicit_partial_label_masks_are_preserved(tmp_path: Path) -> None:
    root = tmp_path / "partial"
    write_scene(root, 1)
    with np.load(root / "keypoints.npz", allow_pickle=False) as source:
        payload = {key: source[key] for key in source.files}
    payload["world_valid"] = np.asarray([True, False, True])
    payload["visibility_valid"] = np.zeros((2, 3), dtype=bool)
    np.savez_compressed(root / "keypoints.npz", **payload)

    dataset = BlendedMVSDataset(
        scenes=[load_scene(root)],
        num_views=2,
        image_width=96,
        image_height=64,
        samples_per_scene=1,
        seed=1,
        use_masked_images=False,
        pfm_flip_y=False,
        canonicalize_camera=False,
        scale_mode="raw",
        top_srcs=2,
        load_keypoints=True,
    )
    sample = dataset[0]
    assert sample["instance_world_valid"][0].tolist() == [True, False, True]
    assert sample["keypoint_valid"].tolist() == [True, False, True]
    assert not sample["instance_keypoint_visibility_valid"].any()
