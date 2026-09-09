"""Dataset loader for mousegeoforge_instance_core23_v1 JSONL manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Core23InstanceManifestDataset(Dataset):
    """Load one homogeneous manifest split without inventing missing labels.

    Samples may have different view/instance counts. Use ``batch_size=1`` or a
    source-aware batch sampler that groups identical ``(V, M)`` signatures.
    """

    def __init__(
        self,
        manifest: str | Path,
        *,
        image_size: tuple[int, int] = (512, 288),
        project_root: str | Path = PROJECT_ROOT,
    ) -> None:
        self.manifest = Path(manifest)
        self.project_root = Path(project_root)
        self.image_width, self.image_height = map(int, image_size)
        if self.image_width % 16 or self.image_height % 16:
            raise ValueError("image_size must be divisible by VGGT patch size 16")
        with self.manifest.open("r", encoding="utf-8") as stream:
            self.records = [json.loads(line) for line in stream if line.strip()]
        if not self.records:
            raise ValueError(f"Empty instance manifest: {self.manifest}")

    def __len__(self) -> int:
        return len(self.records)

    def signature(self, index: int) -> tuple[int, int, str]:
        record = self.records[index]
        return len(record["views"]), len(record["instance_ids"]), record["source"]

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        original_width, original_height = record["image_size"]
        scale_x = self.image_width / float(original_width)
        scale_y = self.image_height / float(original_height)
        images = []
        masks = []
        intrinsics = []
        extrinsics = []
        camera_available = []
        for view_index, view in enumerate(record["views"]):
            with Image.open(self.project_root / view["image"]) as image:
                image = image.convert("RGB").resize(
                    (self.image_width, self.image_height), Image.Resampling.BICUBIC
                )
                array = np.asarray(image, dtype=np.float32).copy() / 255.0
            images.append(torch.from_numpy(array).permute(2, 0, 1))
            view_masks = []
            for instance_index, mask_path in enumerate(record["mask_paths"][view_index]):
                if record["mask_valid"][view_index][instance_index] and mask_path:
                    with Image.open(self.project_root / mask_path) as mask:
                        mask = mask.convert("L").resize(
                            (self.image_width, self.image_height), Image.Resampling.NEAREST
                        )
                        view_masks.append(torch.from_numpy(np.asarray(mask).copy() > 0))
                else:
                    view_masks.append(
                        torch.zeros(self.image_height, self.image_width, dtype=torch.bool)
                    )
            masks.append(torch.stack(view_masks))
            if view["intrinsic"] is None:
                intrinsics.append(torch.eye(3, dtype=torch.float32))
                extrinsics.append(torch.eye(4, dtype=torch.float32))
                camera_available.append(False)
            else:
                intrinsic = torch.tensor(view["intrinsic"], dtype=torch.float32)
                intrinsic[0] *= scale_x
                intrinsic[1] *= scale_y
                intrinsics.append(intrinsic)
                extrinsics.append(torch.tensor(view["extrinsic"], dtype=torch.float32))
                camera_available.append(True)

        keypoint_world = record.get("keypoint_world_xyz")
        instances = len(record["instance_ids"])
        if keypoint_world is None:
            keypoint_world = torch.zeros(instances, 23, 3, dtype=torch.float32)
        else:
            keypoint_world = torch.tensor(keypoint_world, dtype=torch.float32)
        return {
            "sample_id": record["sample_id"],
            "source": record["source"],
            "supervision": record["supervision"],
            "images": torch.stack(images),
            "view_valid": torch.ones(len(images), dtype=torch.bool),
            "instance_valid": torch.ones(len(images), instances, dtype=torch.bool),
            "instance_ids": torch.tensor(record["instance_ids"], dtype=torch.long)[None].expand(
                len(images), -1
            ).clone(),
            "boxes_cxcywh": torch.tensor(record["boxes_cxcywh"], dtype=torch.float32),
            "box_valid": torch.tensor(record["box_valid"], dtype=torch.bool),
            "masks": torch.stack(masks),
            "mask_valid": torch.tensor(record["mask_valid"], dtype=torch.bool),
            "keypoint_xy": torch.tensor(record["keypoint_xy"], dtype=torch.float32),
            "keypoint_valid": torch.tensor(record["keypoint_valid"], dtype=torch.bool),
            "keypoint_visible": torch.tensor(record["keypoint_visible"], dtype=torch.bool),
            "keypoint_visibility_valid": torch.tensor(
                record["keypoint_visibility_valid"], dtype=torch.bool
            ),
            "keypoint_world_xyz": keypoint_world,
            "world_valid": torch.tensor(record["world_valid"], dtype=torch.bool),
            "intrinsics": torch.stack(intrinsics),
            "extrinsics": torch.stack(extrinsics),
            "camera_available": torch.tensor(camera_available, dtype=torch.bool),
            "original_image_size": torch.tensor(
                [original_width, original_height], dtype=torch.long
            ),
        }


def inspect_manifest_sample(manifest: str | Path, index: int = 0) -> dict[str, Any]:
    dataset = Core23InstanceManifestDataset(manifest)
    sample = dataset[index]
    return {
        key: tuple(value.shape) if torch.is_tensor(value) else value
        for key, value in sample.items()
    }
