# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import torch
import torch.nn as nn

from vggt_omega.models.aggregator import Aggregator
from vggt_omega.models.heads import (
    CameraHead,
    DenseHead,
    KeypointHead,
    MultiInstanceMouseHead,
    SurfaceEmbeddingHead,
    TextAlignmentHead,
)


class VGGTOmega(nn.Module):
    """Minimal VGGT-Omega inference model for camera and depth prediction."""

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 1024,
        enable_camera: bool = True,
        enable_depth: bool = True,
        enable_alignment: bool = False,
        enable_keypoints: bool = False,
        num_keypoints: int = 0,
        enable_multi_instance_pose: bool = False,
        num_instance_queries: int = 6,
        enable_surface_embedding: bool = False,
        surface_embedding_dim: int = 16,
    ) -> None:
        super().__init__()

        self.aggregator = Aggregator(patch_size=patch_size, embed_dim=embed_dim)
        _warn_if_rope_not_max(self.aggregator)
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.dense_head = DenseHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_depth else None
        self.text_alignment_head = TextAlignmentHead(dim_in=2 * embed_dim) if enable_alignment else None
        self.keypoint_head = (
            KeypointHead(
                num_keypoints=num_keypoints,
                dim_in=2 * embed_dim,
                patch_size=patch_size,
            )
            if enable_keypoints
            else None
        )
        self.instance_pose_head = (
            MultiInstanceMouseHead(
                num_keypoints=num_keypoints,
                num_queries=num_instance_queries,
                dim_in=2 * embed_dim,
                patch_size=patch_size,
            )
            if enable_multi_instance_pose
            else None
        )
        self.surface_embedding_head = (
            SurfaceEmbeddingHead(
                embedding_dim=surface_embedding_dim,
                dim_in=2 * embed_dim,
                patch_size=patch_size,
            )
            if enable_surface_embedding
            else None
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            aggregated_tokens_list, patch_token_start = self.aggregator(images)

        final_tokens = aggregated_tokens_list[-1]
        if final_tokens is None:
            raise ValueError("Aggregator did not cache the final layer, which VGGTOmega needs.")

        predictions = {
            "camera_and_register_tokens": final_tokens[:, :, :patch_token_start].contiguous(),
        }
        with torch.autocast(device_type="cuda", enabled=False):
            if self.camera_head is not None:
                predictions["pose_enc"] = self.camera_head(
                    aggregated_tokens_list,
                    patch_token_start=patch_token_start,
                )

            if self.dense_head is not None:
                depth, depth_conf = self.dense_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_token_start=patch_token_start,
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.text_alignment_head is not None:
                predictions.update(
                    self.text_alignment_head(
                        aggregated_tokens_list,
                        patch_token_start=patch_token_start,
                    )
                )

            if self.keypoint_head is not None:
                heatmaps, visibility, pixel_sigma = self.keypoint_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_token_start=patch_token_start,
                )
                predictions["keypoint_heatmap_logits"] = heatmaps
                predictions["keypoint_visibility_logits"] = visibility
                predictions["keypoint_pixel_sigma"] = pixel_sigma

            if self.instance_pose_head is not None:
                predictions.update(
                    self.instance_pose_head(
                        aggregated_tokens_list,
                        images=images,
                        patch_token_start=patch_token_start,
                    )
                )

            if self.surface_embedding_head is not None:
                surface_embedding, surface_foreground = self.surface_embedding_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_token_start=patch_token_start,
                )
                predictions["surface_embedding"] = surface_embedding
                predictions["surface_foreground_logits"] = surface_foreground

        if not self.training:
            predictions["images"] = images
        return predictions


def _warn_if_rope_not_max(aggregator: nn.Module) -> None:
    for name, module in (("aggregator.patch_embed", aggregator.patch_embed), ("aggregator", aggregator)):
        rope_embed = getattr(module, "rope_embed", None)
        normalize_coords = getattr(rope_embed, "normalize_coords", None)
        if normalize_coords != "max":
            warnings.warn(
                f"{name} RoPE normalize_coords is {normalize_coords!r}; "
                "the released VGGT-Omega checkpoint was trained with 'max'.",
                stacklevel=2,
            )
