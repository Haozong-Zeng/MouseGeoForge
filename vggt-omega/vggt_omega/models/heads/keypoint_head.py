# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class KeypointHead(nn.Module):
    """Decode per-view keypoint heatmaps from the final VGGT patch tokens."""

    def __init__(
        self,
        num_keypoints: int,
        dim_in: int = 2048,
        patch_size: int = 16,
        hidden_dim: int = 256,
        output_stride: int = 4,
    ) -> None:
        super().__init__()
        if num_keypoints <= 0:
            raise ValueError(f"num_keypoints must be positive, got {num_keypoints}")
        if patch_size % output_stride:
            raise ValueError(
                f"patch_size={patch_size} must be divisible by output_stride={output_stride}"
            )

        self.num_keypoints = int(num_keypoints)
        self.patch_size = int(patch_size)
        self.output_stride = int(output_stride)
        self.norm = nn.LayerNorm(dim_in, eps=1e-5)
        self.patch_projection = nn.Sequential(
            nn.Conv2d(dim_in, hidden_dim, kernel_size=1),
            nn.GroupNorm(16, hidden_dim),
            nn.GELU(),
            _ResidualConvBlock(hidden_dim),
            _ResidualConvBlock(hidden_dim),
        )
        self.heatmap_projection = nn.Conv2d(hidden_dim, num_keypoints, kernel_size=1)
        self.frame_projection = nn.Sequential(
            nn.LayerNorm(dim_in, eps=1e-5),
            nn.Linear(dim_in, hidden_dim),
            nn.GELU(),
        )
        self.visibility_projection = nn.Linear(hidden_dim, num_keypoints)
        self.log_sigma_projection = nn.Linear(hidden_dim, num_keypoints)
        self._init_predictions()

    def _init_predictions(self) -> None:
        nn.init.normal_(self.heatmap_projection.weight, std=1e-3)
        nn.init.zeros_(self.heatmap_projection.bias)
        nn.init.normal_(self.visibility_projection.weight, std=1e-3)
        nn.init.constant_(self.visibility_projection.bias, 1.0)
        nn.init.normal_(self.log_sigma_projection.weight, std=1e-3)
        initial_sigma = math.log(math.expm1(4.0 - 1.0))
        nn.init.constant_(self.log_sigma_projection.bias, initial_sigma)

    def forward(
        self,
        aggregated_tokens_list: list[torch.Tensor | None],
        images: torch.Tensor,
        patch_token_start: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = aggregated_tokens_list[-1]
        if tokens is None:
            raise ValueError("Aggregator did not cache the final layer, which KeypointHead needs.")
        if tokens.dtype != torch.float32:
            tokens = tokens.float()

        batch_size, num_frames, _, channels = tokens.shape
        height, width = images.shape[-2:]
        patch_h = height // self.patch_size
        patch_w = width // self.patch_size
        patch_tokens = tokens[:, :, patch_token_start:]
        if patch_tokens.shape[2] != patch_h * patch_w:
            raise ValueError(
                "Patch-token count does not match the input image grid: "
                f"tokens={patch_tokens.shape[2]} grid={patch_h}x{patch_w}"
            )

        patch_features = self.norm(patch_tokens)
        patch_features = patch_features.reshape(
            batch_size * num_frames,
            patch_h,
            patch_w,
            channels,
        ).permute(0, 3, 1, 2)
        patch_features = self.patch_projection(patch_features)
        heatmap_logits = self.heatmap_projection(patch_features)
        output_size = (height // self.output_stride, width // self.output_stride)
        heatmap_logits = F.interpolate(
            heatmap_logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        heatmap_logits = heatmap_logits.view(
            batch_size,
            num_frames,
            self.num_keypoints,
            *output_size,
        )

        frame_features = self.frame_projection(patch_tokens.mean(dim=2))
        visibility_logits = self.visibility_projection(frame_features)
        pixel_sigma = 1.0 + F.softplus(self.log_sigma_projection(frame_features))
        return heatmap_logits, visibility_logits, pixel_sigma


class _ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(16, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(16, channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.gelu(value + self.block(value))
