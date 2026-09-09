# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SurfaceEmbeddingHead(nn.Module):
    """Decode a category-level surface embedding and foreground mask per view."""

    def __init__(
        self,
        embedding_dim: int = 16,
        dim_in: int = 2048,
        patch_size: int = 16,
        hidden_dim: int = 256,
        output_stride: int = 4,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if patch_size % output_stride:
            raise ValueError(
                f"patch_size={patch_size} must be divisible by output_stride={output_stride}"
            )

        self.embedding_dim = int(embedding_dim)
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
        self.embedding_projection = nn.Conv2d(
            hidden_dim,
            embedding_dim,
            kernel_size=1,
        )
        self.foreground_projection = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self._init_predictions()

    def _init_predictions(self) -> None:
        nn.init.normal_(self.embedding_projection.weight, std=1e-3)
        nn.init.zeros_(self.embedding_projection.bias)
        nn.init.normal_(self.foreground_projection.weight, std=1e-3)
        nn.init.constant_(self.foreground_projection.bias, -2.0)

    def forward(
        self,
        aggregated_tokens_list: list[torch.Tensor | None],
        images: torch.Tensor,
        patch_token_start: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = aggregated_tokens_list[-1]
        if tokens is None:
            raise ValueError(
                "Aggregator did not cache the final layer, which "
                "SurfaceEmbeddingHead needs."
            )
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
        output_size = (height // self.output_stride, width // self.output_stride)

        embedding = self.embedding_projection(patch_features)
        embedding = F.interpolate(
            embedding,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        embedding = F.normalize(embedding, dim=1, eps=1e-8)
        embedding = embedding.view(
            batch_size,
            num_frames,
            self.embedding_dim,
            *output_size,
        )

        foreground_logits = self.foreground_projection(patch_features)
        foreground_logits = F.interpolate(
            foreground_logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        foreground_logits = foreground_logits.view(
            batch_size,
            num_frames,
            1,
            *output_size,
        )
        return embedding, foreground_logits


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
