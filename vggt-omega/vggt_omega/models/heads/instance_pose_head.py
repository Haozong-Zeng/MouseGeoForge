# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Instance-centric mouse detection, segmentation, and pose decoding.

The head deliberately treats decoder slots as unordered detections rather than
persistent identities.  Identity is represented by ``instance_reid_embedding``
and is supervised separately across views/time.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiInstanceMouseHead(nn.Module):
    """Decode a variable-size set of mouse instances from VGGT dense tokens.

    Coordinates and boxes are normalized to ``[0, 1]``. Local heatmaps cover a
    square window around each coarse keypoint and refine its final coordinate.
    """

    def __init__(
        self,
        num_keypoints: int = 23,
        num_queries: int = 6,
        dim_in: int = 2048,
        hidden_dim: int = 256,
        patch_size: int = 16,
        output_stride: int = 4,
        intermediate_layer_idx: tuple[int, ...] = (4, 11, 17, 23),
        decoder_layers: int = 2,
        keypoint_decoder_layers: int = 1,
        num_heads: int = 8,
        reid_dim: int = 64,
        local_heatmap_size: int = 8,
        local_window_radius: float = 0.10,
    ) -> None:
        super().__init__()
        if num_keypoints <= 0 or num_queries <= 0:
            raise ValueError("num_keypoints and num_queries must be positive")
        if patch_size % output_stride:
            raise ValueError("patch_size must be divisible by output_stride")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if local_heatmap_size < 2:
            raise ValueError("local_heatmap_size must be at least 2")

        self.num_keypoints = int(num_keypoints)
        self.num_queries = int(num_queries)
        self.patch_size = int(patch_size)
        self.output_stride = int(output_stride)
        self.hidden_dim = int(hidden_dim)
        self.intermediate_layer_idx = tuple(intermediate_layer_idx)
        self.local_heatmap_size = int(local_heatmap_size)
        self.local_window_radius = float(local_window_radius)

        self.token_norms = nn.ModuleList(
            nn.LayerNorm(dim_in, eps=1e-5) for _ in self.intermediate_layer_idx
        )
        self.lateral_projections = nn.ModuleList(
            nn.Conv2d(dim_in, hidden_dim, kernel_size=1)
            for _ in self.intermediate_layer_idx
        )
        self.pyramid_refine = nn.ModuleList(
            _ConvBlock(hidden_dim) for _ in self.intermediate_layer_idx
        )

        instance_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.instance_decoder = nn.TransformerDecoder(
            instance_layer,
            num_layers=decoder_layers,
            norm=nn.LayerNorm(hidden_dim),
        )
        keypoint_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.keypoint_decoder = nn.TransformerDecoder(
            keypoint_layer,
            num_layers=keypoint_decoder_layers,
            norm=nn.LayerNorm(hidden_dim),
        )

        self.instance_queries = nn.Embedding(num_queries, hidden_dim)
        self.keypoint_queries = nn.Embedding(num_keypoints, hidden_dim)
        self.objectness = nn.Linear(hidden_dim, 1)
        self.box = _MLP(hidden_dim, hidden_dim, 4, 3)
        self.mask_embedding = _MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        self.reid_embedding = _MLP(hidden_dim, hidden_dim, reid_dim, 2)
        self.coarse_keypoint_xy = _MLP(hidden_dim, hidden_dim, 2, 3)
        self.keypoint_visibility = nn.Linear(hidden_dim, 1)
        self.keypoint_log_sigma = nn.Linear(hidden_dim, 1)
        self.local_query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.local_pixel_projection = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self._init_predictions()

    def _init_predictions(self) -> None:
        nn.init.constant_(self.objectness.bias, -2.0)
        nn.init.zeros_(self.box.layers[-1].weight)
        nn.init.zeros_(self.box.layers[-1].bias)
        nn.init.zeros_(self.coarse_keypoint_xy.layers[-1].weight)
        nn.init.zeros_(self.coarse_keypoint_xy.layers[-1].bias)
        nn.init.constant_(self.keypoint_visibility.bias, 1.0)
        initial_sigma = math.log(math.expm1(0.03))
        nn.init.constant_(self.keypoint_log_sigma.bias, initial_sigma)

    def forward(
        self,
        aggregated_tokens_list: list[torch.Tensor | None],
        images: torch.Tensor,
        patch_token_start: int,
    ) -> dict[str, torch.Tensor]:
        if patch_token_start is None:
            raise ValueError("patch_token_start is required")
        batch, views, _, image_h, image_w = images.shape
        patch_h, patch_w = image_h // self.patch_size, image_w // self.patch_size
        projected = self._project_cached_tokens(
            aggregated_tokens_list,
            patch_token_start,
            batch,
            views,
            patch_h,
            patch_w,
        )
        memory_feature = torch.stack(projected, dim=0).mean(dim=0)
        pixel_feature = self._build_pixel_pyramid(projected, image_h, image_w)

        frame_count = batch * views
        position = _sine_position_encoding(
            patch_h,
            patch_w,
            self.hidden_dim,
            dtype=memory_feature.dtype,
            device=memory_feature.device,
        )
        memory = memory_feature.flatten(2).transpose(1, 2) + position
        instance_queries = self.instance_queries.weight.unsqueeze(0).expand(
            frame_count, -1, -1
        )
        instance_tokens = self.instance_decoder(instance_queries, memory)

        objectness_logits = self.objectness(instance_tokens).squeeze(-1)
        raw_box = self.box(instance_tokens)
        box_center = raw_box[..., :2].sigmoid()
        box_size = 0.02 + 0.98 * raw_box[..., 2:].sigmoid()
        boxes = torch.cat((box_center, box_size), dim=-1)
        mask_embedding = self.mask_embedding(instance_tokens)
        mask_logits = torch.einsum("nqd,ndhw->nqhw", mask_embedding, pixel_feature)
        reid = F.normalize(self.reid_embedding(instance_tokens), dim=-1, eps=1e-6)

        keypoint_seed = (
            instance_tokens[:, :, None, :]
            + self.keypoint_queries.weight[None, None, :, :]
        )
        keypoint_tokens = self.keypoint_decoder(
            keypoint_seed.flatten(1, 2),
            memory,
        ).view(frame_count, self.num_queries, self.num_keypoints, self.hidden_dim)
        coarse_xy = self.coarse_keypoint_xy(keypoint_tokens).sigmoid()
        visibility_logits = self.keypoint_visibility(keypoint_tokens).squeeze(-1)
        sigma = 1e-3 + F.softplus(self.keypoint_log_sigma(keypoint_tokens).squeeze(-1))
        local_logits, refined_xy = self._local_refinement(
            pixel_feature,
            keypoint_tokens,
            coarse_xy,
        )

        def restore(value: torch.Tensor) -> torch.Tensor:
            return value.view(batch, views, *value.shape[1:])

        return {
            "instance_objectness_logits": restore(objectness_logits),
            "instance_boxes_cxcywh": restore(boxes),
            "instance_mask_logits": restore(mask_logits),
            "instance_reid_embedding": restore(reid),
            "instance_keypoint_coarse_xy": restore(coarse_xy),
            "instance_keypoint_xy": restore(refined_xy),
            "instance_keypoint_visibility_logits": restore(visibility_logits),
            "instance_keypoint_sigma": restore(sigma),
            "instance_keypoint_local_heatmap_logits": restore(local_logits),
        }

    def _project_cached_tokens(
        self,
        aggregated_tokens_list: list[torch.Tensor | None],
        patch_token_start: int,
        batch: int,
        views: int,
        patch_h: int,
        patch_w: int,
    ) -> list[torch.Tensor]:
        features = []
        expected = patch_h * patch_w
        for norm, projection, layer_idx in zip(
            self.token_norms,
            self.lateral_projections,
            self.intermediate_layer_idx,
            strict=True,
        ):
            tokens = aggregated_tokens_list[layer_idx]
            if tokens is None:
                raise ValueError(
                    f"Aggregator did not cache layer {layer_idx}, which the instance head needs"
                )
            tokens = tokens[:, :, patch_token_start:]
            if tokens.shape[2] != expected:
                raise ValueError(
                    f"Layer {layer_idx} has {tokens.shape[2]} patch tokens; expected {expected}"
                )
            tokens = tokens.float() if tokens.dtype != torch.float32 else tokens
            feature = norm(tokens).reshape(
                batch * views, patch_h, patch_w, tokens.shape[-1]
            ).permute(0, 3, 1, 2)
            features.append(projection(feature))
        return features

    def _build_pixel_pyramid(
        self,
        projected: list[torch.Tensor],
        image_h: int,
        image_w: int,
    ) -> torch.Tensor:
        target_sizes = [
            (image_h // 4, image_w // 4),
            (image_h // 8, image_w // 8),
            (image_h // 16, image_w // 16),
            (max(image_h // 32, 1), max(image_w // 32, 1)),
        ]
        levels = [
            F.interpolate(feature, size=size, mode="bilinear", align_corners=False)
            for feature, size in zip(projected, target_sizes, strict=True)
        ]
        fused = self.pyramid_refine[-1](levels[-1])
        for index in range(len(levels) - 2, -1, -1):
            fused = F.interpolate(
                fused,
                size=levels[index].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            fused = self.pyramid_refine[index](fused + levels[index])
        return fused

    def _local_refinement(
        self,
        pixel_feature: torch.Tensor,
        keypoint_tokens: torch.Tensor,
        coarse_xy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frame_count, queries, keypoints = coarse_xy.shape[:3]
        size = self.local_heatmap_size
        axis = torch.linspace(
            -self.local_window_radius,
            self.local_window_radius,
            size,
            dtype=coarse_xy.dtype,
            device=coarse_xy.device,
        )
        offset_y, offset_x = torch.meshgrid(axis, axis, indexing="ij")
        offsets = torch.stack((offset_x, offset_y), dim=-1)
        sample_grid = coarse_xy[..., None, None, :] * 2.0 - 1.0
        sample_grid = sample_grid + offsets[None, None, None] * 2.0
        sample_grid = sample_grid.reshape(frame_count, queries * keypoints * size, size, 2)
        local_pixels = F.grid_sample(
            self.local_pixel_projection(pixel_feature),
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        local_pixels = local_pixels.view(
            frame_count,
            self.hidden_dim,
            queries,
            keypoints,
            size,
            size,
        ).permute(0, 2, 3, 1, 4, 5)
        local_queries = F.normalize(
            self.local_query_projection(keypoint_tokens), dim=-1, eps=1e-6
        )
        local_pixels = F.normalize(local_pixels, dim=3, eps=1e-6)
        logits = torch.einsum("nqkd,nqkdhw->nqkhw", local_queries, local_pixels)
        probabilities = logits.flatten(-2).softmax(dim=-1).view_as(logits)
        delta = (probabilities[..., None] * offsets).sum(dim=(-3, -2))
        refined = (coarse_xy + delta).clamp(0.0, 1.0)
        return logits, refined


class _ConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class _MLP(nn.Module):
    def __init__(self, dim_in: int, hidden_dim: int, dim_out: int, layers: int) -> None:
        super().__init__()
        dimensions = [dim_in] + [hidden_dim] * (layers - 1) + [dim_out]
        self.layers = nn.ModuleList(
            nn.Linear(dimensions[index], dimensions[index + 1])
            for index in range(layers)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            value = layer(value)
            if index + 1 < len(self.layers):
                value = F.gelu(value)
        return value


def _sine_position_encoding(
    height: int,
    width: int,
    channels: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return a deterministic 2D sine/cosine encoding with shape [1, HW, C]."""
    if channels % 4:
        raise ValueError("Position encoding needs channels divisible by four")
    quarter = channels // 4
    frequency = torch.arange(quarter, dtype=dtype, device=device)
    frequency = 1.0 / (10_000 ** (frequency / max(quarter - 1, 1)))
    y = torch.linspace(0.0, 1.0, height, dtype=dtype, device=device)
    x = torch.linspace(0.0, 1.0, width, dtype=dtype, device=device)
    y_phase = y[:, None] * frequency[None, :] * (2.0 * math.pi)
    x_phase = x[:, None] * frequency[None, :] * (2.0 * math.pi)
    y_embed = torch.cat((y_phase.sin(), y_phase.cos()), dim=-1)
    x_embed = torch.cat((x_phase.sin(), x_phase.cos()), dim=-1)
    encoding = torch.cat(
        (
            y_embed[:, None, :].expand(-1, width, -1),
            x_embed[None, :, :].expand(height, -1, -1),
        ),
        dim=-1,
    )
    return encoding.reshape(1, height * width, channels)
