"""Dense and sparse supervision for a canonical mouse surface embedding."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dense_surface_embedding_loss(
    predicted_embedding: torch.Tensor,
    foreground_logits: torch.Tensor,
    target_vertex_ids: torch.Tensor,
    vertex_codebook: torch.Tensor,
    *,
    foreground_target: torch.Tensor | None = None,
    supervised_pixels: torch.Tensor | None = None,
    embedding_weight: float = 1.0,
    foreground_weight: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Supervise synthetic pixels using canonical mesh vertex IDs.

    ``target_vertex_ids`` uses ``-1`` for background or unavailable dense
    correspondence. A separate foreground target allows real masks to train
    the silhouette branch even when surface correspondence is unavailable.
    """
    if predicted_embedding.ndim != 5:
        raise ValueError("predicted_embedding must have shape [B,V,D,H,W]")
    if target_vertex_ids.shape != predicted_embedding.shape[:2] + predicted_embedding.shape[-2:]:
        raise ValueError("target_vertex_ids does not match embedding image shape")
    if foreground_logits.shape != predicted_embedding.shape[:2] + (1,) + predicted_embedding.shape[-2:]:
        raise ValueError("foreground_logits does not match embedding image shape")
    if vertex_codebook.shape[1] != predicted_embedding.shape[2]:
        raise ValueError("vertex_codebook dimension does not match predicted embedding")

    codebook = F.normalize(
        vertex_codebook.to(
            device=predicted_embedding.device,
            dtype=predicted_embedding.dtype,
        ),
        dim=-1,
        eps=1e-8,
    )
    dense_valid = target_vertex_ids >= 0
    if supervised_pixels is not None:
        dense_valid = dense_valid & supervised_pixels.to(
            device=dense_valid.device,
            dtype=torch.bool,
        )

    predicted_channels_last = predicted_embedding.permute(0, 1, 3, 4, 2)
    if dense_valid.any():
        target_embedding = codebook[target_vertex_ids[dense_valid]]
        cosine_similarity = (
            predicted_channels_last[dense_valid] * target_embedding
        ).sum(dim=-1)
        embedding_loss = (1.0 - cosine_similarity).mean()
        mean_cosine = cosine_similarity.mean()
    else:
        embedding_loss = predicted_embedding.mean() * 0.0
        mean_cosine = predicted_embedding.mean() * 0.0

    if foreground_target is None:
        foreground_target = target_vertex_ids >= 0
    foreground_target = foreground_target.to(
        device=foreground_logits.device,
        dtype=foreground_logits.dtype,
    )
    foreground_target = foreground_target.unsqueeze(2)
    if supervised_pixels is None:
        foreground_loss = F.binary_cross_entropy_with_logits(
            foreground_logits,
            foreground_target,
        )
    else:
        mask = supervised_pixels.to(
            device=foreground_logits.device,
            dtype=torch.bool,
        ).unsqueeze(2)
        foreground_loss = (
            F.binary_cross_entropy_with_logits(
                foreground_logits[mask],
                foreground_target[mask],
            )
            if mask.any()
            else foreground_logits.mean() * 0.0
        )

    total = (
        float(embedding_weight) * embedding_loss
        + float(foreground_weight) * foreground_loss
    )
    metrics = {
        "surface_embedding": embedding_loss.detach(),
        "surface_foreground": foreground_loss.detach(),
        "surface_mean_cosine": mean_cosine.detach(),
        "surface_dense_pixel_count": dense_valid.sum().detach(),
    }
    return total, metrics


def sparse_surface_anchor_loss(
    predicted_embedding: torch.Tensor,
    target_xy: torch.Tensor,
    target_support_vertex_ids: torch.Tensor,
    target_support_weights: torch.Tensor,
    valid: torch.Tensor,
    vertex_codebook: torch.Tensor,
    *,
    image_size: tuple[int, int],
    temperature: float = 0.05,
) -> torch.Tensor:
    """Match heterogeneous sparse landmarks to points or regions on the mesh.

    A broad label such as ``forepaw`` can map to many support vertices, while a
    precise label such as ``index_finger_3`` can map to a narrow vertex ring.
    """
    batch, views, _, _, _ = predicted_embedding.shape
    if target_xy.shape[:3] != valid.shape:
        raise ValueError("target_xy and valid shapes do not agree")
    if target_support_vertex_ids.shape[:2] != (
        batch,
        target_xy.shape[2],
    ):
        raise ValueError("target support map must have shape [B,K,M]")
    if target_support_weights.shape != target_support_vertex_ids.shape:
        raise ValueError("target support weights and vertex IDs must match")

    image_height, image_width = image_size
    grid_x = 2.0 * (target_xy[..., 0] + 0.5) / float(image_width) - 1.0
    grid_y = 2.0 * (target_xy[..., 1] + 0.5) / float(image_height) - 1.0
    sampling_grid = torch.stack([grid_x, grid_y], dim=-1)
    sampling_grid = sampling_grid.reshape(batch * views, -1, 1, 2)
    sampled = F.grid_sample(
        predicted_embedding.reshape(
            batch * views,
            predicted_embedding.shape[2],
            *predicted_embedding.shape[-2:],
        ),
        sampling_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    sampled = sampled.squeeze(-1).permute(0, 2, 1)
    sampled = sampled.reshape(batch, views, -1, predicted_embedding.shape[2])
    sampled = F.normalize(sampled, dim=-1, eps=1e-8)

    codebook = F.normalize(
        vertex_codebook.to(device=sampled.device, dtype=sampled.dtype),
        dim=-1,
        eps=1e-8,
    )
    support_ids = target_support_vertex_ids.clamp_min(0)
    support = codebook[support_ids][:, None].expand(-1, views, -1, -1, -1)
    distances = 1.0 - (sampled[..., None, :] * support).sum(dim=-1)

    support_valid = target_support_vertex_ids >= 0
    weights = target_support_weights.to(
        device=sampled.device,
        dtype=sampled.dtype,
    )
    weights = weights * support_valid.to(dtype=weights.dtype)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    weights = weights[:, None].expand(-1, views, -1, -1)
    soft_match = -float(temperature) * torch.log(
        (
            weights
            * torch.exp(-distances / max(float(temperature), 1e-6))
        ).sum(dim=-1).clamp_min(1e-8)
    )
    sparse_valid = valid & (weights.sum(dim=-1) > 0)
    if sparse_valid.any():
        return soft_match[sparse_valid].mean()
    return predicted_embedding.mean() * 0.0
