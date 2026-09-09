from __future__ import annotations

import torch
import torch.nn.functional as F

from training.surface_embedding_losses import (
    dense_surface_embedding_loss,
    sparse_surface_anchor_loss,
)
from vggt_omega.models.heads import SurfaceEmbeddingHead


def test_surface_embedding_head_shape_norm_and_gradient() -> None:
    torch.manual_seed(11)
    batch, views = 1, 3
    height, width = 64, 96
    patch_tokens = (height // 16) * (width // 16)
    tokens = torch.randn(
        batch,
        views,
        17 + patch_tokens,
        32,
        requires_grad=True,
    )
    cached = [None] * 24
    cached[-1] = tokens
    images = torch.randn(batch, views, 3, height, width)
    head = SurfaceEmbeddingHead(
        embedding_dim=16,
        dim_in=32,
        hidden_dim=32,
    )

    embedding, foreground = head(cached, images, patch_token_start=17)
    assert embedding.shape == (batch, views, 16, height // 4, width // 4)
    assert foreground.shape == (batch, views, 1, height // 4, width // 4)
    torch.testing.assert_close(
        torch.linalg.vector_norm(embedding, dim=2),
        torch.ones_like(embedding[:, :, 0]),
        atol=1e-5,
        rtol=1e-5,
    )

    loss = embedding.square().mean() + foreground.square().mean()
    loss.backward()
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()


def test_dense_surface_loss_prefers_correct_vertex_embedding() -> None:
    codebook = F.normalize(torch.randn(8, 4), dim=-1)
    vertex_ids = torch.tensor([[[[0, 1], [2, -1]]]])
    correct = torch.zeros(1, 1, 4, 2, 2)
    for y in range(2):
        for x in range(2):
            vertex_id = int(vertex_ids[0, 0, y, x])
            correct[0, 0, :, y, x] = codebook[max(vertex_id, 0)]
    wrong = torch.roll(correct, shifts=1, dims=2)
    foreground = torch.tensor([[[[[8.0, 8.0], [8.0, -8.0]]]]])

    correct_loss, correct_metrics = dense_surface_embedding_loss(
        correct,
        foreground,
        vertex_ids,
        codebook,
    )
    wrong_loss, _ = dense_surface_embedding_loss(
        wrong,
        foreground,
        vertex_ids,
        codebook,
    )
    assert correct_loss < wrong_loss
    assert correct_metrics["surface_mean_cosine"] > 0.99
    assert correct_metrics["surface_dense_pixel_count"].item() == 3


def test_sparse_surface_anchor_supports_dataset_specific_regions() -> None:
    codebook = F.normalize(torch.randn(6, 4), dim=-1)
    predicted = torch.zeros(1, 1, 4, 2, 2, requires_grad=True)
    predicted.data[0, 0, :, 0, 0] = codebook[1]
    predicted.data[0, 0, :, 1, 1] = codebook[4]
    target_xy = torch.tensor([[[[0.5, 0.5], [2.5, 2.5]]]])
    support_ids = torch.tensor([[[0, 1, 2], [4, 5, -1]]])
    support_weights = torch.tensor(
        [[[0.1, 0.8, 0.1], [0.9, 0.1, 0.0]]]
    )
    valid = torch.ones(1, 1, 2, dtype=torch.bool)

    loss = sparse_surface_anchor_loss(
        predicted,
        target_xy,
        support_ids,
        support_weights,
        valid,
        codebook,
        image_size=(4, 4),
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()
