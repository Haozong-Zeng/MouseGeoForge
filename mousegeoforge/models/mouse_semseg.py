"""Portable Mask2Former-inspired binary semantic decoder.

Pure PyTorch multi-scale deformable sampling (grid_sample), masked query
attention, and dense Hungarian mask supervision. This is not an MMSeg model
or an exact reproduction of the Depth Anything semantic configuration.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


def position_encoding(height, width, channels, device, dtype):
    if channels % 4:
        raise ValueError('Position channels must be divisible by four')
    y, x = torch.meshgrid(torch.arange(height, device=device),
                          torch.arange(width, device=device), indexing='ij')
    omega = 10000 ** (-torch.arange(channels // 4, device=device).float() / (channels // 4))
    angles = [((coord.flatten().float() + .5) / size * 2 * math.pi)[:, None] * omega
              for coord, size in [(x, width), (y, height)]]
    return torch.cat([a.sin() for a in angles] + [a.cos() for a in angles], -1).to(dtype)


class MultiScaleDeformableAttention(nn.Module):
    def __init__(self, dim, heads=8, levels=3, points=4):
        super().__init__()
        if dim % heads:
            raise ValueError('Attention dimension must divide by heads')
        self.dim, self.heads, self.levels, self.points = dim, heads, levels, points
        self.offsets = nn.Linear(dim, heads * levels * points * 2)
        self.weights = nn.Linear(dim, heads * levels * points)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)
        nn.init.zeros_(self.offsets.weight)
        theta = torch.arange(heads).float() * (2 * math.pi / heads)
        radial = torch.stack([theta.cos(), theta.sin()], -1)
        radial = radial / radial.abs().amax(-1, keepdim=True)
        radial = radial[:, None, None].repeat(1, levels, points, 1)
        radial *= torch.arange(1, points + 1)[None, None, :, None]
        with torch.no_grad():
            self.offsets.bias.copy_(radial.flatten())
        nn.init.zeros_(self.weights.weight)
        nn.init.zeros_(self.weights.bias)

    def forward(self, query, memory, shapes, reference):
        b, n, c = query.shape
        offsets = self.offsets(query).reshape(b, n, self.heads, self.levels, self.points, 2)
        weights = self.weights(query).reshape(b, n, self.heads, -1).softmax(-1)
        weights = weights.reshape(b, n, self.heads, self.levels, self.points)
        values = self.value(memory)
        result = torch.zeros((b, self.heads, c // self.heads, n), device=query.device, dtype=torch.float32)
        cursor = 0
        for level, (h, w) in enumerate(shapes):
            v = values[:, cursor:cursor + h * w]
            v = v.reshape(b, h, w, self.heads, c // self.heads).permute(0, 3, 4, 1, 2)
            v = v.reshape(b * self.heads, c // self.heads, h, w)
            normalizer = query.new_tensor([w, h])
            grid = reference[None, :, None, None, :] + offsets[:, :, :, level] / normalizer
            grid = grid.permute(0, 2, 1, 3, 4).reshape(b * self.heads, n, self.points, 2)
            # FP32 grid_sample is supported on CPU and CUDA, including BF16 callers.
            sampled = F.grid_sample(v.float(), grid.float() * 2 - 1,
                                    mode='bilinear', padding_mode='zeros', align_corners=False)
            sampled = sampled.reshape(b, self.heads, c // self.heads, n, self.points)
            weight = weights[:, :, :, level].permute(0, 2, 1, 3)[:, :, None]
            result = result + (sampled * weight.float()).sum(-1)
            cursor += h * w
        return self.output(result.permute(0, 3, 1, 2).reshape(b, n, c).to(query.dtype))


class DeformableLayer(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attention = MultiScaleDeformableAttention(dim, heads)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x, position, shapes, references):
        x = self.norm1(x + self.attention(x + position, x, shapes, references))
        return self.norm2(x + self.ffn(x))


class QueryLayer(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.cross = nn.MultiheadAttention(dim, heads, dropout=0, batch_first=True)
        self.self_attention = nn.MultiheadAttention(dim, heads, dropout=0, batch_first=True)
        # A constant key bias cancels in softmax. Mask its analytically-zero
        # gradient rather than letting Adam amplify FP32 reduction roundoff.
        def mask_key_bias(gradient):
            gradient = gradient.clone()
            gradient[dim:2 * dim] = 0
            return gradient
        for attention in [self.cross, self.self_attention]:
            attention.in_proj_bias.register_hook(mask_key_bias)
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x, query_position, memory, position, mask):
        # An entirely masked query would produce NaNs; reopen its memory support.
        mask = mask.detach().clone()
        mask[mask.all(-1)] = False
        mask = mask.repeat_interleave(self.heads, 0)
        y, _ = self.cross(x + query_position, memory + position, memory,
                          attn_mask=mask, need_weights=False)
        x = self.norms[0](x + y)
        y, _ = self.self_attention(x + query_position, x + query_position, x, need_weights=False)
        x = self.norms[1](x + y)
        return self.norms[2](x + self.ffn(x))


class MouseMaskDecoder(nn.Module):
    def __init__(self, in_channels=1024, dim=256, queries=16, heads=8,
                 pixel_layers=3, query_layers=3):
        super().__init__()
        groups = math.gcd(32, max(1, dim // 4))
        self.projections = nn.ModuleList([nn.Sequential(nn.Conv2d(in_channels, dim, 1),
                                                       nn.GroupNorm(groups, dim)) for _ in range(4)])
        self.level_embedding = nn.Parameter(torch.randn(3, dim) * .02)
        self.pixel_layers = nn.ModuleList([DeformableLayer(dim, heads) for _ in range(pixel_layers)])
        self.high_resolution = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1),
                                              nn.GroupNorm(groups, dim), nn.GELU())
        self.mask_features = nn.Conv2d(dim, dim, 1)
        self.query_content = nn.Embedding(queries, dim)
        self.query_position = nn.Embedding(queries, dim)
        self.query_layers = nn.ModuleList([QueryLayer(dim, heads) for _ in range(query_layers)])
        self.query_norm = nn.LayerNorm(dim)
        self.classes = nn.Linear(dim, 3)  # background, mouse, no-object
        self.mask_embedding = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def predict(self, query, mask_features):
        query = self.query_norm(query)
        return dict(classes=self.classes(query),
                    masks=torch.einsum('bqc,bchw->bqhw', self.mask_embedding(query), mask_features))

    def forward(self, features):
        if len(features) != 4:
            raise ValueError('Exactly four feature maps required')
        scales = [4, 2, 1, .5]
        pyramid = [F.interpolate(proj(f), size=(max(1, round(f.shape[-2] * s)),
                                                max(1, round(f.shape[-1] * s))),
                                 mode='bilinear', align_corners=False)
                   for f, proj, s in zip(features, self.projections, scales)]
        # Encode three coarser scales; use the finest one as a lateral skip.
        levels = pyramid[1:]
        shapes = [x.shape[-2:] for x in levels]
        positions, references = [], []
        for i, (h, w) in enumerate(shapes):
            positions.append(position_encoding(h, w, levels[i].shape[1], levels[i].device,
                                               levels[i].dtype) + self.level_embedding[i])
            yy, xx = torch.meshgrid(torch.arange(h, device=levels[i].device),
                                    torch.arange(w, device=levels[i].device), indexing='ij')
            references.append(torch.stack([(xx + .5) / w, (yy + .5) / h], -1).reshape(-1, 2))
        position = torch.cat(positions)[None]
        reference = torch.cat(references)
        memory = torch.cat([x.flatten(2).transpose(1, 2) for x in levels], 1)
        for layer in self.pixel_layers:
            memory = layer(memory, position, shapes, reference)
        encoded = []; cursor = 0
        for h, w in shapes:
            encoded.append(memory[:, cursor:cursor + h * w].transpose(1, 2).reshape(
                memory.shape[0], -1, h, w))
            cursor += h * w
        high = self.high_resolution(pyramid[0] + F.interpolate(encoded[0], size=pyramid[0].shape[-2:],
                                                               mode='bilinear', align_corners=False))
        mask_features = self.mask_features(high)
        query = self.query_content.weight[None].expand(memory.shape[0], -1, -1)
        query_position = self.query_position.weight[None]
        predictions = [self.predict(query, mask_features)]
        for i, layer in enumerate(self.query_layers):
            level = i % 3
            h, w = shapes[level]
            mask = F.interpolate(predictions[-1]['masks'].float(), (h, w), mode='bilinear',
                                 align_corners=False).flatten(2).sigmoid() < .5
            mem = encoded[level].flatten(2).transpose(1, 2)
            pos = positions[level][None]
            query = layer(query, query_position, mem, pos, mask)
            predictions.append(self.predict(query, mask_features))
        return predictions


def semantic_scores(prediction, size=None):
    # Exclude no-object; semantic class scores, not independent instance masks.
    cls = prediction['classes'].float().softmax(-1)[..., :2]
    scores = torch.einsum('bqc,bqhw->bchw', cls, prediction['masks'].float().sigmoid())
    if size is not None:
        scores = F.interpolate(scores, size, mode='bilinear', align_corners=False)
    return scores


def hungarian_loss(predictions, targets):
    """Dense fixed-grid costs/losses; per-image then batch mean, with aux outputs.

    Labels: 0 background, 1 visible mouse, 255 unknown. Padding is an explicitly
    supervised background target supplied by the dataset. No uncertainty-point
    subsampling or official MMSeg loss normalization is claimed here.
    """
    from scipy.optimize import linear_sum_assignment
    layer_losses = []
    for pred in predictions:
        target = F.interpolate(targets[:, None].float(), pred['masks'].shape[-2:], mode='nearest')[:, 0].long()
        image_losses = []
        for cls, masks, label in zip(pred['classes'].float(), pred['masks'].float(), target):
            valid = label.flatten() != 255
            if not valid.any():
                image_losses.append((cls.sum() + masks.sum()) * 0)
                continue
            ids = torch.unique(label[label != 255])
            if not all(int(i) in (0, 1) for i in ids):
                raise ValueError('Unexpected semantic label')
            truth = torch.stack([(label == i).flatten()[valid].float() for i in ids])
            logits = masks.flatten(1)[:, valid]
            probability = logits.sigmoid()
            bce_cost = F.softplus(logits).mean(-1)[:, None] - logits @ truth.T / truth.shape[-1]
            dice_cost = 1 - (2 * probability @ truth.T + 1) / (
                probability.sum(-1)[:, None] + truth.sum(-1)[None] + 1)
            cost = 2 * (-cls.softmax(-1)[:, ids]) + 5 * bce_cost + 5 * dice_cost
            query_indices, truth_indices = linear_sum_assignment(cost.detach().cpu().numpy())
            qi = torch.as_tensor(query_indices, device=cls.device)
            ti = torch.as_tensor(truth_indices, device=cls.device)
            class_target = torch.full((len(cls),), 2, dtype=torch.long, device=cls.device)
            class_target[qi] = ids[ti]
            ce = F.cross_entropy(cls, class_target, weight=cls.new_tensor([1., 1., .1]))
            mask_loss = F.binary_cross_entropy_with_logits(logits[qi], truth[ti])
            p = probability[qi]; t = truth[ti]
            dice = (1 - (2 * (p * t).sum(-1) + 1) / (p.sum(-1) + t.sum(-1) + 1)).mean()
            image_losses.append(2 * ce + 5 * mask_loss + 5 * dice)
        layer_losses.append(torch.stack(image_losses).mean())
    return torch.stack(layer_losses).mean()
