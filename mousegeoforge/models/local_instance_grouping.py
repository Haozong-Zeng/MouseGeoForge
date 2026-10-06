"""Isolated RGB-evidence grouping experiment, not a production model default.

Mechanism references: CenterGroup (candidate grouping) and Group Pose (within
instance interaction); this is an original small probe, not either paper's
reproduction. Both arms have identical parameters. Their sole architectural
difference is whether five point-type tokens may interact in self-attention.
GT is accepted ONLY by the separate loss. Missing labels are never negatives.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

from .regional_peak_candidates import regional_top_peaks


def rotate_mask_centroids(centers, valid, quarter_turns):
    """Masks use pixel-cell centers (+.5), unlike legacy point pixel indices."""
    if quarter_turns not in (0, 1, 2, 3):
        raise ValueError('Expected zero to three quarter turns')
    out = centers.clone()
    for _ in range(quarter_turns):
        out = torch.stack([out[..., 1], 1. - out[..., 0]], -1)
    return torch.where(valid[..., None], out, centers)


class LocalInstanceGrouping(nn.Module):
    def __init__(self, *, interaction=False, max_instances=2, width=64):
        super().__init__()
        if max_instances < 1 or width % 4:
            raise ValueError('Positive capacity and width divisible by four required')
        self.interaction = bool(interaction)
        self.max_instances = max_instances
        self.width = width
        self.queries = nn.Parameter(torch.randn(max_instances, width) * .02)
        self.types = nn.Parameter(torch.randn(5, width) * .02)
        self.memory = nn.Sequential(nn.Linear(34, width), nn.GELU(), nn.LayerNorm(width))
        self.instance_attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.instance_norm = nn.LayerNorm(width)
        self.center = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 2))
        self.objectness = nn.Linear(width, 1)
        self.candidate = nn.Sequential(nn.Linear(35, width), nn.GELU(), nn.LayerNorm(width))
        self.point_attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.point_norm = nn.LayerNorm(width)
        self.point_mlp = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))
        self.relative = nn.Sequential(nn.Linear(4, width), nn.GELU(), nn.Linear(width, 1))
        self.null_score = nn.Linear(width, 1)
        self.query_key = nn.Linear(width, width)

    def forward(self, evidence, image_hw):
        """Input is predicted local evidence only; no targets/counts/GT proposals."""
        features = evidence['features']
        if features.ndim != 3 or features.shape[0] != 32:
            raise ValueError('Expected stride-four 32-channel visual features')
        ih, iw = map(int, image_hw)
        h, w = features.shape[-2:]
        valid = evidence['spatial_valid']
        device = features.device
        # Regional extraction is discrete; local CE trains heatmaps, while
        # feature gathering and attention still backpropagate to the backbone.
        candidates = regional_top_peaks(evidence, image_hw, count=6, radius_px=8.)
        xy = candidates['xy'].to(device)
        available = candidates['available'].to(device)
        probability = candidates['probability'].to(device)
        visual = F.layer_norm(features.permute(1, 2, 0), (32,))

        # Pool only real image cells. Absolute coordinates retain orientation;
        # there are no camera labels and no externally specified mouse count.
        vh = int(valid.any(1).sum())
        vw = int(valid.any(0).sum())
        pooled = F.adaptive_avg_pool2d(visual[:vh, :vw].permute(2, 0, 1)[None],
                                     (min(24, vh), min(24, vw)))[0].permute(1, 2, 0)
        gh, gw = pooled.shape[:2]
        yy, xx = torch.meshgrid((torch.arange(gh, device=device) + .5) / gh,
                                (torch.arange(gw, device=device) + .5) / gw, indexing='ij')
        memory = self.memory(torch.cat([pooled, torch.stack([xx, yy], -1)], -1).reshape(1, -1, 34))
        queries = self.queries[None]
        instance, _ = self.instance_attention(queries, memory, memory, need_weights=False)
        instance = self.instance_norm(instance + queries)[0]
        centers = self.center(instance).sigmoid()
        object_logits = self.objectness(instance).squeeze(-1)

        # Our grid is cell-centered at (j+.5)*4; convert exactly to indices.
        ix = (xy[..., 0] * iw / 4 - .5).round().long().clamp(0, w - 1)
        iy = (xy[..., 1] * ih / 4 - .5).round().long().clamp(0, h - 1)
        feature = visual[iy, ix]
        relative_probability = probability / probability.sum(-1, keepdim=True).clamp_min(1e-12)
        prior = relative_probability.clamp_min(1e-12).log()
        keys = self.candidate(torch.cat([feature, xy, prior[..., None] / 12.], -1))
        point = instance[:, None] + self.types[None]
        initial_logits = torch.einsum('qkd,knd->qkn', point, keys) / math.sqrt(self.width)
        initial_logits = initial_logits.masked_fill(~available[None], -1e4)
        weight = initial_logits.softmax(-1) * available[None]
        weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-12)
        point = point + torch.einsum('qkn,knd->qkd', weight, keys)
        diagonal_only = ~torch.eye(5, dtype=torch.bool, device=device)
        attended, _ = self.point_attention(point, point, point,
                                           attn_mask=None if self.interaction else diagonal_only,
                                           need_weights=False)
        point = self.point_norm(point + attended)
        point = point + self.point_mlp(point)
        delta = xy[None] - centers[:, None, None]
        geometry = torch.cat([delta, delta.abs()], -1)
        logits = torch.einsum('qkd,knd->qkn', self.query_key(point), keys) / math.sqrt(self.width)
        logits = logits + self.relative(geometry).squeeze(-1) + .25 * prior[None]
        logits = logits.masked_fill(~available[None], -torch.inf)
        logits = torch.cat([logits, self.null_score(point)], -1)
        return dict(centers=centers, object_logits=object_logits, assignment_logits=logits,
                    candidate_xy=xy, candidate_available=available,
                    candidate_probability=probability)


def assignment_targets(output, target, eligible, image_hw, radius_px=16.):
    """Known points beyond every candidate train candidate-rejection, not visibility.

    Unknown points have zero weight and receive neither candidate nor null loss.
    Nearby alternatives use an 8px Gaussian; there is no regression to an average
    between different feet. Coordinates remain discrete candidates at inference.
    """
    if target.ndim != 3 or target.shape[1:] != (5, 2) or eligible.shape != target.shape[:-1]:
        raise ValueError('Expected instances,5,2 and eligibility')
    if not torch.isfinite(target[eligible]).all():
        raise ValueError('Known coordinates must be finite')
    scale = target.new_tensor([image_hw[1], image_hw[0]])
    safe_target = torch.where(eligible[..., None], target, torch.zeros_like(target))
    distance = ((output['candidate_xy'][None] - safe_target[:, :, None]) * scale).square().sum(-1)
    close = (distance <= radius_px ** 2) & output['candidate_available'][None] & eligible[..., None]
    weights = (-.5 * distance / 8. ** 2).exp() * close
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
    null = eligible & ~close.any(-1)
    return torch.cat([weights, null[..., None].to(weights.dtype)], -1)


def grouping_loss(output, target, eligible, instance_valid, mask_centroid, centroid_valid,
                  image_hw, *, annotation_complete=True):
    """Permutation-invariant supervision; all GT use is confined to this loss."""
    if not torch.equal(instance_valid.bool(), centroid_valid.bool()):
        raise ValueError('This audited pilot requires a valid mask centroid for each active instance')
    if not torch.isfinite(mask_centroid[centroid_valid]).all():
        raise ValueError('Known centers must be finite')
    ids = instance_valid.nonzero(as_tuple=False).flatten()
    if len(ids) > len(output['object_logits']):
        raise ValueError('Instances exceed configured capacity; never silently discard a mouse')
    zero = output['object_logits'].sum() * 0
    object_target = torch.zeros_like(output['object_logits'])
    distributions = assignment_targets(output, target, eligible, image_hw)
    if len(ids):
        cost = torch.cdist(output['centers'].detach(), mask_centroid[ids], p=1)
        query_np, target_np = linear_sum_assignment(cost.cpu().numpy())
        query = torch.as_tensor(query_np, device=target.device)
        matched = ids[torch.as_tensor(target_np, device=target.device)]
        object_target[query] = 1.
        center_loss = F.l1_loss(output['centers'][query], mask_centroid[matched])
        known = eligible[matched]
        logp = output['assignment_logits'][query].log_softmax(-1)
        # Avoid zero times -inf from unavailable candidate slots.
        logp = torch.where(distributions[matched] > 0, logp, torch.zeros_like(logp))
        ce = -(distributions[matched] * logp).sum(-1)
        assignment_loss = ce[known].mean() if known.any() else zero
    else:
        query = ids
        matched = ids
        center_loss = assignment_loss = zero
    if annotation_complete:
        object_loss = F.binary_cross_entropy_with_logits(output['object_logits'], object_target)
    else:
        object_loss = F.binary_cross_entropy_with_logits(output['object_logits'][query], object_target[query]) if len(query) else zero
    total = assignment_loss + object_loss + 10. * center_loss
    return dict(total=total, assignment=assignment_loss, objectness=object_loss,
                center=center_loss, matched_queries=query, matched_targets=matched)


@torch.no_grad()
def decode_grouped(output, object_threshold=.5):
    """GT-free joint candidate selection: each candidate used by at most one mouse.

    Null columns are duplicated so every predicted instance can reject a point.
    Fixed object threshold is preregistered, not adjusted on development labels.
    """
    active = (output['object_logits'].sigmoid() >= object_threshold).nonzero().flatten()
    q = len(output['object_logits'])
    xy = output['centers'].new_zeros(q, 5, 2)
    accepted = torch.zeros(q, 5, dtype=torch.bool, device=xy.device)
    chosen = torch.full((q, 5), -1, dtype=torch.long, device=xy.device)
    if len(active):
        logp = output['assignment_logits'][active].log_softmax(-1)
        n = output['candidate_xy'].shape[1]
        for k in range(5):
            values = torch.cat([logp[:, k, :n], logp[:, k, n:].expand(-1, len(active))], -1)
            rows, cols = linear_sum_assignment(-values.cpu().numpy())
            for row, col in zip(rows, cols):
                if col < n and bool(output['candidate_available'][k, col]):
                    qi = active[row]
                    xy[qi, k] = output['candidate_xy'][k, col]
                    accepted[qi, k] = True
                    chosen[qi, k] = int(col)
    return dict(xy=xy, accepted=accepted, chosen=chosen,
                instance_active=output['object_logits'].sigmoid() >= object_threshold,
                object_probability=output['object_logits'].sigmoid(), centers=output['centers'])
