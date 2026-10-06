"""Frozen semantic-token readout experiment; not enabled in the production head."""
import math

import torch
from torch import nn


class Core23SpatialReadout(nn.Module):
    """Each point queries memory using its own decoder token, not four back tokens.

    Global control replaces pixel memory with its spatial mean but retains the
    same learned position features, parameters, coarse prior and initialization.
    Inputs contain predictions/features only. Matching and labels stay outside.
    """

    def __init__(self, mode="spatial", channels=64, prior_sigma=0.25):
        super().__init__()
        if mode not in ("spatial", "global") or prior_sigma <= 0:
            raise ValueError("invalid readout mode or prior sigma")
        self.mode, self.prior_sigma = mode, float(prior_sigma)
        self.query = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 128),
                                   nn.GELU(), nn.Linear(128, channels))
        self.key = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, channels))
        self.position = nn.Sequential(nn.Linear(2, channels), nn.GELU(),
                                      nn.Linear(channels, channels))
        nn.init.zeros_(self.query[-1].weight)
        nn.init.zeros_(self.query[-1].bias)

    def forward(self, pose, coarse, memory, return_probability=False, predicted_boxes=None, return_log_probability=False, detach_pose_tokens=True, detach_memory=True):
        if pose.ndim != 3 or pose.shape[-1] != 256:
            raise ValueError("pose must be [N,K,256] semantic decoder tokens")
        if coarse.shape != (*pose.shape[:2], 2):
            raise ValueError("coarse must have one predicted XY per token")
        if memory.ndim != 4 or memory.shape[0] != pose.shape[0] or memory.shape[1] != 256:
            raise ValueError("memory must be [N,256,H,W]")
        h, w = memory.shape[-2:]
        y, x = torch.meshgrid(
            (torch.arange(h, device=pose.device, dtype=pose.dtype) + .5) / h,
            (torch.arange(w, device=pose.device, dtype=pose.dtype) + .5) / w,
            indexing="ij")
        grid = torch.stack((x, y), -1).reshape(-1, 2)
        # Experimental pixel-backbone adaptation may explicitly retain this
        # gradient. Default and all existing callers keep the frozen path.
        feature = (memory.detach() if detach_memory else memory).to(pose.dtype).flatten(2).transpose(1, 2)
        if self.mode == "global":
            feature = feature.mean(1, keepdim=True).expand_as(feature)
        keys = self.key(feature) + self.position(grid)[None]
        # Decoder adaptation is an explicit experimental opt-in. Existing
        # frozen-token callers retain exactly the previous default graph.
        query = self.query(pose.detach() if detach_pose_tokens else pose)
        logits = torch.einsum("nkd,npd->nkp", query, keys) / math.sqrt(query.shape[-1])
        prior = (grid[None, None] - coarse.detach()[:, :, None]).square().sum(-1)
        logits = logits - prior / (2 * self.prior_sigma ** 2)
        if predicted_boxes is not None:
            if predicted_boxes.shape != (pose.shape[0], 4):
                raise ValueError("predicted_boxes must be [N,4] normalized cxcywh")
            boxes = predicted_boxes.detach()
            # Soft suppression beyond a 25%-expanded predicted box, never a
            # hard exclusion: low-quality real boxes must remain recoverable.
            extent = boxes[:, None, 2:].clamp_min(.02)
            outside = ((grid[None]-boxes[:, None, :2]).abs() - .625*extent).clamp_min(0)
            penalty = (-2 * (outside / (.2*extent)).square().sum(-1)).clamp_min(-6)
            logits = logits + penalty[:, None]
        probability = logits.softmax(-1)
        xy = torch.einsum("nkp,pd->nkd", probability, grid)
        if return_log_probability:
            return (xy, probability.reshape(*pose.shape[:2], h, w),
                    logits.log_softmax(-1).reshape(*pose.shape[:2], h, w))
        if return_probability:
            return xy, probability.reshape(*pose.shape[:2], h, w)
        return xy
