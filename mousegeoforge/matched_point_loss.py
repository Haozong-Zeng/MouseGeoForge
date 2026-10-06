"""Coordinate-only pose readout loss with equal weight per sampled image."""
import torch
from torch.nn import functional as F


def matched_image_point_loss(pred,gt,valid,pixels,image_indices,image_count,point_weights=None):
    if pred.shape!=gt.shape or valid.shape!=pred.shape[:-1] or pixels.shape!=(len(pred),2):
        raise ValueError('Invalid matched point dimensions')
    if valid.dtype!=torch.bool or image_indices.shape!=(len(pred),):raise ValueError('Invalid mask or image indices')
    if point_weights is not None and (point_weights.shape!=valid.shape or not torch.isfinite(point_weights).all() or (point_weights<0).any()):
        raise ValueError('point_weights must be finite nonnegative and match valid')
    scale=pixels[:,None]/768
    safe=torch.where(valid[...,None],gt,pred.detach())
    point=F.smooth_l1_loss(pred*scale,safe*scale,beta=.01,reduction='none').mean(-1)
    # Keep ORIGINAL valid-point denominator: dropping a noisy observation must
    # not silently increase the gradient weight of reliable landmarks.
    weighted=point if point_weights is None else point*point_weights.detach()
    per_instance=(weighted*valid).sum(-1)/valid.sum(-1).clamp_min(1)
    good=valid.any(-1).to(pred.dtype)
    sums=pred.new_zeros(image_count).index_add(0,image_indices,per_instance*good)
    counts=pred.new_zeros(image_count).index_add(0,image_indices,good)
    # Empty images contribute zero; their missing labels are not negatives.
    return (sums/counts.clamp_min(1)).mean()
