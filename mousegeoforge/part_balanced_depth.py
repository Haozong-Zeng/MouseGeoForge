"""Bounded part-balanced reduction of an existing dense depth error map.

This module changes spatial reduction, not depth targets, scale alignment,
coordinate units, silhouette losses, or the meaning of an anatomical landmark.
All masks/ROI weights are supervision inputs, detached from autograd.
"""
from __future__ import annotations

import math
import torch


def keypoint_regions(points_xy, visible, bbox_xyxy, image_hw, *, radius_fraction=.06):
    """Truncated soft ROIs in integer-centred depth-map pixels, [N,K,H,W].

    Caller must apply the SAME recorded crop/resize transform to points, box,
    depth and validity. `visible` is explicit trusted visibility, not `known`.
    Occluded/out-of-frame/NaN points produce zero support. No depth is inferred
    from a 2D point; in particular joint-Z is never used as surface depth.
    """
    if points_xy.ndim != 3 or points_xy.shape[-1] != 2:
        raise ValueError('points_xy must be [N,K,2]')
    n,k,_ = points_xy.shape
    if visible.shape != (n,k) or visible.dtype != torch.bool or bbox_xyxy.shape != (n,4):
        raise ValueError('Require explicit bool visibility [N,K] and boxes [N,4]')
    h,w = image_hw
    if min(h,w) < 1 or not math.isfinite(radius_fraction) or radius_fraction <= 0:
        raise ValueError('Positive image dimensions and ROI radius required')
    points=points_xy.detach().float(); boxes=bbox_xyxy.detach().float()
    if not torch.isfinite(boxes).all() or torch.any(boxes[:,2:] <= boxes[:,:2]):
        raise ValueError('Boxes must be finite with positive extent')
    good=(visible.detach() & torch.isfinite(points).all(-1) &
          (points[...,0]>=0) & (points[...,0]<w) & (points[...,1]>=0) & (points[...,1]<h))
    points=torch.where(good[...,None],points,torch.zeros_like(points))
    radius=(boxes[:,2:]-boxes[:,:2]).amax(-1)*radius_fraction
    yy,xx=torch.meshgrid(torch.arange(h,device=points.device),torch.arange(w,device=points.device),indexing='ij')
    distance=((xx-points[...,0,None,None])**2+(yy-points[...,1,None,None])**2)/radius[:,None,None,None]**2
    return torch.where(good[...,None,None] & (distance<=1),torch.exp(-2*distance),torch.zeros_like(distance))


def part_balanced_depth(error, valid, regions, *, part_weights=None, confidence=None,
                        strength=.5, max_pixel_boost=4., min_region_pixels=8,
                        min_region_mass=2.):
    """Return spatially reweighted loss plus explicit coverage diagnostics.

    `error`: nonnegative unreduced error of the CURRENT depth recipe [N,H,W].
    Compute any scale alignment once on its original full support, never per ROI.
    `regions`: soft [N,K,H,W] supports or independently defined synthetic parts.

    Each active region is normalised by its own confidence-weighted area. Their
    weighted mixture q sums to one per supported view. Pixel multipliers are
    min(1 + strength * confidence_mass * q / confidence, max_pixel_boost), then
    the loss is normalised per view. Thus no pixel's gradient exceeds the
    original confidence-weighted mean coefficient by max_pixel_boost. Absent
    regions fall back to baseline, never remove a view. `strength=0` is exactly
    the baseline reduction. Source/domain aggregation remains the caller's job.
    """
    if error.ndim != 3 or valid.shape != error.shape or valid.dtype != torch.bool:
        raise ValueError('Require error [N,H,W] and bool validity of identical shape')
    if regions.ndim != 4 or regions.shape[0] != error.shape[0] or regions.shape[2:] != error.shape[1:]:
        raise ValueError('regions must be [N,K,H,W]')
    if regions.shape[1] < 1:
        raise ValueError('At least one declared part slot required')
    if not math.isfinite(strength) or strength < 0 or not math.isfinite(max_pixel_boost) or max_pixel_boost < 1:
        raise ValueError('Invalid strength or pixel boost cap')
    if min_region_pixels < 1 or not math.isfinite(min_region_mass) or min_region_mass <= 0:
        raise ValueError('Positive minimum region support required')
    r=regions.detach().float()
    if not torch.isfinite(r).all() or torch.any((r<0)|(r>1)):
        raise ValueError('Region support must be finite in [0,1]')
    c=torch.ones_like(error,dtype=torch.float32) if confidence is None else confidence.detach().float()
    if c.shape != error.shape or not torch.isfinite(c).all() or torch.any((c<0)|(c>1)):
        raise ValueError('Confidence must be finite [N,H,W] in [0,1]')
    support=valid.detach() & (c>0)
    if not torch.isfinite(error[support]).all() or torch.any(error[support]<0):
        raise ValueError('Valid depth errors must be finite and nonnegative')
    e=torch.where(support,error.float(),torch.zeros_like(error,dtype=torch.float32))
    c=torch.where(support,c,torch.zeros_like(c))
    mass=c.sum((-1,-2)); present=mass>0
    base=(e*c).sum((-1,-2))/mass.clamp_min(1e-12)
    rw=r*c[:,None]; region_mass=rw.sum((-1,-2))
    counts=((r>0)&support[:,None]).sum((-1,-2))
    active=(counts>=min_region_pixels)&(region_mass>=min_region_mass)
    pw=torch.ones(r.shape[1],device=e.device) if part_weights is None else torch.as_tensor(part_weights,device=e.device,dtype=torch.float32).detach()
    if pw.shape != (r.shape[1],) or not torch.isfinite(pw).all() or torch.any(pw<0) or not torch.any(pw>0):
        raise ValueError('Part weights must be finite nonnegative [K], at least one positive')
    budget=active*pw[None]; budget=budget/budget.sum(-1,keepdim=True).clamp_min(1e-12)
    q=(rw/region_mass.clamp_min(1e-12)[...,None,None]*budget[...,None,None]).sum(1)
    region_loss=(rw*e[:,None]).sum((-1,-2))/region_mass.clamp_min(1e-12)
    part_mean=(region_loss*budget).sum(-1)
    boost=(1+strength*mass[:,None,None]*q/c.clamp_min(1e-12)).clamp(max=max_pixel_boost)
    weighted=c*boost
    per_view=(e*weighted).sum((-1,-2))/weighted.sum((-1,-2)).clamp_min(1e-12)
    focus_per_view=(e*c*(boost-1)).sum((-1,-2))/mass.clamp_min(1e-12)
    # Invalid padding can contain NaN, and an entirely empty batch still has a
    # finite zero gradient tied to the original prediction graph.
    zero=e.sum()*0
    loss=per_view[present].mean() if present.any() else zero
    baseline=base[present].mean() if present.any() else zero
    focus_loss=focus_per_view[present].mean() if present.any() else zero
    ratio=boost*mass[:,None,None]/weighted.sum((-1,-2))[:,None,None].clamp_min(1e-12)
    return dict(loss=loss,baseline=baseline,focus_loss=focus_loss,per_view=per_view,part_mean=part_mean.detach(),
                part_losses=region_loss.detach(),active_regions=active.detach(),
                region_pixels=counts.detach(),region_mass=region_mass.detach(),
                valid_views=present.detach(),pixel_boost=boost.detach(),
                gradient_ratio=ratio.detach(),capped_pixels=((boost>=max_pixel_boost)&support).sum().detach())
