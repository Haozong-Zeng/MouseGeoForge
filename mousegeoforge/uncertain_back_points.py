"""Experimental06/07point supervision with explicit uncertain direction sources."""
import torch
from torch.nn import functional as F


def back_axes(observed_xy, observed_valid, *, primary_tangent=None,
              primary_valid=None, use_pair_fallback=False, min_chord_px=20.,
              fallback_reliability=.5):
    """Return detached direction and reliability for06/07 only.

    A06->07chord is a proxy, NOT a measured surface tangent. In the fallback
    policy it has at most0.5reliability; short/missing pairs remain directionless.
    Pixels must already be at training resolution. Caller-provided primary
    neighbor directions also remain proxies, not full-curve observations.
    """
    if observed_xy.shape!=(2,2) or observed_valid.shape!=(2,):raise ValueError('Expected06/07slots')
    if min_chord_px<=0 or not 0<=fallback_reliability<=1:raise ValueError('Invalid proxy settings')
    good=observed_valid.bool()&torch.isfinite(observed_xy).all(-1)
    tangent=torch.zeros_like(observed_xy);reliability=torch.zeros(2,device=observed_xy.device,dtype=observed_xy.dtype)
    if primary_tangent is not None or primary_valid is not None:
        if primary_tangent is None or primary_valid is None or primary_tangent.shape!=(2,2) or primary_valid.shape!=(2,):raise ValueError('Incompleteprimarydirection')
        norm=torch.nan_to_num(primary_tangent).norm(dim=-1)
        valid=good&primary_valid.bool()&torch.isfinite(primary_tangent).all(-1)&(norm>1e-6)
        tangent=torch.where(valid[:,None],torch.nan_to_num(primary_tangent)/norm.clamp_min(1e-6)[:,None],tangent)
        reliability=torch.where(valid,torch.ones_like(reliability),reliability)
    if use_pair_fallback and good.all():
        chord=observed_xy[1]-observed_xy[0];length=chord.norm()
        if length>=min_chord_px:
            fill=reliability==0
            tangent=torch.where(fill[:,None],(chord/length)[None],tangent)
            reliability=torch.where(fill,torch.full_like(reliability,fallback_reliability),reliability)
    return tangent.detach(),reliability.detach()


def uncertain_back_point_loss(predicted_xy,observed_xy,observed_valid,*,
                              tangent,reliability,scale=768.,beta=.01,
                              along_tolerance=4.,unknown_weight=.25):
    """Finite point targets retained; blend anisotropic and weak isotropic loss.

    Unlike free point-to-curve fitting, an arbitrarily long predicted curve
    cannot lower this objective. This does not establish a full dorsal curve.
    Confidence and tangent must originate outside the trainable predictions.
    """
    if predicted_xy.shape!=(2,2) or observed_xy.shape!=(2,2) or observed_valid.shape!=(2,) or tangent.shape!=(2,2) or reliability.shape!=(2,):raise ValueError('Expected06/07shapes')
    if scale<=0 or beta<=0 or along_tolerance<1 or not 0<=unknown_weight<=1:raise ValueError('Invalid settings')
    if not torch.isfinite(reliability).all() or ((reliability<0)|(reliability>1)).any():raise ValueError('Invalid reliability')
    good=observed_valid.bool()&torch.isfinite(observed_xy).all(-1)
    if not good.any():return torch.nan_to_num(predicted_xy).sum()*0
    if not torch.isfinite(predicted_xy[good]).all():raise ValueError('Nonfinite supervisedprediction')
    delta=(predicted_xy[good]-observed_xy[good])/scale
    isotropic=F.smooth_l1_loss(delta,torch.zeros_like(delta),beta=beta,reduction='none').sum(-1)
    t=torch.nan_to_num(tangent.detach()[good]);length=t.norm(dim=-1);t=t/length.clamp_min(1e-6)[:,None]
    confidence=reliability.detach()[good]*(length>1e-6)
    normal=torch.stack([-t[:,1],t[:,0]],-1)
    residual=torch.stack([(delta*t).sum(-1)/along_tolerance,(delta*normal).sum(-1)],-1)
    anisotropic=F.smooth_l1_loss(residual,torch.zeros_like(residual),beta=beta,reduction='none').sum(-1)
    return (confidence*anisotropic+(1-confidence)*unknown_weight*isotropic).mean()/2
