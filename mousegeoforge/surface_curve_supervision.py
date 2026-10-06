"""Opt-in dorsal SURFACE supervision; never interprets RGB as internal spine GT.

Coordinates are pixels. Existing Core23 heads/semantics are unchanged. These
primitives do not guess visibility, fabricate missing points or bridge gaps.
"""
from __future__ import annotations
import torch
from torch.nn import functional as F

BACK_IDS = (5, 6)  # display IDs06/07; NOT neck05 or internal bone locations

def _zero(x):
    return torch.nan_to_num(x).sum()*0

def back_tangents(target, valid):
    """Local chord only when both immediate axial neighbors are observed.

    A chord is an uncertain tangent estimate, not a measured surface tangent.
    Caller may replace it with a validated curve derivative.
    """
    if target.shape[-2:] != (23,2) or valid.shape != target.shape[:-1]:
        raise ValueError('Expected [...,23,2] and [...,23]')
    good=valid.bool() & torch.isfinite(target).all(-1)
    safe=torch.where(good[...,None],target,torch.zeros_like(target))
    tangent=torch.zeros_like(target);known=torch.zeros_like(good)
    for k,lo,hi in [(5,4,6),(6,5,7)]:
        d=safe[...,hi,:]-safe[...,lo,:];norm=d.norm(dim=-1)
        ok=good[...,k]&good[...,lo]&good[...,hi]&(norm>1e-6)
        tangent[...,k,:]=d/norm.clamp_min(1e-6)[...,None];known[...,k]=ok
    return tangent.detach(),known

def point_loss(predicted, target, valid, *, weak_back=None, tangent=None,
               tangent_valid=None, along_tolerance=4., uncertain_weight=.25,
               scale=768., beta=.01):
    """Replace, not add to, strict06/07 point loss in weak real observations.

    All strong points retain the baseline coordinate Huber loss. Tangents are
    detached; only weak06/07 can use anisotropy. Unknown tangents are isotropically
    downweighted, not falsely precise across an invented normal. Normalization
    and denominator stay fixed between experiment arms.
    """
    if predicted.shape!=target.shape or target.shape[-2:]!=(23,2):raise ValueError('Core23 shape mismatch')
    if valid.shape!=target.shape[:-1]:raise ValueError('Validity shape mismatch')
    if along_tolerance<1 or not 0<=uncertain_weight<=1 or scale<=0:raise ValueError('Invalid loss scales')
    good=valid.bool() & torch.isfinite(target).all(-1)
    if not torch.isfinite(predicted[good]).all():raise ValueError('Nonfinite supervised prediction')
    if not good.any():return _zero(predicted)
    safe=torch.where(good[...,None],target,torch.zeros_like(target))
    pp=torch.where(good[...,None],predicted,torch.zeros_like(predicted));delta=(pp-safe)/scale
    base=F.smooth_l1_loss(delta,torch.zeros_like(delta),beta=beta,reduction='none').sum(-1)
    if weak_back is None:return base[good].sum()/(2*good.sum())
    if weak_back.shape!=good.shape:raise ValueError('Weak mask shape mismatch')
    permitted=torch.zeros_like(good);permitted[...,list(BACK_IDS)]=True
    if (weak_back.bool()&~permitted).any():raise ValueError('Only06/07 may be weakened')
    if tangent is None:tangent,tangent_valid=back_tangents(target,good)
    if tangent.shape!=target.shape or tangent_valid is None or tangent_valid.shape!=good.shape:raise ValueError('Tangent shape mismatch')
    tangent=torch.nan_to_num(tangent.detach());norm=tangent.norm(dim=-1);tv=tangent_valid.bool() & (norm>1e-6)
    t=tangent/norm.clamp_min(1e-6)[...,None];normal=torch.stack([-t[...,1],t[...,0]],-1)
    residual=torch.stack([(delta*t).sum(-1)/along_tolerance,(delta*normal).sum(-1)],-1)
    weak=F.smooth_l1_loss(residual,torch.zeros_like(residual),beta=beta,reduction='none').sum(-1)
    weak=torch.where(tv,weak,base*uncertain_weight)
    values=torch.where(weak_back.bool(),weak,base)
    return values[good].sum()/(2*good.sum())

def points_to_polyline(points, line, line_valid):
    """Closest segment distance and ordered material parameter; no missing-gap edges.

    For this primitive only, arrays are [N,2], [M,2], [M]. Piecewise minimum is
    differentiable within the selected segment. Ties are not evidence of unique
    correspondence. Parameter is polyline-index based, not 2D arc length.
    """
    if points.ndim!=2 or line.ndim!=2 or points.shape[-1]!=2 or line.shape[-1]!=2:raise ValueError('Expected2D polylines')
    good=line_valid.bool() & torch.isfinite(line).all(-1)
    clean=torch.where(good[:,None],line,torch.zeros_like(line))
    a,b=clean[:-1],clean[1:];d=b-a;den=d.square().sum(-1);seg=good[:-1]&good[1:]
    if not seg.any():return None
    t=((points[:,None,:]-a)*d).sum(-1)/den.clamp_min(1e-12)
    t=t.clamp(0,1);closest=a+t[...,None]*d
    dist=(points[:,None,:]-closest).square().sum(-1)
    dist=dist.masked_fill(~seg[None,:],float('inf'))
    val,idx=dist.min(-1);frac=t.gather(1,idx[:,None]).squeeze(1)
    param=(idx.to(points.dtype)+frac)/max(1,len(line)-1)
    return val,param

def curve_loss(predicted, target, target_valid, *, full_curve, scale=768.,
               endpoint_weight=1., order_weight=.05):
    """Shape/order loss for ONE matched instance, using actual curve observations.

    Complete synthetic curves: symmetric distances, order and end anchors.
    Partial real fragments: target-to-predicted only, fragment order only, and
    no extrapolated endpoints or reverse full-curve attraction. At least three
    finite target samples are required; sparse2-point data stay point-only.
    Visibility/source reliability must be decided by the caller, not this loss.
    """
    if predicted.ndim!=2 or target.ndim!=2 or predicted.shape[-1]!=2 or target.shape[-1]!=2:raise ValueError('Expected[N,2] curves')
    good=target_valid.bool() & torch.isfinite(target).all(-1)
    if full_curve and not good.all():raise ValueError('Full curve must be fully observed/known')
    if good.sum()<3:return _zero(predicted)
    if len(predicted)<3 or not torch.isfinite(predicted).all():raise ValueError('Invalid predicted curve')
    pred=predicted/scale;gt=torch.where(good[:,None],target,torch.zeros_like(target))/scale
    reverse=points_to_polyline(gt[good],pred,torch.ones(len(pred),dtype=torch.bool,device=pred.device))
    distances,params=reverse
    # Only adjacent observed target indices contribute ordering; do not bridge gaps.
    ids=torch.where(good)[0];adjacent=(ids[1:]-ids[:-1])==1
    # Normalize by nominal sample spacing so the same reversal is not silently
    # weakened quadratically when the curve is sampled more densely.
    order=F.relu((params[:-1]-params[1:])*(len(target)-1))[adjacent].square().mean() if adjacent.any() else _zero(pred)
    loss=distances.mean()+order_weight*order
    if full_curve:
        forward=points_to_polyline(pred,gt,good);dist,p=forward
        loss=.5*(loss+dist.mean())+order_weight*F.relu((p[:-1]-p[1:])*(len(pred)-1)).square().mean()
        loss=loss+endpoint_weight*((pred[0]-gt[0]).square().mean()+(pred[-1]-gt[-1]).square().mean())
    return loss

def material_curve_loss(predicted, target, target_valid, material_u, *,
                        material_parameter_known=False, scale=768., beta=.01):
    """Synthetic-only correspondence at KNOWN rest-mesh material parameters.

    Not2D/3D arc length and NOT fractions assigned to manual Core23 points.
    Complete known geometry only; partial/real lines must use another objective.
    This smooth objective preserves ordering without nearest-segment switches.
    """
    if not material_parameter_known:raise ValueError('Explicit known material provenance required')
    if predicted.ndim!=2 or target.ndim!=2 or predicted.shape[-1]!=2 or target.shape[-1]!=2:raise ValueError('Expected2D curve arrays')
    if len(predicted)<3 or len(target)<3 or target_valid.shape!=(len(target),) or not target_valid.bool().all():raise ValueError('Complete known synthetic curve required')
    if not torch.isfinite(predicted).all() or not torch.isfinite(target).all():raise ValueError('Nonfinite geometry')
    if material_u.shape!=(len(target),) or not torch.isfinite(material_u).all() or not (material_u[1:]>material_u[:-1]).all():raise ValueError('Strict ordered material parameters required')
    if abs(float(material_u[0]))>1e-6 or abs(float(material_u[-1])-1)>1e-6 or scale<=0:raise ValueError('Expected normalized complete0..1 material parameters')
    u=torch.linspace(0,1,len(predicted),device=target.device,dtype=target.dtype)
    hi=torch.searchsorted(material_u.contiguous(),u.contiguous()).clamp(1,len(target)-1);lo=hi-1
    f=(u-material_u[lo])/(material_u[hi]-material_u[lo]);corresponding=target[lo]*(1-f[:,None])+target[hi]*f[:,None]
    return F.smooth_l1_loss(predicted/scale,corresponding/scale,beta=beta)

def bezier(controls, samples=33):
    """Low-DOF cubic curve with four free image-plane control points per instance."""
    if controls.shape[-2:]!=(4,2) or samples<3:raise ValueError('Expected[...,4,2] controls')
    t=torch.linspace(0,1,samples,device=controls.device,dtype=controls.dtype)
    basis=torch.stack([(1-t)**3,3*(1-t)**2*t,3*(1-t)*t*t,t**3],-1)
    return torch.einsum('sk,...kd->...sd',basis,controls)

def cubic_bspline(controls, samples=33):
    """Clamped uniform cubic B-spline; control gradients only, no SciPy runtime.

    Six controls are the initial candidate, but any count>=4 is supported.
    Control points are latent curve parameters, never annotation landmarks.
    """
    count=controls.shape[-2]
    if controls.shape[-1]!=2 or count<4 or samples<3:raise ValueError('Expected[...,C>=4,2]')
    device,dtype=controls.device,controls.dtype
    internal=torch.linspace(0,1,count-2,device=device,dtype=dtype)[1:-1]
    knots=torch.cat([torch.zeros(4,device=device,dtype=dtype),internal,torch.ones(4,device=device,dtype=dtype)])
    t=torch.linspace(0,1,samples,device=device,dtype=dtype)
    basis=((t[:,None]>=knots[:-1])&(t[:,None]<knots[1:])).to(dtype)
    for degree in range(1,4):
        width=len(knots)-degree-1
        dl=knots[degree:degree+width]-knots[:width]
        dr=knots[degree+1:degree+1+width]-knots[1:width+1]
        left=(t[:,None]-knots[:width])/dl.clamp_min(1e-12)
        right=(knots[degree+1:degree+1+width]-t[:,None])/dr.clamp_min(1e-12)
        basis=left*basis[:,:width]*(dl>0)+right*basis[:,1:width+1]*(dr>0)
    basis[-1]=0;basis[-1,-1]=1
    return torch.einsum('sk,...kd->...sd',basis,controls)
