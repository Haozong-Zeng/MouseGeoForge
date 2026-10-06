"""Explicit finite-point heatmap targets; never infer labels from missingness."""
import torch


def gaussian_point_targets(observed_xy,observed_valid,pixels,shape_hw,*,sigma_px=12.):
    """Discretized Gaussian around known points at current image resolution.

    This target is a chosen supervision kernel, NOT calibrated annotation
    uncertainty. It does not create fullcurve labels or visible-point flags.
    """
    if observed_xy.ndim!=3 or observed_xy.shape[-1]!=2 or observed_valid.shape!=observed_xy.shape[:2]:raise ValueError('Expected [N,K,2] coordinates and [N,K] validity')
    if pixels.shape!=(len(observed_xy),2) or not torch.isfinite(pixels).all() or (pixels<=0).any() or sigma_px<=0:raise ValueError('Invalid image size/sigma')
    h,w=shape_hw
    if h<1 or w<1:raise ValueError('Invalid heatmap shape')
    xy=observed_xy.detach();good=observed_valid.bool()&torch.isfinite(xy).all(-1)
    yy,xx=torch.meshgrid((torch.arange(h,device=xy.device,dtype=xy.dtype)+.5)/h,(torch.arange(w,device=xy.device,dtype=xy.dtype)+.5)/w,indexing='ij')
    grid=torch.stack([xx,yy],-1)
    delta=(grid[None,None]-torch.nan_to_num(xy)[:,:,None,None])*pixels.detach()[:,None,None,None]
    log_target=-delta.square().sum(-1)/(2*sigma_px**2)
    target=log_target.flatten(2).softmax(-1).reshape(*good.shape,h,w)
    return target.detach(),good.detach()


def point_heatmap_kl(probability,observed_xy,observed_valid,pixels,*,sigma_px=12.,log_probability=None):
    if probability.ndim!=4 or probability.shape[:2]!=observed_xy.shape[:2]:raise ValueError('Expected [N,K,H,W] probability')
    target,good=gaussian_point_targets(observed_xy,observed_valid,pixels,probability.shape[-2:],sigma_px=sigma_px)
    if not good.any():return torch.nan_to_num(probability).sum()*0
    p=probability[good]
    if not torch.isfinite(p).all() or (p<0).any():raise ValueError('Invalid supervisedprobability')
    q=target[good]
    if log_probability is not None:
        if log_probability.shape!=probability.shape or not torch.isfinite(log_probability[good]).all():raise ValueError('Invalid log probability')
        log_p=log_probability[good]
    else:log_p=p.clamp_min(1e-12).log()
    return (q*(q.clamp_min(1e-12).log()-log_p)).sum((-2,-1)).mean()
