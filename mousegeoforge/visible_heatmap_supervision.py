"""Auxiliary visible-only spatial score supervision for an isolated pilot.

Does not change inference or the loss for occluded coordinates. Scores use the
same finite predicted mask prior as SpatialAnchorResidual, never a GT mask prior.
"""
import math
import torch
from torch.nn import functional as F


def anchor_scores(branch,tokens,pixels,mask):
    query=F.normalize(branch.query(tokens.detach()),dim=-1,eps=1e-6)
    pixel=F.normalize(branch.pixel(pixels.detach()),dim=1,eps=1e-6)
    return (torch.einsum('nqkd,ndhw->nqkhw',query,pixel)*branch.log_scale.clamp(0.,math.log(32.)).exp()+
            F.logsigmoid(mask.detach()).clamp_min(-8.)[:,:,None])


def visible_gaussian_kl(scores,xy,valid,*,image_hw,sigma_px=8.):
    if scores.shape[:-2]!=valid.shape or xy.shape!=(*valid.shape,2):raise ValueError('Score/point/valid shape mismatch')
    if sigma_px<=0:raise ValueError('Positive sigma required')
    if not valid.any():return scores.sum()*0
    selected=scores[valid];target=xy.detach()[valid]
    if not torch.isfinite(target).all() or ((target<0)|(target>1)).any():raise ValueError('Valid targets must be finite in-frame normalized points')
    h,w=scores.shape[-2:];ih,iw=image_hw
    ys=(torch.arange(h,device=scores.device,dtype=scores.dtype)+.5)/h
    xs=(torch.arange(w,device=scores.device,dtype=scores.dtype)+.5)/w
    squared=((xs[None,None,:]-target[:,0,None,None])*iw/sigma_px).square()+((ys[None,:,None]-target[:,1,None,None])*ih/sigma_px).square()
    target_log=(-.5*squared).flatten(1).log_softmax(-1)
    logp=selected.flatten(1).log_softmax(-1)
    return (target_log.exp()*(target_log-logp)).sum(-1).mean()


def matched_visible_loss(branch,sample,*,sigma_px=8.):
    scores=anchor_scores(branch,sample['tokens'],sample['pixels'],sample['mask'])
    xy=torch.zeros_like(sample['raw']);valid=torch.zeros(xy.shape[:-1],dtype=torch.bool,device=xy.device)
    target=sample['target'];batch=sample['batch']
    for v,(queries,instances) in enumerate(sample['matches']):
        for q,m in zip(queries,instances):
            xy[v,q]=target['keypoint_xy'][0,v,m].to(xy.device)
            valid[v,q]=(target['keypoint_valid'][0,v,m]&batch['instance_keypoint_visible'][0,v,m]&
                        batch['instance_keypoint_visibility_valid'][0,v,m]).to(xy.device)
    return visible_gaussian_kl(scores,xy,valid,image_hw=batch['images'].shape[-2:],sigma_px=sigma_px),int(valid.sum())
