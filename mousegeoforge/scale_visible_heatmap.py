"""Training-target-only scale-aware Gaussian widths; inference is unchanged."""
import torch
from .visible_heatmap_supervision import anchor_scores

BODY=list(range(7))+list(range(14,23))


def body_sigma(xy,valid,*,image_hw,factor=.035,minimum=2.,maximum=8.):
    if xy.shape[-2:]!=(23,2) or valid.shape!=xy.shape[:-1]:raise ValueError('Expected Core23 xy and validity')
    if not (factor>0 and 0<minimum<=maximum):raise ValueError('Invalid width policy')
    with torch.no_grad():
        body=xy.detach()[...,BODY,:];mask=valid[...,BODY]&torch.isfinite(body).all(-1)
        lo=body.masked_fill(~mask[...,None],float('inf')).amin(-2)
        hi=body.masked_fill(~mask[...,None],-float('inf')).amax(-2)
        ih,iw=image_hw;diag=((hi-lo)*body.new_tensor([iw,ih])).norm(dim=-1)
        fallback=mask.sum(-1)<2
        sigma=torch.where(fallback,torch.full_like(diag,maximum),(factor*diag).clamp(minimum,maximum))
    return sigma,fallback


def variable_visible_kl(scores,xy,valid,sigma,*,image_hw):
    if scores.shape[:-2]!=valid.shape or xy.shape!=(*valid.shape,2) or sigma.shape!=valid.shape:raise ValueError('Target shapes mismatch')
    if not valid.any():return scores.sum()*0
    selected=scores[valid];target=xy.detach()[valid];width=sigma.detach()[valid,None,None]
    if not torch.isfinite(width).all() or (width<=0).any():raise ValueError('Positive finite widths required')
    if not torch.isfinite(target).all() or ((target<0)|(target>1)).any():raise ValueError('Finite in-frame target required')
    h,w=scores.shape[-2:];ih,iw=image_hw
    ys=(torch.arange(h,device=scores.device,dtype=scores.dtype)+.5)/h;xs=(torch.arange(w,device=scores.device,dtype=scores.dtype)+.5)/w
    squared=((xs[None,None,:]-target[:,0,None,None])*iw/width).square()+((ys[None,:,None]-target[:,1,None,None])*ih/width).square()
    target_log=(-.5*squared).flatten(1).log_softmax(-1);logp=selected.flatten(1).log_softmax(-1)
    return (target_log.exp()*(target_log-logp)).sum(-1).mean()


def matched_scale_visible_loss(branch,sample):
    scores=anchor_scores(branch,sample['tokens'],sample['pixels'],sample['mask'])
    xy=torch.zeros_like(sample['raw']);valid=torch.zeros(xy.shape[:-1],dtype=torch.bool,device=xy.device)
    sigma=torch.full_like(valid,8.,dtype=xy.dtype);target=sample['target'];batch=sample['batch'];fallbacks=0
    for v,(queries,instances) in enumerate(sample['matches']):
        for q,m in zip(queries,instances):
            point=target['keypoint_xy'][0,v,m].to(xy.device);point_valid=target['keypoint_valid'][0,v,m].to(xy.device)
            xy[v,q]=point
            valid[v,q]=(target['keypoint_valid'][0,v,m]&batch['instance_keypoint_visible'][0,v,m]&batch['instance_keypoint_visibility_valid'][0,v,m]).to(xy.device)
            width,fallback=body_sigma(point,point_valid,image_hw=batch['images'].shape[-2:]);sigma[v,q]=width;fallbacks+=int(fallback)
    widths=sigma[valid]
    stats=dict(sigma_mean=float(widths.mean()) if widths.numel() else None,sigma_min=float(widths.min()) if widths.numel() else None,
               sigma_max=float(widths.max()) if widths.numel() else None,fallback_instances=fallbacks)
    return variable_visible_kl(scores,xy,valid,sigma,image_hw=batch['images'].shape[-2:]),int(valid.sum()),stats
