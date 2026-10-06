"""Diagnostic point-type evidence, deliberately without instance conditioning.

Not a replacement pose head. No boxes/counts/labels are accepted by forward.
Coordinates and observability are used only in the separate training loss.
"""
import torch
from torch import nn
from torch.nn import functional as F


class LocalPointEvidenceProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv2d(34,64,3,padding=1),nn.GELU(),nn.Conv2d(64,5,1))
        nn.init.zeros_(self.layers[-1].weight);nn.init.zeros_(self.layers[-1].bias)

    def forward(self,features,image_hw):
        if features.ndim!=3 or features.shape[0]!=32:raise ValueError('Expected 32,H,W frozen features')
        ih,iw=image_hw;h,w=features.shape[-2:]
        yy,xx=torch.meshgrid((torch.arange(h,device=features.device)+.5)*4,(torch.arange(w,device=features.device)+.5)*4,indexing='ij')
        grid=torch.stack([xx/iw,yy/ih],-1);valid=(xx<iw)&(yy<ih)
        if not valid.any():raise ValueError('No unpadded spatial cells')
        visual=F.layer_norm(features.permute(1,2,0),(32,)).permute(2,0,1)
        logits=self.layers(torch.cat([visual,grid.permute(2,0,1)],0)[None])[0]
        return dict(logits=logits.masked_fill(~valid,-torch.inf),grid=grid,spatial_valid=valid)


def gaussian_mixture_targets(output,target,eligible,image_hw,sigma_px=8.):
    if target.ndim!=3 or target.shape[1:]!=(5,2) or eligible.shape!=target.shape[:-1] or eligible.dtype!=torch.bool:raise ValueError('Expected instances,5,2 and boolean eligibility')
    if sigma_px<=0:raise ValueError('Positive sigma required')
    if not torch.isfinite(target[eligible]).all():raise ValueError('Known targets must be finite')
    if ((target[eligible]<0)|(target[eligible]>1)).any():raise ValueError('Known targets must be normalized in-frame')
    ih,iw=image_hw;grid=output['grid'];valid=output['spatial_valid'];distributions=torch.zeros_like(output['logits']);known=eligible.any(0)
    for k in range(5):
        xy=target[eligible[:,k],k]
        if not len(xy):continue
        d=(grid[None]-xy[:,None,None])*target.new_tensor([iw,ih]);log=-.5*d.square().sum(-1)/sigma_px**2
        log=log.masked_fill(~valid,-torch.inf)
        distributions[k]=log.flatten(1).softmax(-1).mean(0).reshape_as(valid)
    return distributions,known


def point_evidence_loss(output,target,eligible,image_hw):
    distribution,known=gaussian_mixture_targets(output,target,eligible,image_hw)
    safe=output['logits'].masked_fill(~output['spatial_valid'],0.)
    if not known.any():return safe.sum()*0
    logp=output['logits'].flatten(1).log_softmax(-1).reshape_as(distribution).masked_fill(~output['spatial_valid'],0.)
    return -(distribution[known]*logp[known]).sum((-1,-2)).mean()


@torch.no_grad()
def fixed_top_peaks(output,image_hw,count=6,radius_px=8.):
    if count<=0 or radius_px<0:raise ValueError('Positive count and nonnegative suppression radius')
    ih,iw=image_hw;grid=output['grid'];prob=output['logits'].flatten(1).softmax(-1).reshape_as(output['logits']);work=prob.masked_fill(~output['spatial_valid'],-torch.inf).clone();coords=[];values=[];available=[]
    for _ in range(count):
        score,idx=work.flatten(1).max(1);xy=grid.reshape(-1,2)[idx];ok=torch.isfinite(score)
        coords.append(xy);values.append(torch.where(ok,score,torch.zeros_like(score)));available.append(ok)
        distance=((grid[None]-xy[:,None,None])*grid.new_tensor([iw,ih])).square().sum(-1)
        work=work.masked_fill(distance<=radius_px**2,-torch.inf)
    return dict(xy=torch.stack(coords,1),probability=torch.stack(values,1),available=torch.stack(available,1),probability_map=prob)
