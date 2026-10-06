"""Experimental shared high-resolution map + instance-conditioned residual.

Never crops by GT or treats a predicted box as a hard boundary. Only known
five-point labels are consumed by the separate loss, not by forward.
"""
import torch
from torch import nn
from torch.nn import functional as F
KP=(0,14,15,16,17)

class HighResolutionInstanceReadout(nn.Module):
    def __init__(self,appearance=True):
        super().__init__();self.appearance=appearance
        self.key=nn.Sequential(nn.Conv2d(34,32,3,padding=1),nn.GELU(),nn.Conv2d(32,32,1))
        self.norm=nn.LayerNorm(256,elementwise_affine=False)
        self.query=nn.Sequential(nn.Linear(262,64),nn.GELU(),nn.Linear(64,32))
        nn.init.zeros_(self.query[-1].weight);nn.init.zeros_(self.query[-1].bias)

    def forward(self,features,tokens,boxes,baseline_xy,image_hw,*,enabled=False):
        if not enabled:return dict(xy=baseline_xy)
        if features.ndim!=3 or features.shape[0]!=32 or tokens.shape!=(len(boxes),5,256) or baseline_xy.shape!=(len(boxes),23,2):raise ValueError('Feature/query shapes')
        h,w=image_hw;hh,ww=features.shape[-2:];device=features.device
        yy,xx=torch.meshgrid((torch.arange(hh,device=device)+.5)*4,(torch.arange(ww,device=device)+.5)*4,indexing='ij');grid=torch.stack([xx/w,yy/h],-1);valid=(xx<w)&(yy<h)
        if not valid.any():raise ValueError('No valid spatial cells')
        # Frozen feature scale normalized per pixel, not by dataset statistics.
        visual=F.layer_norm(features.permute(1,2,0),(32,)).permute(2,0,1) if self.appearance else torch.zeros_like(features)
        key=self.key(torch.cat([visual,grid.permute(2,0,1)],0)[None])[0]
        conditioning=torch.cat([self.norm(tokens),boxes[:,None].expand(-1,5,-1),baseline_xy[:,KP]],-1)
        q=self.query(conditioning);learned=torch.einsum('qkc,chw->qkhw',q,key)/32**.5
        d=(grid[None,None]-baseline_xy[:,KP,None,None])*features.new_tensor([w,h]);prior=-d.square().sum(-1)/(2*64.**2)
        prior=prior.masked_fill(~valid,-torch.inf);logits=prior+learned
        prob=logits.flatten(-2).softmax(-1);priorprob=prior.flatten(-2).softmax(-1);coords=grid.reshape(-1,2)
        delta=(prob-priorprob)@coords;out=baseline_xy.clone();out[:,KP]=baseline_xy[:,KP]+delta
        return dict(xy=out,logits=logits,grid=grid,spatial_valid=valid,delta=delta)

def highres_localization_loss(output,target,known,image_hw):
    valid=known&torch.isfinite(target).all(-1)
    if not valid.any():return output['xy'].sum()*0
    h,w=image_hw;pixels=target.new_tensor([w,h]);pred=output['xy'][:,KP][valid];gt=target[valid]
    coord=F.smooth_l1_loss(pred*pixels/768,gt*pixels/768,beta=.01)
    grid=output['grid'];d=(grid[None]-gt[:,None,None])*pixels
    targetlogits=-d.square().sum(-1)/(2*8.**2);targetlogits=targetlogits.masked_fill(~output['spatial_valid'],-torch.inf)
    p=targetlogits.flatten(1).softmax(-1);logp=output['logits'][valid].flatten(1).log_softmax(-1)
    # Avoid 0 * -inf at padded cells, including the backward path.
    logp=logp.masked_fill(~output['spatial_valid'].flatten()[None],0.)
    return coord+.01*(-p*logp).sum(-1).mean()
