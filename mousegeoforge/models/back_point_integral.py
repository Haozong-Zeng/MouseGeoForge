"""Experimental instance-conditioned spatial expectation, not production head."""
import math
import torch
from torch import nn


class BackPointIntegralProbe(nn.Module):
    """Two points from full-map scores; global control retains explicit XY only.

    Inspired by spatial expectation in DSNT / Integral Pose Regression. This
    independent small probe does not reproduce either complete published model.
    Predicted coarse points give a broad prior, never a ground-truth crop.
    """
    def __init__(self, mode='spatial', channels=64, prior_sigma=.25, num_points=2):
        super().__init__()
        if mode not in ('global','spatial') or prior_sigma<=0 or num_points<1:raise ValueError('Invalid settings')
        self.mode=mode;self.prior_sigma=prior_sigma;self.channels=channels;self.num_points=num_points
        self.query=nn.Sequential(nn.LayerNorm(1024),nn.Linear(1024,128),nn.GELU(),nn.Linear(128,num_points*channels))
        self.key=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,channels))
        self.position=nn.Sequential(nn.Linear(2,channels),nn.GELU(),nn.Linear(channels,channels))
        nn.init.zeros_(self.query[-1].weight);nn.init.zeros_(self.query[-1].bias)

    def forward(self,pose,coarse,memory,return_distribution=False):
        if pose.ndim!=2 or pose.shape[-1]!=1024 or coarse.shape!=(len(pose),self.num_points,2):raise ValueError('Expected per-instance pose and predicted point slots')
        if memory.ndim!=4 or memory.shape[:2]!=(len(pose),256):raise ValueError('Expected per-instance image memory')
        n,_,h,w=memory.shape
        yy,xx=torch.meshgrid((torch.arange(h,device=memory.device,dtype=memory.dtype)+.5)/h,(torch.arange(w,device=memory.device,dtype=memory.dtype)+.5)/w,indexing='ij')
        xy=torch.stack([xx,yy],-1).reshape(-1,2)
        visual=memory.detach().flatten(2).transpose(1,2)
        if self.mode=='global':visual=visual.mean(1,keepdim=True).expand(-1,h*w,-1)
        key=self.key(visual)+self.position(xy)[None]
        query=self.query(pose.detach()).reshape(n,self.num_points,self.channels)
        logits=torch.einsum('nkc,npc->nkp',query,key)/math.sqrt(self.channels)
        logits=logits-((xy[None,None]-coarse.detach()[:,:,None])**2).sum(-1)/(2*self.prior_sigma**2)
        prob=logits.softmax(-1);prediction=torch.einsum('nkp,pc->nkc',prob,xy)
        return (prediction,prob.reshape(n,self.num_points,h,w)) if return_distribution else prediction
