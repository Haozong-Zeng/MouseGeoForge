"""Deterministic grid gradients for frozen-feature spatial-anchor experiments.

Not a general grid_sample replacement: feature gradients deliberately unsupported.
Equivalent bilinear/zero-padding/align_corners=False math, evaluated with gather
from detached pixels and differentiable interpolation weights. No scatter-add
into pixel gradients, because the experiment freezes its encoder/local pixels.
"""
import torch
from torch.nn import functional as F


def frozen_bilinear_sample(pixels,grid):
    if pixels.requires_grad:raise ValueError('Frozen features required; do not silently cut encoder gradients')
    if pixels.ndim!=4 or grid.ndim!=4 or grid.shape[0]!=pixels.shape[0] or grid.shape[-1]!=2:
        raise ValueError('Expected NCHW pixels and NHW2 grid')
    if pixels.dtype!=grid.dtype or pixels.device!=grid.device:raise ValueError('Matching dtype/device required')
    n,c,h,w=pixels.shape;gh,gw=grid.shape[1:3]
    # CUDA grid_sampler contracts multiply/subtract to FMA. Evaluate that pair
    # in float64 then round once for float32 to match its source coordinates.
    # The initial +1 still rounds in the original dtype, as in grid_sampler.
    def unnormalize(coord,size):
        shifted=coord+1
        if coord.dtype==torch.float32:return ((shifted.double()*size-1).float())/2
        return (shifted*size-1)/2
    x=unnormalize(grid[...,0],w);y=unnormalize(grid[...,1],h)
    x0=x.floor();y0=y.floor();x1=x0+1;y1=y0+1
    def gather(ix,iy):
        inside=(ix>=0)&(ix<w)&(iy>=0)&(iy<h)
        index=(iy.clamp(0,h-1).long()*w+ix.clamp(0,w-1).long()).reshape(n,1,-1).expand(-1,c,-1)
        return pixels.flatten(2).gather(2,index).reshape(n,c,gh,gw)*inside[:,None]
    return (gather(x0,y0)*((x1-x)*(y1-y))[:,None]+gather(x1,y0)*((x-x0)*(y1-y))[:,None]+
            gather(x0,y1)*((x1-x)*(y-y0))[:,None]+gather(x1,y1)*((x-x0)*(y-y0))[:,None])


def deterministic_local_refine(sample,coarse):
    n,q,k=coarse.shape[:3];size=sample['size'];radius=sample['radius']
    axis=torch.linspace(-radius,radius,size,device=coarse.device,dtype=coarse.dtype)
    y,x=torch.meshgrid(axis,axis,indexing='ij');offset=torch.stack((x,y),-1)
    grid=(coarse[:,:,:,None,None]*2-1+offset[None,None,None]*2).reshape(n,q*k*size,size,2)
    pixel=frozen_bilinear_sample(sample['local_pixel'],grid)
    pixel=pixel.view(n,-1,q,k,size,size).permute(0,2,3,1,4,5)
    logits=torch.einsum('nqkd,nqkdhw->nqkhw',sample['local_query'],F.normalize(pixel,dim=3,eps=1e-6))
    logits=logits*sample['scale'][...,None,None]
    probs=logits.flatten(-2).softmax(-1).view_as(logits)
    return (coarse+(probs[...,None]*offset).sum((-3,-2))).clamp(0,1)
