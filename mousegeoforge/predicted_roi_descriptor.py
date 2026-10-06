"""Mean-pool image feature maps at predicted boxes, without query embeddings."""
import torch
from torch.nn import functional as F


def pool_predicted_roi(feature_map,boxes,grid_size=5):
    if feature_map.ndim!=4 or boxes.ndim!=3 or boxes.shape[0]!=feature_map.shape[0] or boxes.shape[-1]!=4 or grid_size<1:raise ValueError('Expected image features and predicted boxes')
    if not torch.isfinite(feature_map).all() or not torch.isfinite(boxes).all() or (boxes[...,2:]<=0).any():raise ValueError('Nonfinite features or invalid box extents')
    axis=(torch.arange(grid_size,device=feature_map.device,dtype=torch.float32)+.5)/grid_size-.5
    y,x=torch.meshgrid(axis,axis,indexing='ij');offsets=torch.stack([x,y],-1).reshape(-1,2)
    grid=boxes.float()[:,:,None,:2]+offsets[None,None]*boxes.float()[:,:,None,2:]
    valid=((grid>=0)&(grid<=1)).all(-1)
    samples=F.grid_sample(feature_map.float(),grid*2-1,mode='bilinear',padding_mode='zeros',align_corners=False).permute(0,2,3,1)
    count=valid.sum(-1);descriptor=(samples*valid[...,None]).sum(-2)/count.clamp_min(1)[...,None]
    return dict(descriptor=descriptor,valid=count>0,sampled_fraction=count.float()/(grid_size**2))
