"""Opt-in predicted-ROI residual detector probe, not a DAB-DETR reproduction."""
import torch
from torch import nn
from torch.nn import functional as F


def sample_predicted_roi(memory,boxes):
    """5x5 grid spanning twice predicted box extent; no labels or GT crop."""
    if memory.ndim!=4 or memory.shape[1]!=256 or boxes.ndim!=3 or boxes.shape[0]!=memory.shape[0] or boxes.shape[-1]!=4:
        raise ValueError('Expected memory[N,256,H,W], boxes[N,Q,4]')
    memory=memory.detach().float();boxes=boxes.detach().float()
    axis=torch.linspace(-1,1,5,device=memory.device);y,x=torch.meshgrid(axis,axis,indexing='ij')
    offsets=torch.stack([x,y],-1).reshape(25,2)
    grid=boxes[:,:,None,:2]+offsets[None,None]*boxes[:,:,None,2:]
    # One grid_sample per image; border mode is explicit for out-of-image ROI.
    samples=F.grid_sample(memory,grid*2-1,mode='bilinear',padding_mode='border',align_corners=False)
    return samples.permute(0,2,3,1).contiguous(),memory.mean((-2,-1))


class SpatialDetectionReadout(nn.Module):
    """Matched ROI/global branches, identical parameters and zero residual init."""
    def __init__(self,mode='roi'):
        super().__init__()
        if mode not in ['roi','global']:raise ValueError('mode must be roi or global')
        self.mode=mode
        self.token=nn.LayerNorm(256)
        self.context=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,32),nn.GELU())
        self.residual=nn.Sequential(nn.Linear(256+25*32+4,128),nn.GELU(),nn.Linear(128,5))
        nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)

    def forward(self,tokens,roi,global_memory,original_boxes,original_logits):
        n,q=tokens.shape[:2]
        if tokens.shape!=(n,q,256) or roi.shape!=(n,q,25,256) or global_memory.shape!=(n,256) or original_boxes.shape!=(n,q,4) or original_logits.shape!=(n,q):
            raise ValueError('Invalid spatial detection input shapes')
        context=roi.detach() if self.mode=='roi' else global_memory.detach()[:,None,None].expand(n,q,25,256)
        features=torch.cat([self.token(tokens.detach()),self.context(context.float()).flatten(-2),original_boxes.detach()],-1)
        delta=self.residual(features)
        normalized=torch.cat([original_boxes.detach()[...,:2],(original_boxes.detach()[...,2:]-.02)/.98],-1).clamp(1e-6,1-1e-6)
        raw=torch.logit(normalized)+delta[...,:4]
        boxes=torch.cat([raw[...,:2].sigmoid(),.02+.98*raw[...,2:].sigmoid()],-1)
        return dict(instance_boxes_cxcywh=boxes,instance_objectness_logits=original_logits.detach()+delta[...,4])
