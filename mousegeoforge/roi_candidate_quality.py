"""Region-feature quality probe; no GT inputs in inference and no box/NMS edits."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def region_features(features, boxes, scores, bins=7):
    if features.ndim!=4 or boxes.ndim!=3 or boxes.shape[-1]!=4 or scores.shape!=boxes.shape[:2]:raise ValueError('Expected VCHW, VQ4 and VQ')
    if features.shape[0]!=boxes.shape[0] or bins<1:raise ValueError('View/bins mismatch')
    if not torch.isfinite(boxes).all() or (boxes[...,2:]<=0).any():raise ValueError('Invalid boxes')
    lo=(boxes[...,:2]-boxes[...,2:]/2).clamp(0,1)
    hi=(boxes[...,:2]+boxes[...,2:]/2).clamp(0,1)
    if (hi<=lo).any():raise ValueError('Empty clipped ROI')
    t=(torch.arange(bins,device=boxes.device,dtype=boxes.dtype)+.5)/bins
    yy,xx=torch.meshgrid(t,t,indexing='ij');unit=torch.stack((xx,yy),-1)
    grid=lo[:,:,None,None]+(hi-lo)[:,:,None,None]*unit
    v,q=boxes.shape[:2]
    # Stack ROIs along output-height: one grid_sample per view, no feature copies.
    sampled=F.grid_sample(F.normalize(features,dim=1,eps=1e-6),2*grid.reshape(v,q*bins,bins,2)-1,
                          align_corners=False,padding_mode='zeros')
    sampled=sampled.reshape(v,features.shape[1],q,bins,bins).permute(0,2,1,3,4)
    return torch.cat((sampled.mean((-1,-2)),sampled.amax((-1,-2)),boxes,scores[...,None]),-1)


class RegionQualityProbe(nn.Module):
    def __init__(self,channels=256,hidden=64):
        super().__init__()
        self.hidden=nn.Linear(2*channels+5,hidden)
        self.output=nn.Linear(hidden,1)
        with torch.no_grad():
            self.output.weight.mul_(.01);self.output.bias.fill_(math.log(.1/.9))

    def forward(self,descriptors):
        return self.output(F.relu(self.hidden(descriptors))).squeeze(-1)


def apply_quality_scores(probe,descriptors,original_scores,enabled=False):
    if not enabled:return original_scores
    return probe(descriptors).sigmoid()
