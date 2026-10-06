"""Learned soft relative-box geometry bias, never NMS or hard exclusion."""
import torch
from torch import nn
from torch.nn import functional as F
from mousegeoforge.roi_set_objectness import SetObjectnessProbe


def pair_geometry(boxes,scores):
    if boxes.ndim!=3 or boxes.shape[-1]!=4 or scores.shape!=boxes.shape[:2]:raise ValueError('Expected VQ4 and VQ')
    if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all() or (boxes[...,2:]<=0).any():raise ValueError('Invalid candidate geometry')
    xy=boxes[...,:2];wh=boxes[...,2:].clamp_min(1e-6)
    lo=xy-wh/2;hi=xy+wh/2
    inter=(torch.minimum(hi[:,:,None],hi[:,None])-torch.maximum(lo[:,:,None],lo[:,None])).clamp_min(0).prod(-1)
    area=wh.prod(-1);iou=(inter/(area[:,:,None]+area[:,None]-inter).clamp_min(1e-8)).clamp(0,1)
    offset=((xy[:,None]-xy[:,:,None])/wh[:,:,None]).clamp(-4,4)
    ratio=torch.log(wh[:,None]/wh[:,:,None]).clamp(-4,4)
    difference=scores[:,None]-scores[:,:,None]
    return torch.cat((iou[...,None],offset,ratio,difference[...,None]),-1)


class GeometricObjectnessProbe(SetObjectnessProbe):
    def __init__(self,channels=256,hidden=64,geometry_enabled=False):
        super().__init__(channels,hidden,relation_enabled=True)
        self.geometry_enabled=bool(geometry_enabled)
        self.geometry=nn.Linear(6,4,bias=False)
        nn.init.zeros_(self.geometry.weight)
        self.geometry.weight.requires_grad_(self.geometry_enabled)

    def forward(self,descriptors):
        if not self.geometry_enabled:return super().forward(descriptors)
        hidden=F.relu(self.hidden(descriptors));normalized=self.norm(hidden)
        geometry=pair_geometry(descriptors[...,-5:-1],descriptors[...,-1])
        bias=2*torch.tanh(self.geometry(geometry))
        v,q=descriptors.shape[:2];bias=bias.permute(0,3,1,2).reshape(v*4,q,q)
        hidden=hidden+.1*self.attention(normalized,normalized,normalized,attn_mask=bias,need_weights=False)[0]
        return self.output(hidden).squeeze(-1)
