"""Detection-only adaptation of original frozen instance tokens."""
import torch
from torch import nn
from vggt_omega.models.heads.instance_pose_head import _MLP


class InstanceDetectionReadout(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.objectness=nn.Linear(hidden_dim,1)
        self.box=_MLP(hidden_dim,hidden_dim,4,3)

    def forward(self,tokens):
        if tokens.ndim!=3 or tokens.shape[-1]!=self.hidden_dim:
            raise ValueError('Expected [images,queries,hidden_dim] frozen instance tokens')
        features=tokens.detach()
        raw=self.box(features)
        boxes=torch.cat((raw[...,:2].sigmoid(),.02+.98*raw[...,2:].sigmoid()),-1)
        return dict(instance_boxes_cxcywh=boxes,instance_objectness_logits=self.objectness(features).squeeze(-1))
