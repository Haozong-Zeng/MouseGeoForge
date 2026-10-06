"""One-to-one proposal objectness with optional permutation-equivariant context."""
import torch
from torch import nn
from torch.nn import functional as F
from mousegeoforge.roi_candidate_quality import RegionQualityProbe
from mousegeoforge.dense_center_probe import candidate_metrics


def set_objectness_targets(boxes,scores,targets,valid):
    labels=torch.zeros_like(scores)
    for view in range(len(boxes)):
        gt=targets[view,valid[view]]
        matches=candidate_metrics(boxes[view],scores[view],gt)['matches']
        for match in matches:
            if match['iou']>=.5:labels[view,match['query']]=1
    return labels


class SetObjectnessProbe(RegionQualityProbe):
    def __init__(self,channels=256,hidden=64,relation_enabled=False):
        super().__init__(channels,hidden)
        self.relation_enabled=bool(relation_enabled)
        self.norm=nn.LayerNorm(hidden)
        self.attention=nn.MultiheadAttention(hidden,4,dropout=0.,batch_first=True)
        with torch.no_grad():
            self.attention.out_proj.weight.mul_(.01)
            self.attention.out_proj.bias.zero_()
        for p in list(self.norm.parameters())+list(self.attention.parameters()):p.requires_grad_(self.relation_enabled)

    def forward(self,descriptors):
        hidden=F.relu(self.hidden(descriptors))
        if self.relation_enabled:
            normalized=self.norm(hidden)
            hidden=hidden+.1*self.attention(normalized,normalized,normalized,need_weights=False)[0]
        return self.output(hidden).squeeze(-1)
