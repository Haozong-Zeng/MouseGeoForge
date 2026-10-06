"""Isolated diagnostic visual backbone with local five-point evidence output."""
from torch import nn
from .local_point_evidence_probe import LocalPointEvidenceProbe


class MixedLocalEvidence(nn.Module):
    def __init__(self,backbone,*,adapt_backbone=False):
        super().__init__();self.backbone=backbone;self.readout=LocalPointEvidenceProbe();self.adapt_backbone=adapt_backbone
        self.backbone.requires_grad_(adapt_backbone);self.backbone.eval()

    def train(self,mode=True):
        super().train(mode)
        # Full parameter gradients may flow in eval mode; running BN statistics
        # must not depend on variable-resolution, one-image training microbatches.
        self.backbone.eval()
        return self

    def forward(self,normalized_rgb,image_hw):
        if normalized_rgb.ndim!=4 or normalized_rgb.shape[:2]!=(1,3):raise ValueError('Expected one full-frame normalized RGB, no instance crops')
        features=self.backbone(normalized_rgb)
        if features.ndim!=4 or features.shape[:2]!=(1,32):raise ValueError('Expected native stride4 HRNet32 feature tensor')
        output=self.readout(features[0],image_hw);output['features']=features[0]
        return output
