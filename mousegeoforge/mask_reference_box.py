"""Opt-in explicit mask-reference residual; signed coefficients, not confidence."""
import torch
from torch import nn
from torch.nn import functional as F
from mousegeoforge.mask_box_refiner import MaskConditionedBoxRefiner


class MaskReferenceBoxRefiner(MaskConditionedBoxRefiner):
    def __init__(self,token_dim=256,hidden_dim=128):
        super().__init__(token_dim,hidden_dim)
        self.reference_gate=nn.Linear(hidden_dim,2)
        nn.init.zeros_(self.reference_gate.weight);nn.init.zeros_(self.reference_gate.bias)

    def forward(self,boxes,tokens=None,mask_features=None,*,enabled=False,use_prior=False):
        if not enabled:return boxes
        # Both ablation arms read exactly the same mask/token/box features.
        base=super().forward(boxes,tokens,mask_features,enabled=True,use_mask=True)
        if not use_prior:return base
        shape=mask_features.detach().to(boxes)
        features=torch.cat((F.normalize(tokens.detach().to(boxes),dim=-1,eps=1e-6),boxes.detach(),shape),dim=-1)
        coefficients=.5*self.reference_gate(F.gelu(self.input(features))).tanh()
        # No positive clipping at initialization: both signs retain a gradient.
        # Missing mask has no reference contribution; normal learned residual remains.
        coefficients=coefficients*(shape[...,8:9]>0).to(boxes.dtype)
        references=shape[...,:8].reshape(*boxes.shape[:-1],2,4)
        correction=(coefficients[...,None]*(references-boxes.detach()[...,None,:])).sum(-2)
        updated=base+correction
        return torch.cat((updated[...,:2].clamp(0.,1.),updated[...,2:].clamp(1e-4,1.)),dim=-1)
