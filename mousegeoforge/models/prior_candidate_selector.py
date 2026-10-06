"""Experimental bounded log-probability correction, default disabled.

Unlike unconstrained classification, a candidate cannot overcome an arbitrarily
weak native probability. This does not guarantee anatomical/instance identity.
"""
import math
import torch
from .local_candidate_selector import LocalCandidateSelector


class PriorCandidateSelector(LocalCandidateSelector):
    def __init__(self, mode='geometry', residual_bound=2.0):
        super().__init__(mode)
        if not math.isfinite(residual_bound) or residual_bound <= 0:
            raise ValueError('Positive finite residual bound required')
        self.residual_bound=float(residual_bound)

    def forward(self, semantic, local, scalars, geometry, candidate_xy,
                baseline_xy, eligible, *, enabled=False):
        original=super().forward(semantic,local,scalars,geometry,candidate_xy,
                                 baseline_xy,eligible,enabled=enabled)
        if not enabled:
            return original
        valid=original['eligible']
        # Parent includes baseline +.001; remove it before subtracting its
        # learned score. Exact same network and initialization as control.
        raw=original['logits'].clone()
        raw[...,0]=raw[...,0]-1e-3
        relative=raw-raw[...,:1]
        residual=self.residual_bound*torch.tanh(relative/self.residual_bound)
        # Scalar0 is log(relative native 3x3 mass)/20, not an annotation.
        prior=torch.where(valid[...,None],scalars.detach()[...,0]*20,0).clamp(-20,0)
        prior=prior.clone();prior[...,0]=1e-3
        logits=prior+residual
        choice=logits.argmax(-1)
        chosen=candidate_xy.gather(-2,choice[...,None,None].expand(*choice.shape,1,2)).squeeze(-2)
        active=valid&(choice!=0)
        xy=torch.where(active[...,None],chosen,baseline_xy)
        return dict(xy=xy,logits=logits,choice=choice,active=active,eligible=valid)
