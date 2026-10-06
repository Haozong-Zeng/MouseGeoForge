"""Independent conditional gate prototype; not installed by production models.

No GT visibility, camera role, animal count, or manual confidence enters forward.
The static comparator has K offsets; dynamic adds a shared token/evidence MLP.
This is deliberately NOT described as a parameter-count-matched comparison.
"""
import torch
from torch import nn


class InstanceAdaptiveGate(nn.Module):
    def __init__(self,token_dim=256,keypoints=23,hidden_dim=32,*,dynamic=True):
        super().__init__();self.keypoints=keypoints;self.token_dim=token_dim;self.dynamic=dynamic
        self.offset=nn.Parameter(torch.zeros(keypoints))
        if dynamic:
            self.network=nn.Sequential(nn.Linear(token_dim+11,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,1))
            nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
        else:self.network=None

    def forward(self,raw_logits,anchor,base_gate,tokens,quality):
        # quality = normalized entropy, peak probability, predicted visibility,
        # predicted mask area fraction and predicted objectness, all detached.
        if raw_logits.shape!=anchor.shape or raw_logits.shape[-2:]!=(self.keypoints,2):raise ValueError('Expected N,Q,K,2')
        if tokens.shape!=(*raw_logits.shape[:-1],self.token_dim) or quality.shape!=(*raw_logits.shape[:-1],5):raise ValueError('Evidence shape mismatch')
        if base_gate.shape!=(self.keypoints,) or not torch.isfinite(base_gate).all() or ((base_gate<0)|(base_gate>1)).any():raise ValueError('Base gate must be in [0,1]')
        raw=raw_logits.detach();spatial=anchor.detach().clamp(1e-5,1-1e-5);g0=base_gate.detach()[None,None]
        offset=self.offset[None,None].expand(raw.shape[:-1])
        if self.network is not None:
            xy=raw.sigmoid();features=torch.cat([tokens.detach(),xy,spatial,spatial-xy,quality.detach()],dim=-1)
            offset=offset+self.network(features).squeeze(-1)
        change=offset.tanh()
        gate=g0+torch.where(offset>=0,(1-g0)*change,g0*change)
        output=raw+gate[...,None]*(torch.logit(spatial)-raw)
        return output,gate


def predicted_quality(scores,visibility_logits,mask_logits,objectness_logits):
    """NQKHW + NQK + NQHW + NQ -> NQK5, no label inputs."""
    with torch.no_grad():
        prob=scores.flatten(-2).softmax(-1);size=prob.shape[-1]
        entropy=-(prob*prob.clamp_min(1e-12).log()).sum(-1)/max(1.,float(torch.log(torch.tensor(float(size)))))
        peak=prob.amax(-1);area=mask_logits.sigmoid().mean((-2,-1))[:,:,None].expand_as(peak)
        score=objectness_logits.sigmoid()[:,:,None].expand_as(peak)
        return torch.stack([entropy,peak,visibility_logits.sigmoid(),area,score],-1).detach()
