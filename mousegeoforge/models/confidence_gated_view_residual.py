"""Default-off semantic residual probe, initially for single-instance scenes.

No query-ID correspondence or averaging image coordinates across cameras.
Ambiguous predicted instance counts abstain, but unique predictions cannot prove
single-instance identity. This is NOT a validated multi-animal association model.
"""
import math
import torch
from torch import nn


class ConfidenceGatedViewResidual(nn.Module):
    def __init__(self,channels=32,max_delta=.1,mode='multiview'):
        super().__init__()
        if mode not in ['multiview','self'] or max_delta<=0:raise ValueError('Invalid residual configuration')
        self.mode,self.max_delta=mode,float(max_delta)
        self.feature=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,channels),nn.GELU())
        self.delta=nn.Sequential(nn.Linear(channels*2+2,64),nn.GELU(),nn.Linear(64,2))
        self.gate=nn.Linear(channels*2+2,1)
        nn.init.zeros_(self.delta[-1].weight);nn.init.zeros_(self.delta[-1].bias)
        nn.init.zeros_(self.gate.weight);nn.init.constant_(self.gate.bias,-2.)

    def forward(self,xy,tokens,objectness_logits,quality,view_ids,view_valid,*,enabled=False):
        """Inputs are predictions/features only; quality is NOT visibility GT.

        xy[B,V,Q,K,2], tokens[B,V,Q,K,256], logits[B,V,Q], quality[B,V,Q,K]
        view_ids[B,V] identify physical cameras; view_valid[B,V] marks padding.
        Duplicate copies share support mass; exact cloned views cannot add weight.
        Self control replaces support semantic content with target content while
        preserving prediction-only availability/abstention and parameter count.
        """
        if xy.ndim!=5 or xy.shape[-1]!=2:raise ValueError('Expected xy[B,V,Q,K,2]')
        b,v,q,k=xy.shape[:4]
        if tokens.shape!=(b,v,q,k,256) or objectness_logits.shape!=(b,v,q) or quality.shape!=(b,v,q,k) or view_ids.shape!=(b,v) or view_valid.shape!=(b,v) or view_valid.dtype!=torch.bool:raise ValueError('Invalid input shapes')
        zero=xy.new_zeros(b,v,q,k)
        if not enabled or v==1:return dict(xy=xy,gate=zero,active=zero.bool())
        # Padded tokens may be NaN; remove them before any neural operation.
        clean=torch.where(view_valid[:,:,None,None,None],tokens.detach(),torch.zeros_like(tokens))
        logits=torch.where(view_valid[:,:,None],objectness_logits.detach(),torch.full_like(objectness_logits,-100))
        qual=torch.where(view_valid[:,:,None,None],quality.detach(),torch.zeros_like(quality)).clamp(0,1)
        if not torch.isfinite(clean).all() or not torch.isfinite(logits).all() or not torch.isfinite(qual).all():raise ValueError('Valid views must have finite features and quality')
        scores=logits.sigmoid();confident=(scores>=.5)&view_valid[:,:,None];unique=(confident.sum(-1)==1)&view_valid
        eligible=confident&unique[:,:,None]
        f=self.feature(clean);output=[];gates=[];actives=[]
        # Small V loop keeps explicit physical-camera exclusions and duplicate
        # support normalization auditable; no GT target/instance count is read.
        equal=view_ids[:,:,None]==view_ids[:,None,:]
        copies=(equal&view_valid[:,None,:]).sum(-1).clamp_min(1).to(xy.dtype)
        for target in range(v):
            support=eligible&(view_ids!=view_ids[:,target:target+1])[:,:,None]
            sf=f.reshape(b,v*q,k,-1);sq=qual.reshape(b,v*q,k);ss=scores.reshape(b,v*q)
            target_f=f[:,target];similarity=torch.einsum('bqkd,bskd->bqks',target_f,sf)/math.sqrt(f.shape[-1])
            support_weight=ss[:,:,None]*sq/copies[:,:,None,None].expand(b,v,q,k).reshape(b,v*q,k)
            mask=support.reshape(b,v*q)[:,None,None,:]&(support_weight.permute(0,2,1)[:,None]>0)
            logits_attn=similarity+support_weight.permute(0,2,1)[:,None].clamp_min(1e-12).log()
            weights=logits_attn.masked_fill(~mask,-1e4).softmax(-1)*mask
            weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)
            context=torch.einsum('bqks,bskd->bqkd',weights,sf)
            support_quality=torch.einsum('bqks,bsk->bqk',weights,sq)
            if self.mode=='self':context=target_f;support_quality=qual[:,target]
            inp=torch.cat([target_f,context,qual[:,target,...,None],support_quality[...,None]],-1)
            active=eligible[:,target,:,None]&mask.any(-1)
            gate=self.gate(inp).squeeze(-1).sigmoid()*active
            residual=self.max_delta*self.delta(inp).tanh()*gate[...,None]
            result=torch.where(active[...,None],(xy[:,target]+residual).clamp(0,1),xy[:,target])
            output.append(result);gates.append(gate);actives.append(active)
        return dict(xy=torch.stack(output,1),gate=torch.stack(gates,1),active=torch.stack(actives,1))
