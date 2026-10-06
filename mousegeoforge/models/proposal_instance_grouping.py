"""Predicted-center proposal grouping probe; GT appears only in the loss.

Both arms share architecture/parameters and differ only in within-instance
point-type attention. Inspired by CenterGroup, not a reproduction.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from .dense_instance_centers import decode_centers
from .regional_peak_candidates import regional_top_peaks
from .local_instance_grouping import assignment_targets,decode_grouped


class ProposalInstanceGrouping(nn.Module):
    def __init__(self,*,interaction=False,proposals=8,width=64):
        super().__init__()
        if proposals<2 or width%4:raise ValueError('Need at least two proposals and width divisible by four')
        self.interaction=interaction;self.proposals=proposals;self.width=width
        self.center_embed=nn.Sequential(nn.Linear(36,width),nn.GELU(),nn.LayerNorm(width))
        self.center_attention=nn.MultiheadAttention(width,4,batch_first=True)
        self.center_norm=nn.LayerNorm(width)
        self.types=nn.Parameter(torch.randn(5,width)*.02)
        self.candidate=nn.Sequential(nn.Linear(35,width),nn.GELU(),nn.LayerNorm(width))
        self.point_attention=nn.MultiheadAttention(width,4,batch_first=True)
        self.point_norm=nn.LayerNorm(width)
        self.point_mlp=nn.Sequential(nn.Linear(width,width*2),nn.GELU(),nn.Linear(width*2,width))
        self.relative=nn.Sequential(nn.Linear(4,width),nn.GELU(),nn.Linear(width,1))
        self.query_key=nn.Linear(width,width);self.null_score=nn.Linear(width,1)
        self.objectness=nn.Sequential(nn.Linear(width*2,width),nn.GELU(),nn.Linear(width,1))

    def forward(self,evidence,center_evidence,image_hw):
        features=evidence['features'];device=features.device;ih,iw=map(int,image_hw);h,w=features.shape[-2:]
        if features.ndim!=3 or features.shape[0]!=32:raise ValueError('Need HRNet32 features')
        proposal=decode_centers(center_evidence,image_hw,capacity=self.proposals)
        centers=proposal['centers'];valid=proposal['available'];score=proposal['object_probability']
        # Deliberately ignore instance_active: raw heatmap0.5 is NOT the gate.
        visual=F.layer_norm(features.permute(1,2,0),(32,))
        def gather(xy):
            ix=(xy[...,0]*iw/4-.5).round().long().clamp(0,w-1)
            iy=(xy[...,1]*ih/4-.5).round().long().clamp(0,h-1)
            return visual[iy,ix]
        rank=torch.arange(self.proposals,device=device,dtype=features.dtype)/max(self.proposals-1,1)
        token=self.center_embed(torch.cat([gather(centers),centers,score[:,None],rank[:,None]],-1))
        token=token.masked_fill(~valid[:,None],0.)
        padding=~valid.clone()
        if padding.all():padding[0]=False # finite sentinel; never becomes an available proposal
        context,_=self.center_attention(token[None],token[None],token[None],key_padding_mask=padding[None],need_weights=False)
        instance=self.center_norm(token+context[0])
        p=regional_top_peaks(evidence,image_hw,count=6,radius_px=8.)
        xy=p['xy'].to(device);available=p['available'].to(device);prob=p['probability'].to(device)
        relative_prob=prob/prob.sum(-1,keepdim=True).clamp_min(1e-12);prior=relative_prob.clamp_min(1e-12).log()
        keys=self.candidate(torch.cat([gather(xy),xy,prior[...,None]/12],-1))
        point=instance[:,None]+self.types[None]
        delta=xy[None]-centers[:,None,None];geometry=torch.cat([delta,delta.abs()],-1)
        geometry_score=self.relative(geometry).squeeze(-1)
        initial=torch.einsum('qkd,knd->qkn',point,keys)/math.sqrt(self.width)+geometry_score
        weight=initial.masked_fill(~available[None],-1e4).softmax(-1)*available[None]
        weight=weight/weight.sum(-1,keepdim=True).clamp_min(1e-12)
        point=point+torch.einsum('qkn,knd->qkd',weight,keys)
        mask=None if self.interaction else ~torch.eye(5,dtype=torch.bool,device=device)
        attended,_=self.point_attention(point,point,point,attn_mask=mask,need_weights=False)
        point=self.point_norm(point+attended);point=point+self.point_mlp(point)
        logits=torch.einsum('qkd,knd->qkn',self.query_key(point),keys)/math.sqrt(self.width)+geometry_score+.25*prior[None]
        logits=logits.masked_fill(~available[None],-torch.inf)
        logits=torch.cat([logits,self.null_score(point)],-1)
        obj=self.objectness(torch.cat([instance,point.mean(1)],-1)).squeeze(-1)
        return dict(centers=centers,center_available=valid,center_heatmap_score=score,
                    object_logits=obj,assignment_logits=logits,candidate_xy=xy,
                    candidate_available=available,candidate_probability=prob)


def proposal_grouping_loss(output,target,eligible,instance_valid,mask_centroid,centroid_valid,image_hw,*,annotation_complete=True):
    if not torch.equal(instance_valid.bool(),centroid_valid.bool()):raise ValueError('Every known instance requires a centroid')
    if not torch.isfinite(mask_centroid[centroid_valid]).all():raise ValueError('Nonfinite known center')
    zero=output['object_logits'].sum()*0;device=target.device;ids=instance_valid.nonzero().flatten()
    proposal_ids=output['center_available'].nonzero().flatten()
    if len(ids)>2:raise ValueError('This probe supports at most two targets; do not drop extra mice')
    query=torch.empty(0,dtype=torch.long,device=device);matched=query.clone()
    if len(ids) and len(proposal_ids):
        scale=target.new_tensor([image_hw[1],image_hw[0]])
        distance=torch.cdist(output['centers'][proposal_ids].detach()*scale,mask_centroid[ids]*scale)
        # First maximize admissible matches, then minimize distance; GT only here.
        cost=distance.clone().masked_fill(distance>64,1e6)
        rr,cc=linear_sum_assignment(cost.cpu().numpy())
        rr=torch.as_tensor(rr,device=device);cc=torch.as_tensor(cc,device=device)
        good=distance[rr,cc]<=64;query=proposal_ids[rr[good]];matched=ids[cc[good]]
    distributions=assignment_targets(output,target,eligible,image_hw)
    if len(query):
        logp=output['assignment_logits'][query].log_softmax(-1)
        logp=torch.where(distributions[matched]>0,logp,torch.zeros_like(logp))
        ce=-(distributions[matched]*logp).sum(-1);known=eligible[matched]
        assignment=ce[known].mean() if known.any() else zero
    else:assignment=zero
    object_target=torch.zeros_like(output['object_logits']);object_target[query]=1
    positive=object_target.bool()&output['center_available'];negative=~positive&output['center_available']
    # Equal positive/negative contributions avoid all-background minima from
    # eight proposals for one animal. Missing instances remain reported below.
    components=[]
    if positive.any():components.append(F.binary_cross_entropy_with_logits(output['object_logits'][positive],object_target[positive]))
    if annotation_complete and negative.any():components.append(F.binary_cross_entropy_with_logits(output['object_logits'][negative],object_target[negative]))
    objectness=torch.stack(components).mean() if components else zero
    return dict(total=assignment+objectness,assignment=assignment,objectness=objectness,
                matched_queries=query,matched_targets=matched,missing_center_proposals=zero.detach()+len(ids)-len(query))


@torch.no_grad()
def decode_proposal_grouped(output,object_threshold=.5,max_instances=2):
    if max_instances<1 or not 0<object_threshold<1:raise ValueError('Invalid decode limits')
    logits=output['object_logits'].masked_fill(~output['center_available'],-torch.inf)
    ranked=torch.argsort(logits,descending=True,stable=True)
    keep=torch.zeros_like(output['center_available']);keep[ranked[:max_instances]]=True
    modified=dict(output);modified['object_logits']=logits.masked_fill(~keep,-torch.inf)
    result=decode_grouped(modified,object_threshold)
    result['proposal_object_probability']=output['object_logits'].sigmoid().masked_fill(~output['center_available'],0.)
    return result
