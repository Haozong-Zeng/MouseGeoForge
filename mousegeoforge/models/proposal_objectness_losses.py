"""Isolated objectness-loss control; matching and inference stay unchanged."""
import torch
from torch.nn import functional as F
from .proposal_instance_grouping import proposal_grouping_loss

def balanced_objectness(logits, available, matched_queries, *, gamma=0., annotation_complete=True):
    if gamma < 0:raise ValueError('gamma must be nonnegative')
    target=torch.zeros_like(logits);target[matched_queries]=1
    positive=target.bool()&available;negative=~positive&available
    components=[]
    for mask in [positive,negative if annotation_complete else torch.zeros_like(negative)]:
        if not mask.any():continue
        x=logits[mask];y=target[mask];ce=F.binary_cross_entropy_with_logits(x,y,reduction='none')
        if gamma:
            p=x.sigmoid();pt=p*y+(1-p)*(1-y);ce=ce*(1-pt).pow(gamma)
        components.append(ce.mean())
    return torch.stack(components).mean() if components else logits.sum()*0

def grouping_objectness_loss(output,*args,gamma=0.,annotation_complete=True):
    result=proposal_grouping_loss(output,*args,annotation_complete=annotation_complete)
    # gamma=0 returns the historical tensors exactly, not an approximate replay.
    if gamma:
        result['objectness']=balanced_objectness(output['object_logits'],output['center_available'],result['matched_queries'],gamma=gamma,annotation_complete=annotation_complete)
        result['total']=result['assignment']+result['objectness']
    return result
