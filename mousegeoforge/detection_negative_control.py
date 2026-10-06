"""Opt-in background-only increment; original detector loss remains unchanged."""
import math
import torch
from torch.nn import functional as F
from mousegeoforge.detection_readout_loss import detection_readout_loss,match_detection


def background_control_loss(predictions,boxes,valid,complete,*,additional_negative_weight=0.):
    """Keep original positive/box scale; increase unmatched-query BCE only.

    Complete annotations are required before treating an unmatched query as a
    negative. Assignment uses all valid instances, including the second mouse.
    This is a training target, never a GT-count inference filter.
    """
    if not math.isfinite(additional_negative_weight) or additional_negative_weight<0:
        raise ValueError('additional_negative_weight must be finite and nonnegative')
    base,logs=detection_readout_loss(predictions,boxes,valid,complete)
    if additional_negative_weight==0 or not complete.any():
        return base,dict(logs,additional_negative=base*0)
    pred={k:v[complete] for k,v in predictions.items()}
    logits=pred['instance_objectness_logits'];negative=torch.ones_like(logits,dtype=torch.bool)
    with torch.no_grad():
        matches=match_detection(pred,boxes[complete],valid[complete])
        for i,(qs,_) in enumerate(matches):negative[i,qs]=False
    # Exactly the base loss's denominator, not a new denominator that reduces
    # positive gradients when more background weight is requested.
    denominator=torch.where(negative,.1,1.).sum().clamp_min(1)
    addition=additional_negative_weight*(F.softplus(logits)*negative).sum()/denominator
    return base+addition,dict(logs,additional_negative=addition)
