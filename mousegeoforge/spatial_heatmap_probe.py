"""Experimental spatial readout diagnostics, not a production pose decoder.

An optional predicted-mask prior may bias invisible joints and must not be
interpreted as amodal supervision. Disabled calls return the original object.
"""
import torch
from torch.nn import functional as F


def spatial_expectation(original, logits=None, *, enabled=False, mask_logits=None):
    if not enabled:
        return original
    if logits is None or logits.ndim < 3 or logits.shape[:-2] != original.shape[:-1] or original.shape[-1] != 2:
        raise ValueError('Expected matching [...,K,H,W] logits and [...,K,2] coordinates')
    if not torch.isfinite(logits).all():
        raise ValueError('Non-finite spatial logits')
    h,w=logits.shape[-2:]
    if min(h,w)<1:raise ValueError('Empty spatial grid')
    if mask_logits is not None:
        if mask_logits.shape != (*logits.shape[:-3],h,w) or not torch.isfinite(mask_logits).all():
            raise ValueError('Mask must have matching instance/spatial axes')
        logits=logits+F.logsigmoid(mask_logits).unsqueeze(-3)
    prob=logits.flatten(-2).softmax(-1).reshape_as(logits)
    ys=(torch.arange(h,device=logits.device,dtype=logits.dtype)+.5)/h
    xs=(torch.arange(w,device=logits.device,dtype=logits.dtype)+.5)/w
    return torch.stack(((prob*xs).sum((-2,-1)),(prob*ys[:,None]).sum((-2,-1))),-1)
