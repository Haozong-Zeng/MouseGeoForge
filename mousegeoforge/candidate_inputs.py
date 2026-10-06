"""Frozen prediction-only local candidates plus sampled spatial features."""
import numpy as np
import torch
from torch.nn import functional as F
from .epipolar_candidate_readout import probability_candidates


def local_candidate_inputs(probabilities,baseline_xy,memory):
    """Return six alternatives: original expectation followed by five peaks.

    probabilities[Q,K,H,W], baseline_xy[Q,K,2], memory[1,C,H,W].
    No labels/boxes/camera IDs enter this construction. Samples use only
    predicted positions. Memory is frozen and stored as half precision.
    """
    p=np.asarray(probabilities,np.float32);xy=np.asarray(baseline_xy,np.float32)
    if p.ndim!=4 or xy.shape!=(*p.shape[:2],2) or memory.ndim!=4 or memory.shape[0]!=1 or memory.shape[-2:]!=p.shape[-2:]:raise ValueError('Invalid candidate input shape')
    if not np.isfinite(xy).all():raise ValueError('Nonfinite baseline coordinate')
    q,k,h,w=p.shape;coords=np.empty((q,k,6,2),np.float32);mass=np.empty((q,k,6),np.float32);coords[:,:,0]=xy
    for qi in range(q):
        for ki in range(k):
            coords[qi,ki,1:],mass[qi,ki,1:]=probability_candidates(p[qi,ki],[1,1])
            x,y=np.minimum(np.maximum((xy[qi,ki]*[w,h]).astype(int),0),[w-1,h-1])
            mass[qi,ki,0]=p[qi,ki,max(0,y-1):y+2,max(0,x-1):x+2].sum()
    grid=torch.from_numpy(coords.reshape(1,-1,1,2)).to(memory.device)*2-1
    with torch.no_grad():sample=F.grid_sample(memory.detach().float(),grid,mode='bilinear',padding_mode='border',align_corners=False)
    sample=sample[0,:,:,0].T.reshape(q,k,6,memory.shape[1]).half().cpu()
    return dict(candidate_xy=torch.from_numpy(coords),candidate_mass=torch.from_numpy(mass),candidate_memory=sample)
