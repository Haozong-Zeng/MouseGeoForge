"""Final-coordinate-only control for frozen-representation gate experiments."""
import torch
from torch.nn import functional as F


def refined_coordinate_loss(refined,sample):
    losses=[]
    for v,(queries,instances) in enumerate(sample['matches']):
        for q,m in zip(queries,instances):
            valid=sample['target']['keypoint_valid'][0,v,m].to(refined.device)
            gt=sample['target']['keypoint_xy'][0,v,m].to(refined.device)
            if valid.any():losses.append(F.smooth_l1_loss(refined[v,q,valid],gt[valid],beta=.05))
    if not losses:raise ValueError('No valid coordinate supervision')
    return torch.stack(losses).mean()
