"""Detached matching-cost diagnostics, not an inference selector or loss patch."""
import itertools
import torch
from torch.nn import functional as F
from training.instance_pose_losses import _linear_sum_assignment


@torch.no_grad()
def matching_costs(pred, target, batch_index, view_index):
    ids = torch.where(target['instance_valid'][batch_index,view_index])[0]
    score = pred['instance_objectness_logits'][batch_index,view_index].sigmoid()
    box = pred['instance_boxes_cxcywh'][batch_index,view_index]
    gtbox = target['boxes_cxcywh'][batch_index,view_index,ids]
    box_cost = torch.cdist(gtbox,box,p=1)
    if 'box_valid' in target:
        box_cost = box_cost*target['box_valid'][batch_index,view_index,ids,None]
    point_cost = box_cost.new_zeros(box_cost.shape)
    for row,m in enumerate(ids):
        good = target['keypoint_valid'][batch_index,view_index,m]
        if good.any():
            point_cost[row] = (pred['instance_keypoint_xy'][batch_index,view_index,:,good]-
                              target['keypoint_xy'][batch_index,view_index,m,good]).abs().mean((-2,-1))
    # Preserve the legacy evaluation order exactly.
    original = -score[None]+5*box_cost
    original = original+5*point_cost
    bce = torch.zeros_like(original); dice = torch.zeros_like(original)
    if 'masks' in target:
        valid = target.get('mask_valid', target['instance_valid'])[batch_index,view_index,ids]
        if valid.any():
            logits = pred['instance_mask_logits'][batch_index,view_index].detach().float()
            labels = target['masks'][batch_index,view_index,ids[valid]].float()
            if not torch.isfinite(logits).all() or not torch.isfinite(labels).all():
                raise ValueError('Nonfinite supervised mask inputs')
            labels = F.interpolate(labels[:,None],size=logits.shape[-2:],mode='nearest')[:,0].flatten(1)
            logits = logits.flatten(1); probabilities = logits.sigmoid()
            bce[valid] = F.softplus(logits).mean(1)[None] - labels@logits.T/logits.shape[1]
            dice[valid] = 1-(2*labels@probabilities.T+1)/(labels.sum(1)[:,None]+probabilities.sum(1)[None]+1)
    return ids, dict(original=original, mask_bce=bce, mask_dice=dice,
                     mask_augmented=original+2*bce+2*dice)


def assignment(cost, ids):
    qs, rows = _linear_sum_assignment(cost)
    return {int(ids[m]):int(q) for q,m in zip(qs,rows)}


def assignment_gap(cost):
    """Exact first-to-second permutation gap; bounded pilot has at most2 GT."""
    rows,queries=cost.shape
    if rows==0:return None
    if rows>2:raise ValueError('Diagnostic enumeration is limited to two instances')
    values=sorted(float(sum(cost[m,q] for m,q in enumerate(p))) for p in itertools.permutations(range(queries),rows))
    return values[1]-values[0] if len(values)>1 else None


def objectness_terms(logits, matches, complete, no_object=.1):
    """Exact existing weighted-BCE target/weight and dL/dlogit, no optimization."""
    labels=torch.zeros_like(logits); weights=torch.full_like(logits,no_object)*complete[...,None]
    for b, views in enumerate(matches):
        for v,(qs,_) in enumerate(views):labels[b,v,qs]=1;weights[b,v,qs]=1
    gradient=(logits.sigmoid()-labels)*weights/weights.sum().clamp_min(1)
    return labels,weights,gradient
