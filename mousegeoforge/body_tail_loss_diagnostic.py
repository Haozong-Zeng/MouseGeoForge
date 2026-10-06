"""Partition the existing coordinate loss, without changing its weighting."""
import torch
from torch.nn import functional as F


def partition_coordinate_loss(refined, coarse, sample):
    """Return body/tail contributions whose sum is the original instance mean."""
    terms = {'body': [], 'tail': []}
    for view, (queries, instances) in enumerate(sample['matches']):
        for query, instance in zip(queries, instances):
            valid = sample['target']['keypoint_valid'][0, view, instance].to(refined.device).bool()
            if not valid.any():
                continue
            gt = sample['target']['keypoint_xy'][0, view, instance].to(refined.device)
            ids = torch.arange(valid.numel(), device=valid.device)[valid]
            tail = (ids >= 7) & (ids < 14)
            errors = F.smooth_l1_loss(refined[view, query, valid], gt[valid], beta=.05, reduction='none')
            errors = errors + .5 * F.smooth_l1_loss(coarse[view, query, valid], gt[valid], beta=.05, reduction='none')
            for name, selected in [('body', ~tail), ('tail', tail)]:
                # Divide by ALL valid coordinates, not only this anatomical subset.
                terms[name].append(errors[selected].sum() / errors.numel())
    if not terms['body']:
        raise ValueError('No GT supervision')
    return {name: torch.stack(values).mean() for name, values in terms.items()}
