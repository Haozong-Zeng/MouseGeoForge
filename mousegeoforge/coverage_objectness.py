"""Training-only one-to-one labels preserving fixed-box geometric coverage.

Uses GT exclusively to construct supervision, never to filter inference candidates.
This is a controlled frozen-box diagnostic, not a replacement DETR matcher.
"""
import torch
from mousegeoforge.candidate_ceiling import geometric_ceiling


@torch.no_grad()
def coverage_objectness_targets(boxes,targets,valid):
    if boxes.ndim!=3 or targets.ndim!=3 or boxes.shape[0]!=targets.shape[0]:
        raise ValueError('Expected per-view candidate and target boxes')
    if valid.shape!=targets.shape[:2]:raise ValueError('Invalid target validity shape')
    if not torch.isfinite(boxes).all() or not torch.isfinite(targets[valid]).all():
        raise ValueError('Nonfinite boxes')
    labels=boxes.new_zeros(boxes.shape[:2])
    for view in range(len(boxes)):
        bound=geometric_ceiling(boxes[view],targets[view,valid[view]])
        for match in bound['matches']:
            if match['covered']:labels[view,match['query']]=1
    return labels
