"""Keep proposal sources intact: concatenate only, no suppression or GT selection."""
import torch


def merge_candidate_pools(original_boxes,original_scores,center_boxes,center_scores,enabled=False):
    if not enabled:return center_boxes,center_scores
    for boxes,scores in ((original_boxes,original_scores),(center_boxes,center_scores)):
        if boxes.ndim!=3 or boxes.shape[-1]!=4 or scores.shape!=boxes.shape[:2]:raise ValueError('Expected VQ4/VQ')
    if original_boxes.shape[0]!=center_boxes.shape[0]:raise ValueError('Views differ')
    return torch.cat((original_boxes,center_boxes),1),torch.cat((original_scores,center_scores),1)
