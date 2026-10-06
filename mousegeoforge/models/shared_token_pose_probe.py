"""Diagnostic two-pass2D composition; original multiview geometry retained.

Only six evaluated2D fields are replaced. Other pose/curve auxiliaries remain
unreconciled and are not suitable for joint training. Not a model factory.
"""
import torch
from .independent_view_pose_probe import POSE_KEYS
from .view_token_probe import shared_first_view_tokens

@torch.no_grad()
def shared_token_pose_probe(images, multiview, model):
    if images.ndim!=5 or images.shape[1]<1:raise ValueError('Expected B,V,C,H,W')
    for key in POSE_KEYS:
        if key not in multiview or multiview[key].shape[:2]!=images.shape[:2]:raise ValueError('Invalid2D field '+key)
    if images.shape[1]==1:return dict(multiview)
    with shared_first_view_tokens(model.aggregator,enabled=True):
        shared=model(images,coupling='off')
    result=dict(multiview)
    for key in POSE_KEYS:
        if key not in shared or shared[key].shape!=multiview[key].shape:raise ValueError('Shared2D shape mismatch '+key)
        result[key]=shared[key]
    return result
