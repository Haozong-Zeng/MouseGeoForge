"""Experimental coarse-body auxiliary loss without modifying locked upstream code."""
from types import FunctionType
import torch
from torch.nn import functional as F
import training.instance_pose_losses as upstream


def matched_coarse_body_loss(predictions, targets, matches):
    coarse=predictions['instance_keypoint_coarse_xy']
    if coarse.shape[-2]!=23:raise ValueError('Audited Core23 order required')
    predicted=[];labeled=[];count=0
    for b,views in enumerate(matches):
        for v,(qs,ms) in enumerate(views):
            for q,m in zip(qs.tolist(),ms.tolist()):
                valid=targets['keypoint_valid'][b,v,m].clone().bool()
                valid[7:14]=False  # Tail objective stays unchanged in this arm.
                gt=targets['keypoint_xy'][b,v,m]
                if not torch.isfinite(gt[valid]).all():raise ValueError('Nonfinite valid xy label')
                if valid.any():
                    predicted.append(coarse[b,v,q][valid]);labeled.append(gt[valid]);count+=int(valid.sum())
    if not count:return coarse.sum()*0.,0
    return F.smooth_l1_loss(torch.cat(predicted),torch.cat(labeled),beta=.01),count


def coarse_aux_pose_loss(predictions,targets,weights,*,aux_weight=2.5,**kwargs):
    if aux_weight<0:raise ValueError('Auxiliary weight must be nonnegative')
    # Clone only the Python function's lookup environment, not its implementation.
    # No global monkeypatch or changed matching cost: one match serves both losses.
    matches=upstream.hungarian_match_instances(predictions,targets)
    def fixed_matching(p,t):
        if p is not predictions or t is not targets:raise AssertionError('Matching inputs changed')
        return matches
    base=upstream.instance_pose_loss
    namespace=dict(base.__globals__,hungarian_match_instances=fixed_matching)
    same_loss=FunctionType(base.__code__,namespace,base.__name__,base.__defaults__,base.__closure__)
    same_loss.__kwdefaults__=base.__kwdefaults__
    base_loss,logs=same_loss(predictions,targets,weights,**kwargs)
    aux,count=matched_coarse_body_loss(predictions,targets,matches)
    return base_loss+aux_weight*aux,dict(logs,loss_instance_base_pose=base_loss,
        loss_instance_coarse_body=aux,coarse_body_points=count)
