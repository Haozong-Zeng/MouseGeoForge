"""Experimental fixed-teacher box/pose retention with GT identity alignment."""
from types import FunctionType
import math
import torch
from torch.nn import functional as F
import training.instance_pose_losses as upstream
from mousegeoforge.pilot_metrics import box_iou


@torch.no_grad()
def build_teacher_targets(predictions,targets,matches):
    """Freeze only confident GT-box-qualified foreground targets, keyed by ID."""
    packed=[]
    for b,views in enumerate(matches):
        batch=[]
        for v,(qs,ms) in enumerate(views):
            selected={};seen=set()
            for q,m in zip(qs.tolist(),ms.tolist()):
                identity=int(targets['instance_ids'][b,v,m])
                if identity<0:continue
                if identity in seen:raise ValueError('Duplicate instance identity in one view')
                seen.add(identity)
                if not targets['box_valid'][b,v,m]:continue
                score=predictions['instance_objectness_logits'][b,v,q].sigmoid()
                box=predictions['instance_boxes_cxcywh'][b,v,q]
                iou=box_iou(targets['boxes_cxcywh'][b,v,m:m+1],box[None])[0,0]
                if score<.5 or iou<.5:continue
                xy=predictions['instance_keypoint_xy'][b,v,q]
                valid=targets['keypoint_valid'][b,v,m].bool()
                if not torch.isfinite(box).all() or not torch.isfinite(xy[valid]).all():raise ValueError('Nonfinite teacher')
                selected[identity]=dict(box=box.detach().clone(),xy=xy.detach().clone(),valid=valid.detach().clone(),
                    query=q,target_slot=m,score=float(score),iou=float(iou))
            batch.append(selected)
        packed.append(batch)
    return packed


def matched_replay_retention(predictions,targets,matches,teacher_targets):
    xy=predictions['instance_keypoint_xy'];boxes=predictions['instance_boxes_cxcywh']
    zero=xy.sum()*0.+boxes.sum()*0.;instances={};point_count=0;instance_views=0
    if teacher_targets is None:return zero,dict(retained_instance_views=0,retained_points=0)
    if len(teacher_targets)!=len(matches):raise ValueError('Teacher batch mismatch')
    for b,views in enumerate(matches):
        if len(teacher_targets[b])!=len(views):raise ValueError('Teacher view mismatch')
        for v,(qs,ms) in enumerate(views):
            seen=set()
            for q,m in zip(qs.tolist(),ms.tolist()):
                identity=int(targets['instance_ids'][b,v,m])
                if identity<0:continue
                if identity in seen:raise ValueError('Duplicate instance identity in one view')
                seen.add(identity)
                if identity not in teacher_targets[b][v]:continue
                teacher=teacher_targets[b][v][identity]
                target_box=teacher['box'].detach().to(boxes)
                target_xy=teacher['xy'].detach().to(xy)
                valid=targets['keypoint_valid'][b,v,m].bool() & teacher['valid'].to(device=xy.device).bool()
                if not torch.isfinite(boxes[b,v,q]).all() or not torch.isfinite(xy[b,v,q,valid]).all():raise ValueError('Nonfinite student')
                box_term=F.smooth_l1_loss(boxes[b,v,q],target_box,beta=.01)
                point_term=F.smooth_l1_loss(xy[b,v,q,valid],target_xy[valid],beta=.01) if valid.any() else zero
                instances.setdefault((b,identity),[]).append(box_term+point_term)
                point_count+=int(valid.sum());instance_views+=1
    per_instance=[torch.stack(terms).mean() for terms in instances.values()]
    return (torch.stack(per_instance).mean() if per_instance else zero),dict(retained_instance_views=instance_views,retained_points=point_count)


def replay_retention_pose_loss(predictions,targets,weights,*,teacher_targets=None,aux_weight=1.,**kwargs):
    if not math.isfinite(aux_weight) or aux_weight<0:raise ValueError('Invalid retention weight')
    matches=upstream.hungarian_match_instances(predictions,targets)
    def fixed_matching(p,t):
        if p is not predictions or t is not targets:raise AssertionError('Matching inputs changed')
        return matches
    base=upstream.instance_pose_loss
    same=FunctionType(base.__code__,dict(base.__globals__,hungarian_match_instances=fixed_matching),base.__name__,base.__defaults__,base.__closure__)
    same.__kwdefaults__=base.__kwdefaults__
    original,logs=same(predictions,targets,weights,**kwargs)
    if aux_weight==0 or teacher_targets is None:
        return original,dict(logs,loss_instance_base_pose=original,loss_instance_retention=original.detach()*0.,retained_instance_views=0,retained_points=0)
    aux,counts=matched_replay_retention(predictions,targets,matches,teacher_targets)
    return original+aux_weight*aux,dict(logs,loss_instance_base_pose=original,loss_instance_retention=aux,**counts)
