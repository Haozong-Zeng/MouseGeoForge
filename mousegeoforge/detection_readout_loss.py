"""Set detection loss on complete mask-backed frames only; no pose loss."""
import torch
from torch.nn import functional as F
from training.instance_pose_losses import hungarian_match_instances,box_cxcywh_to_xyxy,generalized_box_iou


def match_detection(predictions,boxes,valid):
    n,q=predictions['instance_objectness_logits'].shape
    m=boxes.shape[1]
    p={k:v[:,None] for k,v in predictions.items()}
    p['instance_keypoint_xy']=boxes.new_zeros(n,1,q,1,2)
    t=dict(instance_valid=valid[:,None],boxes_cxcywh=boxes[:,None],box_valid=valid[:,None],
           keypoint_xy=boxes.new_zeros(n,1,m,1,2),keypoint_valid=torch.zeros(n,1,m,1,dtype=torch.bool,device=boxes.device))
    return [row[0] for row in hungarian_match_instances(p,t)]


def detection_readout_loss(predictions,boxes,valid,complete):
    if complete.shape!=boxes.shape[:1] or complete.dtype!=torch.bool or valid.shape!=boxes.shape[:2]:
        raise ValueError('Invalid complete/box validity shapes')
    zero=predictions['instance_objectness_logits'].sum()*0+predictions['instance_boxes_cxcywh'].sum()*0
    if not complete.any():return zero,dict(objectness=zero,box_l1=zero,box_giou=zero)
    p={k:v[complete] for k,v in predictions.items()};boxes=boxes[complete];valid=valid[complete]
    matches=match_detection(p,boxes,valid)
    target=torch.zeros_like(p['instance_objectness_logits']);weight=torch.full_like(target,.1);pb=[];gb=[]
    for i,(qs,ms) in enumerate(matches):
        target[i,qs]=1.;weight[i,qs]=1.
        if len(qs):pb.append(p['instance_boxes_cxcywh'][i,qs]);gb.append(boxes[i,ms])
    obj=(F.binary_cross_entropy_with_logits(p['instance_objectness_logits'],target,reduction='none')*weight).sum()/weight.sum().clamp_min(1)
    l1=giou=zero
    if pb:
        pp,gg=torch.cat(pb),torch.cat(gb);l1=F.l1_loss(pp,gg)
        giou=(1-torch.diagonal(generalized_box_iou(box_cxcywh_to_xyxy(pp),box_cxcywh_to_xyxy(gg)))).mean()
    return obj+5*l1+2*giou,dict(objectness=obj,box_l1=l1,box_giou=giou)


def synthetic_detection_retention(predictions,teacher_boxes,teacher_logits,synthetic_train):
    """Prediction distillation, not extra GT. Caller explicitly selects TRAIN only.

    All-query soft objectness preserves negatives and positives; box retention
    uses only original confidence>=.5. No real frames enter this constraint.
    """
    logits=predictions['instance_objectness_logits'];boxes=predictions['instance_boxes_cxcywh']
    if synthetic_train.dtype!=torch.bool or synthetic_train.shape!=logits.shape[:1]:
        raise ValueError('synthetic_train must be a per-image boolean mask')
    if teacher_logits.shape!=logits.shape or teacher_boxes.shape!=boxes.shape:
        raise ValueError('Teacher and student query shapes must agree')
    zero=logits.sum()*0+boxes.sum()*0
    if not synthetic_train.any():return zero,dict(retention_objectness=zero,retention_box=zero)
    target=teacher_logits.detach()[synthetic_train].sigmoid()
    obj=F.binary_cross_entropy_with_logits(logits[synthetic_train],target)
    positive=target>=.5
    l1=F.l1_loss(boxes[synthetic_train][positive],teacher_boxes.detach()[synthetic_train][positive]) if positive.any() else zero
    return obj+5*l1,dict(retention_objectness=obj,retention_box=l1)
