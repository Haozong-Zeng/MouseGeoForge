"""Optional ordered Core23 tail supervision; no change to the prediction head."""
import torch
from torch.nn import functional as F
from training.instance_pose_losses import hungarian_match_instances

TAIL = slice(7, 14)  # tail_base, q1..q5, tail_tip


def sample_tail(points, valid, samples=5):
    """Catmull-Rom samples in semantic order; never bridge a missing label.

    Boundary controls repeat the endpoint. All four controls must be labeled.
    The fit interpolates the annotations, it is not an observed tail silhouette.
    """
    if points.shape[-2:] != (7, 2) or valid.shape != points.shape[:-1]:
        raise ValueError('Expected [...,7,2] points and [...,7] validity')
    if samples < 2:
        raise ValueError('At least two samples per segment')
    indices=torch.tensor([[max(i-1,0),i,i+1,min(i+2,6)] for i in range(6)],device=points.device)
    controls=points[...,indices,:]
    p0,p1,p2,p3=controls.unbind(-2)
    t=torch.linspace(0,1,samples,device=points.device,dtype=points.dtype)[...,None]
    curve=.5*(2*p1[...,None,:]+(-p0+p2)[...,None,:]*t+
              (2*p0-5*p1+4*p2-p3)[...,None,:]*t.square()+
              (-p0+3*p1-3*p2+p3)[...,None,:]*t.pow(3))
    mask=valid[...,indices].all(-1)[...,None].expand(*valid.shape[:-1],6,samples)
    return curve,mask


def tail_loss(xy, target, valid, boxes, image_size, mode):
    """Equal-scale point-vs-curve auxiliary position loss (no straightening).

    Both modes keep the original point loss. This isolates the additional
    representation, not a new head or a replacement of missing point labels.
    Known-occluded coordinates remain supervised; unknown ones do not.
    """
    if mode not in ('points','curve'):
        raise ValueError(mode)
    h,w=image_size
    scale=xy.new_tensor([w,h])
    diagonal=(boxes[...,2:].detach()*scale).norm(dim=-1).clamp_min(1)
    good=valid.bool() & torch.isfinite(target).all(-1)
    # Sanitize BEFORE interpolation so invalid NaNs cannot poison gradients.
    gt=torch.where(good[...,None],target,torch.zeros_like(target))
    prediction=torch.where(good[...,None],xy,torch.zeros_like(xy))
    gt=gt*scale/diagonal[...,None,None]
    prediction=prediction*scale/diagonal[...,None,None]
    if mode=='curve':
        prediction,mask=sample_tail(prediction,good)
        gt,_=sample_tail(gt,good)
    else: mask=good
    if not mask.any():
        return xy.sum()*0
    return F.smooth_l1_loss(prediction[mask],gt[mask],beta=.01)


def matched_tail_loss(predictions, targets, image_size, mode):
    matches=hungarian_match_instances(predictions,targets)
    predicted=[]; labeled=[]; valid=[]; boxes=[]
    for b,views in enumerate(matches):
        for v,(qs,ms) in enumerate(views):
            for q,m in zip(qs.tolist(),ms.tolist()):
                predicted.append(predictions['instance_keypoint_xy'][b,v,q,TAIL])
                labeled.append(targets['keypoint_xy'][b,v,m,TAIL])
                valid.append(targets['keypoint_valid'][b,v,m,TAIL])
                boxes.append(targets['boxes_cxcywh'][b,v,m])
    if not predicted: return predictions['instance_keypoint_xy'].sum()*0
    return tail_loss(torch.stack(predicted),torch.stack(labeled),torch.stack(valid),
                     torch.stack(boxes),image_size,mode)
