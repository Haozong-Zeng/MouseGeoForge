"""Experimental body16 epipolar loss; leaves locked upstream losses unchanged."""
from itertools import combinations
import math
from types import FunctionType
import torch
from torch.nn import functional as F
import training.instance_pose_losses as upstream

BODY = tuple(list(range(7))+list(range(14,23)))


@torch.no_grad()
def fundamental_from_calibration(k1,e1,k2,e2):
    """OpenCV world-to-camera calibration; None for a coincident camera pair."""
    k1,k2,e1,e2=[x.detach().double() for x in (k1,k2,e1,e2)]
    if not all(torch.isfinite(x).all() for x in (k1,k2,e1,e2)):
        raise ValueError('Nonfinite available calibration')
    rotation=e2[:3,:3]@e1[:3,:3].T
    translation=e2[:3,3]-rotation@e1[:3,3]
    if float(translation.norm())<1e-10:return None
    x,y,z=translation;zero=x*0
    cross=torch.stack((torch.stack((zero,-z,y)),torch.stack((z,zero,-x)),torch.stack((-y,x,zero))))
    matrix=torch.linalg.inv(k2).T@cross@rotation@torch.linalg.inv(k1)
    return matrix/matrix.norm().clamp_min(1e-15)


def signed_sampson_normalized(xy1,xy2,fundamental,image_size):
    """Differentiable signed Sampson distance / image diagonal; xy are UV."""
    h,w=image_size
    if h<=0 or w<=0:raise ValueError('Invalid image dimensions')
    dtype=torch.float64 if xy1.dtype==torch.float64 else torch.float32
    xy1,xy2=xy1.to(dtype),xy2.to(dtype);matrix=fundamental.to(device=xy1.device,dtype=dtype)
    pixel=xy1.new_tensor([w,h])
    a=torch.cat((xy1*pixel,torch.ones_like(xy1[...,:1])),dim=-1)
    b=torch.cat((xy2*pixel,torch.ones_like(xy2[...,:1])),dim=-1)
    line2=a@matrix.T;line1=b@matrix
    numerator=(b*line2).sum(-1)
    denominator=(line2[...,:2].square().sum(-1)+line1[...,:2].square().sum(-1)).clamp_min(1e-12).sqrt()
    return numerator/denominator/math.hypot(h,w)


def matched_epipolar_body_loss(predictions,targets,matches,*,image_size):
    xy=predictions['instance_keypoint_xy']
    if xy.shape[-2:]!=(23,2):raise ValueError('Audited Core23 order required')
    zero=xy.sum()*0.;logs=dict(epipolar_instances=0,epipolar_pairs=0,epipolar_points=0,epipolar_degenerate_pairs=0)
    required=('intrinsics','extrinsics','camera_available','instance_ids','keypoint_valid')
    if xy.shape[1]<2 or any(k not in targets for k in required):return zero,logs
    instances=[]
    for b,views in enumerate(matches):
        identities={};matrices={}
        for v,(qs,ms) in enumerate(views):
            for q,m in zip(qs.tolist(),ms.tolist()):
                identity=int(targets['instance_ids'][b,v,m])
                if identity<0:continue
                byview=identities.setdefault(identity,{})
                if v in byview:raise ValueError('Duplicate persistent instance ID in one view')
                byview[v]=(q,m)
        for byview in identities.values():
            pairs=[]
            for v1,v2 in combinations(sorted(byview),2):
                if not (targets['camera_available'][b,v1] and targets['camera_available'][b,v2]):continue
                q1,m1=byview[v1];q2,m2=byview[v2]
                valid=targets['keypoint_valid'][b,v1,m1].bool() & targets['keypoint_valid'][b,v2,m2].bool()
                valid=valid.clone();valid[7:14]=False
                if not valid.any():continue
                key=v1,v2
                if key not in matrices:
                    matrices[key]=fundamental_from_calibration(targets['intrinsics'][b,v1],targets['extrinsics'][b,v1],
                        targets['intrinsics'][b,v2],targets['extrinsics'][b,v2])
                matrix=matrices[key]
                if matrix is None:logs['epipolar_degenerate_pairs']+=1;continue
                a,c=xy[b,v1,q1,valid],xy[b,v2,q2,valid]
                if not torch.isfinite(a).all() or not torch.isfinite(c).all():raise ValueError('Nonfinite valid prediction')
                residual=signed_sampson_normalized(a,c,matrix,image_size)
                pairs.append(F.smooth_l1_loss(residual,torch.zeros_like(residual),beta=.01))
                logs['epipolar_pairs']+=1;logs['epipolar_points']+=int(valid.sum())
            if pairs:
                instances.append(torch.stack(pairs).mean());logs['epipolar_instances']+=1
    return (torch.stack(instances).mean() if instances else zero),logs


def epipolar_aux_pose_loss(predictions,targets,weights,*,aux_weight=1.,image_size=None,**kwargs):
    if not math.isfinite(aux_weight) or aux_weight<0:raise ValueError('Invalid auxiliary weight')
    matches=upstream.hungarian_match_instances(predictions,targets)
    def fixed_matching(p,t):
        if p is not predictions or t is not targets:raise AssertionError('Matching inputs changed')
        return matches
    base=upstream.instance_pose_loss
    namespace=dict(base.__globals__,hungarian_match_instances=fixed_matching)
    same_loss=FunctionType(base.__code__,namespace,base.__name__,base.__defaults__,base.__closure__)
    same_loss.__kwdefaults__=base.__kwdefaults__
    original,logs=same_loss(predictions,targets,weights,image_size=image_size,**kwargs)
    if aux_weight==0:
        return original,dict(logs,loss_instance_base_pose=original,loss_instance_epipolar=original.detach()*0.,
            epipolar_instances=0,epipolar_pairs=0,epipolar_points=0,epipolar_degenerate_pairs=0)
    if image_size is None:raise ValueError('Epipolar loss needs image_size')
    aux,counts=matched_epipolar_body_loss(predictions,targets,matches,image_size=image_size)
    return original+aux_weight*aux,dict(logs,loss_instance_base_pose=original,loss_instance_epipolar=aux,**counts)
