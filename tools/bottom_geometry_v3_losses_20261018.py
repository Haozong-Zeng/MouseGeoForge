"""Explicit v3 scale strategies; no experiment launch or global monkey-patching.

All functions receive one independently calibrated sample. Missing objectives
contribute zero to the fixed eight-sample update, never a changed denominator.
Real source-ray supervision eliminates one latent positive depth scale using
training correspondences only. Validation-label fitting is never a primary score.
"""
from __future__ import annotations
import math
import torch
from torch.nn import functional as F
import pure_geometry_bottom_20261018 as geo

MODES=('A0','A1-GT','A1-SI')


def point_distance_scale(b):
    if not b['valid'].any():return None
    device=b['depth'].device;side=b['depth'].shape[-1]
    yy,xx=torch.meshgrid(torch.arange(side,device=device),torch.arange(side,device=device),indexing='ij')
    uv=torch.stack((xx,yy,torch.ones_like(xx)),0).reshape(3,-1).float()
    camera=torch.linalg.solve(b['K'][0],uv.expand(len(b['roles']),-1,-1))*b['depth'][0].reshape(len(b['roles']),1,-1)
    world=torch.linalg.solve(b['E'][0,:,:3,:3],camera-b['E'][0,:,:3,3,None])
    values=world.norm(dim=1)[b['valid'][0].reshape(len(b['roles']),-1)]
    scale=values.mean().clamp_min(1e-6)
    assert torch.isfinite(scale)
    return scale.detach()


def native_gt(b):
    scale=b.get('point_distance_scale')
    if scale is None:scale=point_distance_scale(b)
    if scale is None:return b
    result=dict(b);result['E']=b['E'].clone();result['E'][...,:3,3]/=scale
    result['depth']=b['depth']/scale
    return result


def scaled_prediction(pred,scale):
    pose=pred['pose_enc'];translation=pose[...,:3]*scale
    return dict(pose_enc=torch.cat((translation,pose[...,3:]),-1),depth=pred['depth']*scale)


def relative_components(pose,b):
    pe,_=geo.encoding_to_camera(pose,(512,512));e=b['E']
    pr=pe[:,1:,:3,:3]@pe[:,:1,:3,:3].transpose(-1,-2)
    tr=e[:,1:,:3,:3]@e[:,:1,:3,:3].transpose(-1,-2)
    pt=pe[:,1:,:3,3]-(pr@pe[:,:1,:3,3,None])[...,0]
    tt=e[:,1:,:3,3]-(tr@e[:,:1,:3,3,None])[...,0]
    return pe,pr,tr,pt,tt


def relative_camera_loss(pred,b,translation_vector=False):
    pose=pred['pose_enc'];pe,pr,tr,pt,tt=relative_components(pose,b)
    target=geo.extri_intri_to_pose_encoding(b['E'],b['K'],(512,512))
    loss=.1*F.smooth_l1_loss(pose[...,7:],target[...,7:])
    loss=loss+.1*geo.rotation_angle(pe[:,:1,:3,:3]@b['E'][:,:1,:3,:3].transpose(-1,-2)).mean()
    if pose.shape[1]<2:return loss
    loss=loss+geo.rotation_angle(pr@tr.transpose(-1,-2)).mean()+.05*(pr-tr).square().mean()
    loss=loss+.25*(F.normalize(pt,dim=-1)-F.normalize(tt,dim=-1)).square().sum(-1).mean()
    if translation_vector:
        # Native GT point-distance gauge makes this residual dimensionless;
        # prediction t and Z have already received the SAME positive scale.
        loss=loss+.25*F.smooth_l1_loss(pt,tt)
    elif pt.shape[1]>1:
        pn=pt.norm(dim=-1).clamp_min(1e-6).log();tn=tt.norm(dim=-1).clamp_min(1e-6).log()
        loss=loss+.1*F.smooth_l1_loss(pn-pn.mean(-1,keepdim=True),tn-tn.mean(-1,keepdim=True))
    return loss


def sparse_coefficients(pred,b):
    pairs=[]
    for source in range(len(b['roles'])):
        xy=b['xy'][0,source];uv=torch.cat((xy,torch.ones_like(xy[:,:1])),-1).T
        z=F.grid_sample(pred['depth'][0,source,...,0][None,None],((xy+.5)/512*2-1)[None,None],align_corners=False,padding_mode='border').reshape(-1)
        ray=torch.linalg.solve(b['K'][0,source],uv)*z[None]
        for target in range(len(b['roles'])):
            if source==target:continue
            good=b['known'][0,source]&b['known'][0,target]
            if not good.any():continue
            R=b['E'][0,target,:3,:3]@b['E'][0,source,:3,:3].T
            a=R@ray;offset=b['E'][0,target,:3,3]-R@b['E'][0,source,:3,3]
            pairs.append(dict(a=(b['K'][0,target]@a)[:,good],offset=b['K'][0,target]@offset,
                expected=b['xy'][0,target,good].T,
                role='bottom' if 'bottom'in(b['roles'][source],b['roles'][target]) else 'other'))
    return pairs


def sparse_grid_cost(pairs,scales):
    losses=[]
    for p in pairs:
        hom=scales[:,None,None]*p['a'][None]+p['offset'][None,:,None]
        delta=(hom[:,:2]/hom[:,2:3].clamp_min(.01)-p['expected'][None])/512
        x=delta.abs();h=torch.where(x<.03,.5*x.square()/.03,x-.015)
        losses.append(h.mean((1,2))+F.relu(.01-hom[:,2]).mean(1))
    return torch.stack(losses).mean(0)


@torch.no_grad()
def fit_training_sparse_scale(pred,b,pairs=None):
    """Only for TRAINING sparse loss, never a deployment or validation estimator.

    Search dimension is median-normalized depth, so the fixed search grid is
    itself invariant to common positive rescaling of raw predicted depths.
    The selected latent scale is detached; it is not a learned parameter.
    """
    pairs=sparse_coefficients(pred,b) if pairs is None else pairs
    if not pairs:raise ValueError('No manual cross-view support')
    device=pred['depth'].device
    reference=torch.cat([p['a'][2] for p in pairs]).abs().median().clamp_min(1e-6)
    # a_z is linear in predicted depth and transforms by the same scale.
    logs=torch.linspace(math.log(.05),math.log(20),65,device=device)
    coarse=logs.exp()/reference
    index=sparse_grid_cost(pairs,coarse).argmin();step=logs[1]-logs[0]
    fine=(logs[index]+torch.linspace(-1,1,33,device=device)*step).clamp(logs[0],logs[-1]).exp()/reference
    return fine[sparse_grid_cost(pairs,fine).argmin()].detach()


def sparse_training_loss(pred,b):
    pairs=sparse_coefficients(pred,b);scale=fit_training_sparse_scale(pred,b,pairs)
    return sparse_grid_cost(pairs,scale[None])[0],scale


def terms(pred,b,mode,real_anchor=False,enforce_guard=True):
    assert mode in MODES
    qnorm=pred['pose_enc'][...,3:7].norm(dim=-1)
    if not torch.isfinite(qnorm).all():raise FloatingPointError('Nonfinite quaternion')
    if enforce_guard and bool((qnorm<.05).any()):raise FloatingPointError('Raw quaternion norm below preregistered .05 guard')
    # Do not conceal exact zero quaternions in the model-to-rotation conversion.
    if bool((qnorm<1e-6).any()):raise FloatingPointError('Degenerate quaternion')
    result={'quaternion_radius':(qnorm-1).square().mean()}
    dense=bool(b['valid'].any())
    if b['record']['kind']=='real_labeled':
        result['real_sparse'],_=sparse_training_loss(pred,b)
        if real_anchor:result['real_camera_anchor']=relative_camera_loss(pred,b)
        return result
    if not dense:
        result['camera']=relative_camera_loss(pred,b)
        return result
    proxy=b['record'].get('depth_is_pseudo_label',False)
    target=b if mode=='A0' and not proxy else native_gt(b)
    adjusted=pred
    if mode=='A1-SI' or proxy:
        residual=target['depth'].clamp_min(1e-6).log()-pred['depth'][...,0].clamp_min(1e-6).log()
        masks=[target['valid']&target['body'],target['valid']&~target['body']]
        offset=torch.stack([residual[m].mean() for m in masks if m.any()]).mean()
        adjusted=scaled_prediction(pred,offset.exp())
        result['camera']=relative_camera_loss(adjusted,target,not proxy)
    else:result['camera']=geo.camera_loss(adjusted,target)
    z=adjusted['depth'][...,0].clamp_min(1e-6);gt=target['depth'].clamp_min(1e-6);residual=z.log()-gt.log()
    masks=[target['valid']&target['body'],target['valid']&~target['body']]
    result['depth']=torch.stack([F.smooth_l1_loss(residual[m],torch.zeros_like(residual[m])) for m in masks if m.any()]).mean()
    shapes=[]
    for view in range(z.shape[1]):
        mask=target['valid'][0,view]&target['body'][0,view]
        if mask.sum()>16:
            p=z[0,view][mask];t=gt[0,view][mask]
            shapes.append(F.smooth_l1_loss(p-p.mean(),t-t.mean(),beta=.01))
    if shapes:result['shape']=torch.stack(shapes).mean()
    pairs=[p for source in range(len(b['roles'])) for p in geo.transfer_data(adjusted,target,source)]
    transfer=geo.projection_loss(pairs)
    if transfer is not None:result['transfer']=transfer
    if proxy:
        for key in ('depth','shape','transfer'):
            if key in result:result[key]=.2*result[key]
    return result


def teacher_retention(pred,teacher,b):
    """Only reliable non-bottom TRAIN pairs; no wrong-bottom teacher target.

    Teacher eligibility is fixed by frozen release vs training GT (<10deg).
    A single depth-derived scale is shared by student relative t and depth.
    """
    pe,_=geo.encoding_to_camera(pred['pose_enc'],(512,512));te,_=geo.encoding_to_camera(teacher['pose_enc'],(512,512))
    losses=[]
    for a in range(len(b['roles'])):
        if b['roles'][a]=='bottom':continue
        for c in range(a+1,len(b['roles'])):
            if b['roles'][c]=='bottom':continue
            tR=te[0,c,:3,:3]@te[0,a,:3,:3].T;gtR=b['E'][0,c,:3,:3]@b['E'][0,a,:3,:3].T
            if float(geo.rotation_angle(tR@gtR.T).detach())>math.radians(10):continue
            pR=pe[0,c,:3,:3]@pe[0,a,:3,:3].T
            pt=pe[0,c,:3,3]-pR@pe[0,a,:3,3];tt=te[0,c,:3,3]-tR@te[0,a,:3,3]
            d=pred['depth'][0,[a,c],::16,::16,0].clamp_min(1e-6)
            td=teacher['depth'][0,[a,c],::16,::16,0].clamp_min(1e-6)
            scale=(td.log()-d.log()).mean().exp()
            loss=geo.rotation_angle(pR@tR.T)+.1*F.smooth_l1_loss(pred['pose_enc'][0,[a,c],7:],teacher['pose_enc'][0,[a,c],7:])
            loss=loss+.1*F.smooth_l1_loss((scale*d).log(),td.log())+.1*F.smooth_l1_loss(scale*pt,tt)
            losses.append(loss)
    return torch.stack(losses).mean() if losses else None
