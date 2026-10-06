"""Common assignment and true replacement of strict ambiguous real back losses."""
from dataclasses import replace
import torch
from training.instance_pose_losses import hungarian_match_instances,instance_pose_loss
from mousegeoforge.surface_curve_supervision import point_loss,curve_loss

def surface_instance_loss(prediction,target,weights,*,image_size,mode,is_real=False,
                          tangent=None,tangent_valid=None,curve_target=None,curve_valid=None,
                          along_tolerance=4.,uncertain_weight=.25,curve_weight=1.):
    if mode not in ('strict_points','weak_points','weak_points_curve6'):raise ValueError(mode)
    matching=dict(target);matching['keypoint_valid']=target['keypoint_valid'].clone()
    if is_real:matching['keypoint_valid'][...,5:7]=False
    # Both calls receive identical matching inputs. Hungarian is deterministic;
    # no back-point cost can change the assignment between experiment arms.
    matches=hungarian_match_instances(prediction,matching)
    base_weights=replace(weights,keypoint_l1=0.,local_heatmap=0.,keypoint_uncertainty=0.,keypoint_3d=0.,reprojection=0.)
    base,logs=instance_pose_loss(prediction,matching,base_weights,image_size=image_size)
    h,w=image_size;scale=prediction['instance_keypoint_xy'].new_tensor([w,h]);point_terms=[];curve_terms=[]
    for b,views in enumerate(matches):
        for v,(queries,instances) in enumerate(views):
            for q,m in zip(queries.tolist(),instances.tolist()):
                pp=prediction['instance_keypoint_xy'][b,v,q]*scale;gt=target['keypoint_xy'][b,v,m]*scale;valid=target['keypoint_valid'][b,v,m]
                weak=torch.zeros_like(valid)
                if is_real and mode!='strict_points':weak[5:7]=True
                point_terms.append(point_loss(pp,gt,valid,weak_back=weak,
                    tangent=None if tangent is None else tangent[b,v,m],tangent_valid=None if tangent_valid is None else tangent_valid[b,v,m],
                    along_tolerance=along_tolerance,uncertain_weight=uncertain_weight,scale=max(h,w)))
                if mode=='weak_points_curve6' and curve_target is not None:
                    p=prediction['instance_surface_curve_xy'][b,v,q]*scale
                    curve_terms.append(curve_loss(p,curve_target[b,v,m],curve_valid[b,v,m],full_curve=True,scale=max(h,w)))
    zero=prediction['instance_keypoint_xy'].sum()*0
    points=torch.stack(point_terms).mean() if point_terms else zero
    curves=torch.stack(curve_terms).mean() if curve_terms else zero
    total=base+weights.keypoint_l1*points+curve_weight*curves
    return total,{**logs,'loss_surface_point_replacement':points,'loss_surface_curve':curves,'loss_surface_total':total},matches
