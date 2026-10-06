"""CPU-only diagnostics of saved predictions; never modifies inference gates."""
import torch
from mousegeoforge.pilot_metrics import box_iou
from vggt_omega.utils.robust_pose_geometry import triangulate_robust

BODY = tuple(list(range(7))+list(range(14,23)))


def matched_views(saved):
    p,t=saved['predictions'],saved['targets'];h,w=saved['record']['image_hw']
    pixel=torch.tensor([w,h]);rows=[]
    for v,(qs,ms) in enumerate(saved['oracle_assignment']):
        for q,m in zip(qs,ms):
            score=float(p['instance_objectness_logits'][0,v,q].sigmoid())
            iou=float(box_iou(t['boxes_cxcywh'][0,v,m:m+1],p['instance_boxes_cxcywh'][0,v,q:q+1])[0,0])
            xy=p['instance_keypoint_xy'][0,v,q]*pixel;gt=t['keypoint_xy'][0,v,m]*pixel
            diag=(t['boxes_cxcywh'][0,v,m,2:]*pixel).norm().clamp_min(1.)
            errors=(xy-gt).norm(dim=-1)/diag
            rows.append(dict(view=v,camera=saved['record']['views'][v],role=saved['record']['physical_roles'][v],
                query=q,instance=m,score=score,iou=iou,found=score>=.5 and iou>=.5,
                errors=errors,valid=t['keypoint_valid'][0,v,m],
                visible=t['keypoint_visible'][0,v,m],
                visibility_valid=t['keypoint_visibility_valid'][0,v,m]))
    return rows


def symmetric_epipolar_distance(xy, projections):
    """Mean point-to-epipolar-line distance in pixels; requires exactly two views."""
    if xy.shape!=(2,2) or projections.shape!=(2,3,4):raise ValueError('Expected exactly two views')
    p=projections.double();u=torch.cat((xy.double(),torch.ones(2,1)),dim=-1)
    _,_,vh=torch.linalg.svd(p[0]);center=vh[-1];epipole=p[1]@center
    x,y,z=epipole
    skew=torch.stack((torch.stack((x*0,-z,y)),torch.stack((z,x*0,-x)),torch.stack((-y,x,x*0))))
    fundamental=skew@p[1]@torch.linalg.pinv(p[0])
    line2=fundamental@u[0];line1=fundamental.T@u[1]
    denominator1=line1[:2].norm();denominator2=line2[:2].norm()
    if min(denominator1,denominator2)<1e-12:raise ValueError('Degenerate epipolar lines')
    residual=(u[1]@line2).abs()
    return float(.5*residual*(1/denominator1+1/denominator2))


def geometry_points(saved):
    p,t=saved['predictions'],saved['targets'];r=saved['record'];h,w=r['image_hw']
    projection=t['intrinsics'][0]@t['extrinsics'][0,:,:3]
    qs=[q[m.index(0)] for q,m in saved['oracle_assignment']]
    xy=torch.stack([p['instance_keypoint_xy'][0,v,q] for v,q in enumerate(qs)])*torch.tensor([w,h])
    scores=torch.stack([p['instance_objectness_logits'][0,v,q].sigmoid() for v,q in enumerate(qs)])
    gt=t['keypoint_xy'][0,:,0]*torch.tensor([w,h]);rows=[]
    for k in BODY:
        label=t['keypoint_valid'][0,:,0,k];valid=label & (scores>=.5)
        actual=triangulate_robust(xy[:,k],projection,valid,huber_delta_px=3.)
        no_objectness=triangulate_robust(xy[:,k],projection,label,huber_delta_px=3.)
        oracle=triangulate_robust(gt[:,k],projection,label,huber_delta_px=3.)
        error=float((actual.xyz-t['keypoint_world_xyz'][0,0,k]).norm())*r['normalization_scale'] if actual.supported else None
        row=dict(keypoint=k,valid_views=int(label.sum()),score_gated_views=int(valid.sum()),scores=scores.tolist(),
            supported=actual.supported,reason=actual.reason,error_native=error,
            inliers=actual.inliers.tolist(),reprojection_px=[float(x) if torch.isfinite(x) else None for x in actual.reprojection_px],
            without_objectness_supported=no_objectness.supported,without_objectness_reason=no_objectness.reason,
            gt_supported=oracle.supported,per_view_error_px=(xy[:,k]-gt[:,k]).norm(dim=-1).tolist())
        if len(xy)==2:
            row['epipolar_distance_px']=symmetric_epipolar_distance(xy[:,k],projection)
            row['gt_epipolar_distance_px']=symmetric_epipolar_distance(gt[:,k],projection)
            # Label-assisted counterfactuals localize which view's prediction is
            # inconsistent. They are diagnostics, never deployed GT correction.
            rescues=[]
            for v in range(2):
                fixed=xy[:,k].clone();fixed[v]=gt[v,k]
                result=triangulate_robust(fixed,projection,label,huber_delta_px=3.)
                rescues.append(result.supported)
            row['gt_replace_one_view_supported']=rescues
        rows.append(row)
    return rows
