"""Experimental leave-one-camera-out triangulation, no annotation inputs."""
import itertools
import numpy as np
from .multiview_camera_geometry import weighted_dlt, project_points


def leave_one_out_projection(projections, uv, weights, available, physical_views,
                             target_index, *, robust=True, threshold_px=16.,
                             min_angle_degrees=2.):
    """Project support-only XYZ into target, or abstain.

    Robust mode requires >=3 inliers, so two agreeing cameras alone cannot
    override the target. Physical duplicates and the target camera are excluded.
    This assumes a single-instance cohort; it does NOT solve identity matching.
    """
    p=np.asarray(projections,float);xy=np.asarray(uv,float);w=np.asarray(weights,float)
    available=np.asarray(available,bool);views=np.asarray(physical_views)
    n=len(p)
    if p.shape!=(n,3,4) or xy.shape!=(n,2) or w.shape!=(n,) or available.shape!=(n,) or views.shape!=(n,):
        raise ValueError('Inconsistent camera/point arrays')
    if not 0<=target_index<n or threshold_px<=0:raise ValueError('Invalid target/threshold')
    ids=[];seen={views[target_index]}
    for i in range(n):
        if views[i] in seen:continue
        if available[i] and w[i]>0 and np.isfinite(w[i]) and np.isfinite(xy[i]).all() and np.isfinite(p[i]).all():
            ids.append(i);seen.add(views[i])
    minimum=3 if robust else 2
    if len(ids)<minimum:return None
    ids=np.array(ids);mat=p[ids];points=xy[ids];weight=w[ids]
    weight=weight/weight.max()

    def check(xyz):
        if xyz is None:return None
        errors=[];depths=[]
        for camera,point in zip(mat,points):
            projected,z=project_points(camera,xyz)
            errors.append(np.linalg.norm(projected-point));depths.append(z)
        errors=np.array(errors);positive=np.array(depths)>1e-8
        inlier=(errors<=threshold_px)&positive&np.isfinite(errors)
        return errors,inlier

    if robust:
        candidates=[]
        for a,b in itertools.combinations(range(len(ids)),2):
            xyz=weighted_dlt(mat[[a,b]],points[[a,b]],weight[[a,b]])
            checked=check(xyz)
            if checked is None:continue
            errors,good=checked
            if good.sum()>=minimum:
                candidates.append((int(good.sum()),float(weight[good].sum()),
                                   -float(np.average(errors[good],weights=weight[good])),good))
        if not candidates:return None
        # Fixed predicted-data rule; no target position or target labels used.
        good=max(candidates,key=lambda c:c[:3])[3]
        xyz=weighted_dlt(mat[good],points[good],weight[good])
        checked=check(xyz)
        if checked is None:return None
        errors,refit_good=checked
        good &= refit_good
        if good.sum()<minimum:return None
    else:
        xyz=weighted_dlt(mat,points,weight);checked=check(xyz)
        if checked is None:return None
        errors,_=checked;good=np.ones(len(ids),bool)
        if not all(project_points(camera,xyz)[1]>1e-8 for camera in mat):return None
    centers=[]
    for camera in mat[good]:
        center=np.linalg.svd(camera)[2][-1]
        if abs(center[-1])<1e-12:return None
        centers.append(center[:3]/center[3])
    rays=xyz-np.array(centers);rays/=np.maximum(np.linalg.norm(rays,axis=-1,keepdims=True),1e-12)
    angles=np.degrees(np.arccos(np.clip(rays@rays.T,-1,1)))
    # Acute ray intersection angle; opposing rays are also ill conditioned.
    angles=np.minimum(angles,180-angles)
    if np.max(angles)<min_angle_degrees:return None
    target,z=project_points(p[target_index],xyz)
    if z<=1e-8 or not np.isfinite(target).all():return None
    return dict(uv=target,xyz=xyz,support_indices=ids[good],
                support_error_px=errors[good],max_angle_degrees=float(angles.max()))
