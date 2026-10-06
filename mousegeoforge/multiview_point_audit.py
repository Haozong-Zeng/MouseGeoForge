"""Read-only triangulation diagnostics; no pseudo-ground-truth acceptance."""
import numpy as np
from scipy.optimize import least_squares


def project_point(xyz,K,E):
    camera=np.asarray(E)[:3,:3]@np.asarray(xyz)+np.asarray(E)[:3,3]
    pixel=np.asarray(K)@camera
    return pixel[:2]/pixel[2],float(camera[2])


def triangulate_observations(uv,K,E,*,minimum_angle_deg=5.,huber_px=8.):
    uv=np.asarray(uv,dtype=np.float64);K=np.asarray(K,dtype=np.float64);E=np.asarray(E,dtype=np.float64)
    if uv.ndim!=2 or uv.shape[1]!=2 or len(uv)<2 or K.shape!=(len(uv),3,3) or E.shape!=(len(uv),4,4):return dict(valid=False,reason='insufficient_or_bad_shapes')
    if not all(np.isfinite(x).all() for x in (uv,K,E)):return dict(valid=False,reason='nonfinite')
    rows=[];rays=[]
    for p,k,e in zip(uv,K,E):
        ray=np.linalg.solve(k,np.r_[p,1.]);ray=ray/ray[2];world=e[:3,:3].T@ray;rays.append(world/np.linalg.norm(world))
        rows.extend([ray[0]*e[2]-e[0],ray[1]*e[2]-e[1]])
    rays=np.asarray(rays);angle=np.degrees(np.arccos(np.clip(np.abs(rays@rays.T),0,1))).max()
    if angle<minimum_angle_deg:return dict(valid=False,reason='near_collinear_rays',max_ray_angle_deg=float(angle))
    _,_,vt=np.linalg.svd(np.asarray(rows));x=vt[-1]
    if abs(x[3])<1e-10:return dict(valid=False,reason='point_at_infinity')
    initial=x[:3]/x[3]
    def residual(xyz):return np.concatenate([project_point(xyz,k,e)[0]-p for p,k,e in zip(uv,K,E)])
    fit=least_squares(residual,initial,loss='huber',f_scale=huber_px,max_nfev=100)
    depths=np.asarray([project_point(fit.x,k,e)[1] for k,e in zip(K,E)])
    if not fit.success or not np.isfinite(fit.x).all() or (depths<=0).any():return dict(valid=False,reason='solver_or_cheirality',converged=bool(fit.success))
    return dict(valid=True,xyz_mm=fit.x.tolist(),fit_source_reprojection_px=np.linalg.norm(residual(fit.x).reshape(-1,2),axis=-1).tolist(),max_ray_angle_deg=float(angle))
