"""Experimental prediction-only spatial candidate reranking, single animal.

No GT, labels, boxes or camera identity associations are inferred here. Caller
supplies exactly one prediction-selected query per usable physical camera.
"""
import numpy as np
from .multiview_camera_geometry import fundamental_from_projections, symmetric_epipolar_distance


def probability_candidates(probability, pixels, count=5, suppression_radius=2):
    p=np.asarray(probability,float);pixel=np.asarray(pixels,float)
    if p.ndim!=2 or pixel.shape!=(2,) or not np.isfinite(p).all() or (p<0).any() or p.sum()<=0 or (pixel<=0).any():
        raise ValueError('Expected nonnegative probability map and positive pixel dimensions')
    h,w=p.shape;work=p.copy();positions=[];masses=[]
    for _ in range(min(count,h*w)):
        y,x=np.unravel_index(np.argmax(work),work.shape)
        if work[y,x]<0:break
        ys=slice(max(0,y-1),min(h,y+2));xs=slice(max(0,x-1),min(w,x+2))
        local=p[ys,xs];gy,gx=np.meshgrid(np.arange(ys.start,ys.stop),np.arange(xs.start,xs.stop),indexing='ij')
        mass=float(local.sum());coord=np.stack([(gx+.5)/w,(gy+.5)/h],-1)
        mean=(local[...,None]*coord).sum((0,1))/mass if mass>0 else np.array([(x+.5)/w,(y+.5)/h])
        positions.append(mean*pixel);masses.append(mass)
        work[max(0,y-suppression_radius):y+suppression_radius+1,max(0,x-suppression_radius):x+suppression_radius+1]=-1
    return np.array(positions),np.array(masses)


def rerank_candidates(projections,candidates,masses,available,physical_views,target_index,
                      baseline,*,enabled=False,geometry_weight=1.,sigma_px=16.,minimum_support=2):
    """Return target local candidate or exact baseline on missing support.

    Per-support evidence is the best peak's relative mass times an epipolar
    Gaussian. Average the log evidence of the best two supporting cameras to
    reduce one-camera occlusion failures. This is not a visibility estimate.
    Geometry weight zero is a matched availability, unary-only control.
    """
    if not enabled:return dict(xy=baseline,active=False)
    p=np.asarray(projections,float);xy=np.asarray(candidates,float);weight=np.asarray(masses,float)
    valid=np.asarray(available,bool);views=np.asarray(physical_views);n=len(p)
    if p.shape!=(n,3,4) or xy.ndim!=3 or xy.shape[0]!=n or xy.shape[-1]!=2 or weight.shape!=xy.shape[:2] or valid.shape!=(n,) or views.shape!=(n,):raise ValueError('Inconsistent candidate arrays')
    if not 0<=target_index<n or geometry_weight<0 or sigma_px<=0 or minimum_support<1:raise ValueError('Invalid reranking settings')
    target=target_index
    if not valid[target] or not np.isfinite(xy[target]).all() or not np.isfinite(weight[target]).all() or weight[target].max()<=0:return dict(xy=baseline,active=False)
    seen={views[target]};evidence=[];supports=[]
    for s in range(n):
        if views[s] in seen or not valid[s] or not np.isfinite(xy[s]).all() or not np.isfinite(weight[s]).all() or weight[s].max()<=0:continue
        seen.add(views[s])
        try:f=fundamental_from_projections(p[target],p[s])
        except ValueError:continue
        a=np.broadcast_to(xy[target,:,None],(len(xy[target]),len(xy[s]),2));b=np.broadcast_to(xy[s,None],a.shape)
        distance=symmetric_epipolar_distance(f,a,b)
        if not np.isfinite(distance).all():continue
        likelihood=np.exp(-.5*(distance/sigma_px)**2)*(weight[s]/weight[s].max())[None]
        evidence.append(likelihood.max(-1));supports.append(s)
    if len(evidence)<minimum_support:return dict(xy=baseline,active=False)
    e=np.stack(evidence);best=np.sort(e,axis=0)[-minimum_support:]
    geo=np.log(best.clip(1e-8)).mean(0);unary=np.log((weight[target]/weight[target].max()).clip(1e-8))
    scores=unary+geometry_weight*geo;choice=int(np.argmax(scores))
    return dict(xy=xy[target,choice],active=True,choice=choice,scores=scores,unary=unary,geometry=geo,support_indices=supports)
