"""Prediction-only local/epipolar features for six-candidate ranking."""
import numpy as np
import torch
from .multiview_camera_geometry import fundamental_from_projections,symmetric_epipolar_distance
KP=[0,14,15,16,17]
SCALAR_NAMES=['log_relative_mass_div20','local_mass','is_original','dx','dy','distance_from_original','entropy','normalized_std','query_score']
GEOMETRY_NAMES=['max_log_evidence_div20','top2_mean_log_evidence_div20','all_mean_log_evidence_div20','std_log_evidence_div20','support_count_div8','fraction_evidence_gt_half']


def array(value):
    return value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else np.asarray(value)


def build_selector_inputs(predictions,projections,physical_views,pixels,view_valid=None):
    """Inputs have no target/annotation parameters. Known single-animal pilot.

    A view contributes only its unique score>=.5 prediction, not a GT-selected
    query. Both heads share support eligibility, preventing coverage confounds.
    Duplicate physical cameras count once; conflicting duplicates are ignored.
    """
    p=predictions;xy=array(p['candidate_xy']).astype(float);mass=array(p['candidate_mass']).astype(float);base=array(p['xy'])[:,:,KP].astype(float)
    logits=array(p['objectness_logits']).astype(float);projection=array(projections).astype(float);views=np.asarray(physical_views);pixel=np.asarray(pixels,float)
    v,q,k,c,_=xy.shape
    if (q,k,c)!=(6,5,6) or mass.shape!=(v,q,k,c) or projection.shape!=(v,3,4) or views.shape!=(v,) or base.shape!=(v,6,5,2):raise ValueError('Invalid selector feature shapes')
    if pixel.shape==(2,):pixel=np.tile(pixel,(v,1))
    if pixel.shape!=(v,2) or not np.isfinite(pixel).all() or (pixel<=0).any():raise ValueError('Invalid pixels')
    score=1/(1+np.exp(-logits.clip(-80,80)));available=(score>=.5).sum(-1)==1
    available&=np.isfinite(logits).all(-1)&np.isfinite(projection).all((1,2))
    if view_valid is not None:available&=np.asarray(view_valid,bool)
    queries=score.argmax(-1)
    if 'probability_entropy' in p:
        entropy=array(p['probability_entropy']);variance=array(p['probability_variance'])
    else:
        probability=array(p['probability']).astype(float);h,w=probability.shape[-2:];y,x=np.meshgrid((np.arange(h)+.5)/h,(np.arange(w)+.5)/w,indexing='ij');grid=np.stack([x,y],-1)
        entropy=-(probability*np.log(probability.clip(1e-12))).sum((-1,-2))/np.log(h*w)
        variance=(probability[...,None]*(grid[None,None,None]-base[:,:,:,None,None])**2).sum((-3,-2))
    relative=np.maximum(mass,0)/np.maximum(mass.max(-1,keepdims=True),1e-12);delta=xy-base[:,:,:,None];isbase=np.zeros_like(mass);isbase[:,:,:,0]=1
    broadcast=lambda a:np.broadcast_to(a[...,None],mass.shape)
    scalar=np.stack([np.log(relative.clip(1e-8))/20,mass,isbase,delta[...,0],delta[...,1],np.linalg.norm(delta,axis=-1),broadcast(entropy),broadcast(np.sqrt(np.maximum(variance.sum(-1),0))),np.broadcast_to(score[:,:,None,None],mass.shape)],-1)
    # Do not silently choose one of conflicting physical-camera copies.
    canonical={}
    for physical in sorted(set(views.tolist())):
        indices=np.flatnonzero(views==physical);good=[i for i in indices if available[i]]
        if not good:continue
        first=good[0]
        if any(not (np.allclose(projection[first],projection[j],atol=1e-9,rtol=0) and np.allclose(xy[first],xy[j],atol=1e-7,rtol=0,equal_nan=True) and np.allclose(mass[first],mass[j],atol=1e-7,rtol=0,equal_nan=True) and np.allclose(score[first],score[j],atol=1e-7,rtol=0)) for j in good[1:]):
            available[indices]=False;continue
        canonical[physical]=first
    geometry=np.zeros((*mass.shape,6),np.float32);eligible=np.zeros((v,6,5),bool)
    for target in range(v):
        if not available[target]:continue
        tq=queries[target];evidence=[[] for _ in range(k)]
        for physical,s in canonical.items():
            if physical==views[target]:continue
            try:f=fundamental_from_projections(projection[target],projection[s])
            except ValueError:continue
            sq=queries[s]
            for point in range(k):
                a=xy[target,tq,point]*pixel[target];b=xy[s,sq,point,1:]*pixel[s];weights=mass[s,sq,point,1:]
                if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(weights).all() or weights.max()<=0:continue
                aa=np.broadcast_to(a[:,None],(6,5,2));bb=np.broadcast_to(b[None],(6,5,2));distance=symmetric_epipolar_distance(f,aa,bb)
                if not np.isfinite(distance).all():continue
                evidence[point].append((np.exp(-.5*(distance/16)**2)*(weights/weights.max())[None]).max(-1))
        for point,items in enumerate(evidence):
            if len(items)<2:continue
            e=np.stack(items);logs=np.log(e.clip(1e-8));features=np.stack([logs.max(0)/20,np.sort(logs,axis=0)[-2:].mean(0)/20,logs.mean(0)/20,logs.std(0)/20,np.full(6,len(items)/8),(e>.5).mean(0)],-1)
            geometry[target,tq,point]=features;eligible[target,tq,point]=True
    return dict(semantic=torch.as_tensor(array(p['tokens']),dtype=torch.float32),local=torch.as_tensor(array(p['candidate_memory']),dtype=torch.float32),
        scalars=torch.tensor(scalar,dtype=torch.float32),geometry=torch.from_numpy(geometry),candidate_xy=torch.tensor(xy,dtype=torch.float32),baseline_xy=torch.tensor(base,dtype=torch.float32),eligible=torch.from_numpy(eligible))
