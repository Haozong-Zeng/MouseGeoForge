"""Prediction-only cross-camera evidence, not a matcher or identity guarantee.

Distances use corresponding nose/four-paw predictions in actual input pixels.
Semantic and local-memory cosine evidence are separate, uncalibrated channels.
No GT, animal counts, labels, masks, forced assignments or fused poses accepted.
"""
import numpy as np
import torch
from .multiview_camera_geometry import fundamental_from_projections,symmetric_epipolar_distance
KP=[0,14,15,16,17]


def array(value):
    return value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else np.asarray(value)


def cosine(a,b):
    denom=np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)
    good=np.isfinite(a).all(-1)&np.isfinite(b).all(-1)&(denom>1e-12)
    return float(np.mean(np.sum(a[good]*b[good],-1)/denom[good])) if good.sum()>=3 else np.nan


def pairwise_affinity_evidence(predictions,projections,physical_views,pixels):
    p={k:array(predictions[k]).astype(float) for k in ['xy','boxes','objectness_logits','tokens','candidate_memory']}
    v,q,_,_=p['xy'].shape
    if p['xy'].shape!=(v,q,23,2) or p['tokens'].shape[:3]!=(v,q,5) or p['candidate_memory'].shape[:4]!=(v,q,5,6):raise ValueError('Invalid prediction shapes')
    cameras=array(projections).astype(float);views=np.asarray(physical_views);pixel=np.asarray(pixels,float)
    if pixel.shape==(2,):pixel=np.tile(pixel,(v,1))
    if cameras.shape!=(v,3,4) or views.shape!=(v,) or pixel.shape!=(v,2) or not np.isfinite(pixel).all() or (pixel<=0).any():raise ValueError('Invalid camera/pixel inputs')
    selected=[]
    for view in sorted(set(views.tolist())):
        ids=np.flatnonzero(views==view);first=int(ids[0])
        for j in ids[1:]:
            if not np.allclose(cameras[first],cameras[j],equal_nan=True,atol=1e-9,rtol=0) or not np.array_equal(pixel[first],pixel[j]) or any(not np.allclose(x[first],x[j],equal_nan=True,atol=1e-7,rtol=0) for x in p.values()):
                raise ValueError('Conflicting duplicate physical camera')
        selected.append(first)
    nodes=[]
    for vi in selected:
        logits=p['objectness_logits'][vi]
        for qi in np.flatnonzero(np.isfinite(logits)&(logits>=0)):
            nodes.append(dict(view_index=vi,view=int(views[vi]),query=int(qi),score=float(1/(1+np.exp(-np.clip(logits[qi],-80,80))))))
    n=len(nodes);values={k:np.full((n,n),np.nan) for k in ['epipolar_mean_px','epipolar_median_px','semantic_cosine','local_cosine']}
    count=np.zeros((n,n),int)
    for a,na in enumerate(nodes):
        vi,qi=na['view_index'],na['query']
        for b in range(a+1,n):
            nb=nodes[b];vj,qj=nb['view_index'],nb['query']
            if na['view']==nb['view']:continue
            values['semantic_cosine'][a,b]=values['semantic_cosine'][b,a]=cosine(p['tokens'][vi,qi],p['tokens'][vj,qj])
            values['local_cosine'][a,b]=values['local_cosine'][b,a]=cosine(p['candidate_memory'][vi,qi,:,0],p['candidate_memory'][vj,qj,:,0])
            if not np.isfinite(cameras[[vi,vj]]).all():continue
            try:f=fundamental_from_projections(cameras[vi],cameras[vj])
            except (ValueError,np.linalg.LinAlgError):continue
            xy1=p['xy'][vi,qi,KP]*pixel[vi];xy2=p['xy'][vj,qj,KP]*pixel[vj]
            distance=symmetric_epipolar_distance(f,xy1,xy2);good=np.isfinite(distance)
            count[a,b]=count[b,a]=int(good.sum())
            if good.sum()>=3:
                values['epipolar_mean_px'][a,b]=values['epipolar_mean_px'][b,a]=float(distance[good].mean())
                values['epipolar_median_px'][a,b]=values['epipolar_median_px'][b,a]=float(np.median(distance[good]))
    return dict(nodes=nodes,evidence=values,finite_point_count=count)
