"""Small float64 pinhole-camera diagnostics; no annotation modification."""
import numpy as np


def projection_matrix(intrinsic,world_to_camera):
    k=np.asarray(intrinsic,dtype=np.float64);e=np.asarray(world_to_camera,dtype=np.float64)
    if k.shape!=(3,3) or e.shape not in [(3,4),(4,4)] or not np.isfinite(k).all() or not np.isfinite(e).all():raise ValueError('Expected finite K3x3 and world-to-camera E3x4/4x4')
    return k@e[:3]


def project_points(projection,xyz):
    p=np.asarray(projection,dtype=np.float64);x=np.asarray(xyz,dtype=np.float64)
    if p.shape!=(3,4) or x.shape[-1]!=3:raise ValueError('Invalid projection/XYZ shape')
    h=np.concatenate([x,np.ones((*x.shape[:-1],1))],-1)@p.T;z=h[...,2];good=np.isfinite(h).all(-1)&(np.abs(z)>1e-12);uv=np.full_like(h[...,:2],np.nan);np.divide(h[...,:2],z[...,None],out=uv,where=good[...,None]);return uv,z


def fundamental_from_projections(first,second):
    p1=np.asarray(first,dtype=np.float64);p2=np.asarray(second,dtype=np.float64)
    if p1.shape!=(3,4) or p2.shape!=(3,4) or not np.isfinite(p1).all() or not np.isfinite(p2).all():raise ValueError('Expected two finite projection matrices')
    if min(np.linalg.matrix_rank(p1),np.linalg.matrix_rank(p2))<3:raise ValueError('Rank deficient camera')
    p1=p1/np.linalg.norm(p1);p2=p2/np.linalg.norm(p2);center=np.linalg.svd(p1)[2][-1];ep=p2@center
    if np.linalg.norm(ep)<1e-10:raise ValueError('Coincident camera centers')
    x,y,z=ep;cross=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    f=cross@p2@np.linalg.pinv(p1);u,s,vh=np.linalg.svd(f);s[-1]=0;f=(u*s)@vh
    if np.linalg.norm(f)<1e-12:raise ValueError('Degenerate fundamental matrix')
    return f/np.linalg.norm(f)


def symmetric_epipolar_distance(f,uv1,uv2):
    a=np.asarray(uv1,dtype=np.float64);b=np.asarray(uv2,dtype=np.float64)
    if a.shape!=b.shape or a.shape[-1]!=2:raise ValueError('Corresponding pixel arrays required')
    ah=np.concatenate([a,np.ones((*a.shape[:-1],1))],-1);bh=np.concatenate([b,np.ones((*b.shape[:-1],1))],-1);l2=ah@np.asarray(f).T;l1=bh@np.asarray(f);value=np.abs((bh*l2).sum(-1));n1=np.linalg.norm(l1[...,:2],axis=-1);n2=np.linalg.norm(l2[...,:2],axis=-1)
    good=(n1>1e-12)&(n2>1e-12)&np.isfinite(value);out=np.full_like(value,np.nan);out[good]=value[good]*np.sqrt(.5*(1/n1[good]**2+1/n2[good]**2));return out


def weighted_dlt(projections,uv,weights=None):
    """Return XYZ or None on insufficient/degenerate observations.

    Caller controls cheirality and reprojection gates. A small own-reprojection
    error is not evidence of true anatomical accuracy. No visibility inference.
    """
    p=np.asarray(projections,dtype=np.float64);xy=np.asarray(uv,dtype=np.float64);w=np.ones(len(p)) if weights is None else np.asarray(weights,dtype=np.float64)
    if p.shape!=(len(p),3,4) or xy.shape!=(len(p),2) or w.shape!=(len(p),):raise ValueError('Invalid triangulation shapes')
    if np.any(w<0):raise ValueError('Weights must be nonnegative')
    valid=np.isfinite(p).all((1,2))&np.isfinite(xy).all(-1)&np.isfinite(w)&(w>0);p=p[valid];xy=xy[valid];w=w[valid]
    if len(p)<2:return None
    equations=[]
    for matrix,point,weight in zip(p,xy,w):
        for row,coord in zip([0,1],point):
            a=coord*matrix[2]-matrix[row];norm=np.linalg.norm(a[:3])
            if norm<1e-12:return None
            equations.append(a/norm*np.sqrt(weight))
    _,s,vh=np.linalg.svd(np.stack(equations),full_matrices=False)
    if s[-2]<1e-10*max(s[0],1e-12) or abs(vh[-1,3])<1e-10:return None
    xyz=vh[-1,:3]/vh[-1,3]
    return xyz if np.isfinite(xyz).all() else None
