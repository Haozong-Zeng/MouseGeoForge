"""Frozen-calibration multiview audit of manual labels, NOT accepted 3D GT.

Report unfiltered all-view least-squares and leave-one-view-out errors, not just
RANSAC inlier errors. Holdout gets the same fixed algorithm, never model tuning.
"""
import itertools,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'dataset/PyMouseStudio/annotation_returns/20260930_full_v2'
OUT=ROOT/'output/pymousestudio_manual_multiview_20260930'
ACTIVE={0,5,6,7,13,14,15,16,17,22}
THRESHOLD=15.0 # Existing pilot threshold; frozen before either split is evaluated.

def projection(x,K,E):
    cam=np.einsum('vij,j->vi',E[:,:3,:3],x)+E[:,:3,3]
    h=np.einsum('vij,vj->vi',K,cam)
    den=np.where(np.abs(h[:,2:])<1e-10,1e-10,h[:,2:])
    return h[:,:2]/den,cam[:,2]

def solve(xy,K,E,refine=True):
    xy1=np.c_[xy,np.ones(len(xy))];norm=np.einsum('vij,vj->vi',np.linalg.inv(K),xy1)
    rows=[]
    for q,e in zip(norm,E):
        for r in [q[0]*e[2]-q[2]*e[0],q[1]*e[2]-q[2]*e[1]]:rows.append(r/max(np.linalg.norm(r[:3]),1e-12))
    _,s,vh=np.linalg.svd(rows,full_matrices=False)
    if abs(vh[-1,3])<1e-10:return None
    x=vh[-1,:3]/vh[-1,3]
    if refine:
        def residual(p):return (projection(p,K,E)[0]-xy).ravel()
        opt=least_squares(residual,x,method='lm',max_nfev=50)
        if np.isfinite(opt.x).all():x=opt.x
    return x if np.isfinite(x).all() else None

def robust(xy,K,E):
    best=None
    for pair in itertools.combinations(range(len(xy)),2):
        ids=list(pair);x=solve(xy[ids],K[ids],E[ids],False)
        if x is None:continue
        p,z=projection(x,K,E);error=np.linalg.norm(p-xy,axis=-1)
        inside=(error<THRESHOLD)&(z>0)
        score=(int(inside.sum()),-float(np.minimum(error,THRESHOLD).sum()))
        if best is None or score>best[0]:best=(score,inside,x)
    if best is None or best[1].sum()<2:return None,None
    ids=np.flatnonzero(best[1]);x=solve(xy[ids],K[ids],E[ids])
    if x is None:return None,None
    p,z=projection(x,K,E);mask=(np.linalg.norm(p-xy,axis=-1)<THRESHOLD)&(z>0)
    return x,mask

def angle(x,E):
    centers=-np.einsum('vji,vj->vi',E[:,:3,:3],E[:,:3,3]);r=x-centers
    r/=np.maximum(np.linalg.norm(r,axis=1,keepdims=True),1e-12)
    return max([float(np.degrees(np.arccos(np.clip(np.dot(r[i],r[j]),-1,1)))) for i,j in itertools.combinations(range(len(r)),2)],default=0.)

def fundamental(k1,e1,k2,e2):
    r=e2[:3,:3]@e1[:3,:3].T;t=e2[:3,3]-r@e1[:3,3]
    tx=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
    return np.linalg.inv(k2).T@tx@r@np.linalg.inv(k1)

def epi(x,y,f):
    a=np.r_[x,1];b=np.r_[y,1];l2=f@a;l1=f.T@b;v=abs(b@f@a)
    return float(np.sqrt(.5*((v/max(np.linalg.norm(l1[:2]),1e-12))**2+(v/max(np.linalg.norm(l2[:2]),1e-12))**2)))

def stats(values):
    a=np.asarray(values,float);a=a[np.isfinite(a)]
    if not len(a):return dict(n=0,median=None,p90=None,p95=None,under5=None,under10=None,under15=None)
    return dict(n=len(a),median=float(np.median(a)),p90=float(np.percentile(a,90)),p95=float(np.percentile(a,95)),
                under5=float(np.mean(a<=5)),under10=float(np.mean(a<=10)),under15=float(np.mean(a<=15)))

def self_test():
    K=np.repeat(np.array([[800.,0,540],[0,800,600],[0,0,1]])[None],4,0)
    E=np.repeat(np.eye(4)[None],4,0);E[:,0,3]=[-50,0,50,0];E[:,1,3]=[0,-50,0,50]
    x=np.array([4.,12.,250.]);xy,_=projection(x,K,E)
    assert np.linalg.norm(solve(xy,K,E)-x)<1e-7
    for i,j in itertools.combinations(range(4),2):assert epi(xy[i],xy[j],fundamental(K[i],E[i],K[j],E[j]))<1e-7
    corrupt=xy.copy();corrupt[3]+=[200,-100];rx,mask=robust(corrupt,K,E)
    assert np.linalg.norm(rx-x)<1e-6 and mask.tolist()==[True,True,True,False]
    # With two views one can fit a wrong correspondence surprisingly well; never call it verified 3D.
    print('SYNTHETIC_GEOMETRY_TESTS_PASSED',flush=True)

def main():
    self_test();OUT.mkdir(parents=True,exist_ok=False)
    ingest=json.loads((BASE/'ingestion_manifest.json').read_text());rows=[];observations=[];pairwise=[];summary_sessions=[]
    for session in ingest['sessions']:
        cal=json.loads((BASE/'calibration'/session/'rectified_calibration.json').read_text())
        sg=[g for g in ingest['frame_groups'] if g['session']==session]
        for g in sg:
            split=g['split'];frame=g['frame'];cams=g['cameras']
            K=np.array([cal[c]['K'] for c in cams]);E=np.array([cal[c]['world_to_camera_mm'] for c in cams])
            for c,k,e in zip(cams,K,E):
                assert np.allclose(cal[c]['distortion'],0) and cal[c]['size_wh']==[1080,1200]
                assert np.allclose(e[3],[0,0,0,1]) and np.allclose(e[:3,:3]@e[:3,:3].T,np.eye(3),atol=1e-6)
                assert abs(np.linalg.det(e[:3,:3])-1)<1e-6
            with np.load(BASE/g['normalized_file']) as a:uv=a['uv'][:,0];names=a['semantic_names'].tolist()
            all_xyz=np.full((23,3),np.nan);cons_xyz=all_xyz.copy();all_err=np.full((len(cams),23),np.nan)
            cons_err=all_err.copy();loo_err=all_err.copy();loo_projection=np.full((len(cams),23,2),np.nan)
            inliers=np.zeros((len(cams),23),bool);quality=np.zeros(23,bool);angles=np.full(23,np.nan)
            for k in range(23):
                ids=np.flatnonzero(np.isfinite(uv[:,k]).all(-1));n=len(ids)
                base=dict(split=split,session=session,frame=frame,id=k+1,name=names[k],active=k in ACTIVE,n_views=n,cameras=[cams[i] for i in ids])
                for i,j in itertools.combinations(ids,2):
                    pairwise.append(dict(**{q:base[q] for q in ['split','session','frame','id','name','active']},
                        camera1=cams[i],camera2=cams[j],symmetric_epipolar_px=epi(uv[i,k],uv[j,k],fundamental(K[i],E[i],K[j],E[j]))))
                if n<2:rows.append(dict(**base,status='insufficient_views'));continue
                x=solve(uv[ids,k],K[ids],E[ids]);rx,mask=robust(uv[ids,k],K[ids],E[ids])
                if x is None:rows.append(dict(**base,status='degenerate'));continue
                p,z=projection(x,K[ids],E[ids]);err=np.linalg.norm(p-uv[ids,k],axis=-1);all_xyz[k]=x;all_err[ids,k]=err
                angles[k]=angle(x,E[ids]);behind=int((z<=0).sum())
                if rx is not None:
                    rp,rz=projection(rx,K[ids],E[ids]);re=np.linalg.norm(rp-uv[ids,k],axis=-1)
                    cons_xyz[k]=rx;cons_err[ids,k]=re;inliers[ids,k]=mask
                    # Diagnostic consistency flag, NOT a training-acceptance flag.
                    quality[k]=bool(mask.sum()>=3 and angle(rx,E[ids[mask]])>=2 and (rz[mask]>0).all())
                for local,v in enumerate(ids):
                    looe=None
                    if n>=3:
                        other=ids[ids!=v];lx=solve(uv[other,k],K[other],E[other])
                        if lx is not None:
                            lp,lz=projection(lx,K[v:v+1],E[v:v+1])
                            if lz[0]>0:
                                looe=float(np.linalg.norm(lp[0]-uv[v,k]));loo_err[v,k]=looe;loo_projection[v,k]=lp[0]
                    observations.append(dict(**{q:base[q] for q in ['split','session','frame','id','name','active','n_views']},camera=cams[v],
                        xy=uv[v,k].tolist(),all_ls_reprojection_px=float(err[local]),loo_reprojection_px=looe,
                        consensus_reprojection_px=float(cons_err[v,k]) if np.isfinite(cons_err[v,k]) else None,
                        consensus_inlier=bool(inliers[v,k]),all_ls_depth_mm=float(z[local])))
                rows.append(dict(**base,status='triangulated_diagnostic',all_ls_median_px=float(np.median(err)),all_ls_max_px=float(err.max()),
                    behind_camera_views=behind,max_ray_angle_deg=float(angles[k]),consensus_inliers=int(inliers[:,k].sum()),
                    diagnostic_consensus3=bool(quality[k]),consensus_inlier_error=stats(cons_err[inliers[:,k],k]),
                    consensus_all_observation_error=stats(cons_err[ids,k]),loo_error=stats(loo_err[ids,k])))
            target=BASE/'geometry_diagnostics_NOT_GT'/split/session/f'frame_{frame:06d}.npz';target.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(target,xyz_all_ls_mm=all_xyz,xyz_consensus_mm=cons_xyz,uv_observed=uv,
                error_all_ls_px=all_err,error_consensus_px=cons_err,error_leave_one_view_out_px=loo_err,uv_leave_one_view_out=loo_projection,
                consensus_inlier=inliers,diagnostic_consensus3=quality,max_ray_angle_deg=angles,
                names=np.asarray(names),camera_names=np.asarray(cams),K=K,world_to_camera_mm=E,
                source_frame=np.asarray(frame),train_allowed=np.asarray(False),not_ground_truth=np.asarray(True))
        print('GEOMETRY_AUDITED',session,len(sg),'frames',flush=True)
    def summarize(obs,pts):
        eligible=[p for p in pts if p['n_views']>=3]
        o3=[r for r in obs if r['n_views']>=3]
        return dict(points_total=len(pts),points_with2plus=sum(p['n_views']>=2 for p in pts),points_with3plus=len(eligible),
            consensus3_points=sum(p.get('diagnostic_consensus3',False) for p in pts),
            all_ls_observations_3plus=stats([r['all_ls_reprojection_px'] for r in o3]),
            loo_observations=stats([r['loo_reprojection_px'] for r in o3 if r['loo_reprojection_px'] is not None]),
            consensus_inliers=stats([r['consensus_reprojection_px'] for r in o3 if r['consensus_inlier']]),
            consensus_all_observations=stats([r['consensus_reprojection_px'] for r in o3 if r['consensus_reprojection_px'] is not None]),
            consensus_inlier_fraction=float(np.mean([r['consensus_inlier'] for r in o3])) if o3 else None)
    session_stats={s:summarize([o for o in observations if o['session']==s],[p for p in rows if p['session']==s]) for s in ingest['sessions']}
    perkey={str(k):summarize([o for o in observations if o['id']==k],[p for p in rows if p['id']==k]) for k in range(1,24)}
    percam={c:dict(all_ls=stats([o['all_ls_reprojection_px'] for o in observations if o['camera']==c and o['n_views']>=3]),
        loo=stats([o['loo_reprojection_px'] for o in observations if o['camera']==c and o['loo_reprojection_px'] is not None])) for c in 'ABCDEFG'}
    pair_stats={f'{a}-{b}':stats([p['symmetric_epipolar_px'] for p in pairwise if p['camera1']==a and p['camera2']==b and p['active']]) for a,b in itertools.combinations('ABCDEFG',2)}
    result=dict(method='Normalized DLT plus nonlinear pixel LS; exhaustive-pair consensus 15px; leave-one-view-out LS; symmetric epipolar distance.',
        calibration_unchanged=True,distortion_applied_again=False,world_units='mm from original calibration',
        source_synchronization='Same source indices, hardware synchronization/time offsets unverified',
        tail_edge_filter=False,train_allowed=False,consensus_threshold_px=THRESHOLD,minimum_consensus_views=3,minimum_ray_angle_deg=2,
        all=summarize(observations,rows),active=summarize([o for o in observations if o['active']],[p for p in rows if p['active']]),
        splits={s:summarize([o for o in observations if o['split']==s],[p for p in rows if p['split']==s]) for s in ingest['split_counts']},
        sessions=session_stats,keypoints=perkey,cameras=percam,active_epipolar_pairs=pair_stats,
        failures_note='Small consensus-inlier residual does not verify rejected observations or anatomical truth. Leave-one-view-out uses independent withheld-view evaluation.',
        geometry_tests='passed')
    for name,data in [('summary.json',result),('point_metrics.json',rows),('observation_metrics.json',observations),('pairwise_epipolar_metrics.json',pairwise)]:
        (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps(dict(all=result['all'],active=result['active'],splits=result['splits']),indent=2),flush=True)

if __name__=='__main__':main()
