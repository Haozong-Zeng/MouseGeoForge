"""Diagnostic custom fixed-camera consensus, not new manual GT or Anipose/EKS."""
import argparse
import datetime as dt
import itertools
import json
import os
import time
from pathlib import Path
import numpy as np
from weighted_framewise_cycle_20261002 import Geometry, ROOT, OUT, BASE, CONFIGS
from audit_pymousestudio_anipose_20261001 import group_from_cal, safe_project
from audit_teacher_dlc_predictions_20261001 import stats
from run_teacher_cycle_dlc_20261001 import write_state

SETTINGS = dict(confidence_min=.6, spread_max_px=10, residual_max_px=15,
                minimum_support=3, strict_majority_of_eligible=True,
                ray_angle_min_deg=3, variance_floor_px=4,
                equal_support_ambiguity_mm=5, temporal=False, train_allowed=False)

def paw_collisions(uv,ids):
    """Per-camera only; both identities rejected when predicted paws nearly coincide."""
    js=[list(ids).index(k) for k in [15,16,17,18]]
    paws=uv[...,js,:]
    d=np.linalg.norm(paws[..., :,None,:]-paws[...,None,:,:],axis=-1)
    d[...,np.arange(4),np.arange(4)]=np.inf
    out=np.zeros(uv.shape[:-1],bool);out[...,js]=(d<=15).any(-1)
    return out

def solve(geo, uv, var, conf, spread, available):
    available=np.array(list(available),int)
    eligible=available[(conf[available]>=.6)&(spread[available]<=10)&np.isfinite(uv[available]).all(-1)]
    meta=dict(available=available.tolist(),eligible=eligible.tolist(),inliers=[],rejected=available.tolist(),
              reason='insufficient_views',support=0,subset_max_shift_mm=None)
    fail=np.full(3,np.nan)
    if len(eligible)<3:return fail,meta
    candidates={}
    for pair in itertools.combinations(eligible,2):
        idx=np.array(pair);P=geo.P[idx];p=uv[idx]
        A=np.concatenate([p[:,0,None]*P[:,2]-P[:,0],p[:,1,None]*P[:,2]-P[:,1]])
        _,_,v=np.linalg.svd(A);h=v[-1]
        if abs(h[3])<1e-12:continue
        x=h[:3]/h[3];proj,_,z=geo.project(x,eligible)
        support=eligible[(z>0)&(np.linalg.norm(proj-uv[eligible],axis=-1)<=15)]
        if len(support)<3 or len(support)*2<=len(eligible):continue
        key=tuple(support)
        if key in candidates:continue
        x,m=geo.solve(uv,var,conf,spread,support,CONFIGS['weighted_robust_floor4'])
        if not np.isfinite(x).all() or m['angle_deg']<3:continue
        proj,_,z=geo.project(x,support);err=np.linalg.norm(proj-uv[support],axis=-1)
        if np.any(z<=0) or np.max(err)>15:continue
        candidates[key]=(x,float(np.median(err)),m['angle_deg'])
    if not candidates:meta['reason']='no_three_view_consensus';return fail,meta
    ordered=sorted(candidates,key=lambda k:(-len(k),candidates[k][1],k))
    best=ordered[0];x,err,angle=candidates[best]
    # Equally supported incompatible hypotheses remain unknown, not a favorable pick.
    if any(len(k)==len(best) and np.linalg.norm(candidates[k][0]-x)>5 for k in ordered[1:]):
        meta['reason']='ambiguous_consensus';return fail,meta
    shifts=[]
    if len(best)>=4:
        for leave in best:
            sub=[i for i in best if i!=leave]
            y,_=geo.solve(uv,var,conf,spread,sub,CONFIGS['weighted_robust_floor4'])
            if np.isfinite(y).all():shifts.append(float(np.linalg.norm(y-x)))
    meta.update(reason='accepted',inliers=[int(i) for i in best],rejected=[int(i) for i in available if i not in best],
                support=len(best),median_reprojection_px=err,ray_angle_deg=angle,
                subset_max_shift_mm=max(shifts) if shifts else None)
    return x,meta

def selftest():
    cal={c:dict(K=[[800,0,540],[0,800,600],[0,0,1]],world_to_camera_mm=np.block([[np.eye(3),np.array(t)[:,None]],[np.zeros((1,3)),np.ones((1,1))]]).tolist())
         for c,t in zip('ABCD',[[-50,0,0],[0,-50,0],[50,0,0],[0,50,0]])}
    geo=Geometry(cal,list(cal));truth=np.array([6.,9.,250.]);uv=geo.project(truth,np.arange(4))[0]
    var=np.ones((4,2));conf=np.ones(4);spread=np.zeros(4)
    x,m=solve(geo,uv,var,conf,spread,range(4));assert np.linalg.norm(x-truth)<1e-6 and m['support']==4
    bad=uv.copy();bad[3]+=[190,90]
    x,m=solve(geo,bad,var,conf,spread,range(4));assert np.linalg.norm(x-truth)<1e-6 and m['inliers']==[0,1,2]
    y,n=solve(geo,bad,var,conf,spread,range(3));assert np.linalg.norm(y-truth)<1e-6 and 3 not in n['available']
    bad[3]=np.nan;z,_=solve(geo,bad,var,conf,spread,range(3));assert np.allclose(y,z)
    x,_=solve(geo,uv,var,conf,spread,range(2));assert np.isnan(x).all()
    return dict(exact_geometry=True,outlier_rejection=True,held_camera_invariant=True,two_view_rejected=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input-suffix',default='_exactnorm')
    ap.add_argument('--reference-run',default='weighted_framewise_pilot_floor4')
    ap.add_argument('--output-run',default='consensus_pilot_v1');ap.add_argument('--reject-paw-collisions',action='store_true');args=ap.parse_args()
    assert all('/' not in x for x in [args.input_suffix,args.reference_run,args.output_run])
    tests=selftest();plan=json.loads((OUT/'plan.json').read_text())
    deadline=min(time.time()+1200,dt.datetime.fromisoformat(plan['deadline_utc']).timestamp())
    def check():assert time.time()<deadline and not (OUT/'STOP').exists()
    check();dest=OUT/args.output_run;dest.mkdir(exist_ok=False)
    (dest/'settings.json').write_text(json.dumps(dict(settings=SETTINGS,tests=tests,arguments=vars(args),thresholds='Predeclared heuristic diagnostic; not calibrated guarantees'),indent=2))
    write_state(stage='consensus_running',pid=os.getpid(),active_run=args.output_run,mode='CPU',job_deadline_utc=dt.datetime.fromtimestamp(deadline,dt.timezone.utc).isoformat())
    manifests=[json.loads((OUT/f'continuous_lp_seed{s}_kl{args.input_suffix}/manifest.json').read_text()) for s in [17,31,73]]
    manual=json.loads((BASE/'ingestion_manifest.json').read_text());ids=np.array(plan['approved_body_ids_1based'])
    key=lambda r:(r['session'],r['frame'],r['camera'],r['id'])
    prior={key(r):r for r in json.loads((OUT/args.reference_run/'observations.json').read_text())}
    excluded={(r['session'],r['frame']) for r in json.loads((ROOT/plan['exclusion_manifest']).read_text())['cases']}
    rows=[];clips=[]
    for ci,clip in enumerate(manifests[0]):
        session=clip['session'];assert session in plan['development_validation_sessions']
        ps=[]
        for m in manifests:
            assert m[ci]['session']==session
            with np.load(Path(m[ci]['folder'])/'predictions_source_pixels.npz') as z:ps.append(dict(z))
        cams=ps[0]['cameras'].tolist();frames=ps[0]['source_frames'];C=len(cams);N=len(frames)
        assert all(np.array_equal(p['source_frames'],frames) and p['cameras'].tolist()==cams for p in ps)
        assert not any((session,int(f)) in excluded for f in frames)
        raw=np.stack([p['uv'] for p in ps]);conf=np.mean([p['confidence'] for p in ps],axis=0)
        uv=np.median(raw,axis=0);spread=np.sqrt(np.square(raw-raw.mean(0)).sum(-1).mean(0))
        var=np.var(raw,axis=0)/np.maximum(conf[...,None],1e-12)
        collision=paw_collisions(uv,ids)
        if args.reject_paw_collisions:
            conf=conf.copy();conf[collision]=0
        cal=json.loads((BASE/'calibration'/session/'rectified_calibration.json').read_text());geo=Geometry(cal,cams);cg=group_from_cal(cal,cams)
        for g in manual['frame_groups']:
            if g['session']!=session or g['frame'] not in frames:continue
            fi=frames.tolist().index(g['frame'])
            with np.load(BASE/g['normalized_file']) as z:gt=z['uv'][:,0,ids-1]
            for gi,c in enumerate(g['cameras']):
                if c not in cams:continue
                vi=cams.index(c);other=[i for i in range(C) if i!=vi]
                for j,k in enumerate(ids):
                    if not np.isfinite(gt[gi,j]).all():continue
                    check();r=dict(prior[(session,g['frame'],c,int(k))])
                    x,meta=solve(geo,uv[:,fi,j],var[:,fi,j],conf[:,fi,j],spread[:,fi,j],other)
                    assert vi not in meta['available']
                    proj=safe_project(cg.subset_cameras([vi]),x[None])[0,0]
                    r['consensus']=float(np.linalg.norm(proj-gt[gi,j])) if np.isfinite(proj).all() else None
                    r['consensus_meta']=meta;rows.append(r)
        xyz=np.full((N,7,3),np.nan);mask=np.zeros((C,N,7),bool);eligible=np.zeros_like(mask)
        reasons=np.empty((N,7),dtype='U32');shift=np.full((N,7),np.nan)
        for fi in range(N):
            check()
            for j in range(7):
                xyz[fi,j],meta=solve(geo,uv[:,fi,j],var[:,fi,j],conf[:,fi,j],spread[:,fi,j],range(C))
                mask[meta['inliers'],fi,j]=True;eligible[meta['eligible'],fi,j]=True;reasons[fi,j]=meta['reason']
                if meta['subset_max_shift_mm'] is not None:shift[fi,j]=meta['subset_max_shift_mm']
        projected=safe_project(cg,xyz.reshape(-1,3)).reshape(C,N,7,2)
        with np.load(OUT/args.reference_run/f'{session}.npz') as z:baseline=dict(z)
        np.savez_compressed(dest/f'{session}.npz',xyz_consensus_mm=xyz,uv_consensus=projected,
            xyz_weighted_mm=baseline['xyz_weighted_mm'],uv_weighted=baseline['uv_weighted'],uv_gated=projected,
            inlier_views=mask,eligible_views=eligible,rejected_views=~mask,support_views=mask.sum(0),predicted_paw_collisions=collision,
            rejection_reason=reasons,subset_max_shift_mm=shift,ids=ids,cameras=cams,source_frames=frames,
            imputed=np.zeros((N,7),bool),train_allowed=False)
        clips.append(dict(session=session,accepted_fraction=float(np.isfinite(xyz).all(-1).mean()),
            accepted_by_keypoint={str(k):float(np.isfinite(xyz[:,j]).all(-1).mean()) for j,k in enumerate(ids)},
            inlier_by_camera={c:int(mask[i].sum()) for i,c in enumerate(cams)},
            reason_counts={str(r):int((reasons==r).sum()) for r in np.unique(reasons)},
            subset_shift_mm=stats(shift[np.isfinite(shift)])))
        print('CLIP_DONE',clips[-1],flush=True)
    metrics=['framewise_loo','eks_loo','weighted_robust_floor4','gated_robust_floor4','consensus']
    retained=[r for r in rows if r['consensus'] is not None]
    common_gate=[r for r in retained if r['gated_robust_floor4'] is not None]
    def summary(group,keys):return {k:stats([r[k] for r in group if r[k] is not None]) for k in keys}
    report=dict(manual_observations=len(rows),manual_source_groups=len({(r['session'],r['frame']) for r in rows}),
        all_available={k:dict(coverage=sum(r[k] is not None for r in rows)/len(rows),errors=summary(rows,[k])[k]) for k in metrics},
        same_consensus_retained_cohort=summary(retained,[k for k in metrics if k!='gated_robust_floor4']),
        same_gate_and_consensus_cohort=summary(common_gate,metrics),
        coverage_by_keypoint={str(k):dict(total=int(sum(r['id']==k for r in rows)),retained=int(sum(r['id']==k for r in retained))) for k in ids},
        clips=clips,train_allowed=False,warning='Development-only diagnostics. True camera holdout; no temporal fill. Consensus agreement is not anatomical correctness or3D GT. Repeated points/frames are correlated.')
    (dest/'observations.json').write_text(json.dumps(rows,indent=2));(dest/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    write_state(stage='consensus_evaluated',pid=None,active_run=None,mode='idle_between_heartbeats',job_deadline_utc=None,
                latest_result=str(dest/'summary.json'),next_action='Audit quality and coverage, matched retained cohorts, manual-only geometry and visual failures; no pseudo-GT release.')

if __name__=='__main__':main()
