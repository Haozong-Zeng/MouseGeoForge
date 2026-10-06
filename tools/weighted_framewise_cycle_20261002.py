"""Fixed-camera, NO-temporal uncertainty controls; conservative diagnostic gates.
Custom scipy weighted triangulation, not a reimplementation advertised as EKS.
"""
import argparse,datetime as dt,json,time,os
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from audit_pymousestudio_anipose_20261001 import group_from_cal,safe_project
from audit_teacher_dlc_predictions_20261001 import stats
from run_teacher_cycle_dlc_20261001 import write_state
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/teacher_temporal_cycle_20261001';BASE=ROOT/'dataset/PyMouseStudio/annotation_returns/20260930_full_v2'
CONFIGS={
 'weighted_gaussian_exact':dict(floor=1e-6,loss='linear',gate=False),
 'weighted_gaussian_floor4':dict(floor=4.,loss='linear',gate=False),
 'weighted_robust_floor4':dict(floor=4.,loss='soft_l1',gate=False),
 'gated_robust_floor4':dict(floor=4.,loss='soft_l1',gate=True),
}

class Geometry:
    def __init__(self,cal,cams):
        self.E=np.array([cal[c]['world_to_camera_mm'] for c in cams])
        self.P=np.array([np.array(cal[c]['K'])@e[:3] for c,e in zip(cams,self.E)])
        self.centers=np.array([-e[:3,:3].T@e[:3,3] for e in self.E])
    def project(self,x,idx):
        P=self.P[idx];q=P[:,:,:3]@x+P[:,:,3];z=q[:,2]
        uv=q[:,:2]/z[:,None]
        jac=(P[:,:2,:3]*z[:,None,None]-q[:,:2,None]*P[:,2:3,:3])/z[:,None,None]**2
        return uv,jac,z
    def solve(self,uv,var,confidence,spread,available,config):
        idx=np.array(available,dtype=int)
        if config['gate']:idx=idx[(confidence[idx]>=.6)&(spread[idx]<=10)]
        meta=dict(support=len(idx),reason='insufficient_views',angle_deg=None,median_reproj_px=None)
        fail=np.full(3,np.nan)
        if len(idx)<2:return fail,meta
        P=self.P[idx];pts=uv[idx]
        A=np.concatenate([pts[:,0,None]*P[:,2]-P[:,0],pts[:,1,None]*P[:,2]-P[:,1]],axis=0)
        _,_,vh=np.linalg.svd(A);h=vh[-1]
        if abs(h[3])<1e-12:meta['reason']='infinite_dlt';return fail,meta
        x=h[:3]/h[3];sigma=np.sqrt(np.maximum(var[idx],config['floor']**2))
        def fun(x):return ((self.project(x,idx)[0]-pts)/sigma).ravel()
        def jac(x):return (self.project(x,idx)[1]/sigma[:,:,None]).reshape(-1,3)
        result=least_squares(fun,x,jac=jac,loss=config['loss'],f_scale=3.,max_nfev=80,xtol=1e-8,ftol=1e-8,gtol=1e-8)
        x=result.x;proj,_,z=self.project(x,idx)
        if not result.success or not np.isfinite(x).all():meta['reason']='optimizer_failed';return fail,meta
        if np.any(z<=0):meta['reason']='behind_camera';return fail,meta
        rays=x-self.centers[idx];rays/=np.linalg.norm(rays,axis=1)[:,None]
        angle=np.degrees(np.arccos(np.clip(np.abs(rays@rays.T),0,1))).max()
        reproj=np.linalg.norm(proj-pts,axis=-1)
        meta.update(angle_deg=float(angle),median_reproj_px=float(np.median(reproj)),reason='accepted')
        if config['gate']:
            if angle<3:meta['reason']='ray_angle_under3deg';return fail,meta
            if np.max(reproj)>15:meta['reason']='reprojection_over15px';return fail,meta
        return x,meta

def selftest():
    cal={c:dict(K=[[800,0,540],[0,800,600],[0,0,1]],world_to_camera_mm=np.block([[np.eye(3),np.array(t)[:,None]],[np.zeros((1,3)),np.ones((1,1))]]).tolist()) for c,t in zip('ABC',[[-50,0,0],[0,-50,0],[50,0,0]])}
    geo=Geometry(cal,list(cal));truth=np.array([6.,9.,250.]);uv,jac,_=geo.project(truth,np.arange(3))
    for k in range(3):
        dx=np.zeros(3);dx[k]=1e-4
        numeric=(geo.project(truth+dx,np.arange(3))[0]-geo.project(truth-dx,np.arange(3))[0])/(2e-4)
        assert np.allclose(numeric,jac[:,:,k],atol=1e-6)
    x,meta=geo.solve(uv,np.ones((3,2)),np.ones(3),np.zeros(3),range(3),CONFIGS['gated_robust_floor4'])
    assert np.linalg.norm(x-truth)<1e-6 and meta['reason']=='accepted'
    bad=uv.copy();bad[2]+=60;var=np.ones((3,2));var[2]=1e6
    x,_=geo.solve(bad,var,np.ones(3),np.zeros(3),range(3),CONFIGS['weighted_gaussian_exact'])
    assert np.linalg.norm(x-truth)<.01
    return dict(analytic_jacobian=True,exact_triangulation=True,uncertainty_downweights_outlier=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input-suffix',default='_exactnorm');ap.add_argument('--reference-run',default='official_eks_17_31_73');ap.add_argument('--output-run',default='weighted_framewise_v1');args=ap.parse_args()
    assert all('/' not in x for x in [args.input_suffix,args.reference_run,args.output_run])
    tests=selftest();print('SELFTEST',tests,flush=True)
    plan=json.loads((OUT/'plan.json').read_text());deadline=min(time.time()+1200,dt.datetime.fromisoformat(plan['deadline_utc']).timestamp())
    def check():assert time.time()<deadline and not (OUT/'STOP').exists()
    check();dest=OUT/args.output_run;dest.mkdir(exist_ok=False)
    (dest/'settings.json').write_text(json.dumps(dict(configs=CONFIGS,seeds=[17,31,73],gate_confidence=.6,gate_radial_spread_px=10,gate_max_reproj_px=15,gate_ray_angle_deg=3,
        tests=tests,reference_run=args.reference_run,input_suffix=args.input_suffix,temporal_smoothing=False,bone_constraints=False,thresholds_predeclared_not_calibrated=True,train_allowed=False),indent=2))
    manifests=[json.loads((OUT/f'continuous_lp_seed{s}_kl{args.input_suffix}/manifest.json').read_text()) for s in [17,31,73]]
    manual=json.loads((BASE/'ingestion_manifest.json').read_text());ids=np.array(plan['approved_body_ids_1based'])
    eks_rows=json.loads((OUT/args.reference_run/'observations.json').read_text());key=lambda r:(r['session'],r['frame'],r['camera'],r['id'])
    prior={key(r):r for r in eks_rows};rows=[];clips=[]
    excluded={(r['session'],r['frame']) for r in json.loads((ROOT/'output/pymousestudio_numbered_review_20261001/review_manifest.json').read_text())['cases']}
    write_state(stage='weighted_framewise_running',pid=os.getpid(),active_run=args.output_run,mode='CPU',job_deadline_utc=dt.datetime.fromtimestamp(deadline,dt.timezone.utc).isoformat())
    for ci,clip in enumerate(manifests[0]):
        session=clip['session'];assert session in plan['development_validation_sessions'];ps=[]
        for manifest in manifests:
            assert manifest[ci]['session']==session
            with np.load(Path(manifest[ci]['folder'])/'predictions_source_pixels.npz') as z:ps.append(dict(z))
        cams=ps[0]['cameras'].tolist();frames=ps[0]['source_frames'];C=len(cams);N=len(frames)
        assert not any((session,int(f)) in excluded for f in frames)
        assert all(np.array_equal(p['source_frames'],frames) and p['cameras'].tolist()==cams for p in ps)
        raw=np.stack([p['uv'] for p in ps]);conf=np.mean([p['confidence'] for p in ps],axis=0)
        uv=np.median(raw,axis=0);spread=np.sqrt(np.square(raw-raw.mean(0)).sum(-1).mean(0))
        var=np.var(raw,axis=0)/np.maximum(conf[...,None],1e-12)
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
                    for name,config in CONFIGS.items():
                        x,meta=geo.solve(uv[:,fi,j],var[:,fi,j],conf[:,fi,j],spread[:,fi,j],other,config)
                        projected=safe_project(cg.subset_cameras([vi]),x[None])[0,0]
                        r[name]=float(np.linalg.norm(projected-gt[gi,j])) if np.isfinite(projected).all() else None
                        r[name+'_meta']=meta
                    rows.append(r)
        # Full contiguous sequence only for conservative candidate diagnostic/visualization.
        xyz=np.full((N,7,3),np.nan);xyz_weighted=np.full_like(xyz,np.nan);support=np.zeros((N,7),int);reason=np.empty((N,7),dtype='U30');angle=np.full((N,7),np.nan)
        for fi in range(N):
            check()
            for j in range(7):
                xyz_weighted[fi,j],_=geo.solve(uv[:,fi,j],var[:,fi,j],conf[:,fi,j],spread[:,fi,j],range(C),CONFIGS['weighted_robust_floor4'])
                xyz[fi,j],meta=geo.solve(uv[:,fi,j],var[:,fi,j],conf[:,fi,j],spread[:,fi,j],range(C),CONFIGS['gated_robust_floor4'])
                support[fi,j]=meta['support'];reason[fi,j]=meta['reason'];angle[fi,j]=meta['angle_deg'] if meta['angle_deg'] is not None else np.nan
        np.savez_compressed(dest/f'{session}.npz',xyz_weighted_mm=xyz_weighted,uv_weighted=safe_project(cg,xyz_weighted.reshape(-1,3)).reshape(C,N,7,2),xyz_gated_mm=xyz,uv_gated=safe_project(cg,xyz.reshape(-1,3)).reshape(C,N,7,2),uv_ensemble=uv,
            support_views=support,rejection_reason=reason,ray_angle_deg=angle,ids=ids,cameras=cams,source_frames=frames,
            imputed=np.zeros((N,7),bool),train_allowed=False)
        clips.append(dict(session=session,accepted_fraction=float(np.isfinite(xyz).all(-1).mean()),accepted_by_keypoint={str(k):float(np.isfinite(xyz[:,j]).all(-1).mean()) for j,k in enumerate(ids)},
            reason_counts={r:int((reason==r).sum()) for r in np.unique(reason)}))
        print('CLIP_DONE',session,clips[-1],flush=True)
    keys=['framewise_loo','eks_loo']+list(CONFIGS)
    report=dict(manual_observations=len(rows),reference_run=args.reference_run,all_available={k:dict(coverage=sum(r[k] is not None for r in rows)/len(rows),errors=stats([r[k] for r in rows if r[k] is not None])) for k in keys},paired={},clips=clips,
        camera_fixed=True,held_view_removed_before_gates_and_fit=True,train_allowed=False,
        warning='Small development pilot. Rejection changes cohort, so compare SAME retained observations. Variance floors and gates are heuristics, not calibrated guarantees. No 3D GT and no imputation.')
    for name in CONFIGS:
        common=[r for r in rows if r[name] is not None and r['framewise_loo'] is not None and r['eks_loo'] is not None]
        report['paired'][name]={k:stats([r[k] for r in common]) for k in ['framewise_loo','eks_loo',name]}
    (dest/'observations.json').write_text(json.dumps(rows,indent=2));(dest/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    write_state(stage='weighted_framewise_evaluated',pid=None,active_run=None,mode='idle_between_heartbeats',latest_result=str(dest/'summary.json'),next_action='Inspect paired quality/coverage of fixed-camera weighted controls and sparse candidates; never equate filtering with improved full-cohort accuracy.')

if __name__=='__main__':main()
