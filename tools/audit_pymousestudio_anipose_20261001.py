"""Aniposelib comparison with fixed cameras and review-case exclusion.

Sparse sampled frames: NO temporal/bone regularization. Single-view coordinates,
abdomen and all tail slots remain untouched. All outputs are diagnostic, not GT.
"""
import os
os.environ.setdefault('NUMBA_DISABLE_JIT', '1')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import cv2
import numpy as np
import aniposelib.cameras as camera_module
from aniposelib.cameras import Camera, CameraGroup
from audit_pymousestudio_manual_multiview import stats, projection

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'dataset/PyMouseStudio/annotation_returns/20260930_full_v2'
OLD=ROOT/'output/pymousestudio_manual_multiview_20260930'
REVIEW=ROOT/'output/pymousestudio_numbered_review_20261001/review_manifest.json'
OUT=ROOT/'output/pymousestudio_anipose_20261001'
BODY=np.array(list(range(7))+list(range(14,22)))
ACTIVE={0,5,6,14,15,16,17}
PRESERVED=np.array(list(range(7,14))+[22])


def group_from_cal(cal,cams):
    result=[]
    for c in cams:
        E=np.asarray(cal[c]['world_to_camera_mm'],float)
        assert np.allclose(cal[c]['distortion'],0)
        result.append(Camera(matrix=np.asarray(cal[c]['K'],float),dist=np.zeros(5),size=(1080,1200),
                             rvec=cv2.Rodrigues(E[:3,:3])[0].ravel(),tvec=E[:3,3].copy(),name=c))
    return CameraGroup(result)


def fit(cgroup,uv):
    """Only independently reconstructible points enter the optimizer.

    Batch N dimension enumerates independent points, not temporal frames.
    Both temporal and length weights are zero, so there is no cross-point loss.
    """
    n=uv.shape[1];good=np.isfinite(uv).all(-1).sum(0)>=2
    dlt=np.full((n,3),np.nan);opt=dlt.copy()
    if good.sum():
        points=uv[:,good]
        dlt[good]=cgroup.triangulate(points,undistort=True)  # D=0; only pixel->normalized-ray conversion.
        assert np.isfinite(dlt[good]).all()
        if good.sum()<2:
            raise RuntimeError('Insufficient independent batch size for zero-smoothing aniposelib optimizer')
        opt[good]=cgroup.optim_points(points[:,:,None,:],dlt[good][:,None,:],constraints=[],constraints_weak=[],
            scale_smooth=0,scale_length=0,scale_length_weak=0,reproj_error_threshold=15,reproj_loss='soft_l1',verbose=False)[:,0]
        assert np.isfinite(opt[good]).all()
    return dlt,opt


def safe_project(cgroup,xyz):
    valid=np.isfinite(xyz).all(-1);uv=np.full((len(cgroup.cameras),len(xyz),2),np.nan)
    if valid.any():
        uv[:,valid]=cgroup.project(xyz[valid])
        for i,cam in enumerate(cgroup.cameras):
            E=cam.get_extrinsics_mat()
            z=xyz[valid]@E[2,:3]+E[2,3]
            uv[i,np.flatnonzero(valid)[z<=0]]=np.nan
    return uv


def selftest():
    cal={}
    for c,t in zip('ABCD',[[-50,0,0],[0,-50,0],[50,0,0],[0,50,0]]):
        e=np.eye(4);e[:3,3]=t
        cal[c]={'K':[[800,0,540],[0,800,600],[0,0,1]],'distortion':[0]*5,'world_to_camera_mm':e.tolist()}
    cg=group_from_cal(cal,list(cal))
    xyz=np.array([[4+i*.7,12-i*.5,250+i] for i in range(30)])
    uv=cg.project(xyz);a,b=fit(cg,uv)
    assert np.max(np.abs(a-xyz))<1e-6 and np.max(np.abs(b-xyz))<1e-5
    shuffled=np.arange(30)[::-1];_,rev=fit(cg,uv[:,shuffled])
    np.testing.assert_allclose(rev,xyz[shuffled],atol=1e-5)
    for i,c in enumerate(cal):
        pred,_=projection(xyz[0],np.array([cal[c]['K']]),np.array([cal[c]['world_to_camera_mm']]))
        np.testing.assert_allclose(pred[0],uv[i,0],atol=1e-6)
    return dict(exact_reconstruction=True,projection_conventions_match=True,no_cross_point_regularization=True)


def summarize(rows):
    out={}
    for method in ['baseline_ls','anipose_dlt','anipose_robust']:
        out[method]={kind:stats([r[method+'_'+kind] for r in rows if r[method+'_'+kind] is not None]) for kind in ['fit','loo']}
    paired=[r for r in rows if r['baseline_ls_loo'] is not None and r['anipose_robust_loo'] is not None]
    if paired:
        before=np.array([r['baseline_ls_loo'] for r in paired]);after=np.array([r['anipose_robust_loo'] for r in paired])
        out['paired_loo']=dict(n=len(paired),before=stats(before),after=stats(after),
                              median_after_minus_before_px=float(np.median(after-before)),
                              improved_fraction=float(np.mean(after<before)),
                              improved_over2px_fraction=float(np.mean(after<before-2)),
                              worsened_over2px_fraction=float(np.mean(after>before+2)))
    return out


def main():
    t=time.time();OUT.mkdir(exist_ok=False)
    tests=selftest();print('ANIPOSELIB_SELFTEST_PASSED',flush=True)
    manifest=json.loads((BASE/'ingestion_manifest.json').read_text())
    review=json.loads(REVIEW.read_text())['cases']
    excluded={(r['session'],r['frame']) for r in review}
    assert len(excluded)==24
    selected=[g for g in manifest['frame_groups'] if (g['session'],g['frame']) not in excluded]
    calibration_hashes={r['path']:hashlib.sha256((BASE/r['path']).read_bytes()).hexdigest() for r in manifest['calibration']}
    config=dict(aniposelib_version=importlib.metadata.version('aniposelib'),installed_cameras_source=str(Path(camera_module.__file__)),
                installed_cameras_sha256=hashlib.sha256(Path(camera_module.__file__).read_bytes()).hexdigest(),
                camera_fixed=True,source_images_already_rectified=True,distortion_coefficients_zero=True,
                body_ids=(BODY+1).tolist(),preserved_ids=(PRESERVED+1).tolist(),single_view_policy='preserve exactly; no inferred supervision',
                excluded_cases=[dict(id=r['id'],session=r['session'],frame=r['frame']) for r in review],
                excluded_groups=len(excluded),included_groups=len(selected),
                fit='Actual aniposelib triangulate + optim_points; soft_l1 threshold 15 px; no RANSAC subset selection',
                temporal_smooth=0,bone_constraints=[],scores='equal manual weights; no invented confidences',
                batch_dimension='Independent points batched for efficiency; temporal and length weights zero',
                holdout_policy='same frozen method, diagnostic only, no tuning',train_allowed=False,tests=tests)
    (OUT/'config.json').write_text(json.dumps(config,indent=2))
    rows=[];preserved_count=0;singleton_count=0;frames_out=[]
    for session in manifest['sessions']:
        groups=[g for g in selected if g['session']==session]
        cal=json.loads((BASE/'calibration'/session/'rectified_calibration.json').read_text())
        cams=sorted(cal);cg=group_from_cal(cal,cams);before_params=cg.get_dicts()
        uv=np.full((len(cams),len(groups),23,2),np.nan);old=[]
        for gi,g in enumerate(groups):
            with np.load(BASE/g['normalized_file']) as z:
                for i,c in enumerate(g['cameras']):uv[cams.index(c),gi]=z['uv'][i,0]
            with np.load(BASE/'geometry_diagnostics_NOT_GT'/g['split']/session/f"frame_{g['frame']:06d}.npz") as z:old.append(dict(z))
        flat=uv[:,:,BODY,:].reshape(len(cams),-1,2)
        dlt,opt=fit(cg,flat)
        pdlt=safe_project(cg,dlt).reshape(len(cams),len(groups),len(BODY),2)
        popt=safe_project(cg,opt).reshape(pdlt.shape)
        loo_dlt=np.full_like(pdlt,np.nan);loo_opt=loo_dlt.copy()
        for vi in range(len(cams)):
            ids=[i for i in range(len(cams)) if i!=vi]
            # Withheld labels are never used for initialization, filtering, or fitting.
            a,b=fit(cg.subset_cameras(ids),flat[ids])
            loo_dlt[vi]=safe_project(cg.subset_cameras([vi]),a)[0].reshape(len(groups),len(BODY),2)
            loo_opt[vi]=safe_project(cg.subset_cameras([vi]),b)[0].reshape(len(groups),len(BODY),2)
        dlt=dlt.reshape(len(groups),len(BODY),3);opt=opt.reshape(dlt.shape)
        for gi,g in enumerate(groups):
            numviews=np.isfinite(uv[:,gi]).all(-1).sum(0)
            proposed=uv[:,gi].copy();updated=np.zeros((len(cams),23),bool)
            project_all=np.full_like(proposed,np.nan);project_all[:,BODY]=popt[:,gi]
            for bi,k in enumerate(BODY):
                for vi,c in enumerate(cams):
                    if not np.isfinite(uv[vi,gi,k]).all():continue
                    if numviews[k]<2:
                        singleton_count+=1;continue
                    baseline_vi=g['cameras'].index(c)
                    fits=[float(old[gi]['error_all_ls_px'][baseline_vi,k]),
                          float(np.linalg.norm(pdlt[vi,gi,bi]-uv[vi,gi,k])),float(np.linalg.norm(popt[vi,gi,bi]-uv[vi,gi,k]))]
                    loos=[float(old[gi]['error_leave_one_view_out_px'][baseline_vi,k]),
                          float(np.linalg.norm(loo_dlt[vi,gi,bi]-uv[vi,gi,k])),float(np.linalg.norm(loo_opt[vi,gi,bi]-uv[vi,gi,k]))]
                    row=dict(session=session,split=g['split'],frame=g['frame'],camera=c,id=int(k+1),active=k in ACTIVE,n_views=int(numviews[k]))
                    for m,f,l in zip(['baseline_ls','anipose_dlt','anipose_robust'],fits,loos):
                        row[m+'_fit']=f if np.isfinite(f) else None
                        row[m+'_loo']=l if numviews[k]>=3 and np.isfinite(l) else None
                    rows.append(row)
                    if np.isfinite(popt[vi,gi,bi]).all():
                        proposed[vi,k]=popt[vi,gi,bi];updated[vi,k]=True
            np.testing.assert_equal(proposed[:,PRESERVED],uv[:,gi,PRESERVED])
            one=numviews<2;np.testing.assert_equal(proposed[:,one],uv[:,gi,one])
            assert not updated[:,PRESERVED].any()
            preserved_count+=int(np.isfinite(uv[:,gi,PRESERVED]).all(-1).sum())
            target=OUT/'diagnostic_npz_NOT_GT'/g['split']/session/f"frame_{g['frame']:06d}.npz"
            target.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(target,uv_original=uv[:,gi],uv_proposed_for_review=proposed,uv_projection_all=project_all,
                updated_mask=updated,xyz_anipose_dlt_mm=dlt[gi],xyz_anipose_robust_mm=opt[gi],body_ids=BODY+1,
                uv_loo_dlt=loo_dlt[:,gi],uv_loo_robust=loo_opt[:,gi],camera_names=np.array(cams),
                train_allowed=False,not_ground_truth=True,source_frame=g['frame'])
            frames_out.append(dict(**g,diagnostic_path=target.relative_to(OUT).as_posix()))
        # Compare numeric camera parameters after optimization; optim_points must never calibrate cameras.
        for a,b in zip(before_params,cg.get_dicts()):
            for key in ('matrix','distortions','rotation','translation'):np.testing.assert_equal(a[key],b[key])
        print(f'ANIPOSE {session}: {len(groups)} groups incl all LOO fits | elapsed {time.time()-t:.1f}s',flush=True)
    for path,h in calibration_hashes.items():assert hashlib.sha256((BASE/path).read_bytes()).hexdigest()==h
    result=dict(included_groups=len(selected),excluded_groups=len(excluded),all_non_tail=summarize(rows),
                approved_non_tail=summarize([r for r in rows if r['active']]),
                splits={s:summarize([r for r in rows if r['active'] and r['split']==s]) for s in manifest['split_counts']},
                cameras={c:summarize([r for r in rows if r['active'] and r['camera']==c]) for c in 'ABCDEFG'},
                keypoints={str(k+1):summarize([r for r in rows if r['id']==k+1]) for k in BODY},
                preserved_tail_and_abdomen_observations=preserved_count,preserved_singleview_body_observations=singleton_count,
                camera_hashes_unchanged=True,train_allowed=False,seconds=time.time()-t)
    for name,obj in [('summary.json',result),('observation_metrics.json',rows),('frame_manifest.json',frames_out)]:
        (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps(result['approved_non_tail'],indent=2),flush=True)


if __name__=='__main__':main()
