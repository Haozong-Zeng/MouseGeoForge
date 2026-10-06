"""Official calibrated EKS vs same-observation framewise Anipose; true camera holdout."""
import os
os.environ['JAX_PLATFORMS']='cpu';os.environ['NUMBA_DISABLE_JIT']='1'
os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['OMP_NUM_THREADS']='4'
import argparse,datetime as dt,json,time
from pathlib import Path
import numpy as np
from eks.marker_array import MarkerArray
from eks.multicam_smoother import ensemble_kalman_smoother_multicam
from audit_pymousestudio_anipose_20261001 import group_from_cal,fit,safe_project
from audit_pymousestudio_manual_multiview import stats
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/teacher_temporal_cycle_20261001';BASE=ROOT/'dataset/PyMouseStudio/annotation_returns/20260930_full_v2'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',nargs='+',type=int,default=[17,31,73]);ap.add_argument('--input-suffix',default='_exactnorm')
    ap.add_argument('--variance-floor-px',type=float,default=0);ap.add_argument('--output-run');args=ap.parse_args()
    assert '/' not in args.input_suffix
    assert args.variance_floor_px>=0
    if args.variance_floor_px:
        # Explicit adapter: official EKS unchanged except a fixed observation-variance floor.
        import eks.multicam_smoother as official
        original=official.ensemble
        def floored_ensemble(*a,**kw):
            markers=original(*a,**kw);arr=np.array(markers.array,copy=True)
            arr[...,2:4]=np.maximum(arr[...,2:4],args.variance_floor_px**2)
            return MarkerArray(arr,data_fields=markers.data_fields)
        official.ensemble=floored_ensemble
    plan=json.loads((OUT/'plan.json').read_text());deadline=min(time.time()+1200,dt.datetime.fromisoformat(plan['deadline_utc']).timestamp())
    def check():assert time.time()<deadline and not (OUT/'STOP').exists()
    check()
    name=args.output_run or ('official_eks_'+'_'.join(map(str,args.seeds))+(f'_floor{args.variance_floor_px:g}' if args.variance_floor_px else ''))
    assert '/' not in name
    dest=OUT/name;dest.mkdir(exist_ok=False)
    m=json.loads((OUT/'dlc_dataset_manifest.json').read_text());ids=np.array(m['ids_1based']);names=m['names']
    manifests=[json.loads((OUT/f'continuous_lp_seed{s}_kl{args.input_suffix}/manifest.json').read_text()) for s in args.seeds]
    manual=json.loads((BASE/'ingestion_manifest.json').read_text());rows=[];motions={}
    excluded={(r['session'],r['frame']) for r in json.loads((ROOT/'output/pymousestudio_numbered_review_20261001/review_manifest.json').read_text())['cases']}
    for ci,clip in enumerate(manifests[0]):
        check();session=clip['session'];assert session in plan['development_validation_sessions']
        ps=[]
        for manifest in manifests:
            assert manifest[ci]['session']==session
            with np.load(Path(manifest[ci]['folder'])/'predictions_source_pixels.npz') as z:ps.append(dict(z))
        assert all(np.array_equal(p['cameras'],ps[0]['cameras']) and np.array_equal(p['source_frames'],ps[0]['source_frames']) for p in ps)
        cams=ps[0]['cameras'].tolist();frames=ps[0]['source_frames'];N=len(frames);C=len(cams);J=len(ids)
        assert not any((session,int(f)) in excluded for f in frames)
        raw=np.stack([np.concatenate([p['uv'],p['confidence'][...,None]],axis=-1) for p in ps])
        assert np.isfinite(raw).all()
        uv=np.median(raw[...,:2],axis=0)
        cal=json.loads((BASE/'calibration'/session/'rectified_calibration.json').read_text());cg=group_from_cal(cal,cams)
        def eks(others):
            check();sub=cg.subset_cameras(others)
            dfs,s,df3=ensemble_kalman_smoother_multicam(MarkerArray(raw[:,others],data_fields=['x','y','likelihood']),
                names,[cams[i] for i in others],smooth_param=1.,avg_mode='median',var_mode='confidence_weighted_var',inflate_vars=False,camgroup=sub)
            xyz=df3.to_numpy().reshape(N,J,6)[...,:3];assert np.isfinite(xyz).all()
            return xyz
        _,fx=fit(cg,uv.reshape(C,-1,2));fx=fx.reshape(N,J,3);ex=eks(list(range(C)))
        all_f=safe_project(cg,fx.reshape(-1,3)).reshape(C,N,J,2);all_e=safe_project(cg,ex.reshape(-1,3)).reshape(C,N,J,2)
        loo_f=np.full_like(uv,np.nan);loo_e=np.full_like(uv,np.nan)
        for vi,c in enumerate(cams):
            other=[i for i in range(C) if i!=vi]
            _,x=fit(cg.subset_cameras(other),uv[other].reshape(C-1,-1,2))
            loo_f[vi]=safe_project(cg.subset_cameras([vi]),x)[0].reshape(N,J,2)
            x=eks(other)
            loo_e[vi]=safe_project(cg.subset_cameras([vi]),x.reshape(-1,3))[0].reshape(N,J,2)
            print('EKS_LOO_DONE',session,c,flush=True)
        for g in manual['frame_groups']:
            if g['session']!=session or g['frame'] not in frames:continue
            fi=frames.tolist().index(g['frame'])
            with np.load(BASE/g['normalized_file']) as z:gt=z['uv'][:,0,ids-1]
            for gi,c in enumerate(g['cameras']):
                if c not in cams:continue
                vi=cams.index(c)
                for j,k in enumerate(ids):
                    if not np.isfinite(gt[gi,j]).all():continue
                    r=dict(session=session,frame=g['frame'],camera=c,id=int(k))
                    for name,arr in [('ensemble',uv),('framewise_all',all_f),('eks_all',all_e),('framewise_loo',loo_f),('eks_loo',loo_e)]:
                        r[name]=float(np.linalg.norm(arr[vi,fi,j]-gt[gi,j])) if np.isfinite(arr[vi,fi,j]).all() else None
                    rows.append(r)
        np.savez_compressed(dest/f'{session}.npz',xyz_framewise_mm=fx,xyz_eks_mm=ex,uv_ensemble=uv,uv_framewise=all_f,uv_eks=all_e,
            uv_loo_framewise=loo_f,uv_loo_eks=loo_e,ids=ids,cameras=cams,source_frames=frames,train_allowed=False)
        motions[session]={method:dict(speed_mm_s=stats(np.linalg.norm(np.diff(x,axis=0)*52,axis=-1).ravel()),
            amplitude_p95_p5_mm=np.percentile(x,95,axis=0).tolist(),note='Amplitude/speed not accuracy; no dense 3D truth available.') for method,x in [('framewise',fx),('eks',ex)]}
        for method,x in [('framewise',fx),('eks',ex)]:motions[session][method]['amplitude_p95_p5_mm']=(np.percentile(x,95,axis=0)-np.percentile(x,5,axis=0)).tolist()
    keys=['ensemble','framewise_all','eks_all','framewise_loo','eks_loo'];paired=[r for r in rows if all(r[k] is not None for k in keys)]
    result=dict(seeds=args.seeds,official_package='ensemble-kalman-smoother4.6.2',smooth_param=1.,variance_mode='confidence_weighted_var',variance_floor_px=args.variance_floor_px,
        observations=len(rows),paired_n=len(paired),paired={k:stats([r[k] for r in paired]) for k in keys},motions=motions,
        actual_held_view_input_removed=True,manual_labels_used_for_fitting=False,train_allowed=False,
        warning='Pilot only; smoothing parameter not optimized. Posterior variance is not calibrated correctness. No ground-truth 3D; shared teacher errors and camera/label inconsistencies persist.')
    (dest/'summary.json').write_text(json.dumps(result,indent=2));(dest/'observations.json').write_text(json.dumps(rows,indent=2));print(json.dumps(result['paired'],indent=2))

if __name__=='__main__':main()
