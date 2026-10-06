"""Fixed teacher: framewise vs temporal Anipose; true leave-camera-out diagnostic."""
import os
os.environ['NUMBA_DISABLE_JIT']='1';os.environ['JAX_PLATFORMS']='cpu'
import datetime as dt
import json
import argparse
import time
from pathlib import Path
import numpy as np
from audit_pymousestudio_anipose_20261001 import group_from_cal,safe_project
from audit_pymousestudio_manual_multiview import stats
from scipy.interpolate import interp1d
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/teacher_temporal_cycle_20261001'
BASE=ROOT/'dataset/PyMouseStudio/annotation_returns/20260930_full_v2'

def reconstruct(cg,uv,scales=(0,1,4)):
    C,N,J,_=uv.shape;flat=uv.reshape(C,-1,2)
    xyz=cg.triangulate(flat).reshape(N,J,3)
    valid=np.isfinite(xyz).all(-1)
    good=valid.sum(0)>=10
    result={s:np.full_like(xyz,np.nan) for s in scales}
    if not good.any():return result,valid
    filtered=uv[:,:,good].copy()
    filtered[:,~valid[:,good]]=np.nan
    for scale in scales:
        fitted=cg.optim_points(filtered,xyz[:,good],constraints=[],constraints_weak=[],
            scale_smooth=scale,scale_length=0,scale_length_weak=0,reproj_loss='soft_l1',reproj_error_threshold=15)
        if scale==0:fitted[~valid[:,good]]=np.nan
        result[scale][:,good]=fitted
    return result,valid

def main():
    p=argparse.ArgumentParser();p.add_argument('--input-run',default='continuous_seed17_epoch30');p.add_argument('--output-run',default='temporal_anipose_seed17_epoch30');args=p.parse_args()
    assert '/' not in args.input_run and '/' not in args.output_run
    start=time.monotonic();dest=OUT/args.output_run;dest.mkdir(exist_ok=False)
    plan=json.loads((OUT/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['deadline_utc']).timestamp()
    manifest=json.loads((OUT/args.input_run/'manifest.json').read_text());assert len(manifest)==2
    manual_manifest=json.loads((BASE/'ingestion_manifest.json').read_text())
    rows=[];clips=[]
    for clip in manifest:
        assert time.time()<deadline and time.monotonic()-start<1200
        session=clip['session'];assert session in plan['development_validation_sessions']
        with np.load(Path(clip['folder'])/'predictions_source_pixels.npz') as z:pred=dict(z)
        uv=pred['uv'].copy();uv[pred['confidence']<.3]=np.nan
        cal=json.loads((BASE/'calibration'/session/'rectified_calibration.json').read_text());cams=pred['cameras'].tolist();cg=group_from_cal(cal,cams)
        xs,valid=reconstruct(cg,uv)
        N,J=uv.shape[1:3]
        project={s:safe_project(cg,x.reshape(-1,3)).reshape(len(cams),N,J,2) for s,x in xs.items()}
        loo={s:np.full_like(uv,np.nan) for s in xs}
        for v,c in enumerate(cams):
            assert time.time()<deadline and time.monotonic()-start<1200
            other=[i for i in range(len(cams)) if i!=v]
            held,_=reconstruct(cg.subset_cameras(other),uv[other])
            for s,x in held.items():loo[s][v]=safe_project(cg.subset_cameras([v]),x.reshape(-1,3))[0].reshape(N,J,2)
            print('LOO_DONE',session,c,'seconds',round(time.monotonic()-start,1),flush=True)
        # The sparse manual coordinates below were NOT used in any reconstruction.
        for g in manual_manifest['frame_groups']:
            if g['session']!=session or g['frame'] not in pred['source_frames']:continue
            fi=pred['source_frames'].tolist().index(g['frame'])
            with np.load(BASE/g['normalized_file']) as z:gt=z['uv'][:,0,pred['ids']-1]
            for gi,c in enumerate(g['cameras']):
                if c not in cams:continue
                vi=cams.index(c)
                for j,k in enumerate(pred['ids']):
                    if not np.isfinite(gt[gi,j]).all():continue
                    def err(p):return float(np.linalg.norm(p-gt[gi,j])) if np.isfinite(p).all() else None
                    row=dict(session=session,frame=g['frame'],camera=c,id=int(k),raw_teacher_error_px=err(pred['uv'][vi,fi,j]),
                             teacher_confidence=float(pred['confidence'][vi,fi,j]),support_views=int(np.isfinite(uv[:,fi,j]).all(-1).sum()))
                    for s in xs:
                        row[f'scale{s}_allview_error_px']=err(project[s][vi,fi,j]);row[f'scale{s}_loo_error_px']=err(loo[s][vi,fi,j])
                    rows.append(row)
        measures={}
        for s,x in xs.items():
            speed=np.linalg.norm(np.diff(x,axis=0)*52,axis=-1)
            accel=np.linalg.norm(np.diff(x,n=2,axis=0)*52**2,axis=-1)
            measures[str(s)]=dict(speed_mm_s=stats(speed.ravel()),accel_mm_s2=stats(accel.ravel()),
                                  generated_without_two_view_support=int((np.isfinite(x).all(-1)&~valid).sum()))
        np.savez_compressed(dest/f'{session}.npz',uv_raw=pred['uv'],confidence=pred['confidence'],uv_used=uv,
            xyz_framewise_mm=xs[0],xyz_temporal1_mm=xs[1],xyz_temporal4_mm=xs[4],
            uv_framewise=project[0],uv_temporal1=project[1],uv_temporal4=project[4],
            uv_loo_framewise=loo[0],uv_loo_temporal1=loo[1],uv_loo_temporal4=loo[4],
            source_frames=pred['source_frames'],cameras=cams,ids=pred['ids'],original_two_view_support=valid,
            train_allowed=False,not_ground_truth=True)
        clips.append(dict(session=session,frames=N,motion_measures=measures))
    # Paired cohort common to every compared method; no cherry-picked dropped errors.
    keys=['raw_teacher_error_px']+[f'scale{s}_{mode}_error_px' for mode in ['allview','loo'] for s in [0,1,4]]
    common=[r for r in rows if all(r[k] is not None for k in keys)]
    result=dict(confidence_threshold=.3,temporal_scales=[0,1,4],fps=52,manual_observations=len(rows),paired_observations=len(common),
        paired_metrics={k:stats([r[k] for r in common]) for k in keys},
        all_available_metrics={k:stats([r[k] for r in rows if r[k] is not None]) for k in keys},
        clips=clips,camera_fixed=True,bone_constraints=[],pending_frames_present=False,train_allowed=False,
        warning='Small pilot only. Lower acceleration is NOT evidence of accuracy. Interpolated unsupported points never become strong GT; manual coordinates are not inputs. Hardware sync/calibration applicability not independently validated.',seconds=time.monotonic()-start)
    (dest/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False));(dest/'manual_observation_errors.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    print(json.dumps(result['paired_metrics'],indent=2),flush=True)

if __name__=='__main__':main()
