"""Same-hypothesis independent vs offline temporal identity, true camera holdout."""
import argparse,datetime as dt,json,os,time
import numpy as np
from multipeak_geometry_cycle_20261002 import merge,hypotheses,Geometry,ROOT,OUT,BASE
from temporal_joint_states_20261002 import states,decode,selftest,SETTINGS
from audit_teacher_dlc_predictions_20261001 import stats
from run_teacher_cycle_dlc_20261001 import write_state

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--peak-run',default='temporal_paw_peaks_v1');ap.add_argument('--output-run',default='temporal_joint_states_v1');args=ap.parse_args()
    assert '/' not in args.peak_run and '/' not in args.output_run
    tests=selftest();plan=json.loads((OUT/'plan.json').read_text());deadline=min(time.time()+1200,dt.datetime.fromisoformat(plan['deadline_utc']).timestamp())
    def check():assert time.time()<deadline and not (OUT/'STOP').exists()
    check();dest=OUT/args.output_run;dest.mkdir(exist_ok=False)
    (dest/'settings.json').write_text(json.dumps(dict(settings=SETTINGS,tests=tests,geometry_settings=json.loads((OUT/'multipeak_geometry_v1/settings.json').read_text()),weights=[0,.5]),indent=2))
    write_state(stage='temporal_joint_states_running',pid=os.getpid(),active_run=dest.name,mode='CPU',job_deadline_utc=dt.datetime.fromtimestamp(deadline,dt.timezone.utc).isoformat())
    manifest=json.loads((OUT/args.peak_run/'manifest.json').read_text());assert len(manifest)==12
    excluded={(r['session'],r['frame']) for r in json.loads((ROOT/plan['exclusion_manifest']).read_text())['cases']};excluded.add(('AZR2L2',584))
    manual=json.loads((BASE/'ingestion_manifest.json').read_text());allrows=[];motions=[];parities=[]
    previous={}
    for cohort in ['pilot','stress1']:
        for r in json.loads((OUT/f'joint_paw_assignment_v1/{cohort}_observations.json').read_text()):previous[(cohort,r['session'],r['frame'],r['camera'],r['id'])]=r
    for cohort,session in [('pilot','AZR2L2'),('pilot','FR2'),('stress1','AZR2L2'),('stress1','FR2')]:
        check();assert session in plan['development_validation_sessions'];ps=[]
        for seed in [17,31,73]:
            record=next(r for r in manifest if (r['cohort'],r['session'],r['seed'])==(cohort,session,seed))
            with np.load(record['file']) as z:ps.append(dict(z))
        frames=ps[0]['source_frames'];cams=ps[0]['cameras'].tolist();N=len(frames);C=len(cams)
        assert np.all(np.diff(frames)==1) and not any((session,int(f)) in excluded for f in frames)
        assert all(np.array_equal(p['source_frames'],frames) and p['cameras'].tolist()==cams for p in ps)
        cal=json.loads((BASE/'calibration'/session/'rectified_calibration.json').read_text());geo=Geometry(cal,cams)
        candidates=[[[merge(np.stack([p['peaks_source_px'][vi,fi,j] for p in ps]),np.stack([p['relative_scores'][vi,fi,j] for p in ps])) for vi in range(C)] for j in range(4)] for fi in range(N)]
        for held in [None]+list(range(C)):
            check();others=[i for i in range(C) if i!=held];label='ALL' if held is None else cams[held];xs=[];es=[]
            for fi in range(N):
                check();hs=[]
                for j in range(4):
                    _,meta=hypotheses(geo,candidates[fi][j],others,3)
                    if held is not None:assert held not in meta['available']
                    hs.append(meta['hypotheses'])
                x,e=states(hs);xs.append(x);es.append(e)
            check();independent=decode(xs,es,0);temporal=decode(xs,es,.5)
            # Every accepted point is from a current-frame >=3-view hypothesis, never interpolation.
            for result in [independent,temporal]:
                for fi in range(N):
                    for j in range(4):
                        if np.isfinite(result['xyz'][fi,j]).all():assert np.any(np.linalg.norm(xs[fi][:,j]-result['xyz'][fi,j],axis=-1)<1e-8)
            maxstates=max(map(len,es));padded_x=np.full((N,maxstates,4,3),np.nan,dtype=np.float32);padded_e=np.full((N,maxstates),-np.inf);counts=[]
            for fi,(x,e) in enumerate(zip(xs,es)):padded_x[fi,:len(x)]=x;padded_e[fi,:len(e)]=e;counts.append(len(e))
            file=dest/f'{cohort}_{session}_held{label}.npz'
            np.savez_compressed(file,xyz_framewise_mm=independent['xyz'],xyz_temporal_mm=temporal['xyz'],
                ambiguous_framewise=independent['ambiguous'],ambiguous_temporal=temporal['ambiguous'],
                state_xyz_mm=padded_x,emission_scores=padded_e,state_counts=counts,
                source_frames=frames,cameras=cams,ids=[15,16,17,18],held=label,fps=52,imputed=np.zeros((N,4),bool),train_allowed=False)
            if held is None:
                motion=dict(cohort=cohort,session=session,methods={})
                for name,result in [('framewise',independent),('temporal',temporal)]:
                    x=result['xyz'];valid=np.isfinite(x).all(-1);pair=valid[1:]&valid[:-1];speed=np.linalg.norm(np.diff(x,axis=0)*52,axis=-1)
                    motion['methods'][name]=dict(coverage=float(valid.mean()),adjacent_speed_mm_s=stats(speed[pair]),
                        amplitude_p95_p5_mm=[(np.percentile(x[valid[:,j],j],95,axis=0)-np.percentile(x[valid[:,j],j],5,axis=0)).tolist() if valid[:,j].any() else None for j in range(4)])
                motion['warning']='Motion/amplitude on changing coverage, NOT accuracy. No dense3D truth; true action lag not established.';motions.append(motion)
            else:
                for g in manual['frame_groups']:
                    if g['session']!=session or g['frame'] not in frames or cams[held] not in g['cameras']:continue
                    fi=frames.tolist().index(g['frame'])
                    with np.load(BASE/g['normalized_file']) as z:gt=z['uv'][g['cameras'].index(cams[held]),0,14:18]
                    for j,k in enumerate([15,16,17,18]):
                        if not np.isfinite(gt[j]).all():continue
                        r=dict(cohort=cohort,session=session,frame=g['frame'],camera=cams[held],id=k)
                        for name,result in [('framewise',independent),('temporal',temporal)]:
                            x=result['xyz'][fi,j];p=geo.project(x,np.array([held]))[0][0] if np.isfinite(x).all() else np.full(2,np.nan)
                            r[name]=float(np.linalg.norm(p-gt[j])) if np.isfinite(p).all() else None;r[name+'_uv']=p.tolist()
                        old=previous[(cohort,session,g['frame'],cams[held],k)]['joint'];new=r['framewise']
                        parities.append(dict(key=[cohort,session,g['frame'],cams[held],k],both_missing=old is None and new is None,acceptance_equal=(old is None)==(new is None),absolute_error_difference_px=abs(old-new) if old is not None and new is not None else None))
                        allrows.append(r)
            print('VARIANT_DONE',cohort,session,label,'states',maxstates,flush=True)
            (dest/'progress.json').write_text(json.dumps(dict(cohort=cohort,session=session,held=label,observations=len(allrows)),indent=2))
    def metrics(rs,k):
        vals=[r[k] for r in rs if r[k] is not None]
        return dict(coverage=len(vals)/len(rs) if rs else 0,errors=stats(vals),over50px=sum(v>50 for v in vals),over100px=sum(v>100 for v in vals))
    reports={}
    for cohort in ['pilot','stress1']:
        rs=[r for r in allrows if r['cohort']==cohort];both=[r for r in rs if r['framewise'] is not None and r['temporal'] is not None]
        reports[cohort]=dict(n=len(rs),all_available={k:metrics(rs,k) for k in ['framewise','temporal']},
            same_accepted_cohort={k:metrics(both,k) for k in ['framewise','temporal']},
            by_keypoint={str(j):{k:metrics([r for r in rs if r['id']==j],k) for k in ['framewise','temporal']} for j in [15,16,17,18]},
            transitions=dict(improved_over1px=sum(r['temporal']<r['framewise']-1 for r in both),worsened_over1px=sum(r['temporal']>r['framewise']+1 for r in both),
                newly_accepted=sum(r['framewise'] is None and r['temporal'] is not None for r in rs),newly_rejected=sum(r['framewise'] is not None and r['temporal'] is None for r in rs)))
    report=dict(cohorts=reports,motion=motions,sparse_framewise_parity=dict(acceptance_mismatches=sum(not r['acceptance_equal'] for r in parities),
        max_error_difference_px=max((r['absolute_error_difference_px'] for r in parities if r['absolute_error_difference_px'] is not None),default=0)),
        train_allowed=False,warning='Offline hypothesis selection, not3Dtruth. Same-state/no-temporal control. Fixed heuristic costs; no denseGT action-lag validation.')
    (dest/'observations.json').write_text(json.dumps(allrows,indent=2));(dest/'parity.json').write_text(json.dumps(parities,indent=2));(dest/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report['cohorts'],indent=2))
    write_state(stage='temporal_joint_states_evaluated',pid=None,active_run=None,mode='idle_between_heartbeats',job_deadline_utc=None,latest_result=str(dest/'summary.json'),next_action='Review trueLOO temporal identity errors/coverage, sparseparity, motion changes and visual examples; no pseudoGT release.')

if __name__=='__main__':main()
