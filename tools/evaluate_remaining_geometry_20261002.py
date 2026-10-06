"""Frozen 34-group development extension, no tuning or independent-test claim."""
import datetime as dt,json,os,time
import numpy as np
from multipeak_geometry_cycle_20261002 import merge,hypotheses,Geometry,ROOT,OUT,BASE
from joint_paw_assignment_cycle_20261002 import assign
from weighted_framewise_cycle_20261002 import CONFIGS
from audit_teacher_dlc_predictions_20261001 import stats
from run_teacher_cycle_dlc_20261001 import write_state

def main():
    plan=json.loads((OUT/'plan.json').read_text());declaration=json.loads((OUT/'remaining_development_geometry_predeclaration.json').read_text())
    deadline=min(time.time()+1200,dt.datetime.fromisoformat(plan['deadline_utc']).timestamp())
    def check():assert time.time()<deadline and not (OUT/'STOP').exists()
    check();dest=OUT/'remaining_geometry_frozen_v1';dest.mkdir(exist_ok=False)
    (dest/'settings.json').write_text(json.dumps(dict(declaration=declaration,
        multipeak_settings=json.loads((OUT/'multipeak_geometry_v1/settings.json').read_text()),
        joint_settings=json.loads((OUT/'joint_paw_assignment_v1/settings.json').read_text()),
        no_parameters_changed=True,train_allowed=False),indent=2))
    write_state(stage='remaining_geometry_evaluation_running',pid=os.getpid(),active_run=dest.name,mode='CPU',job_deadline_utc=dt.datetime.fromtimestamp(deadline,dt.timezone.utc).isoformat())
    m=json.loads((OUT/'dlc_dataset_manifest.json').read_text());groups={(g['session'],g['frame']):g for g in json.loads((BASE/'ingestion_manifest.json').read_text())['frame_groups']}
    peaks=[];teachers=[]
    for seed in [17,31,73]:
        with np.load(OUT/f'paw_heatmap_peaks_v1/seed{seed}.npz') as z:peaks.append(dict(z))
        with np.load(OUT/f'lp_seed{seed}_kl/predictions_source_pixels.npz') as z:teachers.append(dict(z))
    assert all(np.array_equal(p['indices'],peaks[0]['indices']) for p in peaks)
    lookup={}
    for pos,idx in enumerate(peaks[0]['indices']):
        e=m['entries'][int(idx)];assert e['session'] in plan['development_validation_sessions'];lookup[(e['session'],e['frame'],e['camera'])]=(pos,int(idx))
    excluded={(r['session'],r['frame']) for r in json.loads((ROOT/plan['exclusion_manifest']).read_text())['cases']};excluded.add(('AZR2L2',584))
    prior=set()
    for c in ['pilot','stress1']:
        prior.update((r['session'],r['frame']) for r in json.loads((OUT/f'multipeak_geometry_v1/{c}_observations.json').read_text()))
    rows=[];pawjs=[m['ids_1based'].index(k) for k in [15,16,17,18]]
    for group in declaration['groups']:
        check();session=group['session'];frame=group['frame'];key=(session,frame)
        assert key not in excluded and key not in prior and session in plan['development_validation_sessions']
        g=groups[key];cams=g['cameras'];C=len(cams);geo=Geometry(json.loads((BASE/'calibration'/session/'rectified_calibration.json').read_text()),cams)
        candidates=[[] for _ in range(4)];raw=[];conf=[]
        for c in cams:
            pos,idx=lookup[(session,frame,c)]
            for j in range(4):candidates[j].append(merge(np.stack([p['peaks_source_px'][pos,j] for p in peaks]),np.stack([p['relative_scores'][pos,j] for p in peaks])))
            raw.append(np.stack([p['uv'][idx,pawjs] for p in teachers]));conf.append(np.mean([p['confidence'][idx,pawjs] for p in teachers],axis=0))
        raw=np.array(raw);conf=np.array(conf);uv=np.median(raw,axis=1);var=np.var(raw,axis=1)/np.maximum(conf[...,None],1e-12)
        predictions={}
        for vi,c in enumerate(cams):
            check();others=[i for i in range(C) if i!=vi];hs=[];perpaw=[]
            for j in range(4):
                rec={}
                for name,k in [('merged_top1',1),('merged_top3',3)]:
                    x,meta=hypotheses(geo,candidates[j],others,k);assert vi not in meta['available']
                    rec[name]=(x,meta)
                    if k==3:hs.append(meta['hypotheses'])
                x,meta=geo.solve(uv[:,j],var[:,j],conf[:,j],np.zeros(C),others,CONFIGS['weighted_robust_floor4'])
                rec['weighted_robust_floor4']=(x,meta);perpaw.append(rec)
            xyz,meta=assign(hs)
            for j in range(4):perpaw[j]['joint']=(xyz[j],meta)
            predictions[c]=perpaw
        # Only now open manual coordinates to measure predictions already fixed without GT.
        with np.load(BASE/g['normalized_file']) as z:gt=z['uv'][:,0,np.array([15,16,17,18])-1]
        for vi,c in enumerate(cams):
            for j,k in enumerate([15,16,17,18]):
                if not np.isfinite(gt[vi,j]).all():continue
                r=dict(session=session,frame=frame,camera=c,id=k)
                for method,(x,meta) in predictions[c][j].items():
                    p=geo.project(x,np.array([vi]))[0][0] if np.isfinite(x).all() else np.full(2,np.nan)
                    r[method]=float(np.linalg.norm(p-gt[vi,j])) if np.isfinite(p).all() else None
                    r[method+'_uv']=p.tolist();r[method+'_meta']=meta
                others=[i for i in range(C) if i!=vi and np.isfinite(gt[i,j]).all()]
                x,_=geo.solve(gt[:,j],np.ones((C,2))*16,np.ones(C),np.zeros(C),others,CONFIGS['weighted_robust_floor4'])
                r['manual_only_loo']=float(np.linalg.norm(geo.project(x,np.array([vi]))[0][0]-gt[vi,j])) if np.isfinite(x).all() else None
                rows.append(r)
        print('GROUP_DONE',session,frame,len(rows),flush=True)
    methods=['weighted_robust_floor4','merged_top1','merged_top3','joint']
    def metrics(rs,k):
        values=[r[k] for r in rs if r[k] is not None]
        return dict(coverage=len(values)/len(rs) if rs else 0,errors=stats(values),over50px=sum(v>50 for v in values),over100px=sum(v>100 for v in values))
    def compare(rs):return {k:metrics(rs,k) for k in methods}
    common=[r for r in rows if r['merged_top1'] is not None and r['joint'] is not None]
    report=dict(n=len(rows),source_groups=len(declaration['groups']),all_available=compare(rows),
        same_top1_joint_cohort=compare(common),by_keypoint={str(k):compare([r for r in rows if r['id']==k]) for k in [15,16,17,18]},
        by_camera={c:compare([r for r in rows if r['camera']==c]) for c in sorted({r['camera'] for r in rows})},
        by_session={s:compare([r for r in rows if r['session']==s]) for s in sorted({r['session'] for r in rows})},
        manual_only_control=metrics(rows,'manual_only_loo'),train_allowed=False,
        warning='Same two teacher development sources, NOT independent test. Human2D residuals not3D truth; manual-only control neither lower bound nor independent teacher. SEM01excluded before this run; historicalmetrics unchanged.')
    (dest/'observations.json').write_text(json.dumps(rows,indent=2));(dest/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report['all_available'],indent=2))
    write_state(stage='remaining_geometry_evaluation_complete',pid=None,active_run=None,mode='idle_between_heartbeats',job_deadline_utc=None,
        latest_result=str(dest/'summary.json'),next_action='Inspect broader fixed-setting quality/coverage and severe errors before choosing any further teacher or temporal work. No pseudo-GT release.')

if __name__=='__main__':main()
