"""Separate diagnostic joint-paw assignment; unknown allowed; no manual input."""
import datetime as dt,itertools,json,os,time
import numpy as np
from multipeak_geometry_cycle_20261002 import merge,hypotheses,Geometry,ROOT,OUT,BASE
from audit_teacher_dlc_predictions_20261001 import stats
from run_teacher_cycle_dlc_20261001 import write_state

SETTINGS=dict(max_hypotheses_per_paw=3,min_distinct_paw_distance_mm=5,
    score='sum(support*log(10)+heatmap_geometric_score) for chosen hypotheses; unknown score0',
    plausible_plan_margin=float(np.log(3)),allow_unknown=True,
    policy='If any plan within margin changes a paw by>5mm or marks it unknown, that paw stays unknown. No identity swapping of original labels.',
    warning='Paw contacts closer than5mm may be real; rejection is diagnostic uncertainty, not anatomy GT. No calibrated probabilities.',train_allowed=False)

def assign(hypotheses_per_paw):
    alternatives=[[None]+h[:3] for h in hypotheses_per_paw];plans=[]
    for combination in itertools.product(*alternatives):
        chosen=[h for h in combination if h is not None]
        if any(np.linalg.norm(np.array(a['xyz_mm'])-b['xyz_mm'])<5 for a,b in itertools.combinations(chosen,2)):continue
        score=float(sum(h['support']*np.log(10)+h['score'] for h in chosen))
        plans.append((score,combination))
    plans.sort(key=lambda x:-x[0]);best=plans[0];plausible=[p for p in plans if best[0]-p[0]<np.log(3)]
    xyz=np.full((4,3),np.nan);reasons=[]
    for j,h in enumerate(best[1]):
        if h is None:reasons.append('best_unknown');continue
        if any(p[1][j] is None or np.linalg.norm(np.array(p[1][j]['xyz_mm'])-h['xyz_mm'])>5 for p in plausible):
            reasons.append('ambiguous_assignment');continue
        xyz[j]=h['xyz_mm'];reasons.append('accepted')
    return xyz,dict(best_score=best[0],plausible_plans=len(plausible),reasons=reasons,
        best_plan=[h for h in best[1]],plan_count=len(plans))

def selftest():
    def h(x,score=0):return dict(xyz_mm=[x,0,0],support=4,score=score)
    x,m=assign([[h(0)],[h(10)],[h(20)],[h(30)]]);assert np.isfinite(x).all()
    x,m=assign([[h(0)],[h(0)],[h(20)],[h(30)]]);assert np.isnan(x[:2]).all() and np.isfinite(x[2:]).all()
    x,m=assign([[h(0)],[h(0),h(10,-2)],[h(20)],[h(30)]]);assert np.allclose(x[:,0],[0,10,20,30])
    return dict(distinct_points=True,equal_identity_conflict_unknown=True,alternative_restores_unique_assignment=True)

def main():
    tests=selftest();plan=json.loads((OUT/'plan.json').read_text());deadline=min(time.time()+1200,dt.datetime.fromisoformat(plan['deadline_utc']).timestamp())
    def check():assert time.time()<deadline and not (OUT/'STOP').exists()
    check();dest=OUT/'joint_paw_assignment_v1';dest.mkdir(exist_ok=False)
    (dest/'settings.json').write_text(json.dumps(dict(settings=SETTINGS,tests=tests),indent=2))
    write_state(stage='joint_paw_assignment_running',pid=os.getpid(),active_run=dest.name,mode='CPU',job_deadline_utc=dt.datetime.fromtimestamp(deadline,dt.timezone.utc).isoformat())
    m=json.loads((OUT/'dlc_dataset_manifest.json').read_text());groups={(g['session'],g['frame']):g for g in json.loads((BASE/'ingestion_manifest.json').read_text())['frame_groups']}
    peaks=[]
    for seed in [17,31,73]:
        with np.load(OUT/f'paw_heatmap_peaks_v1/seed{seed}.npz') as z:peaks.append(dict(z))
    lookup={}
    for pos,idx in enumerate(peaks[0]['indices']):
        e=m['entries'][int(idx)];assert e['session'] in plan['development_validation_sessions'];lookup[(e['session'],e['frame'],e['camera'])]=pos
    excluded={(r['session'],r['frame']) for r in json.loads((ROOT/plan['exclusion_manifest']).read_text())['cases']}
    cache={};reports={}
    for cohort in ['pilot','stress1']:
        rows=json.loads((OUT/f'multipeak_geometry_v1/{cohort}_observations.json').read_text())
        for r in rows:
            check();session=r['session'];frame=r['frame'];held=r['camera'];assert (session,frame) not in excluded
            ck=(session,frame,held);g=groups[(session,frame)];cams=g['cameras'];vi=cams.index(held)
            if ck not in cache:
                geo=Geometry(json.loads((BASE/'calibration'/session/'rectified_calibration.json').read_text()),cams)
                available=[i for i in range(len(cams)) if i!=vi];hs=[]
                for j in range(4):
                    candidates=[]
                    for c in cams:
                        # No held-camera candidates even assembled for joint inference.
                        if c==held:candidates.append([]);continue
                        pos=lookup[(session,frame,c)]
                        candidates.append(merge(np.stack([p['peaks_source_px'][pos,j] for p in peaks]),np.stack([p['relative_scores'][pos,j] for p in peaks])))
                    _,meta=hypotheses(geo,candidates,available,3);hs.append(meta['hypotheses'])
                x,meta=assign(hs);cache[ck]=(geo,x,meta)
            geo,x,meta=cache[ck];j=r['id']-15
            with np.load(BASE/g['normalized_file']) as z:gt=z['uv'][vi,0,r['id']-1]
            pred=geo.project(x[j],np.array([vi]))[0][0] if np.isfinite(x[j]).all() else np.array([np.nan,np.nan])
            r['joint']=float(np.linalg.norm(pred-gt)) if np.isfinite(pred).all() else None
            r['joint_uv']=pred.tolist();r['joint_meta']=meta
        def metrics(rs,k):
            values=[r[k] for r in rs if r[k] is not None]
            return dict(coverage=len(values)/len(rs) if rs else 0,errors=stats(values),over50px=sum(v>50 for v in values),over100px=sum(v>100 for v in values))
        keys=['merged_top1','merged_top3','joint'];common=[r for r in rows if all(r[k] is not None for k in keys)]
        report=dict(n=len(rows),all_available={k:metrics(rows,k) for k in keys},same_all_three_cohort={k:metrics(common,k) for k in keys},
            by_keypoint={str(j):{k:metrics([r for r in rows if r['id']==j],k) for k in keys} for j in [15,16,17,18]},
            train_allowed=False,warning='Paw-only, frozen same-source development windows; no3D truth or manual-guided hypothesis selection. Unknown/contact rejection reduces coverage.')
        (dest/f'{cohort}_observations.json').write_text(json.dumps(rows,indent=2));reports[cohort]=report
        print(cohort,json.dumps(report['all_available']),flush=True)
    (dest/'summary.json').write_text(json.dumps(reports,indent=2))
    write_state(stage='joint_paw_assignment_evaluated',pid=None,active_run=None,mode='idle_between_heartbeats',job_deadline_utc=None,latest_result=str(dest/'summary.json'),next_action='Review matched-cohort errors and identity ambiguity; do not expand to dense pseudo-labels unless real evidence warrants it.')

if __name__=='__main__':main()
