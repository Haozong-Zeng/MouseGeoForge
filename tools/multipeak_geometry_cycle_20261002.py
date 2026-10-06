"""Frozen diagnostic multi-peak geometry; no GT enters candidate selection."""
import datetime as dt,itertools,json,os,time
import numpy as np
from weighted_framewise_cycle_20261002 import Geometry,ROOT,OUT,BASE,CONFIGS
from audit_teacher_dlc_predictions_20261001 import stats
from run_teacher_cycle_dlc_20261001 import write_state

SETTINGS=dict(seed_topk=3,relative_strength_min=.1,merge_radius_px=10,max_merged_candidates=3,
    min_views=3,strict_majority=True,max_reprojection_px=15,min_ray_angle_deg=3,
    competing_cluster_distance_mm=5,minimum_same_support_score_margin=float(np.log(3)),
    score='sum(log(seed-mean relative peak strength)-0.5*(residual/15)^2) on inliers; rank support first, score second. Heuristic NOT probability.',
    variance_px2=16,temporal=False,train_allowed=False)

def merge(peaks,ratios):
    items=sorted([(float(ratios[s,k]),s,k,peaks[s,k]) for s in range(3) for k in range(3)
                  if ratios[s,k]>=.1 and np.isfinite(peaks[s,k]).all()],key=lambda x:(-x[0],x[1],x[2]))
    groups=[]
    for q,s,k,xy in items:
        match=next((g for g in groups if np.linalg.norm(xy-g['anchor'])<=10),None)
        if match is None:match=dict(anchor=xy,items=[]);groups.append(match)
        match['items'].append((q,s,k,xy))
    result=[]
    for g in groups:
        seed_scores=[max([q for q,s,_,_ in g['items'] if s==seed],default=0) for seed in range(3)]
        q=float(np.mean(seed_scores));xy=np.average([a[3] for a in g['items']],axis=0,weights=[a[0] for a in g['items']])
        result.append(dict(uv=xy,score=q,members=[dict(seed=[17,31,73][s],rank=k+1,ratio=q0) for q0,s,k,_ in g['items']]))
    return sorted(result,key=lambda x:-x['score'])[:3]

def hypotheses(geo,candidates,available,kmax):
    available=list(available);eligible=[i for i in available if candidates[i]]
    fail=np.full(3,np.nan);meta=dict(available=available,reason='insufficient_views',hypotheses=[])
    if len(eligible)<3:return fail,meta
    xy=np.full((len(candidates),3,2),np.nan);quality=np.zeros((len(candidates),3))
    for i in eligible:
        for j,item in enumerate(candidates[i][:kmax]):xy[i,j]=item['uv'];quality[i,j]=item['score']
    seen=set();solutions=[]
    for a,b in itertools.combinations(eligible,2):
        for ia,ib in itertools.product(range(min(kmax,len(candidates[a]))),range(min(kmax,len(candidates[b])))):
            P=geo.P[[a,b]];uv=xy[[a,b],[ia,ib]]
            A=np.concatenate([uv[:,0,None]*P[:,2]-P[:,0],uv[:,1,None]*P[:,2]-P[:,1]])
            _,_,vh=np.linalg.svd(A);h=vh[-1]
            if abs(h[3])<1e-12:continue
            x=h[:3]/h[3];proj,_,z=geo.project(x,np.array(eligible))
            distance=np.linalg.norm(proj[:,None]-xy[eligible],axis=-1)
            distance=np.where(np.isfinite(distance),distance,np.inf);choice=distance.argmin(-1)
            residual=distance[np.arange(len(eligible)),choice]
            support=[i for n,i in enumerate(eligible) if residual[n]<=15 and z[n]>0]
            if len(support)<3 or len(support)*2<=len(eligible):continue
            assignment={i:int(choice[n]) for n,i in enumerate(eligible) if i in support}
            key=tuple(sorted(assignment.items()))
            if key in seen:continue
            seen.add(key);observed=np.full((len(candidates),2),np.nan)
            for i,j in assignment.items():observed[i]=xy[i,j]
            x,m=geo.solve(observed,np.ones_like(observed)*16,np.ones(len(candidates)),np.zeros(len(candidates)),support,CONFIGS['weighted_robust_floor4'])
            if not np.isfinite(x).all() or m['angle_deg']<3:continue
            p,_,z=geo.project(x,np.array(support));error=np.linalg.norm(p-observed[support],axis=-1)
            if np.any(z<=0) or np.max(error)>15:continue
            score=float(sum(np.log(max(quality[i,assignment[i]],1e-9))-.5*(e/15)**2 for i,e in zip(support,error)))
            solutions.append(dict(xyz_mm=x.tolist(),support=len(support),inliers=support,
                peak_indices=[assignment[i] for i in support],score=score,median_reprojection_px=float(np.median(error))))
    if not solutions:meta['reason']='no_consensus';return fail,meta
    solutions.sort(key=lambda s:(-s['support'],-s['score']))
    distinct=[]
    for s in solutions:
        if all(np.linalg.norm(np.array(s['xyz_mm'])-d['xyz_mm'])>5 for d in distinct):distinct.append(s)
    meta['hypotheses']=distinct[:8];best=distinct[0]
    if len(distinct)>1 and distinct[1]['support']==best['support'] and best['score']-distinct[1]['score']<np.log(3):
        meta['reason']='ambiguous_identity_geometry';return fail,meta
    meta['reason']='accepted';return np.array(best['xyz_mm']),meta

def selftest():
    cal={c:dict(K=[[800,0,540],[0,800,600],[0,0,1]],world_to_camera_mm=np.block([[np.eye(3),np.array(t)[:,None]],[np.zeros((1,3)),np.ones((1,1))]]).tolist())
         for c,t in zip('ABCD',[[-50,0,0],[0,-50,0],[50,0,0],[0,50,0]])}
    geo=Geometry(cal,list(cal));truth=np.array([6.,9.,250.]);uv=geo.project(truth,np.arange(4))[0]
    candidates=[[dict(uv=p,score=1.)] for p in uv]
    x,_=hypotheses(geo,candidates,[0,1,2],3);assert np.linalg.norm(x-truth)<1e-6
    candidates[3]=[dict(uv=np.array([1e6,1e6]),score=1.)]
    y,_=hypotheses(geo,candidates,[0,1,2],3);assert np.allclose(x,y)
    assert np.isnan(hypotheses(geo,candidates,[0,1],3)[0]).all()
    # Three weak but correct secondary peaks recover inconsistent primary positions.
    for i in range(3):candidates[i]=[dict(uv=uv[i]+np.array([80*(i+1),60*(-1)**i]),score=1.),dict(uv=uv[i],score=.5)]
    x,_=hypotheses(geo,candidates,[0,1,2],3);assert np.linalg.norm(x-truth)<1e-6
    return dict(exact_geometry=True,held_camera_invariance=True,two_view_rejection=True,secondary_peak_recovery=True)

def main():
    tests=selftest();plan=json.loads((OUT/'plan.json').read_text());deadline=min(time.time()+1200,dt.datetime.fromisoformat(plan['deadline_utc']).timestamp())
    def check():assert time.time()<deadline and not (OUT/'STOP').exists()
    check();dest=OUT/'multipeak_geometry_v1';dest.mkdir(exist_ok=False)
    (dest/'settings.json').write_text(json.dumps(dict(settings=SETTINGS,tests=tests),indent=2))
    write_state(stage='multipeak_geometry_running',pid=os.getpid(),active_run=dest.name,mode='CPU',job_deadline_utc=dt.datetime.fromtimestamp(deadline,dt.timezone.utc).isoformat())
    m=json.loads((OUT/'dlc_dataset_manifest.json').read_text());manual=json.loads((BASE/'ingestion_manifest.json').read_text())
    groups={(g['session'],g['frame']):g for g in manual['frame_groups']};peaks=[]
    for seed in [17,31,73]:
        with np.load(OUT/f'paw_heatmap_peaks_v1/seed{seed}.npz') as z:peaks.append(dict(z))
    assert all(np.array_equal(p['indices'],peaks[0]['indices']) for p in peaks)
    entrymap={}
    for pos,idx in enumerate(peaks[0]['indices']):
        e=m['entries'][int(idx)];assert e['session'] in plan['development_validation_sessions']
        entrymap[(e['session'],e['frame'],e['camera'])]=pos
    excluded={(r['session'],r['frame']) for r in json.loads((ROOT/plan['exclusion_manifest']).read_text())['cases']}
    cache={};all_reports={}
    for cohort,previous in [('pilot','consensus_pilot_v1_final'),('stress1','consensus_stress1_v1')]:
        prior=json.loads((OUT/previous/'observations.json').read_text());rows=[]
        for r0 in prior:
            if r0['id'] not in [15,16,17,18]:continue
            check();r=dict(r0);session=r['session'];frame=r['frame'];assert (session,frame) not in excluded
            g=groups[(session,frame)];cams=g['cameras'];vi=cams.index(r['camera']);j=r['id']-15
            ck=(session,frame,r['id'])
            if ck not in cache:
                cal=json.loads((BASE/'calibration'/session/'rectified_calibration.json').read_text());geo=Geometry(cal,cams);candidates=[]
                for c in cams:
                    assert (session,frame,c) in entrymap
                    pos=entrymap[(session,frame,c)]
                    candidates.append(merge(np.stack([p['peaks_source_px'][pos,j] for p in peaks]),np.stack([p['relative_scores'][pos,j] for p in peaks])))
                cache[ck]=(geo,candidates)
            geo,candidates=cache[ck];others=[i for i in range(len(cams)) if i!=vi]
            with np.load(BASE/g['normalized_file']) as z:gt=z['uv'][vi,0,r['id']-1]
            assert np.isfinite(gt).all()
            for name,kmax in [('merged_top1',1),('merged_top3',3)]:
                x,meta=hypotheses(geo,candidates,others,kmax)
                assert vi not in meta['available']
                prediction=geo.project(x,np.array([vi]))[0][0] if np.isfinite(x).all() else np.array([np.nan,np.nan])
                r[name]=float(np.linalg.norm(prediction-gt)) if np.isfinite(prediction).all() else None
                r[name+'_meta']=meta;r[name+'_uv']=prediction.tolist()
            rows.append(r)
        names=['weighted_robust_floor4','consensus','merged_top1','merged_top3']
        def summarize(rs,key):
            v=[r[key] for r in rs if r[key] is not None]
            return dict(errors=stats(v),coverage=len(v)/len(rs) if rs else 0,over50px=sum(x>50 for x in v),over100px=sum(x>100 for x in v))
        common=[r for r in rows if r['merged_top1'] is not None and r['merged_top3'] is not None]
        report=dict(n=len(rows),source_groups=len({(r['session'],r['frame']) for r in rows}),
            all_available={k:summarize(rows,k) for k in names},same_top1_top3_cohort={k:summarize(common,k) for k in names},
            by_keypoint={str(k):{name:summarize([r for r in rows if r['id']==k],name) for name in names} for k in [15,16,17,18]},
            reasons={name:{reason:sum(r[name+'_meta']['reason']==reason for r in rows) for reason in sorted({r[name+'_meta']['reason'] for r in rows})} for name in ['merged_top1','merged_top3']},
            train_allowed=False,warning='Actual candidate selection uses other cameras only, no manual coordinates. All thresholds heuristic; no3D truth. Paw-only metrics not comparable to previous all-body metrics.')
        (dest/f'{cohort}_observations.json').write_text(json.dumps(rows,indent=2));all_reports[cohort]=report
        print(cohort,json.dumps(report['all_available']),flush=True)
    (dest/'summary.json').write_text(json.dumps(all_reports,indent=2))
    write_state(stage='multipeak_geometry_evaluated',pid=None,active_run=None,mode='idle_between_heartbeats',job_deadline_utc=None,latest_result=str(dest/'summary.json'),next_action='Inspect trueLOO multi-peak accuracy vs coverage and ambiguity before continuous inference. No GT release.')

if __name__=='__main__':main()
