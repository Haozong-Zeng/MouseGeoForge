"""Frozen bottom-ray consensus. Manual side labels are evaluation-only.

Each API call receives only retained views, making held-view leakage impossible.
No original labels, coordinates, or prior-cycle state are modified.
"""
import datetime as dt
import json
import os
import numpy as np
from scipy.optimize import least_squares
from bottom_anchor_common_20261002 import ROOT, OUT, PRIOR, BASE, budget, write_state
from weighted_framewise_cycle_20261002 import Geometry
from audit_teacher_dlc_predictions_20261001 import stats

GATE = 15.
MARGIN = 2.


def hypotheses(geo, cal, cams, anchor, points, j, flexible):
    a = cams.index('A')
    if not np.isfinite(anchor).all():
        return []
    ray = geo.E[a, :3, :3].T @ np.linalg.solve(np.asarray(cal['A']['K']), np.r_[anchor, 1.])
    ray /= np.linalg.norm(ray)
    origin = geo.centers[a]
    sides = np.array([i for i, c in enumerate(cams) if c != 'A'])
    choices = [0, 1] if flexible else [j]
    candidates = {}

    def assign(depth):
        xyz = origin + depth * ray
        proj, _, z = geo.project(xyz, sides)
        err = np.linalg.norm(proj[:, None] - points[sides], axis=-1)
        err[~np.isfinite(err)] = np.inf
        mapping = np.argmin(err, axis=1) if flexible else np.full(len(sides), j)
        dist = err[np.arange(len(sides)), mapping]
        valid = (z > 0) & (dist < GATE)
        if flexible:
            valid &= np.abs(err[:, 0] - err[:, 1]) > MARGIN
        mapping = np.where(valid, mapping, -1)
        return xyz, mapping, dist, valid

    for vi in sides:
        for k in choices:
            if not np.isfinite(points[vi, k]).all():
                continue
            p = geo.P[vi]
            u, v = points[vi, k]
            mat = np.stack([u * p[2] - p[0], v * p[2] - p[1]])
            w = mat[:, :3] @ ray
            b = mat[:, :3] @ origin + mat[:, 3]
            if w @ w < 1e-12:
                continue
            depth = -float(w @ b) / float(w @ w)
            if depth <= 0:
                continue
            xyz, mapping, dist, valid = assign(depth)
            if valid.sum() < 2:
                continue
            fit_views = sides[valid]
            fit_points = points[fit_views, mapping[valid]]
            result = least_squares(
                lambda x: ((geo.project(origin + x[0] * ray, fit_views)[0] - fit_points) / 4.).ravel(),
                [depth], bounds=(1e-6, np.inf), loss='soft_l1', f_scale=3., max_nfev=80)
            if not result.success:
                continue
            xyz, mapping, dist, valid = assign(result.x[0])
            if valid.sum() < 2:
                continue
            # Require positive depth in every retained camera, not only its supporters.
            if np.any(geo.project(xyz, np.arange(len(cams)))[2] <= 0):
                continue
            rays = xyz - geo.centers[np.r_[a, sides[valid]]]
            rays /= np.linalg.norm(rays, axis=1)[:, None]
            angle = np.degrees(np.arccos(np.clip(np.abs(rays @ rays.T), 0, 1))).max()
            if angle < 3.:
                continue
            key = tuple(mapping.tolist())
            item = dict(xyz=xyz, mapping=mapping, support=int(valid.sum()),
                        cost=float(np.minimum(dist, GATE).mean()), angle=float(angle))
            if key not in candidates or item['cost'] < candidates[key]['cost']:
                candidates[key] = item
    return list(candidates.values())


def fit_pair(cal, cams, anchor, points, flexible):
    """No GT, no omitted camera predictions, no omitted camera metadata as input."""
    assert points.shape == (len(cams), 2, 2)
    fail = dict(accepted=False, reason='insufficient_or_ambiguous_support', xyz=None, mapping=None)
    if not np.isfinite(anchor).all():
        return dict(fail, reason='missing_anchor_pair')
    geo = Geometry(cal, cams)
    candidates = [hypotheses(geo, cal, cams, anchor[j], points, j, flexible) for j in range(2)]
    pairs = []
    for left in candidates[0]:
        for right in candidates[1]:
            shared = (left['mapping'] >= 0) & (right['mapping'] >= 0)
            if np.any(left['mapping'][shared] == right['mapping'][shared]):
                continue
            pairs.append((left, right))
    if not pairs:
        return fail
    pairs.sort(key=lambda p: (-(p[0]['support'] + p[1]['support']), p[0]['cost'] + p[1]['cost']))
    best = pairs[0]
    support = sum(p['support'] for p in best)
    cost = sum(p['cost'] for p in best) / 2.
    if len(pairs) > 1:
        second = pairs[1]
        if sum(p['support'] for p in second) == support and sum(p['cost'] for p in second) / 2. - cost < MARGIN:
            return dict(fail, reason='competing_pair_hypotheses')
    mapping = np.full((len(cams), 2), -1, int)
    sides = [i for i, c in enumerate(cams) if c != 'A']
    mapping[sides] = np.stack([p['mapping'] for p in best], axis=-1)
    mapping[cams.index('A')] = [0, 1]
    return dict(accepted=True, reason='accepted_diagnostic', xyz=np.stack([p['xyz'] for p in best]).tolist(),
                mapping=mapping.tolist(), side_support=[p['support'] for p in best], cost_px=cost)


def tests():
    cams = list('ABCDE')
    cal = {c: dict(K=[[800,0,540],[0,800,600],[0,0,1]], world_to_camera_mm=np.block(
        [[np.eye(3),np.array(t)[:,None]], [np.zeros((1,3)),np.ones((1,1))]]).tolist())
        for c,t in zip(cams, [[0,0,0],[-70,0,0],[70,0,0],[0,-70,0],[0,70,0]])}
    geo = Geometry(cal, cams)
    xyz = np.array([[-25,-15,250],[25,15,260]])
    uv = np.stack([geo.project(x,np.arange(5))[0] for x in xyz],axis=1)
    wrong = uv.copy(); wrong[1] = wrong[1,::-1]
    fixed = fit_pair(cal,cams,uv[0],wrong,True)
    assert fixed['accepted'] and np.allclose(fixed['xyz'],xyz,atol=1e-5)
    assert fixed['mapping'][1] == [1,0]
    exact = fit_pair(cal,cams,uv[0],uv,False)
    assert exact['accepted'] and np.allclose(exact['xyz'],xyz,atol=1e-5)
    keep = [0,1,2,3]; sub = {cams[i]:cal[cams[i]] for i in keep}
    before = fit_pair(sub,[cams[i] for i in keep],uv[0],wrong[keep],True)
    wrong[4] = 1e9
    after = fit_pair(sub,[cams[i] for i in keep],uv[0],wrong[keep],True)
    assert before == after
    assert not fit_pair({c:cal[c] for c in cams[:2]},cams[:2],uv[0],uv[:2],True)['accepted']
    return dict(exact_identity=True,known_swap=True,held_view_invariance=True,one_side_rejected=True)


def main():
    plan,end,check = budget(1200)
    dest = OUT/'multiview_consensus_v1'; dest.mkdir(exist_ok=False)
    setting = dict(tests=tests(),gate_px=GATE,margin_px=MARGIN,minimum_side_support_per_paw=2,
        min_ray_angle_deg=3,anchor_constraint='exact bottom ray; diagnostic, not calibrated human uncertainty',
        identity='joint bilateral candidate exclusivity; missing/ambiguous assignments unknown',
        controls='same fitter with original-ID candidates only vs flexible bilateral candidates',
        held_view_removed_before_all_selection=True,manual_side_GT_evaluation_only=True,
        thresholds_frozen_no_sweep=True,train_allowed=False)
    (dest/'settings.json').write_text(json.dumps(setting,indent=2))
    write_state(stage='multiview_consensus_running',pid=os.getpid(),active_run=dest.name,mode='CPU',
        job_deadline_utc=dt.datetime.fromtimestamp(end,dt.timezone.utc).isoformat(),job_log=None)
    m = json.loads((PRIOR/'lp_context_dataset_v1/manifest.json').read_text())
    grouped = {}
    exclusions = {(r['session'],r['frame']) for r in json.loads((ROOT/plan['exclusion_manifest']).read_text())['cases']}
    exclusions.add(('AZR2L2',584))
    with np.load(PRIOR/'lp_context_seed17_kl/predictions_source_pixels.npz') as z:
        for p,idx in enumerate(z['manifest_indices']):
            e=m['entries'][int(idx)];key=(e['session'],e['frame'])
            assert key not in exclusions and e['session'] in plan['development_sessions']
            grouped.setdefault(key,{})[e['camera']]=dict(entry=e,pred=z['uv_selected'][p,3:].copy(),gt=z['gt'][p,3:].copy())
    rows=[];groups=[]
    prior_rows=json.loads((OUT/'epipolar_pilot_v1/observations.json').read_text())
    previous={(r['session'],r['frame'],r['camera'],r['id']):r for r in prior_rows}
    for (session,frame),items in sorted(grouped.items()):
        check();cams=sorted(items);a=cams.index('A')
        cal=json.loads((BASE/'calibration'/session/'rectified_calibration.json').read_text());geo=Geometry(cal,cams)
        uv=np.stack([items[c]['pred'] for c in cams]);truth=np.stack([items[c]['gt'] for c in cams])
        full={};loo={}
        for source,anchor in [('manual',truth[a]),('predicted',uv[a])]:
            for variant,flexible in [('fixed',False),('flexible',True)]:
                name=source+'_'+variant;full[name]=[];loo[name]={}
                for pair in [[0,1],[2,3]]:
                    check();full[name].append(fit_pair(cal,cams,anchor[pair],uv[:,pair],flexible))
                for held,c in enumerate(cams):
                    if held==a:continue
                    keep=[i for i in range(len(cams)) if i!=held]
                    subcams=[cams[i] for i in keep];subcal={c:cal[c] for c in subcams}
                    loo[name][c]=[fit_pair(subcal,subcams,anchor[pair],uv[keep][:,pair],flexible) for pair in [[0,1],[2,3]]]
        for vi,c in enumerate(cams):
            if vi==a:continue
            for j,kp in enumerate([15,16,17,18]):
                if not np.isfinite(truth[vi,j]).all():continue
                old=previous[(session,frame,c,kp)]
                row=dict(session=session,frame=frame,camera=c,id=kp,gt_uv=truth[vi,j].tolist(),
                    raw_uv=uv[vi,j].tolist(),raw_error=old['raw_error'],source=items[c]['entry']['source'],
                    prior_manual_loo=old['methods']['manual_no_swap']['loo_error'],
                    prior_predicted_loo=old['methods']['predicted_no_swap']['loo_error'],methods={})
                for name in full:
                    result=full[name][j//2];fit=loo[name][c][j//2]
                    mapped=result['mapping'][vi][j%2] if result['accepted'] else -1
                    point=uv[vi,2*(j//2)+mapped] if mapped>=0 else None
                    projected=geo.project(np.array(fit['xyz'][j%2]),np.array([vi]))[0][0] if fit['accepted'] else None
                    row['methods'][name]=dict(mapped_from_id=15+2*(j//2)+mapped if mapped>=0 else None,
                        uv=point.tolist() if point is not None else None,
                        direct_error=float(np.linalg.norm(point-truth[vi,j])) if point is not None else None,
                        loo_uv=projected.tolist() if projected is not None else None,
                        loo_error=float(np.linalg.norm(projected-truth[vi,j])) if projected is not None else None,
                        loo_reason=fit['reason'])
                rows.append(row)
        groups.append(dict(session=session,frame=frame,cameras=cams,full=full,loo=loo))
        (dest/'progress.json').write_text(json.dumps(dict(groups_done=len(groups),side_points=len(rows))))
        print('GROUP_DONE',session,frame,len(groups),flush=True)
    reports={}
    for name in full:
        direct=[r for r in rows if r['methods'][name]['direct_error'] is not None]
        accepted=[r for r in rows if r['methods'][name]['loo_error'] is not None]
        swapped=[r for r in direct if r['methods'][name]['mapped_from_id']!=r['id']]
        reports[name]=dict(direct_n=len(direct),direct_coverage=len(direct)/len(rows),
            direct=stats([r['methods'][name]['direct_error'] for r in direct]),raw_same_cohort=stats([r['raw_error'] for r in direct]),
            swapped_n=len(swapped),swap_improved_gt5=sum(r['raw_error']-r['methods'][name]['direct_error']>5 for r in swapped),
            swap_harmed_gt5=sum(r['raw_error']-r['methods'][name]['direct_error']<-5 for r in swapped),
            loo_n=len(accepted),loo_coverage=len(accepted)/len(rows),loo=stats([r['methods'][name]['loo_error'] for r in accepted]),
            loo_over100=sum(r['methods'][name]['loo_error']>100 for r in accepted))
    paired={}
    for source in ['manual','predicted']:
        names=[source+'_fixed',source+'_flexible']
        common=[r for r in rows if all(r['methods'][n]['loo_error'] is not None for n in names)]
        paired[source]=dict(n=len(common),methods={n:stats([r['methods'][n]['loo_error'] for r in common]) for n in names},
            prior_ungated_same_cohort=stats([r['prior_'+source+'_loo'] for r in common if r['prior_'+source+'_loo'] is not None]))
    report=dict(settings=setting,groups=len(groups),side_points=len(rows),methods=reports,paired=paired,
        warning='Reused development videos, frozen single seed. Human bottom inputs are not independent validation. Coverage selection is not full-cohort improvement.',
        pseudo_gt_released=False,original_labels_modified=False,locked_holdout_opened=False)
    for file,data in [('observations',rows),('groups',groups),('summary',report)]:
        (dest/(file+'.json')).write_text(json.dumps(data,indent=2))
    write_state(stage='multiview_consensus_ready',pid=None,active_run=None,mode='idle_between_heartbeats',job_deadline_utc=None,
        latest_result=str(dest/'summary.json'),next_action='Inspect paired coverage, severe errors and actual visual gains/harms/unknowns before temporal propagation.')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
