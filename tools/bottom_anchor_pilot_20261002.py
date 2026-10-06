"""Explicit paw-ID permutation, with human-bottom inputs separated from evaluation.
No original coordinates are edited. Epipolar-only pilot is intentionally simple.
"""
import datetime as dt
import json
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from weighted_framewise_cycle_20261002 import Geometry
from audit_teacher_dlc_predictions_20261001 import stats
from bottom_anchor_common_20261002 import ROOT,OUT,PRIOR,BASE,write_state,budget


def fundamental(cal,a,b):
    ea=np.asarray(cal[a]['world_to_camera_mm']); eb=np.asarray(cal[b]['world_to_camera_mm'])
    R=eb[:3,:3]@ea[:3,:3].T; t=eb[:3,3]-R@ea[:3,3]
    tx=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0.]])
    F=np.linalg.inv(cal[b]['K']).T@tx@R@np.linalg.inv(cal[a]['K'])
    return F/np.linalg.norm(F)


def distances(F,anchor,points):
    a=np.r_[anchor,1.]; b=np.column_stack([points,np.ones(len(points))])
    la=F@a; lb=b@F
    numerator=np.abs(b@la)
    return numerator/np.sqrt(la[:2]@la[:2]+np.square(lb[:,:2]).sum(-1)+1e-12)


def reassign(anchor,points,F,conservative):
    mapping=np.arange(4);meta=[]
    for pair in [(0,1),(2,3)]:
        a,b=pair
        if not np.isfinite(anchor[[a,b]]).all():
            meta.append(dict(pair=list(pair),swapped=False,reason='missing_anchor_pair'));continue
        cost=np.stack([distances(F,anchor[j],points[[a,b]]) for j in pair])
        identity=float((cost[0,0]+cost[1,1])/2); swapped=float((cost[0,1]+cost[1,0])/2)
        margin=identity-swapped
        choose=margin>1e-6 if not conservative else margin>2. and swapped<15.
        if choose:mapping[[a,b]]=[b,a]
        meta.append(dict(pair=list(pair),identity_cost_px=identity,swapped_cost_px=swapped,margin_px=margin,
                         swapped=bool(choose),reason='swap' if choose else 'retain_ambiguous_or_identity'))
    return points[mapping].copy(),mapping,meta


def tests():
    cal={c:dict(K=[[800,0,540],[0,800,600],[0,0,1]],world_to_camera_mm=np.block([[np.eye(3),np.array(t)[:,None]],[np.zeros((1,3)),np.ones((1,1))]]).tolist()) for c,t in zip('ABC',[[-50,0,0],[50,0,0],[0,-50,0]])}
    geo=Geometry(cal,list(cal));xyz=np.array([[-15,-30,250],[15,-10,250],[-15,10,250],[15,30,250]])
    uv=np.stack([geo.project(x,np.arange(3))[0] for x in xyz],axis=1)
    F=fundamental(cal,'A','B')
    assert max(distances(F,uv[0,j],uv[1,j:j+1])[0] for j in range(4))<1e-6
    fixed,perm,_=reassign(uv[0],uv[1][[1,0,3,2]],F,True)
    assert np.allclose(fixed,uv[1]) and perm.tolist()==[1,0,3,2]
    _,p,_=reassign(uv[0],uv[1],F,True);assert p.tolist()==[0,1,2,3]
    missing=uv[0].copy();missing[0]=np.nan
    _,p,meta=reassign(missing,uv[1][[1,0,3,2]],F,True);assert p[:2].tolist()==[0,1] and meta[0]['reason']=='missing_anchor_pair'
    tie=uv[0].copy();tie[1]=tie[0]
    _,p,_=reassign(tie,uv[1],F,True);assert p[:2].tolist()==[0,1]
    return dict(epipolar_exact=True,known_bilateral_swap_recovers=True,identity_unchanged=True,missing_pair_nochange=True,tie_nochange=True)


def main():
    plan,end,check=budget(1200);test=tests()
    dest=OUT/'epipolar_pilot_v1';dest.mkdir(exist_ok=False)
    settings=dict(tests=test,teacher='native LP MHCRNN seed17 temporalON, frozen bestepoch24',
        anchor='camA original manual labels vs same teacher camA predictions; manual anchor NOT evaluation',
        states=['identity','frontswap','hindswap','bothswap'],forehind_exchange=False,
        score='pairmean Sampson distance in originalsourcepixels',ungated='swap iff cost reduction>1e-6px',
        conservative='swap iff pairmean reduction>2px AND swappedcost<15px; otherwise unchanged with ambiguity metadata',
        geometry=dict(floor_px=4,loss='soft_l1',f_scale=3,gate=False),
        caveat='Frozen diagnostic heuristics, not calibrated thresholds. Retain-ambiguous is not certification.',
        held_side_view_absent_from_reassignment_of_other_views_and_3D_fit=True,side_manual_labels_evaluation_only=True,
        no_original_edits=True,train_allowed=False)
    (dest/'settings.json').write_text(json.dumps(settings,indent=2))
    m=json.loads((PRIOR/'lp_context_dataset_v1/manifest.json').read_text())
    with np.load(PRIOR/'lp_context_seed17_kl/predictions_source_pixels.npz') as z:
        pred=z['uv_selected'][:,3:].copy(); gt=z['gt'][:,3:].copy();ix=z['manifest_indices'].copy()
    grouped={}
    excluded={(r['session'],r['frame']) for r in json.loads((ROOT/plan['exclusion_manifest']).read_text())['cases']};excluded.add(('AZR2L2',584))
    for pos,idx in enumerate(ix):
        e=m['entries'][int(idx)];key=(e['session'],e['frame'])
        assert key not in excluded and e['session'] in plan['development_sessions']
        grouped.setdefault(key,{})[e['camera']]=dict(entry=e,pred=pred[pos],gt=gt[pos])
    write_state(stage='epipolar_anchor_pilot_running',pid=os.getpid(),active_run=dest.name,mode='CPU',
        job_deadline_utc=dt.datetime.fromtimestamp(end,dt.timezone.utc).isoformat(),job_log=str(OUT/'epipolar_pilot_v1.log'))
    rows=[];groups_out=[];anchor_groups=[]
    for (session,frame),items in sorted(grouped.items()):
        check()
        if 'A' not in items:continue
        cams=sorted(items);a=cams.index('A'); C=len(cams)
        cal=json.loads((BASE/'calibration'/session/'rectified_calibration.json').read_text());geo=Geometry(cal,cams)
        uv=np.stack([items[c]['pred'] for c in cams]);truth=np.stack([items[c]['gt'] for c in cams])
        anchors={'manual':truth[a],'predicted':uv[a]}; versions={};maps={};metadata={}
        for source,anchor in anchors.items():
            for variant in ['no_swap','argmin','conservative']:
                name=source+'_'+variant; versions[name]=uv.copy(); versions[name][a]=anchor
                maps[name]=np.tile(np.arange(4),(C,1));metadata[name]={}
                for vi,c in enumerate(cams):
                    if c=='A' or variant=='no_swap':continue
                    versions[name][vi],maps[name][vi],metadata[name][c]=reassign(anchor,uv[vi],fundamental(cal,'A',c),variant=='conservative')
        # Held-view invariance check: altering its coordinates cannot alter any other camera assignment.
        for held in [i for i in range(C) if i!=a][:1]:
            perturbed=uv.copy();perturbed[held]=1e6
            for vi,c in enumerate(cams):
                if vi in [a,held]:continue
                changed,_,_=reassign(truth[a],perturbed[vi],fundamental(cal,'A',c),True)
                assert np.array_equal(changed,versions['manual_conservative'][vi])
        for vi,c in enumerate(cams):
            if vi==a:continue
            for j,kp in enumerate([15,16,17,18]):
                if not np.isfinite(truth[vi,j]).all():continue
                row=dict(session=session,frame=frame,camera=c,id=kp,gt_uv=truth[vi,j].tolist(),source=items[c]['entry']['source'],anchor_uv=truth[a,j].tolist(),
                    anchor_available=bool(np.isfinite(truth[a,j]).all()),raw_uv=uv[vi,j].tolist(),raw_error=float(np.linalg.norm(uv[vi,j]-truth[vi,j])),methods={})
                for name,arr in versions.items():
                    check();other=[i for i in range(C) if i!=vi and np.isfinite(arr[i,j]).all()]
                    assert vi not in other
                    x,meta=geo.solve(arr[:,j],np.full((C,2),16.),np.ones(C),np.zeros(C),other,dict(floor=4.,loss='soft_l1',gate=False))
                    projected=geo.project(x,np.array([vi]))[0][0] if np.isfinite(x).all() else np.full(2,np.nan)
                    row['methods'][name]=dict(uv=arr[vi,j].tolist(),mapped_from_id=int(maps[name][vi,j]+15),
                        direct_error=float(np.linalg.norm(arr[vi,j]-truth[vi,j])),
                        loo_uv=projected.tolist(),loo_error=float(np.linalg.norm(projected-truth[vi,j])) if np.isfinite(projected).all() else None,
                        fit_views=[cams[i] for i in other],fit_meta=meta)
                rows.append(row)
        groups_out.append(dict(session=session,frame=frame,cameras=cams,metadata=metadata,
            anchor_complete_pairs=[bool(np.isfinite(truth[a,pair]).all()) for pair in [[0,1],[2,3]]]))
        anchor_groups.append((session,frame,items['A']))
        (dest/'progress.json').write_text(json.dumps(dict(groups_done=len(groups_out),side_manual_points=len(rows)),indent=2))
    names=list(rows[0]['methods'])
    common=[r for r in rows if r['anchor_available'] and all(r['methods'][n]['loo_error'] is not None for n in names)]
    reports={}
    for name in names:
        swapped=[r for r in rows if r['methods'][name]['mapped_from_id']!=r['id']]
        reports[name]=dict(side2d=stats([r['methods'][name]['direct_error'] for r in rows]),
            side2d_over100=sum(r['methods'][name]['direct_error']>100 for r in rows),
            swaps_on_labeled_points=len(swapped),swap_improved_gt5=sum(r['raw_error']-r['methods'][name]['direct_error']>5 for r in swapped),
            swap_harmed_gt5=sum(r['raw_error']-r['methods'][name]['direct_error']<-5 for r in swapped),
            common_loo=stats([r['methods'][name]['loo_error'] for r in common]),
            common_loo_over100=sum(r['methods'][name]['loo_error']>100 for r in common))
    report=dict(settings=settings,groups=len(groups_out),side_manual_points=len(rows),common_loo_n=len(common),methods=reports,
        complete_anchor_pairs=sum(sum(g['anchor_complete_pairs']) for g in groups_out),
        original_labels_modified=False,locked_holdout_opened=False,pseudo_gt_released=False,
        warning='Manualbottom anchors are INPUT, sideGT onlyevaluation. Compare manual_conservative with manual_no_swap to isolate assignment. Developmentvideos reused, no independenttest.')
    (dest/'observations.json').write_text(json.dumps(rows,indent=2));(dest/'groups.json').write_text(json.dumps(groups_out,indent=2));(dest/'summary.json').write_text(json.dumps(report,indent=2))
    # Actual bottom images for human/assistant anatomical audit, not synthetic illustrations.
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',22)
    selection=np.linspace(0,len(anchor_groups)-1,12,dtype=int)
    sheet=Image.new('RGB',(540*4,640*3),'#202020')
    for cell,index in enumerate(selection):
        session,frame,item=anchor_groups[index];im=Image.open(ROOT/item['entry']['source']).convert('RGB');draw=ImageDraw.Draw(im)
        for j,kp in enumerate([15,16,17,18]):
            if not np.isfinite(item['gt'][j]).all():continue
            x,y=item['gt'][j];color=['#ff7777','#55eaff','#fff04a','#7fff70'][j]
            draw.ellipse((x-9,y-9,x+9,y+9),outline=color,width=4);draw.text((x+12,y-30),str(kp),font=font,fill=color,stroke_width=2,stroke_fill='black')
        im=im.resize((540,600),Image.Resampling.LANCZOS);cx=(cell%4)*540;cy=(cell//4)*640
        sheet.paste(im,(cx,cy+40));ImageDraw.Draw(sheet).text((cx+6,cy+6),f'{session} frame{frame} camA HUMAN',font=font,fill='white')
    sheet.save(dest/'bottom_manual_audit.jpg',quality=95)
    write_state(stage='epipolar_anchor_pilot_ready',pid=None,active_run=None,mode='idle_between_heartbeats',job_deadline_utc=None,
        latest_result=str(dest/'summary.json'),next_action='Visually inspect bottommanualaudit and rank gains/harms before interpreting permutation results. No label release.')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
