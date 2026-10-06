"""Evaluation-only identity/candidate ceiling and manual-ray consistency audit.
Oracle matching here is NOT an inference algorithm and is never exported as labels.
"""
import datetime as dt
import json
import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from bottom_anchor_common_20261002 import ROOT,OUT,PRIOR,BASE,budget,write_state
from bottom_anchor_pilot_20261002 import fundamental,distances
from audit_teacher_dlc_predictions_20261001 import stats


def main():
    plan,end,check=budget(600)
    dest=OUT/'anchor_candidate_audit_v1';dest.mkdir(exist_ok=False)
    write_state(stage='anchor_candidate_audit_running',pid=os.getpid(),active_run=dest.name,mode='CPU',
        job_deadline_utc=dt.datetime.fromtimestamp(end,dt.timezone.utc).isoformat(),job_log=None)
    manifest=json.loads((PRIOR/'lp_context_dataset_v1/manifest.json').read_text())
    excluded={(r['session'],r['frame']) for r in json.loads((ROOT/plan['exclusion_manifest']).read_text())['cases']};excluded.add(('AZR2L2',584))
    groups={}
    with np.load(PRIOR/'lp_context_seed17_kl/predictions_source_pixels.npz') as z:
        for p,idx in enumerate(z['manifest_indices']):
            entry=manifest['entries'][int(idx)];key=(entry['session'],entry['frame'])
            assert key not in excluded and entry['session'] in plan['development_sessions']
            groups.setdefault(key,{})[entry['camera']]=dict(entry=entry,pred=z['uv_selected'][p,3:].copy(),gt=z['gt'][p,3:].copy())
    pairs=[];bottom=[];side=[];rays=[]
    for (session,frame),items in sorted(groups.items()):
        check();a=items['A'];cal=json.loads((BASE/'calibration'/session/'rectified_calibration.json').read_text())
        for pair in [[0,1],[2,3]]:
            if not np.isfinite(a['gt'][pair]).all():continue
            same=np.linalg.norm(a['pred'][pair]-a['gt'][pair],axis=-1)
            swapped=np.linalg.norm(a['pred'][pair[::-1]]-a['gt'][pair],axis=-1)
            pairs.append(dict(session=session,frame=frame,ids=[j+15 for j in pair],identity_mean_px=float(same.mean()),
                swapped_mean_px=float(swapped.mean()),swap_gain_px=float(same.mean()-swapped.mean()),
                both_points_improve_gt5=bool(np.all(same-swapped>5)),both_swapped_under15=bool(np.all(swapped<15))))
        for c,item in items.items():
            for j,kp in enumerate([15,16,17,18]):
                if not np.isfinite(item['gt'][j]).all():continue
                error=np.linalg.norm(item['pred']-item['gt'][j],axis=-1);pair=[2*(j//2),2*(j//2)+1]
                row=dict(session=session,frame=frame,camera=c,id=kp,raw_error=float(error[j]),
                    pair_oracle_error=float(min(error[pair])),all4_oracle_error=float(min(error)),
                    pair_best_candidate_id=15+int(pair[np.argmin(error[pair])]),all4_best_candidate_id=15+int(np.argmin(error)))
                (bottom if c=='A' else side).append(row)
                if c!='A' and np.isfinite(a['gt'][j]).all():
                    F=fundamental(cal,'A',c)
                    residuals={str(k+15):float(distances(F,a['gt'][k],item['gt'][j:j+1])[0]) for k in range(4) if np.isfinite(a['gt'][k]).all()}
                    rays.append(dict(session=session,frame=frame,camera=c,id=kp,same_id_sampson_px=residuals[str(kp)],
                        bottom_id_sampson_px=residuals,warning='Evaluation-only epipolar compatibility; a small value does not establish identity/depth or prove either annotation correct.'))
    def metric(rows):
        return dict(n=len(rows),errors={name:stats([r[name] for r in rows]) for name in ['raw_error','pair_oracle_error','all4_oracle_error']},
            within_px={str(t):{name:sum(r[name]<=t for r in rows) for name in ['raw_error','pair_oracle_error','all4_oracle_error']} for t in [10,15,25]},
            severe_raw_over100=sum(r['raw_error']>100 for r in rows),
            severe_recoverable_within15_samepair=sum(r['raw_error']>100 and r['pair_oracle_error']<=15 for r in rows),
            severe_recoverable_within15_only_otherpair=sum(r['raw_error']>100 and r['pair_oracle_error']>15 and r['all4_oracle_error']<=15 for r in rows),
            severe_no_candidate_within15=sum(r['raw_error']>100 and r['all4_oracle_error']>15 for r in rows))
    summary=dict(groups=len(groups),complete_bottom_pairs=len(pairs),bottom=metric(bottom),side=metric(side),
        sessions={s:dict(bottom=metric([r for r in bottom if r['session']==s]),side=metric([r for r in side if r['session']==s])) for s in plan['development_sessions']},
        bottom_pair_swaps_gt5=[r for r in pairs if r['swap_gain_px']>5],
        manual_same_id_sampson=stats([r['same_id_sampson_px'] for r in rays]),
        manual_sampson_over15=sum(r['same_id_sampson_px']>15 for r in rays),
        interpretation='Independent pointwise oracle minima reuse candidates and ignore exclusivity. Optimistic diagnostic lower bounds ONLY, not deployable matching or new GT. Bottom manual used only to evaluate the frozen bottom teacher here.',
        no_original_edits=True,pseudo_gt_released=False,locked_holdout_opened=False,
        caveats=['Reused development videos, one seed, 48 sparse groups, not independent confirmation.',
                 '15px is a descriptive source-resolution threshold, not fitted or calibrated.',
                 'Manual ray residuals can reflect annotation identity, calibration, synchronization or localization errors; audit cannot uniquely determine the cause.'])
    for name,data in [('summary',summary),('bottom_pairs',pairs),('bottom_observations',bottom),('side_observations',side),('manual_ray_audit',rays)]:
        (dest/(name+'.json')).write_text(json.dumps(data,indent=2))
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',18)
    colors=['#ff7878','#60ebff','#fff45a','#8afa79']
    def draw_item(item,lines=None):
        im=Image.open(ROOT/item['entry']['source']).convert('RGB').resize((540,600),Image.Resampling.LANCZOS);d=ImageDraw.Draw(im)
        if lines is not None:
            for j,line in lines:
                a,b,c=line;c/=2
                ends=[]
                if abs(b)>1e-9:
                    ends += [(0,-c/b),(539,-(a*539+c)/b)]
                if abs(a)>1e-9:
                    ends += [(-c/a,0),(-(b*599+c)/a,599)]
                good=[(x,y) for x,y in ends if 0<=x<=539 and 0<=y<=599]
                if len(good)>=2:d.line((*good[0],*good[1]),fill=colors[j],width=1)
        for j,kp in enumerate([15,16,17,18]):
            if np.isfinite(item['gt'][j]).all():
                x,y=item['gt'][j]/2;d.rectangle((x-5,y-5,x+5,y+5),outline='#60ff60',width=2)
                d.text((x+6,y-22),str(kp),font=font,fill='#60ff60',stroke_width=1,stroke_fill='black')
            x,y=item['pred'][j]/2;d.line((x-5,y-5,x+5,y+5),fill='#ff75d5',width=2);d.line((x-5,y+5,x+5,y-5),fill='#ff75d5',width=2)
            d.text((x+6,y+6),str(kp),font=font,fill='#ff75d5',stroke_width=1,stroke_fill='black')
        return im
    selected=[]
    for r in sorted(pairs,key=lambda r:r['swap_gain_px'],reverse=True):
        key=(r['session'],r['frame'])
        if key not in selected:selected.append(key)
        if len(selected)==8:break
    sheet=Image.new('RGB',(2160,1320),'#202020');d=ImageDraw.Draw(sheet)
    for i,key in enumerate(selected):
        im=draw_item(groups[key]['A']);x=(i%4)*540;y=(i//4)*660
        sheet.paste(im,(x,y+60));d.text((x+5,y+3),f'{key[0]} frame{key[1]} camA',font=font,fill='white')
        d.text((x+5,y+27),'Green=manual; pink=frozen teacher',font=font,fill='white')
    sheet.save(dest/'bottom_ranked_identity_audit.jpg',quality=95)
    gallery=['bottom_ranked_identity_audit.jpg']
    for key in [('FR2',1039),('FR2',1104),('AZR2L2',1169)]:
        check();items=groups[key];cams=sorted(items);cal=json.loads((BASE/'calibration'/key[0]/'rectified_calibration.json').read_text())
        sheet=Image.new('RGB',(2160,1320),'#202020');d=ImageDraw.Draw(sheet)
        for i,c in enumerate(cams):
            lines=None
            if c!='A':
                F=fundamental(cal,'A',c)
                lines=[(j,F@np.r_[items['A']['gt'][j],1.]) for j in range(4) if np.isfinite(items['A']['gt'][j]).all()]
            im=draw_item(items[c],lines);x=(i%4)*540;y=(i//4)*660;sheet.paste(im,(x,y+60))
            d.text((x+5,y+3),f'{key[0]} frame{key[1]} cam{c}',font=font,fill='white')
            d.text((x+5,y+27),'Lines: bottom15 red,16 cyan,17 yellow,18 mint',font=font,fill='white')
        file=f'{key[0]}_{key[1]:06d}_allview_manual_rays.jpg';sheet.save(dest/file,quality=95);gallery.append(file)
    missed=[r for r in side if r['raw_error']>100]
    selection=[]
    for category,predicate in [('samepair',lambda r:r['pair_oracle_error']<=15),('otherpair',lambda r:r['pair_oracle_error']>15 and r['all4_oracle_error']<=15),('absent',lambda r:r['all4_oracle_error']>15)]:
        used=set()
        for r in sorted(filter(predicate,missed),key=lambda r:r['raw_error'],reverse=True):
            key=(r['session'],r['frame'],r['camera'])
            if key in used:continue
            used.add(key);selection.append((category,r))
            if len(used)==2:break
    sheet=Image.new('RGB',(540*len(selection),680),'#202020');d=ImageDraw.Draw(sheet)
    for i,(category,r) in enumerate(selection):
        sheet.paste(draw_item(groups[(r['session'],r['frame'])][r['camera']]),(i*540,80))
        d.text((i*540+5,3),f'{category}: {r["session"]} f{r["frame"]} cam{r["camera"]} kp{r["id"]}',font=font,fill='white')
        d.text((i*540+5,28),f'raw {r["raw_error"]:.1f}px; pair oracle {r["pair_oracle_error"]:.1f}px',font=font,fill='white')
        d.text((i*540+5,52),f'all4 oracle {r["all4_oracle_error"]:.1f}px (EVAL ONLY)',font=font,fill='white')
    sheet.save(dest/'side_candidate_ceiling_examples.jpg',quality=95);gallery.append('side_candidate_ceiling_examples.jpg')
    page='<html><meta charset="utf-8"><h1>底视身份与侧视候选可用性审计</h1><p>仅诊断：人工点为绿框，教师预测为粉叉；oracle使用人工答案挑选候选，只是理想下界，不能当算法结果或新标签。各视图线表示人工底视点的极线，不是3D重建。</p>'
    page+='<pre>'+json.dumps(summary,indent=2)+'</pre>'+''.join(f'<h2>{f}</h2><img width="1600" src="{f}">' for f in gallery)+'</html>'
    (dest/'index.html').write_text(page)
    export=Path('/mnt/c/Users/55395/Downloads/MouseGeoForge-BottomAnchor-20261002/anchor_candidate_audit_v1');assert not export.exists();export.mkdir()
    for file in gallery+['index.html','summary.json','manual_ray_audit.json']:shutil.copy2(dest/file,export/file)
    write_state(stage='anchor_candidate_audit_ready',pid=None,active_run=None,mode='idle_between_heartbeats',job_deadline_utc=None,
        latest_result=str(dest/'summary.json'),latest_visual_directory=str(export),next_action='Inspect bottom swaps, severe side candidate absence and manual ray all-view audits; no corrections derived from oracle.')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
