"""CPU-only paired teacher evaluation. No labels, checkpoints or training mutation."""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bottom_anchor_common_20261002 import ROOT, OUT, PRIOR, budget, write_state
from audit_teacher_dlc_predictions_20261001 import stats


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def metrics(uv, gt, entries, ids, scale):
    err = np.linalg.norm(uv-gt, axis=-1)
    ei = np.linalg.norm((uv-gt)*scale, axis=-1)
    def group(mask):
        a, b = err[mask], ei[mask]
        return dict(source=stats(a.ravel()), input=stats(b.ravel()),
                    paws_source=stats(a[:, 3:].ravel()), paws_input=stats(b[:, 3:].ravel()),
                    over100=int((a>100).sum()), paws_over100=int((a[:,3:]>100).sum()))
    result = dict(all=group(np.ones(len(entries), bool)))
    result['cameras'] = {c:group(np.array([e['camera']==c for e in entries])) for c in sorted({e['camera'] for e in entries})}
    result['sessions'] = {c:group(np.array([e['session']==c for e in entries])) for c in sorted({e['session'] for e in entries})}
    result['bottom'] = group(np.array([e['camera']=='A' for e in entries]))
    result['side'] = group(np.array([e['camera']!='A' for e in entries]))
    result['keypoints'] = {str(k):dict(source=stats(err[:,j]), input=stats(ei[:,j])) for j,k in enumerate(ids)}
    # GT-assisted pair diagnostic only; never changes predictions or feeds training.
    result['swap_diagnostic'] = {}
    for name, pair in [('fore', [3,4]), ('hind',[5,6])]:
        p, t = uv[:,pair], gt[:,pair]
        finite = np.isfinite(t).all((1,2))
        direct = np.linalg.norm(p-t,axis=-1).mean(-1)
        swapped = np.linalg.norm(p-t[:,::-1],axis=-1).mean(-1)
        result['swap_diagnostic'][name] = dict(pairs=int(finite.sum()),
            swap_improves_over5=int((finite & (direct-swapped>5)).sum()),
            swapped_both_within15=int((finite & (direct-swapped>5) & (np.linalg.norm(p-t[:,::-1],axis=-1).max(-1)<15)).sum()))
    return result


def main(dlc200=False):
    plan,_,check = budget(600)
    dest = OUT/('extended_teacher_review_v2' if dlc200 else 'extended_teacher_review_v1');dest.mkdir(exist_ok=False)
    dr = OUT/('dlc_150to200_v1' if dlc200 else 'dlc_75to150_v1'); lr = OUT/'lp_context_30to60_v2'
    dm = json.loads((dr/'dataset_manifest.json').read_text())
    vi = np.array(dm['validation_indices']); entries=[dm['entries'][i] for i in vi]
    assert len(entries)==281 and all(e['session'] in plan['development_sessions'] for e in entries)
    A=np.array(dm['affine_source_to_input']); scale=np.diag(A)[:2]; ids=dm['ids_1based']
    project=Path(dm['project_config']).parent
    h5=list((project/'evaluation-results-pytorch').rglob('*snapshot_best-*.h5'));assert len(h5)==1
    best_epoch=int(h5[0].stem.split('snapshot_best-')[-1])
    new_name=f'DLC{200 if dlc200 else 150}_best{best_epoch}'
    truth=pd.read_hdf(project/'labeled-data/camA/CollectedData_manualv3.h5')
    pred=pd.read_hdf(h5[0]);assert set(truth.index)==set(pred.index)
    pred=pred.reindex(truth.index)
    uv=np.stack([pred.xs(n,level='bodyparts',axis=1).droplevel(['scorer','individuals'],axis=1)[['x','y']].values for n in dm['names']],axis=1)
    gt=(truth.values.reshape(-1,7,2)-A[:2,2])/scale
    uv=(uv-A[:2,2])/scale
    with np.load(PRIOR/'dlc_extension_review/predictions_source_pixels.npz') as z:
        old=z['uv'][dm['old_indices']].copy()
        assert np.allclose(gt,z['gt'][dm['old_indices']],equal_nan=True,atol=.002)
    gt=gt[vi]
    models={'DLC70':old[vi], new_name:uv[vi]}
    if dlc200:
        with np.load(OUT/'extended_teacher_review_v1/paired_predictions.npz') as z:
            assert np.allclose(z['gt'],gt,equal_nan=True)
            models['DLC150_best135']=z['DLC150_best135'].copy()
    with np.load(lr/'predictions_source_pixels.npz') as z:
        assert np.array_equal(z['context_manifest_indices'],np.array(dm['context_manifest_indices'])[vi])
        assert np.allclose(z['gt'],gt,equal_nan=True,atol=.002)
        models.update(LP30_best24=z['uv_old'].copy(),LP60_best45=z['uv'].copy())
    assert all(np.isfinite(v).all() for v in models.values())
    report=dict(validation_images=len(vi),observed_points=int(np.isfinite(gt).all(-1).sum()),
        source_wh=[1080,1200],input_wh=[704,768],confidence_filter=False,
        caveats=['Reused development sessions, not independent holdout. Single seed.',
                 'LP native validation selection and exported SF/MF-confidence selection are different metrics.',
                 'Swap counts are GT-assisted diagnostics, NOT corrections or new predictions.'],
        models={n:metrics(v,gt,entries,ids,scale) for n,v in models.items()})
    traini=np.array(dm['train_indices'])
    fullgt=(truth.values.reshape(-1,7,2)-A[:2,2])/scale
    report['DLC_train_separate']=metrics(uv[traini],fullgt[traini],[dm['entries'][i] for i in traini],ids,scale)
    (dest/'summary.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(dest/'paired_predictions.npz',gt=gt,**models,ids=ids,
                        context_manifest_indices=np.array(dm['context_manifest_indices'])[vi])
    dlcrows=rows(dr/'epochs.jsonl')
    if dlc200:dlcrows=rows(OUT/'dlc_75to150_v1/epochs.jsonl')+dlcrows
    dlc=[r for r in dlcrows if r['mode']=='eval']
    lp=rows(lr/'epochs.jsonl')
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    axes[0,0].plot([r['epoch'] for r in dlc],[r['metrics']['metrics/test.rmse'] for r in dlc],'o-')
    axes[0,0].set_title('DLC native validation error (input pixels)')
    axes[0,1].plot([r['epoch'] for r in dlc],[r['metrics']['metrics/test.mAP'] for r in dlc],'o-')
    axes[0,1].set_title('DLC validation mAP')
    axes[1,0].plot([r['epoch'] for r in lp],[r['metrics']['val_supervised_rmse'] for r in lp],'o-')
    axes[1,0].set_title('LP native validation RMSE (not exported EPE)')
    for key in ['train_supervised_loss','val_supervised_loss']:
        axes[1,1].plot([r['epoch'] for r in lp],[r['metrics'].get(key,float('nan')) for r in lp],label=key)
    axes[1,1].set_title('LP loss');axes[1,1].legend();axes[1,1].set_yscale('log')
    for ax in axes.ravel():ax.set_xlabel('total epochs');ax.grid(alpha=.3)
    fig.savefig(dest/'learning_curves.png',dpi=140);plt.close(fig)
    # Explicitly choose gains, regressions and bottom/side failures; no representativeness claim.
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',23)
    selected=[]
    for prefix, before, after in [('DLC','DLC150_best135' if dlc200 else 'DLC70',new_name),('LP','LP30_best24','LP60_best45')]:
        eb=np.linalg.norm(models[before]-gt,axis=-1); ea=np.linalg.norm(models[after]-gt,axis=-1)
        delta=np.nanmean(ea-eb,axis=1)
        choices=[('gain',int(np.argmin(delta))),('regression',int(np.argmax(delta)))]
        for label,mask in [('bottom',[e['camera']=='A' for e in entries]),('side',[e['camera']!='A' for e in entries])]:
            idx=np.flatnonzero(mask);choices.append((label+'_paw_failure',int(idx[np.nanargmax(np.nanmean(ea[idx,3:],axis=1))])))
        for category,i in choices:
            check();e=entries[i]; panels=[]
            for name in [before,after]:
                im=Image.open(ROOT/e['source']).convert('RGB');assert im.size==(1080,1200)
                draw=ImageDraw.Draw(im)
                for j,k in enumerate(ids):
                    if not np.isfinite(gt[i,j]).all():continue
                    x,y=gt[i,j];u,v=models[name][i,j]
                    draw.ellipse((x-7,y-7,x+7,y+7),outline='#70ff70',width=3)
                    draw.text((x+9,y-24),str(k),font=font,fill='#70ff70',stroke_width=1,stroke_fill='black')
                    draw.line((x,y,u,v),fill='#ff77df',width=2)
                    draw.line((u-7,v-7,u+7,v+7),fill='#ff77df',width=3)
                    draw.line((u-7,v+7,u+7,v-7),fill='#ff77df',width=3)
                    draw.text((u+8,v+3),str(k),font=font,fill='#ff77df',stroke_width=1,stroke_fill='black')
                panel=Image.new('RGB',(1080,1250),'#161616');panel.paste(im,(0,50))
                ImageDraw.Draw(panel).text((10,10),f'{name} | {e["session"]} f{e["frame"]} cam{e["camera"]} | {category}',font=font,fill='white')
                panels.append(panel)
            canvas=Image.new('RGB',(2160,1250));canvas.paste(panels[0],(0,0));canvas.paste(panels[1],(1080,0))
            filename=f'{prefix}_{category}_{e["session"]}_{e["frame"]}_cam{e["camera"]}.jpg'
            canvas.save(dest/filename,quality=95)
            selected.append(dict(file=filename,model=prefix,category=category,entry=e,mean_delta=float(delta[i])))
    (dest/'visual_selection.json').write_text(json.dumps(selected,indent=2))
    page='<meta charset="utf-8"><h1>教师延长训练：同281张开发图对照</h1><p>绿圈：人工标注；粉叉：预测。左右为延长前/后。展示最大改善、最大退步、底视与侧视四足最差样本，非随机代表样本；不按置信度删点。</p><img width="1200" src="learning_curves.png">'
    page+=''.join(f'<h2>{r["file"]}</h2><img style="max-width:100%" src="{r["file"]}">' for r in selected)
    page+='<pre>'+json.dumps(report,indent=2)+'</pre>'
    (dest/'index.html').write_text(page)
    export=Path('/mnt/c/Users/55395/Downloads/MouseGeoForge-BottomAnchor-20261002')/dest.name
    shutil.copytree(dest,export)
    write_state(latest_paired_review=str(dest/'summary.json'),latest_training_visual_directory=str(export))
    print(json.dumps(dict(export=str(export),models={n:r['all'] for n,r in report['models'].items()}),indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--record-decision',action='store_true');parser.add_argument('--dlc200',action='store_true')
    args=parser.parse_args()
    if args.record_decision:
        decision=OUT/'heartbeat_20261002_1005_decision.json'
        assert decision.is_file()
        a=json.loads((OUT/'dlc_75to150_v1/dataset_manifest.json').read_text())
        b=json.loads((OUT/'dlc_150to200_v1/dataset_manifest.json').read_text())
        for key in ['entries','old_indices','context_manifest_indices','train_indices','validation_indices']:
            assert a[key]==b[key],key
        write_state(latest_decision=str(decision),
            next_action='Finish active DLC150to200_v1 then train audited official SuperAnimal-Quadruped HRNet. LP60 completed without sustained gain; preserve oldbest24, do not restart blindly. See latest_decision and superanimal_followup.json.',
            convergence_decision='DLC still improves; extending to200 at current LR. LP30to60 no sustained gain, pause extensions. Quadruped weights and CPU preflight verified.',
            global_step=None,batches_seen=None)
        print('Decision recorded; DLC150 and200 manifests exactly equal; live PID/epoch/lock untouched.')
    else:main(args.dlc200)
