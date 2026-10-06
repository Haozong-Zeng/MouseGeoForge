"""Paired checkpoint comparison and severe-error gallery; original data read-only."""
import datetime as dt
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from bottom_anchor_common_20261002 import ROOT,OUT,PRIOR,write_state


def main():
    dest=OUT/'quadruped_stage150_comparison_v1';dest.mkdir(exist_ok=False)
    run=OUT/'superanimal_quadruped_hrnet_v1';replica=OUT/'superanimal_quadruped_hrnet_seed29_v1'
    dm=json.loads((run/'dataset_manifest.json').read_text())
    entries=[dm['entries'][i] for i in dm['validation_indices']];ids=dm['ids_1based']
    models={};summaries={}
    for ep in [110,150,45]:
        folder=OUT/f'quadruped_epoch{ep:03d}_review_v1'
        with np.load(folder/'predictions_source_pixels.npz') as z:
            if models:assert np.allclose(gt,z['gt'],equal_nan=True)
            gt=z['gt'].copy();models[ep]=z['uv'].copy()
        summaries[ep]=json.loads((folder/'summary.json').read_text())['models'][f'Quadruped_epoch{ep}']
    errors={ep:np.linalg.norm(uv-gt,axis=-1) for ep,uv in models.items()}
    severe=[]
    for ep in [110,150]:
        for i,j in np.argwhere(errors[ep]>100):
            e=entries[i]
            severe.append(dict(epoch=ep,session=e['session'],frame=e['frame'],camera=e['camera'],id=ids[j],
                error_source=float(errors[ep][i,j]),image_index=int(i),gt=gt[i,j].tolist(),prediction=models[ep][i,j].tolist()))
    chosen=sorted({r['image_index'] for r in severe})
    dorsal_score=np.nanmean(errors[110][:,1:3],axis=-1)
    chosen.append(int(np.nanargmax(dorsal_score)))
    chosen=list(dict.fromkeys(chosen))
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',22);images=[]
    for i in chosen:
        e=entries[i];assert e['session'] in ['AZR2L2','FR2']
        canvas=Image.new('RGB',(3240,1250),'#181818')
        for col,ep in enumerate([110,150,45]):
            im=Image.open(ROOT/e['source']).convert('RGB');d=ImageDraw.Draw(im)
            for j,k in enumerate(ids):
                if not np.isfinite(gt[i,j]).all():continue
                x,y=gt[i,j];u,v=models[ep][i,j]
                d.ellipse((x-7,y-7,x+7,y+7),outline='#70ff70',width=3)
                d.text((x+8,y-23),str(k),font=font,fill='#70ff70',stroke_width=1,stroke_fill='black')
                d.line((x,y,u,v),fill='#ff77df',width=2)
                d.line((u-7,v-7,u+7,v+7),fill='#ff77df',width=3);d.line((u-7,v+7,u+7,v-7),fill='#ff77df',width=3)
                d.text((u+9,v+3),str(k),font=font,fill='#ff77df',stroke_width=1,stroke_fill='black')
            canvas.paste(im,(1080*col,50))
            ImageDraw.Draw(canvas).text((1080*col+8,10),f'Quadruped epoch{ep} | {e["session"]} f{e["frame"]} cam{e["camera"]}',font=font,fill='white')
        file=f'{e["session"]}_{e["frame"]}_cam{e["camera"]}.jpg';canvas.save(dest/file,quality=95);images.append(file)
    a=json.loads((run/'settings.json').read_text());b=json.loads((replica/'settings.json').read_text())
    for key in ['target_epoch','initial_lr','milestones','gamma','batch_size','input_wh','source_wh','train_images','validation_images']:
        assert a[key]==b[key],key
    ua=dict(a['updates']);ub=dict(b['updates']);ua.pop('train_settings.seed');ub.pop('train_settings.seed');assert ua==ub
    assert a['seed']==17 and b['seed']==29
    other=json.loads((replica/'dataset_manifest.json').read_text())
    for key in ['entries','train_indices','validation_indices','old_indices','context_manifest_indices']:assert dm[key]==other[key]
    first=json.loads((replica/'first_gpu_step.json').read_text());assert first['device']=='cuda:0' and first['optimizer_state_entries']>0
    report=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),completed_epoch=150,canonical_metrics=summaries,
        severe_errors=severe,source_pixel_units=True,no_confidence_filter=True,
        decision='Do notextendseed17to200/300: nativevalidation7.76-7.99inputpxsince100 and45to150pawmean8.20to7.84sourcepx. Keepofficialbest-mAP110asincumbent;150meanissimilarbutsevereerrorsincreased. Preserveboth.',
        next='Repeatidentical150epochQuadrupedtrainingwithseed29toassessrunvariability, thencompareseverecases/pawidentity/6-7dorsalerrors. Samevideo-split; NOTindependentdatasetvalidation. No newpseudoGT or originalannotationedits.',
        replication=dict(run=replica.name,only_seed_changed=True,first_GPU_step=first),
        remaining_bottleneck='Dorsal6/7means22.77/27.47sourcepxat110; substantiallyworsethanpaws. Investigatesemantics/annotationuncertaintyafterteacherreplication, notautomaticrelabeling.',
        prior_STOP_preserved=(PRIOR/'STOP').exists(),locked_holdout_accessed=False,pseudo_gt_released=False)
    (dest/'summary.json').write_text(json.dumps(report,indent=2))
    page='<meta charset="utf-8"><h1>Quadruped 150轮阶段总结</h1><p>左110最佳mAP、中150末轮、右45中期；绿圈人工、粉叉预测。展示110/150超过100原图像素的全部误差所在图片，以及背部点误差最大图片。仅诊断，不修改标签。</p>'
    page+=''.join(f'<h2>{file}</h2><img style="max-width:100%" src="{file}">' for file in images)
    page+='<pre>'+json.dumps(report,indent=2)+'</pre>'; (dest/'index.html').write_text(page)
    export=Path('/mnt/c/Users/55395/Downloads/MouseGeoForge-BottomAnchor-20261002')/dest.name;shutil.copytree(dest,export)
    write_state(latest_decision=str(dest/'summary.json'),latest_stage_comparison=str(dest/'summary.json'),
        latest_training_visual_directory=str(export),convergence_decision=report['decision'],next_action=report['next'])
    print(json.dumps(dict(export=str(export),severe=severe,replication=report['replication']),indent=2))


if __name__=='__main__':main()
