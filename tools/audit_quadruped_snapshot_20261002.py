"""CPU-only immutable checkpoint audit alongside active single-GPU training."""
import argparse
import datetime as dt
import hashlib
import json
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['DLClight']='True'
from pathlib import Path
import shutil
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from bottom_anchor_common_20261002 import ROOT,OUT,budget,write_state
from audit_extended_teachers_20261002 import metrics


def main(epoch,from_evaluation=False,run_name='superanimal_quadruped_hrnet_v1'):
    plan,_,check=budget(1200)
    torch.set_num_threads(4)
    from deeplabcut.pose_estimation_pytorch.models import PoseModel
    assert run_name in ('superanimal_quadruped_hrnet_v1','superanimal_quadruped_hrnet_seed29_v1')
    run=OUT/run_name
    prefix='quadruped_seed29' if run_name.endswith('seed29_v1') else 'quadruped'
    dest=OUT/f'{prefix}_epoch{epoch:03d}_review_v1'
    assert not dest.exists() or not any(dest.iterdir()),'Never overwrite an existing audit'
    dest.mkdir(exist_ok=True)
    paths=list((run/'project/dlc-models-pytorch').rglob(f'snapshot-{epoch:03d}.pt'))
    if not paths:paths=list((run/'project/dlc-models-pytorch').rglob(f'snapshot-best-{epoch:03d}.pt'))
    assert len(paths)==1
    snapshot=paths[0];digest=hashlib.sha256(snapshot.read_bytes()).hexdigest()
    checkpoint=torch.load(snapshot,map_location='cpu',weights_only=False)
    assert checkpoint['metadata']['epoch']==epoch
    cfg=json.loads((OUT/'superanimal_quadruped_preparation_v1/model_config.json').read_text())
    if not from_evaluation:
        model=PoseModel.build(cfg,pretrained_backbone=False)
        model.load_state_dict(checkpoint['model'],strict=True);model.cpu().eval()
    assert not torch.cuda.is_available()
    dm=json.loads((run/'dataset_manifest.json').read_text())
    vi=dm['validation_indices'];entries=[dm['entries'][i] for i in vi]
    assert len(entries)==281 and all(e['session'] in plan['development_sessions'] for e in entries)
    A=np.array(dm['affine_source_to_input']);scale=np.diag(A)[:2]
    with np.load(OUT/'extended_teacher_review_v2/paired_predictions.npz') as z:
        assert np.array_equal(z['context_manifest_indices'],np.array(dm['context_manifest_indices'])[vi])
        gt=z['gt'].copy();old_dlc=z['DLC200_best200'].copy();old_lp=z['LP30_best24'].copy()
    predictions=[]
    if from_evaluation:
        import pandas as pd
        h5=list((run/'project/evaluation-results-pytorch').rglob(f'*snapshot_best-{epoch}.h5'));assert len(h5)==1
        truth=pd.read_hdf(run/'project/labeled-data/camA/CollectedData_manualv3.h5')
        pred=pd.read_hdf(h5[0]);assert set(pred.index)==set(truth.index)
        pred=pred.reindex(truth.index)
        cached=np.stack([pred.xs(n,level='bodyparts',axis=1).droplevel(['scorer','individuals'],axis=1)[['x','y','likelihood']].values for n in dm['names']],axis=1)
        predictions=cached[vi]
        assert np.allclose((truth.values.reshape(-1,7,2)[vi]-A[:2,2])/scale,gt,equal_nan=True,atol=.002)
    norm_mean=torch.tensor([.485,.456,.406]).view(1,3,1,1)
    norm_std=torch.tensor([.229,.224,.225]).view(1,3,1,1)
    with torch.inference_mode():
        for i,e in enumerate([] if from_evaluation else entries):
            check()
            im=np.asarray(Image.open(ROOT/e['derived']).convert('RGB'),dtype=np.float32)
            assert im.shape==(768,704,3)
            x=torch.from_numpy(im.copy()).permute(2,0,1).unsqueeze(0)/255.
            outputs=model((x-norm_mean)/norm_std)
            poses=model.get_predictions(outputs)['bodypart']['poses'].detach().cpu().numpy()
            assert poses.size==7*3,poses.shape
            predictions.append(poses.reshape(7,3))
            if i%25==0:print(f'CPU_AUDIT {i+1}/{len(entries)}',flush=True)
    predictions=np.stack(predictions);uv=(predictions[:,:,:2]-A[:2,2])/scale
    assert np.isfinite(uv).all()
    models={f'Quadruped_epoch{epoch}':uv,'DLC200_best200':old_dlc,'LP30_best24':old_lp}
    comparison_models=dict(models)
    if run_name.endswith('seed29_v1'):
        with np.load(OUT/'quadruped_epoch110_review_v1/predictions_source_pixels.npz') as z:
            assert np.allclose(z['gt'],gt,equal_nan=True)
            assert np.array_equal(z['context_manifest_indices'],np.array(dm['context_manifest_indices'])[vi])
            comparison_models['Quadruped_seed17_best110']=z['uv'].copy()
    if epoch!=45:
        with np.load(OUT/'quadruped_epoch045_review_v1/predictions_source_pixels.npz') as z:
            assert np.allclose(z['gt'],gt,equal_nan=True)
            comparison_models['Quadruped_epoch45']=z['uv'].copy()
    report=dict(run=run_name,epoch=epoch,checkpoint=str(snapshot),checkpoint_sha256=digest,device='CPU audit of native GPU evaluation H5' if from_evaluation else 'CPU',input_wh=[704,768],source_wh=[1080,1200],
        validation_images=len(entries),confidence_filter=False,locked_holdout_accessed=False,pseudo_gt_released=False,
        models={name:metrics(value,gt,entries,dm['ids_1based'],scale) for name,value in comparison_models.items()},
        caveats=['Checkpoint-specific result; convergence and selection require full training-curve review.','Same reused development sessions, single seed.',
                 'Different pretraining/backbone/head/optimization: not isolated architecture ablation.','Swap checks are GT-assisted diagnostics only.'])
    # Independent CPU export must reproduce the logged unfiltered native error.
    evals=[]
    for line in (run/'epochs.jsonl').read_text().splitlines():
        try:r=json.loads(line)
        except json.JSONDecodeError:continue
        if r['mode']=='eval':evals.append(r)
    native=next(r for r in evals if r['epoch']==epoch)['metrics']['metrics/test.rmse']
    measured=report['models'][f'Quadruped_epoch{epoch}']['all']['input']['mean']
    report['native_input_mean']=native;report['cpu_export_input_mean']=measured
    report['prediction_source']='native_evaluation_H5' if from_evaluation else 'independent_CPU_forward'
    assert abs(measured-native)<.02,(measured,native)
    np.savez_compressed(dest/'predictions_source_pixels.npz',uv=uv,confidence=predictions[:,:,2],gt=gt,
        context_manifest_indices=np.array(dm['context_manifest_indices'])[vi],ids=dm['ids_1based'])
    (dest/'summary.json').write_text(json.dumps(report,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    x=[r['epoch'] for r in evals]
    axes[0].plot(x,[r['metrics']['metrics/test.rmse'] for r in evals],'o-',label='Quadruped unfiltered')
    axes[0].axhline(21.78005,color='orange',label='ResNet200');axes[0].set_title('Validation INPUT pixel error');axes[0].legend()
    axes[1].plot(x,[r['metrics']['metrics/test.mAP'] for r in evals],'o-');axes[1].set_title('Validation mAP')
    axes[2].step(x,[r['lr'][0] for r in evals],where='post');axes[2].set_title('Actual LR; decays planned100/130')
    for ax in axes:ax.grid(alpha=.3);ax.set_xlabel('Quadruped epochs')
    fig.savefig(dest/'learning_curve.png',dpi=140);plt.close(fig)
    err=np.linalg.norm(uv-gt,axis=-1);err_old=np.linalg.norm(old_dlc-gt,axis=-1)
    fixed=[('AZR2L2',1169,'A'),('FR2',1364,'A'),('FR2',974,'A'),('FR2',584,'D'),('FR2',974,'E'),('AZR2L2',1429,'D')]
    selected=[]
    for key in fixed:
        i=next(i for i,e in enumerate(entries) if (e['session'],e['frame'],e['camera'])==key)
        selected.append(('previous_failure',i))
    selected += [('new_worst',int(np.nanargmax(np.nanmean(err,axis=1)))),
                 ('max_gain',int(np.nanargmin(np.nanmean(err-err_old,axis=1))))]
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',23);images=[]
    for label,i in selected:
        check();e=entries[i];canvas=Image.new('RGB',(3240,1250),'#161616')
        for col,(name,points) in enumerate(models.items()):
            im=Image.open(ROOT/e['source']).convert('RGB');d=ImageDraw.Draw(im)
            for j,k in enumerate(dm['ids_1based']):
                if not np.isfinite(gt[i,j]).all():continue
                x,y=gt[i,j];u,v=points[i,j]
                d.ellipse((x-7,y-7,x+7,y+7),outline='#70ff70',width=3)
                d.text((x+9,y-24),str(k),font=font,fill='#70ff70',stroke_width=1,stroke_fill='black')
                d.line((x,y,u,v),fill='#ff77df',width=2)
                d.line((u-7,v-7,u+7,v+7),fill='#ff77df',width=3);d.line((u-7,v+7,u+7,v-7),fill='#ff77df',width=3)
                d.text((u+9,v+3),str(k),font=font,fill='#ff77df',stroke_width=1,stroke_fill='black')
            canvas.paste(im,(1080*col,50))
            ImageDraw.Draw(canvas).text((1080*col+10,10),f'{name} | {e["session"]} f{e["frame"]} cam{e["camera"]}',font=font,fill='white')
        filename=f'{label}_{e["session"]}_{e["frame"]}_cam{e["camera"]}.jpg'
        canvas.save(dest/filename,quality=95);images.append(filename)
    page=f'<meta charset="utf-8"><h1>四足版SuperAnimal 第{epoch}轮核验</h1><p>每图左：Quadruped；中：ResNet200；右：LP旧最佳。绿圈人工，粉叉预测。既往困难样本及自动挑选大错/改善，不是随机样本。无置信度筛点；收敛结论须结合完整训练曲线。</p><img width="1400" src="learning_curve.png">'
    page+=''.join(f'<h2>{p}</h2><img style="max-width:100%" src="{p}">' for p in images)
    page+='<pre>'+json.dumps(report,indent=2)+'</pre>'
    (dest/'index.html').write_text(page)
    export=Path('/mnt/c/Users/55395/Downloads/MouseGeoForge-BottomAnchor-20261002')/dest.name
    shutil.copytree(dest,export)
    decision=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),stage='intermediate_CPU_audit',
        run=run.name,checkpoint_epoch=epoch,report=str(dest/'summary.json'),
        decision='Checkpoint audit complete; compare final150 with best-mAP epoch and45 reference before extending. Preserve all labels/models; no pseudoGT release.',
        native_CPU_error_match=abs(measured-native)<.02,GPU_training_untouched=True,locked_holdout_accessed=False)
    (dest/'decision.json').write_text(json.dumps(decision,indent=2))
    write_state(latest_paired_review=str(dest/'summary.json'),latest_training_visual_directory=str(export),
        latest_decision=str(dest/'decision.json'),convergence_decision=decision['decision'])
    print(json.dumps(dict(export=str(export),models={name:{k:r[k] for k in ['all','bottom','side','swap_diagnostic']} for name,r in report['models'].items()}),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--epoch',type=int,required=True);ap.add_argument('--from-evaluation',action='store_true');ap.add_argument('--run-name',default='superanimal_quadruped_hrnet_v1');args=ap.parse_args();main(args.epoch,args.from_evaluation,args.run_name)
