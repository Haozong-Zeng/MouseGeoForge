"""Actual Lightning Pose supervised teacher; fixed source split, bounded GPU job."""
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from run_teacher_cycle_dlc_20261001 import write_state

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/teacher_temporal_cycle_20261001'
RUN=OUT/'lp_seed17'

def run(smoke=False,loss_name='mse',seed=17):
    guard=(OUT/'gpu_job.lock').open('a+')
    fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((OUT/'plan.json').read_text())
    deadline=min(dt.datetime.fromisoformat(plan['deadline_utc']).timestamp(),time.time()+45*60)
    assert time.time()<deadline
    import numpy as np
    import pandas as pd
    import torch
    import imgaug.augmenters as iaa
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import Callback,ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from lightning_pose.data.datasets import HeatmapDataset
    from lightning_pose.models.heatmap_tracker import HeatmapTracker
    from lightning_pose.losses.factory import LossFactory
    from lightning_pose.callbacks import UnfreezeBackbone
    from torch.utils.data import DataLoader,Subset
    from importlib.metadata import distributions
    assert torch.cuda.is_available()
    torch.set_num_threads(8)
    pl.seed_everything(seed,workers=True)
    RUN.mkdir(exist_ok=True)
    m=json.loads((OUT/'dlc_dataset_manifest.json').read_text())
    with np.load(OUT/'dlc_baseline_review/predictions_source_pixels.npz') as z:gt=z['gt'].copy()
    A=np.array(m['affine_source_to_input'])
    xy=gt*np.diag(A)[:2]+A[:2,2]
    assert len(m['entries'])==len(xy)==1336
    assert set(m['ids_1based'])==set(plan['approved_body_ids_1based'])
    for role,key,sessions in [('train','train_indices',plan['training_sessions']),('validation','validation_indices',plan['development_validation_sessions'])]:
        assert all(m['entries'][i]['role']==role and m['entries'][i]['session'] in sessions for i in m[key])
    # Sole input is the already verified, pending-free teacher manifest. No other sessions opened.
    columns=pd.MultiIndex.from_product([['manualv3'],m['names'],['x','y']],names=['scorer','bodyparts','coords'])
    df=pd.DataFrame(xy.reshape(len(xy),-1),columns=columns,index=[e['derived'] for e in m['entries']])
    csv=RUN/'labels.csv';df.to_csv(csv)
    common=dict(root_directory=ROOT,csv_path=str(csv),image_resize_height=768,image_resize_width=704,
                downsample_factor=2,uniform_heatmaps=False,imgaug_hflip=False,do_context=False)
    trainset=HeatmapDataset(**common,imgaug_transform=iaa.Sequential([
        iaa.Sometimes(.5,iaa.Affine(rotate=(-15,15),scale=(.9,1.1))),
        iaa.AdditiveGaussianNoise(scale=(0,6))]))
    evalset=HeatmapDataset(**common,imgaug_transform=iaa.Sequential([]))
    assert np.array_equal(np.isnan(evalset.keypoints.numpy()),np.isnan(xy))
    assert evalset.keypoint_names==m['names']
    assert np.allclose(evalset.keypoints.numpy(),xy,equal_nan=True,atol=.001)
    sample=evalset[next(i for i in m['train_indices'] if np.isnan(xy[i]).any())]
    missing=torch.isnan(sample['keypoints'].reshape(7,2)).any(-1)
    assert missing.any() and torch.all(sample['heatmaps'][missing]==0)
    # Check every labeled channel at the target heatmap peak, not only the NaN mask.
    for i in m['train_indices'][::100]:
        item=evalset[i]
        pts=item['keypoints'].reshape(7,2)
        for j in range(7):
            if not torch.isfinite(pts[j]).all():continue
            peak=int(item['heatmaps'][j].argmax());h,w=item['heatmaps'].shape[-2:]
            coord=torch.tensor([peak%w,peak//w])*4
            assert float(torch.linalg.vector_norm(coord-pts[j]))<6
    factory=LossFactory({f'heatmap_{loss_name}':{'log_weight':0.0}},data_module=None)
    hm=sample['heatmaps'].unsqueeze(0)
    test_pred=torch.rand_like(hm,requires_grad=True)
    loss,_=factory(heatmaps_targ=hm,heatmaps_pred=test_pred)
    loss.backward();assert torch.all(test_pred.grad[0,missing]==0)
    settings=dict(framework='lightning-pose 2.4.2',backbone='resnet50',seed=seed,epochs=30,batch_size=8,
        input_wh=[704,768],downsample_factor=2,heatmap_loss=loss_name,optimizer='Adam',learning_rate=.0001,
        milestones=[20,27],gamma=.3,backbone_unfreeze_epoch=1,missing_gradient_zero_verified=True,
        train_indices=m['train_indices'],validation_indices=m['validation_indices'],
        augmentation='affine p=.5 rotate +/-15 scale .9-1.1; noise0-6; NO flip; NO crop',
        camera_inputs=False,temporal_inputs=False,train_allowed_pseudo_labels=False,
        dependency_note='Isolated lp_env packages; existing VGGT torch runtime imported read-only via .pth. No DALI used or installed.')
    (RUN/'settings.json').write_text(json.dumps(settings,indent=2))
    (RUN/'runtime_packages.json').write_text(json.dumps(sorted([dict(name=d.metadata['Name'],version=d.version,path=str(d.locate_file(''))) for d in distributions()],key=lambda x:x['name'].lower()),indent=2))
    if smoke:
        print('LP_DATA_AND_MISSING_GRADIENT_OK',len(trainset),sample['images'].shape,flush=True)
        return
    assert not list((RUN/'checkpoints').glob('*.ckpt')),'Never overwrite an existing training run'
    write_state(stage='lp_initializing',framework='Lightning Pose',pid=os.getpid(),active_run=RUN.name,
        torch_version=torch.__version__,project_config=str(RUN/'settings.json'),shuffle=None,mode='train',
        job_log=str(OUT/f'{RUN.name}.log'),job_deadline_utc=dt.datetime.fromtimestamp(deadline,dt.timezone.utc).isoformat(),
        next_action='Evaluate LP against fixed DLC baseline and inspect paw correspondence before pseudo-label export')
    model=HeatmapTracker(num_keypoints=7,loss_factory=factory,backbone='resnet50',pretrained=True,torch_seed=seed,
        downsample_factor=2,optimizer_params={'learning_rate':.0001},
        lr_scheduler_params={'milestones':[20,27],'gamma':.3})
    class Guard(Callback):
        def on_train_batch_end(self,trainer,pl_module,outputs,batch,batch_idx):
            if trainer.global_step%25==0:
                write_state(stage='lp_training',epoch=trainer.current_epoch+1,global_step=trainer.global_step,
                    torch_version=torch.__version__,project_config=str(RUN/'settings.json'),shuffle=None,mode='train')
            if time.time()>=deadline or (OUT/'STOP').exists():
                trainer.save_checkpoint(RUN/'deadline.ckpt');trainer.should_stop=True
        def on_validation_epoch_end(self,trainer,pl_module):
            if trainer.sanity_checking:return
            metrics={k:float(v.detach().cpu()) for k,v in trainer.callback_metrics.items()}
            record=dict(epoch=trainer.current_epoch+1,step=trainer.global_step,metrics=metrics,utc=dt.datetime.now(dt.timezone.utc).isoformat())
            with (RUN/'epochs.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            write_state(stage='lp_training',last_epoch_record=record)
            print('LP_EPOCH',json.dumps(record),flush=True)
    checkpoint=ModelCheckpoint(dirpath=RUN/'checkpoints',filename='epoch{epoch:03d}',monitor='val_supervised_rmse',mode='min',save_top_k=2,save_last=True)
    trainer=pl.Trainer(accelerator='gpu',devices=1,max_epochs=30,precision='32-true',
        callbacks=[Guard(),UnfreezeBackbone(unfreeze_epoch=1),checkpoint],
        logger=CSVLogger(str(RUN),name='metrics'),enable_progress_bar=False,log_every_n_steps=25,
        num_sanity_val_steps=2,max_time={'minutes':45},check_val_every_n_epoch=1)
    trainloader=DataLoader(Subset(trainset,m['train_indices']),batch_size=8,shuffle=True,num_workers=4,pin_memory=True)
    valloader=DataLoader(Subset(evalset,m['validation_indices']),batch_size=8,num_workers=4,pin_memory=True)
    trainer.fit(model,train_dataloaders=trainloader,val_dataloaders=valloader)
    assert checkpoint.best_model_path
    write_state(stage='lp_evaluating',checkpoint=checkpoint.best_model_path)
    model.load_state_dict(torch.load(checkpoint.best_model_path,map_location='cpu',weights_only=False)['state_dict'])
    model.to('cuda').eval();uv=[];conf=[]
    with torch.inference_mode():
        for batch in DataLoader(evalset,batch_size=8,num_workers=4):
            assert time.time()<deadline,'Prediction budget reached; checkpoint is safe'
            batch={k:v.to('cuda') if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
            p,c=model.predict_step(batch,0);uv.append(p.cpu().numpy().reshape(-1,7,2));conf.append(c.cpu().numpy())
    uv=(np.concatenate(uv)-A[:2,2])/np.diag(A)[:2];conf=np.concatenate(conf)
    err=np.linalg.norm(uv-gt,axis=-1)
    np.savez_compressed(RUN/'predictions_source_pixels.npz',uv=uv,confidence=conf,gt=gt,error_px=err,ids=m['ids_1based'])
    from audit_teacher_dlc_predictions_20261001 import stats
    valid=m['validation_indices']
    summary=dict(checkpoint=checkpoint.best_model_path,units='original1080x1200 pixels',validation=stats(err[valid].ravel()),
        keypoints={str(k):stats(err[valid,j]) for j,k in enumerate(m['ids_1based'])},train_allowed=False)
    (RUN/'summary.json').write_text(json.dumps(summary,indent=2))
    write_state(stage='lp_evaluated',result=summary,next_action='Compare DLC and LP source-pixel errors, visible paw semantics and confidence coverage; then choose bounded second seed or improve teacher.')
    print('LP_RESULT',json.dumps(summary),flush=True)

def main():
    global RUN
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');p.add_argument('--smoke',action='store_true')
    p.add_argument('--loss',choices=['mse','kl'],default='mse');p.add_argument('--seed',type=int,default=17);args=p.parse_args()
    RUN=OUT/(f'lp_seed{args.seed}_kl' if args.loss=='kl' else f'lp_seed{args.seed}')
    if args.launch:
        log=OUT/f'{RUN.name}.log';assert not log.exists(),'Existing job log; inspect before retry'
        with log.open('w') as f:
            proc=subprocess.Popen([sys.executable,'-u',__file__,'--loss',args.loss,'--seed',str(args.seed)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        print('LAUNCHED',proc.pid)
        return
    try:run(args.smoke,args.loss,args.seed)
    except Exception as e:
        if not args.smoke:write_state(stage='failed',framework='Lightning Pose',pid=os.getpid(),error=repr(e))
        traceback.print_exc();raise

if __name__=='__main__':main()
