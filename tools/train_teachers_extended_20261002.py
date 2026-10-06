"""New-cycle teacher continuation: isolated projects, explicit optimizer/LR resume.
Original labels/checkpoints and the closed cycle are read-only. No pseudo-GT.
"""
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
from bottom_anchor_common_20261002 import ROOT,OUT,PRIOR,budget,write_state,exclusions

class BudgetStop(Exception):pass

def dump(path,data):path.write_text(json.dumps(data,indent=2,default=str))

def manifest():
    plan=json.loads((OUT/'plan.json').read_text());excluded=exclusions(plan)
    prior=json.loads((PRIOR/'lp_context_dataset_v1/manifest.json').read_text())
    allowed=[]
    for i,e in enumerate(prior['entries']):
        assert e['session'] not in plan['locked_holdout_sessions']
        if any((e['session'],int(f)) in excluded for f in e['context_frames']):continue
        if (e['session'],e['frame']) in excluded:continue
        allowed.append(i)
    train=[i for i in allowed if prior['entries'][i]['role']=='train']
    valid=[i for i in allowed if prior['entries'][i]['role']=='validation']
    assert len(train)==1043 and len(valid)==281,(len(train),len(valid))
    return prior,allowed,train,valid

def run_dlc(run,deadline,check,extend200=False):
    os.environ['DLClight']='True'
    import numpy as np
    import pandas as pd
    import torch
    import deeplabcut
    from deeplabcut.utils import auxiliaryfunctions
    from deeplabcut.pose_estimation_pytorch.runners.train import TrainingRunner,PoseTrainingRunner
    assert torch.cuda.is_available();torch.set_num_threads(8)
    m,allowed,train,valid=manifest()
    dm=json.loads((PRIOR/'dlc_dataset_manifest.json').read_text())
    old_project=Path(dm['project_config']).parent
    project=run/'project';project.mkdir();config=project/'config.yaml'
    cfg=auxiliaryfunctions.read_config(dm['project_config']);cfg['project_path']=str(project)
    cfg['TrainingFraction']=[round(len(train)/len(allowed),2)]
    auxiliaryfunctions.write_config(config,cfg)
    target=project/'labeled-data/camA';target.mkdir(parents=True)
    old_df=pd.read_hdf(old_project/'labeled-data/camA/CollectedData_manualv3.h5')
    old_indices=[m['entries'][i]['old_index'] for i in allowed]
    df=old_df.iloc[old_indices].copy()
    entries=[m['entries'][i] for i in allowed]
    for e,index in zip(entries,df.index):
        source=ROOT/e['derived'];assert source.is_file()
        assert tuple(index)==tuple(dm['entries'][e['old_index']]['index'])
        (target/index[-1]).symlink_to(source)
    df.to_hdf(target/'CollectedData_manualv3.h5',key='df_with_missing',mode='w')
    train_new=[i for i,e in enumerate(entries) if e['role']=='train'];valid_new=[i for i,e in enumerate(entries) if e['role']=='validation']
    data=dict(project_config=str(config),names=dm['names'],ids_1based=dm['ids_1based'],entries=entries,
              context_manifest_indices=allowed,old_indices=old_indices,train_indices=train_new,validation_indices=valid_new,
              affine_source_to_input=m['affine_source_to_input'],input_wh=[704,768],source_wh=[1080,1200])
    dump(run/'dataset_manifest.json',data)
    deeplabcut.create_training_dataset(str(config),Shuffles=[1],trainIndices=[train_new],testIndices=[valid_new],
        net_type='resnet_50',engine=deeplabcut.Engine.PYTORCH,userfeedback=False)
    merged=list((project/'training-datasets').rglob('CollectedData_manualv3.h5'));assert len(merged)==1
    pd.testing.assert_frame_equal(pd.read_hdf(merged[0]),df)
    start_epoch,target_epoch=(150,200) if extend200 else (75,150)
    resume_lr=1e-5/3 if extend200 else 3e-5
    milestones,gamma=([180],.3) if extend200 else ([110,140],1/3)
    source=(list((OUT/'dlc_75to150_v1/project/dlc-models-pytorch').glob('iteration-0/*shuffle1/train/snapshot-150.pt'))
        if extend200 else list((old_project/'dlc-models-pytorch').glob('iteration-0/*shuffle2/train/snapshot-075.pt')))
    assert len(source)==1,source
    checkpoint=torch.load(source[0],map_location='cpu',weights_only=False)
    assert checkpoint['metadata']['epoch']==start_epoch and checkpoint['optimizer']['state']
    original_step=PoseTrainingRunner.step;original_epoch=TrainingRunner._epoch;original_fit=TrainingRunner.fit
    ticks=0
    def guarded_step(self,batch,mode='train'):
        nonlocal ticks
        if time.time()>deadline or (OUT/'STOP').exists():
            self.snapshot_manager.update(self.current_epoch,self.state_dict(),last=True)
            dump(run/'budget_checkpoint.json',dict(epoch=self.current_epoch,partial_epoch=True))
            raise BudgetStop('DLC deadline/STOP: checkpoint saved')
        value=original_step(self,batch,mode);ticks+=1
        if ticks%50==0:write_state(stage='teacher_extended_training',epoch=self.current_epoch,mode=mode,batches_seen=ticks)
        return value
    def logged_epoch(self,*args,**kwargs):
        value=original_epoch(self,*args,**kwargs)
        row=dict(epoch=self.current_epoch,mode=kwargs.get('mode','train'),loss=float(value),
            lr=[g['lr'] for g in self.optimizer.param_groups],metrics=self._metadata.get('metrics',{}),utc=dt.datetime.now(dt.timezone.utc).isoformat())
        with (run/'epochs.jsonl').open('a') as f:f.write(json.dumps(row,default=str)+'\n')
        write_state(last_epoch_record=row)
        return value
    def fit(self,*args,**kwargs):
        assert self.starting_epoch==start_epoch and len(self.optimizer.state)>0
        before=[g['lr'] for g in self.optimizer.param_groups]
        if extend200:assert all(abs(v-resume_lr)<1e-12 for v in before),before
        for g in self.optimizer.param_groups:g['lr']=resume_lr;g['initial_lr']=resume_lr
        self.scheduler=torch.optim.lr_scheduler.MultiStepLR(self.optimizer,milestones=milestones,gamma=gamma)
        self.scheduler.last_epoch=start_epoch;self.scheduler._last_lr=[resume_lr for _ in self.optimizer.param_groups]
        dump(run/'resume_verified.json',dict(start_epoch=start_epoch,optimizer_state_entries=len(self.optimizer.state),
             before_lr=before,resume_lr=[g['lr'] for g in self.optimizer.param_groups],milestones=milestones,
             gamma=gamma,schedule_changed_explicitly=True,optimizer_moments_preserved=True,source=str(source[0])))
        return original_fit(self,*args,**kwargs)
    PoseTrainingRunner.step=guarded_step;TrainingRunner._epoch=logged_epoch;TrainingRunner.fit=fit
    updates={'device':'cuda','train_settings.seed':17,'train_settings.dataloader_workers':4,
        'train_settings.dataloader_pin_memory':True,'train_settings.display_iters':100,
        'model.heads.bodypart.target_generator.gradient_masking':True,
        'model.heads.bodypart.target_generator.background_weight':0.,
        'data.train.crop_sampling':None,'data.train.hflip':False,
        'data.train.affine':dict(p=.5,rotation=15,scaling=[.9,1.1],translation=0),
        'data.train.gaussian_noise':6,'data.train.motion_blur':False,
        'runner.eval_interval':5,'runner.snapshots.max_snapshots':20,'runner.snapshots.save_optimizer_state':True}
    dump(run/'settings.json',dict(framework=deeplabcut.__version__,source_snapshot=str(source[0]),source_epoch=start_epoch,target_epoch=target_epoch,
        extra_epochs=target_epoch-start_epoch,batch_size=8,updates=updates,resume_lr=resume_lr,milestones=milestones,gamma=gamma,
        dataset=data['project_config'],validation_images=len(valid_new),train_images=len(train_new),
        note='DLC150to200 keeps current LR without reheating, adds decay at180; earlier75to150 explicitly reheated. No data/augmentation changes.'))
    write_state(framework='DLC',gpu=torch.cuda.get_device_name(0),epoch=start_epoch,project_config=str(config),error=None)
    deeplabcut.train_network(str(config),shuffle=1,engine=deeplabcut.Engine.PYTORCH,epochs=target_epoch-start_epoch,batch_size=8,
        save_epochs=5,max_snapshots_to_keep=20,device='cuda',pytorch_cfg_updates=updates,snapshot_path=str(source[0]))
    check()
    deeplabcut.evaluate_network(str(config),shuffles=[1],engine=deeplabcut.Engine.PYTORCH,plotting=False,per_keypoint_evaluation=True)
    dump(run/'completed.json',dict(target_epoch=target_epoch,utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        next='Compute paired pixel metrics and visuals. Then prepare explicitly requested SuperAnimal-Quadruped.'))

def run_lp(run,deadline,check):
    import numpy as np
    import pandas as pd
    import torch
    import imgaug.augmenters as iaa
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import Callback,ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from lightning_pose.data.datasets import HeatmapDataset
    from lightning_pose.models.heatmap_tracker_mhcrnn import HeatmapTrackerMHCRNN
    from lightning_pose.losses.factory import LossFactory
    from lightning_pose.utils.io import get_context_img_paths
    from torch.utils.data import DataLoader,Subset
    from torchvision.transforms import ToTensor,Normalize
    from PIL import Image
    from audit_teacher_dlc_predictions_20261001 import stats
    torch.set_num_threads(8);pl.seed_everything(17,workers=True);assert torch.cuda.is_available()
    m,allowed,train,valid=manifest();old=np.array([e['old_index'] for e in m['entries']])
    with np.load(PRIOR/'dlc_baseline_review/predictions_source_pixels.npz') as z:gt=z['gt'][old].copy()
    A=np.array(m['affine_source_to_input']);xy=gt*np.diag(A)[:2]+A[:2,2]
    # Only allowed images can enter the dataset, including context files.
    entries=[m['entries'][i] for i in allowed];old_filtered=old[allowed]
    for e in entries:
        center=ROOT/e['context_center'];paths=get_context_img_paths(center)
        expected=[center.with_name(f'frame_{f:06d}.png') for f in e['context_frames']]
        assert all(p.exists() for p in expected),f'Missing recorded source context: {center}'
        for path in paths:
            if not Path(path).exists():
                # Match the original native LP policy: repeat center only beyond a
                # real video endpoint. Missing interior context is always fatal.
                frame=int(Path(path).stem.split('_')[-1])
                assert e['boundary_clamped'] and frame>max(e['context_frames']),f'Missing interior context: {path}'
        if not e['boundary_clamped']:
            assert [str(center.with_name(f'frame_{f:06d}.png')) for f in e['context_frames']]==[str(p) for p in paths]
    cols=pd.MultiIndex.from_product([['manualv3'],m['names'],['x','y']],names=['scorer','bodyparts','coords'])
    csv=run/'labels.csv';pd.DataFrame(xy[allowed].reshape(len(allowed),-1),columns=cols,index=[e['context_center'] for e in entries]).to_csv(csv)
    train_new=[i for i,e in enumerate(entries) if e['role']=='train'];valid_new=[i for i,e in enumerate(entries) if e['role']=='validation']
    dump(run/'dataset_manifest.json',dict(entries=entries,context_manifest_indices=allowed,old_indices=old_filtered.tolist(),
        train_indices=train_new,validation_indices=valid_new,affine_source_to_input=A.tolist(),ids_1based=m['ids_1based'],names=m['names']))
    common=dict(root_directory=ROOT,csv_path=str(csv),image_resize_height=768,image_resize_width=704,
        downsample_factor=2,uniform_heatmaps=False,imgaug_hflip=False,do_context=True)
    trainset=HeatmapDataset(**common,imgaug_transform=iaa.Sequential([iaa.Sometimes(.5,iaa.Affine(rotate=(-15,15),scale=(.9,1.1))),iaa.AdditiveGaussianNoise(scale=(0,6))]))
    evalset=HeatmapDataset(**common,imgaug_transform=iaa.Sequential([]))
    assert np.allclose(evalset.keypoints.numpy(),xy[allowed],equal_nan=True,atol=.001)
    for i in [0,valid_new[0]]:
        expected=Normalize([.485,.456,.406],[.229,.224,.225])(ToTensor()(Image.open(ROOT/entries[i]['derived']).convert('RGB')))
        assert torch.equal(evalset[i]['images'][2],expected)
    factory=LossFactory({'heatmap_kl':{'log_weight':0.0}},data_module=None)
    sample=evalset[next(i for i in train_new if np.isnan(xy[allowed[i]]).any())]
    missing=torch.isnan(sample['keypoints'].reshape(7,2)).any(-1)
    p=torch.rand_like(sample['heatmaps'].unsqueeze(0),requires_grad=True)
    loss,_=factory(heatmaps_targ=sample['heatmaps'].unsqueeze(0),heatmaps_pred=p);loss.backward();assert torch.all(p.grad[0,missing]==0)
    model=HeatmapTrackerMHCRNN(num_keypoints=7,loss_factory=factory,backbone='resnet50',pretrained=True,
        downsample_factor=2,torch_seed=17,optimizer_params={'learning_rate':3e-5},lr_scheduler_params={'milestones':[45,55],'gamma':.3})
    source=PRIOR/'lp_context_seed17_kl/checkpoints/last.ckpt'
    ckpt=torch.load(source,map_location='cpu',weights_only=False)
    assert ckpt['epoch']==29 and ckpt['global_step']==7830
    assert ckpt['optimizer_states'][0]['state']
    before=[g['lr'] for g in ckpt['optimizer_states'][0]['param_groups']]
    for optimizer in ckpt['optimizer_states']:
        for group in optimizer['param_groups']:group['lr']=3e-5;group['initial_lr']=3e-5
    from collections import Counter
    for scheduler in ckpt['lr_schedulers']:
        scheduler.update(milestones=Counter([45,55]),gamma=.3,base_lrs=[3e-5]*len(before),_last_lr=[3e-5]*len(before))
    # Old checkpoint-selection callback paths must not delete or reuse prior checkpoints.
    ckpt['callbacks']={}
    resume=run/'resume_schedule.ckpt';torch.save(ckpt,resume)
    settings=dict(framework='lightning-pose2.4.2',model='MHCRNN',source_epoch=30,target_epoch=60,source=str(source),
        batch_size=4,seed=17,resume_optimizer_moments=True,source_lr=before,resume_lr=3e-5,milestones=[45,55],gamma=.3,
        train_images=len(train_new),validation_images=len(valid_new),input_wh=[704,768],source_wh=[1080,1200],
        missing_gradient_zero=True,center_input_exact=True,schedule_change_explicit=True,
        note='Full Lightning loop/optimizer resume, with explicit LR/scheduler extension; old callback state cleared to isolate output paths. All backbone parameters remain trainable.')
    dump(run/'settings.json',settings)
    for p in model.parameters():p.requires_grad_(True)
    class Guard(Callback):
        def on_train_start(self,trainer,module):
            assert trainer.current_epoch==30 and trainer.global_step==7830
            assert all(abs(g['lr']-3e-5)<1e-12 for g in trainer.optimizers[0].param_groups)
            dump(run/'resume_verified.json',dict(epoch=trainer.current_epoch,step=trainer.global_step,
                optimizer_state_entries=len(trainer.optimizers[0].state),lr=[g['lr'] for g in trainer.optimizers[0].param_groups]))
        def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
            if trainer.global_step%25==0:write_state(epoch=trainer.current_epoch+1,global_step=trainer.global_step)
            if time.time()>deadline-180 or (OUT/'STOP').exists():
                trainer.save_checkpoint(run/'deadline.ckpt');trainer.should_stop=True
        def on_validation_epoch_end(self,trainer,module):
            if trainer.sanity_checking:return
            row=dict(epoch=trainer.current_epoch+1,step=trainer.global_step,lr=[g['lr'] for g in trainer.optimizers[0].param_groups],
                metrics={k:float(v.detach().cpu()) for k,v in trainer.callback_metrics.items()})
            with (run/'epochs.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            write_state(last_epoch_record=row);print('EXTENDED_EPOCH',json.dumps(row),flush=True)
    checkpoint=ModelCheckpoint(dirpath=run/'checkpoints',filename='epoch{epoch:03d}',monitor='val_supervised_rmse',mode='min',save_top_k=3,save_last=True)
    trainer=pl.Trainer(accelerator='gpu',devices=1,max_epochs=60,precision='32-true',callbacks=[Guard(),checkpoint],
        logger=CSVLogger(str(run),name='metrics'),enable_progress_bar=False,log_every_n_steps=25,num_sanity_val_steps=2,
        max_time={'seconds':int(deadline-time.time()-180)})
    write_state(framework='LP',gpu=torch.cuda.get_device_name(0),epoch=30)
    trainer.fit(model,train_dataloaders=DataLoader(Subset(trainset,train_new),batch_size=4,shuffle=True,num_workers=4,pin_memory=True),
        val_dataloaders=DataLoader(Subset(evalset,valid_new),batch_size=4,num_workers=4,pin_memory=True),ckpt_path=str(resume))
    check();assert checkpoint.best_model_path
    model.load_state_dict(torch.load(checkpoint.best_model_path,map_location='cpu',weights_only=False)['state_dict']);model.cuda().eval()
    results=[]
    with torch.inference_mode():
        for batch in DataLoader(Subset(evalset,valid_new),batch_size=4,num_workers=4):
            check();sf,mf=model(batch['images'].cuda());ps,cs=model.head.run_subpixelmaxima(sf);pm,cm=model.head.run_subpixelmaxima(mf)
            ps,pm=ps.reshape(-1,7,2),pm.reshape(-1,7,2)
            results.append(torch.where((cm>cs).unsqueeze(-1),pm,ps).cpu().numpy())
    uv_input=np.concatenate(results);uv=(uv_input-A[:2,2])/np.diag(A)[:2];truth=gt[np.array(allowed)[valid_new]]
    with np.load(PRIOR/'lp_context_seed17_kl/predictions_source_pixels.npz') as z:
        lookup={int(i):p for p,i in enumerate(z['manifest_indices'])}
        old_uv=z['uv_selected'][[lookup[allowed[i]] for i in valid_new]]
    report=dict(checkpoint=checkpoint.best_model_path,validation_images=len(valid_new),models={})
    for name,points in [('extended',uv),('old_best24',old_uv)]:
        err=np.linalg.norm(points-truth,axis=-1);err_input=np.linalg.norm((points-truth)*np.diag(A)[:2],axis=-1)
        report['models'][name]=dict(source=stats(err.ravel()),input=stats(err_input.ravel()),paws_source=stats(err[:,3:].ravel()),
            source_over100=int((err>100).sum()),paw_source_over100=int((err[:,3:]>100).sum()))
    np.savez_compressed(run/'predictions_source_pixels.npz',uv=uv,gt=truth,uv_old=old_uv,
        context_manifest_indices=np.array(allowed)[valid_new],ids=m['ids_1based'])
    dump(run/'summary.json',report);dump(run/'completed.json',dict(epoch=trainer.current_epoch,global_step=trainer.global_step,summary=str(run/'summary.json')))

def worker(kind,run_name=None):
    import logging
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',force=True)
    plan,end,check=budget(7200);check()
    name=run_name or ('dlc_75to150_v1' if kind=='dlc' else 'lp_context_30to60_v1')
    assert name.isidentifier();run=OUT/name
    guard=(ROOT/plan['gpu_lock']).open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run.mkdir(exist_ok=False)
    write_state(stage='teacher_extended_preparing',pid=os.getpid(),active_run=run.name,mode='GPU',
        job_deadline_utc=dt.datetime.fromtimestamp(end,dt.timezone.utc).isoformat(),job_log=str(OUT/(run.name+'.log')))
    try:
        if kind=='dlc200':run_dlc(run,end,check,extend200=True)
        else:(run_dlc if kind=='dlc' else run_lp)(run,end,check)
        write_state(stage='teacher_extended_stage_complete',pid=None,active_run=None,mode='idle_between_queue_jobs',latest_result=str(run),job_deadline_utc=None)
    except BudgetStop as exc:
        dump(run/'stopped.json',dict(reason=str(exc)));write_state(stage='teacher_extended_budget_checkpoint',pid=None,active_run=None,job_deadline_utc=None)
    except Exception as exc:
        dump(run/'failed.json',dict(error=repr(exc)));write_state(stage='teacher_extended_failed',pid=None,active_run=None,error=repr(exc));raise

def queue():
    plan,end,check=budget(8*3600)
    guard=(OUT/'training_queue.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write_state(queue_pid=os.getpid(),training_priority=str(OUT/'training_priority.json'))
    jobs=[('dlc',ROOT/'DeepLabCut/.venv-mousegeoforge/bin/python','dlc_75to150_v1'),
          ('lp',PRIOR/'lp_env/bin/python','lp_context_30to60_v1')]
    queue_results=[]
    for kind,python,name in jobs:
        check()
        if (OUT/name).exists():
            queue_results.append(dict(run=name,status='existing_not_restarted'));continue
        log=OUT/(name+'.log')
        with log.open('x') as f:
            process=subprocess.Popen([str(python),'-u',__file__,'--worker',kind],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        write_state(queue_pid=os.getpid(),queue_child_pid=process.pid,next_action='User priority: sufficiently train BOTH teachers. Queue runs DLC thenLP; monitor convergence before geometry.')
        local_end=min(end,time.time()+7250)
        while process.poll() is None:
            if time.time()>local_end or (OUT/'STOP').exists():
                process.terminate()
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                break
            time.sleep(5)
        queue_results.append(dict(run=name,returncode=process.returncode))
        dump(OUT/'training_queue_results.json',queue_results)
        # Isolated failure must not indefinitely prevent the second requested model.
    write_state(queue_pid=None,queue_child_pid=None,pid=None,active_run=None,stage='teacher_training_queue_finished',mode='idle_between_heartbeats',
        next_action='Inspect BOTH run results/logs, same-cohort metrics and convergence. Extend DLC to200 / LP to100 if needed; do not claim sufficient convergence merely because stages ended.')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worker',choices=['dlc','dlc200','lp']);ap.add_argument('--launch',action='store_true');ap.add_argument('--preflight',action='store_true');ap.add_argument('--run-name');ap.add_argument('--launch-worker',action='store_true');args=ap.parse_args()
    if args.launch_worker:
        assert args.worker and args.run_name and args.run_name.isidentifier()
        plan,end,check=budget();check()
        guard=(ROOT/plan['gpu_lock']).open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(guard,fcntl.LOCK_UN)
        assert not (OUT/args.run_name).exists()
        with (OUT/(args.run_name+'.log')).open('x') as f:
            p=subprocess.Popen([sys.executable,'-u',__file__,'--worker',args.worker,'--run-name',args.run_name],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(pid=p.pid,run=args.run_name)))
    elif args.preflight:
        import torch
        m,allowed,train,valid=manifest()
        ckpt=torch.load(PRIOR/'lp_context_seed17_kl/checkpoints/last.ckpt',map_location='cpu',weights_only=False)
        assert ckpt['epoch']==29 and ckpt['global_step']==7830 and ckpt['optimizer_states'][0]['state']
        assert len(ckpt['lr_schedulers'])==1
        data=dict(train=len(train),validation=len(valid),lp_epoch_zero_based=ckpt['epoch'],lp_step=ckpt['global_step'],
            lp_lr=[g['lr'] for g in ckpt['optimizer_states'][0]['param_groups']],scheduler=ckpt['lr_schedulers'],
            LP_checkpoints_read_CPU_only=True,old_STOP_preserved=(PRIOR/'STOP').exists())
        dump(OUT/'training_preflight.json',data);print(json.dumps(data,default=str))
    elif args.launch:
        plan,end,check=budget();check()
        lock=(OUT/'training_queue.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_UN)
        log=OUT/'training_queue.log'
        with log.open('x') as f:p=subprocess.Popen([sys.executable,'-u',__file__],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(queue_pid=p.pid,log=str(log))))
    elif args.worker:worker(args.worker,args.run_name)
    else:queue()
