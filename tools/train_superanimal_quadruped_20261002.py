"""Isolated official Quadruped backbone transfer, full-image masked 7-point teacher."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from bottom_anchor_common_20261002 import ROOT,OUT,PRIOR,budget,write_state
from train_teachers_extended_20261002 import manifest,dump,BudgetStop


def train(run,deadline,check,seed=17):
    os.environ['DLClight']='True'
    import numpy as np
    import pandas as pd
    import torch
    import deeplabcut
    from deeplabcut.utils import auxiliaryfunctions as aux
    from deeplabcut.modelzoo import build_weight_init
    from deeplabcut.pose_estimation_pytorch.runners.train import TrainingRunner,PoseTrainingRunner
    from deeplabcut.pose_estimation_pytorch.models.model import filter_state_dict
    torch.set_num_threads(8);torch.manual_seed(seed);np.random.seed(seed)
    assert torch.cuda.is_available()
    prep=OUT/'superanimal_quadruped_preparation_v1'
    proof=json.loads((prep/'preflight.json').read_text());weights=Path(proof['checkpoint'])
    assert proof['model']=='SuperAnimal-Quadruped' and proof['not_topviewmouse']
    assert hashlib.sha256(weights.read_bytes()).hexdigest()==proof['sha256']
    source=torch.load(weights,map_location='cpu',weights_only=True)
    source_backbone=filter_state_dict(source['model'],'backbone')
    base=OUT/'dlc_150to200_v1';dm=json.loads((base/'dataset_manifest.json').read_text())
    m,allowed,train_indices,valid_indices=manifest()
    assert allowed==dm['context_manifest_indices']
    project=run/'project';project.mkdir();config=project/'config.yaml'
    cfg=aux.read_config(dm['project_config']);cfg['project_path']=str(project);cfg['default_net_type']='hrnet_w32'
    aux.write_config(config,cfg)
    labels=project/'labeled-data/camA';labels.mkdir(parents=True)
    df=pd.read_hdf(Path(dm['project_config']).parent/'labeled-data/camA/CollectedData_manualv3.h5')
    for e,index in zip(dm['entries'],df.index):
        assert tuple(index)==tuple(e['index'])
        (labels/index[-1]).symlink_to(ROOT/e['derived'])
    df.to_hdf(labels/'CollectedData_manualv3.h5',key='df_with_missing',mode='w')
    dm['project_config']=str(config);dump(run/'dataset_manifest.json',dm)
    wi=build_weight_init(cfg,super_animal='superanimal_quadruped',model_name='hrnet_w32',
        detector_name=None,with_decoder=False,customized_pose_checkpoint=weights)
    deeplabcut.create_training_dataset(str(config),Shuffles=[1],trainIndices=[dm['train_indices']],
        testIndices=[dm['validation_indices']],net_type='hrnet_w32',engine=deeplabcut.Engine.PYTORCH,
        userfeedback=False,weight_init=wi)
    merged=list((project/'training-datasets').rglob('CollectedData_manualv3.h5'));assert len(merged)==1
    pd.testing.assert_frame_equal(pd.read_hdf(merged[0]),df)
    model_cfg=json.loads((prep/'model_config.json').read_text())
    # Whole model section is intentionally replaced with the exact CPU-audited model,
    # not merged with DLC's different default HRNet/locref decoder.
    base_cfgs=list((base/'project/dlc-models-pytorch').rglob('pytorch_config.yaml'));assert len(base_cfgs)==1
    from deeplabcut.core.config import read_config_as_dict
    baseline=read_config_as_dict(base_cfgs[0])
    updates={'device':'cuda','method':'bu','detector':None,'model':model_cfg,
        'train_settings.weight_init':wi.to_dict(),'train_settings.seed':seed,
        'train_settings.dataloader_workers':4,'train_settings.dataloader_pin_memory':True,
        'train_settings.display_iters':100,'data':baseline['data'],
        'runner.optimizer':{'type':'AdamW','params':{'lr':1e-4}},
        'runner.eval_interval':5,'runner.snapshots.save_optimizer_state':True,
        'runner.snapshots.max_snapshots':35,'runner.snapshots.save_epochs':5}
    assert updates['data']['train'].get('crop_sampling') is None
    assert not updates['data']['train'].get('hflip')
    initial_lr=1e-4;milestones=[100,130];target_epoch=150
    dump(run/'settings.json',dict(model='SuperAnimal-Quadruped HRNetW32',preflight=str(prep/'preflight.json'),
        target_epoch=target_epoch,initial_lr=initial_lr,milestones=milestones,gamma=.3,batch_size=8,
        seed=seed,input_wh=[704,768],source_wh=[1080,1200],train_images=1043,validation_images=281,
        head_initialization='New7 channels; official Quadruped backbone, no ImageNet fallback, no decoder semantic mapping.',
        comparison='Not architecture-only ablation: pretraining,backbone,Gaussian heatmap head and optimization differ. Same cohort,input,augmentation,missing-label policy.',
        policy='No detector/GTROI. No validation adaptation. All updates confined to new project.',updates=updates))
    original_fit=TrainingRunner.fit;original_step=PoseTrainingRunner.step;original_epoch=TrainingRunner._epoch
    ticks=0
    def checked_fit(self,train_loader,valid_loader,*args,**kwargs):
        assert self.starting_epoch==0 and len(self.optimizer.state)==0
        assert len(train_loader.dataset)==1043 and len(valid_loader.dataset)==281
        actual=self.model.backbone.state_dict()
        assert all(torch.equal(actual[k].cpu(),v) for k,v in source_backbone.items())
        # Target-level gradient test on the same newly instantiated model/loss.
        heatmap=torch.rand(1,7,192,176,device='cuda',requires_grad=True)
        keypoints=torch.tensor([[[[352.,384.,2.]]*7]])
        keypoints[0,0,1:3,2]=0
        outputs={'bodypart':{'heatmap':heatmap}}
        targets=self.model.get_target(outputs,{'keypoints':keypoints})
        loss=self.model.get_loss(outputs,targets)['total_loss'];loss.backward()
        assert torch.count_nonzero(heatmap.grad[:,1:3])==0
        assert torch.count_nonzero(heatmap.grad[:,0])>0
        for group in self.optimizer.param_groups:group['lr']=initial_lr;group['initial_lr']=initial_lr
        self.scheduler=torch.optim.lr_scheduler.MultiStepLR(self.optimizer,milestones=milestones,gamma=.3)
        dump(run/'initialization_verified.json',dict(backbone_tensors_exact=len(source_backbone),
            checkpoint_sha256=proof['sha256'],pretrained_dataset='superanimal_quadruped',
            missing_gradient_exactly_zero=True,observed_gradient_nonzero=True,
            optimizer_fresh=True,lr=[g['lr'] for g in self.optimizer.param_groups],milestones=milestones,
            train_samples=len(train_loader.dataset),validation_samples=len(valid_loader.dataset)))
        return original_fit(self,train_loader,valid_loader,*args,**kwargs)
    def step(self,batch,mode='train'):
        nonlocal ticks
        if time.time()>deadline-30 or (OUT/'STOP').exists():
            self.snapshot_manager.update(self.current_epoch,self.state_dict(),last=True)
            dump(run/'budget_checkpoint.json',dict(epoch=self.current_epoch,partial_epoch=True))
            raise BudgetStop('Quadruped deadline/STOP; checkpoint saved')
        assert tuple(batch['image'].shape[-2:])==(768,704),batch['image'].shape
        if mode=='eval':
            assert np.allclose(np.asarray(batch['scales']),1.)
            assert np.allclose(np.asarray(batch['offsets']),0.)
        value=original_step(self,batch,mode);ticks+=1
        assert np.isfinite(float(value['total_loss']))
        if ticks==1:
            dump(run/'first_gpu_step.json',dict(epoch=self.current_epoch,input_shape=list(batch['image'].shape),
                loss=float(value['total_loss']),optimizer_state_entries=len(self.optimizer.state),
                device=str(next(self.model.parameters()).device),utc=dt.datetime.now(dt.timezone.utc).isoformat()))
        if ticks%25==0:write_state(stage='superanimal_quadruped_training',epoch=self.current_epoch,mode=mode,batches_seen=ticks)
        return value
    def epoch(self,*args,**kwargs):
        value=original_epoch(self,*args,**kwargs)
        row=dict(epoch=self.current_epoch,mode=kwargs.get('mode','train'),loss=float(value),
            lr=[g['lr'] for g in self.optimizer.param_groups],metrics=self._metadata.get('metrics',{}),
            utc=dt.datetime.now(dt.timezone.utc).isoformat())
        with (run/'epochs.jsonl').open('a') as f:f.write(json.dumps(row,default=str)+'\n')
        write_state(last_epoch_record=row)
        return value
    TrainingRunner.fit=checked_fit;TrainingRunner._epoch=epoch;PoseTrainingRunner.step=step
    write_state(framework='DLC SuperAnimal-Quadruped HRNetW32',epoch=0,global_step=None,batches_seen=0,
        last_epoch_record=None,project_config=str(config),gpu=torch.cuda.get_device_name(0))
    deeplabcut.train_network(str(config),shuffle=1,engine=deeplabcut.Engine.PYTORCH,epochs=target_epoch,
        batch_size=8,save_epochs=5,max_snapshots_to_keep=35,device='cuda',pytorch_cfg_updates=updates)
    check()
    deeplabcut.evaluate_network(str(config),shuffles=[1],engine=deeplabcut.Engine.PYTORCH,plotting=False,per_keypoint_evaluation=True)
    dump(run/'completed.json',dict(epoch=target_epoch,utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        next='Export full-cohort metrics and actual visuals; compare DLC200 and LPoldbest24, bottom/side separately. No production replacement.'))


def worker(name,seed=17):
    import logging
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',force=True)
    plan,end,check=budget(7200)
    guard=(ROOT/plan['gpu_lock']).open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run=OUT/name;run.mkdir(exist_ok=False)
    write_state(stage='superanimal_quadruped_preparing',pid=os.getpid(),active_run=name,mode='GPU',error=None,seed=seed,
        job_deadline_utc=dt.datetime.fromtimestamp(end,dt.timezone.utc).isoformat(),job_log=str(OUT/(name+'.log')),
        next_action='Monitor Quadruped transfer training through meaningful LR phases; do not duplicate GPU work; checkpoint at budget.')
    try:
        train(run,end,check,seed=seed)
        write_state(stage='superanimal_quadruped_complete',pid=None,active_run=None,mode='idle',job_deadline_utc=None,latest_result=str(run))
    except BudgetStop as exc:
        dump(run/'stopped.json',dict(reason=str(exc)))
        write_state(stage='superanimal_quadruped_budget_checkpoint',pid=None,active_run=None,mode='idle',job_deadline_utc=None)
    except Exception as exc:
        dump(run/'failed.json',dict(error=repr(exc)))
        write_state(stage='superanimal_quadruped_failed',pid=None,active_run=None,mode='idle',error=repr(exc),job_deadline_utc=None)
        raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--launch',action='store_true');ap.add_argument('--run-name',default='superanimal_quadruped_hrnet_v1');ap.add_argument('--seed',type=int,default=17);args=ap.parse_args()
    assert args.run_name.isidentifier()
    if args.launch:
        plan,_,check=budget();check()
        guard=(ROOT/plan['gpu_lock']).open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(guard,fcntl.LOCK_UN)
        assert not (OUT/args.run_name).exists()
        with (OUT/(args.run_name+'.log')).open('x') as f:
            p=subprocess.Popen([sys.executable,'-u',__file__,'--run-name',args.run_name,'--seed',str(args.seed)],cwd=ROOT,
                stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(pid=p.pid,run=args.run_name)))
    else:worker(args.run_name,args.seed)
