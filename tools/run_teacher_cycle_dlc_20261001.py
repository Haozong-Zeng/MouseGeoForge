"""Single GPU DLC pilot with durable log/state, process lock and deadline checkpoint."""
import argparse
import datetime as dt
import fcntl
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ['DLClight']='True'
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/teacher_temporal_cycle_20261001'

def write_state(**kwargs):
    p=OUT/'state.json'
    state=json.loads(p.read_text()) if p.exists() else {}
    if kwargs.get('stage') and kwargs['stage']!='failed':
        state.pop('error',None)
    state.update(kwargs);state['updated_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(p)

class BudgetStop(Exception):pass

def run(extension=False):
    run_name='dlc_seed17_extension' if extension else 'dlc_seed17'
    shuffle=2 if extension else 1
    epochs=45 if extension else 30
    guard=(OUT/'gpu_job.lock').open('a+')
    fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((OUT/'plan.json').read_text())
    global_deadline=dt.datetime.fromisoformat(plan['deadline_utc']).timestamp()
    deadline=min(global_deadline,time.time()+45*60)
    if time.time()>=global_deadline:raise RuntimeError('Cycle deadline reached; not starting')
    write_state(stage='preparing_dlc_extension' if extension else 'preparing_dlc_baseline',pid=os.getpid(),framework='DLC',active_run=run_name,shuffle=shuffle,job_deadline_utc=dt.datetime.fromtimestamp(deadline,dt.timezone.utc).isoformat(),job_log=str(OUT/f'{run_name}.log'))
    if not (OUT/'dlc_prepared.json').exists():
        subprocess.run([sys.executable,'-u',str(ROOT/'tools/prepare_teacher_cycle_dlc_20261001.py')],cwd=ROOT,check=True,timeout=max(1,deadline-time.time()))
    import torch
    import deeplabcut
    from deeplabcut.pose_estimation_pytorch.runners.train import TrainingRunner,PoseTrainingRunner
    assert torch.cuda.is_available(), 'GPU required; never silently train on CPU'
    torch.set_num_threads(8)
    prepared=json.loads((OUT/'dlc_prepared.json').read_text())
    snapshot=None
    if extension:
        dm=json.loads((OUT/'dlc_dataset_manifest.json').read_text())
        project=Path(prepared['project_config']).parent
        snapshots=list((project/'dlc-models-pytorch').rglob('snapshot-best-030.pt'))
        assert len(snapshots)==1
        snapshot=str(snapshots[0])
        assert not list((project/'dlc-models-pytorch').glob('iteration-0/*shuffle2')), 'Never overwrite an extension run'
        deeplabcut.create_training_dataset(prepared['project_config'],Shuffles=[2],trainIndices=[dm['train_indices']],testIndices=[dm['validation_indices']],net_type='resnet_50',engine=deeplabcut.Engine.PYTORCH,userfeedback=False)
    original_step=PoseTrainingRunner.step
    original_epoch=TrainingRunner._epoch
    ticks=0
    def guarded_step(self,batch,mode='train'):
        nonlocal ticks
        if time.time()>=deadline or (OUT/'STOP').exists():
            self.snapshot_manager.update(self.current_epoch,self.state_dict(),last=True)
            write_state(stage='dlc_budget_checkpoint_saved',epoch=self.current_epoch,partial_epoch=True)
            raise BudgetStop('Deadline or STOP requested; checkpoint saved')
        loss=original_step(self,batch,mode)
        ticks+=1
        if ticks%25==0:write_state(stage='dlc_training',epoch=self.current_epoch,batches_seen=ticks,mode=mode)
        return loss
    def logged_epoch(self,*args,**kwargs):
        value=original_epoch(self,*args,**kwargs)
        record=dict(epoch=self.current_epoch,mode=kwargs.get('mode','train'),loss=float(value),metrics=self._metadata.get('metrics',{}),timestamp=dt.datetime.now(dt.timezone.utc).isoformat())
        with (OUT/f'{run_name}_epochs.jsonl').open('a') as f:f.write(json.dumps(record,default=str)+'\n')
        write_state(stage='dlc_training',epoch=self.current_epoch,last_epoch_record=record)
        return value
    PoseTrainingRunner.step=guarded_step;TrainingRunner._epoch=logged_epoch
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',force=True)
    # DLC 3.0.1 set_nested replaces a supplied section rather than deep-merging it.
    # Dotted leaf overrides preserve backbone, predictor, inference and logger fields.
    updates={
        'device':'cuda','train_settings.seed':17,'train_settings.dataloader_workers':4,
        'train_settings.dataloader_pin_memory':True,'train_settings.display_iters':25,
        'model.heads.bodypart.target_generator.gradient_masking':True,
        'model.heads.bodypart.target_generator.background_weight':0.0,
        'data.train.crop_sampling':None,'data.train.hflip':False,
        'data.train.affine':dict(p=.5,rotation=15,scaling=[.9,1.1],translation=0),
        'data.train.gaussian_noise':6,'data.train.motion_blur':False,
        'runner.eval_interval':5,'runner.optimizer.params.lr':0.0001,
        'runner.scheduler.params.lr_list':[[0.00005],[0.00001]],
        'runner.scheduler.params.milestones':[20,27],
        'runner.snapshots.max_snapshots':16,'runner.snapshots.save_optimizer_state':True}
    (OUT/f'{run_name}_settings.json').write_text(json.dumps(dict(epochs=epochs,batch_size=8,save_epochs=5,updates=updates,snapshot_path=snapshot,shuffle=shuffle),indent=2))
    write_state(stage='dlc_starting_gpu_training',gpu=torch.cuda.get_device_name(0),torch_version=torch.__version__,project_config=prepared['project_config'])
    try:
        deeplabcut.train_network(prepared['project_config'],shuffle=shuffle,engine=deeplabcut.Engine.PYTORCH,
            epochs=epochs,batch_size=8,save_epochs=5,max_snapshots_to_keep=16,device='cuda',pytorch_cfg_updates=updates,snapshot_path=snapshot)
    except BudgetStop as exc:
        print(str(exc),flush=True);write_state(stage='dlc_budget_complete',next_action='Inspect saved checkpoint and development validation before extending');return
    write_state(stage='dlc_training_complete',next_action='Inspect per-keypoint source-coordinate errors, examples and convergence; run continuous development clips; decide next teacher')
    if time.time()<deadline-180:
        deeplabcut.evaluate_network(prepared['project_config'],shuffles=[shuffle],engine=deeplabcut.Engine.PYTORCH,
            plotting=False,per_keypoint_evaluation=True)
        write_state(stage='dlc_baseline_evaluated',next_action='Read DLC evaluation files and per-keypoint errors, visualize examples, infer continuous development clips, compare temporal methods')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');parser.add_argument('--resume-prepared',action='store_true');parser.add_argument('--extension',action='store_true');args=parser.parse_args()
    run_name='dlc_seed17_extension' if args.extension else 'dlc_seed17'
    if args.launch:
        if (OUT/'state.json').exists():
            existing=json.loads((OUT/'state.json').read_text());pid=existing.get('pid')
            if pid and Path(f'/proc/{pid}').exists():raise RuntimeError(f'Existing worker PID {pid}; do not duplicate')
        if args.resume_prepared:
            assert (OUT/'dlc_prepared.json').exists(), 'Preparation must have completed before resume'
            assert not (OUT/'dlc_seed17_epochs.jsonl').exists(), 'This option cannot silently restart a trained run'
            log=(OUT/'dlc_seed17.log').open('a')
            log.write('\nResuming prepared project before first training batch; explicit missing-keypoint gradient masking enabled.\n');log.flush()
        else:
            assert not (OUT/f'{run_name}.log').exists(),'Use a separately named run for retries or extensions'
            log=(OUT/f'{run_name}.log').open('x')
        command=[sys.executable,'-u',str(Path(__file__).resolve())]+(['--extension'] if args.extension else [])
        p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(pid=p.pid,log=str(OUT/f'{run_name}.log'))));return
    try:run(args.extension)
    except Exception as exc:
        write_state(stage='failed',error=repr(exc),pid=os.getpid());raise

if __name__=='__main__':main()
