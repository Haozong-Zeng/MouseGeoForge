"""Prospectively bounded TRAIN-safe control/feature-adapter comparison."""
from pathlib import Path
import datetime as dt,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mousegeoforge.semseg_training import digest
from mousegeoforge.semseg_adapter_safety_20261025 import violations,persistent_stop
OUT=ROOT/'output/calibrated_mouse_next12h_20261025';BASE=OUT/'semseg_adapter_pair_v1'
ENGINE='tools/train_calibrated_mouse_semseg_adapter_20261025.py'
PARENT=OUT/'semseg_roi_pair_v2/full_006000/step_006000.pt'
def command(policy,endpoint,directory,resume=None,evaluate=False):
 args=[sys.executable,ENGINE,'evaluate' if evaluate else 'train','--items',str(OUT/'semseg_roi_coarse_cache_v1/items.json'),'--parent',str(PARENT),'--arm','M','--crop-mode','full','--adapter-mode',policy,'--label-policy','center_exact','--side','512','--steps','3000','--global-batch','4','--micro-batch','1','--precision','fp32','--device','cuda','--workers','2','--seed','17','--save-every','500','--eval-every','500','--run-dir',str(directory)]
 if not evaluate:args+=['--stop-after',str(endpoint),'--allow-unregistered-fixed-budget']
 if resume:args+=['--resume',str(resume)]
 return args
def now():return dt.datetime.now(dt.timezone.utc)
def main():
 assert not BASE.exists(),'Never duplicate/restart a formally started comparison'
 registration=json.loads((OUT/'registrations/semseg_adapter_pair_v1.json').read_text())
 for p,h in registration['source_sha256'].items():assert digest(ROOT/p)==h,p
 readiness=OUT/'semseg_adapter_readiness_snapshot_v1.json';assert digest(readiness)==registration['readiness_sha256'];assert json.loads(readiness.read_text())['FULL_ready_for_trial_design']
 assert digest(PARENT)==registration['parent_sha256']
 assert digest(OUT/'semseg_adapter_CPU_preflight_v1/receipt.json')==registration['CPU_preflight_sha256']
 smoke_file=OUT/'semseg_adapter_actual_GPU_preflight_v1/receipt.json';smoke=json.loads(smoke_file.read_text());assert smoke['passed'] and smoke['actual_weights'] and smoke['strict_cross_policy_resume_rejected'] and smoke['actual_smoke_optimizer_LRs_verified']
 items=OUT/'semseg_roi_coarse_cache_v1/items.json';assert digest(items)==registration['items_sha256']
 baseline_file=OUT/'semseg_roi_pair_v2/full_006000/train_monitor_006000.json';assert digest(baseline_file)==registration['baseline_TRAIN_sha256'];baseline=json.loads(baseline_file.read_text())
 cutoff=dt.datetime.fromisoformat(json.loads((OUT/'plan.json').read_text())['optimization_cutoff_utc']);remaining=(cutoff-now()).total_seconds();BASE.mkdir()
 result=dict(started_utc=now().isoformat(),registration_sha256=digest(OUT/'registrations/semseg_adapter_pair_v1.json'),full_paired_endpoint=0,evaluations={},status='running',TRAIN_safety=[],actual_GPU_preflight_sha256=digest(smoke_file))
 def save(name,data):(BASE/name).write_text(json.dumps(data,indent=2)+'\n')
 def run(args,name,maximum):
  seconds=min(maximum,(cutoff-now()).total_seconds()-5)
  if seconds<=0:raise TimeoutError('Original optimization cutoff')
  with (BASE/(name+'.log')).open('w') as log:r=subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=seconds,env={**os.environ,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1'})
  if r.returncode:raise RuntimeError(f'{name} returncode{r.returncode}')
 if remaining<registration['minimum_seconds']:result.update(status='skipped_budget',remaining_seconds=remaining);save('completion.json',result);return
 try:
  parents={};durations=[];previous={policy:[] for policy in ['control','adapter']}
  for endpoint in range(500,3001,500):
   if durations:
    required=max(durations[-3:])*(3000-result['full_paired_endpoint'])/500+600
    if (cutoff-now()).total_seconds()<required:result.update(status='stopped_budget_before_next_pair',estimated_remaining_seconds=required);break
   began=time.monotonic();stop=False
   for policy in ['control','adapter']:
    directory=BASE/f'{policy}_{endpoint:06d}';run(command(policy,endpoint,directory,parents.get(policy)),f'{policy}_{endpoint:06d}',1200)
    complete=json.loads((directory/'completion.json').read_text());assert complete['segment_endpoint']==endpoint and complete['frozen_unchanged'] and complete['encoder_always_frozen'];parents[policy]=directory/f'step_{endpoint:06d}.pt'
    failures=violations(json.loads((directory/f'train_monitor_{endpoint:06d}.json').read_text()),baseline);persistent=persistent_stop(failures,previous[policy]);result['TRAIN_safety'].append(dict(policy=policy,endpoint=endpoint,violations=failures,persistent=persistent));previous[policy]=failures;stop|=persistent
   result['full_paired_endpoint']=endpoint;result['parents']={k:str(v) for k,v in parents.items()};durations.append(time.monotonic()-began);save('progress.json',dict(**result,last_pair_seconds=durations[-1]));print('ADAPTER_PAIR',endpoint,round(durations[-1],2),'TRAINstop',stop,flush=True)
   if stop:result['status']='stopped_persistent_TRAIN_regression';break
  if result['full_paired_endpoint']:
   for policy in ['control','adapter']:
    directory=BASE/(policy+'_dev');run(command(policy,result['full_paired_endpoint'],directory,parents[policy],True),policy+'_dev',600);evaluation=json.loads((directory/'evaluation.json').read_text());assert len(evaluation['ema'])==879 and len(evaluation['raw'])==879;result['evaluations'][policy]=str(directory)
  if result['status']=='running':result['status']='completed' if result['full_paired_endpoint']==3000 else 'partial'
  result['finished_utc']=now().isoformat();save('completion.json',result)
 except Exception as error:result.update(status='failed',error=repr(error),finished_utc=now().isoformat());save('completion.json',result);raise
if __name__=='__main__':main()
