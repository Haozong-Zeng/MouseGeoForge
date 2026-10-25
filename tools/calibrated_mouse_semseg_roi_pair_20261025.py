"""Finite matched legacy/center label-head pilot, independent of head convergence gate."""
from pathlib import Path
import datetime as dt,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mousegeoforge.semseg_training import digest
OUT=ROOT/'output/calibrated_mouse_next12h_20261025';BASE=OUT/'semseg_roi_pair_v1'
ENGINE='tools/train_mouse_semseg_roi_20261025.py'
PARENT=ROOT/'output/calibrated_mouse_10h_20261025/semseg_campaign_s17_v1/M_018000/step_018000.pt'
def command(policy,endpoint,directory,resume=None,evaluate=False):
    args=[sys.executable,ENGINE,'evaluate' if evaluate else 'train','--items',str(OUT/'semseg_roi_coarse_cache_v1/items.json'),'--parent',str(PARENT),'--arm','M','--crop-mode',policy,'--label-policy','center_exact','--side','512','--steps','6000','--global-batch','4','--micro-batch','1','--precision','fp32','--device','cuda','--workers','2','--seed','17','--save-every','500','--eval-every','500','--run-dir',str(directory)]
    if not evaluate:args+=['--stop-after',str(endpoint),'--allow-unregistered-fixed-budget']
    if resume:args+=['--resume',str(resume)]
    return args

def now():return dt.datetime.now(dt.timezone.utc)
def main():
    assert not BASE.exists(),'Never duplicate/restart a formally started pilot'
    registration=json.loads((OUT/'registrations/semseg_roi_pair_v1.json').read_text())
    for p,h in registration['source_sha256'].items():assert digest(ROOT/p)==h,p
    cache=OUT/'semseg_roi_coarse_cache_v1/completion.json';cache_receipt=json.loads(cache.read_text());assert cache_receipt['images']==3496
    items_file=OUT/'semseg_roi_coarse_cache_v1/items.json';assert digest(items_file)==cache_receipt['items_sha256']
    assert digest(OUT/'semseg_roi_CPU_preflight_v1/receipt.json')==registration['CPU_preflight_sha256']
    for row in json.loads(items_file.read_text())['items']:
        assert digest(ROOT/row['coarse_mask'])==row['coarse_mask_sha256']
    smoke_file=OUT/'semseg_roi_actual_GPU_preflight_v1/receipt.json';smoke=json.loads(smoke_file.read_text());assert smoke['passed'] and smoke['actual_weights'] and smoke['strict_cross_crop_resume_rejected']
    plan=json.loads((OUT/'plan.json').read_text());cutoff=dt.datetime.fromisoformat(plan['optimization_cutoff_utc'])
    remaining=(cutoff-now()).total_seconds();BASE.mkdir()
    result=dict(started_utc=now().isoformat(),registration_sha256=digest(OUT/'registrations/semseg_roi_pair_v1.json'),full_paired_endpoint=0,evaluations={},status='running')
    def save(name,data): (BASE/name).write_text(json.dumps(data,indent=2)+'\n')
    def run(args,name,maximum):
        seconds=min(maximum,(cutoff-now()).total_seconds()-5)
        if seconds<=0:raise TimeoutError('Original optimization cutoff')
        with (BASE/(name+'.log')).open('w') as log:r=subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=seconds,env={**os.environ,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1'})
        if r.returncode:raise RuntimeError(f'{name} returncode{r.returncode}')
    if remaining<registration['minimum_seconds']:
        result.update(status='skipped_budget',remaining_seconds=remaining);save('completion.json',result);return
    try:
        result['actual_GPU_smoke_sha256']=digest(smoke_file)
        result['coarse_cache_sha256']=digest(cache)
        parents={};durations=[]
        for endpoint in range(500,6001,500):
            if durations:
                required=max(durations[-3:])*(6000-result['full_paired_endpoint'])/500+600
                if (cutoff-now()).total_seconds()<required:
                    result.update(status='stopped_budget_before_next_pair',estimated_remaining_seconds=required);break
            began=time.monotonic()
            for policy in ['full','roi']:
                directory=BASE/f'{policy}_{endpoint:06d}'
                run(command(policy,endpoint,directory,parents.get(policy)),f'{policy}_{endpoint:06d}',1200)
                complete=json.loads((directory/'completion.json').read_text());assert complete['segment_endpoint']==endpoint and complete['frozen_unchanged']
                parents[policy]=directory/f'step_{endpoint:06d}.pt'
            result['full_paired_endpoint']=endpoint;result['parents']={k:str(v) for k,v in parents.items()};durations.append(time.monotonic()-began)
            save('progress.json',dict(**result,last_pair_seconds=durations[-1]));print('SEMSEG_ROI_PAIR',endpoint,round(durations[-1],2),flush=True)
        if result['full_paired_endpoint']==6000:
            for policy in ['full','roi']:
                directory=BASE/(policy+'_dev');run(command(policy,6000,directory,parents[policy],True),policy+'_dev',600)
                evaluation=json.loads((directory/'evaluation.json').read_text());assert len(evaluation['ema'])==879 and len(evaluation['raw'])==879
                result['evaluations'][policy]=str(directory)
            result['status']='completed'
        result['finished_utc']=now().isoformat();save('completion.json',result)
    except Exception as error:
        result.update(status='failed',error=repr(error),finished_utc=now().isoformat());save('completion.json',result);raise
if __name__=='__main__':main()
