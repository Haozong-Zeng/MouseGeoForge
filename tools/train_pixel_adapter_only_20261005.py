"""Complete the same-budget 2x2 with a frozen-head LoRA-only arm."""
import datetime as dt,fcntl,json,math,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from train_pixel_adapter_matched_20261005 import prepare_rows,metrics,read,KP,STEPS,SEED
from pixel_adaptation_runtime import PixelAdaptationRuntime
from preflight_pixel_adapter_20261005 import frozen_digest
from run_surface_curve_pilot_20261003 import load_item,atomic_torch
from run_depth_pose_ablation import write_json,fingerprint
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';BASE=CYCLE/'pixel_adapter_matched_pilot_v1';OUT=CYCLE/'pixel_adapter_only_pilot_v1'
def now():return dt.datetime.now(dt.timezone.utc)
def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),updated_utc=now().isoformat(),**kw))
def main():
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));assert now()<deadline
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2);torch.manual_seed(SEED);torch.backends.cuda.matmul.allow_tf32=False
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='pixel_adapter_only_running',active_job=OUT.name,active_pid=os.getpid(),updated_utc=now().isoformat());write_json(CYCLE/'state.json',state)
    original=json.loads((BASE/'protocol.json').read_text());schedule=original['schedule'];fit=original['fit_monitor'];cal=original['calibration'];val=original['validation']
    assert len(schedule)==STEPS
    write_json(OUT/'protocol.json',dict(arm='lora_only',seed=SEED,steps=STEPS,initial='Same frozen original pose and exact saved initial A/B as matchedpilot',schedule_source=str(BASE.relative_to(ROOT)/'protocol.json'),schedule_sha256=fingerprint(BASE/'protocol.json'),
        optimizer='Exactly same AdamW adapter lr1e-4 cosine to.1*initial,wd.01,clip1,4image accumulation; allposeparameters frozen.',
        decision='Exploratory follow-up after matchedpilot negative; completes frozen/adapted head x off/on image LoRA2x2 at same400imageupdates. No new data/LR/labels. Not a preregistered pristine test. Train/cal stage metrics every100, final400VAL only. No bestcheckpoint selection.',
        inference='Original predicted boxes/logits fixed as anchors (notGT). ActualRGB independentV1 all6queries138tokens. Loss nose4feet known afteroriginalboxmatch. Conditionalpose not detectionAP. Unsupervised otherpoints maydrift, doNOTpromote.',
        script_sha256=fingerprint(Path(__file__)),training_stop_utc=plan['training_stop_utc']))
    rows=prepare_rows();images={};hashes={}
    status('loading_rgb',images=0)
    for i,r in enumerate(rows):
        if now()>=deadline:raise TimeoutError('Deadline during loading')
        key=(r['record']['scene'],tuple(r['record']['views']))
        if key not in images:
            item=load_item(r['record']);images[key]={v:item['batch']['images'][:,j:j+1].contiguous().clone() for j,v in enumerate(r['record']['views'])};hashes.update(item['source_hashes'])
        r['image']=images[key][r['metadata']['view']]
        if (i+1)%200==0:status('loading_rgb',images=i+1);print('RGB',i+1,flush=True)
    write_json(OUT/'source_hashes.json',hashes)
    runtime=PixelAdaptationRuntime(rank=8);initial=read(BASE/'initial.pt');runtime.pose.load_state_dict(initial['pose'])
    with torch.no_grad():
        for k,v in initial['adapter'].items():getattr(runtime.adapter,k).weight.copy_(v)
    runtime.adapter.enabled=True;params=[p for n,p in runtime.named_all_parameters() if p.requires_grad];assert len(params)==2 and sum(p.numel() for p in params)==32768
    before=frozen_digest(runtime);write_json(OUT/'frozen_before.json',before)
    baseline=torch.stack([r['baseline']['xy'] for r in rows]);curves=[]
    def forward(i):
        r=rows[i];return runtime.native_forward(r['image'].cuda(),r['baseline']['boxes'][None].cuda(),allow_input_grad=True)['xy'][0]
    @torch.no_grad()
    def evaluate(ids):
        result=baseline.clone()
        for i in ids:
            if now()>=deadline:raise TimeoutError('Deadline in evaluation')
            result[i]=forward(i).cpu()
        return result
    p=evaluate(fit+cal);delta=float((p[fit+cal]-baseline[fit+cal]).abs().max());assert delta<2e-5
    curves.append(dict(step=0,fit=metrics(rows,p,fit),calibration=metrics(rows,p,cal),initial_max_abs_delta=delta));write_json(OUT/'curves.json',curves)
    opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01);seconds=0.;peak=0;losses=[];start=time.monotonic()
    for step,indices in enumerate(schedule,1):
        if now()>=deadline:raise TimeoutError('Training cutoff')
        status('training',step=step-1,total=STEPS)
        torch.cuda.synchronize();t0=time.monotonic();torch.cuda.reset_peak_memory_stats();opt.zero_grad(set_to_none=True)
        factor=.1+.9*.5*(1+math.cos(math.pi*(step-1)/STEPS))
        for group in opt.param_groups:group['lr']=1e-4*factor
        total=0.
        for i in indices:
            xy=forward(i);r=rows[i];loss=(xy[:,KP]-r['gt'].cuda()).square().sum(-1)[r['valid'].cuda()].mean()/len(indices)
            assert torch.isfinite(loss);loss.backward();total+=float(loss.detach());del xy,loss
        if step==2:assert all(p0.grad is not None and torch.isfinite(p0.grad).all() and p0.grad.norm()>0 for p0 in params)
        assert all(p0.grad is None for n,p0 in runtime.named_all_parameters() if not p0.requires_grad)
        torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True);opt.step();torch.cuda.synchronize();seconds+=time.monotonic()-t0;peak=max(peak,torch.cuda.max_memory_allocated());losses.append(total)
        if step%25==0:print('lora_only',step,'loss',float(np.mean(losses[-25:])),flush=True)
        if step%100==0:
            p=evaluate(fit+cal);curves.append(dict(step=step,fit=metrics(rows,p,fit),calibration=metrics(rows,p,cal),recent_loss=float(np.mean(losses[-100:]))));write_json(OUT/'curves.json',curves)
            atomic_torch(OUT/f'lora_only_step{step:04d}.pt',dict(adapter={k:getattr(runtime.adapter,k).weight.detach().cpu() for k in ['a','b']},step=step,seed=SEED,optimizer=opt.state_dict(),frozen_pose_sha256=runtime.hashes['pose']))
    train_elapsed=time.monotonic()-start;status('final_evaluation')
    p=evaluate(fit+cal+val);after=frozen_digest(runtime);assert before==after
    summary={stage:metrics(rows,p,ids) for stage,ids in [('fit_monitor',fit),('calibration',cal),('validation',val)]}
    torch.save(dict(predictions=p,evaluated_ids=fit+cal+val),OUT/'predictions.pt')
    write_json(OUT/'results.json',dict(summary=summary,frozen_before=before,frozen_after=after,train_seconds=seconds,train_plus_cal_seconds=train_elapsed,peak_gpu_bytes=peak,losses=losses,claim='Exploratory frozen-head LoRA-only400, not new pristine test; same schedule/data and alloriginalweights frozen.'))
    status('complete');state.update(stage='pixel_adapter_only_complete',active_job=None,active_pid=None,next='Compare full2x2 train/cal/VAL, visualize and audit unsigned confidence and otherpoint drift; choose next step oncal evidence, no automatic extension or promotion.',updated_utc=now().isoformat());state['completed'].append('pixel_adapter_only_actualRGB400_seed17');write_json(CYCLE/'state.json',state);print('COMPLETE',json.dumps(summary),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write_json(OUT/'failure.json',dict(traceback=traceback.format_exc()));status('failed')
        raise

