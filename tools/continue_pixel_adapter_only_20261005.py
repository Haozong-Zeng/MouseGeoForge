"""Bounded convergence continuation of frozen-head LoRA, calibration-only stop gates."""
import datetime as dt,fcntl,json,os,random,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from train_pixel_adapter_matched_20261005 import prepare_rows,metrics,read,KP,SEED
from pixel_adaptation_runtime import PixelAdaptationRuntime
from preflight_pixel_adapter_20261005 import frozen_digest
from run_surface_curve_pilot_20261003 import load_item,atomic_torch
from run_depth_pose_ablation import write_json,fingerprint
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';BASE=CYCLE/'pixel_adapter_matched_pilot_v1';SOLO=CYCLE/'pixel_adapter_only_pilot_v1';OUT=CYCLE/'pixel_adapter_only_continuation_v1'
START=400;END=1600
def now():return dt.datetime.now(dt.timezone.utc)
def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),updated_utc=now().isoformat(),**kw))
def main():
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));assert now()<deadline
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2);torch.manual_seed(SEED);torch.backends.cuda.matmul.allow_tf32=False
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='pixel_adapter_only_continuation_running',active_job=OUT.name,active_pid=os.getpid(),updated_utc=now().isoformat());write_json(CYCLE/'state.json',state)
    source=json.loads((BASE/'protocol.json').read_text());fit=source['fit_monitor'];cal=source['calibration'];val=source['validation'];rows=prepare_rows()
    pools={k:[i for i,r in enumerate(rows) if r['metadata']['stage']=='fit' and r['metadata']['kind']==k and r['valid'].any()] for k in ['real','single','dual']}
    rng=random.Random(SEED);fullschedule=[rng.sample(pools['real'],2)+rng.sample(pools['single'],1)+rng.sample(pools['dual'],1) for _ in range(END)];assert fullschedule[:START]==source['schedule']
    # Reference frozen baseline, never development metrics, determines stop ceilings.
    baseline_cal=json.loads((BASE/'results.json').read_text())['summary']['baseline']['calibration']
    thresholds={k:baseline_cal[k]['mean']+margin for k,margin in [('real',2.),('single',1.),('dual',2.),('real_bottom_feet',3.)]}
    protocol=dict(start_step=START,max_total_steps=END,new_steps=END-START,seed=SEED,
        checkpoint_sha256=fingerprint(SOLO/'lora_only_step0400.pt'),source_protocol_sha256=fingerprint(BASE/'protocol.json'),schedule=fullschedule[START:],
        optimizer='Resume exact AdamW state from400; freeze alloriginalparameters and original pose. Constant1e-5 matches terminal LR floor, no warm restart. wd.01clip1;4images2real1single1dual exactcontinuedRNG.',
        decision='Convergence probe after400: fit stillimproves, calibration real/single nearbaseline, bottom31.13->30.35. Not universalgain and not matched-budget superiority. Development alreadyseen; no checkpoint selection byit.',
        calibration_stop='Every200steps evaluate full220cal and72fixedfit; if any mean ceiling exceeded for two consecutive evals, stop atcurrentendpoint, preserveweights, no bestVALcheckpoint. Also abortaftertwo evals with confidentcal unsupervisedtail mean coordinate drift >8px vsoriginal. Drift isnotaccuracy.',
        calibration_mean_ceilings=thresholds,tail_drift_ceiling_px=8.,training_stop_utc=plan['training_stop_utc'],hard_deadline_utc=plan['hard_deadline_utc'],
        scope='Same source split/actualRGB768/fixedoriginalpredboxes/qualifiednose4feet/all6queries138tokens; no newdata, pseudoGT, labels orproduction changes. Finalendpointdevelopment only, no intermediateVAL.',
        script_sha256=fingerprint(Path(__file__)))
    write_json(OUT/'protocol.json',protocol);status('loading_rgb',images=0);cache={};hashes={}
    for i,r in enumerate(rows):
        if now()>=deadline:raise TimeoutError('Deadline loading')
        key=(r['record']['scene'],tuple(r['record']['views']))
        if key not in cache:
            item=load_item(r['record']);cache[key]={v:item['batch']['images'][:,j:j+1].contiguous().clone() for j,v in enumerate(r['record']['views'])};hashes.update(item['source_hashes'])
        r['image']=cache[key][r['metadata']['view']]
        if (i+1)%200==0:status('loading_rgb',images=i+1);print('RGB',i+1,flush=True)
    assert hashes==json.loads((SOLO/'source_hashes.json').read_text()),'Source files changed'
    write_json(OUT/'source_hashes.json',hashes)
    runtime=PixelAdaptationRuntime(rank=8);checkpoint=read(SOLO/'lora_only_step0400.pt')
    with torch.no_grad():
        for k,v in checkpoint['adapter'].items():getattr(runtime.adapter,k).weight.copy_(v)
    runtime.adapter.enabled=True;params=[p for n,p in runtime.named_all_parameters() if p.requires_grad];assert len(params)==2 and sum(p.numel() for p in params)==32768
    before=frozen_digest(runtime);assert before==json.loads((SOLO/'results.json').read_text())['frozen_after']
    opt=torch.optim.AdamW(params,lr=1e-5,weight_decay=.01);opt.load_state_dict(checkpoint['optimizer'])
    for g in opt.param_groups:g['lr']=1e-5
    baseline=torch.stack([r['baseline']['xy'] for r in rows]);prior=read(SOLO/'predictions.pt')['predictions'];curves=[];losses=[];seconds=0.;peak=0;bad_streak=0;stop_reason='max_total_steps';started=time.monotonic()
    def forward(i):
        r=rows[i];return runtime.native_forward(r['image'].cuda(),r['baseline']['boxes'][None].cuda(),allow_input_grad=True)['xy'][0]
    @torch.no_grad()
    def evaluate(ids):
        p=baseline.clone()
        for i in ids:
            if now()>=deadline:raise TimeoutError('Deadline evaluating')
            p[i]=forward(i).cpu()
        return p
    def tail_drift(p):
        values=[]
        for i in cal:
            mask=rows[i]['baseline']['objectness_logits'].sigmoid()>=.5
            if mask.any():values.extend((((p[i,:,7:14]-baseline[i,:,7:14])*rows[i]['pixels']).norm(dim=-1)[mask]).flatten().tolist())
        return float(np.mean(values))
    p=evaluate(fit+cal);delta=float((p[fit+cal]-prior[fit+cal]).abs().max());assert delta<1e-5,delta
    curves.append(dict(step=START,fit=metrics(rows,p,fit),calibration=metrics(rows,p,cal),tail_drift=tail_drift(p),resume_delta=delta));write_json(OUT/'curves.json',curves)
    for step,indices in enumerate(fullschedule[START:],START+1):
        if now()>=deadline:raise TimeoutError('Training cutoff')
        status('training',step=step-1,total=END,bad_streak=bad_streak);torch.cuda.synchronize();t0=time.monotonic();torch.cuda.reset_peak_memory_stats();opt.zero_grad(set_to_none=True);total=0.
        for i in indices:
            r=rows[i];xy=forward(i);loss=(xy[:,KP]-r['gt'].cuda()).square().sum(-1)[r['valid'].cuda()].mean()/len(indices)
            assert torch.isfinite(loss);loss.backward();total+=float(loss.detach());del xy,loss
        assert all(p0.grad is None for n,p0 in runtime.named_all_parameters() if not p0.requires_grad)
        torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True);opt.step();torch.cuda.synchronize();seconds+=time.monotonic()-t0;peak=max(peak,torch.cuda.max_memory_allocated());losses.append(total)
        if step%50==0:print('continuation',step,'loss',float(np.mean(losses[-50:])),flush=True)
        if step%200==0:
            status('calibration',step=step,total=END);p=evaluate(fit+cal);cm=metrics(rows,p,cal);td=tail_drift(p);reasons=[k for k,limit in thresholds.items() if cm[k]['mean']>limit]
            if td>8:reasons.append('unsupervised_tail_drift')
            bad_streak=bad_streak+1 if reasons else 0
            curves.append(dict(step=step,fit=metrics(rows,p,fit),calibration=cm,tail_drift=td,stop_gate_reasons=reasons,bad_streak=bad_streak,recent_loss=float(np.mean(losses[-200:]))));write_json(OUT/'curves.json',curves)
            atomic_torch(OUT/f'lora_only_step{step:04d}.pt',dict(adapter={k:getattr(runtime.adapter,k).weight.detach().cpu() for k in ['a','b']},step=step,seed=SEED,optimizer=opt.state_dict(),frozen_pose_sha256=runtime.hashes['pose']))
            if bad_streak>=2:stop_reason='calibration_safety_gate';break
    status('final_evaluation',step=step,stop_reason=stop_reason);p=evaluate(fit+cal+val);after=frozen_digest(runtime);assert before==after
    summary={stage:metrics(rows,p,ids) for stage,ids in [('fit_monitor',fit),('calibration',cal),('validation',val)]}
    torch.save(dict(predictions=p,evaluated_ids=fit+cal+val,step=step),OUT/'predictions.pt')
    write_json(OUT/'results.json',dict(step=step,stop_reason=stop_reason,summary=summary,frozen_before=before,frozen_after=after,incremental_train_seconds=seconds,total_seconds=time.monotonic()-started,peak_gpu_bytes=peak,losses=losses,claim='Bounded lowLR continuation, no equivalent newhead budget. Reused development; inspect before any further research or promotion.'))
    status('complete',step=step,stop_reason=stop_reason);state.update(stage='pixel_adapter_only_continuation_complete',active_job=None,active_pid=None,next='Inspect400->endpoint fullcal curves andactualRGB changes. No automaticbudgetextension; recordfailure orboundedgain honestly, then assess data coverage versusoptimization.',updated_utc=now().isoformat());state['completed'].append('pixel_adapter_only_lowLR_continuation');write_json(CYCLE/'state.json',state);print('COMPLETE',step,stop_reason,json.dumps(summary),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write_json(OUT/'failure.json',dict(traceback=traceback.format_exc()));status('failed')
        raise

