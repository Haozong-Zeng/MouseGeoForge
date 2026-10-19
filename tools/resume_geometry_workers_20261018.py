"""Full-state continuation of the active geometry arm and disposable A/B timing.

The original protocol and trainer stay immutable. This runtime amendment changes
only loading/profiling and adds safe resume; no next-arm scheduler exists.
"""
from __future__ import annotations
import argparse
import collections
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import time
import traceback

import numpy as np
import torch
import pure_geometry_bottom_20261018 as geo
import geometry_prefetch_20261018 as io

BASE=geo.OUT
DEST=BASE/'geometry_seed17'
SESSION=BASE/'resume_workers4'


def digest(value):
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            y=x.detach().cpu().contiguous()
            h.update(str((y.dtype,tuple(y.shape))).encode());h.update(y.numpy().tobytes())
        elif isinstance(x,np.ndarray):h.update(str((x.dtype,x.shape)).encode());h.update(x.tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(list,tuple)):
            h.update(str(type(x)).encode())
            for y in x:visit(y)
        else:h.update(repr(x).encode())
    visit(value)
    return h.hexdigest()


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def restore(source,manifest,work):
    ck=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
    protocol=geo.read(BASE/'protocol.json')
    assert ck['protocol_sha256']==geo.sha(BASE/'protocol.json')
    assert ck['base_sha256']==protocol['checkpoint_sha256']
    assert geo.sha(geo.__file__)==protocol['code_sha256']
    assert geo.sha(geo.old.__file__)==protocol['loader_sha256']
    assert geo.sha(BASE/'manifest.json')==protocol['manifest_sha256']
    model,opt,shared=geo.make_model();params=dict(model.named_parameters())
    names={n for n,p in params.items() if p.requires_grad}
    assert names==set(ck['model'])==set(ck['ema'])
    with torch.no_grad():
        for n,value in ck['model'].items():params[n].copy_(value)
    opt.load_state_dict(ck['optimizer'])
    ema={n:value.to(params[n].device).clone() for n,value in ck['ema'].items()}
    step=ck['step'];expected=collections.Counter(i['sampling_role'] for ms in work[:step] for i in ms)
    assert dict(expected)==ck['exposures'],(expected,ck['exposures'])
    # Verify complete optimizer state, parameters and EMA, before consuming data.
    checks=dict(model=digest({n:params[n] for n in names})==digest(ck['model']),
                ema=digest(ema)==digest(ck['ema']),
                optimizer=digest(opt.state_dict())==digest(ck['optimizer']))
    assert all(checks.values()),checks
    restore_rng(ck['rng'])
    checks['rng']=digest(geo.old.rng_state())==digest(ck['rng'])
    assert checks['rng']
    meta=dict(step=step,loss_weights=ck['loss_weights'],exposures=expected,
              checks=checks,rng_digest=digest(ck['rng']))
    del ck
    return model,opt,shared,params,ema,meta


class Feed:
    def __init__(self,manifest,items,workers):
        self.manifest=manifest;self.items=items;self.workers=workers;self.position=0
        self.dl=io.loader(manifest,items,workers,True) if workers else None
        self.it=iter(self.dl) if self.dl else None

    def next(self,item):
        assert item==self.items[self.position]
        if self.it:
            row=next(self.it)
            assert row['sequence']==self.position and row['item']==item
            b=io.to_device(row['batch'])
        else:b=geo.load_item(self.manifest['records'][item['index']],self.manifest,item['views'])
        self.position+=1
        return b

    def close(self):
        self.it=None;self.dl=None


def update(model,opt,shared,params,ema,weights,exposures,feed,micros,step,updates,instrument=True):
    opt.zero_grad(set_to_none=True);totals=collections.Counter();details=[];wait=0.
    warm=min(1.,(step+1)/500)
    progress=max(0.,(step-500)/max(1,updates-500))
    factor=warm*(.1+.9*(1+math.cos(math.pi*min(progress,1)))/2)
    for pg in opt.param_groups:pg['lr']=pg['base_lr']*factor
    for item in micros:
        began=time.perf_counter();b=feed.next(item);wait+=time.perf_counter()-began
        pred=model(b['images']);losses=geo.terms(pred,b)
        if instrument and step%100==0:
            norms,cosines=geo.task_gradients(losses,shared)
            details.append(dict(role=item['sampling_role'],raw_gradient_norms=norms,
                weighted_gradient_norms={k:v*weights[k] for k,v in norms.items()},cosines=cosines))
        loss=sum(weights[k]*v for k,v in losses.items())/len(micros)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite objective')
        loss.backward();exposures[item['sampling_role']]+=1
        for k,v in losses.items():totals[k]+=float(v.detach())/len(micros)
        del pred,losses,loss,b
    norm=torch.nn.utils.clip_grad_norm_([p for p in params.values() if p.requires_grad],10.)
    if not torch.isfinite(norm):raise FloatingPointError('Nonfinite accumulated gradient')
    opt.step();step+=1;decay=min(.999,(step+1)/(step+10))
    with torch.no_grad():
        for n,value in ema.items():value.lerp_(params[n].detach(),1-decay)
    return dict(utc=geo.utc(),step=step,losses=dict(totals),gradient_norm=float(norm),
                lr_factor=factor,ema_decay=decay,gradient_details=details,loader_wait_s=wait)


@contextlib.contextmanager
def ema_weights(params,ema):
    backup={n:params[n].detach().clone() for n in ema}
    rng=geo.old.rng_state()
    try:
        with torch.no_grad():
            for n,v in ema.items():params[n].copy_(v)
        yield
    finally:
        with torch.no_grad():
            for n,v in backup.items():params[n].copy_(v)
        restore_rng(rng)


def checkpoint(path,model,opt,ema,step,weights,exposures,work,elapsed,clipped):
    geo.checkpoint(path,model,opt,ema,step,weights,exposures)
    # Same authoritative step selects the next registered update; prefetched but
    # unconsumed samples do not count and can always be discarded safely.
    geo.write(path.with_suffix('.resume.json'),dict(utc=geo.utc(),step=step,
        next_update_index=step,next_microbatch_offset=step*8,
        schedule_sha256=geo.sha(DEST/'schedule.json'),elapsed_training_s=elapsed,
        clipped_updates=clipped,workers=4,automatic_next_arm=False,
        next_update=work[step] if step<len(work) else None,
        runtime_amendment_sha256=geo.sha(SESSION/'runtime_amendment.json')))


def profile_one(model,feed,item,output):
    b=feed.next(item);torch.cuda.synchronize()
    # Profiling is disposable and excluded from both benchmark and formal steps.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
        began=time.perf_counter();pred=model(b['images']);losses=geo.terms(pred,b)
        sum(losses.values()).backward();torch.cuda.synchronize();wall=time.perf_counter()-began
    events=prof.key_averages();rows=[]
    for e in events:
        rows.append(dict(name=e.key,count=e.count,self_cpu_us=e.self_cpu_time_total,
                         self_device_us=getattr(e,'self_device_time_total',0.)))
    gpu_us=sum(e['self_device_us'] for e in rows)
    geo.write(output,dict(wall_s=wall,device_self_time_s=gpu_us/1e6,
        cpu_top=sorted(rows,key=lambda x:x['self_cpu_us'],reverse=True)[:30],
        device_top=sorted(rows,key=lambda x:x['self_device_us'],reverse=True)[:30],
        synchronizations=[r for r in rows if 'Synchronize' in r['name'] or '_local_scalar_dense' in r['name']],
        images_shape=list(b['images'].shape),peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
        note='One disposable forward/loss/backward microbatch after A/B timing; profiler adds overhead. Kernel self time is not nvidia-smi utilization.'))


def benchmark(args,manifest,work,protocol):
    label=f'benchmark_w{args.workers}_{args.repeat}'
    output=SESSION/f'{label}.json';assert not output.exists()
    model,opt,shared,params,ema,meta=restore(args.source,manifest,work)
    step=meta['step'];start=step;weights=meta['loss_weights'];exposures=meta['exposures']
    count=args.warmup+args.measured
    extra=1 if args.profile else 0
    items=[i for ms in work[step:step+count+extra] for i in ms]
    feed=Feed(manifest,items,args.workers)
    assert digest(geo.old.rng_state())==meta['rng_digest']
    samples=[]
    with (SESSION/f'{label}_gpu.csv').open('w') as gpu_log:
        monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw',
            '--format=csv,noheader,nounits','--loop-ms=200'],stdout=gpu_log,stderr=subprocess.DEVNULL)
        try:
            for index,micros in enumerate(work[start:start+count]):
                torch.cuda.synchronize();began=time.perf_counter()
                row=update(model,opt,shared,params,ema,weights,exposures,feed,micros,step,protocol['budget']['updates'])
                torch.cuda.synchronize();elapsed=time.perf_counter()-began;step+=1
                samples.append(dict(step=step,seconds=elapsed,loader_wait_s=row['loader_wait_s'],
                    measured=index>=args.warmup,losses=row['losses']))
        finally:
            monitor.terminate();monitor.wait(timeout=10)
    timed=[s for s in samples if s['measured']]
    result=dict(utc=geo.utc(),workers=args.workers,repeat=args.repeat,initial_step=start,
        final_step=step,disposable=True,warmup_updates=args.warmup,measured_updates=len(timed),
        measured_total_s=sum(x['seconds'] for x in timed),
        seconds_per_update=sum(x['seconds'] for x in timed)/len(timed),
        loader_wait_s_per_update=sum(x['loader_wait_s'] for x in timed)/len(timed),samples=samples,
        restored_state_checks=meta['checks'],peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
        final_digests=dict(model=digest({n:p for n,p in params.items() if p.requires_grad}),
            optimizer=digest(opt.state_dict()),ema=digest(ema),rng=digest(geo.old.rng_state()),exposures=digest(dict(exposures))),
        runtime_code_sha256=geo.sha(__file__))
    geo.write(output,result);print(json.dumps({k:v for k,v in result.items() if k not in ('samples','final_digests')}),flush=True)
    if args.profile:profile_one(model,feed,work[step][0],SESSION/'gpu_profile.json')
    feed.close()


def continuation(args,manifest,work,protocol):
    assert args.workers==4
    gate=geo.read(SESSION/'continuation_gate.json')
    assert gate['passed']
    assert gate['benchmark_summary_sha256']==geo.sha(SESSION/'benchmark_summary.json')
    assert gate['numerical_diagnostic_sha256']==geo.sha(SESSION/'numerical_resume_diagnostic.json')
    diagnostic=geo.read(SESSION/'numerical_resume_diagnostic.json')
    assert diagnostic['passed'] and diagnostic['cpu_inputs_bitwise_equal']
    assert diagnostic['original_serial_cuda_backward_not_bitwise_repeatable']
    amendment=geo.read(SESSION/'runtime_amendment.json')
    assert amendment['runtime_code_sha256']==geo.sha(__file__)
    assert amendment['prefetch_code_sha256']==geo.sha(io.__file__)
    model,opt,shared,params,ema,meta=restore(args.source,manifest,work)
    step=meta['step'];start=step;weights=meta['loss_weights'];exposures=meta['exposures']
    lines=[json.loads(x) for x in (DEST/'train.jsonl').read_text().splitlines()]
    assert lines[-1]['step']==step,'Training log must end at source checkpoint'
    clipped=sum(r['gradient_norm']>10 for r in lines)
    elapsed_before=float(amendment['elapsed_training_s_before_pause']);began=time.monotonic();feed=None
    def elapsed():return elapsed_before+time.monotonic()-began
    def status(state,**more):
        row=dict(utc=geo.utc(),pid=os.getpid(),state=state,step=step,elapsed_s=elapsed(),
            resumed_from_step=start,workers=4,exposures=dict(exposures),automatic_next_arm=False,**more)
        geo.write(DEST/'status.json',row);geo.write(SESSION/'status.json',row)
    geo.write(SESSION/'resume_integrity.json',dict(utc=geo.utc(),restored_from_step=step,
        restored_state_checks=meta['checks'],next_update_index=step,next_microbatch_offset=step*8,
        schedule_sha256=geo.sha(DEST/'schedule.json'),exposures=dict(exposures),
        lr_factor_for_next_update=min(1.,(step+1)/500)*(.1+.9*(1+math.cos(math.pi*max(0.,(step-500)/(protocol['budget']['updates']-500))))/2),
        ema_next_decay=min(.999,(step+2)/(step+11)),no_optimizer_step_reset=True,
        no_data_replay_or_skip=True,automatic_next_arm=False))
    geo.write(BASE/'pipeline_status.json',dict(utc=geo.utc(),state='resume_workers4_training',
        child_pid=os.getpid(),supervisor_pid=None,automatic_next_arm=False))
    def validate(name):
        status('ema_validation')
        with ema_weights(params,ema):geo.evaluate(model,manifest,DEST/name)
    try:
        # Complete interrupted validation without consuming a training sample.
        missing=DEST/f'ema_step{step:06d}.json'
        if step%500==0 and not missing.exists():validate(missing.name)
        items=[i for micros in work[step:] for i in micros]
        feed=Feed(manifest,items,4)
        assert digest(geo.old.rng_state())==meta['rng_digest']
        status('training')
        with (DEST/'train.jsonl').open('a',buffering=1) as log:
            while step<len(work):
                if geo.STOP or (BASE/'STOP').exists() or elapsed()>protocol['budget']['max_hours']*3600:break
                began_step=time.perf_counter()
                row=update(model,opt,shared,params,ema,weights,exposures,feed,work[step],step,protocol['budget']['updates'])
                step+=1;clipped+=row['gradient_norm']>10
                row.update(clip_fraction=clipped/step,workers=4,resume_session='resume_workers4',
                    update_wall_s=time.perf_counter()-began_step)
                log.write(json.dumps(row)+'\n')
                if step%10==0 or step==start+1:
                    status('training',last=row,peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3)
                    print(json.dumps(row),flush=True)
                if step%250==0:checkpoint(DEST/'latest.pt',model,opt,ema,step,weights,exposures,work,elapsed(),clipped)
                if step%500==0:validate(f'ema_step{step:06d}.json')
        status('saving');checkpoint(DEST/'latest.pt',model,opt,ema,step,weights,exposures,work,elapsed(),clipped)
        if not geo.STOP and not (BASE/'STOP').exists():
            validate(f'ema_final_step{step:06d}.json');geo.evaluate(model,manifest,DEST/f'raw_final_step{step:06d}.json')
        state='completed' if step==len(work) else 'stopped_or_budget'
        status(state,promotion='None; no automatic frozen control or keypoint stage')
        geo.write(BASE/'pipeline_status.json',dict(utc=geo.utc(),state=state,child_pid=os.getpid(),automatic_next_arm=False))
    except BaseException as exc:
        # EMA context always restores raw parameters, including interrupted eval.
        checkpoint(DEST/'resumed_interrupted.pt',model,opt,ema,step,weights,exposures,work,elapsed(),clipped)
        status('interrupted' if isinstance(exc,InterruptedError) else 'failed',error=str(exc),traceback=traceback.format_exc())
        raise
    finally:
        if feed is not None:feed.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['benchmark','continue'])
    p.add_argument('--source',type=Path,default=SESSION/'pre_stop_checkpoint.pt')
    p.add_argument('--workers',type=int,choices=[0,4],default=4)
    p.add_argument('--repeat',type=int,default=1);p.add_argument('--warmup',type=int,default=4)
    p.add_argument('--measured',type=int,default=12);p.add_argument('--profile',action='store_true')
    args=p.parse_args();torch.set_num_threads(4);geo.old.cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    def stop(*_):geo.STOP=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    manifest=geo.read(BASE/'manifest.json');work=geo.read(DEST/'schedule.json');protocol=geo.read(BASE/'protocol.json')
    amendment=geo.read(SESSION/'runtime_amendment.json')
    assert geo.sha(args.source)==amendment['source_checkpoint_sha256']
    assert geo.sha(DEST/'schedule.json')==amendment['schedule_sha256']
    assert geo.sha(__file__)==amendment['runtime_code_sha256']
    lock=(geo.ROOT/'output/teacher_temporal_cycle_20261001/gpu_job.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        if args.mode=='benchmark':benchmark(args,manifest,work,protocol)
        else:continuation(args,manifest,work,protocol)
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':main()
