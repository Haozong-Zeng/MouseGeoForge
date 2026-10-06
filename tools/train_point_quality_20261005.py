"""Matched scalar/semantic frozen-point quality pilot, no pose modification."""
import datetime as dt,fcntl,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from mousegeoforge.models.point_localization_quality import PointLocalizationQuality,masked_quality_loss
from diagnose_cross_query_affinity_20261005 import auc
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';SOURCE=CYCLE/'point_quality_inputs_v1';OUT=CYCLE/'point_quality_pilot_v1'


def quality_metrics(score,data,ids,*,probability=True,point_mask=None):
    s=score[ids].detach().cpu().float().numpy();known=data['valid'][ids].numpy().copy();error=data['error'][ids].numpy()
    confident=data['confident'][ids].numpy()[...,None].repeat(5,-1)
    if point_mask is not None:
        known&=np.asarray(point_mask)[None,None,:];confident&=np.asarray(point_mask)[None,None,:]
    y=(error<=32).astype(float);ks=s[known];ky=y[known];bins=[]
    if probability:
        for i in range(10):
            mask=(ks>=i/10)&((ks<(i+1)/10) if i<9 else (ks<=1))
            bins.append(dict(low=i/10,high=(i+1)/10,n=int(mask.sum()),confidence=float(ks[mask].mean()) if mask.any() else None,accuracy=float(ky[mask].mean()) if mask.any() else None))
    risk=[]
    for fraction in [.1,.25,.5,.75,1.]:
        values=s[confident]
        if not len(values):continue
        cutoff=float(np.sort(values)[::-1][max(0,int(np.ceil(fraction*len(values)))-1)])
        selected=confident&(s>=cutoff);measured=selected&known;e=error[measured]
        risk.append(dict(requested_coverage=fraction,actual_coverage=float(selected.sum()/confident.sum()),cutoff=cutoff,selected=int(selected.sum()),
            known=int(measured.sum()),unknown=int((selected&~known).sum()),mean_error=float(e.mean()) if len(e) else None,p95=float(np.percentile(e,95)) if len(e) else None,
            accuracy32=float((e<=32).mean()) if len(e) else None))
    return dict(images=len(ids),known=int(known.sum()),confident=int(confident.sum()),unknown=int((confident&~known).sum()),
        auc=auc(ks,ky),brier=float(np.mean((ks-ky)**2)) if probability and len(ks) else None,
        ece=sum(b['n']*abs(b['confidence']-b['accuracy']) for b in bins if b['n'])/len(ks) if probability and len(ks) else None,
        accuracy32=float(ky.mean()) if len(ks) else None,mean_error=float(error[known].mean()) if known.any() else None,reliability_bins=bins,risk_coverage=risk)


def eval_sets(score,data,stage,probability=True):
    metadata=data['metadata'];out={}
    for kind in ['all','real','single','dual','real_bottom_feet']:
        ids=[i for i,m in enumerate(metadata) if m['stage']==stage and (kind=='all' or m['kind']==kind or kind=='real_bottom_feet' and m['physical_bottom'])]
        if ids:out[kind]=quality_metrics(score,data,ids,probability=probability,point_mask=[False,True,True,True,True] if kind=='real_bottom_feet' else None)
    return out


def main():
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));assert dt.datetime.now(dt.timezone.utc)<deadline
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    data=torch.load(SOURCE/'inputs.pt',map_location='cpu',weights_only=False);meta=data['metadata']
    pools={kind:[i for i,m in enumerate(meta) if m['stage']=='fit' and m['kind']==kind and data['valid'][i].any()] for kind in ['real','single','dual']}
    fitcal=[i for i,m in enumerate(meta) if m['stage']!='validation']
    proto=dict(steps=800,seeds=[17,29,43],arms=['scalar','semantic'],batch_images=dict(real=8,single=4,dual=4),optimizer='AdamW',lr=[.001,.0001],weight_decay=.01,clip_norm=1,
        loss='BCE P(error<=32px), mean knownpoints/image thenmean nonemptyimages. Balanced8/4/4image schedule, no positive-class reweighting. Unknownmask only in loss.',
        architecture='Shared identical seeded initial state: semantic LayerNorm(noaffine)256->32GELU, concatenate12scalars,44->32GELU->1. Scalar arm replaces32semantic channels with zero; same nominal9697parameters,1473active versus9697semantic, NOT equal effective capacity.',
        protocol_sha256=fingerprint(SOURCE/'protocol.json'),inputs_sha256=fingerprint(SOURCE/'inputs.pt'),fit_pool_sizes={k:len(v) for k,v in pools.items()},
        scope='Fixed detector/pose/encoder and all coordinates. No threshold deployment. Fit/calcurves every100; all6fixedbudget fits before development metrics. Do not select steps on reused development. Stagecal seen by frozen base pose, unseen by new quality head. No quality-inferred visibility.')
    write_json(OUT/'protocol.json',proto)
    tensors={k:data[k].cuda() for k in ['tokens','scalars','valid','target']};prior=tensors['scalars'][...,3]
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='point_quality_training',active_job=OUT.name,active_pid=os.getpid());write_json(CYCLE/'state.json',state)
    curves=[];runs=[];started=time.monotonic()
    def train_eval(model,seed,arm,step):
        with torch.no_grad():
            scores=torch.zeros_like(prior);scores[fitcal]=model(tensors['tokens'][fitcal],tensors['scalars'][fitcal],prior[fitcal],enabled=True)['probability']
            scores=scores.cpu()
        for stage in ['fit','calibration']:curves.append(dict(seed=seed,arm=arm,step=step,stage=stage,metrics=eval_sets(scores,data,stage)))
    for seed in proto['seeds']:
        torch.manual_seed(seed);initial=PointLocalizationQuality();initial_state={k:v.cpu().clone() for k,v in initial.state_dict().items()};torch.save(initial_state,OUT/f'seed{seed}_initial.pt')
        rng=np.random.default_rng(seed);schedule=np.array([rng.permutation(np.concatenate([rng.choice(pools[k],size=n,replace=True) for k,n in proto['batch_images'].items()])) for _ in range(800)])
        np.save(OUT/f'seed{seed}_schedule.npy',schedule)
        for arm in proto['arms']:
            assert dt.datetime.now(dt.timezone.utc)<deadline
            model=PointLocalizationQuality(arm).cuda();model.load_state_dict(initial_state);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,800,eta_min=.0001)
            train_eval(model,seed,arm,0);losses=[];t0=time.monotonic();torch.cuda.reset_peak_memory_stats()
            for step,indices in enumerate(schedule,1):
                if step%100==1:assert dt.datetime.now(dt.timezone.utc)<deadline
                ids=torch.as_tensor(indices,device='cuda');opt.zero_grad(set_to_none=True)
                output=model(tensors['tokens'][ids],tensors['scalars'][ids],prior[ids],enabled=True)
                loss=masked_quality_loss(output['logits'],tensors['target'][ids],tensors['valid'][ids]);assert torch.isfinite(loss)
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step();scheduler.step();losses.append(float(loss.detach()))
                if step%100==0:
                    train_eval(model,seed,arm,step);torch.save(dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},seed=seed,arm=arm,step=step),OUT/f'{arm}_seed{seed}_step{step:04d}.pt')
                    write_json(OUT/'curves.json',curves);write_json(OUT/'status.json',dict(stage='training',pid=os.getpid(),seed=seed,arm=arm,step=step))
            torch.cuda.synchronize();runs.append(dict(seed=seed,arm=arm,seconds=time.monotonic()-t0,last100_loss=float(np.mean(losses[-100:])),previous100_loss=float(np.mean(losses[-200:-100])),peak_gpu_bytes=torch.cuda.max_memory_allocated()))
            print('TRAINED',runs[-1],flush=True);del model,opt,scheduler
    # Fixed-budget training has finished. Development is evaluated only now.
    scores={};results={}
    for arm in ['objectness','inverse_std_rank']:
        score=data['scalars'][...,3] if arm=='objectness' else -data['std_px'];scores[arm]=score
        results[arm]={stage:eval_sets(score,data,stage,probability=arm=='objectness') for stage in ['fit','calibration','validation']}
    for seed in proto['seeds']:
        for arm in proto['arms']:
            model=PointLocalizationQuality(arm).cuda().eval();model.load_state_dict(torch.load(OUT/f'{arm}_seed{seed}_step0800.pt',map_location='cpu',weights_only=False)['state'])
            with torch.no_grad():score=model(tensors['tokens'],tensors['scalars'],prior,enabled=True)['probability'].cpu()
            name=f'{arm}_seed{seed}';scores[name]=score;results[name]={stage:eval_sets(score,data,stage) for stage in ['fit','calibration','validation']}
            del model
    torch.save(scores,OUT/'scores.pt');write_json(OUT/'results.json',dict(results=results,runs=runs,elapsed_seconds=time.monotonic()-started,production_changed=False,coordinates_changed=False,
        module_sha256=fingerprint(ROOT/'mousegeoforge/models/point_localization_quality.py'),script_sha256=fingerprint(Path(__file__)),protocol_sha256=fingerprint(OUT/'protocol.json')))
    write_json(OUT/'status.json',dict(stage='complete',pid=os.getpid()));state.update(stage='point_quality_pilot_complete',active_job=None,active_pid=None,next='Inspect allseed point-quality Brier/AUC/ECE and risk-coverage, source-calibration curves and wrong confident foot overlays; no pose changes or confidence fusion yet.');state['completed'].append('point_quality_scalar_semantic_800x3x2');write_json(CYCLE/'state.json',state)
    print(json.dumps({k:{kind:{m:s[m] for m in ['known','auc','brier','ece']} for kind,s in v['validation'].items()} for k,v in results.items()},indent=2))


if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise

