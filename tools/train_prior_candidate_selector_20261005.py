"""Predeclared bounded-prior intervention; identical 800-step pilot schedule."""
import datetime as dt,fcntl,json,math,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from mousegeoforge.models.prior_candidate_selector import PriorCandidateSelector
from mousegeoforge.candidate_selector_features import KP
from audit_independent_unary_20261005 import metrics,independent_inputs
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';SOURCE=CYCLE/'local_candidate_selector_pilot_v1';OUT=CYCLE/'bounded_prior_candidate_pilot_v1'


def main():
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'))
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='training_bounded_prior',active_job=OUT.name,active_pid=os.getpid());write_json(CYCLE/'state.json',state)
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    status('preparing');start=time.monotonic()
    protocol=dict(steps=800,residual_bound=2.0,prior='Candidate0 baseline logit+.001, other5 log(native localmass/max over6 masses); bounded correction 2*tanh((learnedscore-learnedbaseline)/2). Weak peaks below exp(-2) relative mass cannot win against baseline. This is not a guarantee against cross-instance mistakes.',bound_selection='Predeclared 2 log-units (odds boost <exp(2)), no threshold/VAL sweep. Only this hypothesis; no extra auxiliary loss, no dual TRAIN additions.',control='Reuse immutable unconstrained 800-step unary/geometry controls. Same initial tensors, image schedule, GT soft-target CE sigma16, eligibility (>=2 other cameras), batch weighting, optimizer and lr. Only score parameterization changed. Independent unary gate and dual per-query tests are posttraining diagnostics, not selection.',learning='AdamW lr.001 cosine->.0001,wd.001,gradclip1; frozen all baseline/candidates/encoder; final step800 only.',safety='No original annotations/production/default factory changed. Reused development VAL not external test.',source_hashes={n:fingerprint(SOURCE/n) for n in ['initial.pt','schedule.json','results.json','train_inputs.pt','validation_inputs.pt']},module_sha256=fingerprint(ROOT/'mousegeoforge/models/prior_candidate_selector.py'))
    write_json(OUT/'protocol.json',protocol)
    schedule=json.loads((SOURCE/'schedule.json').read_text());assert len(schedule)==800
    initial=torch.load(SOURCE/'initial.pt',map_location='cpu',weights_only=True)
    cpu={role:torch.load(SOURCE/f'{role}_inputs.pt',map_location='cpu',weights_only=False) for role in ['train','validation']}
    gpu={role:{k:({kk:vv.cuda() for kk,vv in v.items()} if k=='inputs' else v.cuda() if isinstance(v,torch.Tensor) else v) for k,v in d.items()} for role,d in cpu.items()}
    dual=torch.load(CYCLE/'dual_unary_diagnostic_v1/inputs.pt',map_location='cpu',weights_only=False)
    results=[];arrays={}
    @torch.no_grad()
    def evaluate(model,d,inputs,tag):
        xy=d['base_xy'].clone();active=torch.zeros_like(inputs['eligible']);choices=torch.zeros_like(active,dtype=torch.long)
        for start in range(0,len(xy),32):
            ids=list(range(start,min(start+32,len(xy))));b={k:v[ids].cuda() for k,v in inputs.items()};out=model(**b,enabled=True)
            block=xy[ids].clone();block[:,:,KP]=out['xy'].cpu();xy[ids]=block;active[ids]=out['active'].cpu();choices[ids]=out['choice'].cpu()
        values,points=metrics(d,xy,active,choices);write_json(OUT/f'{tag}_points.json',points)
        arrays[tag]=xy.numpy();arrays[tag+'_active']=active.numpy();results.append(dict(tag=tag,metrics=values,changed_all_queries=int(active.sum())))
        print(tag,{k:round(v['mean_error'],3) for k,v in values.items() if k in ['real/reliable','real/view0/feet','synthetic/reliable']},flush=True)
    for arm in ['unary','geometry']:
        model=PriorCandidateSelector(arm,2.0).cuda();model.load_state_dict(initial,strict=True)
        b={k:v[:4] for k,v in gpu['validation']['inputs'].items()};torch.testing.assert_close(model(**b,enabled=True)['xy'],b['baseline_xy'],atol=0,rtol=0)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);history=[];model.train();d=gpu['train']
        for step,ids in enumerate(schedule,1):
            assert dt.datetime.now(dt.timezone.utc)<deadline
            lr=.0001+.5*(.001-.0001)*(1+math.cos(math.pi*(step-1)/800))
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.zero_grad(set_to_none=True);out=model(**{k:v[ids] for k,v in d['inputs'].items()},enabled=True)
            valid=d['train_valid'][ids]&out['eligible'];point=-(d['target_prob'][ids]*out['logits'].log_softmax(-1)).sum(-1)
            loss=((point*valid).sum((-1,-2))/valid.sum((-1,-2)).clamp_min(1)).mean();assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            history.append(dict(step=step,loss=float(loss.detach()),lr=lr))
            if step%100==0:
                status('training',arm=arm,step=step,loss=history[-1]['loss']);write_json(OUT/f'{arm}_history.json',history);torch.save(dict(arm=arm,step=step,state={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),residual_bound=2.0),OUT/f'{arm}_latest.pt');print(arm,step,history[-1]['loss'],flush=True)
        model.eval();assert model(**b)['xy'] is b['baseline_xy']
        for role in ['train','validation']:evaluate(model,cpu[role],cpu[role]['inputs'],arm+'_'+role+'_matched')
        if arm=='unary':
            evaluate(model,cpu['validation'],independent_inputs(cpu['validation']),'unary_validation_independent')
            gate=(dual['logits'].sigmoid()>=.5)[...,None].expand(-1,-1,5)
            evaluate(model,dual,dict(dual['inputs'],eligible=gate),'unary_dual_perquery')
    np.savez_compressed(OUT/'predictions.npz',**arrays);write_json(OUT/'results.json',dict(results=results,protocol=protocol,elapsed_seconds=time.monotonic()-start,script_sha256=fingerprint(Path(__file__)),production_changed=False))
    status('complete');state.update(stage='bounded_prior_pilot_complete',active_job=None,active_pid=None,next='Inspect bounded_prior_candidate_pilot_v1 results and curves, compare immutable unconstrained controls, raw overlays and failure mass diagnostics before further steps. No promotion.');state['completed'].append('bounded_prior_candidate_800x2');write_json(CYCLE/'state.json',state)


if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise
