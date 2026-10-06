"""Matched selector-training seed repeats; frozen baseline remains seed17."""
import datetime as dt,fcntl,json,math,os,random,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from mousegeoforge.models.prior_candidate_selector import PriorCandidateSelector
from mousegeoforge.candidate_selector_features import KP
from audit_independent_unary_20261005 import metrics,independent_inputs
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';SOURCE=CYCLE/'local_candidate_selector_pilot_v1';OUT=CYCLE/'bounded_prior_selector_seed_repeats_v1'


def main():
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='repeating_bounded_prior_selector_seeds',active_job=OUT.name,active_pid=os.getpid());write_json(CYCLE/'state.json',state)
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    status('preparing');start=time.monotonic()
    protocol=dict(seeds=[29,43],steps=800,baseline_seed=17,residual_bound=2.,meaning='Only candidate-selector initialization and balanced image schedule change. NOT a full encoder/pose or dataset split repeat. Same initialization/schedule both arms within seed. Prior seed17 immutable reference.',loss='Same soft CE sigma16, matched >=2 other camera eligibility,4realbottom+4otherreal+8syn perbatch,AdamW.001 cosine->.0001 wd.001 clip1. No dual training. No validation selection.',hashes={n:fingerprint(SOURCE/n) for n in ['train_inputs.pt','validation_inputs.pt','initial.pt','schedule.json']},seed17_reference=fingerprint(CYCLE/'bounded_prior_candidate_pilot_v1/results.json'),module_sha256=fingerprint(ROOT/'mousegeoforge/models/prior_candidate_selector.py'))
    write_json(OUT/'protocol.json',protocol)
    cpu={role:torch.load(SOURCE/f'{role}_inputs.pt',map_location='cpu',weights_only=False) for role in ['train','validation']}
    dual=torch.load(CYCLE/'dual_unary_diagnostic_v1/inputs.pt',map_location='cpu',weights_only=False)
    d={k:({kk:vv.cuda() for kk,vv in v.items()} if k=='inputs' else v.cuda() if isinstance(v,torch.Tensor) else v) for k,v in cpu['train'].items()}
    groups=[[i for i,m in enumerate(d['metadata']) if m['kind']=='real' and m['view']==0],[i for i,m in enumerate(d['metadata']) if m['kind']=='real' and m['view']!=0],[i for i,m in enumerate(d['metadata']) if m['kind']=='synthetic']]
    def make_schedule(seed):
        rng=random.Random(seed)
        return [rng.choices(groups[0],k=4)+rng.choices(groups[1],k=4)+rng.choices(groups[2],k=8) for _ in range(800)]
    assert make_schedule(17)==json.loads((SOURCE/'schedule.json').read_text())
    torch.manual_seed(17);initial_check=PriorCandidateSelector().state_dict();old=torch.load(SOURCE/'initial.pt',weights_only=True)
    for k in old:torch.testing.assert_close(initial_check[k],old[k],atol=0,rtol=0)
    results=[];arrays={}
    @torch.no_grad()
    def evaluate(model,data,inputs,tag):
        xy=data['base_xy'].clone();active=torch.zeros_like(inputs['eligible']);choice=torch.zeros_like(active,dtype=torch.long)
        for start in range(0,len(xy),32):
            ids=list(range(start,min(start+32,len(xy))));out=model(**{k:v[ids].cuda() for k,v in inputs.items()},enabled=True);block=xy[ids].clone();block[:,:,KP]=out['xy'].cpu();xy[ids]=block;active[ids]=out['active'].cpu();choice[ids]=out['choice'].cpu()
        values,points=metrics(data,xy,active,choice);write_json(OUT/f'{tag}_points.json',points);arrays[tag]=xy.numpy();arrays[tag+'_active']=active.numpy();results.append(dict(tag=tag,metrics=values,changed_all_queries=int(active.sum())))
        print(tag,{k:round(v['mean_error'],3) for k,v in values.items() if k in ['real/reliable','real/view0/feet','synthetic/reliable']},flush=True)
    for seed in [29,43]:
        torch.manual_seed(seed);m=PriorCandidateSelector();initial={k:v.clone() for k,v in m.state_dict().items()};torch.save(initial,OUT/f'seed{seed}_initial.pt');del m
        schedule=make_schedule(seed);write_json(OUT/f'seed{seed}_schedule.json',schedule)
        for arm in ['unary','geometry']:
            model=PriorCandidateSelector(arm).cuda();model.load_state_dict(initial,strict=True);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);history=[];model.train()
            for step,ids in enumerate(schedule,1):
                assert dt.datetime.now(dt.timezone.utc)<deadline
                lr=.0001+.5*(.001-.0001)*(1+math.cos(math.pi*(step-1)/800))
                for group in optimizer.param_groups:group['lr']=lr
                optimizer.zero_grad(set_to_none=True);out=model(**{k:v[ids] for k,v in d['inputs'].items()},enabled=True);valid=d['train_valid'][ids]&out['eligible'];point=-(d['target_prob'][ids]*out['logits'].log_softmax(-1)).sum(-1);loss=((point*valid).sum((-1,-2))/valid.sum((-1,-2)).clamp_min(1)).mean();assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();history.append(dict(step=step,loss=float(loss.detach()),lr=lr))
                if step%100==0:
                    status('training',seed=seed,arm=arm,step=step,loss=history[-1]['loss']);write_json(OUT/f'seed{seed}_{arm}_history.json',history);torch.save(dict(seed=seed,arm=arm,step=step,state={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),residual_bound=2.),OUT/f'seed{seed}_{arm}_latest.pt');print(seed,arm,step,history[-1]['loss'],flush=True)
            model.eval()
            for role in ['train','validation']:evaluate(model,cpu[role],cpu[role]['inputs'],f'seed{seed}_{arm}_{role}_matched')
            if arm=='unary':
                evaluate(model,cpu['validation'],independent_inputs(cpu['validation']),f'seed{seed}_unary_validation_independent')
                gate=(dual['logits'].sigmoid()>=.5)[...,None].expand(-1,-1,5);evaluate(model,dual,dict(dual['inputs'],eligible=gate),f'seed{seed}_unary_dual_perquery')
            write_json(OUT/'partial_results.json',dict(results=results))
    np.savez_compressed(OUT/'predictions.npz',**arrays);write_json(OUT/'results.json',dict(results=results,protocol=protocol,elapsed_seconds=time.monotonic()-start,script_sha256=fingerprint(Path(__file__)),production_changed=False))
    status('complete');state.update(stage='bounded_prior_selector_repeats_complete',active_job=None,active_pid=None,next='Inspect bounded_prior_viewstress_v1 and bounded_prior_selector_seed_repeats_v1, compare all3 selector seeds but frozenbaseline17, visualfailure review then coherenthandoff. No promotion.');state['completed'].append('bounded_prior_selector_training_seed29_43');write_json(CYCLE/'state.json',state)


if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise
