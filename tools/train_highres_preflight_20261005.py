"""Fixed300-step two-arm TRAIN overfit sanity, no development validation."""
import sys,datetime as dt,fcntl,json,os,time,hashlib,random,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np,torch
from mousegeoforge.models.highres_instance_readout import HighResolutionInstanceReadout,highres_localization_loss,KP
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';INPUT=CYCLE/'highres_preflight_inputs_v1';FEATURES=CYCLE/'highres_preflight_features_v1';OUT=CYCLE/'highres_preflight_pilot_v1'
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def read(p):return torch.load(p,weights_only=False,map_location='cpu')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    global OUT,FEATURES
    pre_relu='--pre-relu' in sys.argv;official='--official' in sys.argv
    assert not (pre_relu and official)
    if pre_relu:OUT=CYCLE/'highres_pre_relu_pilot_v1';FEATURES=CYCLE/'highres_pre_relu_features_v1'
    if official:OUT=CYCLE/'official_quadruped_readout_pilot_v1';FEATURES=CYCLE/'official_quadruped_features_v2'
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));assert dt.datetime.now(dt.timezone.utc)<deadline
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);OUT.mkdir(exist_ok=False);torch.set_num_threads(2);torch.manual_seed(17);torch.backends.cuda.matmul.allow_tf32=False
    def status(stage,**kw):write(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='highres_preflight_training',active_job=OUT.name,active_pid=os.getpid());write(CYCLE/'state.json',state)
    manifest=json.loads((FEATURES/'manifest.json').read_text());inf={r['index']:r for r in read(INPUT/'inference.pt')};ev={r['index']:r for r in read(INPUT/'evaluation_only.pt')};rows={}
    for z in manifest:
        i=z['index'];assert digest(ROOT/z['path'])==z['sha256'];cached=read(ROOT/z['path']);rows[i]=dict(**inf[i],**{k:v for k,v in ev[i].items() if k!='index'},**cached)
        assert rows[i]['metadata']['stage'] in ['fit','calibration']
        for k,v in rows[i].items():
            if torch.is_tensor(v):rows[i][k]=v.cuda().float() if v.dtype!=torch.bool else v.cuda()
        assert torch.isfinite(rows[i]['features']).all()
    pools={kind:[i for i,r in rows.items() if r['metadata']['stage']=='fit' and r['metadata']['kind']==kind] for kind in ['real','single','dual']};rng=random.Random(17);schedule=[[rng.choice(pools[k]) for k in pools] for _ in range(300)]
    initial=HighResolutionInstanceReadout();initial_state={k:v.clone() for k,v in initial.state_dict().items()};torch.save(initial_state,OUT/'initial.pt')
    protocol=json.loads((INPUT/'protocol.json').read_text());protocol.update(tests=9,parameter_count=sum(p.numel() for p in initial.parameters()),schedule=schedule,initial_sha256=digest(OUT/'initial.pt'),code_sha256=digest(ROOT/'mousegeoforge/models/highres_instance_readout.py'),controls='same nominal architecture/parameter count and identical initialstate; zeroappearance zeros normalized32visualchannels but retains coordinategrid/tokens/boxes/coarseXY; different effective feature information, not independent pretrained representation ablation.',pre_relu=pre_relu)
    if pre_relu or official:
        old=CYCLE/'highres_preflight_pilot_v1';reference=read(old/'initial.pt');assert all(torch.equal(v,reference[k]) for k,v in initial_state.items());assert schedule==json.loads((old/'protocol.json').read_text())['schedule'];protocol['arms']=['appearance'];protocol['comparison']='Only swap postReLU32features for same32channels before finalReLU; same normalization/readout/initial/schedule/300steps. Reuse immutable prior postReLU/zeroappearance results; no development images.'
    if official:
        protocol['official_backbone']=True;protocol['comparison']='Only swap frozen finetuned32channelpostReLU features for officialSuperAnimalpretrained32channelpostReLU features. Same torch2.11trainingenv/readout/init/schedule/loss/300endpoint. No decodersemantic reuse, no newcal/validation selection; oldfinetuned andzeroappearance controls reused. Sourcebackbone pretrainingprovenance differs, not architecture/compute change.';protocol['feature_cache_sha256']=digest(FEATURES/'manifest.json')
    write(OUT/'protocol.json',protocol)
    def forward(m,i):
        r=rows[i];return m(r['features'],r['tokens'],r['boxes'],r['baseline_xy'],r['image_hw'],enabled=True)
    def metrics(pred,stage):
        ans={}
        for kind in ['all','real','single','dual','real_bottom_feet']:
            errors=[]
            for i,r in rows.items():
                if r['metadata']['stage']!=stage:continue
                if kind=='real_bottom_feet':
                    if r['metadata']['kind']!='real' or not r['metadata']['physical_bottom']:continue
                elif kind!='all' and r['metadata']['kind']!=kind:continue
                v=r['valid'].clone()
                if kind=='real_bottom_feet':v[:,0]=False
                e=((pred[i][:,KP].cuda()-r['gt'])*r['pixels']).norm(dim=-1);errors+=e[v].detach().cpu().tolist()
            ans[kind]=dict(known=len(errors),mean=float(np.mean(errors)) if errors else None,pck32=float(np.mean(np.asarray(errors)<=32)) if errors else None)
        return ans
    @torch.no_grad()
    def evaluate(m):
        pred={i:forward(m,i)['xy'].cpu() for i in rows};return pred,{stage:metrics(pred,stage) for stage in ['fit','calibration']}
    baseline={i:r['baseline_xy'].cpu() for i,r in rows.items()};summaries={'baseline':{s:metrics(baseline,s) for s in ['fit','calibration']}};curves=[];performance={};predictions={'baseline':baseline}
    for arm in (['appearance'] if pre_relu or official else ['appearance','zero_appearance']):
        model=HighResolutionInstanceReadout(appearance=arm=='appearance').cuda();model.load_state_dict(initial_state);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);p,s=evaluate(model);assert all(torch.equal(p[i],baseline[i]) for i in rows);curves.append(dict(arm=arm,step=0,metrics=s));losses=[];seconds=0.;peak=0
        for step,ids in enumerate(schedule,1):
            assert dt.datetime.now(dt.timezone.utc)<deadline
            opt.zero_grad(set_to_none=True);factor=.1+.9*.5*(1+math.cos(math.pi*(step-1)/300))
            for g in opt.param_groups:g['lr']=.001*factor
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic();lossvalue=0.
            for i in ids:
                r=rows[i];out=forward(model,i);loss=highres_localization_loss(out,r['gt'],r['valid'],r['image_hw'])/3;assert torch.isfinite(loss);loss.backward();lossvalue+=float(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step();torch.cuda.synchronize();seconds+=time.monotonic()-start;peak=max(peak,torch.cuda.max_memory_allocated());losses.append(lossvalue)
            if step%25==0:status('training',arm=arm,step=step,total=300);print(arm,step,float(np.mean(losses[-25:])),flush=True)
            if step in [50,100,200,300]:
                p,s=evaluate(model);curves.append(dict(arm=arm,step=step,metrics=s,loss_recent=float(np.mean(losses[-50:]))));write(OUT/'curves.json',curves)
                torch.save(dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},step=step,arm=arm,optimizer=opt.state_dict()),OUT/f'{arm}_step{step:04d}.pt')
        predictions[arm]=p;summaries[arm]=s;performance[arm]=dict(training_seconds=seconds,peak_gpu_bytes=peak,losses=losses);del model,opt
    gates={k:bool(summaries['appearance']['fit'][k]['mean']<=12 and summaries['appearance']['fit'][k]['mean']<=.5*summaries['baseline']['fit'][k]['mean']) for k in ['real','single','dual']}
    torch.save(predictions,OUT/'predictions.pt');write(OUT/'results.json',dict(summary=summaries,fit_gate=gates,all_fit_gates_pass=all(gates.values()),performance=performance,development_images_evaluated=0,production_changed=False));assert all(digest(ROOT/z['path'])==z['sha256'] for z in manifest)
    status('complete');state.update(stage='highres_preflight_complete',active_job=None,active_pid=None,next='Review preflight300 fit/cal vs zeroappearance, gates and rawRGB; no development evaluation or automaticpilot until evidence.');state['completed'].append('official_quadruped_readout300' if official else ('highres_pre_relu300' if pre_relu else 'highres_instance_preflight300x2'));write(CYCLE/'state.json',state);print(json.dumps(dict(summary=summaries,fit_gate=gates)),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise
