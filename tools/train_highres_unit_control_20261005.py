"""Matched-runtime frozen vs final-fusion-adapted300 control on42TRAIN/cal inputs."""
import sys,datetime as dt,fcntl,json,os,time,hashlib,math
os.environ['DLClight']='True'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np,torch
from mousegeoforge.models.highres_fusion_adapter import HighresFusionAdapter,buffer_digest
from mousegeoforge.models.highres_instance_readout import highres_localization_loss,KP
from run_specialist_roi_20261005 import digest,write
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';INPUT=CYCLE/'highres_preflight_inputs_v1';CACHE=CYCLE/'highres_unit_inputs_v1';OLD=CYCLE/'highres_pre_relu_pilot_v1';OUT=CYCLE/'highres_unit_control_v1'
def read(p):return torch.load(p,weights_only=False,map_location='cpu')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));assert dt.datetime.now(dt.timezone.utc)<deadline
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);OUT.mkdir(exist_ok=False);torch.set_num_threads(2);torch.manual_seed(17);torch.backends.cuda.matmul.allow_tf32=False
    def status(stage,**kw):write(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='highres_unit_control_training',active_job=OUT.name,active_pid=os.getpid());write(CYCLE/'state.json',state)
    manifest=json.loads((CACHE/'manifest.json').read_text());inf={r['index']:r for r in read(INPUT/'inference.pt')};ev={r['index']:r for r in read(INPUT/'evaluation_only.pt')};rows={}
    for z in manifest:
        i=z['index'];assert sha(ROOT/z['path'])==z['sha256'];cache=read(ROOT/z['path']);r=dict(**inf[i],**{k:v for k,v in ev[i].items() if k!='index'},**cache);assert r['metadata']['stage'] in ['fit','calibration']
        r['branches']=[v.cuda() for v in r['branches']]
        for k,v in r.items():
            if torch.is_tensor(v):r[k]=v.cuda().float() if v.dtype!=torch.bool else v.cuda()
        rows[i]=r
    from deeplabcut.pose_estimation_pytorch.models import PoseModel
    cfg=json.loads((ROOT/'output/bottom_anchor_cycle_20261002/superanimal_quadruped_preparation_v1/model_config.json').read_text());builder=PoseModel.build(cfg,pretrained_backbone=False);native=builder.backbone.model.stage4[-1];native.load_state_dict(read(CACHE/'native_initial.pt'),strict=True);native.eval();del builder
    initial=read(CYCLE/'highres_preflight_pilot_v1/initial.pt');protocol=json.loads((OLD/'protocol.json').read_text());schedule=protocol['schedule'];assert len(schedule)==300
    protocol.update(arms=['frozen_unit','adapt_unit'],torch=torch.__version__,cache_sha256=sha(CACHE/'manifest.json'),unit_initial_sha256=sha(CACHE/'native_initial.pt'),unit_lr='1e-5 cosine floor1e-6',readout_lr='original1e-3 cosine floor1e-4',bn_policy='always eval, all running buffers fixed; adaptation allows affine parameter gradients',runtime_control='Previousreadout ran torch2.11, nativeHRNet/timm available inDLCtorch2.7; rerun BOTH frozen andadapt arms inDLC environment, not attributing runtime changes to architecture.',extra_unit_parameters=sum(p.numel() for p in native.parameters()),scope='same42TRAIN/internalcal only, sameinit/schedule300/objective/wd1e-4/clip1. Additionaltrainablecapacity andcompute disclosed. Never deploy or evaluate development automatically.');write(OUT/'protocol.json',protocol)
    def forward(m,i):
        r=rows[i];return m(r['branches'],r['tokens'],r['boxes'],r['baseline_xy'],r['image_hw'],enabled=True)
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
    baseline={i:r['baseline_xy'].cpu() for i,r in rows.items()};summaries={'baseline':{s:metrics(baseline,s) for s in ['fit','calibration']}};curves=[];performance={};predictions={'baseline':baseline};audits={}
    for arm in ['frozen_unit','adapt_unit']:
        adapt=arm=='adapt_unit';m=HighresFusionAdapter(native,adapt=adapt).cuda().eval();m.readout.load_state_dict(initial,strict=True);unit_before=digest(m.unit);bn_before=buffer_digest(m.unit);groups=[dict(params=list(m.readout.parameters()),lr=.001,initial_lr=.001)]
        if adapt:groups.append(dict(params=list(m.unit.parameters()),lr=.00001,initial_lr=.00001))
        opt=torch.optim.AdamW(groups,weight_decay=.0001);p,s=evaluate(m);assert all(torch.equal(p[i],baseline[i]) for i in rows);curves.append(dict(arm=arm,step=0,metrics=s));losses=[];seconds=0.;peak=0.;grad_max=0.
        for step,ids in enumerate(schedule,1):
            assert dt.datetime.now(dt.timezone.utc)<deadline;opt.zero_grad(set_to_none=True);factor=.1+.9*.5*(1+math.cos(math.pi*(step-1)/300))
            for g in opt.param_groups:g['lr']=g['initial_lr']*factor
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic();value=0.
            for i in ids:
                r=rows[i];out=forward(m,i);loss=highres_localization_loss(out,r['gt'],r['valid'],r['image_hw'])/3;assert torch.isfinite(loss);loss.backward();value+=float(loss.detach())
            grads=[p0.grad for p0 in m.unit.parameters() if p0.grad is not None];assert all(torch.isfinite(g).all() for g in grads);grad_max=max(grad_max,sum(float(g.norm()) for g in grads))
            torch.nn.utils.clip_grad_norm_([p0 for p0 in m.parameters() if p0.requires_grad],1,error_if_nonfinite=True);opt.step();torch.cuda.synchronize();seconds+=time.monotonic()-start;peak=max(peak,torch.cuda.max_memory_allocated());losses.append(value)
            if step%25==0:status('training',arm=arm,step=step,total=300);print(arm,step,float(np.mean(losses[-25:])),flush=True)
            if step in [50,100,200,300]:
                p,s=evaluate(m);curves.append(dict(arm=arm,step=step,metrics=s,loss_recent=float(np.mean(losses[-50:]))));write(OUT/'curves.json',curves);torch.save(dict(state={k:v.detach().cpu() for k,v in m.state_dict().items()},step=step,arm=arm,optimizer=opt.state_dict()),OUT/f'{arm}_step{step:04d}.pt')
        assert bn_before==buffer_digest(m.unit);assert (unit_before!=digest(m.unit))==adapt;assert (grad_max>0)==adapt
        predictions[arm]=p;summaries[arm]=s;performance[arm]=dict(training_seconds=seconds,peak_gpu_bytes=peak,losses=losses);audits[arm]=dict(unit_before=unit_before,unit_after=digest(m.unit),bn_before=bn_before,bn_after=buffer_digest(m.unit),unit_gradnorm_max=grad_max);del m,opt;torch.cuda.empty_cache()
    gates={k:bool(summaries['adapt_unit']['fit'][k]['mean']<=12 and summaries['adapt_unit']['fit'][k]['mean']<=.5*summaries['baseline']['fit'][k]['mean']) for k in ['real','single','dual']}
    oldpred=read(OLD/'predictions.pt')['appearance'];runtime_delta=max(float((predictions['frozen_unit'][i]-oldpred[i]).abs().max()) for i in rows)
    torch.save(predictions,OUT/'predictions.pt');write(OUT/'results.json',dict(summary=summaries,fit_gate=gates,all_fit_gates_pass=all(gates.values()),performance=performance,audits=audits,frozen_runtime_replay_max_normalized_xy=runtime_delta,development_images_evaluated=0,production_changed=False));assert all(sha(ROOT/z['path'])==z['sha256'] for z in manifest);assert sha(CACHE/'native_initial.pt')==protocol['unit_initial_sha256']
    status('complete');state.update(stage='highres_unit_control_complete',active_job=None,active_pid=None,next='Review frozen-vs-adapt300fit/cal gates, curves andrawRGB; no automaticfullpilot ordevelopment.');state['completed'].append('highres_unit_matched300x2');write(CYCLE/'state.json',state);print(json.dumps(dict(summary=summaries,fit_gate=gates)),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise
