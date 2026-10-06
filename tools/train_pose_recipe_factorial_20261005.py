"""Bounded 2x2 loss-recipe/LR-recipe diagnosis; reuse immutable high-LR MSE control."""
import datetime as dt,fcntl,json,math,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import torch
from torch.nn import functional as F
from train_pixel_adapter_matched_20261005 import prepare_rows,metrics,read,KP,SEED
from pixel_adaptation_runtime import PixelAdaptationRuntime
from preflight_pixel_adapter_20261005 import frozen_digest
from run_surface_curve_pilot_20261003 import load_item,atomic_torch
from run_depth_pose_ablation import write_json,fingerprint
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';BASE=CYCLE/'pixel_adapter_matched_pilot_v1';OUT=CYCLE/'pose_recipe_factorial_v1'
def now():return dt.datetime.now(dt.timezone.utc)
def objective(xy,gt,valid,pixels,recipe):
    if recipe=='mse':return (xy-gt).square().sum(-1)[valid].mean()
    if recipe!='pixel_huber':raise ValueError(recipe)
    scale=pixels/768
    # The recipe includes input-aspect scaling and robust coordinate averaging.
    # It is NOT described as an isolated robustness-only intervention.
    return F.smooth_l1_loss(xy[valid]*scale,gt[valid]*scale,beta=.01,reduction='none').mean()
def main():
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));assert now()<deadline
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2);torch.manual_seed(SEED);torch.backends.cuda.matmul.allow_tf32=False
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='pose_recipe_factorial_running',active_job=OUT.name,active_pid=os.getpid(),updated_utc=now().isoformat());write_json(CYCLE/'state.json',state)
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=now().isoformat(),**kw))
    status('loading');source=json.loads((BASE/'protocol.json').read_text());rows=prepare_rows();schedule=source['schedule'];fit=source['fit_monitor'];cal=source['calibration'];val=source['validation'];assert len(schedule)==400
    arms={'mse_low':('mse','low'),'huber_high':('pixel_huber','high'),'huber_low':('pixel_huber','low')}
    protocol=dict(seed=17,steps=400,arms=arms,reused_control='pixel_adapter_matched_pilot_v1/head_only highLR MSE400; immutable saved predictions/curves/checkpoint, no new initialization or data.',
        hypothesis='Recent head regressions confound original loss recipe and LR recipe. Test their2x2 with same5point target/sampling; not full original training restoration.',
        loss_mse='Mean normalized squared Euclidean XY over all qualified nose+4feet points per image, includingbothinstances, mean4images.',
        loss_huber='Input-pixel normalized by768 SmoothL1 beta.01, coordinate mean, same qualified points/image normalization. Includes aspect scaling and robust/scale change; do not attribute difference only to robustness. No per-instance renormalization change.',
        learning_rates='high: bothreadout+decoder1e-4->1e-5. low:readout5e-5->1e-5,decoder1e-5->1e-6. Samecosine400,AdamWwd.01 andglobalclip1. No weight-decay/cohort-weight change.',
        fixed='Original43singleTRAINposes/originalviews, exact2real1single1dual400schedule, originalinitialpose, all6queries138tokens, samefrozen originalpredicted boxes, notGTcrop. Encoder/instance embeddings/coarse/detector frozen; LoRA disabled. Train1,116,928poseparams. Nose4feet only; no newtail/back/abdomen targets.',
        evaluation='Samefit72/cal220every100; finishall400armsbeforeVAL144. No selectionbyVAL orautoextension. Base sawTRAIN; stagecalonlyheldnewupdate, repeateddevelopment notexternaltst. Conditional qualifiedmetrics, notAP.',
        split_sha256=fingerprint(BASE/'protocol.json'),initial_sha256=fingerprint(BASE/'initial.pt'),script_sha256=fingerprint(Path(__file__)),training_stop_utc=plan['training_stop_utc'])
    write_json(OUT/'protocol.json',protocol)
    images={};hashes={};required=sorted(set(fit+cal+val+sum(schedule,[])))
    for j,i in enumerate(required):
        if now()>=deadline:raise TimeoutError('RGB deadline')
        r=rows[i];key=r['record']['scene'],tuple(r['record']['views'])
        if key not in images:
            item=load_item(r['record']);images[key]={v:item['batch']['images'][:,k:k+1].contiguous().clone() for k,v in enumerate(r['record']['views'])};hashes.update(item['source_hashes'])
        r['image']=images[key][r['metadata']['view']]
        if (j+1)%200==0:status('loading_rgb',images=j+1,total=len(required));print('RGB',j+1,flush=True)
    previous=json.loads((BASE/'source_hashes.json').read_text());assert all(previous.get(k)==v for k,v in hashes.items());write_json(OUT/'source_hashes.json',hashes)
    runtime=PixelAdaptationRuntime(rank=8);initial=read(BASE/'initial.pt');runtime.pose.load_state_dict(initial['pose']);runtime.adapter.enabled=False;runtime.adapter.a.requires_grad_(False);runtime.adapter.b.requires_grad_(False)
    with torch.no_grad():
        for k,v in initial['adapter'].items():getattr(runtime.adapter,k).weight.copy_(v)
    runtime.pose.spatial_readout.requires_grad_(True);runtime.pose.set_decoder_trainable(True);params=[p for p in runtime.pose.parameters() if p.requires_grad];assert sum(p.numel() for p in params)==1116928
    before=frozen_digest(runtime);baseline=torch.stack([r['baseline']['xy'] for r in rows]);curves=[];performance={}
    def forward(i):
        r=rows[i];return runtime.native_forward(r['image'].cuda(),r['baseline']['boxes'][None].cuda(),allow_input_grad=False)['xy'][0]
    @torch.no_grad()
    def evaluate(ids):
        p=baseline.clone()
        for i in ids:
            if now()>=deadline:raise TimeoutError('Evaluation deadline')
            p[i]=forward(i).cpu()
        return p
    for arm,(loss_recipe,lr_recipe) in arms.items():
        runtime.pose.load_state_dict(initial['pose']);runtime.pose.eval();torch.manual_seed(SEED)
        for _,p in runtime.named_all_parameters():p.grad=None
        if lr_recipe=='high':groups=[dict(params=params,peak=1e-4,floor=1e-5)]
        else:groups=[dict(params=runtime.pose.spatial_readout.parameters(),peak=5e-5,floor=1e-5),dict(params=runtime.pose.keypoint_decoder.parameters(),peak=1e-5,floor=1e-6)]
        opt=torch.optim.AdamW(groups,lr=1e-4,weight_decay=.01);p=evaluate(fit+cal);delta=float((p[fit+cal]-baseline[fit+cal]).abs().max());assert delta<2e-5
        curves.append(dict(arm=arm,step=0,fit=metrics(rows,p,fit),calibration=metrics(rows,p,cal),initial_delta=delta));write_json(OUT/'curves.json',curves)
        seconds=0.;peak=0;losses=[];start=time.monotonic()
        for step,ids in enumerate(schedule,1):
            if now()>=deadline:
                atomic_torch(OUT/f'{arm}_deadline.pt',dict(pose=runtime.pose.state_dict(),optimizer=opt.state_dict(),step=step-1));raise TimeoutError('Training cutoff saved')
            status('training',arm=arm,step=step-1,total=400);torch.cuda.synchronize();t0=time.monotonic();torch.cuda.reset_peak_memory_stats();opt.zero_grad(set_to_none=True)
            for g in opt.param_groups:g['lr']=g['floor']+(g['peak']-g['floor'])*.5*(1+math.cos(math.pi*(step-1)/400))
            total=0.
            for i in ids:
                r=rows[i];xy=forward(i);loss=objective(xy[:,KP],r['gt'].cuda(),r['valid'].cuda(),r['pixels'].cuda(),loss_recipe)/4
                assert torch.isfinite(loss);loss.backward();total+=float(loss.detach());del loss,xy
            assert all(p0.grad is None for _,p0 in runtime.named_all_parameters() if not p0.requires_grad)
            torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True);opt.step();torch.cuda.synchronize();seconds+=time.monotonic()-t0;peak=max(peak,torch.cuda.max_memory_allocated());losses.append(total)
            if step%100==0:
                p=evaluate(fit+cal);curves.append(dict(arm=arm,step=step,fit=metrics(rows,p,fit),calibration=metrics(rows,p,cal),recent_loss=sum(losses[-100:])/100));write_json(OUT/'curves.json',curves)
                atomic_torch(OUT/f'{arm}_step{step:04d}.pt',dict(pose={k:v.cpu() for k,v in runtime.pose.state_dict().items()},optimizer=opt.state_dict(),step=step,arm=arm));print(arm,step,'realcal',curves[-1]['calibration']['real']['mean'],'bottomcal',curves[-1]['calibration']['real_bottom_feet']['mean'],flush=True)
        performance[arm]=dict(train_seconds=seconds,total_seconds=time.monotonic()-start,peak_gpu_bytes=peak);write_json(OUT/'performance.json',performance);del opt
    old=read(BASE/'predictions.pt');oldsummary=json.loads((BASE/'results.json').read_text())['summary'];predictions={'baseline':old['predictions']['baseline'],'mse_high':old['predictions']['head_only']};summary={'baseline':oldsummary['baseline'],'mse_high':oldsummary['head_only']}
    for arm in arms:
        status('final_evaluation',arm=arm);runtime.pose.load_state_dict(read(OUT/f'{arm}_step0400.pt')['pose']);p=evaluate(fit+cal+val);predictions[arm]=p;summary[arm]={stage:metrics(rows,p,ids) for stage,ids in [('fit_monitor',fit),('calibration',cal),('validation',val)]}
    after=frozen_digest(runtime);assert before==after
    torch.save(dict(predictions=predictions,evaluated_ids=fit+cal+val),OUT/'predictions.pt');write_json(OUT/'results.json',dict(summary=summary,performance=performance,frozen_before=before,frozen_after=after,production_changed=False))
    status('complete');state.update(stage='pose_recipe_factorial_complete',active_job=None,active_pid=None,updated_utc=now().isoformat(),next='Review lossrecipe x LRrecipe400 usingfit/calcurves thenfixedVAL; rawRGB andunsuperviseddrift. NotpureHubercomparison andnotfulloriginalrecipe. No automaticextension orpromotion.');state['completed'].append('pose_recipe_factorial400');write_json(CYCLE/'state.json',state);print('COMPLETE',flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write_json(OUT/'failure.json',dict(traceback=traceback.format_exc(),utc=now().isoformat()))
        raise
