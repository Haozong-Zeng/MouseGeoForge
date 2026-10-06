"""Controlled synthetic non-target supervision retention, without new labels."""
import datetime as dt,fcntl,json,math,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import torch
from torch.nn import functional as F
from train_pixel_adapter_matched_20261005 import prepare_rows,metrics,read,KP,SEED
from review_pose_recipe_20261005 import full_targets
from pixel_adaptation_runtime import PixelAdaptationRuntime
from preflight_pixel_adapter_20261005 import frozen_digest
from run_surface_curve_pilot_20261003 import load_item,atomic_torch
from run_depth_pose_ablation import write_json,fingerprint
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';BASE=CYCLE/'pixel_adapter_matched_pilot_v1';REF=CYCLE/'pose_recipe_factorial_v1';OUT=CYCLE/'synthetic_supervision_retention_v1'
def now():return dt.datetime.now(dt.timezone.utc)
def objective(xy,gt,valid,pixels,recipe):
    if recipe=='mse':return (xy-gt).square().sum(-1)[valid].mean()
    if recipe!='pixel_huber':raise ValueError(recipe)
    scale=pixels/768
    # The recipe includes input-aspect scaling and robust coordinate averaging.
    # It is NOT described as an isolated robustness-only intervention.
    return F.smooth_l1_loss(xy[valid]*scale,gt[valid]*scale,beta=.01,reduction='none').mean()
def retention_objective(xy,gt,valid,full_gt,aux_valid,pixels,kind,weight=.25):
    if aux_valid[:,KP].any():raise ValueError('Auxiliary includes primary points')
    if kind=='real' and aux_valid.any():raise ValueError('Real auxiliary forbidden')
    if kind=='dual' and aux_valid[:,7:14].any():raise ValueError('Dual tail forbidden')
    primary=objective(xy[:,KP],gt,valid,pixels,'pixel_huber')
    aux=objective(xy,full_gt,aux_valid,pixels,'pixel_huber') if aux_valid.any() else xy.sum()*0
    return primary+weight*aux
def main():
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));assert now()<deadline
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2);torch.manual_seed(SEED);torch.backends.cuda.matmul.allow_tf32=False
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='synthetic_supervision_retention_running',active_job=OUT.name,active_pid=os.getpid(),updated_utc=now().isoformat());write_json(CYCLE/'state.json',state)
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=now().isoformat(),**kw))
    status('loading');source=json.loads((BASE/'protocol.json').read_text());rows=prepare_rows();schedule=source['schedule'];fit=source['fit_monitor'];cal=source['calibration'];val=source['validation'];assert len(schedule)==400
    arms={'huber_low_retention':('pixel_huber','low')}
    full_gt,full_valid=full_targets(rows);aux_valid=full_valid.clone();aux_valid[:,:,KP]=False
    for i,r in enumerate(rows):
        if r['metadata']['kind']=='real':aux_valid[i]=False
        if r['metadata']['kind']=='dual':assert not aux_valid[i,:,7:14].any()
    assert not aux_valid[:,:,KP].any()
    protocol=dict(seed=17,steps=400,arms=arms,reused_control='pose_recipe_factorial_v1/huber_low_step0400.pt; same initial, schedule, primary loss and optimizer, unchanged control predictions.',
        hypothesis='Narrow5point updates degraded other existing synthetic GT landmarks. Add separate .25 auxiliary known-coordinate loss without diluting primary5point denominator.',
        primary='Exactly huber_low pixel/768SmoothL1 beta.01, same qualified nose4feet, perimagepointmean, fourimagebatchmean.',
        auxiliary='Only existing matched-qualified synthetic non-target points; single18indices, dual11because tail7-13zero-basedoff. Realaux alwayszero, no realback/tail/abdomen coordinate targets added. Knownvalid points only; annotations immutable, no pseudoGT. Loss primary+.25aux, eachownvalidpointmean perimage thenmean4.',
        fixed='Original400schedule2real1single1dual, originalviews, actualRGB768, all6/138queries, frozen originalpredictedboxesnotGTcrop. Train same1116928poseparams; lowspatial5e-5->1e-5/decoder1e-5->1e-6,wd.01clip1; encoder/LoRA/detector/coarse/queries frozen.',
        evaluation='fit72/cal220every100; final400beforeVAL144. Report primary and non-targetaccuracy separately; noautomaticextension/promotion. Reuseddevelopment, conditionalerrors notAP.',
        split_sha256=fingerprint(BASE/'protocol.json'),initial_sha256=fingerprint(BASE/'initial.pt'),control_sha256=fingerprint(REF/'huber_low_step0400.pt'),script_sha256=fingerprint(Path(__file__)),training_stop_utc=plan['training_stop_utc'],
        auxiliary_fit_points=sum(int(aux_valid[i].sum()) for i,r in enumerate(rows) if r['metadata']['stage']=='fit'))
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
                r=rows[i];xy=forward(i)
                loss=retention_objective(xy,r['gt'].cuda(),r['valid'].cuda(),full_gt[i].cuda(),aux_valid[i].cuda(),r['pixels'].cuda(),r['metadata']['kind'])/4
                assert torch.isfinite(loss);loss.backward();total+=float(loss.detach());del loss,xy
            assert all(p0.grad is None for _,p0 in runtime.named_all_parameters() if not p0.requires_grad)
            torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True);opt.step();torch.cuda.synchronize();seconds+=time.monotonic()-t0;peak=max(peak,torch.cuda.max_memory_allocated());losses.append(total)
            if step%100==0:
                p=evaluate(fit+cal);curves.append(dict(arm=arm,step=step,fit=metrics(rows,p,fit),calibration=metrics(rows,p,cal),recent_loss=sum(losses[-100:])/100));write_json(OUT/'curves.json',curves)
                atomic_torch(OUT/f'{arm}_step{step:04d}.pt',dict(pose={k:v.cpu() for k,v in runtime.pose.state_dict().items()},optimizer=opt.state_dict(),step=step,arm=arm));print(arm,step,'realcal',curves[-1]['calibration']['real']['mean'],'bottomcal',curves[-1]['calibration']['real_bottom_feet']['mean'],flush=True)
        performance[arm]=dict(train_seconds=seconds,total_seconds=time.monotonic()-start,peak_gpu_bytes=peak);write_json(OUT/'performance.json',performance);del opt
    old=read(REF/'predictions.pt');oldsummary=json.loads((REF/'results.json').read_text())['summary'];predictions={k:old['predictions'][k] for k in ['baseline','huber_low']};summary={k:oldsummary[k] for k in predictions}
    for arm in arms:
        status('final_evaluation',arm=arm);runtime.pose.load_state_dict(read(OUT/f'{arm}_step0400.pt')['pose']);p=evaluate(fit+cal+val);predictions[arm]=p;summary[arm]={stage:metrics(rows,p,ids) for stage,ids in [('fit_monitor',fit),('calibration',cal),('validation',val)]}
    after=frozen_digest(runtime);assert before==after
    torch.save(dict(predictions=predictions,evaluated_ids=fit+cal+val),OUT/'predictions.pt');write_json(OUT/'results.json',dict(summary=summary,performance=performance,frozen_before=before,frozen_after=after,production_changed=False))
    status('complete');state.update(stage='synthetic_supervision_retention_complete',active_job=None,active_pid=None,updated_utc=now().isoformat(),next='Compare matchedhuber_low vsretention400: primaryfeet andexisting syntheticnon-targetaccuracy separately, curvesandRGB. .25auxweightprespecified, no newrealGT ordualtail. No blindextension orpromotion.');state['completed'].append('synthetic_supervision_retention400');write_json(CYCLE/'state.json',state);print('COMPLETE',flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write_json(OUT/'failure.json',dict(traceback=traceback.format_exc(),utc=now().isoformat()))
        raise
