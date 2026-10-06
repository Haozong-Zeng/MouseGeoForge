"""Data-only side-azimuth coverage control; preserve pose/cohort/bottom exposure."""
import collections,datetime as dt,fcntl,json,math,os,random,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np,torch
from train_pixel_adapter_matched_20261005 import prepare_rows,metrics,read,KP,STEPS,SEED
from pixel_adaptation_runtime import PixelAdaptationRuntime
from preflight_pixel_adapter_20261005 import frozen_digest
from run_surface_curve_pilot_20261003 import load_item,atomic_torch
from run_depth_pose_ablation import write_json,fingerprint
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';BASE=CYCLE/'pixel_adapter_matched_pilot_v1';EXTRA=CYCLE/'single_train_allview_rgb_inputs_v1';OUT=CYCLE/'single_side_azimuth_control_v1'
def now():return dt.datetime.now(dt.timezone.utc)
def main():
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));assert now()<deadline
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2);torch.manual_seed(SEED);torch.backends.cuda.matmul.allow_tf32=False
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='single_side_azimuth_control_running',active_job=OUT.name,active_pid=os.getpid(),updated_utc=now().isoformat());write_json(CYCLE/'state.json',state)
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),updated_utc=now().isoformat(),**kw))
    original_protocol=json.loads((BASE/'protocol.json').read_text());rows=prepare_rows();original_count=len(rows);lookup={(r['metadata']['scene'],r['metadata']['view']):i for i,r in enumerate(rows)}
    expanded=read(EXTRA/'rows.pt');candidates=collections.defaultdict(list);orientations={}
    for r in expanded:
        m=r['metadata'];key=m['scene'],m['view'];orientation=m['synthetic_orientation']
        if key not in lookup:lookup[key]=len(rows);rows.append(r)
        i=lookup[key];candidates[m['scene'],orientation].append(i);orientations[key]=orientation
    rng=random.Random(71);schedule=[];changes=[];before_counts=collections.Counter();after_counts=collections.Counter()
    for step,ids in enumerate(original_protocol['schedule'],1):
        old=ids[2];m=rows[old]['metadata'];assert m['stage']=='fit' and m['kind']=='single';o=orientations[m['scene'],m['view']]
        new=rng.choice(sorted(candidates[m['scene'],o]));n=rows[new]['metadata'];assert n['scene']==m['scene']
        new_ids=ids[:2]+[new]+ids[3:];assert new_ids[:2]==ids[:2] and new_ids[3:]==ids[3:];schedule.append(new_ids)
        before_counts[o]+=1;after_counts[orientations[n['scene'],n['view']]]+=1
        changes.append(dict(step=step,original_index=old,new_index=new,scene=m['scene'],old_view=m['view'],new_view=n['view'],orientation=o,known_points=int(rows[new]['valid'].sum())))
    assert before_counts==after_counts and all(c['old_view']==c['new_view'] for c in changes if c['orientation']!='side')
    fit=original_protocol['fit_monitor'];cal=original_protocol['calibration'];val=original_protocol['validation']
    protocol=dict(seed=SEED,camera_draw_seed=71,steps=400,architecture='Exactly originalhead-only:1,116,928pose decoder+spatialparameters, no LoRA. Queries/coarse/originalencoder/detector frozen.',
        control='pixel_adapter_matched_pilot_v1/head_only_step0400.pt; same originalinitial.pt, AdamW1e-4cosfloor.1wd.01clip1, exact real2/dual1 slots andsinglepose sequence of400schedule. Only syntheticSIDE cameraazimuth draw changes.',
        exposure='Preserve every originalsinglemouse scene andphysicalorientation draw, includingallbottom/top exposures. Cylinder21poses sidepool6cameras; Homecage22poses sidepool3. Existingbottom/topsingleton views unchanged. NewCylinderTOP notadded since it wouldchangeorientationmix;16/21top lackboxqualification is separatelyrecorded.',
        data='same43 existingTRAINmoseq poses; no newframes/sourcevideos/labels/cal/VAL, noRealCapture21. All additionalviews independentlyencodedV1. OriginalpredboxesnotGT. All6queries138tokensforward.',
        loss='Same normalizedXYMSE nose4feet; originalscore>=.5 maskboxIoU>=.5knownlabelafterforward. Any selectednewviewwithzeroqualifiedpoints stillforwarded withzero image loss (not resampled), batchlossdivides4 unchanged. Detection remainsfrozen.',
        evaluation='Same72fixedfit and220cal every100; final400VAL only. Reused development, one seed, no promotion.',
        orientation_draws=dict(before_counts),changed_draws=sum(c['old_view']!=c['new_view'] for c in changes),zero_known_draws=sum(c['known_points']==0 for c in changes),changes=changes,schedule=schedule,
        source_manifest_sha256=fingerprint(EXTRA/'manifest.json'),original_protocol_sha256=fingerprint(BASE/'protocol.json'),script_sha256=fingerprint(Path(__file__)),training_stop_utc=plan['training_stop_utc'])
    write_json(OUT/'protocol.json',protocol);status('loading_rgb',images=0);cache={};hashes={}
    # Load only original eval/fit-monitor and images actually sampled; unused new views do not train.
    required=sorted(set(fit+cal+val+sum(schedule,[])))
    for j,i in enumerate(required):
        if now()>=deadline:raise TimeoutError('Deadline RGB load')
        r=rows[i];key=(r['record']['scene'],tuple(r['record']['views']))
        if key not in cache:
            item=load_item(r['record']);cache[key]={v:item['batch']['images'][:,k:k+1].contiguous().clone() for k,v in enumerate(r['record']['views'])};hashes.update(item['source_hashes'])
        r['image']=cache[key][r['metadata']['view']]
        if (j+1)%200==0:status('loading_rgb',images=j+1,total=len(required));print('RGB',j+1,len(required),flush=True)
    write_json(OUT/'source_hashes.json',hashes)
    runtime=PixelAdaptationRuntime(rank=8);initial=read(BASE/'initial.pt');runtime.pose.load_state_dict(initial['pose'])
    with torch.no_grad():
        for k,v in initial['adapter'].items():getattr(runtime.adapter,k).weight.copy_(v)
    runtime.adapter.enabled=False;runtime.adapter.a.requires_grad_(False);runtime.adapter.b.requires_grad_(False)
    runtime.pose.spatial_readout.requires_grad_(True);runtime.pose.set_decoder_trainable(True);params=[p for _,p in runtime.named_all_parameters() if p.requires_grad];assert sum(p.numel() for p in params)==1116928
    before=frozen_digest(runtime);baseline=torch.stack([r['baseline']['xy'] for r in rows]);curves=[]
    def forward(i):
        r=rows[i];return runtime.native_forward(r['image'].cuda(),r['baseline']['boxes'][None].cuda(),allow_input_grad=False)['xy'][0]
    @torch.no_grad()
    def evaluate(ids):
        p=baseline.clone()
        for i in ids:
            if now()>=deadline:raise TimeoutError('Deadline evaluation')
            p[i]=forward(i).cpu()
        return p
    p=evaluate(fit+cal);delta=float((p[fit+cal]-baseline[fit+cal]).abs().max());assert delta<2e-5
    curves.append(dict(step=0,fit=metrics(rows,p,fit),calibration=metrics(rows,p,cal),initial_delta=delta));write_json(OUT/'curves.json',curves)
    torch.manual_seed(SEED);opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01);seconds=0.;peak=0;losses=[];start=time.monotonic()
    for step,ids in enumerate(schedule,1):
        if now()>=deadline:raise TimeoutError('Training cutoff')
        status('training',step=step-1,total=400);torch.cuda.synchronize();t0=time.monotonic();torch.cuda.reset_peak_memory_stats();opt.zero_grad(set_to_none=True)
        factor=.1+.9*.5*(1+math.cos(math.pi*(step-1)/400))
        for g in opt.param_groups:g['lr']=1e-4*factor
        total=0.
        for i in ids:
            xy=forward(i);r=rows[i];mask=r['valid'].cuda()
            loss=(xy[:,KP]-r['gt'].cuda()).square().sum(-1)[mask].mean()/4 if mask.any() else xy.sum()*0
            assert torch.isfinite(loss);loss.backward();total+=float(loss.detach());del xy,loss
        assert all(p0.grad is None for n,p0 in runtime.named_all_parameters() if not p0.requires_grad)
        torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True);opt.step();torch.cuda.synchronize();seconds+=time.monotonic()-t0;peak=max(peak,torch.cuda.max_memory_allocated());losses.append(total)
        if step%25==0:print('azimuth',step,'loss',float(np.mean(losses[-25:])),flush=True)
        if step%100==0:
            p=evaluate(fit+cal);curves.append(dict(step=step,fit=metrics(rows,p,fit),calibration=metrics(rows,p,cal),recent_loss=float(np.mean(losses[-100:]))));write_json(OUT/'curves.json',curves)
            atomic_torch(OUT/f'azimuth_step{step:04d}.pt',dict(pose={k:v.cpu() for k,v in runtime.pose.state_dict().items()},optimizer=opt.state_dict(),step=step,seed=SEED))
    status('final_evaluation');p=evaluate(fit+cal+val);after=frozen_digest(runtime);assert before==after
    summary={stage:metrics(rows,p,ids) for stage,ids in [('fit_monitor',fit),('calibration',cal),('validation',val)]}
    torch.save(dict(predictions=p[:original_count],evaluated_ids=fit+cal+val),OUT/'predictions.pt')
    write_json(OUT/'results.json',dict(summary=summary,frozen_before=before,frozen_after=after,train_seconds=seconds,total_seconds=time.monotonic()-start,peak_gpu_bytes=peak,losses=losses,claim='Data-side-azimuth-only400 control versusoriginalhead-only400. Samepose/orientation/cohortexposure; qualifications maydiffer fornewviews, zero-knowncountreported. No promotion.'))
    status('complete');state.update(stage='single_side_azimuth_control_complete',active_job=None,active_pid=None,next='Inspect matchedhead-only versusnewsideazimuth curves,conditionalmetrics andRGB. Do not infernewvideos orbottomtrainingincrease: bottomorientation exposure unchanged. No blindextensions.',updated_utc=now().isoformat());state['completed'].append('single_side_azimuth_matched400');write_json(CYCLE/'state.json',state);print('COMPLETE',json.dumps(summary),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write_json(OUT/'failure.json',dict(traceback=traceback.format_exc()))
        raise

