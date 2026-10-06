"""Matched actual-RGB head-only versus image-backbone LoRA conditional pose pilot."""
import collections,datetime as dt,fcntl,json,math,os,random,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from pixel_adaptation_runtime import PixelAdaptationRuntime
from preflight_pixel_adapter_20261005 import frozen_digest
from run_surface_curve_pilot_20261003 import load_item,atomic_torch
from mousegeoforge.detection_readout_loss import match_detection
from mousegeoforge.pilot_metrics import box_iou
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';OUT=CYCLE/'pixel_adapter_matched_pilot_v1';KP=[0,14,15,16,17];STEPS=400;SEED=17

def now():return dt.datetime.now(dt.timezone.utc)
def read(path):return torch.load(path,map_location='cpu',weights_only=False)
def status(stage,**kwargs):
    write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),updated_utc=now().isoformat(),**kwargs))
def prepare_rows():
    quality=read(CYCLE/'point_quality_inputs_v1/inputs.pt');lookup={}
    for role in ['train','validation']:
        old=read(CYCLE/'singleview_fusion_inputs_seed17_v1'/f'{role}.pt')
        for i,m in enumerate(old['metadata']):
            lookup[role,m['record']['scene'],m['view']]=dict(record=m['record'],baseline={k:old[k][i] for k in ['xy','boxes','objectness_logits']},pixels=old['pixels'][i],target={k:old[k][i] for k in ['gt_xy','gt_boxes','point_valid','box_valid','instance_valid']})
        folder='dual_train_allview_affinity_context_v1' if role=='train' else 'dual_allscene_association_audit_v1'
        for g in read(CYCLE/folder/'groups.pt'):
            for vi,v in enumerate(g['record']['views']):
                lookup[role,g['record']['scene'],v]=dict(record=g['record'],baseline={k:g['predictions'][k][vi] for k in ['xy','boxes','objectness_logits']},pixels=torch.tensor(g['pixels']).float(),target={k:g['evaluation_data'][k][vi] for k in ['gt_xy','gt_boxes','point_valid','box_valid','instance_valid']})
    rows=[]
    for i,m in enumerate(quality['metadata']):
        r=lookup[m['role'],m['scene'],m['view']];d=r['target'];b=r['baseline']
        qs,ts=match_detection(dict(instance_boxes_cxcywh=b['boxes'][None],instance_objectness_logits=b['objectness_logits'][None]),d['gt_boxes'][None],(d['box_valid']&d['instance_valid'])[None])[0]
        target=torch.zeros(6,5,2);valid=torch.zeros(6,5,dtype=torch.bool)
        for q,t in zip(qs.tolist(),ts.tolist()):
            if b['objectness_logits'][q].sigmoid()<.5 or box_iou(b['boxes'][q:q+1],d['gt_boxes'][t:t+1])[0,0]<.5:continue
            target[q]=d['gt_xy'][t,KP];valid[q]=d['point_valid'][t,KP]
        assert torch.equal(valid,quality['valid'][i])
        assert float((b['xy']-quality['xy'][i]).abs().max())<1e-5
        rows.append(dict(metadata=m,record=r['record'],baseline=b,pixels=r['pixels'].float(),gt=target,valid=valid))
    assert len(rows)==1388
    return rows

def metrics(rows,predictions,ids):
    answer={}
    for kind in ['all','real','single','dual','real_bottom_feet']:
        errors=[];known=0;confident=0
        for i in ids:
            m=rows[i]['metadata']
            if kind=='real_bottom_feet':
                if not m['physical_bottom']:continue
            elif kind!='all' and m['kind']!=kind:continue
            r=rows[i];v=r['valid'].clone();c=(r['baseline']['objectness_logits'].sigmoid()>=.5)[:,None].expand(6,5).clone()
            if kind=='real_bottom_feet':v[:,0]=False;c[:,0]=False
            e=((predictions[i][:,KP]-r['gt'])*r['pixels']).norm(dim=-1)
            errors.extend(e[v].tolist());known+=int(v.sum());confident+=int(c.sum())
        a=np.asarray(errors)
        answer[kind]=dict(known=known,confident=confident,unknown=confident-known,mean=float(a.mean()) if len(a) else None,p95=float(np.quantile(a,.95)) if len(a) else None,pck32=float((a<=32).mean()) if len(a) else None)
    return answer

def main():
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'))
    assert now()<deadline
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2);torch.manual_seed(SEED);torch.backends.cuda.matmul.allow_tf32=False
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='pixel_adapter_matched_pilot_running',active_job=OUT.name,active_pid=os.getpid(),updated_utc=now().isoformat());write_json(CYCLE/'state.json',state)
    rows=prepare_rows()
    pools={k:[i for i,r in enumerate(rows) if r['metadata']['stage']=='fit' and r['metadata']['kind']==k and r['valid'].any()] for k in ['real','single','dual']}
    rng=random.Random(SEED);schedule=[rng.sample(pools['real'],2)+rng.sample(pools['single'],1)+rng.sample(pools['dual'],1) for _ in range(STEPS)]
    fit_monitor=sorted(sum([random.Random(SEED).sample(pools[k],min(24,len(pools[k]))) for k in pools],[]))
    cal=[i for i,r in enumerate(rows) if r['metadata']['stage']=='calibration'];val=[i for i,r in enumerate(rows) if r['metadata']['stage']=='validation']
    protocol=dict(seed=SEED,steps=STEPS,arms=['head_only','head_plus_lora'],batch='4images accumulated independently V1:2real1single1dual; same schedule/order, no augmentation',optimizer='AdamW pose1e-4, LoRA1e-4; cosine to0.1*initial, wd0.01, total gradnormclip1',
        pose_trainable='keypoint_decoder + spatial_readout; query embedding and coarse head frozen; all6instances/all138tokens forwarded',
        adapter='rank8 alpha8 on final DINOv3 image block qkv, zeroB;32768extra trainable parameters; no original backbone weights modified',
        loss='mean normalized squared Euclidean XY on nose+4feet, mean known points perimage then mean4images. Fixed original predicted boxes/logits; GTbox association only after baselineprediction; IoU>=.5,score>=.5. Missing labels excluded. Both arms same objective.',
        split='Reuse point_quality_inputs_v1 source-group fit/cal/validation. Curve21unverified sources omitted. No new data. Base pose already saw all TRAIN; only new adaptation held-cal. Reused FR2/AZR2L2 development NOT pristine external. Dual TRAINactionA vsVALactionB but sameassets.',
        scope='Conditional localization with frozen ORIGINAL predicted boxes (notGT). No detector improvement claim. No tail/back/abdomen supervision or pseudoGT. No production change.',
        controls='Actual floating RGB via original load_item, no roundedPNG; same head initial state botharms; original pose checkpoint + frozen baseline reported; extra LoRA param and compute are NOT capacity/compute matched. Same image-update budget only.',
        evaluation='All220cal +fixed72fit monitor at0/100/200/300/400; save both final400before VAL metrics. No earlystop or checkpointselection byVAL. Baseline matched qualified known points; report unknown counts. TRAINfit pools excludeimages with zero known points, withcounts.',
        pool_images={k:len(v) for k,v in pools.items()},fit_monitor=fit_monitor,calibration=cal,validation=val,schedule=schedule,
        deadline=plan['training_stop_utc'],inputs_sha256=fingerprint(CYCLE/'point_quality_inputs_v1/inputs.pt'),script_sha256=fingerprint(Path(__file__)))
    write_json(OUT/'protocol.json',protocol);status('loading_rgb',images=0)
    images={};hashes={}
    for i,r in enumerate(rows):
        if now()>=deadline:raise TimeoutError('Deadline during RGB loading')
        key=(r['record']['scene'],tuple(r['record']['views']))
        if key not in images:
            item=load_item(r['record']);views=r['record']['views']
            images[key]={v:item['batch']['images'][:,j:j+1].contiguous().clone() for j,v in enumerate(views)};hashes.update(item['source_hashes'])
        r['image']=images[key][r['metadata']['view']]
        if (i+1)%100==0:status('loading_rgb',images=i+1);print('RGB',i+1,flush=True)
    write_json(OUT/'source_hashes.json',hashes)
    runtime=PixelAdaptationRuntime(rank=8);runtime.pose.spatial_readout.requires_grad_(True);runtime.pose.set_decoder_trainable(True)
    pose_params=[p for p in runtime.pose.parameters() if p.requires_grad]
    initial_pose={k:v.detach().cpu().clone() for k,v in runtime.pose.state_dict().items()}
    initial_adapter={k:getattr(runtime.adapter,k).weight.detach().cpu().clone() for k in ['a','b']}
    torch.save(dict(pose=initial_pose,adapter=initial_adapter),OUT/'initial.pt')
    before=frozen_digest(runtime);write_json(OUT/'trainable.json',dict(head_params=sum(p.numel() for p in pose_params),lora_params=32768,frozen_before=before,source_sha256=runtime.hashes,names=[n for n,p in runtime.named_all_parameters() if p.requires_grad]))
    baseline=torch.stack([r['baseline']['xy'] for r in rows]);all_predictions={'baseline':baseline};curves=[];performance={}
    def forward(i,adapt):
        r=rows[i]
        return runtime.native_forward(r['image'].cuda(),r['baseline']['boxes'][None].cuda(),allow_input_grad=adapt)['xy'][0]
    @torch.no_grad()
    def evaluate(ids,adapt):
        result=baseline.clone()
        for i in ids:
            if now()>=deadline:raise TimeoutError('Deadline during eval')
            result[i]=forward(i,adapt).cpu()
        return result
    for arm in protocol['arms']:
        runtime.pose.load_state_dict(initial_pose)
        with torch.no_grad():
            for k,v in initial_adapter.items():getattr(runtime.adapter,k).weight.copy_(v)
        adapt=arm=='head_plus_lora';runtime.adapter.enabled=adapt
        for _,p in runtime.named_all_parameters():p.grad=None
        opt=torch.optim.AdamW([dict(params=pose_params,lr=1e-4)]+([dict(params=list(runtime.adapter.a.parameters())+list(runtime.adapter.b.parameters()),lr=1e-4)] if adapt else []),weight_decay=.01)
        torch.manual_seed(SEED);runtime.pose.eval()
        start=time.monotonic();train_seconds=0.;max_memory=0;losses=[]
        p=evaluate(fit_monitor+cal,adapt);delta=float((p[fit_monitor+cal]-baseline[fit_monitor+cal]).abs().max());assert delta<2e-5,delta
        curves.append(dict(arm=arm,step=0,fit=metrics(rows,p,fit_monitor),calibration=metrics(rows,p,cal),initial_max_abs_delta=delta));write_json(OUT/'curves.json',curves)
        for step,indices in enumerate(schedule,1):
            if now()>=deadline:raise TimeoutError('Training cutoff')
            status('training',arm=arm,step=step-1,total=STEPS)
            torch.cuda.synchronize();t0=time.monotonic();torch.cuda.reset_peak_memory_stats();opt.zero_grad(set_to_none=True)
            factor=.1+.9*.5*(1+math.cos(math.pi*(step-1)/STEPS))
            for group in opt.param_groups:group['lr']=1e-4*factor
            total=0.
            for i in indices:
                xy=forward(i,adapt);r=rows[i];mask=r['valid'].cuda()
                loss=(xy[:,KP]-r['gt'].cuda()).square().sum(-1)[mask].mean()/len(indices)
                assert torch.isfinite(loss);loss.backward();total+=float(loss.detach());del loss,xy
            if step==2 and adapt:
                for p0 in list(runtime.adapter.a.parameters())+list(runtime.adapter.b.parameters()):assert p0.grad is not None and torch.isfinite(p0.grad).all() and p0.grad.norm()>0
            assert all(p0.grad is None for _,p0 in runtime.named_all_parameters() if not p0.requires_grad)
            params=pose_params+(list(runtime.adapter.a.parameters())+list(runtime.adapter.b.parameters()) if adapt else [])
            torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True);opt.step()
            torch.cuda.synchronize();train_seconds+=time.monotonic()-t0;max_memory=max(max_memory,torch.cuda.max_memory_allocated());losses.append(total)
            if step%25==0:print(arm,step,'loss',float(np.mean(losses[-25:])),'training_seconds',train_seconds,flush=True)
            if step%100==0:
                p=evaluate(fit_monitor+cal,adapt);curves.append(dict(arm=arm,step=step,fit=metrics(rows,p,fit_monitor),calibration=metrics(rows,p,cal),train_loss_recent100=float(np.mean(losses[-100:]))));write_json(OUT/'curves.json',curves)
                atomic_torch(OUT/f'{arm}_step{step:04d}.pt',dict(pose={k:v.cpu() for k,v in runtime.pose.state_dict().items()},adapter={k:getattr(runtime.adapter,k).weight.detach().cpu() for k in ['a','b']},step=step,seed=SEED,arm=arm,optimizer=opt.state_dict()))
        performance[arm]=dict(train_seconds=train_seconds,total_seconds=time.monotonic()-start,peak_gpu_bytes=max_memory,loss=losses)
        write_json(OUT/'performance.json',performance)
        del opt;p=None;torch.cuda.empty_cache()
    # Only now, after both fixed-step checkpoints exist, evaluate developmentVAL.
    summaries={'baseline':{stage:metrics(rows,baseline,ids) for stage,ids in [('fit_monitor',fit_monitor),('calibration',cal),('validation',val)]}}
    for arm in protocol['arms']:
        checkpoint=read(OUT/f'{arm}_step0400.pt');runtime.pose.load_state_dict(checkpoint['pose'])
        with torch.no_grad():
            for k,v in checkpoint['adapter'].items():getattr(runtime.adapter,k).weight.copy_(v)
        adapt=arm=='head_plus_lora';runtime.adapter.enabled=adapt;status('final_evaluation',arm=arm)
        ids=fit_monitor+cal+val;p=evaluate(ids,adapt);all_predictions[arm]=p
        summaries[arm]={stage:metrics(rows,p,subset) for stage,subset in [('fit_monitor',fit_monitor),('calibration',cal),('validation',val)]}
    after=frozen_digest(runtime);assert before==after
    torch.save(dict(predictions=all_predictions,metadata=[r['metadata'] for r in rows],gt=torch.stack([r['gt'] for r in rows]),valid=torch.stack([r['valid'] for r in rows]),pixels=torch.stack([r['pixels'] for r in rows]),evaluated_ids=fit_monitor+cal+val),OUT/'predictions.pt')
    write_json(OUT/'results.json',dict(summary=summaries,frozen_before=before,frozen_after=after,performance={k:{n:v for n,v in a.items() if n!='loss'} for k,a in performance.items()},claim='One matched seed, actualRGB conditional localization. Fixed400endpoint not necessarily converged. Reused development, no production promotion.'))
    status('complete');state.update(stage='pixel_adapter_matched_pilot_complete',active_job=None,active_pid=None,next='Inspect400step matched curves/metrics and RGB overlays before deciding extension or repeat; not a detector gain.',updated_utc=now().isoformat());state['completed'].append('pixel_adapter_actualRGB_matched400_seed17');write_json(CYCLE/'state.json',state)
    print('COMPLETE',json.dumps(summaries),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write_json(OUT/'failure.json',dict(traceback=traceback.format_exc()));status('failed')
        raise

