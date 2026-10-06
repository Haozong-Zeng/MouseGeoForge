"""Three-TRAIN-image pixel-backbone LoRA gradient and memory preflight only."""
import datetime as dt,fcntl,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import torch
from pixel_adaptation_runtime import PixelAdaptationRuntime
from run_surface_curve_pilot_20261003 import load_item
from mousegeoforge.detection_readout_loss import match_detection
from mousegeoforge.pilot_metrics import box_iou
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';OUT=CYCLE/'pixel_adapter_preflight_v1';KP=[0,14,15,16,17]


def frozen_digest(runtime):
    digest=hashlib.sha256();count=0
    for name,p in runtime.named_all_parameters():
        if p.requires_grad:continue
        digest.update(name.encode());digest.update(p.detach().cpu().contiguous().numpy().tobytes());count+=p.numel()
    return dict(sha256=digest.hexdigest(),parameters=count)


def loss_for(xy,baseline,d):
    boxes=baseline['boxes'].cpu();logits=baseline['objectness_logits'].cpu()
    qs,ts=match_detection(dict(instance_boxes_cxcywh=boxes[None],instance_objectness_logits=logits[None]),d['gt_boxes'][None],(d['box_valid']&d['instance_valid'])[None])[0]
    terms=[]
    for q,t in zip(qs.tolist(),ts.tolist()):
        if logits[q].sigmoid()<.5 or box_iou(boxes[q:q+1],d['gt_boxes'][t:t+1])[0,0]<.5:continue
        valid=d['point_valid'][t,KP];indices=[kp for j,kp in enumerate(KP) if valid[j]]
        if indices:terms.append((xy[0,q,indices]-d['gt_xy'][t,indices].cuda()).square().sum(-1).mean())
    assert terms,'No known TRAIN point in predeclared image'
    return torch.stack(terms).mean()


def main():
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));assert dt.datetime.now(dt.timezone.utc)<deadline
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2);torch.manual_seed(17);torch.backends.cuda.matmul.allow_tf32=False
    old=torch.load(CYCLE/'singleview_fusion_inputs_seed17_v1/train.pt',map_location='cpu',weights_only=False);cases=[]
    for label,predicate in [('real_bottom',lambda m:m['record']['kind']=='real' and m['view']==0),('single',lambda m:m['record'].get('source_group')=='moseq_sample10')]:
        i=next(i for i,m in enumerate(old['metadata']) if predicate(m));m=old['metadata'][i]
        cases.append(dict(label=label,record=m['record'],view=m['view'],target={k:old[k][i] for k in ['gt_boxes','gt_xy','box_valid','point_valid','instance_valid']},cached_xy=old['xy'][i]))
    dual=torch.load(CYCLE/'dual_train_allview_affinity_context_v1/groups.pt',map_location='cpu',weights_only=False)[0]
    cases.append(dict(label='dual_top',record=dual['record'],view=dual['record']['views'][0],target={k:dual['evaluation_data'][k][0] for k in ['gt_boxes','gt_xy','box_valid','point_valid','instance_valid']},cached_xy=dual['predictions']['xy'][0]))
    write_json(OUT/'protocol.json',dict(rank=8,alpha=8,trainable_location='aggregator.patch_embed.blocks.23.attn.qkv lowrank A/B only',trainable_params=32768,steps_per_case=2,optimizer='AdamW lr1e-4 wd0',cases=[dict(label=c['label'],record=c['record'],view=c['view']) for c in cases],
        scope='TRAIN-only feasibility, not accuracy trial. Original all1b weights frozen; last DINO image block actualRGB adaptation, all frozen Omega blocks retain backward graph. Frozen ORIGINAL predicted boxes anchor pose readout, notGT crops.3images2updates each, reset LoRA aftereverycase. No deployed checkpoint. Coarse prior remains detached, native-memory/instance inputgrad opt-in.',
        reference='Low-rank residual idea: https://github.com/microsoft/LoRA ; own small wrapper preserves base LinearKMaskedBias.forward, no merged/pretrained weights downloaded.'))
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='pixel_adapter_preflight_running',active_job=OUT.name,active_pid=os.getpid());write_json(CYCLE/'state.json',state)
    started=time.monotonic();runtime=PixelAdaptationRuntime(rank=8)
    trainable=[(n,p) for n,p in runtime.named_all_parameters() if p.requires_grad];assert len(trainable)==2 and sum(p.numel() for n,p in trainable)==32768
    frozen_before=frozen_digest(runtime);initial={k:getattr(runtime.adapter,k).weight.detach().clone() for k in ['a','b']}
    torch.save({k:v.cpu() for k,v in initial.items()},OUT/'initial_lowrank.pt');results=[]
    print('LOADED',dict(trainable=[n for n,p in trainable],frozen=frozen_before),flush=True)
    for case in cases:
        assert dt.datetime.now(dt.timezone.utc)<deadline
        item=load_item(case['record']);vi=case['record']['views'].index(case['view']);images=item['batch']['images'][:,vi:vi+1].cuda()
        runtime.adapter.enabled=False
        with torch.no_grad():
            reference=runtime.from_images(images);reference_delta=float((reference['xy'].cpu()-case['cached_xy']).abs().max());assert reference_delta<1e-5
            boxes=reference['boxes'][None].detach()
            custom=runtime.native_forward(images,boxes)['xy'][0];native_delta=float((custom-reference['xy']).abs().max());assert native_delta<1e-6
            runtime.adapter.enabled=True;zero=runtime.native_forward(images,boxes,allow_input_grad=True)['xy'][0];initial_delta=float((zero-reference['xy']).abs().max());assert initial_delta<1e-6
        blocked=runtime.native_forward(images,boxes,allow_input_grad=False)['xy'];assert not blocked.requires_grad;del blocked
        opt=torch.optim.AdamW([p for n,p in trainable],lr=1e-4,weight_decay=0);steps=[]
        for step in range(2):
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t0=time.monotonic();opt.zero_grad(set_to_none=True)
            prediction=runtime.native_forward(images,boxes,allow_input_grad=True)
            # Labels only enter the loss, after the RGB-only prediction.
            loss=loss_for(prediction['xy'],reference,case['target']);assert loss.requires_grad and torch.isfinite(loss)
            loss.backward();norms={n:float(p.grad.float().norm()) if p.grad is not None else None for n,p in trainable}
            assert all(v is not None and torch.isfinite(torch.tensor(v)) for v in norms.values()) and norms[trainable[1][0]]>0,norms
            if step==1:assert norms[trainable[0][0]]>0,norms
            assert all(p.grad is None for n,p in runtime.named_all_parameters() if not p.requires_grad)
            torch.nn.utils.clip_grad_norm_([p for n,p in trainable],1,error_if_nonfinite=True);opt.step();torch.cuda.synchronize()
            steps.append(dict(step=step,loss=float(loss.detach()),gradient_norms=norms,seconds=time.monotonic()-t0,peak_gpu_bytes=torch.cuda.max_memory_allocated()))
            print(case['label'],steps[-1],flush=True);del prediction,loss
        runtime.adapter.enabled=False
        with torch.no_grad():disabled=runtime.native_forward(images,boxes)['xy'][0];disabled_delta=float((disabled-reference['xy']).abs().max());assert disabled_delta<1e-6
        results.append(dict(label=case['label'],shape=list(images.shape),reference_cached_delta=reference_delta,native_path_delta=native_delta,zero_adapter_delta=initial_delta,disabled_after_updates_delta=disabled_delta,blocked_route_has_gradient=False,steps=steps))
        with torch.no_grad():
            for k,v in initial.items():getattr(runtime.adapter,k).weight.copy_(v)
        del opt,images,reference,custom,zero,disabled,boxes;runtime.adapter.a.weight.grad=None;runtime.adapter.b.weight.grad=None;torch.cuda.empty_cache()
        write_json(OUT/'status.json',dict(stage='checking',pid=os.getpid(),cases_done=len(results)))
    frozen_after=frozen_digest(runtime);assert frozen_before==frozen_after
    write_json(OUT/'results.json',dict(cases=results,frozen_before=frozen_before,frozen_after=frozen_after,trainable_params=32768,elapsed_seconds=time.monotonic()-started,
        source_sha256=runtime.hashes,module_sha256=fingerprint(ROOT/'mousegeoforge/models/low_rank_linear.py'),runtime_sha256=fingerprint(ROOT/'tools/pixel_adaptation_runtime.py'),
        claim='Feasibility only: nonzero TRAIN loss gradients reach actualimagebackbone lowrank weights. Numerical tolerances recorded. No validation accuracy or production change; all learned trial updates reset/discarded.'))
    write_json(OUT/'status.json',dict(stage='complete',pid=os.getpid()));state.update(stage='pixel_adapter_preflight_complete',active_job=None,active_pid=None,next='Inspect gradient/memory/timing and default parity; design bounded matched pixel-adaptation trial with same source split, no new data confound.');state['completed'].append('pixel_adapter_rgb_gradient_preflight');write_json(CYCLE/'state.json',state)
    print('COMPLETE',json.dumps(results),flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise

