"""Batch-aware geometry through explicit per-sample objectives.

The batch axis never becomes a camera/time axis. Each sample retains its own
gauge, masks, correspondence support and mean reductions. Eight ordered sample
objectives are summed with coefficient1/8; physical grouping only shares model
forward/backward launches.
"""
from __future__ import annotations
import collections
import math
import time
import torch
import pure_geometry_bottom_20261018 as geo


def groups(micros,cap):
    assert len(micros)==8 and cap in (1,2,4)
    begin=0
    while begin<len(micros):
        end=begin+1
        while end<len(micros) and end-begin<cap and len(micros[end]['views'])==len(micros[begin]['views']):
            end+=1
        yield micros[begin:end]
        begin=end


def model_images(samples):
    assert samples and all(b['images'].shape[0]==1 for b in samples)
    shape=samples[0]['images'].shape[1:]
    assert all(b['images'].shape[1:]==shape for b in samples)
    return torch.cat([b['images'] for b in samples],dim=0)


def sample_prediction(pred,index,count):
    # Only these predictions enter the geometry objectives. Metadata lives in
    # the per-sample list; no merged body/valid/point-count denominator is used.
    assert pred['pose_enc'].shape[0]==pred['depth'].shape[0]==count
    return {k:pred[k][index:index+1] for k in ('pose_enc','depth')}


def per_sample_terms(pred,samples):
    return [geo.terms(sample_prediction(pred,i,len(samples)),b) for i,b in enumerate(samples)]


def geometry_forward(model,samples):
    """Keep BF16 encoder arithmetic sample-wise; batch FP32 geometry heads.

    Full encoder batching exceeded the registered numerical tolerances. Its
    batch-dependent BF16 rounding compounds over the frozen feature stack.
    Encoder graphs for the trainable last blocks stay connected and their
    gradients still accumulate from every sample's batched-head objective.
    """
    if len(samples)==1:return model(samples[0]['images'])
    amp_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    caches=[];start=None
    for b in samples:
        with torch.autocast(device_type='cuda',dtype=amp_dtype):
            features,index=model.aggregator(b['images'])
        assert start is None or start==index
        start=index;caches.append(features)
    merged=[]
    for layer in range(len(caches[0])):
        values=[f[layer] for f in caches]
        assert all((v is None)==(values[0] is None) for v in values)
        merged.append(None if values[0] is None else torch.cat(values,dim=0))
    with torch.autocast(device_type='cuda',enabled=False):
        pose=model.camera_head(merged,patch_token_start=start)
        depth,confidence=model.dense_head(merged,images=model_images(samples),patch_token_start=start)
    return dict(pose_enc=pose,depth=depth,depth_conf=confidence)


def objective(losses,weights,total_samples=8):
    # Missing tasks contribute zero for that sample; never divide by the number
    # of samples that happen to have a task or by all valid pixels in the batch.
    assert total_samples==8
    return sum(sum(weights[k]*value for k,value in terms.items())/total_samples for terms in losses)


def update(model,opt,shared,params,ema,weights,exposures,feed,micros,step,updates,cap,instrument=True,
           gradient_observer=None):
    assert len(micros)==8
    opt.zero_grad(set_to_none=True);totals=collections.Counter();details=[];wait=0.;sizes=[]
    warm=min(1.,(step+1)/500);progress=max(0.,(step-500)/max(1,updates-500))
    factor=warm*(.1+.9*(1+math.cos(math.pi*min(progress,1)))/2)
    for pg in opt.param_groups:pg['lr']=pg['base_lr']*factor
    for group in groups(micros,cap):
        batches=[]
        for item in group:
            began=time.perf_counter();batches.append(feed.next(item));wait+=time.perf_counter()-began
        pred=geometry_forward(model,batches);losses=per_sample_terms(pred,batches);sizes.append(len(group))
        for item,terms in zip(group,losses):
            if instrument and step%100==0:
                norms,cosines=geo.task_gradients(terms,shared)
                details.append(dict(role=item['sampling_role'],raw_gradient_norms=norms,
                    weighted_gradient_norms={k:v*weights[k] for k,v in norms.items()},cosines=cosines))
        loss=objective(losses,weights,8)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite batched objective')
        loss.backward()
        for item,terms in zip(group,losses):
            exposures[item['sampling_role']]+=1
            for k,v in terms.items():totals[k]+=float(v.detach())/8
        del pred,losses,loss,batches
    if gradient_observer is not None:gradient_observer(params)
    norm=torch.nn.utils.clip_grad_norm_([p for p in params.values() if p.requires_grad],10.)
    if not torch.isfinite(norm):raise FloatingPointError('Nonfinite accumulated gradient')
    opt.step();step+=1;decay=min(.999,(step+1)/(step+10))
    with torch.no_grad():
        for n,value in ema.items():value.lerp_(params[n].detach(),1-decay)
    return dict(utc=geo.utc(),step=step,losses=dict(totals),gradient_norm=float(norm),lr_factor=factor,
        ema_decay=decay,gradient_details=details,loader_wait_s=wait,physical_batch_cap=cap,
        physical_group_sizes=sizes,effective_batch=8)
