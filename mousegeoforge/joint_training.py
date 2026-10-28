"""Task-interleaved optimisation and complete resumable state, no job dispatch."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import random
import numpy as np
import torch


def _train_batch(batch):
    if batch.get('split')!='train':
        raise ValueError('Optimisation requires an explicitly admitted TRAIN batch')
    return batch['images']


def joint_train_step(model,optimizer,depth_batch,keypoint_batch,depth_loss,keypoint_loss,
                     *,depth_weight=1.,keypoint_weight=1.,clip_norm=1.):
    """One optimizer update, two registered task microbatches, live shared encoder.

    Callbacks receive (head output, original batch) and return the task scalar.
    The depth callback retains current mask/metric-anchor/plane terms; only its
    unreduced depth-error reduction may use part_balanced_depth. Domain/source
    weights and partial-label keypoint masking remain in these callbacks.
    Each callback's forward is independent; gradients accumulate before one
    step. The heads do not consume or invent each other's GT.
    """
    if not np.isfinite([depth_weight,keypoint_weight,clip_norm]).all() or min(depth_weight,keypoint_weight,clip_norm)<=0:
        raise ValueError('Positive finite task weights and clipping norm required')
    # Audit BOTH roles before doing any forward or backward.
    depth_rgb=_train_batch(depth_batch); keypoint_rgb=_train_batch(keypoint_batch)
    ids=[id(p) for g in optimizer.param_groups for p in g['params']]
    if len(ids)!=len(set(ids)):
        raise ValueError('Shared parameters must occur exactly once in the optimizer')
    expected={id(p) for p in model.parameters() if p.requires_grad}
    if set(ids)!=expected:
        raise ValueError('Optimizer must cover precisely the selected trainable model parameters')
    optimizer.zero_grad(set_to_none=True); model.train()
    values={}
    try:
        for task,batch,rgb,fn,weight in [('depth',depth_batch,depth_rgb,depth_loss,depth_weight),
                                       ('keypoints',keypoint_batch,keypoint_rgb,keypoint_loss,keypoint_weight)]:
            loss=fn(model(rgb,task=task)[task],batch)
            if loss.ndim!=0 or not torch.isfinite(loss):
                raise ValueError(f'{task} callback returned a nonfinite/non-scalar loss')
            (weight*loss).backward();values[task]=float(loss.detach())
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),clip_norm,error_if_nonfinite=True)
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        raise
    optimizer.step()
    return dict(**values,grad_norm=float(norm),optimizer_updates=1,task_microbatches=2)


@torch.no_grad()
def update_ema(ema_model,model,decay=.999):
    if not 0<=decay<1: raise ValueError('EMA decay must be in [0,1)')
    src=model.state_dict();dst=ema_model.state_dict()
    if src.keys()!=dst.keys(): raise ValueError('EMA topology mismatch')
    for name,value in src.items():
        if torch.is_floating_point(value):dst[name].lerp_(value,1-decay)
        else:dst[name].copy_(value)


def validate_identity(identity):
    required={'code_sha256','data_sha256','release_sha256','heads_sha256','recipe_sha256'}
    if set(identity)!=required or any(not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v) for v in identity.values()):
        raise ValueError('All five immutable SHA256 identities required')


def save_joint_checkpoint(path,model,optimizer,ema_model,*,step,samplers,identity):
    """Write raw/EMA/both heads/optimizer/RNG/sampler offsets atomically.

    identity must include immutable code/data/release/head/recipe hashes. Use
    only this routine after a complete successful optimizer+EMA update.
    """
    validate_identity(identity)
    if step<0 or not samplers: raise ValueError('Step and complete sampler state required')
    ns=np.random.get_state()
    payload=dict(format='mgf-shared-patch-two-heads-v1',model=model.state_dict(),ema=ema_model.state_dict(),
        optimizer=optimizer.state_dict(),step=step,samplers=copy.deepcopy(samplers),identity=dict(identity),
        python_rng=random.getstate(),numpy_rng=(ns[0],torch.from_numpy(ns[1].astype(np.int64)),ns[2],ns[3],ns[4]),
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if any(p.is_cuda for p in model.parameters()) else None)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    torch.save(payload,tmp);os.replace(tmp,path)


def load_joint_checkpoint(path,model,optimizer,ema_model,*,identity):
    validate_identity(identity)
    data=torch.load(path,map_location='cpu',weights_only=True)
    if data['format']!='mgf-shared-patch-two-heads-v1' or data['identity']!=identity:
        raise ValueError('Checkpoint recipe/data/source identity mismatch')
    uses_cuda=any(p.is_cuda for p in model.parameters())
    if (data['cuda_rng'] is None)!= (not uses_cuda):
        raise ValueError('Resume requires the same CPU/CUDA RNG topology')
    if data['cuda_rng'] is not None and len(data['cuda_rng'])!=torch.cuda.device_count():
        raise ValueError('CUDA device count changed across resume')
    model.load_state_dict(data['model'],strict=True);ema_model.load_state_dict(data['ema'],strict=True)
    optimizer.load_state_dict(data['optimizer']);random.setstate(data['python_rng'])
    name,keys,pos,has_gauss,cached=data['numpy_rng']
    np.random.set_state((name,keys.numpy().astype(np.uint32),pos,has_gauss,cached))
    torch.set_rng_state(data['torch_rng'])
    if data['cuda_rng'] is not None:torch.cuda.set_rng_state_all(data['cuda_rng'])
    return data['step'],data['samplers']
