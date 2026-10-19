"""CPU-only, ordered DataLoader for the pinned pure-geometry experiment.

Workers never initialize CUDA. A private generator keeps worker seeding from
consuming the model's RNG. No batching, augmentation, shuffling or drop_last is
introduced: each item is one already registered microbatch.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import pure_geometry_bottom_20261018 as geo

ROOT=geo.ROOT


def load_cpu(rec,manifest,pack):
    b=geo.old.load_item(rec,manifest,pack)
    views=geo.old.views_for(rec,manifest)
    if rec['kind']=='real_labeled':
        observed_scale=geo.old.median_camera_baseline(np.array([v['E'] for v in views]))
        b['E'][...,:3,3]*=observed_scale/geo.known_rig_scale()
    if any(v.get('truth') for v in views):
        scale=geo.old.median_camera_baseline(np.array([v['E'] for v in views]))
        for j,vid in enumerate(pack):
            v=next(v for v in views if v['id']==vid)
            with np.load(ROOT/v['truth']) as a:
                z=geo.old.resized(Image.fromarray(a['depth'].astype(np.float32)),512,True)[0]/scale
                mask=geo.old.resized(Image.fromarray(a['body'].astype(np.uint8)*255),512,True)[0]>0
                valid=geo.old.resized(Image.fromarray(a['valid'].astype(np.uint8)*255),512,True)[0]>0
                h,w=a['depth'].shape
                b['K'][0,j,0,2]+=(512/w-1)/2
                b['K'][0,j,1,2]+=(512/h-1)/2
            b['depth'][0,j]=torch.tensor(z)
            b['body'][0,j]=torch.tensor(mask)
            b['valid'][0,j]=torch.tensor(valid)
    if len(views)==1 and b['valid'].any():
        s=b['depth'][b['valid']].median().clamp_min(1e-6)
        b['depth']/=s
    assert all(v.device.type=='cpu' for v in b.values() if isinstance(v,torch.Tensor))
    return b


class ScheduledGeometry(Dataset):
    def __init__(self,manifest,items):
        self.manifest=manifest
        self.items=items

    def __len__(self):return len(self.items)

    def __getitem__(self,index):
        item=self.items[index]
        b=load_cpu(self.manifest['records'][item['index']],self.manifest,item['views'])
        return dict(sequence=index,item=item,batch=b)


def worker_init(_):
    torch.set_num_threads(1)
    geo.old.cv2.setNumThreads(1)
    assert not torch.cuda.is_initialized(), 'CUDA must stay in the parent trainer'


def loader(manifest,items,workers=4,pin_memory=True):
    assert workers in (0,1,2,4,8)
    options=dict(dataset=ScheduledGeometry(manifest,items),batch_size=None,
                 shuffle=False,num_workers=workers,pin_memory=pin_memory,
                 generator=torch.Generator().manual_seed(20261018),worker_init_fn=worker_init)
    if workers:
        options.update(multiprocessing_context='spawn',persistent_workers=True,
                       prefetch_factor=2,timeout=120)
    return DataLoader(**options)


def to_device(batch,device='cuda'):
    return {k:v.to(device,non_blocking=True) if isinstance(v,torch.Tensor) else v
            for k,v in batch.items()}


def digest(batch):
    h=hashlib.sha256()
    for key,value in sorted(batch.items()):
        h.update(key.encode())
        if isinstance(value,torch.Tensor):
            h.update(str((value.dtype,tuple(value.shape))).encode())
            h.update(value.contiguous().numpy().tobytes())
        else:h.update(json.dumps(value,sort_keys=True).encode())
    return h.hexdigest()


def benchmark(dest,samples=64):
    torch.set_num_threads(1)
    geo.old.cv2.setNumThreads(1)
    manifest=geo.read(geo.OUT/'manifest.json')
    registered=geo.read(geo.OUT/'geometry_seed17/schedule.json')
    items=[x for update in registered for x in update][:samples]
    assert samples>=32 and len(items)==samples
    # Obtain the unchanged pinned loader's CPU output without touching CUDA.
    original=geo.old.on_device
    geo.old.on_device=lambda b:b
    expected=[]
    try:
        for item in items:
            b=geo.load_item(manifest['records'][item['index']],manifest,item['views'])
            expected.append(digest(b))
    finally:geo.old.on_device=original
    rows=[]
    for workers in (0,2,4):
        dl=loader(manifest,items,workers,pin_memory=False)
        passes=[]
        for repetition in range(3):
            began=time.perf_counter();latencies=[];last=began
            for index,result in enumerate(dl):
                now=time.perf_counter();latencies.append(now-last)
                assert result['sequence']==index and result['item']==items[index]
                if repetition==0:assert digest(result['batch'])==expected[index],(workers,index)
                last=time.perf_counter()
            elapsed=time.perf_counter()-began
            passes.append(dict(repetition=repetition,seconds=elapsed,
                samples_per_second=samples/elapsed,wait_seconds=sum(latencies),
                median_wait_s=statistics.median(latencies)))
        # First pass includes worker startup and parity hashing, kept separately.
        rows.append(dict(workers=workers,passes=passes,
            steady_samples_per_second=statistics.median(p['samples_per_second'] for p in passes[1:])))
        print(json.dumps(rows[-1]),flush=True)
        del dl
    base=rows[0]['steady_samples_per_second']
    for row in rows:row['loader_only_speedup']=row['steady_samples_per_second']/base
    result=dict(utc=geo.utc(),pid=os.getpid(),samples=samples,passed=True,
        bitwise_parity=True,ordered=True,cuda_used=False,pin_memory=False,
        code_sha256=geo.sha(__file__),trainer_sha256=geo.sha(geo.__file__),
        manifest_sha256=geo.sha(geo.OUT/'manifest.json'),rows=rows,
        limits='Warm-cache CPU loader-only benchmark while existing trainer runs; startup/parity pass excluded from throughput. Not end-to-end GPU training speedup; worker memory/shared-memory and GPU pinning need preflight in next run.')
    geo.write(dest,result)
    return result


def smoke(dest,workers=4):
    """Check real pinned-memory -> CUDA transfer, worker order and RNG isolation."""
    torch.set_num_threads(1)
    manifest=geo.read(geo.OUT/'manifest.json')
    items=[x for update in geo.read(geo.OUT/'geometry_seed17/schedule.json')[:2] for x in update]
    torch.cuda.init()
    state=torch.get_rng_state().clone()
    cuda_state=torch.cuda.get_rng_state().clone()
    dl=loader(manifest,items,workers,pin_memory=True)
    expected=[digest(load_cpu(manifest['records'][i['index']],manifest,i['views'])) for i in items]
    for index,result in enumerate(dl):
        assert result['sequence']==index and result['item']==items[index]
        b=result['batch']
        assert all(v.is_pinned() for v in b.values() if isinstance(v,torch.Tensor))
        gpu=to_device(b)
        back={k:v.cpu() if isinstance(v,torch.Tensor) else v for k,v in gpu.items()}
        assert digest(back)==expected[index]
    assert torch.equal(state,torch.get_rng_state())
    assert torch.equal(cuda_state,torch.cuda.get_rng_state())
    geo.write(dest,dict(utc=geo.utc(),passed=True,samples=len(items),workers=workers,
        pin_memory=True,non_blocking=True,bitwise_cpu_cuda_parity=True,
        model_cpu_and_cuda_rng_unchanged=True,worker_cuda_initialized=False,
        code_sha256=geo.sha(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--benchmark',type=Path);g.add_argument('--smoke',type=Path)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--samples',type=int,default=64);args=p.parse_args()
    if args.benchmark:benchmark(args.benchmark,args.samples)
    else:smoke(args.smoke,args.workers)
