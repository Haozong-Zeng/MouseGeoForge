"""Explicit model loading and input policies; never dispatches an experiment."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import re

import numpy as np
import torch
from torch.nn import functional as F

import pure_geometry_bottom_20261018 as geo
import bottom_geometry_v3_data_20261018 as data
import geometry_prefetch_20261018 as io


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def gpu_lock_path():
    """Local global lock by default; explicit UUID binding for independent H100 jobs."""
    identity = os.environ.get('MOUSEGEOFORGE_GPU_LOCK_ID')
    if identity is None:
        return geo.ROOT/'output/teacher_temporal_cycle_20261001/gpu_job.lock'
    if not re.fullmatch(r'GPU-[A-Za-z0-9-]+', identity) or os.environ.get('CUDA_VISIBLE_DEVICES') != identity:
        raise ValueError('Cluster lock requires the SAME single physical GPU UUID in both environment variables')
    return geo.ROOT/'output/geometry_v4_gpu_locks'/f'{identity}.lock'


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def read_manifest(path):
    m = json.loads(Path(path).read_text())
    for r in m['records']:
        tag = re.sub('[^a-z0-9]', '', json.dumps(r).lower())
        if 'waterreaching' in tag or r['group'].upper() in {'GR2', 'UR2', 'UR3'}:
            raise ValueError('Excluded source in manifest: ' + r['group'])
        if 'danncemarkerlessmouse2' in tag or r.get('rig') == 'dannce/2' or r['group'].lower() in ('dannce_mouse_2', 'dannce2'):
            raise ValueError('DANNCE2 is excluded')
        if r.get('protected_test') or any(str(r.get(k, '')).lower() == 'test' for k in ('source_split', 'admitted_motion_split')) or r['group'].startswith('anipose'):
            raise ValueError('Protected source TEST is excluded')
        if r['split'] not in ('train', 'val', 'sealed'):
            raise ValueError('Unexpected split: ' + r['split'])
    return m


def model(release, *, checkpoint=None, state='ema', trainable=False, seed=17,
          camera_lr=1e-4, encoder_lr=1e-6, depth_lr=1e-5):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    m = geo.VGGTOmega(enable_camera=True, enable_depth=True)
    weights = torch.load(release, map_location='cpu', mmap=True, weights_only=True)
    weights = {k: v for k, v in weights.items()
               if k.startswith(('aggregator.', 'camera_head.', 'dense_head.'))}
    buffers = {}
    if checkpoint:
        ck = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
        if ck['base_sha256'] != sha(release):
            raise ValueError('Adapted checkpoint does not match the pinned release')
        weights.update(ck['ema' if state == 'ema' else 'model']); buffers = ck['buffers']
    m.load_state_dict(weights, strict=True)
    with torch.no_grad():
        for name, value in m.named_buffers():
            if name in buffers:
                value.copy_(buffers[name])
    del weights, buffers
    if checkpoint:
        del ck
    m.requires_grad_(False)
    if trainable:
        m.camera_head.requires_grad_(True); m.dense_head.requires_grad_(True)
        m.dense_head.proj_conf.requires_grad_(False)
        for block in list(m.aggregator.frame_blocks[-2:]) + list(m.aggregator.inter_frame_blocks[-2:]):
            block.requires_grad_(True)
    if m.keypoint_head is not None:
        raise ValueError('Geometry experiment must not have a keypoint head')
    m = m.cuda().eval()
    if not trainable:
        return m
    groups = []
    for prefix, lr in [('aggregator.', encoder_lr), ('camera_head.', camera_lr), ('dense_head.', depth_lr)]:
        groups.append(dict(params=[v for n, v in m.named_parameters() if n.startswith(prefix) and v.requires_grad],
                           lr=lr, base_lr=lr))
    return m, torch.optim.AdamW(groups, weight_decay=.01, foreach=False)


def evaluation_packs(record, manifest):
    # Same input ordering for TRAIN calibration and development inference.
    for pack in geo.old.packs(geo.old.views_for(record, manifest)):
        yield pack[1:] + pack[:1] if len(pack) > 1 else pack


def background(batch, mode, dilation=8, blur_kernel=31):
    if mode == 'original':
        return batch
    if mode not in ('gray', 'blur'):
        raise ValueError(mode)
    mask = batch['body'].float()
    if not all(bool(m.any()) for m in mask[0]):
        raise ValueError('Background intervention requires a verified body mask in EVERY view')
    n, v, h, w = mask.shape
    # Preserve body edges, then create a fixed two-pixel soft transition.
    mask = F.max_pool2d(mask.reshape(n*v, 1, h, w), 2*dilation+1, stride=1, padding=dilation)
    mask = F.avg_pool2d(mask, 5, stride=1, padding=2).reshape(n, v, 1, h, w)
    images = batch['images']
    if mode == 'gray':
        backdrop = torch.full_like(images, .5)
    else:
        backdrop = F.avg_pool2d(images.reshape(n*v, 3, h, w), blur_kernel, stride=1,
                               padding=blur_kernel//2, count_include_pad=False).reshape_as(images)
    return dict(batch, images=images*mask + backdrop*(1-mask))


def load(record, manifest, pack, intervention='original'):
    return background(io.to_device(data.load_cpu(record, manifest, pack)), intervention)
