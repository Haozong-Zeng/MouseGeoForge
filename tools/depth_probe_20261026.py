"""Cheap depth probe (2026-10-26): does the keypoint-adapted DINO image encoder (UX_flat1e-4, the
three-seed winner of the head-first keypoint sweep) give better mouse depth than the release
encoder when the encoder is FROZEN and only a small depth head is trained on top?

Data: synthetic views with dense depth truth from the v3 geometry manifest
(output/bottom_geometry_v3_local12h_20261018/manifest.json): the crossed-rig renders
(48 TRAIN rigs, 12 unseen DEV rigs; the 8 sealed rigs are never read) and the legacy
BlenderPyMouse pools (CRIM13 homecage, MoSeq, ClimbWall, WashFace). Real captures have no
depth truth and are not used. Both arms share the item list, the crop cache, the sampling
order and the augmentation stream of a seed; only the frozen encoder weights differ.

Target: per-view relative depth of the mouse. Loss: scale-invariant log loss (SILog, lambda 1)
on body pixels plus a 0.01 anchor on the mean log residual. Metric: per-view scale alignment
by the median log residual on body pixels (the v4 evaluator's per-view body oracle), then
AbsRel, delta<1.05/1.10/1.25 and the log-residual std. Weight-free: logs and metrics only.

    prepare                            view-level item list from the manifest (JSON only)
    cache [--workers N]                512 crops + 256 depth/flag targets for every item (CPU)
    run --encoder release|kp --seed N  train the depth head on the frozen encoder (GPU)
    table                              paired report over the finished runs
"""
import argparse
import collections
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import head_first_kp_20261024 as kp  # noqa: E402  release parts, token conventions, sampler, hashing

OUT = ROOT/'output/depth_probe_20261026'
MANIFEST = ROOT/'output/bottom_geometry_v3_local12h_20261018/manifest.json'
ENDPOINTS = ROOT/'output/head_first_kp_20261024/retrain'
RUNS = OUT/'runs'
LAYERS = (5, 11, 17, 23)          # 0-based encoder blocks whose normalized tokens feed the head
CROP, TARGET, MARGIN = 512, 256, 1.3
EVAL_AT = (2000, 4000, 6000)
CFG = dict(
    updates=6000, batch_bottom=8, batch_other=8, peak_lr=3e-4, warmup=200, floor=0.05, weight_decay=0.01,
    grad_clip=10.0, ema_decay=0.995,
    loss=dict(name='silog', anchor=0.01, min_pixels=16),
    augment=dict(scale=(0.8, 1.25), rotate_deg=30.0, translate=0.08, hflip=0.5, p_photo=0.8, brightness=0.2,
                 contrast=(0.8, 1.2), gamma=(0.8, 1.25), gray=0.2, noise=0.02))
EXPECTED = {'train/crossed/bottom': 288, 'train/crossed/other': 576, 'train/legacy/bottom': 291,
            'train/legacy/other': 3728, 'unseen_rig/crossed/bottom': 72, 'unseen_rig/crossed/other': 144,
            'replay_val/crossed/bottom': 6, 'replay_val/crossed/other': 12, 'legacy_val/legacy/bottom': 91,
            'legacy_val/legacy/other': 1528}

sha, write, write_new = kp.sha, kp.write, kp.write_new


def load_items():
    return json.loads((OUT/'items.json').read_text())


# --------------------------------------------------------------------------- items

def admit(record):
    """Pool and evaluation group of a manifest record, or None when the probe may not use it."""
    if record.get('kind') != 'synthetic':
        return None
    if record.get('split') == 'sealed' or record.get('evaluation_role') == 'sealed_confirmation' \
            or 'sealed' in str(record.get('rig', '')):
        return None
    crossed = str(record.get('scene', '')).startswith('output/')
    if record['split'] == 'train':
        group = 'train'
    elif crossed:
        group = 'unseen_rig' if record.get('evaluation_role') == 'unseen_rig_gate' else 'replay_val'
    else:
        group = 'legacy_val'
    rig = record.get('rig') or 'legacy/' + str(record['group']).split('/')[0]
    return dict(pool='crossed' if crossed else 'legacy', group=group, rig=rig)


def motion_of(group):
    for key in ('climb', 'moseq', 'wash', 'crim13'):
        if key in group:
            return key
    return 'other'


def prepare():
    if (OUT/'items.json').exists():
        raise SystemExit('items.json exists: the item list is frozen; use a new namespace to change it')
    manifest = json.loads(MANIFEST.read_text())
    items, skipped = [], collections.Counter()
    for r in manifest['records']:
        a = admit(r)
        if not a:
            skipped[f"{r.get('kind')}/{r.get('split')}"] += 1
            continue
        for v in sorted(r['views'], key=lambda v: v['id']):
            if a['pool'] == 'crossed':
                assert v.get('truth') and not v.get('depth'), r['scene']
            else:
                assert v.get('depth') and v.get('mask'), r['scene']
            items.append(dict(id=len(items), scene=r['scene'], motion=motion_of(r['group']), source_group=r['group'],
                              rig=a['rig'], split=r['split'], pool=a['pool'], group=a['group'], view=v['id'],
                              physical_role=v['physical_role'], role='bottom' if v['physical_role'] == 'bottom' else 'other',
                              image=v['image'], truth=v.get('truth'), depth=v.get('depth'), mask=v.get('mask'), probe=False))
    rng = random.Random(20261026); probe = []
    for pool in ('crossed', 'legacy'):
        for role, n in (('bottom', 16), ('other', 32)):
            ids = [i['id'] for i in items if i['group'] == 'train' and i['pool'] == pool and i['role'] == role]
            probe += rng.sample(ids, n)
    for i in probe:
        items[i]['probe'] = True
    counts = collections.Counter(f"{i['group']}/{i['pool']}/{i['role']}" for i in items)
    if dict(counts) != EXPECTED:
        raise SystemExit(f'unexpected counts {dict(counts)} vs {EXPECTED}')
    unseen = sorted({i['rig'] for i in items if i['group'] == 'unseen_rig'})
    assert len(unseen) == 12 and not any('sealed' in i['rig'] for i in items)
    OUT.mkdir(parents=True, exist_ok=True)
    write_new(OUT/'items.json', items)
    write_new(OUT/'input_receipt.json', dict(
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        manifest=dict(path=str(MANIFEST.relative_to(ROOT)), sha256=sha(MANIFEST)),
        items=dict(n=len(items), sha256=sha(OUT/'items.json'), counts=dict(counts), probe=len(probe)),
        skipped_records=dict(skipped), unseen_rigs=unseen, sealed='never read: split sealed / sealed_confirmation excluded',
        real_data='none: real captures have no dense depth truth', tool_sha256=sha(Path(__file__))))
    print('prepared', len(items), dict(counts))


# --------------------------------------------------------------------------- crop cache

def read_pfm(path, flip_y=True):
    """vggt-omega/training/finetune_blendedmvs.read_pfm (copied; that module is not imported here)."""
    with Path(path).open('rb') as f:
        header = f.readline().decode('ascii').rstrip()
        if header not in {'PF', 'Pf'}:
            raise ValueError(f'Not a PFM file: {path}')
        dims = f.readline().decode('ascii').strip()
        while dims.startswith('#'):
            dims = f.readline().decode('ascii').strip()
        width, height = map(int, dims.split())
        scale = float(f.readline().decode('ascii').strip())
        endian = '<' if scale < 0 else '>'
        channels = 3 if header == 'PF' else 1
        data = np.fromfile(f, endian + 'f', width * height * channels)
    arr = np.reshape(data, (height, width, channels) if channels == 3 else (height, width))
    if flip_y:
        arr = np.flipud(arr)
    if channels == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.float32, copy=False)


def crop_box(body, margin=MARGIN, min_side=32.0):
    """Square source box [x0, y0, side] around the body mask, or None when the mask is empty."""
    ys, xs = np.nonzero(body)
    if len(xs) == 0:
        return None
    w, h = xs.max() + 1 - xs.min(), ys.max() + 1 - ys.min()
    side = max(margin * max(w, h), min_side)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2  # pixel-index (centre) coordinates, as warpAffine uses
    return [float(cx - side / 2), float(cy - side / 2), float(side)]


def crop_affine(box, size):
    x0, y0, side = box; s = size / side
    return np.array([[s, 0, -s * x0], [0, s, -s * y0]], np.float64)


def load_truth(it):
    """RGB (H, W, 3) uint8, depth (H, W) float32, body and valid (H, W) bool in source pixels."""
    import cv2
    cv2.setNumThreads(0)  # worker processes; no second thread pool next to torch's
    rgb = cv2.imread(str(ROOT/it['image']), cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(it['image'])
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    if it.get('truth'):
        with np.load(ROOT/it['truth']) as a:
            z, body, valid = a['depth'].astype(np.float32), a['body'].astype(bool), a['valid'].astype(bool)
    else:
        from PIL import Image
        z = read_pfm(ROOT/it['depth'], flip_y=True)
        with Image.open(ROOT/it['mask']) as m:
            body = np.asarray(m.convert('L')) > 0
        valid = np.isfinite(z) & (z > 1e-6)
    if z.shape != rgb.shape[:2] or body.shape != z.shape or valid.shape != z.shape:
        raise ValueError(f"shape mismatch {it['id']}: rgb {rgb.shape} depth {z.shape} body {body.shape} valid {valid.shape}")
    valid &= np.isfinite(z) & (z > 0)
    return rgb, np.nan_to_num(z).astype(np.float32), body, valid


def crop_item(it):
    """One cache row: 512 crop, 256 depth and flag targets (bit0 body, bit1 valid), box, body pixels."""
    import cv2
    rgb, z, body, valid = load_truth(it)
    box = crop_box(body)
    if box is None:
        h, w = z.shape; side = float(min(h, w)); box = [(w - side) / 2, (h - side) / 2, side]
    crop = cv2.warpAffine(rgb, crop_affine(box, CROP), (CROP, CROP), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    a = crop_affine(box, TARGET)
    zt = cv2.warpAffine(z, a, (TARGET, TARGET), flags=cv2.INTER_NEAREST, borderValue=0)
    flags = cv2.warpAffine((body.astype(np.uint8) | (valid.astype(np.uint8) << 1)), a, (TARGET, TARGET),
                           flags=cv2.INTER_NEAREST, borderValue=0)
    body_px = int(((flags & 3) == 3).sum())
    return crop, zt.astype(np.float32), flags.astype(np.uint8), box, body_px


def cache(workers=12):
    import torch
    receipt = OUT/'cache/receipt.json'
    if receipt.exists():
        print('cache exists'); return
    items = load_items(); N = len(items)
    (OUT/'cache').mkdir(exist_ok=True)
    crops = np.lib.format.open_memmap(OUT/'cache/crops.u8.npy.partial', 'w+', np.uint8, (N, CROP, CROP, 3))
    depth = np.lib.format.open_memmap(OUT/'cache/depth.f32.npy.partial', 'w+', np.float32, (N, TARGET, TARGET))
    flags = np.lib.format.open_memmap(OUT/'cache/flags.u8.npy.partial', 'w+', np.uint8, (N, TARGET, TARGET))

    class Rows(torch.utils.data.Dataset):
        def __len__(self):
            return N

        def __getitem__(self, i):
            crop, zt, fl, box, body_px = crop_item(items[i])
            return i, crop, zt, fl, np.array(box, np.float64), body_px

    loader = torch.utils.data.DataLoader(Rows(), batch_size=None, num_workers=workers, prefetch_factor=4)
    meta, t0 = [None] * N, time.time()
    for i, crop, zt, fl, box, body_px in loader:
        i = int(i)
        crops[i], depth[i], flags[i] = crop.numpy(), zt.numpy(), fl.numpy()
        meta[i] = dict(id=int(i), box=[float(x) for x in box], body_px=int(body_px), ok=int(body_px) >= 64)
        if i % 1000 == 0:
            print('CACHE', int(i), N, round(time.time() - t0), flush=True)
    crops.flush(); depth.flush(); flags.flush(); del crops, depth, flags
    for name in ('crops.u8.npy', 'depth.f32.npy', 'flags.u8.npy'):
        os.replace(OUT/'cache'/f'{name}.partial', OUT/'cache'/name)
    write(OUT/'cache/meta.json', meta)
    bad = [m['id'] for m in meta if not m['ok']]
    write_new(receipt, dict(items_sha256=sha(OUT/'items.json'), n=N, excluded_small_body=len(bad), excluded_ids=bad[:50],
                            crops=dict(shape=[N, CROP, CROP, 3], dtype='uint8'), depth=dict(shape=[N, TARGET, TARGET], dtype='float32'),
                            flags=dict(shape=[N, TARGET, TARGET], dtype='uint8', bits='0 body, 1 valid'),
                            preprocessing=f'body-mask square box, margin {MARGIN}, warpAffine INTER_LINEAR (rgb) / INTER_NEAREST '
                                          f'(depth, flags), border 0; legacy PFM read with flip_y; depth = camera Z in mm',
                            meta_sha256=sha(OUT/'cache/meta.json'), tool_sha256=sha(Path(__file__)), seconds=round(time.time() - t0)))
    print('CACHE done', N, 'small-body excluded', len(bad), round(time.time() - t0))


class Store:
    """Cache arrays on the GPU when they fit (else CPU); `ok` marks items with enough body pixels."""

    def __init__(self, items, crops, depth, flags, ok, device='cpu', reserve_gb=12):
        import torch
        self.items, self.device = items, device
        size = crops.nbytes + depth.nbytes + flags.nbytes
        on_gpu = device != 'cpu' and torch.cuda.mem_get_info()[0] - size > reserve_gb * 2**30
        self.placement = 'gpu' if on_gpu else 'cpu'
        self.crops, self.depth, self.flags = (self._load(a, on_gpu) for a in (crops, depth, flags))
        self.ok = np.asarray(ok, bool)

    @classmethod
    def from_cache(cls, items, device='cuda'):
        meta = json.loads((OUT/'cache/meta.json').read_text())
        assert [m['id'] for m in meta] == [i['id'] for i in items]
        arrays = [np.load(OUT/'cache'/n, mmap_mode='r') for n in ('crops.u8.npy', 'depth.f32.npy', 'flags.u8.npy')]
        return cls(items, *arrays, [m['ok'] for m in meta], device)

    def _load(self, arr, on_gpu):
        import torch
        out = torch.empty(arr.shape, dtype=torch.from_numpy(np.asarray(arr[:1]).copy()).dtype,
                          device=self.device if on_gpu else 'cpu')
        for i in range(0, len(arr), 256):
            out[i:i + 256].copy_(torch.from_numpy(np.array(arr[i:i + 256])))
        return out

    def batch(self, ids):
        """rgb01 (B, 3, 512, 512) float, depth (B, 256, 256) float, flags (B, 256, 256) uint8 on `device`."""
        idx = list(ids)
        rgb = self.crops[idx].to(self.device, non_blocking=True).permute(0, 3, 1, 2).float().contiguous() / 255
        return rgb, self.depth[idx].to(self.device, non_blocking=True).float(), self.flags[idx].to(self.device, non_blocking=True)


# --------------------------------------------------------------------------- model pieces

def encode_layers(pe, mean, std, rgb01, layers=LAYERS, autocast=True):
    """Normalized patch tokens after each block in `layers`, as (B, C, H, W) float; no autograd.
    The last entry (block 23) equals head_first_kp_20261024.encode(..., first_active=24)."""
    import torch
    ctx = torch.autocast('cuda', dtype=torch.bfloat16) if autocast else torch.autocast('cpu', enabled=False)
    B, out = rgb01.shape[0], []
    with torch.no_grad(), ctx:
        x, (H, W) = pe.prepare_tokens_with_masks((rgb01 - mean) / std)
        rope = pe.rope_embed(H=H, W=W) if pe.rope_embed is not None else None
        for i, blk in enumerate(pe.blocks):
            x = blk(x, rope)
            if i in layers:
                t = pe.norm(x)[:, pe.n_storage_tokens + 1:]
                # transpose+reshape is a channels-last VIEW of the token tensor; the CPU GroupNorm backward of the
                # head segfaults on it (torch 2.11, input without grad), so materialize NCHW-contiguous features.
                out.append(t.transpose(1, 2).reshape(B, t.shape[-1], H, W).contiguous().float())
    return out


def build_head(feat_dim=1024, width=128, layers=len(LAYERS)):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def up(w):
        return nn.Sequential(nn.Conv2d(w, w, 3, padding=1), nn.GroupNorm(8, w), nn.GELU())

    class DepthHead(nn.Module):
        """Multi-layer token projection summed at 32x32, three x2 upsampling stages to 256,
        an RGB skip at 256, one output channel = log relative depth."""

        def __init__(self):
            super().__init__()
            self.project = nn.ModuleList([nn.Sequential(nn.GroupNorm(16, feat_dim), nn.Conv2d(feat_dim, width, 1))
                                          for _ in range(layers)])
            self.fuse = nn.Sequential(nn.GELU(), nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.GELU())
            self.up = nn.ModuleList([up(width) for _ in range(3)])
            self.rgb = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.GroupNorm(4, 16), nn.GELU())
            self.out = nn.Sequential(nn.Conv2d(width + 16, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(), nn.Conv2d(64, 1, 1))

        def forward(self, feats, rgb):
            x = self.fuse(sum(p(f) for p, f in zip(self.project, feats)))
            for u in self.up:
                x = u(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False))
            return self.out(torch.cat([x, self.rgb(rgb)], 1))
    return DepthHead()


def target_mask(flags, z):
    return ((flags & 3) == 3) & (z > 0)


def silog(pred_log, z, mask, anchor=0.01, min_pixels=16):
    """Per-view scale-invariant log loss (variance of the log residual) + anchor * mean residual^2,
    averaged over the views with at least `min_pixels` target pixels. Returns (loss, views used)."""
    import torch
    d = torch.where(mask, pred_log[:, 0] - torch.log(z.clamp_min(1e-6)), torch.zeros_like(z))
    n = mask.sum((1, 2)); nf = n.clamp_min(1).float()
    mean = d.sum((1, 2)) / nf
    var = (d * d).sum((1, 2)) / nf - mean ** 2
    ok = n >= min_pixels
    if not ok.any():
        return pred_log.sum() * 0, 0
    return (var + anchor * mean ** 2)[ok].mean(), int(ok.sum())


def view_metrics(pred_log, z, mask, min_pixels=16):
    """Per-view body metrics after scale alignment by the median log residual (v4 per-view body oracle)."""
    import torch
    if int(mask.sum()) < min_pixels:
        return None
    lp, zt = pred_log[mask], z[mask]
    d = torch.log(zt) - lp
    med = d.median()
    ratio = torch.exp((d - med).abs())   # max(s*p/t, t/(s*p)) with s = exp(med)
    absrel = ((torch.exp(lp + med) - zt).abs() / zt).mean()
    return dict(n=int(mask.sum()), scale=float(torch.exp(med)), absrel=float(absrel),
                d105=float((ratio < 1.05).float().mean()), d110=float((ratio < 1.10).float().mean()),
                d125=float((ratio < 1.25).float().mean()), silog=float(d.std(unbiased=False)))


# --------------------------------------------------------------------------- augmentation

def apply_warp(rgb, z, flags, theta):
    """Joint affine resampling: rgb bilinear at 512, depth and flags nearest at 256, zeros outside."""
    import torch
    import torch.nn.functional as F
    B = rgb.shape[0]
    g512 = F.affine_grid(theta, (B, 3, CROP, CROP), align_corners=False)
    g256 = F.affine_grid(theta, (B, 1, TARGET, TARGET), align_corners=False)
    rgb = F.grid_sample(rgb, g512, mode='bilinear', padding_mode='zeros', align_corners=False)
    z = F.grid_sample(z[:, None], g256, mode='nearest', padding_mode='zeros', align_corners=False)[:, 0]
    flags = F.grid_sample(flags[:, None].float(), g256, mode='nearest', padding_mode='zeros', align_corners=False)[:, 0]
    return rgb, z, flags.round().to(torch.uint8)


def apply_photometric(rgb, on, brightness, contrast, gamma, gray, noise):
    import torch
    x = ((rgb - 0.5) * contrast + 0.5 + brightness).clamp(0, 1) ** gamma
    x = torch.where(gray, x.mean(1, keepdim=True).expand_as(x), x)
    x = (x + noise).clamp(0, 1)
    return torch.where(on, x, rgb)


def augment(rgb, z, flags, gen, a):
    import torch
    B, dev = rgb.shape[0], rgb.device

    def U(lo, hi, shape=(B,)):
        return torch.rand(shape, generator=gen, device=dev) * (hi - lo) + lo
    sc = torch.exp(U(math.log(a['scale'][0]), math.log(a['scale'][1])))
    ang = U(-a['rotate_deg'], a['rotate_deg']) * math.pi / 180
    tx, ty = U(-a['translate'], a['translate']) * 2, U(-a['translate'], a['translate']) * 2
    flip = torch.where(U(0, 1) < a['hflip'], -1.0, 1.0)
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos * flip, -sin, tx], -1), torch.stack([sin * flip, cos, ty], -1)], 1)
    rgb, z, flags = apply_warp(rgb, z, flags, theta)
    v = lambda t: t.view(B, 1, 1, 1)  # noqa: E731
    rgb = apply_photometric(rgb, v(U(0, 1) < a['p_photo']), v(U(-a['brightness'], a['brightness'])), v(U(*a['contrast'])),
                            v(U(*a['gamma'])), v(U(0, 1) < a['gray']), torch.randn(rgb.shape, generator=gen, device=dev) * a['noise'])
    return rgb, z, flags


# --------------------------------------------------------------------------- training

class Trainer:
    def __init__(self, seed, store, head, pe, mean, std, cfg, device='cuda', autocast=True):
        import torch
        self.store, self.head, self.pe, self.mean, self.std = store, head, pe, mean, std
        self.cfg, self.device, self.autocast, self.torch = cfg, device, autocast, torch
        decay, no_decay = [], []
        for module in head.modules():
            for name, p in module.named_parameters(recurse=False):
                (no_decay if kp.is_no_decay(module, name) else decay).append(p)
        self.opt = torch.optim.AdamW([dict(params=decay, weight_decay=cfg['weight_decay']), dict(params=no_decay, weight_decay=0.0)],
                                     lr=0.0, betas=(0.9, 0.999))
        self.ema = {n: p.detach().clone() for n, p in head.named_parameters()}
        pool = [i['id'] for i in store.items if i['group'] == 'train' and store.ok[i['id']]]
        bottom = [i for i in pool if store.items[i]['role'] == 'bottom']
        other = [i for i in pool if store.items[i]['role'] != 'bottom']
        self.s_bottom, self.s_other = kp.Sampler(bottom, seed), kp.Sampler(other, seed + 1000)
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.u, self.log, self.clipped, self.exposure = 0, [], 0, collections.Counter()
        self.pool_sizes = dict(bottom=len(bottom), other=len(other))

    def lr(self, u):
        c = self.cfg
        warm = min(1.0, u / c['warmup'])
        env = c['floor'] + (1 - c['floor']) * 0.5 * (1 + math.cos(math.pi * (u - 1) / c['updates']))
        return c['peak_lr'] * warm * env

    def features(self, rgb):
        return encode_layers(self.pe, self.mean, self.std, rgb, LAYERS, self.autocast)

    def step(self):
        torch, c = self.torch, self.cfg
        u = self.u + 1
        for g in self.opt.param_groups:
            g['lr'] = self.lr(u)
        ids = self.s_bottom.take(c['batch_bottom']) + self.s_other.take(c['batch_other'])
        rgb, z, flags = self.store.batch(ids)
        rgb, z, flags = augment(rgb, z, flags, self.gen, c['augment'])
        pred = self.head(self.features(rgb), rgb)
        loss, used = silog(pred, z, target_mask(flags, z), c['loss']['anchor'], c['loss']['min_pixels'])
        if not torch.isfinite(loss):
            raise FloatingPointError(f'non-finite loss at update {u}')
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(self.head.parameters(), c['grad_clip']))
        if not math.isfinite(gn):
            raise FloatingPointError(f'non-finite gradient norm at update {u}')
        self.clipped += gn > c['grad_clip']
        self.opt.step()
        with torch.no_grad():
            for n, p in self.head.named_parameters():
                self.ema[n].mul_(c['ema_decay']).add_(p.detach(), alpha=1 - c['ema_decay'])
        self.u = u
        for i in ids:
            self.exposure[self.store.items[i]['role']] += 1
        self.exposure['views_with_targets'] += used
        if u % 50 == 0 or u == 1:
            self.log.append(dict(update=u, loss=float(loss), grad_norm=gn, lr=self.lr(u), views_with_targets=used))
        return float(loss.detach())

    def swap_ema(self):
        with self.torch.no_grad():
            for n, p in self.head.named_parameters():
                tmp = p.detach().clone(); p.copy_(self.ema[n]); self.ema[n].copy_(tmp)

    def evaluate(self, ids, use_ema=True, batch=32):
        if use_ema:
            self.swap_ema()
        try:
            self.head.eval(); rows = []
            with self.torch.no_grad():
                for k in range(0, len(ids), batch):
                    chunk = ids[k:k + batch]
                    rgb, z, flags = self.store.batch(chunk)
                    pred = self.head(self.features(rgb), rgb)
                    m = target_mask(flags, z)
                    for j, i in enumerate(chunk):
                        r = view_metrics(pred[j, 0], z[j], m[j])
                        if r is None:
                            continue
                        it = self.store.items[i]
                        rows.append(dict(id=i, group='train_probe' if it['group'] == 'train' else it['group'], pool=it['pool'],
                                         rig=it['rig'], motion=it['motion'], role=it['role'], **r))
            return rows
        finally:
            self.head.train()
            if use_ema:
                self.swap_ema()


def summarize(rows):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[f"{r['group']}/{r['role']}"].append(r)
        groups[f"{r['group']}/all"].append(r)
        if r['group'] == 'unseen_rig':
            groups[f"rig/{r['rig']}/{r['role']}"].append(r)
            groups[f"unseen_rig_motion/{r['motion']}/{r['role']}"].append(r)
    out = {}
    for k, rs in sorted(groups.items()):
        a = np.array([r['absrel'] for r in rs])
        out[k] = dict(views=len(rs), absrel_mean=float(a.mean()), absrel_median=float(np.median(a)), absrel_p90=float(np.percentile(a, 90)),
                      d105=float(np.mean([r['d105'] for r in rs])), d110=float(np.mean([r['d110'] for r in rs])),
                      d125=float(np.mean([r['d125'] for r in rs])), silog_mean=float(np.mean([r['silog'] for r in rs])),
                      scale_median=float(np.median([r['scale'] for r in rs])))
    return out


def apply_endpoint(pe, ep, seed=None):
    """Load the EMA encoder of a retrained keypoint endpoint into `pe` (strict), after checks."""
    if ep.get('release_sha256') != kp.RELEASE_SHA:
        raise ValueError('endpoint was not trained from the pinned release')
    if ep.get('encoder_prefix') != 'aggregator.patch_embed' or ep.get('weights') != 'ema':
        raise ValueError('endpoint is not an EMA aggregator.patch_embed state')
    if seed is not None and ep.get('seed') != seed:
        raise ValueError(f"endpoint seed {ep.get('seed')} != {seed}")
    pe.load_state_dict(ep['ema_encoder'], strict=True)
    return dict(variant=ep.get('variant'), seed=ep.get('seed'), update=ep.get('update'))


def load_encoder(pe, seed):
    import torch
    path = ENDPOINTS/f'UX_flat1e-4_seed{seed}/endpoint.pt'
    ep = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    info = apply_endpoint(pe, ep, seed)
    done = json.loads((path.parent/'DONE.json').read_text())
    info.update(path=str(path.relative_to(ROOT)), sha256=done['endpoint']['sha256'], bytes=done['endpoint']['bytes'],
                dev_pano_bottom_paw_pck05=ep.get('dev_summary', {}).get('pano/bottom/paws', {}).get('pck05'))
    del ep
    return info


def eval_ids(items, ok, smoke=False):
    ids = [i['id'] for i in items if ok[i['id']] and (i['group'] != 'train' or i['probe'])]
    if smoke:
        per, keep = collections.Counter(), []
        for i in ids:
            k = (items[i]['group'], items[i]['role'])
            if per[k] < 8:
                per[k] += 1; keep.append(i)
        ids = keep
    return ids


def run(encoder, seed, updates=None, smoke=False):
    import torch
    cfg = json.loads(json.dumps(CFG))
    if updates:
        cfg['updates'] = updates
    eval_at = sorted({min(e, cfg['updates']) for e in EVAL_AT} | {cfg['updates']})
    name = f'{encoder}_seed{seed}'
    dest = (OUT/'smoke' if smoke else RUNS)/name
    if (dest/'DONE.json').exists() and not smoke:
        print('done already:', dest); return
    dest.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    items = load_items()
    receipt = json.loads((OUT/'cache/receipt.json').read_text())
    if receipt['items_sha256'] != sha(OUT/'items.json'):
        raise SystemExit('cache was built from a different item list')
    model, pe, mean, std = kp.release_parts()
    release_hashes = kp.param_hashes(pe, 'aggregator.patch_embed')
    provenance = dict(encoder='release', release_sha256=kp.RELEASE_SHA)
    if encoder == 'kp':
        provenance.update(load_encoder(pe, seed), encoder='kp')
        changed = sum(h != release_hashes[n] for n, h in kp.param_hashes(pe, 'aggregator.patch_embed').items())
        provenance['encoder_tensors_changed_from_release'] = changed
        if changed == 0:
            raise RuntimeError('kp endpoint loaded but no encoder tensor differs from the release')
    pe.eval().requires_grad_(False)
    start_hashes = kp.param_hashes(pe, 'aggregator.patch_embed')
    del model
    store = Store.from_cache(items, 'cuda')
    torch.manual_seed(seed)
    head = build_head().cuda()
    t = Trainer(seed, store, head, pe, mean, std, cfg)
    ids = eval_ids(items, store.ok, smoke)
    write(dest/'run_protocol.json', dict(
        name=name, encoder=provenance, seed=seed, cfg=cfg, layers=list(LAYERS), crop=CROP, target=TARGET, eval_at=eval_at,
        head_parameters=sum(p.numel() for p in head.parameters()), train_pool=t.pool_sizes, eval_views=len(ids),
        items_sha256=receipt['items_sha256'], cache_receipt_sha256=sha(OUT/'cache/receipt.json'),
        store_placement=store.placement, smoke=smoke,
        metric='per-view body pixels (body & valid & depth>0), scale = exp(median(log gt - log pred)) [v4 per-view body oracle], '
               'AbsRel of scale*pred, delta thresholds on max ratio, silog = std of the log residual',
        loss='per-view variance of (log pred - log gt) over body pixels + anchor * mean^2, mean over views with >= min_pixels',
        encoder_forward='no autograd, bf16 autocast, blocks 5/11/17/23 normalized tokens; the head trains in fp32',
        weights_saved=False, tool_sha256=sha(Path(__file__)), kp_tool_sha256=sha(ROOT/'tools/head_first_kp_20261024.py'),
        started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    print(f'{name}: encoder {provenance}, head {sum(p.numel() for p in head.parameters())} params, pools {t.pool_sizes}, '
          f'eval views {len(ids)}, store on {store.placement}', flush=True)
    t0, last = time.time(), None
    for _ in range(cfg['updates']):
        loss = t.step()
        u = t.u
        if u % 200 == 0 or u == 1:
            print(f'u{u} loss {loss:.4f} lr {t.lr(u):.2e} {(time.time() - t0) / u:.3f}s/u', flush=True)
        if u in eval_at:
            for use_ema in ((True, False) if u == cfg['updates'] else (True,)):
                rows = t.evaluate(ids, use_ema)
                summary = summarize(rows)
                write(dest/f"dev_u{u:05d}_{'ema' if use_ema else 'raw'}.json", dict(update=u, ema=use_ema, summary=summary, rows=rows))
                if use_ema:
                    last = summary
                key = {k: (round(v['absrel_mean'], 4), round(v['absrel_median'], 4), round(v['d110'], 3))
                       for k, v in summary.items() if k in ('unseen_rig/bottom', 'unseen_rig/other', 'legacy_val/bottom', 'train_probe/bottom')}
                print(f"DEV u{u} {'ema' if use_ema else 'raw'} (absrel mean, median, d110): {key}", flush=True)
            write(dest/'train_log.json', dict(log=t.log, exposure=dict(t.exposure)))
    if kp.param_hashes(pe, 'aggregator.patch_embed') != start_hashes:
        raise RuntimeError('frozen encoder changed during training')
    seconds = time.time() - t0
    write(dest/'train_log.json', dict(log=t.log, exposure=dict(t.exposure)))
    write(dest/'DONE.json', dict(name=name, encoder=provenance, seed=seed, updates=t.u, weights_saved=False, seconds=round(seconds),
                                train_seconds_per_update=seconds / t.u, peak_memory_gb=torch.cuda.max_memory_allocated() / 2**30,
                                clipped_head_updates=t.clipped, exposure=dict(t.exposure), summary=last, smoke=smoke,
                                finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    print('DONE', name, round(seconds), 's', flush=True)


def table(base=RUNS):
    keys = ['unseen_rig/bottom', 'unseen_rig/other', 'replay_val/bottom', 'legacy_val/bottom', 'legacy_val/other', 'train_probe/bottom']
    runs = {d.name: json.loads((d/'DONE.json').read_text()) for d in sorted(base.glob('*_seed*')) if (d/'DONE.json').exists()}
    print(f"{'run':16s} " + ' '.join(f'{k:>22s}' for k in keys) + '   (absrel mean / median / d110)')
    for n, d in runs.items():
        s = d['summary'] or {}
        print(f'{n:16s} ' + ' '.join(f"{s[k]['absrel_mean']:.4f}/{s[k]['absrel_median']:.4f}/{s[k]['d110']:.3f}".rjust(22)
                                     if k in s else ' ' * 22 for k in keys))
    pairs = {}
    for seed in sorted({d['seed'] for d in runs.values()}):
        a, b = runs.get(f'release_seed{seed}'), runs.get(f'kp_seed{seed}')
        if a and b and a['summary'] and b['summary']:
            pairs[seed] = {k: dict(release=a['summary'][k]['absrel_mean'], kp=b['summary'][k]['absrel_mean'],
                                   diff_mean=b['summary'][k]['absrel_mean'] - a['summary'][k]['absrel_mean'],
                                   diff_median=b['summary'][k]['absrel_median'] - a['summary'][k]['absrel_median'],
                                   diff_d110=b['summary'][k]['d110'] - a['summary'][k]['d110'])
                           for k in keys if k in a['summary'] and k in b['summary']}
    for seed, p in pairs.items():
        print(f'seed {seed} kp - release: ' + ', '.join(f"{k} {v['diff_mean']:+.4f} (median {v['diff_median']:+.4f}, d110 {v['diff_d110']:+.3f})"
                                                        for k, v in p.items() if k.startswith('unseen')))
    write(base/'table.json', dict(runs={n: dict(encoder=d['encoder'], seed=d['seed'], updates=d['updates'], summary=d['summary']) for n, d in runs.items()},
                                 paired=pairs, keys=keys, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['prepare', 'cache', 'run', 'table'])
    ap.add_argument('--encoder', choices=['release', 'kp'], default='release')
    ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--updates', type=int, default=None)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--smoke', action='store_true', help='run: short run under smoke/, small evaluation subsets')
    a = ap.parse_args()
    if a.step == 'prepare':
        prepare()
    elif a.step == 'cache':
        cache(a.workers)
    elif a.step == 'run':
        run(a.encoder, a.seed, a.updates, a.smoke)
    else:
        table()
