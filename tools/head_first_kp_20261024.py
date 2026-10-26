"""Head-first keypoint route (2026-10-24): H0 head convergence, then matched F/U continuation.

Design: configs/experiments/head_first_progressive_encoder_20261024.json and
docs/head_first_progressive_encoder_route_20261024.md. Historical B conventions are
imported unchanged from tools/panomouse_stage2_kp_20261023.py (reference only; its
entry points are never run): GT-box 512 crop (margin 1.3), frozen release image
encoder `aggregator.patch_embed` -> x_norm_patchtokens under bf16 autocast, 128x128
heatmaps, sigma 2 cells, 8 own + 8 PanoMouse views per update, CE averaged over the
known points of each domain and summed. Everything is written under
output/head_first_kp_20261024/ (data disk); historical outputs are never written.

    prepare                      items via the historical logic (hash-checked) + receipts
    cache                        frozen final tokens + crops for every item (GPU)
    preflight                    topology, cache/online parity, gradient reach, resume (GPU; updates discarded)
    train --mode H0 --seed N     head to the registered TRAIN plateau (6,000-18,000 updates)
    train --mode F|U --seed N    6,000-update continuation from the H0 EMA endpoint
    preflight-protected          UP gradient reach, exclusions, LR table, speed/memory (GPU; updates discarded)
    train --mode FP|UP --seed N  registered protected pair (configs/experiments/head_first_protected_encoder_20261024.json)
    gate --seed N [--pair protected]  U-vs-F (or UP-vs-FP) screening/retention for one seed (exit 3 = stop expansion)
    summary                      paired three-seed report
"""
import argparse
import collections
import hashlib
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
import panomouse_stage2_kp_20261023 as hist  # noqa: E402  historical B conventions

OUT = ROOT/'output/head_first_kp_20261024'
DESIGN = ROOT/'configs/experiments/head_first_progressive_encoder_20261024.json'
PROTECTED = ROOT/'configs/experiments/head_first_protected_encoder_20261024.json'
RELEASE = ROOT/'vggt-omega/vggt_omega_1b_512.pt'
EVIDENCE = ROOT/'docs/experiments/panomouse_head_first_evidence_20261024.json'
RELEASE_SHA = 'c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934'
# MGF_SMOKE=1: shortened schedules, small evaluation subsets and run folders under OUT/smoke;
# the frozen item list and the feature cache are shared with the real runs.
SMOKE = os.environ.get('MGF_SMOKE') == '1'
RUNS = OUT/'smoke' if SMOKE else OUT
CHECK_EVERY = 10 if SMOKE else 500
PAWS = (1, 2, 3, 4)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=1); f.write('\n')


def write(path, value):
    path = Path(path); tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=1) + '\n'); os.replace(tmp, path)


def design():
    cfg = json.loads(DESIGN.read_text())
    if SMOKE:
        cfg['h0'].update(min_updates=20, max_updates=40, warmup_updates=5, cosine_total_updates=40)
        cfg['h0']['convergence']['span_updates'] = (cfg['h0']['convergence']['checks'] - 1) * CHECK_EVERY
        cfg['pair'].update(updates_per_branch=30, cosine_total_updates=30, head_warmup_updates=5,
                           encoder_new_group_warmup_updates=5, dev_eval_updates=[10, 20, 30],
                           phases=[dict(first_update=1, last_update=10, unfrozen_blocks=[23]),
                                   dict(first_update=11, last_update=20, unfrozen_blocks=[22, 23]),
                                   dict(first_update=21, last_update=30, unfrozen_blocks=[20, 21, 22, 23])])
    return cfg


def protected_design():
    """Parent design with the registered protected pair (FP/UP) settings and encoder schedule."""
    cfg = design()
    reg = json.loads(PROTECTED.read_text())
    cfg['pair'].update({k: reg['pair'][k] for k in (
        'updates_per_branch', 'head_peak_lr', 'head_warmup_updates', 'cosine_total_updates', 'cosine_floor',
        'weight_decay', 'grad_clip', 'ema_decay', 'dev_eval_updates')})
    cfg['protected'] = reg['protected_encoder']
    if SMOKE:
        cfg['pair'].update(updates_per_branch=30, cosine_total_updates=30, head_warmup_updates=5, dev_eval_updates=[10, 20, 30])
        cfg['protected']['warmup_updates'] = 10
    return cfg


def design_for(mode):
    return protected_design() if mode in ('FP', 'UP') else design()


def subset(items, ids, per_stratum):
    """First `per_stratum` ids of each dataset x view stratum (smoke evaluation subsets)."""
    out, seen = [], collections.Counter()
    for i in ids:
        key = (items[i]['dataset'], items[i]['bottom'])
        if seen[key] < per_stratum:
            out.append(i); seen[key] += 1
    return out


def load_items():
    return json.loads((OUT/'items.json').read_text())


def evidence_pins():
    """Historical hashes recorded in the published evidence (items, stage-2 tool, stage-0 annotations)."""
    e = json.loads(EVIDENCE.read_text())
    pins = {p['path']: p['sha256'] for p in e['source_pins']}
    return dict(items=pins['output/panomouse_stage2_kp_20261023/items.json'],
                tool=pins['tools/panomouse_stage2_kp_20261023.py'], annotations=e['stage0_annotation_hashes'])


# --------------------------------------------------------------------------- prepare

def prepare():
    if (OUT/'items.json').exists():
        raise SystemExit('items.json exists: the item list is frozen; use a new namespace to change it')
    cfg = design()
    pano = ROOT/'dataset/PanoMouse'
    linked = sorted(p.name for p in (pano/'unpacked/images').iterdir())
    if set(linked) & set(cfg['inputs']['pano_sealed']):
        raise SystemExit(f'sealed PanoMouse sessions are reachable: {linked}')
    pins = evidence_pins()
    annotation = {k: sha(pano/k) for k in pins['annotations']}
    if annotation != pins['annotations']:
        raise SystemExit(f'PanoMouse annotation hashes differ from stage 0: {annotation}')
    if sha(ROOT/'tools/panomouse_stage2_kp_20261023.py') != pins['tool']:
        raise SystemExit('historical stage-2 reference changed')
    replay = OUT/'historical_replay'
    hist.OUT = replay  # the historical prepare writes only into this new folder
    hist.prepare()
    items_sha = sha(replay/'items.json')
    os.link(replay/'items.json', OUT/'items.json')
    items = load_items(); proto = json.loads((replay/'protocol.json').read_text())
    if proto['release_sha256'] != RELEASE_SHA:
        raise SystemExit('release checkpoint differs from the pinned release')
    strata = collections.Counter(f"{i['dataset']}/{i['split']}/{'bottom' if i['bottom'] else 'other'}" for i in items)
    counts = collections.Counter(f"{i['dataset']}_{i['split']}" for i in items)
    expected = cfg['inputs']['counts']
    observed = dict(own_train=counts['own_train'], own_dev=counts['own_dev'],
                    pano_train=counts['pano_train'], pano_dev=counts['pano_dev'])
    sessions = sorted({i['session'] for i in items})
    if set(sessions) & set(cfg['inputs']['pano_sealed'] + ['GR2', 'UR2', 'UR3']):
        raise SystemExit('excluded session in items')
    eff = ROOT/'dataset/PyMouseStudio/annotation_versions/20261003_effective_v1/manifest.json'
    write_new(OUT/'input_receipt.json', dict(
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        release=dict(path=str(RELEASE.relative_to(ROOT)), sha256=proto['release_sha256']),
        pano_annotations=annotation, pano_sessions_linked=linked,
        effective_manifest=dict(path=str(eff.relative_to(ROOT)), sha256=sha(eff)),
        items=dict(sha256=items_sha, historical_sha256=pins['items'], matches_historical=items_sha == pins['items'],
                   counts=observed, expected_counts=expected, counts_match=observed == expected, strata=dict(strata)),
        pano_label_counts=proto['pano_label_counts'], sessions=sessions,
        sources=dict(tool=sha(Path(__file__)), historical_tool=pins['tool'], design=sha(DESIGN), evidence=sha(EVIDENCE)),
        historical_outputs='output/panomouse_stage2_kp_20261023 is not on the cluster; items reproduced by the '
                           'historical prepare() and compared with the evidence hash instead'))
    write_new(OUT/'admission_receipt.json', dict(
        task='head_first_progressive_encoder_20261024: supervised still-image 2D five-point keypoints',
        authority='User published the head-first route (commits 8df5596, ea93a0a) and asked the cluster to write '
                  'and run it on 2 H100 nodes (2026-10-24).',
        splits=dict(own_train=cfg['inputs']['own_train'], own_dev=cfg['inputs']['own_dev'],
                    pano_train=cfg['inputs']['pano_train'], pano_dev=cfg['inputs']['pano_dev'],
                    pano_sealed_not_linked=cfg['inputs']['pano_sealed']),
        exclusions=cfg['inputs']['exclusions'], mapping=cfg['inputs']['mapping'],
        semantic_audit='Mapping as in docs/panomouse_stage01_20261023.md: nose direct; Lhand/Rhand = forepaw centre '
                       '(12 TRAIN images checked); hindpaw centre = midpoint(heel, foot), an approximation of the '
                       'plantar centre, not an exact synonym. Anatomical left verified in 3D for both datasets.',
        pano_label_rule=cfg['inputs']['pano_label_rule'],
        license='PanoMouse CC BY-NC 4.0: models trained here are tagged non-commercial',
        video_sync='not verified: only still-image supervised 2D is admitted; no video/temporal constraints',
        historical_receipt='PanoMouse stage0 admission stays evaluation-only and is not modified',
        scope='This receipt admits only this task and these splits.'))
    write_new(OUT/'protocol.json', dict(
        design=cfg, items_sha256=items_sha,
        train_probe='all TRAIN items (own + PanoMouse), EMA head, unaugmented; CE = mean known own + mean known Pano',
        convergence_metrics='bottom-view paw PCK@0.05 and swap proxy pooled over both domains (TRAIN probe)',
        engineering_probe='16 TRAIN items per stratum own/pano x bottom/other, Random(20261024)',
        samplers='H0: Random(seed) own / Random(seed+1000) pano (historical B); F and U share Random(seed+2000) / '
                 'Random(seed+3000)',
        features='H0/F: frozen cache (batch-1 bf16 autocast, fp16 storage, as historical B); U: online encoder on the '
                 'cached 512 crops, same autocast, batch of 16',
        no_decay='parameters of normalization modules and every *.bias; everything else in trainable groups wd 0.01',
        h0_weight_decay='historical B: wd 0.01 on all head parameters',
        lr='per update u (1-indexed): peak * min(1, (u - start + 1) / warmup) * envelope(u); H0 envelope '
           '0.05 + 0.95 * cos_half((u-1)/18000); F/U envelope 0.1 + 0.9 * cos_half((u-1)/6000)',
        accumulation='none: the physical batch equals the registered effective batch (8 own + 8 Pano views)',
        dev_eval='H0 endpoint; F/U at updates 2000/4000/6000; EMA primary, raw reported; DEV never drives training'))
    print('prepared', observed, 'items match historical:', items_sha == pins['items'])


# --------------------------------------------------------------------------- model pieces

def build_head(k=5, feat_dim=1024):
    """Historical B head (identical to hist.build_head for feat_dim=1024)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class KPHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = nn.Sequential(nn.GroupNorm(16, feat_dim), nn.Conv2d(feat_dim, 128, 1), nn.GELU())
            self.rgb = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.GroupNorm(4, 16), nn.GELU(),
                                     nn.Conv2d(16, 16, 3, 2, 1), nn.GroupNorm(4, 16), nn.GELU())
            self.decode = nn.Sequential(nn.Conv2d(144, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
                                        nn.Conv2d(128, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
                                        nn.Conv2d(128, k, 1))

        def forward(self, feat, rgb):
            f = F.interpolate(self.project(feat), size=(128, 128), mode='bilinear', align_corners=False)
            return self.decode(torch.cat([f, self.rgb(rgb)], 1))
    return KPHead()


def point_ce(logits, pts, sigma=2.0):
    """Per-point CE of hist.heatmap_loss (B, K); pts in 512-crop pixels."""
    import torch
    B, K, H, W = logits.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=logits.device), torch.arange(W, device=logits.device), indexing='ij')
    c = (pts + .5) / 4 - .5
    g = -((xx[None, None] - c[..., 0, None, None]) ** 2 + (yy[None, None] - c[..., 1, None, None]) ** 2) / (2 * sigma ** 2)
    return -(g.flatten(2).softmax(-1) * logits.flatten(2).log_softmax(-1)).sum(-1)


def known_mean(ce, known):
    return ce[known].mean() if known.any() else ce.sum() * 0


def decode(logits):
    """Vectorized hist.decode: argmax, clamped 5x5 window, local soft-argmax."""
    import torch
    B, K, H, W = logits.shape
    flat = logits.flatten(2); idx = flat.argmax(-1)
    y0, x0 = idx // W, idx % W
    off = torch.arange(-2, 3, device=logits.device)
    ys = (y0[..., None] + off).clamp(0, H - 1); xs = (x0[..., None] + off).clamp(0, W - 1)
    window = flat.gather(2, (ys[..., :, None] * W + xs[..., None, :]).flatten(2))
    p = window.softmax(-1).view(B, K, 5, 5)
    y = (p.sum(-1) * ys.float()).sum(-1); x = (p.sum(-2) * xs.float()).sum(-1)
    return torch.stack([(x + .5) * 4 - .5, (y + .5) * 4 - .5], -1)


def encode(pe, mean, std, rgb01, first_active, autocast=True, train_embed=False):
    """Final normalized patch tokens of `pe` (B, C, 32, 32). Blocks before `first_active`
    run without autograd; blocks from `first_active` and the final norm build the graph.
    `train_embed` also keeps the patch embedding and tokens in the graph (needs first_active 0)."""
    import torch
    ctx = torch.autocast('cuda', dtype=torch.bfloat16) if autocast else torch.autocast('cpu', enabled=False)
    grad_embed = train_embed and torch.is_grad_enabled()
    if grad_embed and first_active:
        raise ValueError('a trainable embedding needs every block in the graph')
    with ctx:
        with torch.set_grad_enabled(grad_embed):
            x, (H, W) = pe.prepare_tokens_with_masks((rgb01 - mean) / std)
        with torch.no_grad():
            rope = pe.rope_embed(H=H, W=W) if pe.rope_embed is not None else None
            for blk in pe.blocks[:first_active]:
                x = blk(x, rope)
        if not grad_embed:
            x = x.detach()
        for blk in pe.blocks[first_active:]:
            x = blk(x, rope)
        patch = pe.norm(x)[:, pe.n_storage_tokens + 1:]
    B = rgb01.shape[0]
    return patch.transpose(1, 2).reshape(B, patch.shape[-1], H, W).float()


def is_no_decay(module, name):
    return name.endswith('bias') or 'Norm' in type(module).__name__


def filter_keeps(name, filt):
    """Exploratory parameter filters inside a block (UX mode)."""
    leaf = name.split('.')
    if filt == 'norm_ls':
        return leaf[-2].startswith('norm') or leaf[-1] == 'gamma'
    if filt == 'attn':
        return '.attn.' in name
    if filt == 'mlp':
        return '.mlp.' in name
    if filt == 'lora':
        return 'lora_' in leaf[-1]
    raise ValueError(f'unknown filter {filt}')


def add_lora(pe, rank=16, alpha=16, targets=('attn.qkv', 'attn.proj')):
    """Wrap linear layers of every block with a zero-initialised low-rank update (base stays frozen)."""
    import torch
    import torch.nn as nn

    class LoRALinear(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base, self.scale = base, alpha / rank
            self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
            self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            self.in_features, self.out_features = base.in_features, base.out_features

        @property
        def weight(self):
            return self.base.weight

        @property
        def bias(self):
            return self.base.bias

        def forward(self, x):
            return self.base(x) + (x @ self.lora_a.t() @ self.lora_b.t()) * self.scale

    for blk in pe.blocks:
        for target in targets:
            parent_name, attr = target.rsplit('.', 1)
            parent = blk.get_submodule(parent_name)
            setattr(parent, attr, LoRALinear(getattr(parent, attr)))
    return pe


class ParamPlan:
    """Named parameter groups, per-update LRs and staged requires_grad for H0/F/U."""

    def __init__(self, mode, head, pe, cfg):
        self.mode, self.cfg = mode, cfg
        pair = cfg['pair']
        self.groups, self.excluded = [], []
        decay = 0.0 if mode == 'H0' else pair['weight_decay']
        head_peak = cfg['h0']['peak_head_lr'] if mode == 'H0' else pair['head_peak_lr']
        self.head_peak = head_peak
        hd, hn = [], []
        for mname, mod in head.named_modules():
            for pname, p in mod.named_parameters(recurse=False):
                full = f'head.{mname}.{pname}' if mname else f'head.{pname}'
                (hn if mode != 'H0' and is_no_decay(mod, pname) else hd).append((full, p))
        if mode == 'H0':
            self.groups.append(dict(name='head', params=hd, peak=head_peak, start=1, wd=cfg['h0']['weight_decay']))
        else:
            self.groups.append(dict(name='head_decay', params=hd, peak=head_peak, start=1, wd=decay))
            self.groups.append(dict(name='head_no_decay', params=hn, peak=head_peak, start=1, wd=0.0))
        if mode == 'U':
            starts = {}
            for phase in pair['phases']:
                for b in phase['unfrozen_blocks']:
                    starts.setdefault(b, phase['first_update'])
            mods = [('norm', pe.norm, pair['final_norm']['peak_lr'], pair['final_norm']['first_update'])]
            mods += [(f'blocks.{b}', pe.blocks[b], pair['encoder_output_peak_lr'] * pair['encoder_layer_decay'] ** (23 - b), s)
                     for b, s in sorted(starts.items())]
            for gname, root, peak, start in mods:
                d, n = [], []
                for mname, mod in root.named_modules():
                    for pname, p in mod.named_parameters(recurse=False):
                        full = f'aggregator.patch_embed.{gname}' + (f'.{mname}' if mname else '') + f'.{pname}'
                        mask = getattr(mod, 'bias_mask', None) if pname == 'bias' else None
                        if mask is not None and not bool(mask.any()):
                            # The release multiplies this bias by an all-zero mask: its gradient is
                            # identically zero, so it stays frozen (and byte-audited) instead.
                            self.excluded.append(full); p.requires_grad_(False)
                            continue
                        (n if is_no_decay(mod, pname) else d).append((full, p))
                for suffix, plist, wd in (('decay', d, decay), ('no_decay', n, 0.0)):
                    if plist:
                        self.groups.append(dict(name=f'{gname}_{suffix}', params=plist, peak=peak, start=start, wd=wd))
            self.block_start = dict(starts)
        elif mode in ('UP', 'UX'):  # lp_anyup's protected schedule; UX = exploratory scope/filter/rate options
            enc, top = cfg['protected'], len(pe.blocks)
            x = cfg.get('explore', {}) if mode == 'UX' else {}
            cap, layer_decay = x.get('top_cap_lr', enc['top_cap_lr']), x.get('layer_decay', enc['layer_decay'])
            scope, embed, filt = set(x.get('blocks', range(top))), x.get('embed', True), x.get('filter')
            by, trained_blocks = collections.defaultdict(lambda: ([], [])), set()
            for n, p in pe.named_parameters():
                full, pname = f'aggregator.patch_embed.{n}', n.rsplit('.', 1)[-1]
                owner = pe.get_submodule(n.rsplit('.', 1)[0]) if '.' in n else pe
                mask = getattr(owner, 'bias_mask', None) if pname == 'bias' else None
                if n == 'mask_token' or (mask is not None and not bool(mask.any())):
                    self.excluded.append(full); p.requires_grad_(False)  # structurally zero gradient: frozen, audited
                    continue
                if n.startswith('blocks.'):
                    b = int(n.split('.')[1]); d = b + 1
                    on = b in scope and (filt is None or filter_keeps(n, filt))
                    if on:
                        trained_blocks.add(b)
                elif n.startswith('norm.'):
                    d, on = top, x.get('final_norm', True)
                elif n.startswith('patch_embed.') or n in ('cls_token', 'storage_tokens'):
                    d, on = 0, embed
                else:
                    raise ValueError(f'unmapped encoder parameter {n}')
                if not on:
                    p.requires_grad_(False)  # outside the explored scope: frozen and byte-audited
                    continue
                by[d][1 if owner is pe or is_no_decay(owner, pname) else 0].append((full, p))
            for d in sorted(by, reverse=True):
                for suffix, plist, wd in (('decay', by[d][0], decay), ('no_decay', by[d][1], 0.0)):
                    if plist:
                        self.groups.append(dict(name=f'depth{d:02d}_{suffix}', params=plist, start=enc['start_update'],
                                                peak=cap * layer_decay ** (top - d), wd=wd))
            self.ramp = (enc['warmup_initial_ratio'], x.get('warmup_updates', enc['warmup_updates']))
            self.guard = x.get('guard', True)
            self.train_embed = embed and 0 in by
            self.first_block = 0 if self.train_embed else min(trained_blocks, default=top)
        names = [n for g in self.groups for n, _ in g['params']]
        assert len(names) == len(set(names)), 'parameter listed twice'
        assert len({id(p) for g in self.groups for _, p in g['params']}) == len(names)
        self.warmup = cfg['h0']['warmup_updates'] if mode == 'H0' else pair['head_warmup_updates']
        self.enc_warmup = pair['encoder_new_group_warmup_updates']
        self.total = cfg['h0']['cosine_total_updates'] if mode == 'H0' else pair['cosine_total_updates']
        self.floor = cfg['h0']['cosine_floor'] if mode == 'H0' else pair['cosine_floor']

    def envelope(self, u):
        t = min(u - 1, self.total) / self.total
        return self.floor + (1 - self.floor) * .5 * (1 + math.cos(math.pi * t))

    def lr(self, group, u):
        if u < group['start']:
            return 0.0
        if self.mode in ('UP', 'UX') and not group['name'].startswith('head'):
            r0, w = self.ramp  # geometric 0.1x -> 1x over w updates, then the head's envelope
            value = group['peak'] * min(1.0, r0 * (1 / r0) ** ((u - group['start']) / w)) * self.envelope(u)
            head_now = self.head_peak * min(1.0, u / self.warmup) * self.envelope(u)
            return min(value, head_now) if self.guard else value  # parent rule: encoder never above the head
        warm = self.warmup if group['name'].startswith('head') else self.enc_warmup
        return group['peak'] * min(1.0, (u - group['start'] + 1) / warm) * self.envelope(u)

    def first_active(self, u):
        if self.mode in ('UP', 'UX'):
            return self.first_block
        if self.mode != 'U':
            return 24
        active = [b for b, s in self.block_start.items() if u >= s]
        return min(active) if active else 24

    def optimizer(self):
        import torch
        return torch.optim.AdamW([dict(params=[p for _, p in g['params']], weight_decay=g['wd'], lr=0.0)
                                  for g in self.groups], lr=0.0)

    def apply(self, opt, u):
        """Set requires_grad and LR for update u; inactive parameters have no gradient."""
        for g, og in zip(self.groups, opt.param_groups):
            on = u >= g['start']
            for _, p in g['params']:
                p.requires_grad_(on)
                if not on:
                    p.grad = None
            og['lr'] = self.lr(g, u)

    def named(self):
        return [(n, p) for g in self.groups for n, p in g['params']]

    def active(self, u):
        return [p for g in self.groups if u >= g['start'] for _, p in g['params']]

    def describe(self, u):
        return [dict(name=g['name'], start=g['start'], peak=g['peak'], weight_decay=g['wd'], lr=self.lr(g, u),
                     active=u >= g['start'], params=len(g['params']), numel=sum(p.numel() for _, p in g['params']),
                     names=[n for n, _ in g['params']]) for g in self.groups]


class Sampler:
    """Historical B stream: refill with a seeded shuffle of the pool, pop from the end."""

    def __init__(self, pool, seed):
        self.pool, self.rng, self.order = list(pool), random.Random(seed), []

    def take(self, n):
        if len(self.order) < n:
            o = self.pool[:]; self.rng.shuffle(o); self.order += o
        return [self.order.pop() for _ in range(n)]

    def state(self):
        return dict(rng=self.rng.getstate(), order=list(self.order))

    def restore(self, s):
        self.rng.setstate(s['rng']); self.order = list(s['order'])


def tuple_state(x):
    if isinstance(x, list):
        return tuple(tuple_state(v) for v in x)
    return x


# --------------------------------------------------------------------------- data

class Store:
    """Crop-space targets for every item; frozen features (H0/F) and crops on GPU when they fit."""

    def __init__(self, items, device, need_feats, reserve_gb=12):
        import torch
        self.items, self.device = items, device
        A = np.array([it['affine'] for it in items], np.float64)
        pts = np.array([it['points'] for it in items], np.float64)
        crop = np.einsum('nkj,nij->nki', pts, A[:, :, :2]) + A[:, None, :, 2]
        known = np.array([it['known'] for it in items]) & (crop >= 0).all(-1) & (crop < 512).all(-1)
        self.pts = torch.tensor(crop, dtype=torch.float32, device=device)
        self.known = torch.tensor(known, device=device)  # training targets: known and inside the crop
        self.known_np = known
        self.scale = A[:, 0, 0]; self.shift = A[:, :, 2]
        self.need_feats = need_feats
        crops = np.load(OUT/'cache/crops.u8.npy', mmap_mode='r')
        feats = np.load(OUT/'cache/feats.f16.npy', mmap_mode='r') if need_feats else None
        size = crops.nbytes + (feats.nbytes if need_feats else 0)
        on_gpu = device != 'cpu' and torch.cuda.mem_get_info()[0] - size > reserve_gb * 2**30
        self.placement = 'gpu' if on_gpu else 'cpu'
        self.crops = self._load(crops, on_gpu)
        self.feats = self._load(feats, on_gpu) if need_feats else None

    def _load(self, arr, on_gpu):
        import torch
        out = torch.empty(arr.shape, dtype=torch.from_numpy(arr[:1].copy()).dtype, device=self.device if on_gpu else 'cpu')
        for i in range(0, len(arr), 256):
            out[i:i + 256].copy_(torch.from_numpy(np.array(arr[i:i + 256])))
        return out

    def feats_for(self, ids):
        if not self.need_feats:
            raise RuntimeError('U must recompute the trainable encoder online; the frozen final-token cache is refused')
        return self.feats[ids].to(self.device, non_blocking=True).float()

    def rgb_for(self, ids):
        # Contiguous NCHW like the historical cache input: a channels-last view makes cuDNN choose a
        # different bf16 patch-embedding kernel (preflight measured max 0.70 feature difference).
        return self.crops[ids].to(self.device, non_blocking=True).permute(0, 3, 1, 2).float().contiguous() / 255


# --------------------------------------------------------------------------- cache

def cache():
    import torch
    import bottom_geometry_v4_runtime_20261019 as rt
    receipt = OUT/'cache/receipt.json'
    if receipt.exists():
        print('cache exists'); return
    items = load_items(); N = len(items)
    (OUT/'cache').mkdir(exist_ok=True)
    feats = np.lib.format.open_memmap(OUT/'cache/feats.f16.npy.partial', 'w+', np.float16, (N, 1024, 32, 32))
    crops = np.lib.format.open_memmap(OUT/'cache/crops.u8.npy.partial', 'w+', np.uint8, (N, 512, 512, 3))

    class Crops(torch.utils.data.Dataset):
        def __len__(self):
            return N

        def __getitem__(self, i):
            import cv2
            it = items[i]
            im = cv2.cvtColor(cv2.imread(str(ROOT/it['image'])), cv2.COLOR_BGR2RGB)
            return cv2.warpAffine(im, np.array(it['affine']), (512, 512), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

    model = rt.model(RELEASE, trainable=False).eval(); a = model.aggregator
    loader = torch.utils.data.DataLoader(Crops(), batch_size=None, num_workers=8, prefetch_factor=8)
    t0 = time.time()
    with torch.inference_mode():
        for i, crop in enumerate(loader):
            crop = crop.numpy()
            x = torch.from_numpy(crop).permute(2, 0, 1).float().cuda()[None] / 255
            with torch.autocast('cuda', dtype=torch.bfloat16):  # exactly the historical cache path
                o = a.patch_embed((x - a._resnet_mean.view(1, 3, 1, 1)) / a._resnet_std.view(1, 3, 1, 1))
            feats[i] = o['x_norm_patchtokens'].transpose(1, 2).reshape(1024, 32, 32).half().cpu().numpy()
            crops[i] = crop
            if i % 1000 == 0:
                print('CACHE', i, N, round(time.time() - t0), flush=True)
    feats.flush(); crops.flush(); del feats, crops
    for name in ('feats.f16.npy', 'crops.u8.npy'):
        os.replace(OUT/'cache'/f'{name}.partial', OUT/'cache'/name)
    write_new(receipt, dict(items_sha256=sha(OUT/'items.json'), release_sha256=RELEASE_SHA, n=N,
                            feats=dict(shape=[N, 1024, 32, 32], dtype='float16'), crops=dict(shape=[N, 512, 512, 3], dtype='uint8'),
                            preprocessing='historical B: BGR->RGB, warpAffine INTER_LINEAR border 0, /255, resnet mean/std, '
                                          'patch_embed forward under bf16 autocast at batch 1, x_norm_patchtokens',
                            tool_sha256=sha(Path(__file__)), seconds=round(time.time() - t0)))
    print('CACHE done', N, round(time.time() - t0))


# --------------------------------------------------------------------------- evaluation

def predict(head, store, ids, feat_fn, batch=64):
    import torch
    xy, ce = [], []
    with torch.no_grad():
        for j in range(0, len(ids), batch):
            b = ids[j:j + batch]
            logits = head(feat_fn(b), store.rgb_for(b))
            xy.append(decode(logits).cpu()); ce.append(point_ce(logits, store.pts[b]).cpu())
    return torch.cat(xy).numpy(), torch.cat(ce).numpy()


def rows_for(store, ids, xy, ce):
    rows = []
    for n, i in enumerate(ids):
        it = store.items[i]
        native = (xy[n] - store.shift[i]) / store.scale[i]
        gt = np.array(it['points']); known = np.array(it['known'])  # historical evaluation rule
        err = np.linalg.norm(native - gt, axis=1)
        swap = [False] * 5
        for a, b in ((1, 2), (3, 4)):
            if known[a] and known[b]:
                swap[a] = bool(err[a] > np.linalg.norm(native[a] - gt[b]) and np.linalg.norm(gt[a] - gt[b]) > 0)
                swap[b] = bool(err[b] > np.linalg.norm(native[b] - gt[a]))
        box = it['box']; size = max(box[2] - box[0], box[3] - box[1])
        for k in range(5):
            if known[k]:
                rows.append(dict(item=i, dataset=it['dataset'], session=it['session'], group=it['group'],
                                 bottom=it['bottom'], k=k, err_native=float(err[k]),
                                 err_ref=float(err[k] * (.64 if it['dataset'] == 'own' else 1)),
                                 err_norm=float(err[k] / size), swap=swap[k], ce=float(ce[n, k]),
                                 in_crop=bool(store.known_np[i, k])))
    return rows


def stats(rows):
    if not rows:
        return None
    n = np.array([r['err_norm'] for r in rows]); e = np.sort(np.array([r['err_ref'] for r in rows]))
    return dict(n=len(rows), median_norm=float(np.median(n)), pck05=float((n < .05).mean()), pck10=float((n < .10).mean()),
                median_px=float(np.median(e)), cov32_ref=float((e <= 32).mean()),
                p95_ref=float(e[math.ceil(.95 * len(e)) - 1]),  # nearest rank, reference pixels (own x0.64)
                swap_rate=float(np.mean([r['swap'] for r in rows])))


def summarize(rows):
    table = {}
    for ds in ('own', 'pano'):  # deployment-target view: every known point of every view
        s = stats([r for r in rows if r['dataset'] == ds])
        if s:
            table[f'{ds}/all/all'] = s
        for view in ('bottom', 'other'):
            for part, ks in (('all', range(5)), ('paws', PAWS), ('hind', (3, 4))):
                s = stats([r for r in rows if r['dataset'] == ds and r['bottom'] == (view == 'bottom') and r['k'] in ks])
                if s:
                    table[f'{ds}/{view}/{part}'] = s
    for s in sorted({r['session'] for r in rows}):
        for view in ('bottom', 'other'):
            st = stats([r for r in rows if r['session'] == s and r['bottom'] == (view == 'bottom') and r['k'] in PAWS])
            if st:
                table[f'session/{s}/{view}/paws'] = st
    return table


def probe_metrics(rows):
    ce = {ds: [r['ce'] for r in rows if r['dataset'] == ds and r['in_crop']] for ds in ('own', 'pano')}
    bottom = [r for r in rows if r['bottom'] and r['k'] in PAWS]
    out = dict(ce=sum(float(np.mean(v)) for v in ce.values() if v),
               bottom_paw_pck05=float(np.mean([r['err_norm'] < .05 for r in bottom])),
               bottom_paw_swap=float(np.mean([r['swap'] for r in bottom])))
    for ds in ('own', 'pano'):
        b = [r for r in bottom if r['dataset'] == ds]
        out[f'{ds}_ce'] = float(np.mean(ce[ds])) if ce[ds] else None
        out[f'{ds}_bottom_paw_pck05'] = float(np.mean([r['err_norm'] < .05 for r in b])) if b else None
    return out


def plateau(history, conv, min_updates=6000):
    if len(history) < conv['checks'] or history[-1]['update'] < min_updates:
        return False
    w = history[-conv['checks']:]
    if w[-1]['update'] - w[0]['update'] != conv['span_updates']:
        return False
    ce = [r['ce'] for r in w]
    reduction = max(0.0, ce[0] - min(ce)) / max(ce[0], 1e-12)
    span = lambda key: max(r[key] for r in w) - min(r[key] for r in w)
    return (reduction < conv['relative_best_ce_reduction_lt']
            and span('bottom_paw_pck05') <= conv['bottom_paw_pck05_span_pp_le'] / 100 + 1e-12
            and span('bottom_paw_swap') <= conv['bottom_paw_swap_proxy_span_pp_le'] / 100 + 1e-12)


# --------------------------------------------------------------------------- training

class Trainer:
    """One H0/F/U run. All state needed for exact resume lives in `state_dict()`."""

    def __init__(self, mode, seed, cfg, store, head, pe=None, mean=None, std=None, device='cuda', autocast=True):
        import torch
        self.mode, self.seed, self.cfg, self.store, self.head, self.pe = mode, seed, cfg, store, head, pe
        self.mean, self.std, self.device, self.autocast = mean, std, device, autocast
        self.plan = ParamPlan(mode, head, pe, cfg)
        self.opt = self.plan.optimizer()
        self.ema = {n: p.detach().clone() for n, p in self.plan.named()}
        own = [i for i, it in enumerate(store.items) if it['dataset'] == 'own' and it['split'] == 'train']
        pano = [i for i, it in enumerate(store.items) if it['dataset'] == 'pano' and it['split'] == 'train']
        base = 0 if mode == 'H0' else 2000
        self.s_own, self.s_pano = Sampler(own, seed + base), Sampler(pano, seed + base + 1000)
        self.u, self.history, self.log, self.clipped, self.clipped_encoder = 0, [], [], 0, 0
        self.exposure = collections.Counter()
        self.batch = (cfg['task']['own_views_per_update'], cfg['task']['pano_views_per_update'])
        self.ema_decay = cfg['h0']['ema_decay'] if mode == 'H0' else cfg['pair']['ema_decay']
        self.clip = cfg['pair']['grad_clip']
        self.torch = torch

    def features(self, ids, u):
        if self.mode in ('U', 'UP', 'UX'):  # rounded to fp16 like the frozen cache that F reads
            return encode(self.pe, self.mean, self.std, self.store.rgb_for(ids), self.plan.first_active(u), self.autocast,
                          train_embed=getattr(self.plan, 'train_embed', False)).half().float()
        return self.store.feats_for(ids)

    def eval_features(self, ids):
        if self.mode in ('U', 'UP', 'UX'):
            return encode(self.pe, self.mean, self.std, self.store.rgb_for(ids), 24, self.autocast).half().float()
        return self.store.feats_for(ids)

    def step(self):
        torch = self.torch
        u = self.u + 1
        self.plan.apply(self.opt, u)
        b_own, b_pano = self.s_own.take(self.batch[0]), self.s_pano.take(self.batch[1])
        ids = b_own + b_pano
        logits = self.head(self.features(ids, u), self.store.rgb_for(ids))
        ce = point_ce(logits, self.store.pts[ids]); kn = self.store.known[ids]
        n = len(b_own)
        loss_own, loss_pano = known_mean(ce[:n], kn[:n]), known_mean(ce[n:], kn[n:])
        loss = loss_own + loss_pano
        if not torch.isfinite(loss):
            raise FloatingPointError(f'non-finite loss at update {u}')
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        # Head and encoder are clipped separately, so the head's update rule is identical in F and U.
        head = [p for g in self.plan.groups if g['name'].startswith('head') and u >= g['start'] for _, p in g['params']]
        enc = [p for g in self.plan.groups if not g['name'].startswith('head') and u >= g['start'] for _, p in g['params']]
        gn = float(torch.nn.utils.clip_grad_norm_(head, self.clip))
        gn_enc = float(torch.nn.utils.clip_grad_norm_(enc, self.clip)) if enc else 0.0
        if not (math.isfinite(gn) and math.isfinite(gn_enc)):
            raise FloatingPointError(f'non-finite gradient norm at update {u}')
        self.clipped += gn > self.clip
        self.clipped_encoder += gn_enc > self.clip
        self.opt.step()
        with torch.no_grad():  # unopened encoder parameters keep their exact release bytes in the EMA
            for g in self.plan.groups:
                if u >= g['start']:
                    for name, p in g['params']:
                        self.ema[name].mul_(self.ema_decay).add_(p.detach(), alpha=1 - self.ema_decay)
        self.u = u
        for i in ids:
            it = self.store.items[i]
            self.exposure[f"{it['dataset']}/{'bottom' if it['bottom'] else 'other'}"] += 1
        self.exposure['known_points_own'] += int(kn[:n].sum()); self.exposure['known_points_pano'] += int(kn[n:].sum())
        if u % 50 == 0 or u == 1:
            self.log.append(dict(update=u, loss_own=loss_own.item(), loss_pano=loss_pano.item(), grad_norm=gn,
                                 grad_norm_encoder=gn_enc,
                                 lr={g['name']: og['lr'] for g, og in zip(self.plan.groups, self.opt.param_groups)}))
        return float(loss.detach())

    def swap_ema(self):
        """Exchange live parameters with their EMA copies (call twice to undo)."""
        torch = self.torch
        with torch.no_grad():
            for name, p in self.plan.named():
                tmp = p.detach().clone(); p.copy_(self.ema[name]); self.ema[name].copy_(tmp)

    def evaluate(self, ids, use_ema=True):
        if use_ema:
            self.swap_ema()
        try:
            self.head.eval()
            xy, ce = predict(self.head, self.store, ids, self.eval_features)
            return rows_for(self.store, ids, xy, ce)
        finally:
            if use_ema:
                self.swap_ema()

    def state_dict(self):
        torch = self.torch
        return dict(mode=self.mode, seed=self.seed, u=self.u,
                    params={n: p.detach().cpu().clone() for n, p in self.plan.named()},
                    ema={n: v.cpu().clone() for n, v in self.ema.items()},
                    opt=self.opt.state_dict(), samplers=[self.s_own.state(), self.s_pano.state()],
                    rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                             cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []),
                    history=self.history, log=self.log, clipped=self.clipped, clipped_encoder=self.clipped_encoder,
                    exposure=dict(self.exposure),
                    requires_grad={n: p.requires_grad for n, p in self.plan.named()},
                    groups=[dict(name=g['name'], names=[n for n, _ in g['params']]) for g in self.plan.groups])

    def load_state_dict(self, s):
        torch = self.torch
        assert (s['mode'], s['seed']) == (self.mode, self.seed)
        assert [g['names'] for g in s['groups']] == [[n for n, _ in g['params']] for g in self.plan.groups]
        with torch.no_grad():
            for n, p in self.plan.named():
                p.copy_(s['params'][n]); self.ema[n].copy_(s['ema'][n])
        self.opt.load_state_dict(s['opt'])
        self.s_own.restore(s['samplers'][0]); self.s_pano.restore(s['samplers'][1])
        random.setstate(tuple_state(s['rng']['python'])); np.random.set_state(s['rng']['numpy'])
        torch.set_rng_state(s['rng']['torch'])
        if s['rng']['cuda']:
            torch.cuda.set_rng_state_all(s['rng']['cuda'])
        self.u, self.history, self.log, self.clipped = s['u'], s['history'], s['log'], s['clipped']
        self.clipped_encoder = s['clipped_encoder']
        self.exposure = collections.Counter(s['exposure'])
        self.plan.apply(self.opt, self.u)  # staged requires_grad is a function of the committed update
        if self.u:
            assert {n: p.requires_grad for n, p in self.plan.named()} == s['requires_grad']

    def save(self, path):
        tmp = Path(str(path) + '.tmp')
        self.torch.save(self.state_dict(), tmp); os.replace(tmp, path)


def release_parts(device='cuda'):
    import torch
    import bottom_geometry_v4_runtime_20261019 as rt
    model = rt.model(RELEASE, trainable=False).eval()
    a = model.aggregator
    return model, a.patch_embed, a._resnet_mean.view(1, 3, 1, 1).to(device), a._resnet_std.view(1, 3, 1, 1).to(device)


def param_hashes(module, prefix):
    return {f'{prefix}.{n}': hashlib.sha256(p.detach().cpu().contiguous().view(-1).view(dtype=__import__('torch').uint8)
                                            .numpy().tobytes()).hexdigest() for n, p in module.named_parameters()}


def stage_audit(trainer, release_hashes, release_copy):
    """Frozen byte identity of not-yet-opened encoder parameters + drift of opened ones."""
    plan, u = trainer.plan, trainer.u
    opened = {n for g in plan.groups if u >= g['start'] for n, _ in g['params']}
    now = param_hashes(trainer.pe, 'aggregator.patch_embed')
    changed_frozen = sorted(n for n, h in now.items() if n not in opened and h != release_hashes[n])
    drift = {}
    for g in plan.groups:
        if g['name'].startswith('head') or u < g['start']:
            continue
        num = sum(float((p.detach().cpu().double() - release_copy[n].double()).norm() ** 2) for n, p in g['params'])
        den = sum(float(release_copy[n].double().norm() ** 2) for n, _ in g['params'])
        drift[g['name']] = math.sqrt(num / den) if den > 0 else math.sqrt(num)  # absolute norm for zero-init (LoRA B)
    return dict(update=u, frozen_unchanged=not changed_frozen, changed_frozen=changed_frozen[:20],
                relative_drift_from_release=drift, groups=plan.describe(u), clipped_head_updates=trainer.clipped,
                clipped_encoder_updates=trainer.clipped_encoder)


def train(mode, seed):
    import torch
    cfg = design_for(mode)
    run = RUNS/f'{mode}_seed{seed}'
    if (run/'DONE.json').exists():
        print('done already:', run); return
    run.mkdir(parents=True, exist_ok=True)
    items = load_items()
    if mode in ('F', 'U', 'FP', 'UP'):
        h0 = json.loads((RUNS/f'H0_seed{seed}/DONE.json').read_text())
        if not h0['converged'] and not SMOKE:  # smoke checks mechanics only
            raise SystemExit('H0 did not converge (head_not_converged): F/U are not started')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    pe = mean = std = None
    online = mode in ('U', 'UP')
    if online:
        _, pe, mean, std = release_parts()
    store = Store(items, 'cuda', need_feats=not online)
    torch.manual_seed(seed)
    head = build_head(5).cuda()
    if mode != 'H0':
        head.load_state_dict(torch.load(RUNS/f'H0_seed{seed}/endpoint.pt', weights_only=False)['ema_head'])
    head.eval()  # GroupNorm has no running state; eval forward mode in every branch
    t = Trainer(mode, seed, cfg, store, head, pe, mean, std)
    release_hashes = release_copy = None
    if online:
        release_hashes = param_hashes(pe, 'aggregator.patch_embed')
        release_copy = {n: p.detach().cpu().clone() for n, p in t.plan.named() if not n.startswith('head')}
    if (run/'state.pt').exists():
        t.load_state_dict(torch.load(run/'state.pt', weights_only=False))
        print('resumed at update', t.u, flush=True)
    else:
        write(run/'run_protocol.json', dict(mode=mode, seed=seed, start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                            host=os.uname().nodename, cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
                                            slurm_job=os.environ.get('SLURM_JOB_ID'), slurm_step=os.environ.get('SLURM_STEP_ID'),
                                            data_placement=store.placement, groups=t.plan.describe(1),
                                            structurally_unused_frozen=t.plan.excluded,
                                            clipping='head and encoder parameter sets clipped separately, each to max norm '
                                                     f"{cfg['pair']['grad_clip']}; online features rounded to fp16 like the cache",
                                            registration=str(PROTECTED.relative_to(ROOT)) if mode in ('FP', 'UP') else None,
                                            registration_sha256=sha(PROTECTED) if mode in ('FP', 'UP') else None,
                                            tool_sha256=sha(Path(__file__)), items_sha256=sha(OUT/'items.json')))
    train_ids = [i for i, it in enumerate(items) if it['split'] == 'train']
    dev_ids = [i for i, it in enumerate(items) if it['split'] == 'dev']
    if SMOKE:
        train_ids, dev_ids = subset(items, train_ids, 32), subset(items, dev_ids, 16)
    conv = cfg['h0']['convergence']
    total = cfg['h0']['max_updates'] if mode == 'H0' else cfg['pair']['updates_per_branch']
    stage_ends = set(cfg['pair']['dev_eval_updates']) if mode != 'H0' else set()
    t0 = time.time()
    # A resumed H0 that had already reached the plateau must not train past it.
    converged = mode == 'H0' and t.u % CHECK_EVERY == 0 and t.u >= cfg['h0']['min_updates'] \
        and plateau(t.history, conv, cfg['h0']['min_updates'])
    while t.u < total and not converged:
        t.step()
        u = t.u
        if mode == 'H0' and u % CHECK_EVERY == 0:
            metrics = probe_metrics(t.evaluate(train_ids))
            t.history.append(dict(update=u, **metrics))
            print(f'H0 seed{seed} u{u} ce {metrics["ce"]:.4f} bottom paw PCK {100*metrics["bottom_paw_pck05"]:.2f}% '
                  f'swap {100*metrics["bottom_paw_swap"]:.2f}% {round(time.time()-t0)}s', flush=True)
            converged = u >= cfg['h0']['min_updates'] and plateau(t.history, conv, cfg['h0']['min_updates'])
        if u in stage_ends:
            snapshot = dict(update=u, raw_head=t.head.state_dict(),
                            ema_head={k[5:]: v.cpu() for k, v in t.ema.items() if k.startswith('head.')})
            if online:
                snapshot['raw_encoder'] = {n: p.detach().cpu() for n, p in t.plan.named() if not n.startswith('head')}
                snapshot['ema_encoder'] = {n: v.cpu() for n, v in t.ema.items() if not n.startswith('head')}
                write(run/f'stage_audit_u{u:05d}.json', stage_audit(t, release_hashes, release_copy))
            torch.save(snapshot, run/f'snapshot_u{u:05d}.pt')
            for kind in ('ema', 'raw'):
                rows = t.evaluate(dev_ids, use_ema=kind == 'ema')
                write(run/f'dev_u{u:05d}_{kind}.json', dict(update=u, weights=kind, summary=summarize(rows), rows=rows))
            print(f'{mode} seed{seed} stage endpoint {u} evaluated', flush=True)
        if u % CHECK_EVERY == 0 or u in stage_ends:  # after any endpoint work, so a resume never skips it
            t.save(run/'state.pt')
    if mode == 'H0':
        ema_head = {k[5:]: v.cpu() for k, v in t.ema.items() if k.startswith('head.')}
        torch.save(dict(update=t.u, converged=converged, raw_head=t.head.state_dict(), ema_head=ema_head), run/'endpoint.pt')
        for kind in ('ema', 'raw'):
            rows = t.evaluate(dev_ids, use_ema=kind == 'ema')
            write(run/f'dev_endpoint_{kind}.json', dict(update=t.u, weights=kind, summary=summarize(rows), rows=rows))
    write(run/'train_log.json', dict(log=t.log, history=t.history, exposure=dict(t.exposure), clipped_head=t.clipped,
                                     clipped_encoder=t.clipped_encoder))
    write_new(run/'DONE.json', dict(mode=mode, seed=seed, updates=t.u, converged=converged if mode == 'H0' else None,
                                    status=('converged' if converged else 'head_not_converged') if mode == 'H0' else 'complete',
                                    clipped_head_updates=t.clipped, clipped_encoder_updates=t.clipped_encoder,
                                    exposure=dict(t.exposure), seconds=round(time.time() - t0),
                                    finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    print('DONE', mode, seed, t.u, 'converged' if converged else '', flush=True)


# --------------------------------------------------------------------------- preflight

def preflight():
    """GPU engineering checks on 64 fixed TRAIN items; every update is discarded."""
    import torch
    cfg = design()
    dest = RUNS/'preflight'
    if (dest/'preflight.json').exists():
        if json.loads((dest/'preflight.json').read_text())['passed']:
            print('preflight passed earlier'); return
        raise SystemExit('preflight failed earlier: inspect preflight/preflight.json before removing it')
    dest.mkdir(parents=True, exist_ok=True)
    items = load_items()
    model, pe, mean, std = release_parts()
    topo = dict(feature_flow=cfg['feature_flow'], modules={})
    for name, mod in [('aggregator.patch_embed.patch_embed', pe.patch_embed), ('aggregator.patch_embed.rope_embed', pe.rope_embed),
                      ('aggregator.patch_embed.norm', pe.norm)] + [(f'aggregator.patch_embed.blocks.{i}', b) for i, b in enumerate(pe.blocks)] \
            + [('aggregator.frame_blocks', model.aggregator.frame_blocks), ('aggregator.inter_frame_blocks', model.aggregator.inter_frame_blocks),
               ('camera_head', model.camera_head), ('dense_head', model.dense_head)]:
        topo['modules'][name] = dict(params=sum(p.numel() for p in mod.parameters()),
                                     tensors=[n for n, _ in mod.named_parameters()][:12])
    topo['tokens'] = dict(cls=1, storage=pe.n_storage_tokens, patch_grid=[32, 32], dim=pe.norm.weight.shape[0])
    write(RUNS/'parameter_topology.json', topo)
    rng = random.Random(20261024); probe = []
    for ds in ('own', 'pano'):
        for bottom in (True, False):
            pool = [i for i, it in enumerate(items) if it['split'] == 'train' and it['dataset'] == ds and it['bottom'] == bottom]
            probe += sorted(rng.sample(pool, 16))
    store = Store(items, 'cuda', need_feats=True)
    report = dict(probe=probe, data_placement=store.placement)
    with torch.no_grad():
        cached = store.feats_for(probe)
        one = torch.cat([encode(pe, mean, std, store.rgb_for([i]), 24) for i in probe])
        many = torch.cat([encode(pe, mean, std, store.rgb_for(probe[j:j + 16]), 24) for j in range(0, 64, 16)])
    report['cache_vs_online_batch1_max_abs'] = float((cached - one.half().float()).abs().max())
    report['cache_vs_online_batch16_max_abs'] = float((cached - many.half().float()).abs().max())
    report['online_last_matches_cache_batch1'] = report['cache_vs_online_batch1_max_abs'] == 0.0
    # Gradient reach in each U phase: only the opened blocks, the final norm and the head receive gradients.
    reach = {}
    for u in [p['first_update'] for p in cfg['pair']['phases']]:
        torch.manual_seed(17); head = build_head(5).cuda().eval()
        plan = ParamPlan('U', head, pe, cfg); opt = plan.optimizer(); plan.apply(opt, u)
        model.zero_grad(set_to_none=True)
        feat = encode(pe, mean, std, store.rgb_for(probe[:16]), plan.first_active(u))
        loss = known_mean(point_ce(head(feat, store.rgb_for(probe[:16])), store.pts[probe[:16]]), store.known[probe[:16]])
        loss.backward()
        opened = {n for n, p in plan.named() if p.requires_grad}
        with_grad = {f'aggregator.{n}' for n, p in model.aggregator.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
        with_grad |= {f'head.{n}' for n, p in head.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
        others = {n for n, p in model.named_parameters() if p.grad is not None and not n.startswith('aggregator.patch_embed')}
        reach[u] = dict(first_active=plan.first_active(u), opened=len(opened), with_nonzero_grad=len(with_grad),
                        opened_without_grad=sorted(opened - with_grad), structurally_unused_frozen=plan.excluded,
                        all_opened_have_grad=opened <= with_grad, no_frozen_grad=with_grad <= opened and not others)
        model.zero_grad(set_to_none=True); head.zero_grad(set_to_none=True)
        for _, p in plan.named():
            p.requires_grad_(False)
    report['gradient_reach'] = reach
    # Resume across the 2,000/2,001 stage boundary on a disposable state, then frozen byte identity.
    hashes = param_hashes(pe, 'aggregator.patch_embed')
    pristine = {n: p.detach().clone() for n, p in pe.named_parameters()}
    sub = [i for i, it in enumerate(items) if it['split'] == 'train']

    def fresh():
        torch.manual_seed(17); head = build_head(5).cuda().eval()
        with torch.no_grad():
            for n, p in pe.named_parameters():
                p.copy_(pristine[n])
        tr = Trainer('U', 17, cfg, store, head, pe, mean, std)
        tr.s_own.pool = [i for i in sub if items[i]['dataset'] == 'own']
        tr.s_pano.pool = [i for i in sub if items[i]['dataset'] == 'pano']
        tr.u = cfg['pair']['phases'][1]['first_update'] - 4  # three updates, save, then cross the stage boundary
        return tr
    store.need_feats = False
    a = fresh()
    for _ in range(3):
        a.step()
    a.save(dest/'boundary_state.pt')
    for _ in range(2):
        a.step()
    ref = {n: p.detach().clone() for n, p in a.plan.named()}
    b = fresh(); b.load_state_dict(torch.load(dest/'boundary_state.pt', weights_only=False))
    for _ in range(2):
        b.step()
    diff = max(float((p.detach() - ref[n]).abs().max()) for n, p in b.plan.named())
    report['resume_across_stage_boundary_max_abs_param_diff'] = diff
    report['resume_requires_grad_match'] = {n: p.requires_grad for n, p in a.plan.named()} == {n: p.requires_grad for n, p in b.plan.named()}
    audit = stage_audit(b, hashes, {n: pristine[n.replace('aggregator.patch_embed.', '')].cpu() for n, _ in b.plan.named() if not n.startswith('head')})
    report['frozen_unchanged_after_updates'] = audit['frozen_unchanged']
    report['opened_after_boundary'] = [g['name'] for g in audit['groups'] if g['active']]
    report['accumulation'] = 'none: physical batch = registered effective batch'
    report['passed'] = bool(all(r['all_opened_have_grad'] and r['no_frozen_grad'] for r in reach.values())
                            and audit['frozen_unchanged'] and report['resume_requires_grad_match'] and diff < 1e-5
                            and report['cache_vs_online_batch1_max_abs'] < 1e-2)
    report['updates_discarded'] = True
    (dest/'boundary_state.pt').unlink()
    write_new(dest/'preflight.json', report)
    print('PREFLIGHT', json.dumps({k: v for k, v in report.items() if k not in ('probe', 'gradient_reach')}))
    if not report['passed']:
        raise SystemExit('preflight failed')


def preflight_protected():
    """GPU checks for the protected arm on fixed TRAIN items; every update is discarded."""
    import torch
    cfg = protected_design()
    dest = RUNS/'preflight_protected'
    if (dest/'preflight.json').exists():
        if json.loads((dest/'preflight.json').read_text())['passed']:
            print('protected preflight passed earlier'); return
        raise SystemExit('protected preflight failed earlier: inspect preflight_protected/preflight.json before removing it')
    dest.mkdir(parents=True, exist_ok=True)
    items = load_items()
    model, pe, mean, std = release_parts()
    store = Store(items, 'cuda', need_feats=False)
    probe = subset(items, [i for i, it in enumerate(items) if it['split'] == 'train'], 4)
    torch.manual_seed(17); head = build_head(5).cuda().eval()
    plan = ParamPlan('UP', head, pe, cfg); opt = plan.optimizer(); plan.apply(opt, 1)
    feat = encode(pe, mean, std, store.rgb_for(probe), plan.first_active(1), train_embed=True)
    loss = known_mean(point_ce(head(feat, store.rgb_for(probe)), store.pts[probe]), store.known[probe])
    loss.backward()
    opened = {n for n, _ in plan.named()}
    named = dict(plan.named())
    without = sorted(n for n in opened if named[n].grad is None or float(named[n].grad.abs().sum()) == 0)
    frozen_with_grad = sorted(n for n, p in model.named_parameters() if p.grad is not None and n not in opened)
    excluded_grad = [n for n in plan.excluded if dict(model.named_parameters())[n].grad is not None]
    groups = {g['name']: g for g in plan.groups}
    lr_table = {u: {name: plan.lr(groups[name], u) for name in ('head_decay', 'depth24_decay', 'depth23_decay', 'depth01_decay', 'depth00_decay')}
                for u in (1, 1001, 2001, 6001, cfg['pair']['updates_per_branch'])}
    model.zero_grad(set_to_none=True); head.zero_grad(set_to_none=True)
    for _, p in plan.named():
        p.requires_grad_(False)
    # A few real updates on a disposable state: speed, memory, excluded bytes unchanged.
    before = {n: p.detach().clone() for n, p in model.named_parameters() if n in plan.excluded}
    torch.manual_seed(17); head = build_head(5).cuda().eval()
    t = Trainer('UP', 17, cfg, store, head, pe, mean, std)
    torch.cuda.reset_peak_memory_stats(); t.step(); torch.cuda.synchronize(); t0 = time.time()
    for _ in range(10):
        t.step()
    torch.cuda.synchronize()
    seconds = (time.time() - t0) / 10
    params = dict(model.named_parameters())
    report = dict(opened_tensors=len(opened), opened_without_grad=without, frozen_with_grad=frozen_with_grad,
                  excluded=plan.excluded, excluded_with_grad=excluded_grad,
                  excluded_unchanged=all(torch.equal(params[n], before[n]) for n in before),
                  encoder_params_trained=sum(p.numel() for n, p in plan.named() if not n.startswith('head')),
                  groups=[dict(name=g['name'], peak=g['peak'], tensors=len(g['params'])) for g in plan.groups],
                  lr_table=lr_table, seconds_per_update=seconds, peak_memory_gb=torch.cuda.max_memory_allocated() / 2**30,
                  estimated_hours_for_pair=seconds * cfg['pair']['updates_per_branch'] / 3600, updates_discarded=True)
    report['passed'] = bool(not without and not frozen_with_grad and not excluded_grad and report['excluded_unchanged'])
    write_new(dest/'preflight.json', report)
    print('PROTECTED PREFLIGHT', json.dumps({k: v for k, v in report.items() if k not in ('groups', 'lr_table', 'excluded')}))
    if not report['passed']:
        raise SystemExit('protected preflight failed')


# --------------------------------------------------------------------------- gates

def paw_table(path):
    return json.loads(Path(path).read_text())['summary']


def gate(seed, pair='FU'):
    ref, arm = ('F', 'U') if pair == 'FU' else ('FP', 'UP')
    full = design_for(arm); cfg = full['screening']; end = full['pair']['updates_per_branch']
    f = paw_table(RUNS/f'{ref}_seed{seed}/dev_u{end:05d}_ema.json'); u = paw_table(RUNS/f'{arm}_seed{seed}/dev_u{end:05d}_ema.json')
    integrity = all(json.loads((RUNS/f'{m}_seed{seed}/DONE.json').read_text())['status'] == 'complete' for m in (ref, arm))
    audits = [json.loads(p.read_text()) for p in sorted((RUNS/f'{arm}_seed{seed}').glob('stage_audit_u*.json'))]
    integrity &= len(audits) == len(full['pair']['dev_eval_updates']) and all(a['frozen_unchanged'] for a in audits)
    rows, retention_ok = {}, True
    for key in ('own/bottom/paws', 'own/other/paws', 'pano/bottom/paws', 'pano/other/paws'):
        d_pck = 100 * (u[key]['pck05'] - f[key]['pck05']); d_swap = 100 * (u[key]['swap_rate'] - f[key]['swap_rate'])
        ok = d_pck >= -cfg['retention']['each_domain_and_layer_paw_pck05_drop_pp_le'] - 1e-9 and \
            d_swap <= cfg['retention']['each_domain_and_layer_paw_swap_proxy_increase_pp_le'] + 1e-9
        retention_ok &= ok
        rows[key] = dict(F_pck05=f[key]['pck05'], U_pck05=u[key]['pck05'], delta_pck_pp=d_pck,
                         F_swap=f[key]['swap_rate'], U_swap=u[key]['swap_rate'], delta_swap_pp=d_swap, retention_ok=ok)
    pb = rows['pano/bottom/paws']
    rel_swap = (pb['F_swap'] - pb['U_swap']) / pb['F_swap'] if pb['F_swap'] > 0 else 0.0
    benefit = pb['delta_pck_pp'] >= cfg['benefit']['pano_bottom_paw_pck05_gain_pp_ge'] or \
        (rel_swap >= cfg['benefit']['or_swap_proxy_relative_reduction_ge'] and pb['delta_pck_pp'] >= 0)
    result = dict(seed=seed, pair=f'{arm} minus {ref} at update {end} (EMA)', integrity_ok=integrity, retention_ok=retention_ok,
                  benefit_signal=benefit,
                  pano_bottom_swap_relative_reduction=rel_swap, strata=rows,
                  expand_to_more_seeds=integrity and retention_ok,
                  rule='expand only without integrity failure or a registered >0.5pp degradation; benefit is judged on '
                       'three paired seeds, not on seed 17 alone')
    write(RUNS/(f'gate_seed{seed}.json' if pair == 'FU' else f'gate_protected_seed{seed}.json'), result)
    print('GATE', json.dumps({k: v for k, v in result.items() if k != 'strata'}))
    sys.exit(0 if result['expand_to_more_seeds'] else 3)


def summary():
    seeds = [s for s in design()['seeds'] if (RUNS/f'gate_seed{s}.json').exists()]
    gates = {s: json.loads((RUNS/f'gate_seed{s}.json').read_text()) for s in seeds}
    h0 = {s: json.loads((RUNS/f'H0_seed{s}/DONE.json').read_text()) for s in design()['seeds'] if (RUNS/f'H0_seed{s}/DONE.json').exists()}
    deltas = [g['strata']['pano/bottom/paws']['delta_pck_pp'] for g in gates.values()]
    report = dict(h0=h0, gates=gates, seeds_with_pairs=seeds,
                  mean_pano_bottom_delta_pp=float(np.mean(deltas)) if deltas else None,
                  positive_directions=sum(d > 0 for d in deltas),
                  candidate_beneficial=len(seeds) == 3 and all(g['retention_ok'] and g['integrity_ok'] for g in gates.values())
                  and sum(g['benefit_signal'] for g in gates.values()) >= 2
                  and float(np.mean(deltas)) >= design()['screening']['benefit']['pano_bottom_paw_pck05_gain_pp_ge'],
                  note='U is a candidate only with three paired seeds, >=2/3 positive and no seed past the 0.5pp '
                       'degradation gate; otherwise F/H0 is the preferred result. 2D only: no geometry/unknown-rig claim.')
    write(RUNS/'summary.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('gates', 'h0')}, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['prepare', 'cache', 'preflight', 'preflight-protected', 'train', 'gate', 'summary'])
    ap.add_argument('--mode', choices=['H0', 'F', 'U', 'FP', 'UP'])
    ap.add_argument('--seed', type=int)
    ap.add_argument('--pair', choices=['FU', 'protected'], default='FU')
    a = ap.parse_args()
    {'prepare': prepare, 'cache': cache, 'preflight': preflight, 'preflight-protected': preflight_protected,
     'train': lambda: train(a.mode, a.seed), 'gate': lambda: gate(a.seed, a.pair), 'summary': summary}[a.step]()
