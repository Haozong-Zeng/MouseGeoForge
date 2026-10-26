"""Visible-mouse semantic segmentation on VGGT-Omega image-encoder features (Depth-Anything-inspired).

Design: configs/experiments/mouse_semseg_depth_anything_20261024.json and
docs/mouse_semseg_depth_anything_route_20261024.md. Borrowed from Depth Anything's
semseg recipe: the last four encoder layers -> 4/2/1/0.5 feature pyramid ->
Mask2Former-style decoder. The encoder is VGGT-Omega's `aggregator.patch_embed`
(user decision 2026-10-24: no actual Depth Anything encoder, so SEG-DA is not run;
the historical S1 checkpoint is not on the cluster, so SEG-R is not run).

Arms share the decoder, its initial parameters, data order, loss and schedule and
train in one process: SEG-L = block 23 replicated four times, SEG-M = blocks
20/21/22/23. Full-image 512 letterbox, padding supervised as background. Outputs
go to output/mouse_semseg_vggt_20261024/ (data disk).

    prepare                       frozen item list, letterbox geometry, probes, receipts
    cache --shard i --shards n    frozen four-layer features + 512 labels (GPU)
    preflight                     API/parity/round-trip/gradient/resume checks (GPU; updates discarded)
    train --seed N                SEG-L and SEG-M to the common TRAIN plateau, DEV at fixed updates
    gate --seed N                 SEG-M minus SEG-L screening (exit 3 = stop seed expansion)
    summary                       paired three-seed report
"""
import argparse
import collections
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
OUT = ROOT/'output/mouse_semseg_vggt_20261024'
DESIGN = ROOT/'configs/experiments/mouse_semseg_depth_anything_20261024.json'
ITEMS_SRC = Path('/grid/hou/data/Tony/dataset/MouseGeoForge-experiment-data-20261023/reproduction/output/'
                 'vggt_mouse_segmentation_16h_20261020/items.json')
RELEASE = ROOT/'vggt-omega/vggt_omega_1b_512.pt'
RELEASE_SHA = 'c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934'
SIDE = 512
LAYERS = [20, 21, 22, 23]
ARMS = {'SEG_L': [3, 3, 3, 3], 'SEG_M': [0, 1, 2, 3]}
STRATA = ['real/bottom', 'real/other', 'synthetic/bottom', 'synthetic/other']
# MGF_SMOKE=1: shortened schedule, small probe/DEV subsets, run folders under OUT/smoke;
# the frozen item list and the feature cache are shared with the real runs.
SMOKE = os.environ.get('MGF_SMOKE') == '1'
RUNS = OUT/'smoke' if SMOKE else OUT
CHECK_EVERY = 10 if SMOKE else 500
IGNORE = 255


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
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=1) + '\n'); os.replace(tmp, path)


def design():
    cfg = json.loads(DESIGN.read_text())
    if SMOKE:
        cfg['frozen_head_pair'].update(min_updates=20, max_updates=40, warmup_updates=5, cosine_total_updates=40,
                                       dev_eval_updates=[20, 40])
    return cfg


def stratum(it):
    return f"{it['domain']}/{'bottom' if it['bottom'] else 'other'}"


def balance_key(it):
    return f"{it['cohort']}|{it['rig'] or it['group']}"


# --------------------------------------------------------------------------- letterbox and labels

def geometry(h, w, side=SIDE):
    s = side / max(h, w)
    nh, nw = round(h * s), round(w * s)
    top, left = (side - nh) // 2, (side - nw) // 2
    return dict(scale=s, size=[nh, nw], top=top, left=left, roi=[left, top, left + nw, top + nh])


def letterbox(arr, g, interpolation, fill=0, side=SIDE):
    out = np.full((side, side) + arr.shape[2:], fill, arr.dtype)
    nh, nw = g['size']
    out[g['top']:g['top'] + nh, g['left']:g['left'] + nw] = cv2.resize(arr, (nw, nh), interpolation=interpolation)
    return out


def roi(arr, g):
    """Crop the valid (non-padding) region of a (..., side, side) array."""
    nh, nw = g['size']
    return arr[..., g['top']:g['top'] + nh, g['left']:g['left'] + nw]


def to_native(prob, g, h, w):
    """(C, side, side) probabilities -> (C, h, w): crop the ROI first, then resize."""
    return np.stack([cv2.resize(np.ascontiguousarray(c), (w, h), interpolation=cv2.INTER_LINEAR) for c in roi(prob, g)])


def read_label(it):
    """Native labels: 1 visible mouse, 0 background, 255 unknown."""
    p = ROOT/it['label']
    if p.suffix == '.npz':
        lab = np.load(p)['body'].astype(np.uint8)  # primary-ray skin truth; glass/hair excluded
    else:
        lab = (cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
    if list(lab.shape) != list(it['native_hw']):
        raise ValueError(f"label shape {lab.shape} != native_hw {it['native_hw']} for {it['id']}")
    return lab


def read_image(it):
    return cv2.cvtColor(cv2.imread(str(ROOT/it['image'])), cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- metrics

def boundary(mask):
    er = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return mask & ~er.astype(bool)


def boundary_f1(pred, gt, valid, tol):
    """BF score: boundary pixels (mask minus 3x3 erosion) matched within `tol` px (Euclidean)."""
    pb, gb = boundary(pred) & valid, boundary(gt) & valid
    if not pb.any() and not gb.any():
        return 1.0
    if not pb.any() or not gb.any():
        return 0.0
    dg = cv2.distanceTransform((~gb).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dp = cv2.distanceTransform((~pb).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    precision, recall = float((dg[pb] <= tol).mean()), float((dp[gb] <= tol).mean())
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def overlap(pred, gt, valid):
    p, g = pred & valid, gt & valid
    inter, union, total = int((p & g).sum()), int((p | g).sum()), int(p.sum() + g.sum())
    return (inter / union if union else 1.0), (2 * inter / total if total else 1.0)


def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


# --------------------------------------------------------------------------- decoder

def sine_pos(h, w, d, device):
    import torch
    y = torch.arange(1, h + 1, dtype=torch.float32, device=device)[:, None].expand(h, w) / (h + 1e-6) * 2 * math.pi
    x = torch.arange(1, w + 1, dtype=torch.float32, device=device)[None, :].expand(h, w) / (w + 1e-6) * 2 * math.pi
    dim_t = 10000 ** (2 * (torch.arange(d // 2, device=device) // 2) / (d // 2))
    px, py = x[..., None] / dim_t, y[..., None] / dim_t
    px = torch.stack((px[..., 0::2].sin(), px[..., 1::2].cos()), -1).flatten(-2)
    py = torch.stack((py[..., 0::2].sin(), py[..., 1::2].cos()), -1).flatten(-2)
    return torch.cat((py, px), -1).reshape(h * w, d)


def build_decoder(in_ch=1024, d=256, pixel_layers=3, decoder_layers=3, heads=8, queries=16, classes=2, ffn=1024):
    """Compact Mask2Former: Feature2Pyramid(4/2/1/0.5) on 256-d projections, a multi-scale
    transformer pixel decoder (standard attention), masked-attention query decoder."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Layer(nn.Module):  # post-norm attention; the residual is the query without its position code
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
            self.norm = nn.LayerNorm(d)

        def forward(self, x, q, k, v, mask=None):
            return self.norm(x + self.attn(q, k, v, attn_mask=mask, need_weights=False)[0])

    class FFN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(d, ffn), nn.ReLU(), nn.Linear(ffn, d))
            self.norm = nn.LayerNorm(d)

        def forward(self, x):
            return self.norm(x + self.net(x))

    class Mask2FormerLite(nn.Module):
        def __init__(self):
            super().__init__()
            gn = lambda: nn.GroupNorm(32, d)
            self.proj = nn.ModuleList([nn.Sequential(nn.Conv2d(in_ch, d, 1), gn()) for _ in range(4)])
            self.up4 = nn.Sequential(nn.ConvTranspose2d(d, d, 2, 2), gn(), nn.GELU(), nn.ConvTranspose2d(d, d, 2, 2))
            self.up2 = nn.ConvTranspose2d(d, d, 2, 2)
            self.down2 = nn.MaxPool2d(2)
            self.level_embed = nn.Parameter(torch.zeros(3, d))
            self.enc_attn = nn.ModuleList([Layer() for _ in range(pixel_layers)])
            self.enc_ffn = nn.ModuleList([FFN() for _ in range(pixel_layers)])
            self.lateral = nn.Sequential(nn.Conv2d(d, d, 1), gn())
            self.output_conv = nn.Sequential(nn.Conv2d(d, d, 3, padding=1), gn(), nn.ReLU())
            self.mask_features = nn.Conv2d(d, d, 1)
            self.query_feat = nn.Embedding(queries, d)
            self.query_pos = nn.Embedding(queries, d)
            self.dec_level_embed = nn.Embedding(3, d)
            self.cross = nn.ModuleList([Layer() for _ in range(decoder_layers)])
            self.self_attn = nn.ModuleList([Layer() for _ in range(decoder_layers)])
            self.dec_ffn = nn.ModuleList([FFN() for _ in range(decoder_layers)])
            self.decoder_norm = nn.LayerNorm(d)
            self.class_embed = nn.Linear(d, classes + 1)
            self.mask_embed = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
            nn.init.normal_(self.level_embed)

        def heads(self, x, mask_features, size):
            x = self.decoder_norm(x)
            masks = torch.einsum('bqc,bchw->bqhw', self.mask_embed(x), mask_features)
            attn = F.interpolate(masks, size=size, mode='bilinear', align_corners=False)
            attn = (attn.sigmoid().flatten(2) < 0.5).repeat_interleave(heads, 0).detach()
            attn[attn.all(-1)] = False  # a query that masks everything attends everywhere
            return self.class_embed(x), masks, attn

        def forward(self, feats):  # feats: (B, 4, C, h, w), early -> late layer
            B = feats.shape[0]
            p = [proj(feats[:, i]) for i, proj in enumerate(self.proj)]
            pyramid = [self.up4(p[0]), self.up2(p[1]), p[2], self.down2(p[3])]  # 4x, 2x, 1x, 0.5x
            levels = [pyramid[3], pyramid[2], pyramid[1]]  # 1/32, 1/16, 1/8 of the 512 input
            shapes = [lv.shape[-2:] for lv in levels]
            pos = [sine_pos(h, w, d, feats.device) + self.level_embed[i] for i, (h, w) in enumerate(shapes)]
            src = torch.cat([lv.flatten(2).transpose(1, 2) for lv in levels], 1)
            pos_all = torch.cat(pos, 0)[None]
            for attn, ffn_ in zip(self.enc_attn, self.enc_ffn):
                src = ffn_(attn(src, src + pos_all, src + pos_all, src))
            outs, start = [], 0
            for h, w in shapes:
                outs.append(src[:, start:start + h * w].transpose(1, 2).reshape(B, -1, h, w)); start += h * w
            fine = F.interpolate(outs[2], size=pyramid[0].shape[-2:], mode='bilinear', align_corners=False)
            mask_features = self.mask_features(self.output_conv(self.lateral(pyramid[0]) + fine))
            memory = [o.flatten(2).transpose(1, 2) + self.dec_level_embed.weight[i] for i, o in enumerate(outs)]
            mem_pos = [sine_pos(h, w, d, feats.device)[None] for h, w in shapes]
            qpos = self.query_pos.weight[None].expand(B, -1, -1)
            x = self.query_feat.weight[None].expand(B, -1, -1)
            logits, masks, attn = self.heads(x, mask_features, shapes[0])
            out_logits, out_masks = [logits], [masks]
            for i, (cross, self_, ffn_) in enumerate(zip(self.cross, self.self_attn, self.dec_ffn)):
                lv = i % 3
                x = cross(x, x + qpos, memory[lv] + mem_pos[lv], memory[lv], attn)
                x = self_(x, x + qpos, x + qpos, x)
                x = ffn_(x)
                logits, masks, attn = self.heads(x, mask_features, shapes[(i + 1) % 3])
                out_logits.append(logits); out_masks.append(masks)
            return dict(logits=out_logits, masks=out_masks)
    return Mask2FormerLite()


def semantic(logits, masks, size=SIDE):
    """Mask2Former semantic inference: (B, 2, size, size) class scores (sums over queries; argmax = label)."""
    import torch
    import torch.nn.functional as F
    m = F.interpolate(masks, size=(size, size), mode='bilinear', align_corners=False).sigmoid()
    return torch.einsum('bqc,bqhw->bchw', logits.softmax(-1)[..., :-1], m)


def criterion(out, lab, weights=(2.0, 5.0, 5.0), eos=0.1):
    """Hungarian-matched class CE + mask BCE + Dice on every prediction layer.
    lab: (B, h, w) long at mask resolution, 255 = ignore (excluded from mask terms)."""
    import torch
    import torch.nn.functional as F
    from scipy.optimize import linear_sum_assignment
    B = lab.shape[0]
    lab = lab.flatten(1)
    valid = (lab != IGNORE).float()
    tgt = torch.stack([(lab == 0), (lab == 1)], 1).float() * valid[:, None]
    present = tgt.sum(-1) > 0
    nvalid = valid.sum(-1).clamp(min=1)
    class_weight = torch.tensor([1.0, 1.0, eos], device=lab.device)
    total = 0.0
    for logits, masks in zip(out['logits'], out['masks']):
        x = masks.flatten(2)
        with torch.no_grad():
            pos, neg = F.softplus(-x) * valid[:, None], F.softplus(x) * valid[:, None]
            c_bce = (torch.einsum('bqn,bcn->bqc', pos, tgt) + torch.einsum('bqn,bcn->bqc', neg, valid[:, None] - tgt)) \
                / nvalid[:, None, None]
            s = x.sigmoid() * valid[:, None]
            c_dice = 1 - (2 * torch.einsum('bqn,bcn->bqc', s, tgt) + 1) / (s.sum(-1)[..., None] + tgt.sum(-1)[:, None] + 1)
            cost = (weights[0] * -logits.softmax(-1)[..., :2] + weights[1] * c_bce + weights[2] * c_dice).cpu()
        target_class = torch.full(logits.shape[:2], 2, dtype=torch.long, device=lab.device)
        bq, bc = [], []
        for b in range(B):
            cols = present[b].nonzero().flatten().tolist()
            if not cols:
                continue
            r, c = linear_sum_assignment(cost[b][:, cols].numpy())
            for q, j in zip(r.tolist(), c.tolist()):
                target_class[b, q] = cols[j]; bq.append((b, q)); bc.append((b, cols[j]))
        loss_ce = F.cross_entropy(logits.transpose(1, 2), target_class, weight=class_weight)
        if bq:
            bi = torch.tensor([b for b, _ in bq], device=lab.device)
            xm = x[bi, torch.tensor([q for _, q in bq], device=lab.device)]
            tm = tgt[bi, torch.tensor([c for _, c in bc], device=lab.device)]
            vm = valid[bi]
            bce = (F.binary_cross_entropy_with_logits(xm, tm, reduction='none') * vm).sum(-1) / vm.sum(-1).clamp(min=1)
            sm = xm.sigmoid() * vm
            dice = 1 - (2 * (sm * tm).sum(-1) + 1) / (sm.sum(-1) + tm.sum(-1) + 1)
            loss_bce, loss_dice = bce.sum() / len(bq), dice.sum() / len(bq)
        else:
            loss_bce = loss_dice = x.sum() * 0
        total = total + weights[0] * loss_ce + weights[1] * loss_bce + weights[2] * loss_dice
    return total


def downsample_labels(lab512):
    """512 labels -> mask resolution (128): the pixel nearest each 4x4 cell centre."""
    return lab512[:, 2::4, 2::4]


# --------------------------------------------------------------------------- prepare / cache

def prepare():
    if (OUT/'items.json').exists():
        raise SystemExit('items.json exists: the item list is frozen; use a new namespace to change it')
    cfg = design()
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ITEMS_SRC, OUT/'items.json')
    items = json.loads((OUT/'items.json').read_text())
    excluded = {'GR2', 'UR2', 'UR3'}
    bad = [it['id'] for it in items if it['split'] not in ('train', 'dev') or str(it['rig']).startswith('sealed')
           or it['group'] in excluded or 'water' in json.dumps(it).lower()]
    if bad:
        raise SystemExit(f'excluded items present: {bad[:5]}')
    missing = [p for it in items for p in (it['image'], it['label']) if not (ROOT/p).exists()]
    if missing:
        raise SystemExit(f'missing inputs: {missing[:5]}')
    counts = collections.Counter(f"{it['split']}_{it['domain']}_{'bottom' if it['bottom'] else 'other'}" for it in items)
    expected = cfg['inputs']['historical_counts']
    if dict(counts) != expected:
        raise SystemExit(f'counts {dict(counts)} != design {expected}')
    geo = [geometry(*it['native_hw']) for it in items]
    rng = random.Random(20261024)
    probes = []
    for s in STRATA:
        pool = [i for i, it in enumerate(items) if it['split'] == 'train' and stratum(it) == s]
        probes += sorted(rng.sample(pool, 64))
    write_new(OUT/'geometry.json', geo)
    label_kinds = collections.Counter((it['domain'], Path(it['label']).suffix, it['cohort']) for it in items)
    write_new(OUT/'input_receipt.json', dict(
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        items=dict(source=str(ITEMS_SRC), sha256=sha(OUT/'items.json'), counts=dict(counts), expected=expected),
        release=dict(path=str(RELEASE.relative_to(ROOT)), sha256=RELEASE_SHA),
        label_policies={'real (.png)': 'confirmed visible-mouse manual mask, >127 = mouse; boundary may include hair',
                        'synthetic replay (.png)': 'rendered visible-body mask, >127 = mouse',
                        'synthetic v3 crossed (.npz)': 'truth body: primary-ray skin surface, glass/hair excluded '
                                                       '(surrogate boundary); depth validity is not a label',
                        'padding': 'letterbox padding is supervised background (0)', 'ignore': 255},
        label_kinds={'/'.join(map(str, k)): v for k, v in label_kinds.items()},
        not_run=dict(SEG_DA='user decision 2026-10-24: use VGGT-Omega directly, no Depth Anything weights',
                     SEG_R='historical S1 checkpoint (output/vggt_mouse_segmentation_16h_20261020/S1/latest.pt) '
                           'is not on the cluster',
                     pano_proxy='PanoMouse SAM2/visual-hull proxy masks are not on the cluster'),
        exclusions=cfg['inputs']['exclusions'], tool_sha256=sha(Path(__file__)), design_sha256=sha(DESIGN)))
    write_new(OUT/'protocol.json', dict(
        design=cfg, probes=probes, probe_rule='64 TRAIN items per stratum, Random(20261024)',
        letterbox='aspect-preserving resize to long side 512 (INTER_LINEAR image, INTER_NEAREST labels), centred, '
                  'zero padding; ROI cropped before inverse resize',
        features='get_intermediate_layers(x, n=[20,21,22,23], norm=True, reshape=True) under bf16 autocast, fp16 cache',
        arms=ARMS, sampler='per update 2 items from each of 4 strata; inside a stratum round-robin over shuffled '
                           'balance groups (cohort|rig or session), each group a shuffled queue; Random(seed)',
        decoder_deviations=[
            'pixel decoder uses standard multi-head attention over the 1/32,1/16,1/8 levels (5,376 tokens) instead of '
            'multi-scale deformable attention',
            'mask loss and matching cost computed densely on all valid 128x128 pixels instead of 12,544 '
            'importance-sampled points',
            'Feature2Pyramid applied after the 1024->256 projection, with GroupNorm(32) instead of BatchNorm',
            'decoder FFN width 1024; no-object class weight 0.1; deep supervision on all 4 predictions'],
        plateau_metrics='per arm on TRAIN probes with EMA weights: full criterion loss, mean foreground IoU and mean '
                        'boundary F1 (long side 512, 2 px) inside the ROI',
        common_endpoint='first TRAIN check >= 6,000 where both arms plateau; otherwise 18,000. Training continues to '
                        '18,000 for the fixed DEV evaluations at 6,000/12,000/18,000',
        boundary_f1='boundary = mask minus 3x3 erosion; precision/recall = fraction of boundary pixels within tol px '
                    '(Euclidean distance transform); both empty = 1, one empty = 0; ignore pixels excluded'))
    print('prepared', dict(counts))


def cache(shard, shards):
    import torch
    import bottom_geometry_v4_runtime_20261019 as rt
    done = OUT/f'cache/shard{shard}_of{shards}.json'
    if done.exists():
        print('shard exists'); return
    items = json.loads((OUT/'items.json').read_text()); geo = json.loads((OUT/'geometry.json').read_text())
    ids = list(range(shard, len(items), shards))
    (OUT/'cache').mkdir(exist_ok=True)
    stem = OUT/f'cache/shard{shard}_of{shards}'
    feats = np.lib.format.open_memmap(f'{stem}.feats.f16.npy.partial', 'w+', np.float16, (len(ids), 4, 1024, 32, 32))
    labels = np.lib.format.open_memmap(f'{stem}.labels.u8.npy.partial', 'w+', np.uint8, (len(ids), SIDE, SIDE))

    class Data(torch.utils.data.Dataset):
        def __len__(self):
            return len(ids)

        def __getitem__(self, j):
            it, g = items[ids[j]], geo[ids[j]]
            return (letterbox(read_image(it), g, cv2.INTER_LINEAR),
                    letterbox(read_label(it), g, cv2.INTER_NEAREST))

    model = rt.model(RELEASE, trainable=False).eval(); a = model.aggregator
    mean, std = a._resnet_mean.view(1, 3, 1, 1), a._resnet_std.view(1, 3, 1, 1)
    loader = torch.utils.data.DataLoader(Data(), batch_size=8, num_workers=8, prefetch_factor=4)
    t0, j = time.time(), 0
    with torch.inference_mode():
        for img, lab in loader:
            x = img.cuda().permute(0, 3, 1, 2).float() / 255
            with torch.autocast('cuda', dtype=torch.bfloat16):
                outs = a.patch_embed.get_intermediate_layers((x - mean) / std, n=LAYERS, norm=True, reshape=True)
            feats[j:j + len(img)] = torch.stack(outs, 1).half().cpu().numpy()
            labels[j:j + len(img)] = lab.numpy()
            j += len(img)
            if j % 800 < 8:
                print('CACHE', shard, j, len(ids), round(time.time() - t0), flush=True)
    feats.flush(); labels.flush(); del feats, labels
    for kind in ('feats.f16', 'labels.u8'):
        os.replace(f'{stem}.{kind}.npy.partial', f'{stem}.{kind}.npy')
    write_new(done, dict(shard=shard, shards=shards, ids=ids, items_sha256=sha(OUT/'items.json'), release_sha256=RELEASE_SHA,
                         layers=LAYERS, feats=dict(shape=[len(ids), 4, 1024, 32, 32], dtype='float16'),
                         labels=dict(shape=[len(ids), SIDE, SIDE], dtype='uint8'), tool_sha256=sha(Path(__file__)),
                         seconds=round(time.time() - t0)))
    print('CACHE done', shard, len(ids), round(time.time() - t0))


class Bank:
    """All cached features/labels in item order; on GPU when they fit, else pinned host memory."""

    def __init__(self, device, reserve_gb=24):
        import torch
        self.items = json.loads((OUT/'items.json').read_text())
        self.geo = json.loads((OUT/'geometry.json').read_text())
        receipts = sorted((OUT/'cache').glob('shard*_of*.json'))
        shards = {json.loads(r.read_text())['shards'] for r in receipts}
        if len(shards) != 1 or len(receipts) != shards.pop():
            raise SystemExit('feature cache incomplete')
        N = len(self.items)
        size = N * (4 * 1024 * 32 * 32 * 2 + SIDE * SIDE)
        on_gpu = device != 'cpu' and torch.cuda.mem_get_info()[0] - size > reserve_gb * 2**30
        self.placement = 'gpu' if on_gpu else 'cpu'
        where = device if on_gpu else 'cpu'
        self.feats = torch.empty((N, 4, 1024, 32, 32), dtype=torch.float16, device=where)
        self.labels = torch.empty((N, SIDE, SIDE), dtype=torch.uint8, device=where)
        for r in receipts:
            meta = json.loads(r.read_text()); stem = str(r)[:-5]
            f = np.load(f'{stem}.feats.f16.npy', mmap_mode='r'); lab = np.load(f'{stem}.labels.u8.npy', mmap_mode='r')
            ids = meta['ids']
            for j in range(0, len(ids), 128):
                idx = torch.tensor(ids[j:j + 128])
                self.feats[idx.to(where)] = torch.from_numpy(np.array(f[j:j + 128])).to(where)
                self.labels[idx.to(where)] = torch.from_numpy(np.array(lab[j:j + 128])).to(where)
        self.device = device

    def batch(self, ids):
        import torch
        idx = torch.tensor(ids, device=self.feats.device)
        return self.feats[idx].to(self.device, non_blocking=True).float(), \
            self.labels[idx].to(self.device, non_blocking=True).long()


# --------------------------------------------------------------------------- training

class StratSampler:
    def __init__(self, items, ids, seed, per_stratum=2):
        self.rng, self.per = random.Random(seed), per_stratum
        self.groups = {s: collections.defaultdict(list) for s in STRATA}
        for i in ids:
            self.groups[stratum(items[i])][balance_key(items[i])].append(i)
        self.cycle = {s: [] for s in STRATA}
        self.queue = {s: {g: [] for g in self.groups[s]} for s in STRATA}

    def take(self):
        out = []
        for s in STRATA:
            for _ in range(self.per):
                if not self.cycle[s]:
                    keys = sorted(self.groups[s]); self.rng.shuffle(keys); self.cycle[s] = keys
                g = self.cycle[s].pop()
                if not self.queue[s][g]:
                    q = list(self.groups[s][g]); self.rng.shuffle(q); self.queue[s][g] = q
                out.append(self.queue[s][g].pop())
        return out

    def state(self):
        return dict(rng=self.rng.getstate(), cycle={k: list(v) for k, v in self.cycle.items()},
                    queue={s: {g: list(q) for g, q in d.items()} for s, d in self.queue.items()})

    def restore(self, st):
        self.rng.setstate(st['rng']); self.cycle = st['cycle']; self.queue = st['queue']


def plateau(history, conv, min_updates):
    if len(history) < conv['checks'] or history[-1]['update'] < min_updates:
        return False
    w = history[-conv['checks']:]
    if w[-1]['update'] - w[0]['update'] != (conv['checks'] - 1) * CHECK_EVERY:
        return False
    loss = [r['loss'] for r in w]
    reduction = max(0.0, loss[0] - min(loss)) / max(loss[0], 1e-12)
    span = lambda k: max(r[k] for r in w) - min(r[k] for r in w)
    return (reduction < conv['relative_loss_improvement_lt'] and span('iou') <= conv['iou_span_pp_le'] / 100 + 1e-12
            and span('bf1') <= conv['boundary_f1_span_pp_le'] / 100 + 1e-12)


class Arm:
    def __init__(self, name, seed, cfg, device, decoder_kwargs=None):
        import torch
        torch.manual_seed(seed)  # identical decoder initialisation in both arms
        self.name, self.layers = name, ARMS[name]
        self.net = build_decoder(**(decoder_kwargs or {})).to(device)
        pair = cfg['frozen_head_pair']
        decay, no_decay = [], []
        for mname, mod in self.net.named_modules():
            for pname, p in mod.named_parameters(recurse=False):
                (no_decay if pname.endswith('bias') or 'Norm' in type(mod).__name__ else decay).append(p)
        self.opt = torch.optim.AdamW([dict(params=decay, weight_decay=pair['weight_decay']),
                                      dict(params=no_decay, weight_decay=0.0)], lr=0.0)
        self.ema = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
        self.history, self.plateau_at = [], None
        self.pair = pair

    def lr(self, u):
        p = self.pair
        t = min(u - 1, p['cosine_total_updates']) / p['cosine_total_updates']
        env = p['cosine_floor'] + (1 - p['cosine_floor']) * .5 * (1 + math.cos(math.pi * t))
        return p['peak_head_lr'] * min(1.0, u / p['warmup_updates']) * env

    def step(self, feats, lab128, u):
        import torch
        for g in self.opt.param_groups:
            g['lr'] = self.lr(u)
        self.net.train()
        loss = criterion(self.net(feats[:, self.layers]), lab128)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'{self.name}: non-finite loss at update {u}')
        self.opt.zero_grad(set_to_none=True); loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.pair['grad_clip']))
        self.opt.step()
        with torch.no_grad():
            d = self.pair['ema_decay']
            for k, v in self.net.state_dict().items():
                self.ema[k].mul_(d).add_(v.detach(), alpha=1 - d) if v.dtype.is_floating_point else self.ema[k].copy_(v)
        return float(loss.detach()), gn

    def weights(self, kind):
        return self.ema if kind == 'ema' else {k: v.detach().clone() for k, v in self.net.state_dict().items()}

    def state_dict(self):
        return dict(net={k: v.cpu() for k, v in self.net.state_dict().items()}, ema={k: v.cpu() for k, v in self.ema.items()},
                    opt=self.opt.state_dict(), history=self.history, plateau_at=self.plateau_at)

    def load_state_dict(self, s):
        self.net.load_state_dict(s['net']); self.opt.load_state_dict(s['opt'])
        for k, v in s['ema'].items():
            self.ema[k].copy_(v)
        self.history, self.plateau_at = s['history'], s['plateau_at']


def predict(net, state, bank, ids, layers, batch=16):
    """Class probabilities at 512 (B, 2, 512, 512) on CPU, plus the mean criterion loss."""
    import torch
    live = {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state); net.eval()
    probs, losses = [], []
    try:
        with torch.no_grad():
            for j in range(0, len(ids), batch):
                f, lab = bank.batch(ids[j:j + batch])
                out = net(f[:, layers])
                losses.append(float(criterion(out, downsample_labels(lab))) * len(f))
                probs.append(semantic(out['logits'][-1], out['masks'][-1]).cpu())
    finally:
        net.load_state_dict(live)
    return torch.cat(probs).numpy(), sum(losses) / len(ids)


def probe_metrics(bank, ids, probs, loss):
    labels = bank.labels[ids].cpu().numpy()
    iou, bf = [], []
    for n, i in enumerate(ids):
        g = bank.geo[i]
        pred = roi(probs[n].argmax(0) == 1, g); lab = roi(labels[n], g)
        valid = lab != IGNORE; gt = lab == 1
        iou.append(overlap(pred, gt, valid)[0]); bf.append(boundary_f1(pred, gt, valid, 2))
    return dict(loss=loss, iou=float(np.mean(iou)), bf1=float(np.mean(bf)))


def dev_metrics(bank, ids, probs, save_dir=None):
    rows = []
    for n, i in enumerate(ids):
        it, g = bank.items[i], bank.geo[i]
        h, w = it['native_hw']
        native_lab = read_label(it)
        native_pred = to_native(probs[n], g, h, w).argmax(0) == 1
        nvalid = native_lab != IGNORE
        lab512 = letterbox(native_lab, g, cv2.INTER_NEAREST)
        pred512 = probs[n].argmax(0) == 1
        rpred, rlab = roi(pred512, g), roi(lab512, g)
        rvalid = rlab != IGNORE
        iou, dice = overlap(native_pred, native_lab == 1, nvalid)
        pad = np.ones((SIDE, SIDE), bool); pad[g['top']:g['top'] + g['size'][0], g['left']:g['left'] + g['size'][1]] = False
        rows.append(dict(item=i, id=it['id'], stratum=stratum(it), domain=it['domain'], bottom=it['bottom'],
                         session=it['group'] if it['domain'] == 'real' else None, rig=it['rig'],
                         iou=iou, dice=dice, bf1_native_1px=boundary_f1(native_pred, native_lab == 1, nvalid, 1),
                         bf1_ref512_1px=boundary_f1(rpred, rlab == 1, rvalid, 1),
                         bf1_ref512_2px=boundary_f1(rpred, rlab == 1, rvalid, 2),
                         empty=not native_pred.any(), all_foreground=bool(native_pred[nvalid].all()) if nvalid.any() else False,
                         padding_foreground_px=int((pred512 & pad).sum()), gt_foreground_px=int((native_lab == 1).sum())))
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_dir/f"{it['id']}_mask.png"), native_pred.astype(np.uint8) * 255)
            frac = probs[n][1] / np.maximum(probs[n].sum(0), 1e-12)  # mouse share of the two class scores
            cv2.imwrite(str(save_dir/f"{it['id']}_prob512.png"), np.clip(frac * 255, 0, 255).astype(np.uint8))
    return rows


def summarize(rows):
    keys = ['iou', 'dice', 'bf1_native_1px', 'bf1_ref512_1px', 'bf1_ref512_2px']
    out = {}
    groups = {s: [r for r in rows if r['stratum'] == s] for s in STRATA}
    groups.update({f"session/{s}": [r for r in rows if r['session'] == s] for s in sorted({r['session'] for r in rows if r['session']})})
    groups.update({f"rig/{s}": [r for r in rows if r['rig'] == s] for s in sorted({r['rig'] for r in rows if r['rig']})})
    for name, rr in groups.items():
        if rr:
            out[name] = dict(n=len(rr), **{k: mean([r[k] for r in rr]) for k in keys},
                             empty_fraction=mean([r['empty'] for r in rr]), all_foreground_fraction=mean([r['all_foreground'] for r in rr]),
                             padding_foreground_px_mean=mean([r['padding_foreground_px'] for r in rr]))
    return out


def train(seed):
    import torch
    cfg = design(); pair = cfg['frozen_head_pair']
    run = RUNS/f'seed{seed}'
    if (run/'DONE.json').exists():
        print('done already:', run); return
    run.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    protocol = json.loads((OUT/'protocol.json').read_text())
    bank = Bank('cuda')
    items = bank.items
    train_ids = [i for i, it in enumerate(items) if it['split'] == 'train']
    dev_ids = [i for i, it in enumerate(items) if it['split'] == 'dev']
    probes = protocol['probes']
    if SMOKE:  # 8 probes and 16 DEV items per stratum
        probes = [i for s in STRATA for i in [p for p in probes if stratum(items[p]) == s][:8]]
        dev_ids = [i for s in STRATA for i in [d for d in dev_ids if stratum(items[d]) == s][:16]]
    arms = {name: Arm(name, seed, cfg, 'cuda') for name in ARMS}
    sampler = StratSampler(items, train_ids, seed)
    u, common, log = 0, None, []
    if (run/'state.pt').exists():
        s = torch.load(run/'state.pt', weights_only=False)
        for name, arm in arms.items():
            arm.load_state_dict(s['arms'][name])
        sampler.restore(s['sampler']); u, common, log = s['u'], s['common'], s['log']
        random.setstate(s['rng']['python']); np.random.set_state(s['rng']['numpy']); torch.set_rng_state(s['rng']['torch'])
        torch.cuda.set_rng_state_all(s['rng']['cuda'])
        print('resumed at update', u, flush=True)
    else:
        write(run/'run_protocol.json', dict(seed=seed, start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                            host=os.uname().nodename, slurm_job=os.environ.get('SLURM_JOB_ID'),
                                            slurm_step=os.environ.get('SLURM_STEP_ID'), data_placement=bank.placement,
                                            tool_sha256=sha(Path(__file__)), items_sha256=sha(OUT/'items.json'),
                                            decoder_params=sum(p.numel() for p in arms['SEG_M'].net.parameters())))
    total = pair['max_updates']
    dev_points = set(pair['dev_eval_updates'])

    def save():
        torch.save(dict(u=u, common=common, log=log, sampler=sampler.state(), arms={n: a.state_dict() for n, a in arms.items()},
                        rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                                 cuda=torch.cuda.get_rng_state_all())), run/'state.pt.tmp')
        os.replace(run/'state.pt.tmp', run/'state.pt')

    def evaluate(tag, save_masks=False):
        """Snapshot first, then score from the snapshot, so an interrupted evaluation can be completed later."""
        for name, arm in arms.items():
            for kind in ('ema', 'raw'):
                snap = run/f'snapshot_{tag}_{name}_{kind}.pt'
                if not snap.exists():
                    torch.save(dict(update=u, weights=kind, state={k: v.cpu() for k, v in arm.weights(kind).items()}), snap)
        for name, arm in arms.items():
            for kind in ('ema', 'raw'):
                if (run/f'dev_{tag}_{name}_{kind}.json').exists():
                    continue
                snap = torch.load(run/f'snapshot_{tag}_{name}_{kind}.pt', weights_only=False)
                state = {k: v.to('cuda') for k, v in snap['state'].items()}
                probs, loss = predict(arm.net, state, bank, dev_ids, arm.layers)
                rows = dev_metrics(bank, dev_ids, probs, run/f'masks_{tag}_{name}_{kind}' if save_masks else None)
                write(run/f'dev_{tag}_{name}_{kind}.json', dict(update=snap['update'], arm=name, weights=kind, loss=loss,
                                                                 summary=summarize(rows), rows=rows))
        print(f'seed{seed} DEV evaluated at {tag}', flush=True)

    t0 = time.time()
    while u < total:
        u += 1
        ids = sampler.take()
        feats, lab = bank.batch(ids)
        lab128 = downsample_labels(lab)
        row = dict(update=u)
        for name, arm in arms.items():
            row[f'{name}_loss'], row[f'{name}_grad_norm'] = arm.step(feats, lab128, u)
        if u % 50 == 0 or u == 1:
            log.append(dict(row, lr=arms['SEG_M'].lr(u)))
        if u % CHECK_EVERY == 0:
            for name, arm in arms.items():
                probs, loss = predict(arm.net, arm.ema, bank, probes, arm.layers)
                arm.history.append(dict(update=u, **probe_metrics(bank, probes, probs, loss)))
                if arm.plateau_at is None and plateau(arm.history, pair['plateau'], pair['min_updates']):
                    arm.plateau_at = u
            if common is None and all(plateau(a.history, pair['plateau'], pair['min_updates']) for a in arms.values()):
                common = dict(update=u, converged=True)
            print(f'seed{seed} u{u} ' + ' '.join(f"{n} loss {a.history[-1]['loss']:.4f} IoU {100*a.history[-1]['iou']:.2f} "
                                                 f"BF1 {100*a.history[-1]['bf1']:.2f}" for n, a in arms.items())
                  + f' {round(time.time()-t0)}s', flush=True)
            if common and common['update'] == u:
                evaluate('common', save_masks=True)
        if u in dev_points:
            evaluate(f'u{u:05d}')
        if u % CHECK_EVERY == 0 or u in dev_points:  # after endpoint work, so a resume never skips it
            save()
    if common is not None:  # complete an evaluation interrupted after its snapshot was written
        evaluate('common', save_masks=True)
    if common is None:
        common = dict(update=total, converged=False,
                      arms_not_converged=[n for n, a in arms.items() if a.plateau_at is None])
        evaluate('common', save_masks=True)
    write(run/'train_log.json', dict(log=log, history={n: a.history for n, a in arms.items()}, common=common))
    write_new(run/'DONE.json', dict(seed=seed, updates=u, common_endpoint=common,
                                    plateau_at={n: a.plateau_at for n, a in arms.items()}, seconds=round(time.time() - t0),
                                    finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    print('DONE seed', seed, common, flush=True)


# --------------------------------------------------------------------------- preflight

def preflight():
    import torch
    import bottom_geometry_v4_runtime_20261019 as rt
    dest = RUNS/'preflight'
    if (dest/'preflight.json').exists():
        if json.loads((dest/'preflight.json').read_text())['passed']:
            print('preflight passed earlier'); return
        raise SystemExit('preflight failed earlier: inspect preflight/preflight.json before removing it')
    dest.mkdir(parents=True, exist_ok=True)
    cfg = design()
    bank = Bank('cuda')
    items, geo = bank.items, bank.geo
    rng = random.Random(7)
    sample = sorted(rng.sample([i for i, it in enumerate(items) if it['split'] == 'train'], 8))
    model = rt.model(RELEASE, trainable=False).eval(); a = model.aggregator
    mean, std = a._resnet_mean.view(1, 3, 1, 1), a._resnet_std.view(1, 3, 1, 1)
    x = torch.stack([torch.from_numpy(letterbox(read_image(items[i]), geo[i], cv2.INTER_LINEAR)) for i in sample]).cuda()
    x = (x.permute(0, 3, 1, 2).float() / 255 - mean) / std
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        outs = a.patch_embed.get_intermediate_layers(x, n=LAYERS, norm=True, reshape=True)
        final = a.patch_embed(x)['x_norm_patchtokens'].transpose(1, 2).reshape(len(sample), 1024, 32, 32)
    report = dict(shapes=[list(o.shape) for o in outs],
                  last_layer_vs_x_norm_patchtokens_max_abs=float((outs[-1].float() - final.float()).abs().max()),
                  online_vs_cache_max_abs=float((torch.stack(outs, 1).half().float() - bank.batch(sample)[0]).abs().max()))
    f = bank.batch(sample)[0]
    report['seg_l_replication_exact'] = bool(all(torch.equal(f[:, ARMS['SEG_L']][:, k], f[:, 3]) for k in range(4)))
    trips = []
    for i in sorted(rng.sample(range(len(items)), 16)):
        lab = read_label(items[i]); g = geo[i]; h, w = lab.shape
        one = np.stack([(lab == 0), (lab == 1)]).astype(np.float32)
        back = to_native(letterbox(one.transpose(1, 2, 0), g, cv2.INTER_NEAREST).transpose(2, 0, 1), g, h, w).argmax(0) == 1
        # The same resolution loss without letterbox padding/offset: any difference is a geometry error.
        small = cv2.resize(one.transpose(1, 2, 0), (g['size'][1], g['size'][0]), interpolation=cv2.INTER_NEAREST)
        plain = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1).argmax(0) == 1
        lb = letterbox(lab, g, cv2.INTER_NEAREST)
        trips.append(dict(item=i, iou=overlap(back, lab == 1, lab != IGNORE)[0],
                          vs_plain_resize_iou=overlap(back, plain, np.ones_like(back))[0],
                          padding_is_background=bool((lb[:g['top']] == 0).all() and (lb[g['top'] + g['size'][0]:] == 0).all()
                                                     and (lb[:, :g['left']] == 0).all() and (lb[:, g['left'] + g['size'][1]:] == 0).all())))
    report['letterbox_round_trip_min_iou_resolution_loss'] = min(t['iou'] for t in trips)  # informational
    report['letterbox_vs_plain_resize_min_iou'] = min(t['vs_plain_resize_iou'] for t in trips)
    report['padding_supervised_background'] = all(t['padding_is_background'] for t in trips)
    # Short training on disposable arms, then resume equality; decoders only (encoder never receives gradients).
    train_ids = [i for i, it in enumerate(items) if it['split'] == 'train']

    def arms_from_scratch():
        return {n: Arm(n, 17, cfg, 'cuda') for n in ARMS}, StratSampler(items, train_ids, 17)
    arms, sampler = arms_from_scratch()
    for u in (1, 2):
        feats, lab = bank.batch(sampler.take())
        for arm in arms.values():
            arm.step(feats, downsample_labels(lab), u)
    snap = dict(arms={n: a.state_dict() for n, a in arms.items()}, sampler=sampler.state())
    torch.save(snap, dest/'resume.pt')
    for u in (3, 4):
        feats, lab = bank.batch(sampler.take())
        losses = [arm.step(feats, downsample_labels(lab), u)[0] for arm in arms.values()]
    ref = {n: {k: v.clone() for k, v in a.net.state_dict().items()} for n, a in arms.items()}
    arms2, sampler2 = arms_from_scratch()
    s = torch.load(dest/'resume.pt', weights_only=False)
    for n, a in arms2.items():
        a.load_state_dict(s['arms'][n])
    sampler2.restore(s['sampler'])
    for u in (3, 4):
        feats, lab = bank.batch(sampler2.take())
        for arm in arms2.values():
            arm.step(feats, downsample_labels(lab), u)
    report['resume_max_abs_param_diff'] = max(float((v - ref[n][k]).abs().max()) for n, a in arms2.items()
                                              for k, v in a.net.state_dict().items() if v.dtype.is_floating_point)
    report['finite_losses'] = all(math.isfinite(x) for x in losses)
    report['encoder_grads'] = sum(p.grad is not None for p in model.parameters())
    report['passed'] = bool(report['last_layer_vs_x_norm_patchtokens_max_abs'] < 1e-3 and report['seg_l_replication_exact']
                            and report['letterbox_vs_plain_resize_min_iou'] >= 0.999 and report['padding_supervised_background']
                            and report['resume_max_abs_param_diff'] < 1e-4 and report['finite_losses']
                            and report['encoder_grads'] == 0 and report['online_vs_cache_max_abs'] < 5e-2)
    report['updates_discarded'] = True
    (dest/'resume.pt').unlink()
    write_new(dest/'preflight.json', report)
    print('PREFLIGHT', json.dumps(report))
    if not report['passed']:
        raise SystemExit('preflight failed')


# --------------------------------------------------------------------------- gates

def gate(seed):
    cfg = design()['screening']
    run = RUNS/f'seed{seed}'
    L = json.loads((run/'dev_common_SEG_L_ema.json').read_text())['summary']
    M = json.loads((run/'dev_common_SEG_M_ema.json').read_text())['summary']
    strata, degraded = {}, False
    for s in STRATA:
        d_iou = 100 * (M[s]['iou'] - L[s]['iou']); d_bf = 100 * (M[s]['bf1_ref512_2px'] - L[s]['bf1_ref512_2px'])
        bad = d_iou < -cfg['each_stratum_mean_iou_drop_pp_le'] - 1e-9 or d_bf < -cfg['each_stratum_mean_boundary_f1_drop_pp_le'] - 1e-9 \
            or M[s]['empty_fraction'] > cfg['empty_fraction_le'] + 1e-9
        degraded |= bad
        strata[s] = dict(L=L[s], M=M[s], delta_iou_pp=d_iou, delta_bf1_ref512_2px_pp=d_bf, degraded=bad)
    gain = strata['real/bottom']['delta_bf1_ref512_2px_pp']
    signal = gain >= cfg['benefit_real_bottom_mean_boundary_f1_ref512_tol2_gain_pp_ge']
    result = dict(seed=seed, real_bottom_bf1_gain_pp=gain, benefit_signal=signal, degraded=degraded,
                  expand_to_more_seeds=signal and not degraded, strata=strata,
                  rule='seeds 29/43 only if seed 17 shows the registered signal without degradation')
    write(RUNS/f'gate_seed{seed}.json', result)
    print('GATE', json.dumps({k: v for k, v in result.items() if k != 'strata'}))
    sys.exit(0 if result['expand_to_more_seeds'] else 3)


def summary():
    seeds = [s for s in design()['frozen_head_pair']['seeds'] if (RUNS/f'gate_seed{s}.json').exists()]
    gates = {s: json.loads((RUNS/f'gate_seed{s}.json').read_text()) for s in seeds}
    gains = [g['real_bottom_bf1_gain_pp'] for g in gates.values()]
    report = dict(seeds=seeds, gates=gates, mean_real_bottom_bf1_gain_pp=float(np.mean(gains)) if gains else None,
                  positive=sum(x > 0 for x in gains),
                  candidate_beneficial=len(seeds) == 3 and float(np.mean(gains)) >= 2 and sum(x > 0 for x in gains) >= 2
                  and not any(g['degraded'] for g in gates.values()),
                  note='Multi-layer features are a candidate only with three paired seeds, mean gain >= 2pp, >= 2/3 '
                       'positive and no seed degraded. No sealed scoring; PanoMouse proxy not run.')
    write(RUNS/'summary.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'gates'}, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['prepare', 'cache', 'preflight', 'train', 'gate', 'summary'])
    ap.add_argument('--seed', type=int)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    a = ap.parse_args()
    {'prepare': prepare, 'cache': lambda: cache(a.shard, a.shards), 'preflight': preflight, 'train': lambda: train(a.seed),
     'gate': lambda: gate(a.seed), 'summary': summary}[a.step]()
