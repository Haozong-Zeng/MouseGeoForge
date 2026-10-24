"""Data, scheduling and state utilities for the independent semantic branch."""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from mousegeoforge.models.mouse_semseg import MouseMaskDecoder

ROOT = Path(__file__).resolve().parents[1]
RELEASE_SHA = 'c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934'
STRATA = [('real', True), ('real', False), ('synthetic', True), ('synthetic', False)]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(16 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_items(path, root, verify=True):
    content = json.loads(Path(path).read_text())
    rows = content['items'] if isinstance(content, dict) else content
    if isinstance(content, dict) and content.get('schema') != 'mouse-semseg-v1':
        raise ValueError('Unknown data manifest schema')
    seen = set()
    for row in rows:
        if row['split'] not in ('train', 'dev'):
            raise ValueError('Only explicit train/dev items allowed, no sealed/test')
        tag = ' '.join(str(row.get(k, '')) for k in ('group', 'rig', 'image', 'label')).lower()
        compact = re.sub('[^a-z0-9]', '', tag)
        if any(x in compact for x in ('waterreaching', 'dannce2', 'danncemarkerlessmouse2', 'walkaround')):
            raise ValueError('Excluded source')
        if row.get('protected_test') or row.get('admitted_motion_split') == 'test':
            raise ValueError('Protected TEST')
        if row.get('group') in ('GR2', 'UR2', 'UR3', 'S04', 'S05') or (row.get('rig') or '').startswith('sealed'):
            raise ValueError('Sealed or excluded group')
        if 'panomouse' in tag:
            raise ValueError('Pano proxy masks are not segmentation supervision')
        if row['domain'] not in ('real', 'synthetic') or type(row['bottom']) is not bool:
            raise ValueError('Unknown source stratum')
        if row['domain'] == 'real':
            allowed = (['AZL3', 'AZR2', 'CR2', 'EL2', 'EL2L2', 'ER3L2'] if row['split'] == 'train'
                       else ['AZR2L2', 'FR2', 'FR3'])
            if row['group'] not in allowed:
                raise ValueError('Real session in wrong split')
        if row['id'] in seen:
            raise ValueError('Duplicate image ID')
        seen.add(row['id'])
        for key, pin in [('image', 'image_sha256'), ('label', 'label_sha256')]:
            source = (Path(root) / row[key]).resolve()
            if not source.is_relative_to(Path(root).resolve()):
                raise ValueError('Source path escapes workspace')
            if pin not in row:
                raise ValueError('Missing input pin: ' + pin)
            if verify and digest(source) != row[pin]:
                raise ValueError('Input hash mismatch: ' + row[key])
    return rows


def letterbox(image, label, side):
    h, w = image.shape[:2]
    if label.shape != (h, w):
        raise ValueError('Image and label shape mismatch')
    scale = side / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    y, x = (side - nh) // 2, (side - nw) // 2
    rgb = np.zeros((side, side, 3), np.uint8)
    target = np.zeros((side, side), np.int64)  # Known background in padding.
    rgb[y:y + nh, x:x + nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    target[y:y + nh, x:x + nw] = cv2.resize(label.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST)
    metadata = dict(native_hw=[h, w], roi=[y, x, nh, nw], resized_hw=[nh, nw],
                    pixel_center_affine=[[nw / w, 0, x + .5 * nw / w - .5],
                                         [0, nh / h, y + .5 * nh / h - .5]])
    return torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255, torch.from_numpy(target), metadata


def native_mask(prediction, metadata):
    y, x, h, w = metadata['roi']
    crop = prediction[y:y + h, x:x + w].astype(np.uint8)
    nh, nw = metadata['native_hw']
    return cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_NEAREST)


def read_sample(row, root, side):
    image = np.asarray(Image.open(Path(root) / row['image']).convert('RGB'))
    p = Path(root) / row['label']
    if p.suffix == '.npz':
        with np.load(p, allow_pickle=False) as truth:
            label = truth['body'].astype(np.uint8)
    else:
        # Reviewed binary masks; 255 convention means foreground in these files.
        raw = np.asarray(Image.open(p).convert('L'))
        label = raw if row.get('label_encoding') == 'class_index_0_1_255' else (raw > 0).astype(np.uint8)
    if not set(np.unique(label)) <= {0, 1, 255}:
        raise ValueError('Invalid pixel labels')
    return letterbox(image, label, side)


class BalancedSchedule:
    """Stateless global update schedule; resume has no prefetched hidden cursor."""
    def __init__(self, items, batch, seed):
        if batch % 4:
            raise ValueError('Global batch must divide into four equal strata')
        self.batch, self.seed, self.groups = batch, seed, {}
        for stratum in STRATA:
            groups = {}
            for i, row in enumerate(items):
                if row['split'] == 'train' and (row['domain'], row['bottom']) == stratum:
                    groups.setdefault((row.get('cohort', ''), row.get('rig') or row['group']), []).append(i)
            if not groups:
                raise ValueError('Missing TRAIN stratum ' + str(stratum))
            self.groups[stratum] = groups

    def at(self, step):
        rng = random.Random(f'{self.seed}:{step}:mouse-semseg-v1')
        selected = []
        for stratum in STRATA:
            groups = self.groups[stratum]
            cohorts = sorted({g[0] for g in groups})
            for _ in range(self.batch // 4):
                cohort = rng.choice(cohorts)
                group = rng.choice(sorted(g for g in groups if g[0] == cohort))
                selected.append(rng.choice(groups[group]))
        return selected


def make_encoder(checkpoint=None, fixture=False):
    sys.path.insert(0, str(ROOT / 'vggt-omega'))
    from vggt_omega.models.layers.vision_transformer import DinoVisionTransformer
    encoder = DinoVisionTransformer(img_size=224, patch_size=16, embed_dim=32 if fixture else 1024,
        depth=24, num_heads=4 if fixture else 16, ffn_ratio=4, qkv_bias=True,
        drop_path_rate=0, layerscale_init=1e-5, norm_layer='layernormbf16',
        n_storage_tokens=4, mask_k_bias=True, pos_embed_rope_normalize_coords='max')
    if fixture:
        encoder.init_weights()
    else:
        if checkpoint is None or digest(checkpoint) != RELEASE_SHA:
            raise ValueError('Exact released encoder checkpoint is required')
        weights = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)
        prefix = 'aggregator.patch_embed.'
        state = {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}
        encoder.load_state_dict(state, strict=True)
    return encoder


class MouseSemseg(nn.Module):
    def __init__(self, encoder, dim=256, queries=16, heads=8, pixel_layers=3, query_layers=3):
        super().__init__()
        self.encoder = encoder
        self.head = MouseMaskDecoder(encoder.embed_dim, dim, queries, heads, pixel_layers, query_layers)
        self.register_buffer('image_mean', torch.tensor([.485, .456, .406])[None, :, None, None], persistent=False)
        self.register_buffer('image_std', torch.tensor([.229, .224, .225])[None, :, None, None], persistent=False)

    def features(self, image, arm):
        normalized = (image - self.image_mean) / self.image_std
        active = any(p.requires_grad for p in self.encoder.parameters())
        with torch.set_grad_enabled(torch.is_grad_enabled() and active):
            layers = self.encoder.get_intermediate_layers(normalized, n=[20, 21, 22, 23], norm=True, reshape=True)
        return [layers[-1]] * 4 if arm == 'L' else layers

    def forward(self, image, arm='M'):
        return self.head(self.features(image, arm))


def stage_for_step(arm, step, phase_steps=2000):
    if arm != 'U':
        return []
    return [[23], [22, 23], [20, 21, 22, 23]][min(2, step // phase_steps)]


def configure_stage(model, arm, step, phase_steps=2000):
    active = stage_for_step(arm, step, phase_steps)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith('head.') or (
            bool(active) and (name.startswith('encoder.norm.') or any(
                name.startswith(f'encoder.blocks.{i}.') for i in active)))
    model.eval()  # Same forward mode in both branches; autograd remains active.
    return active


def optimizer_for(model, arm, head_lr, weight_decay=.01):
    groups = {}
    for name, p in model.named_parameters():
        if name.startswith('head.'):
            owner, peak = 'head', head_lr
        elif arm == 'U' and name.startswith('encoder.norm.'):
            owner, peak = 'norm', 1e-6
        else:
            match = re.match(r'encoder.blocks.(2[0-3])\.', name)
            if arm != 'U' or not match:
                continue
            index = int(match[1]); owner, peak = f'block{index}', 1e-6 * .65 ** (23 - index)
        decay = 0. if p.ndim <= 1 or name.endswith('.bias') else weight_decay
        key = (owner, decay)
        group = groups.setdefault(key, dict(params=[], names=[], name=f'{owner}/wd{decay}', peak_lr=peak,
                                            lr=peak, weight_decay=decay, owner=owner))
        group['params'].append(p); group['names'].append(name)
    return torch.optim.AdamW(list(groups.values()), foreach=False)


def set_learning_rates(optimizer, step, total, warmup, floor, phase_steps=2000):
    envelope = floor + (1 - floor) * .5 * (1 + math.cos(math.pi * min(step, total) / total))
    for group in optimizer.param_groups:
        owner = group['owner']
        start = 0 if owner in ('head', 'norm', 'block23') else phase_steps if owner == 'block22' else 2 * phase_steps
        length = warmup if owner == 'head' else 200
        ramp = min(1., max(0., (step - start + 1) / max(1, length)))
        group['lr'] = group['peak_lr'] * envelope * ramp


class EMA:
    def __init__(self, decay=.995):
        self.decay, self.state = decay, {}

    @torch.no_grad()
    def update(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad:
                if name not in self.state:
                    self.state[name] = p.detach().clone()
                else:
                    self.state[name].lerp_(p.detach(), 1 - self.decay)

    @contextlib.contextmanager
    def apply(self, model):
        parameters = dict(model.named_parameters())
        raw = {k: parameters[k].detach().clone() for k in self.state}
        try:
            with torch.no_grad():
                for k, v in self.state.items():
                    parameters[k].copy_(v)
            yield
        finally:
            with torch.no_grad():
                for k, v in raw.items():
                    parameters[k].copy_(v)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None)


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['cpu'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'])


def save_checkpoint(path, model, optimizer, ema, step, protocol, rank_rngs, metrics):
    weights = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    state = dict(schema='mouse-semseg-state-v1', model=weights, optimizer=optimizer.state_dict(),
                 ema={k: v.cpu() for k, v in ema.state.items()}, step=step, protocol=protocol,
                 requires_grad={n: p.requires_grad for n, p in model.named_parameters()},
                 rng_by_rank=rank_rngs, metrics=metrics,
                 buffers={n: b.detach().cpu() for n, b in model.named_buffers()})
    temporary = Path(path).with_name(Path(path).name + '.tmp')
    torch.save(state, temporary); temporary.replace(path)


def load_checkpoint(path, model, optimizer, ema, protocol, rank=0, inference=False):
    state = torch.load(path, map_location='cpu', weights_only=False)
    expected = dict(protocol)
    if inference:
        # Evaluate an eight-rank checkpoint on one GPU; training resume is strict.
        expected['world_size'] = state['protocol'].get('world_size')
    if state['schema'] != 'mouse-semseg-state-v1' or state['protocol'] != expected:
        raise ValueError('Resume protocol mismatch (data/model/schedule/world size must match)')
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state['model'].items():
            parameters[name].copy_(value)
        for name, value in state['buffers'].items():
            dict(model.named_buffers())[name].copy_(value)
    for name, enabled in state['requires_grad'].items():
        parameters[name].requires_grad = enabled
    if not inference:
        optimizer.load_state_dict(state['optimizer'])
    device = next(model.parameters()).device
    for values in optimizer.state.values():
        for key, value in values.items():
            if torch.is_tensor(value) and key != 'step':
                values[key] = value.to(device)
    ema.state = {k: v.to(device) for k, v in state['ema'].items()}
    if not inference:
        restore_rng(state['rng_by_rank'][rank])
    return state['step'], state['metrics']


def mask_metrics(prediction, truth, tolerance=1):
    valid = truth != 255; pred = (prediction == 1) & valid; gt = (truth == 1) & valid
    intersection = int((pred & gt).sum()); union = int((pred | gt).sum())
    kernel = np.ones((3, 3), np.uint8)
    def edge(mask):
        return mask & ~(cv2.erode(mask.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT,
                                  borderValue=0).astype(bool))
    p, t = edge(pred), edge(gt)
    p &= valid; t &= valid
    size = 2 * tolerance + 1
    dilate = np.ones((size, size), np.uint8)
    precision = (p & cv2.dilate(t.astype(np.uint8), dilate).astype(bool)).sum() / max(1, p.sum())
    recall = (t & cv2.dilate(p.astype(np.uint8), dilate).astype(bool)).sum() / max(1, t.sum())
    bf = 2 * precision * recall / max(1e-12, precision + recall)
    if not p.any() and not t.any():
        bf = 1.
    return dict(iou=intersection / union if union else 1.,
                dice=2 * intersection / max(1, int(pred.sum() + gt.sum())) if union else 1.,
                boundary_f1=float(bf), empty=not bool(pred.any()),
                all_foreground=bool(pred[valid].all()) if valid.any() else False)
