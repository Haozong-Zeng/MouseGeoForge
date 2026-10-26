"""Synthetic keypoints as a THIRD domain for the chosen head-first recipe (2026-10-26).

Question: does adding Blender five-point labels (nose, L/R forepaw centre, L/R hindpaw centre, the same
landmark names as the real five-point set) to the UX_flat1e-4 recipe improve the REAL held-out sessions?
Data: the legacy BlenderPyMouse pools of the v3 geometry manifest (CRIM13 homecage, MoSeq, ClimbWall,
WashFace), which carry Core23 keypoints with visibility; the crossed-rig renders have no keypoints and are
not used. Real items are the pinned head-first list (output/head_first_kp_20261024/items.json), unchanged.
Every update draws 8 own + 8 PanoMouse + N synthetic crops; loss = CE_own + CE_pano + w * CE_synth.
Judged on the real DEV sessions only; the synthetic validation pool is reported for monitoring.
Weight-free: logs and metrics only.

    prepare                          real items (hash-checked) + synthetic five-point items -> items.json
    cache [--workers N]              crops.u8.npy for every item (real crops copied from the existing cache)
    run --variant synth8 --seed N    UX_flat1e-4 + synthetic domain (one H100, ~85 min)
    table --seed N                   against the sweep/retrain UX_flat1e-4 of the same seed
"""
import argparse
import collections
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import head_first_kp_20261024 as kp  # noqa: E402
import head_first_kp_sweep_20261024 as sw  # noqa: E402

REAL = kp.OUT                                   # pinned real items + their crop cache
OUT = ROOT/'output/head_first_kp_synth_20261026'
RUNS = OUT/'runs'
MANIFEST = ROOT/'output/bottom_geometry_v3_local12h_20261018/manifest.json'
FIVE = ['nose', 'left_forepaw_center', 'right_forepaw_center', 'left_hindpaw_center', 'right_hindpaw_center']
BASE = 'UX_flat1e-4'
VARIANTS = {
    'synth8': dict(synth_per_update=8, synth_weight=1.0),    # a full third domain
    'synth4': dict(synth_per_update=4, synth_weight=0.5),    # a half-weight regulariser
}
KEYS = sw.KEYS
sha, write, write_new = kp.sha, kp.write, kp.write_new


def load_items():
    return json.loads((OUT/'items.json').read_text())


def motion_of(group):
    for key in ('climb', 'moseq', 'wash', 'crim13'):
        if key in group:
            return key
    return 'other'


# --------------------------------------------------------------------------- synthetic items

def synth_rows(z, record, views, five=FIVE, min_box=8.0):
    """Items for every (view, instance) of one keypoints.npz. Box = in-frame valid landmarks (all 23);
    points = the five landmarks; known = label valid & in frame & surface visible."""
    names = z['names'].tolist()
    idx = [names.index(n) for n in five]
    uv, valid, vis, inf = z['uv'], z['xy_label_valid'], z['surface_visible'], z['in_frame']
    multi = uv.ndim == 4
    motion, nviews = motion_of(record['group']), len(z['view_ids'])
    out = []
    for k, vid in enumerate(z['view_ids'].tolist()):
        v = views.get(int(vid))
        if v is None:
            continue
        for i in range(uv.shape[1] if multi else 1):
            u = (uv[k, i] if multi else uv[k]).astype(np.float64)
            known23 = (valid[k, i] if multi else valid[k]) & (inf[k, i] if multi else inf[k]) & np.isfinite(u).all(-1)
            vis23 = known23 & (vis[k, i] if multi else vis[k])
            if known23.sum() < 3 or vis23[idx].sum() == 0:
                continue
            p = u[known23]
            box = [float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())]
            if box[2] - box[0] < min_box or box[3] - box[1] < min_box:
                continue
            out.append(dict(id=hashlib.sha256(f"{record['scene']}/{vid}/{i}".encode()).hexdigest()[:20], dataset='synth',
                            split='train' if record['split'] == 'train' else 'dev', session=f'synth_{motion}{nviews}',
                            group=f"{record['scene']}#{i}", camera=str(vid), bottom=v['physical_role'] == 'bottom',
                            image=v['image'], box=box, points=u[idx].tolist(), known=[bool(x) for x in vis23[idx]],
                            aux=None, aux_known=None, affine=kp.hist.crop_affine(box).tolist(), instance=i))
    return out


def prepare():
    if (OUT/'items.json').exists():
        raise SystemExit('items.json exists: the item list is frozen; use a new namespace to change it')
    real = kp.load_items()
    receipt = json.loads((REAL/'input_receipt.json').read_text())
    if sha(REAL/'items.json') != receipt['items']['sha256']:
        raise SystemExit('real items.json differs from its receipt')
    manifest = json.loads(MANIFEST.read_text())
    synth, skipped = [], collections.Counter()
    for r in manifest['records']:
        if r['kind'] != 'synthetic' or not r.get('keypoints'):
            skipped[f"{r['kind']}/no_keypoints" if r['kind'] == 'synthetic' else r['kind']] += 1; continue
        if r['split'] == 'sealed' or 'sealed' in str(r.get('rig', '')):
            skipped['sealed'] += 1; continue
        views = {v['id']: v for v in r['views']}
        with np.load(ROOT/r['keypoints'], allow_pickle=False) as z:
            synth += synth_rows(z, r, views)
    items = real + synth
    assert len({i['id'] for i in items}) == len(items)
    counts = collections.Counter(f"{i['dataset']}/{i['split']}/{'bottom' if i['bottom'] else 'other'}" for i in items)
    known = collections.Counter()
    for i in synth:
        known[f"{i['session']}/{'bottom' if i['bottom'] else 'other'}"] += sum(i['known'])
    OUT.mkdir(parents=True, exist_ok=True)
    write_new(OUT/'items.json', items)
    write_new(OUT/'input_receipt.json', dict(
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        real=dict(path=str(REAL.relative_to(ROOT)/'items.json'), sha256=receipt['items']['sha256'], n=len(real), unchanged=True),
        synthetic=dict(manifest=str(MANIFEST.relative_to(ROOT)), manifest_sha256=sha(MANIFEST), n=len(synth), five=FIVE,
                       known_points=dict(known), skipped_records=dict(skipped),
                       semantics='Blender joint-based landmarks named like the real five points; forepaw/hindpaw "center" '
                                 'is the Blender paw centre joint, an approximation of the real plantar centre',
                       box='bounding box of the in-frame valid Core23 landmarks per instance (no annotation boxes exist); '
                           'two-mouse CRIM13 scenes give one crop per mouse', visibility='xy_label_valid & in_frame & surface_visible'),
        items=dict(n=len(items), sha256=sha(OUT/'items.json'), counts=dict(counts)), tool_sha256=sha(Path(__file__))))
    print('prepared', len(real), 'real +', len(synth), 'synthetic =', len(items), dict(counts))


# --------------------------------------------------------------------------- crops

def cache(workers=12):
    import torch
    receipt = OUT/'cache/receipt.json'
    if receipt.exists():
        print('cache exists'); return
    items = load_items(); N = len(items)
    real = json.loads((REAL/'items.json').read_text()); n_real = len(real)
    assert [i['id'] for i in items[:n_real]] == [i['id'] for i in real], 'real items must lead the list in their original order'
    old = np.load(REAL/'cache/crops.u8.npy', mmap_mode='r'); assert old.shape == (n_real, 512, 512, 3)
    (OUT/'cache').mkdir(exist_ok=True)
    crops = np.lib.format.open_memmap(OUT/'cache/crops.u8.npy.partial', 'w+', np.uint8, (N, 512, 512, 3))
    t0 = time.time()
    for i in range(0, n_real, 256):
        j = min(i + 256, n_real)  # the last block must not spill into the synthetic rows
        crops[i:j] = old[i:j]
    print('copied', n_real, 'real crops', round(time.time() - t0), flush=True)

    class Crops(torch.utils.data.Dataset):
        def __len__(self):
            return N - n_real

        def __getitem__(self, j):
            import cv2
            cv2.setNumThreads(0)
            it = items[n_real + j]
            im = cv2.imread(str(ROOT/it['image']), cv2.IMREAD_COLOR)
            if im is None:
                raise FileNotFoundError(it['image'])
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            return j, cv2.warpAffine(im, np.array(it['affine']), (512, 512), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

    loader = torch.utils.data.DataLoader(Crops(), batch_size=None, num_workers=workers, prefetch_factor=4)
    for j, crop in loader:
        crops[n_real + int(j)] = crop.numpy()
        if int(j) % 1000 == 0:
            print('CACHE', int(j), N - n_real, round(time.time() - t0), flush=True)
    crops.flush(); del crops
    os.replace(OUT/'cache/crops.u8.npy.partial', OUT/'cache/crops.u8.npy')
    write_new(receipt, dict(items_sha256=sha(OUT/'items.json'), n=N, n_real=n_real, n_synth=N - n_real,
                            real_cache=dict(path=str((REAL/'cache/crops.u8.npy').relative_to(ROOT)), copied=True),
                            crops=dict(shape=[N, 512, 512, 3], dtype='uint8'),
                            preprocessing='historical B: BGR->RGB, warpAffine INTER_LINEAR border 0 (synthetic); real crops byte-copied',
                            tool_sha256=sha(Path(__file__)), seconds=round(time.time() - t0)))
    print('CACHE done', N, round(time.time() - t0))


# --------------------------------------------------------------------------- training

class SynthTrainer(kp.Trainer):
    """kp.Trainer with a third sampler; the step is the historical one plus the weighted synthetic term."""

    def __init__(self, mode, seed, cfg, store, head, pe=None, mean=None, std=None, **kw):
        super().__init__(mode, seed, cfg, store, head, pe, mean, std, **kw)
        synth = [i for i, it in enumerate(store.items) if it['dataset'] == 'synth' and it['split'] == 'train']
        self.s_synth = kp.Sampler(synth, seed + 4000)
        self.n_synth, self.w_synth = cfg['task']['synth_views_per_update'], cfg['synth']['weight']

    def step(self):
        torch = self.torch
        u = self.u + 1
        self.plan.apply(self.opt, u)
        b_own, b_pano = self.s_own.take(self.batch[0]), self.s_pano.take(self.batch[1])
        b_synth = self.s_synth.take(self.n_synth) if self.n_synth else []
        ids = b_own + b_pano + b_synth
        logits = self.head(self.features(ids, u), self.store.rgb_for(ids))
        ce = kp.point_ce(logits, self.store.pts[ids]); kn = self.store.known[ids]
        n, m = len(b_own), len(b_own) + len(b_pano)
        loss_own, loss_pano = kp.known_mean(ce[:n], kn[:n]), kp.known_mean(ce[n:m], kn[n:m])
        loss_synth = kp.known_mean(ce[m:], kn[m:]) if b_synth else ce.sum() * 0
        loss = loss_own + loss_pano + self.w_synth * loss_synth
        if not torch.isfinite(loss):
            raise FloatingPointError(f'non-finite loss at update {u}')
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        head = [p for g in self.plan.groups if g['name'].startswith('head') and u >= g['start'] for _, p in g['params']]
        enc = [p for g in self.plan.groups if not g['name'].startswith('head') and u >= g['start'] for _, p in g['params']]
        gn = float(torch.nn.utils.clip_grad_norm_(head, self.clip))
        gn_enc = float(torch.nn.utils.clip_grad_norm_(enc, self.clip)) if enc else 0.0
        if not (math.isfinite(gn) and math.isfinite(gn_enc)):
            raise FloatingPointError(f'non-finite gradient norm at update {u}')
        self.clipped += gn > self.clip
        self.clipped_encoder += gn_enc > self.clip
        self.opt.step()
        with torch.no_grad():
            for g in self.plan.groups:
                if u >= g['start']:
                    for name, p in g['params']:
                        self.ema[name].mul_(self.ema_decay).add_(p.detach(), alpha=1 - self.ema_decay)
        self.u = u
        for i in ids:
            it = self.store.items[i]
            self.exposure[f"{it['dataset']}/{'bottom' if it['bottom'] else 'other'}"] += 1
        self.exposure['known_points_own'] += int(kn[:n].sum()); self.exposure['known_points_pano'] += int(kn[n:m].sum())
        self.exposure['known_points_synth'] += int(kn[m:].sum())
        if u % 50 == 0 or u == 1:
            self.log.append(dict(update=u, loss_own=loss_own.item(), loss_pano=loss_pano.item(), loss_synth=float(loss_synth),
                                 grad_norm=gn, grad_norm_encoder=gn_enc,
                                 lr={g['name']: og['lr'] for g, og in zip(self.plan.groups, self.opt.param_groups)}))
        return float(loss.detach())


def variant_cfg(name):
    v = VARIANTS[name]
    cfg = sw.variant_cfg(sw.VARIANTS[BASE])
    cfg['task']['synth_views_per_update'] = v['synth_per_update']
    cfg['synth'] = dict(weight=v['synth_weight'], base=BASE)
    return cfg


def run(name, seed):
    import torch
    cfg = variant_cfg(name); base = sw.VARIANTS[BASE]
    dest = RUNS/f'{name}_seed{seed}'
    if (dest/'DONE.json').exists():
        print('done already:', dest); return
    dest.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    items = load_items()
    receipt = json.loads((OUT/'cache/receipt.json').read_text())
    if receipt['items_sha256'] != sha(OUT/'items.json'):
        raise SystemExit('crop cache was built from a different item list')
    _, pe, mean, std = kp.release_parts()
    kp.OUT = OUT  # kp.Store reads the crop cache of THIS namespace; kp.RUNS (H0 heads) stays on the real namespace
    store = kp.Store(items, 'cuda', need_feats=False)
    torch.manual_seed(seed)
    head = kp.build_head(5).cuda()
    h0 = kp.RUNS/f'H0_seed{seed}/endpoint.pt'
    head.load_state_dict(torch.load(h0, weights_only=False)['ema_head']); head.eval()
    t = SynthTrainer('UX', seed, cfg, store, head, pe, mean, std)
    release_hashes = kp.param_hashes(pe, 'aggregator.patch_embed')
    release_copy = {n: p.detach().cpu().clone() for n, p in t.plan.named() if not n.startswith('head')}
    n_synth_train = len(t.s_synth.pool)
    kp.write(dest/'run_protocol.json', dict(
        variant=name, seed=seed, mode='UX', base=BASE, base_variant=base, synth=VARIANTS[name],
        per_update=dict(own=cfg['task']['own_views_per_update'], pano=cfg['task']['pano_views_per_update'], synth=cfg['task']['synth_views_per_update']),
        updates=base['updates'], crops=base['updates'] * (16 + cfg['task']['synth_views_per_update']),
        loss='CE_own + CE_pano + weight * CE_synth (each the mean over the known points of its domain)',
        pair={k: cfg['pair'][k] for k in ('head_peak_lr', 'head_warmup_updates', 'cosine_total_updates', 'cosine_floor',
                                          'weight_decay', 'grad_clip', 'ema_decay', 'dev_eval_updates')},
        protected=cfg.get('protected'), explore=cfg.get('explore'), h0_head=str(h0.relative_to(ROOT)), h0_sha256=sha(h0),
        groups=[dict(name=g['name'], peak=g['peak'], start=g['start'], tensors=len(g['params'])) for g in t.plan.groups],
        excluded_frozen=t.plan.excluded, synth_train_pool=n_synth_train, items_sha256=receipt['items_sha256'],
        weights_saved=False, data_placement=store.placement, judged_on='real DEV sessions only (own + PanoMouse); synthetic DEV is monitoring',
        start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), host=os.uname().nodename, slurm_job=os.environ.get('SLURM_JOB_ID'),
        tool_sha256=sha(Path(__file__)), sweep_tool_sha256=sha(Path(sw.__file__)), kp_tool_sha256=sha(Path(kp.__file__))))
    dev_ids = [i for i, it in enumerate(items) if it['split'] == 'dev']
    points = set(cfg['pair']['dev_eval_updates'])
    last_summary = None
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time(); train_seconds = 0.0
    print(f'{name} seed{seed}: synth train pool {n_synth_train}, dev views {len(dev_ids)}, store on {store.placement}', flush=True)
    while t.u < base['updates']:
        s0 = time.time(); t.step(); train_seconds += time.time() - s0
        if t.u % 500 == 0:
            l = t.log[-1]
            print(f"u{t.u} own {l['loss_own']:.3f} pano {l['loss_pano']:.3f} synth {l['loss_synth']:.3f} {train_seconds / t.u:.3f}s/u", flush=True)
        if t.u in points:
            kp.write(dest/f'audit_u{t.u:05d}.json', kp.stage_audit(t, release_hashes, release_copy))
            for kind in ('ema', 'raw'):
                rows = t.evaluate(dev_ids, use_ema=kind == 'ema')
                summary = kp.summarize(rows)
                if kind == 'ema':
                    last_summary = summary
                kp.write(dest/f'dev_u{t.u:05d}_{kind}.json', dict(update=t.u, weights=kind, summary=summary, rows=rows))
            kp.write(dest/'train_log.json', dict(log=t.log, exposure=dict(t.exposure)))
            s = last_summary
            print(f"{name} seed{seed} u{t.u} DEV ema: pano/bottom/paws {100 * s['pano/bottom/paws']['pck05']:.2f} "
                  f"swap {100 * s['pano/bottom/paws']['swap_rate']:.2f} own/bottom/paws {100 * s['own/bottom/paws']['pck05']:.2f}, "
                  f"{round(time.time() - t0)}s", flush=True)
    kp.write(dest/'train_log.json', dict(log=t.log, exposure=dict(t.exposure)))
    kp.write_new(dest/'DONE.json', dict(variant=name, seed=seed, updates=t.u, weights_saved=False, endpoint=None,
                                        seconds=round(time.time() - t0), train_seconds_per_update=train_seconds / t.u,
                                        peak_memory_gb=torch.cuda.max_memory_allocated() / 2**30,
                                        clipped_head_updates=t.clipped, clipped_encoder_updates=t.clipped_encoder,
                                        summary={k: last_summary[k] for k in KEYS if k in last_summary},
                                        finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    print('DONE', name, seed, flush=True)


def table(seed):
    rows = {}
    sources = {f'{BASE} (sweep)': sw.SWEEP/f'{BASE}_seed{seed}', f'{BASE} (retrain)': sw.RETRAIN/f'{BASE}_seed{seed}'}
    sources.update({name: RUNS/f'{name}_seed{seed}' for name in VARIANTS})
    for label, dest in sources.items():
        if not (dest/'DONE.json').exists():
            rows[label] = None; continue
        done = json.loads((dest/'DONE.json').read_text())
        s = kp.summarize(json.loads((dest/f"dev_u{done['updates']:05d}_ema.json").read_text())['rows'])
        rows[label] = dict(done={k: done.get(k) for k in ('updates', 'seconds', 'peak_memory_gb', 'clipped_encoder_updates')},
                           ema={k: dict(pck05=s[k]['pck05'], swap=s[k]['swap_rate'], median_px=s[k]['median_px'],
                                        cov32_ref=s[k]['cov32_ref'], p95_ref=s[k]['p95_ref']) for k in KEYS if k in s},
                           sessions={k: dict(pck05=v['pck05'], p95_ref=v['p95_ref']) for k, v in s.items() if k.startswith('session/')})
    kp.write(RUNS/f'table_seed{seed}.json', dict(seed=seed, rows=rows, keys=KEYS, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    print(f'{"run":22s} {"pano/bot PCK":>12s} {"swap":>6s} {"pano/oth":>8s} {"own/bot":>7s} {"own/oth":>7s} {"own P95":>7s} {"pano P95":>8s} {"s/upd":>6s} {"GB":>5s}')
    for label, r in rows.items():
        if r is None:
            print(f'{label:22s} (not finished)'); continue
        e = r['ema']
        print(f"{label:22s} {100*e['pano/bottom/paws']['pck05']:12.2f} {100*e['pano/bottom/paws']['swap']:6.2f} "
              f"{100*e['pano/other/paws']['pck05']:8.2f} {100*e['own/bottom/paws']['pck05']:7.2f} {100*e['own/other/paws']['pck05']:7.2f} "
              f"{e['own/all/all']['p95_ref']:7.1f} {e['pano/all/all']['p95_ref']:8.1f} "
              f"{r['done']['train_seconds_per_update'] or 0:6.3f} {r['done']['peak_memory_gb'] or 0:5.1f}")
        synth = {k.split('/')[1]: v for k, v in r['sessions'].items() if k.startswith('session/synth') and k.endswith('/bottom/paws')}
        if synth:
            print(' ' * 22 + 'synthetic DEV bottom paws (monitoring): ' + ', '.join(f"{k} {100*v['pck05']:.1f}" for k, v in synth.items()))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['prepare', 'cache', 'run', 'table'])
    ap.add_argument('--variant', choices=sorted(VARIANTS), default='synth8')
    ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--workers', type=int, default=12)
    a = ap.parse_args()
    if a.step == 'prepare':
        prepare()
    elif a.step == 'cache':
        cache(a.workers)
    elif a.step == 'run':
        run(a.variant, a.seed)
    else:
        table(a.seed)
