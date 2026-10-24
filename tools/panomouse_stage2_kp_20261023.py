"""PanoMouse stage 2: does PanoMouse data improve 2D keypoints (bottom-view paws, left/right)?

Arms (paired; seeds 17/29/43; identical update count, head init seed and own-data order):
  A  own PyMouseStudio data only: 8 own views per update, 5 outputs
  B  A + 8 PanoMouse views per update with the 5 mapped points (extra compute, disclosed)
  C  B + 24 PanoMouse body points as auxiliary outputs (supervised on PanoMouse only)
Features: frozen released VGGT image-encoder patch tokens (as the S1 segmentation head), on a
square crop around the mouse resized to 512 (crop from mask bbox / annotation bbox: GT-box
top-down protocol). Head: spatial heatmap decoder at 128x128, softmax CE to Gaussian targets.
Evaluation: own held-out sessions AZR2L2/FR2/FR3 and PanoMouse dev S03/S11. Sealed S04/S05,
GR2/UR2/UR3 never read.

Sub-commands: prepare | cache | train --arm A|B|C --seed N | evaluate
"""
import argparse
import collections
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
OUT = ROOT / 'output/panomouse_stage2_kp_20261023'
PANO = ROOT / 'dataset/PanoMouse'
EFF = ROOT / 'dataset/PyMouseStudio/annotation_versions/20261003_effective_v1/manifest.json'
RELEASE = ROOT / 'vggt-omega/vggt_omega_1b_512.pt'
OWN_TRAIN = ['AZL3', 'AZR2', 'CR2', 'EL2', 'EL2L2', 'ER3L2']
OWN_DEV = ['AZR2L2', 'FR2', 'FR3']
PANO_TRAIN = ['S01', 'S02', 'S06', 'S07', 'S08', 'S09', 'S10']
PANO_DEV = ['S03', 'S11']
OWN_POINTS = ['nose', 'left_forepaw_center', 'right_forepaw_center', 'left_hindpaw_center', 'right_hindpaw_center']
QA_THRESHOLD = 81.48566734926273  # stage 0: P99 TRAIN reprojection error, 2704x2028 px
UPDATES, OWN_BATCH, PANO_BATCH = 6000, 8, 8


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def crop_affine(box, margin=1.3, side_min=64):
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0, side_min) * margin
    s = 512 / side
    return np.array([[s, 0, 256 - s * cx], [0, s, 256 - s * cy]], np.float64)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    items = []
    # --- own PyMouseStudio
    em = json.loads(EFF.read_text())
    prov = {(v['session'], v['frame'], v['camera']): v for v in em['provenance'] if v.get('mask') and v.get('mask_valid')}
    for g in em['frame_groups']:
        s = g['session']
        if g['excluded'] or s not in OWN_TRAIN + OWN_DEV:
            continue
        z = np.load(ROOT / g['effective_normalized_file'])
        names = [str(n) for n in z['names']]; idx = [names.index(n) for n in OWN_POINTS]
        cams = [str(c) for c in z['camera_names']]
        for vi, c in enumerate(cams):
            v = prov.get((s, g['frame'], c))
            if v is None:
                continue
            uv = z['uv'][vi, 0][idx]; known = z['xy_observed'][vi, 0][idx] & np.isfinite(uv).all(-1)
            if not known.any():
                continue
            m = cv2.imread(str(ROOT / v['mask']), 0) > 0
            ys, xs = np.nonzero(m)
            box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
            items.append(dict(id=hashlib.sha256(f'own/{s}/{g["frame"]}/{c}'.encode()).hexdigest()[:20], dataset='own',
                              split='train' if s in OWN_TRAIN else 'dev', session=s, group=f'{s}_{g["frame"]}', camera=c,
                              bottom=c == 'A', image=v['source_image'], box=box,
                              points=np.nan_to_num(uv).tolist(), known=known.tolist(), aux=None, aux_known=None))
    # --- PanoMouse
    a2 = json.loads((PANO / 'raw/annotations/annotations_2d.json').read_text())
    a3 = json.loads((PANO / 'raw/annotations/annotations_3d.json').read_text())
    cal = json.loads((PANO / 'raw/annotations/calib_data.json').read_text())
    names = a2['categories'][0]['keypoints'][:24]; ix = {n: names.index(n) for n in names}
    images = {im['id']: im for im in a2['images']}; ann = {an['image_id']: an for an in a2['annotations']}
    dropped = collections.Counter()
    for g in a2['groups']:
        s = g['scene_name'].split('_')[0]
        if s not in PANO_TRAIN + PANO_DEV:
            continue
        X = np.array(a3[g['scene_name']])[:24]
        for c, iid in zip(g['cams'], g['image_id']):
            d = cal[s][c]; K, R, T = np.array(d['K']), np.array(d['R']), np.array(d['T']).reshape(3)
            kp = np.array(ann[iid]['keypoints']).reshape(-1, 3)
            P = (K @ (R @ X[:, :3].T + T[:, None])).T; proj = P[:, :2] / P[:, 2:]
            ok = (kp[:, 2] == 2) & (X[:, 3] > 0) & (np.linalg.norm(proj - kp[:, :2], axis=1) <= QA_THRESHOLD)
            dropped['labelled_flag2'] += int((kp[:, 2] == 2).sum()); dropped['kept'] += int(ok.sum())
            def mid(a, b):
                return ((kp[ix[a], :2] + kp[ix[b], :2]) / 2, bool(ok[ix[a]] and ok[ix[b]]))
            five = [(kp[ix['nose'], :2], bool(ok[ix['nose']])), (kp[ix['Lhand'], :2], bool(ok[ix['Lhand']])),
                    (kp[ix['Rhand'], :2], bool(ok[ix['Rhand']])), mid('Lheel', 'Lfoot'), mid('Rheel', 'Rfoot')]
            x, y, w, h = ann[iid]['bbox']
            items.append(dict(id=hashlib.sha256(f'pano/{iid}'.encode()).hexdigest()[:20], dataset='pano',
                              split='train' if s in PANO_TRAIN else 'dev', session=s, group=g['scene_name'], camera=c,
                              bottom=int(c[3:]) >= 16, image=str((PANO / 'unpacked/images' / images[iid]['file_name']).relative_to(ROOT)),
                              box=[x, y, x + w, y + h], points=[p.tolist() for p, _ in five], known=[k for _, k in five],
                              aux=kp[:, :2].tolist(), aux_known=ok.tolist()))
    for it in items:
        it['affine'] = crop_affine(it['box']).tolist()
    (OUT / 'items.json').write_text(json.dumps(items))
    cnt = collections.Counter((i['dataset'], i['split'], 'bottom' if i['bottom'] else 'other') for i in items)
    protocol = dict(created=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), doc=__doc__, updates=UPDATES, own_batch=OWN_BATCH,
                    pano_batch=PANO_BATCH, seeds=[17, 29, 43], lr=3e-4, weight_decay=.01, warmup=200, ema=.995, sigma_cells=2.0,
                    mapping=dict(nose='nose', left_forepaw_center='Lhand', right_forepaw_center='Rhand',
                                 left_hindpaw_center='midpoint(Lheel, Lfoot)', right_hindpaw_center='midpoint(Rheel, Rfoot)'),
                    pano_label_rule='flag==2, released 3D valid, reprojection <= stage-0 TRAIN P99 81.49 px; hindpaw needs both heel and foot',
                    pano_label_counts=dict(dropped), counts={'/'.join(k): v for k, v in sorted(cnt.items())},
                    release_sha256=sha(RELEASE), items_sha256=sha(OUT / 'items.json'), no_flip_augmentation='left/right semantics')
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=1))
    print(json.dumps(protocol['counts'], indent=1), dict(dropped))


def cache():
    import torch
    import bottom_geometry_v4_runtime_20261019 as rt
    items = json.loads((OUT / 'items.json').read_text())
    (OUT / 'cache').mkdir(exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    model = rt.model(RELEASE, trainable=False).eval(); a = model.aggregator
    t0 = time.time()
    with torch.inference_mode():
        for i, it in enumerate(items):
            dest = OUT / 'cache' / f'{it["id"]}.pt'
            if dest.exists():
                continue
            im = cv2.cvtColor(cv2.imread(str(ROOT / it['image'])), cv2.COLOR_BGR2RGB)
            crop = cv2.warpAffine(im, np.array(it['affine']), (512, 512), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
            x = torch.from_numpy(crop).permute(2, 0, 1).float().cuda()[None] / 255
            with torch.autocast('cuda', dtype=torch.bfloat16):
                o = a.patch_embed((x - a._resnet_mean.view(1, 3, 1, 1)) / a._resnet_std.view(1, 3, 1, 1))
            f = o['x_norm_patchtokens'].transpose(1, 2).reshape(1024, 32, 32).half().cpu()
            torch.save(dict(feat=f, rgb=torch.from_numpy(crop).permute(2, 0, 1).contiguous()), dest)
            if i % 500 == 0:
                print('CACHE', i, len(items), round(time.time() - t0), flush=True)
    print('CACHE done', len(items), round(time.time() - t0))


def build_head(k):
    import torch.nn as nn
    class KPHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = nn.Sequential(nn.GroupNorm(16, 1024), nn.Conv2d(1024, 128, 1), nn.GELU())
            self.rgb = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.GroupNorm(4, 16), nn.GELU(), nn.Conv2d(16, 16, 3, 2, 1), nn.GroupNorm(4, 16), nn.GELU())
            self.decode = nn.Sequential(nn.Conv2d(144, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
                                        nn.Conv2d(128, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.GELU(), nn.Conv2d(128, k, 1))
        def forward(self, feat, rgb):
            import torch.nn.functional as F
            f = F.interpolate(self.project(feat), size=(128, 128), mode='bilinear', align_corners=False)
            return self.decode(__import__('torch').cat([f, self.rgb(rgb)], 1))
    return KPHead()


class Data:
    def __init__(self, items, ids):
        self.items = items; self.ids = ids
    def batch(self, ids, k_out, with_aux):
        import torch
        feats, rgbs, tgt, known = [], [], [], []
        for i in ids:
            it = self.items[i]; c = torch.load(OUT / 'cache' / f'{it["id"]}.pt', weights_only=True)
            feats.append(c['feat']); rgbs.append(c['rgb'])
            A = np.array(it['affine'])
            pts = np.array(it['points']) @ A[:, :2].T + A[:, 2]; kn = np.array(it['known'])
            if k_out > 5:
                if it['aux'] is not None:
                    aux = np.array(it['aux']) @ A[:, :2].T + A[:, 2]; akn = np.array(it['aux_known'])
                else:
                    aux = np.zeros((24, 2)); akn = np.zeros(24, bool)
                pts = np.concatenate([pts, aux]); kn = np.concatenate([kn, akn])
            kn = kn & (pts >= 0).all(-1) & (pts < 512).all(-1)
            tgt.append(pts); known.append(kn)
        return (torch.stack(feats).cuda().float(), torch.stack(rgbs).cuda().float() / 255,
                torch.tensor(np.array(tgt), dtype=torch.float32).cuda(), torch.tensor(np.array(known)).cuda())


def heatmap_loss(logits, pts, known, sigma=2.0):
    import torch
    B, K, H, W = logits.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=logits.device), torch.arange(W, device=logits.device), indexing='ij')
    c = (pts + .5) / 4 - .5  # 512 px -> 128 cells (pixel-centre convention)
    g = -((xx[None, None] - c[..., 0, None, None]) ** 2 + (yy[None, None] - c[..., 1, None, None]) ** 2) / (2 * sigma ** 2)
    target = g.flatten(2).softmax(-1)
    ce = -(target * logits.flatten(2).log_softmax(-1)).sum(-1)
    return ce[known].mean() if known.any() else logits.sum() * 0


def decode(logits):
    import torch
    B, K, H, W = logits.shape
    flat = logits.flatten(2); idx = flat.argmax(-1)
    y0, x0 = idx // W, idx % W
    off = torch.arange(-2, 3, device=logits.device)
    ys = (y0[..., None] + off).clamp(0, H - 1); xs = (x0[..., None] + off).clamp(0, W - 1)
    patch = torch.stack([torch.stack([logits[b, k][ys[b, k]][:, xs[b, k]] for k in range(K)]) for b in range(B)])
    p = patch.flatten(2).softmax(-1).view(B, K, 5, 5)
    y = (p.sum(-1) * ys.float()).sum(-1); x = (p.sum(-2) * xs.float()).sum(-1)
    return torch.stack([(x + .5) * 4 - .5, (y + .5) * 4 - .5], -1)


def train(arm, seed):
    import torch
    items = json.loads((OUT / 'items.json').read_text())
    dest = OUT / f'{arm}_seed{seed}'; dest.mkdir(exist_ok=False)
    own = [i for i, it in enumerate(items) if it['dataset'] == 'own' and it['split'] == 'train']
    pano = [i for i, it in enumerate(items) if it['dataset'] == 'pano' and it['split'] == 'train']
    k_out = 29 if arm == 'C' else 5
    torch.manual_seed(seed); head = build_head(k_out).cuda()
    if k_out == 29:  # same initial weights as A/B for everything except the extra output rows
        torch.manual_seed(seed); ref = build_head(5).cuda()
        sd = head.state_dict(); r = ref.state_dict()
        for k, v in r.items():
            sd[k] = v if sd[k].shape == v.shape else torch.cat([v, sd[k][v.shape[0]:]], 0)
        head.load_state_dict(sd)
    ema = {k: v.detach().clone() for k, v in head.state_dict().items()}
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200) * (.05 + .95 * .5 * (1 + math.cos(math.pi * min(s, UPDATES) / UPDATES))))
    rng_own = random.Random(seed); rng_pano = random.Random(seed + 1000)
    D = Data(items, None); log = []; t0 = time.time()
    order_own, order_pano = [], []
    for step in range(UPDATES):
        if len(order_own) < OWN_BATCH:
            o = own[:]; rng_own.shuffle(o); order_own += o
        b_own = [order_own.pop() for _ in range(OWN_BATCH)]
        f, rgb, pts, kn = D.batch(b_own, k_out, False)
        loss_own = heatmap_loss(head(f, rgb)[:, :5] if k_out > 5 else head(f, rgb), pts[:, :5], kn[:, :5])
        loss = loss_own; loss_pano = None
        if arm in ('B', 'C'):
            if len(order_pano) < PANO_BATCH:
                o = pano[:]; rng_pano.shuffle(o); order_pano += o
            b_p = [order_pano.pop() for _ in range(PANO_BATCH)]
            f2, rgb2, pts2, kn2 = D.batch(b_p, k_out, arm == 'C')
            loss_pano = heatmap_loss(head(f2, rgb2), pts2, kn2)
            loss = loss + loss_pano
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(head.parameters(), 10.)); opt.step(); sched.step()
        with torch.no_grad():
            for k, v in head.state_dict().items():
                ema[k].mul_(.995).add_(v.detach(), alpha=.005) if v.dtype.is_floating_point else ema[k].copy_(v)
        if step % 50 == 0 or step == UPDATES - 1:
            log.append(dict(step=step, loss_own=float(loss_own), loss_pano=None if loss_pano is None else float(loss_pano), grad_norm=gn, lr=sched.get_last_lr()[0]))
    torch.save(dict(head=head.state_dict(), ema=ema, arm=arm, seed=seed, k_out=k_out), dest / 'final.pt')
    (dest / 'train_log.json').write_text(json.dumps(dict(log=log, seconds=time.time() - t0, own_views_seen=UPDATES * OWN_BATCH,
                                                          pano_views_seen=UPDATES * PANO_BATCH if arm != 'A' else 0)))
    print('TRAINED', arm, seed, round(time.time() - t0), 'final own loss %.3f' % log[-1]['loss_own'], flush=True)


def evaluate():
    import torch
    items = json.loads((OUT / 'items.json').read_text())
    dev = [i for i, it in enumerate(items) if it['split'] == 'dev']
    train_probe = random.Random(5).sample([i for i, it in enumerate(items) if it['split'] == 'train' and it['dataset'] == 'own'], 300)
    D = Data(items, None); results = {}
    for run in sorted(p for p in OUT.iterdir() if (p / 'final.pt').exists()):
        ck = torch.load(run / 'final.pt', weights_only=False); head = build_head(ck['k_out']).cuda(); head.load_state_dict(ck['ema']); head.eval()
        rows = []
        with torch.no_grad():
            for subset, ids in (('dev', dev), ('own_train_probe', train_probe)):
                for j in range(0, len(ids), 16):
                    b = ids[j:j + 16]; f, rgb, pts, kn = D.batch(b, 5, False)
                    xy = decode(head(f, rgb)[:, :5]).cpu().numpy()
                    for bi, i in enumerate(b):
                        it = items[i]; A = np.array(it['affine']); s = A[0, 0]
                        native = (xy[bi] - A[:, 2]) / s
                        gt = np.array(it['points']); known = np.array(it['known'])
                        err = np.linalg.norm(native - gt, axis=1)
                        # left/right swap proxy on paw pairs (1,2) and (3,4)
                        swap = [False] * 5
                        for a_, b_ in ((1, 2), (3, 4)):
                            if known[a_] and known[b_]:
                                swap[a_] = bool(err[a_] > np.linalg.norm(native[a_] - gt[b_]) and np.linalg.norm(gt[a_] - gt[b_]) > 0)
                                swap[b_] = bool(err[b_] > np.linalg.norm(native[b_] - gt[a_]))
                        box = it['box']; size = max(box[2] - box[0], box[3] - box[1])
                        for k in range(5):
                            if known[k]:
                                rows.append(dict(subset=subset, dataset=it['dataset'], session=it['session'], bottom=it['bottom'], k=k,
                                                 err_native=float(err[k]), err_ref=float(err[k] * (.64 if it['dataset'] == 'own' else 1)),
                                                 err_norm=float(err[k] / size), swap=swap[k]))
        results[run.name] = rows
        print('EVAL', run.name, flush=True)
    (OUT / 'eval_rows.json').write_text(json.dumps(results))
    summarize(results)


def summarize(results):
    def stats(rr):
        if not rr:
            return None
        n = np.array([r['err_norm'] for r in rr]); e = np.array([r['err_ref'] for r in rr])
        return dict(n=len(rr), median_norm=float(np.median(n)), pck05=float((n < .05).mean()), pck10=float((n < .10).mean()),
                    median_px=float(np.median(e)), cov32_ref=float((e <= 32).mean()), swap_rate=float(np.mean([r['swap'] for r in rr])))
    table = {}
    for run, rows in results.items():
        for subset in ('dev', 'own_train_probe'):
            for ds in ('own', 'pano'):
                for view in ('bottom', 'other'):
                    for part, ks in (('all', range(5)), ('paws', range(1, 5)), ('hind', (3, 4))):
                        rr = [r for r in rows if r['subset'] == subset and r['dataset'] == ds and r['bottom'] == (view == 'bottom') and r['k'] in ks]
                        s = stats(rr)
                        if s:
                            table.setdefault(f'{subset}/{ds}/{view}/{part}', {})[run] = s
    (OUT / 'summary.json').write_text(json.dumps(table, indent=1))
    for key in [k for k in table if k.startswith('dev/')]:
        print(key)
        for run, s in sorted(table[key].items()):
            print(f"   {run:12s} n={s['n']:5d} PCK@0.05 {100*s['pck05']:5.1f}%  PCK@0.10 {100*s['pck10']:5.1f}%  median {s['median_px']:6.1f}px  swap {100*s['swap_rate']:4.1f}%")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('step', choices=['prepare', 'cache', 'train', 'evaluate'])
    ap.add_argument('--arm'); ap.add_argument('--seed', type=int)
    a = ap.parse_args()
    {'prepare': prepare, 'cache': cache, 'train': lambda: train(a.arm, a.seed), 'evaluate': evaluate}[a.step]()
