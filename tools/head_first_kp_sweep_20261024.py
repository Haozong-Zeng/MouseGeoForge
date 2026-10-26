"""Weight-free exploratory sweep over keypoint continuation recipes (2026-10-24).

Wave 1: encoder recipe {F, U, UP} x batch {16, 32}. Wave 2 (batch 16, same budget): UX
variants of the protected schedule -- unfreezing depth (top 8 / top 12 blocks), layer decay
(0.75 / 0.85 / 1.0), cap (3e-6 / 3e-5), head LR 1e-4, top-4 blocks at the protected rate, parameter subsets (norm+LayerScale,
attention, MLP; LoRA r16 is defined but not run: the user prefers direct weight training) -- and head-from-scratch feature interfaces on the frozen encoder
(HS: layer 23 control vs layers 20-23 concatenated).

Crosses encoder recipe {F frozen, U progressive last-4-block, UP lp_anyup protected} with
batch {8 own + 8 Pano, 16 own + 16 Pano}; every variant starts from the converged H0 EMA
head of the seed and has the same sample budget (192,000 crops: 12,000 updates at batch 16,
6,000 at batch 32), with the same learning rates. U's stage boundaries sit at 1/3 and 2/3 of
the run, UP's warm-up covers 1/6 of it, as in their registered versions.

Nothing is checkpointed (user request: no weight storage until a best recipe is chosen).
Each variant trains in memory and leaves only JSON: losses/gradient norms/LRs every 50
updates, encoder drift and DEV summaries + per-point rows (EMA and raw) at 1/3, 2/3 and the
end, timing and peak memory. DEV is used to compare many variants here, so this is
exploratory selection: the chosen recipe must be re-run with weights and confirmed on more
seeds before it is reported as an improvement.

    python tools/head_first_kp_sweep_20261024.py run --variant UP_b16 [--seed 17]
    python tools/head_first_kp_sweep_20261024.py table [--seed 17]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import head_first_kp_20261024 as kp  # noqa: E402

SWEEP = kp.RUNS/'sweep'
RETRAIN = kp.RUNS/'retrain'  # the chosen recipe re-run with its endpoint weights saved (run --save)
BUDGET = 192000  # crops per variant
VARIANTS = {f'{mode}_b{2 * per}': dict(mode=mode, per_domain=per, updates=BUDGET // (2 * per))
            for mode in ('F', 'U', 'UP') for per in (8, 16)}


def ux(**explore):
    return dict(mode='UX', per_domain=8, updates=BUDGET // 16, explore=explore)


WAVE2 = {
    'UX_top4': ux(blocks=[20, 21, 22, 23], embed=False),  # U's scope at the protected rate
    'UX_top8': ux(blocks=list(range(16, 24)), embed=False),
    'UX_top12': ux(blocks=list(range(12, 24)), embed=False),
    'UX_llrd075': ux(layer_decay=0.75),
    'UX_llrd085': ux(layer_decay=0.85),
    'UX_llrd100': ux(layer_decay=1.0),
    'UX_cap3e-6': ux(top_cap_lr=3e-6),
    'UX_cap3e-5': ux(top_cap_lr=3e-5),
    'UX_head1e-4': ux(head_peak_lr=1e-4),
    'UX_normls': ux(filter='norm_ls', embed=False, top_cap_lr=1e-4, layer_decay=1.0, guard=False),
    'UX_attn': ux(filter='attn', embed=False),
    'UX_mlp': ux(filter='mlp', embed=False),
    'UX_lora16': ux(filter='lora', embed=False, lora=dict(rank=16, alpha=16), top_cap_lr=1e-4, layer_decay=1.0, guard=False),
    'HS_L23': dict(mode='HS', per_domain=8, updates=BUDGET // 16, layers=[23]),
    'HS_L20to23': dict(mode='HS', per_domain=8, updates=BUDGET // 16, layers=[20, 21, 22, 23]),
}
VARIANTS.update(WAVE2)
# wave 3: follows UX_llrd100 (flat rate, seed 17), whose gain came with 1-1.5% drift of the embedding and lower blocks
WAVE3 = {
    'UX_flat3e-6': ux(layer_decay=1.0, top_cap_lr=3e-6),
    'UX_flat3e-5': ux(layer_decay=1.0, top_cap_lr=3e-5),
    'UX_low12': ux(blocks=list(range(12)), final_norm=False, layer_decay=1.0),  # embedding + blocks 1-12 only
    'UX_flat_24k': dict(ux(layer_decay=1.0), updates=2 * BUDGET // 16),  # convergence check: twice the budget
}
VARIANTS.update(WAVE3)
# wave 4: UX_flat3e-5 (cap = head peak) led UX_llrd100 on seed 17; is the encoder rate saturated? Raise head and cap together
WAVE4 = {
    'UX_flat1e-4': ux(layer_decay=1.0, top_cap_lr=1e-4, head_peak_lr=1e-4),
    'UX_flat3e-4': ux(layer_decay=1.0, top_cap_lr=3e-4, head_peak_lr=3e-4),  # where does the rate stop helping?
}
VARIANTS.update(WAVE4)


class FrozenLayersTrainer(kp.Trainer):
    """Head-only training on frozen intermediate-layer features computed online (HS variants)."""

    def __init__(self, *args, layers, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = layers

    def _features(self, ids):
        import torch
        rgb = self.store.rgb_for(ids)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.autocast):
            outs = self.pe.get_intermediate_layers((rgb - self.mean) / self.std, n=self.layers, norm=True, reshape=True)
        return torch.cat(outs, 1).half().float()

    def features(self, ids, u):
        return self._features(ids)

    def eval_features(self, ids):
        return self._features(ids)
KEYS = ('pano/bottom/paws', 'pano/other/paws', 'own/bottom/paws', 'own/other/paws', 'pano/bottom/hind', 'own/all/all', 'pano/all/all')


def variant_cfg(v):
    cfg = kp.protected_design() if v['mode'] in ('UP', 'UX') else kp.design()
    n = v['updates']
    cfg['task'].update(own_views_per_update=v['per_domain'], pano_views_per_update=v['per_domain'])
    cfg['pair'].update(updates_per_branch=n, cosine_total_updates=n, dev_eval_updates=[n // 3, 2 * n // 3, n])
    if v['mode'] == 'U':
        a, b = n // 3, 2 * n // 3
        cfg['pair']['phases'] = [dict(first_update=1, last_update=a, unfrozen_blocks=[23]),
                                 dict(first_update=a + 1, last_update=b, unfrozen_blocks=[22, 23]),
                                 dict(first_update=b + 1, last_update=n, unfrozen_blocks=[20, 21, 22, 23])]
    if v['mode'] in ('UP', 'UX'):
        cfg['protected']['warmup_updates'] = n // 6
    if v['mode'] == 'UX':
        explore = dict(v['explore'])
        if 'head_peak_lr' in explore:
            cfg['pair']['head_peak_lr'] = explore.pop('head_peak_lr')
        cfg['explore'] = explore
    if v['mode'] == 'HS':  # head from scratch: the H0 recipe, fixed budget instead of the plateau stop
        cfg['h0'].update(cosine_total_updates=n)
    return cfg


def endpoint_weights(t, name, seed, pe, head, cfg, dev_summary):
    """EMA endpoint of a retrain: head and encoder state dicts (EMA values for the trained tensors, release bytes
    for the rest), the raw head, and the provenance needed to load the encoder back into VGGT-Omega."""
    def cpu(sd):
        return {k: v.detach().cpu().clone() for k, v in sd.items()}
    raw_head = cpu(head.state_dict())
    t.swap_ema()
    try:
        ema_head = cpu(head.state_dict())
        ema_encoder = cpu(pe.state_dict()) if pe is not None else None
    finally:
        t.swap_ema()
    return dict(variant=name, seed=seed, update=t.u, weights='ema', ema_head=ema_head, raw_head=raw_head,
                ema_encoder=ema_encoder, encoder_prefix='aggregator.patch_embed', release_sha256=kp.RELEASE_SHA,
                excluded_frozen=t.plan.excluded,
                recipe=dict(pair=cfg['pair'], protected=cfg.get('protected'), explore=cfg.get('explore')),
                dev_summary=dev_summary, clipped_head_updates=t.clipped, clipped_encoder_updates=t.clipped_encoder,
                tool_sha256=kp.sha(Path(__file__)), kp_tool_sha256=kp.sha(Path(kp.__file__)))


def run(name, seed, save=False):
    import torch
    v = VARIANTS[name]; cfg = variant_cfg(v); mode = v['mode']
    dest = (RETRAIN if save else SWEEP)/f'{name}_seed{seed}'
    if (dest/'DONE.json').exists():
        print('done already:', dest); return
    dest.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    items = kp.load_items()
    online = mode in ('U', 'UP', 'UX', 'HS')
    pe = mean = std = None
    if online:
        _, pe, mean, std = kp.release_parts()
    lora = cfg.get('explore', {}).get('lora')
    if lora:
        kp.add_lora(pe, rank=lora['rank'], alpha=lora['alpha'])
    store = kp.Store(items, 'cuda', need_feats=not online)
    torch.manual_seed(seed)
    if mode == 'HS':
        head = kp.build_head(5, feat_dim=1024 * len(v['layers'])).cuda().eval()
        t = FrozenLayersTrainer('H0', seed, cfg, store, head, pe, mean, std, layers=v['layers'])
    else:
        head = kp.build_head(5).cuda()
        head.load_state_dict(torch.load(kp.RUNS/f'H0_seed{seed}/endpoint.pt', weights_only=False)['ema_head'])
        head.eval()
        t = kp.Trainer(mode, seed, cfg, store, head, pe, mean, std)
    release_hashes = release_copy = None
    if online and mode != 'HS':
        release_hashes = kp.param_hashes(pe, 'aggregator.patch_embed')
        release_copy = {n: p.detach().cpu().clone() for n, p in t.plan.named() if not n.startswith('head')}
    if save:  # exercise the endpoint save before spending the budget; at update 0 the EMA equals the start weights
        pf = dest/'endpoint.preflight.pt'
        torch.save(endpoint_weights(t, name, seed, pe, head, cfg, None), pf)
        w = torch.load(pf, weights_only=False)
        ok = all(torch.equal(w['ema_head'][k], x.detach().cpu()) for k, x in head.state_dict().items()) and (
            pe is None or all(torch.equal(w['ema_encoder'][k], x.detach().cpu()) for k, x in pe.state_dict().items()))
        size = pf.stat().st_size; pf.unlink()
        if not ok:
            raise RuntimeError('endpoint preflight: the reloaded weights differ from the start weights')
        print(f'endpoint preflight ok: {size / 2**30:.2f} GB per endpoint', flush=True)
    kp.write(dest/'run_protocol.json', dict(
        variant=name, seed=seed, mode=mode, per_domain_batch=v['per_domain'], updates=v['updates'], crops=BUDGET,
        pair={k: cfg['pair'][k] for k in ('head_peak_lr', 'head_warmup_updates', 'cosine_total_updates', 'cosine_floor',
                                          'weight_decay', 'grad_clip', 'ema_decay', 'dev_eval_updates')},
        phases=cfg['pair']['phases'] if mode == 'U' else None,
        protected=cfg.get('protected') if mode in ('UP', 'UX') else None, explore=cfg.get('explore'),
        layers=v.get('layers'), h0_schedule=cfg['h0'] if mode == 'HS' else None,
        groups=[dict(name=g['name'], peak=g['peak'], start=g['start'], tensors=len(g['params'])) for g in t.plan.groups],
        excluded_frozen=t.plan.excluded, weights_saved=save, data_placement=store.placement,
        start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), host=os.uname().nodename,
        slurm_job=os.environ.get('SLURM_JOB_ID'), tool_sha256=kp.sha(Path(__file__)), kp_tool_sha256=kp.sha(Path(kp.__file__)),
        selection_note='exploratory: DEV compares many variants; confirm the chosen recipe with weights on more seeds'))
    dev_ids = [i for i, it in enumerate(items) if it['split'] == 'dev']
    points = set(cfg['pair']['dev_eval_updates'])
    last_summary = None
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time(); train_seconds = 0.0
    while t.u < v['updates']:
        s0 = time.time(); t.step(); train_seconds += time.time() - s0
        if t.u in points:
            if release_hashes is not None:
                kp.write(dest/f'audit_u{t.u:05d}.json', kp.stage_audit(t, release_hashes, release_copy))
            for kind in ('ema', 'raw'):
                rows = t.evaluate(dev_ids, use_ema=kind == 'ema')
                summary = kp.summarize(rows)
                if kind == 'ema':
                    last_summary = summary
                kp.write(dest/f'dev_u{t.u:05d}_{kind}.json', dict(update=t.u, weights=kind, summary=summary, rows=rows))
            kp.write(dest/'train_log.json', dict(log=t.log, exposure=dict(t.exposure)))
            print(f'{name} seed{seed} u{t.u} DEV scored, {round(time.time() - t0)}s', flush=True)
    kp.write(dest/'train_log.json', dict(log=t.log, exposure=dict(t.exposure)))
    endpoint = None
    if save:  # after the final evaluation, so the saved EMA is the scored one
        tmp = dest/'endpoint.pt.tmp'
        torch.save(endpoint_weights(t, name, seed, pe, head, cfg, last_summary), tmp); os.replace(tmp, dest/'endpoint.pt')
        endpoint = dict(path=str(dest/'endpoint.pt'), bytes=(dest/'endpoint.pt').stat().st_size, sha256=kp.sha(dest/'endpoint.pt'))
        print('saved', endpoint, flush=True)
    kp.write_new(dest/'DONE.json', dict(variant=name, seed=seed, updates=t.u, weights_saved=save, endpoint=endpoint,
                                        seconds=round(time.time() - t0), train_seconds_per_update=train_seconds / t.u,
                                        crops_per_second=BUDGET / train_seconds, peak_memory_gb=torch.cuda.max_memory_allocated() / 2**30,
                                        clipped_head_updates=t.clipped, clipped_encoder_updates=t.clipped_encoder,
                                        finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    print('DONE', name, seed, flush=True)


def table(seed, base=SWEEP):
    rows = {}
    for name, v in VARIANTS.items():
        dest = base/f'{name}_seed{seed}'
        if not (dest/'DONE.json').exists():
            rows[name] = None; continue
        done = json.loads((dest/'DONE.json').read_text())
        end = v['updates']
        out = dict(done=done)
        for kind in ('ema', 'raw'):  # recomputed from the saved rows, so every variant has the same metrics
            s = kp.summarize(json.loads((dest/f'dev_u{end:05d}_{kind}.json').read_text())['rows'])
            out[kind] = {k: dict(pck05=s[k]['pck05'], swap=s[k]['swap_rate'], median_px=s[k]['median_px'],
                                 cov32_ref=s[k]['cov32_ref'], p95_ref=s[k]['p95_ref']) for k in KEYS if k in s}
        audit = dest/f'audit_u{end:05d}.json'
        out['drift'] = json.loads(audit.read_text())['relative_drift_from_release'] if audit.exists() else None
        rows[name] = out
    kp.write(base/f'table_seed{seed}.json', dict(seed=seed, variants=rows, budget_crops=BUDGET,
                                                  note='retrain of the chosen recipe with endpoint weights' if base == RETRAIN
                                                  else 'exploratory selection on DEV; weights were not saved'))
    print(f'{"variant":12s} {"pano/bot PCK":>12s} {"swap":>6s} {"pano/oth":>8s} {"own/bot":>7s} {"own/oth":>7s} '
          f'{"own cov32":>9s} {"own P95":>7s} {"pano cov32":>10s} {"pano P95":>8s} {"s/upd":>6s} {"GB":>5s}')
    for name, r in rows.items():
        if r is None:
            print(f'{name:12s} (not finished)'); continue
        e = r['ema']
        print(f"{name:12s} {100*e['pano/bottom/paws']['pck05']:12.2f} {100*e['pano/bottom/paws']['swap']:6.2f} "
              f"{100*e['pano/other/paws']['pck05']:8.2f} {100*e['own/bottom/paws']['pck05']:7.2f} {100*e['own/other/paws']['pck05']:7.2f} "
              f"{100*e['own/all/all']['cov32_ref']:9.2f} {e['own/all/all']['p95_ref']:7.1f} "
              f"{100*e['pano/all/all']['cov32_ref']:10.2f} {e['pano/all/all']['p95_ref']:8.1f} "
              f"{r['done']['train_seconds_per_update']:6.3f} {r['done']['peak_memory_gb']:5.1f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['run', 'table'])
    ap.add_argument('--variant', choices=sorted(VARIANTS))
    ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--save', action='store_true', help='run: retrain under retrain/ and save the EMA endpoint weights')
    ap.add_argument('--retrain', action='store_true', help='table: the retrain/ runs instead of the sweep')
    a = ap.parse_args()
    run(a.variant, a.seed, a.save) if a.step == 'run' else table(a.seed, RETRAIN if a.retrain else SWEEP)
