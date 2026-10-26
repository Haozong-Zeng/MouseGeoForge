"""CPU fixtures for tools/head_first_kp_20261024.py (no release weights, data or GPU)."""
import copy
import math
import random
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import head_first_kp_20261024 as kp  # noqa: E402
import panomouse_stage2_kp_20261023 as hist  # noqa: E402
from vggt_omega.models.layers.vision_transformer import DinoVisionTransformer  # noqa: E402

DIM = 32


def tiny_encoder():
    torch.manual_seed(0)
    pe = DinoVisionTransformer(img_size=512, patch_size=16, embed_dim=DIM, depth=24, num_heads=2, ffn_ratio=1,
                               pos_embed_rope_normalize_coords='max', pos_embed_rope_dtype='fp32',
                               layerscale_init=1e-5, norm_layer='layernormbf16', n_storage_tokens=4, mask_k_bias=True)
    pe.init_weights()
    with torch.no_grad():  # LayerScale starts at 1e-5; make block updates visible in a few steps
        for n, p in pe.named_parameters():
            if n.endswith('gamma'):
                p.fill_(0.1)
    return pe.eval()


def small_cfg():
    cfg = copy.deepcopy(kp.design())
    pair = cfg['pair']
    pair['phases'] = [dict(first_update=1, last_update=2, unfrozen_blocks=[23]),
                      dict(first_update=3, last_update=4, unfrozen_blocks=[22, 23]),
                      dict(first_update=5, last_update=6, unfrozen_blocks=[20, 21, 22, 23])]
    pair.update(updates_per_branch=6, cosine_total_updates=6, head_warmup_updates=2, encoder_new_group_warmup_updates=2,
                head_peak_lr=1e-2, encoder_output_peak_lr=1e-2)
    cfg['task'].update(own_views_per_update=2, pano_views_per_update=2)
    return cfg


class FakeStore:
    def __init__(self, n=8, need_feats=True):
        g = torch.Generator().manual_seed(1)
        self.items = [dict(dataset='own' if i % 2 else 'pano', split='train', bottom=i % 4 < 2, points=[[0, 0]] * 5,
                           known=[True] * 5, box=[0, 0, 100, 100], session='S', group='G') for i in range(n)]
        self.pts = torch.rand(n, 5, 2, generator=g) * 400 + 50
        self.known = torch.rand(n, 5, generator=g) > 0.2
        self.known_np = self.known.numpy()
        self.scale = np.ones(n); self.shift = np.zeros((n, 2))
        self.crops = torch.randint(0, 255, (n, 512, 512, 3), dtype=torch.uint8, generator=g)
        self.feats = torch.randn(n, DIM, 32, 32, generator=g)
        self.need_feats = need_feats

    def feats_for(self, ids):
        if not self.need_feats:
            raise RuntimeError('cache refused')
        return self.feats[ids]

    def rgb_for(self, ids):
        return self.crops[ids].permute(0, 3, 1, 2).float() / 255


def trainer(mode, cfg, store, pe=None, seed=17):
    torch.manual_seed(seed)
    head = kp.build_head(5, DIM).eval()
    mean, std = torch.zeros(1, 3, 1, 1), torch.ones(1, 3, 1, 1)
    return kp.Trainer(mode, seed, cfg, store, head, pe, mean, std, device='cpu', autocast=False)


class HistoricalConventions(unittest.TestCase):
    def test_decode_matches_historical(self):
        torch.manual_seed(3)
        logits = torch.randn(3, 5, 128, 128)
        logits[0, 0, 0, 0] = 50; logits[1, 1, 127, 127] = 50; logits[2, 2, 0, 127] = 50  # clamped windows
        self.assertTrue(torch.allclose(kp.decode(logits), hist.decode(logits), atol=1e-5))

    def test_head_initialisation_matches_historical(self):
        torch.manual_seed(17); a = hist.build_head(5).state_dict()
        torch.manual_seed(17); b = kp.build_head(5).state_dict()
        self.assertEqual(a.keys(), b.keys())
        self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))

    def test_loss_matches_historical_and_missing_labels(self):
        torch.manual_seed(4)
        logits, pts = torch.randn(4, 5, 128, 128), torch.rand(4, 5, 2) * 512
        known = torch.rand(4, 5) > 0.5
        self.assertTrue(torch.allclose(kp.known_mean(kp.point_ce(logits, pts), known), hist.heatmap_loss(logits, pts, known)))
        logits.requires_grad_(True)
        zero = kp.known_mean(kp.point_ce(logits, pts), torch.zeros(4, 5, dtype=torch.bool))
        zero.backward()
        self.assertEqual(float(zero), 0.0)
        self.assertEqual(float(logits.grad.abs().sum()), 0.0)


class Plan(unittest.TestCase):
    def setUp(self):
        self.cfg, self.pe = kp.design(), tiny_encoder()

    def test_groups_unique_and_scoped(self):
        head = kp.build_head(5, DIM)
        plan = kp.ParamPlan('U', head, self.pe, self.cfg)
        names = [n for n, _ in plan.named()]
        self.assertEqual(len(names), len(set(names)))
        enc = {n for n in names if not n.startswith('head')}
        expected = {f'aggregator.patch_embed.{n}' for n, _ in self.pe.named_parameters()
                    if n.startswith(('norm.', 'blocks.20.', 'blocks.21.', 'blocks.22.', 'blocks.23.'))}
        self.assertEqual(enc, expected)
        self.assertEqual({n for n in names if n.startswith('head')}, {f'head.{n}' for n, _ in head.named_parameters()})
        for g in plan.groups:
            if g['name'].endswith('no_decay'):
                self.assertEqual(g['wd'], 0.0)
                self.assertTrue(all(n.endswith('bias') or '.norm' in n or 'norm' in n.split('.')[-2] or n.startswith('head')
                                    for n, _ in g['params']))
        f = kp.ParamPlan('F', head, None, self.cfg)
        self.assertTrue(all(g['name'].startswith('head') for g in f.groups))
        h0 = kp.ParamPlan('H0', head, None, self.cfg)
        self.assertEqual([(g['name'], g['wd']) for g in h0.groups], [('head', 0.01)])

    def test_fully_masked_bias_stays_frozen(self):
        pe = tiny_encoder()
        with torch.no_grad():
            pe.blocks[23].attn.qkv.bias_mask.zero_()  # as in the release checkpoint
        plan = kp.ParamPlan('U', kp.build_head(5, DIM), pe, self.cfg)
        names = {n for n, _ in plan.named()}
        self.assertEqual(plan.excluded, ['aggregator.patch_embed.blocks.23.attn.qkv.bias'])
        self.assertNotIn('aggregator.patch_embed.blocks.23.attn.qkv.bias', names)
        self.assertIn('aggregator.patch_embed.blocks.22.attn.qkv.bias', names)  # partly open mask: trained

    def test_layerwise_lrs_and_stages(self):
        plan = kp.ParamPlan('U', kp.build_head(5, DIM), self.pe, self.cfg)
        by = {g['name']: g for g in plan.groups}
        self.assertAlmostEqual(by['head_decay']['peak'], 3e-5)
        for b, peak in {23: 1e-6, 22: 6.5e-7, 21: 4.225e-7, 20: 2.74625e-7}.items():
            self.assertAlmostEqual(by[f'blocks.{b}_decay']['peak'], peak, places=15)
        self.assertEqual({b: by[f'blocks.{b}_decay']['start'] for b in (20, 21, 22, 23)}, {23: 1, 22: 2001, 21: 4001, 20: 4001})
        self.assertEqual(by['norm_no_decay']['start'], 1)
        env1 = plan.envelope(1)
        self.assertAlmostEqual(plan.lr(by['head_decay'], 1), 3e-5 / 200 * env1)
        self.assertAlmostEqual(plan.lr(by['blocks.23_decay'], 1), 1e-6 / 200 * env1)
        self.assertEqual(plan.lr(by['blocks.22_decay'], 2000), 0.0)
        self.assertAlmostEqual(plan.lr(by['blocks.22_decay'], 2001), 6.5e-7 / 200 * plan.envelope(2001))
        u = 4500
        lrs = [plan.lr(by[f'blocks.{b}_decay'], u) for b in (23, 22, 21, 20)]
        self.assertTrue(all(a > b for a, b in zip(lrs, lrs[1:])))
        self.assertTrue(all(x < plan.lr(by['head_decay'], u) for x in lrs))
        self.assertAlmostEqual(lrs[1] / lrs[0], 0.65)
        envs = [plan.envelope(u) for u in range(1, 6001)]
        self.assertTrue(all(a >= b for a, b in zip(envs, envs[1:])))
        self.assertAlmostEqual(envs[0], 1.0); self.assertGreaterEqual(min(envs), 0.1)
        self.assertEqual([plan.first_active(u) for u in (1, 2000, 2001, 4000, 4001, 6000)], [23, 23, 22, 22, 20, 20])


class Training(unittest.TestCase):
    def test_encode_matches_forward(self):
        pe = tiny_encoder(); x = torch.rand(2, 3, 512, 512)
        with torch.no_grad():
            ref = pe(x)['x_norm_patchtokens'].transpose(1, 2).reshape(2, DIM, 32, 32)
            got = kp.encode(pe, torch.zeros(1, 3, 1, 1), torch.ones(1, 3, 1, 1), x, 24, autocast=False)
        self.assertTrue(torch.equal(ref, got))

    def test_gradient_reach_frozen_bytes_and_ema(self):
        cfg, pe, store = small_cfg(), tiny_encoder(), FakeStore(need_feats=False)
        t = trainer('U', cfg, store, pe)
        before = {n: p.detach().clone() for n, p in pe.named_parameters()}
        t.step()  # update 1: block 23 + final norm + head
        grads = {n for n, p in pe.named_parameters() if p.grad is not None}
        self.assertTrue(grads and all(n.startswith(('blocks.23.', 'norm.')) for n in grads))
        self.assertTrue(any(n.startswith('blocks.23.') for n in grads))
        self.assertTrue(all(p.grad is not None for p in t.head.parameters()))
        for _ in range(5):
            t.step()
        opened = ('blocks.20.', 'blocks.21.', 'blocks.22.', 'blocks.23.', 'norm.')
        for n, p in pe.named_parameters():
            if n.startswith(opened):
                continue
            self.assertTrue(torch.equal(p, before[n]), n)
        self.assertFalse(torch.equal(pe.blocks[20].mlp.fc1.weight, before['blocks.20.mlp.fc1.weight']))
        t2 = trainer('U', cfg, FakeStore(need_feats=False), tiny_encoder())
        for _ in range(4):  # blocks 20/21 open at update 5: their EMA must still be the exact release bytes
            t2.step()
        self.assertTrue(torch.equal(t2.ema['aggregator.patch_embed.blocks.20.mlp.fc1.weight'], before['blocks.20.mlp.fc1.weight']))

    def test_u_refuses_final_token_cache(self):
        store = FakeStore(need_feats=False)
        with self.assertRaises(RuntimeError):
            store.feats_for([0])
        t = trainer('U', small_cfg(), store, tiny_encoder())
        t.step()  # online encoder path; the cache is never consulted
        self.assertEqual(t.u, 1)

    def test_resume_across_stage_boundaries_is_exact(self):
        cfg = small_cfg()
        a = trainer('U', cfg, FakeStore(need_feats=False), tiny_encoder())
        for _ in range(6):
            a.step()
        b = trainer('U', cfg, FakeStore(need_feats=False), tiny_encoder())
        for _ in range(2):
            b.step()
        state = copy.deepcopy(b.state_dict())
        c = trainer('U', cfg, FakeStore(need_feats=False), tiny_encoder())
        random.seed(999); torch.manual_seed(999)  # restored RNG must override this
        c.load_state_dict(state)
        for _ in range(4):
            c.step()
        for (n, p), (_, q) in zip(a.plan.named(), c.plan.named()):
            self.assertTrue(torch.equal(p, q), n)
        for n in a.ema:
            self.assertTrue(torch.equal(a.ema[n], c.ema[n]), n)
        self.assertEqual(a.s_own.state(), c.s_own.state())
        self.assertEqual({n: p.requires_grad for n, p in a.plan.named()}, {n: p.requires_grad for n, p in c.plan.named()})

    def test_h0_and_f_use_cache_and_train_head_only(self):
        for mode in ('H0', 'F'):
            t = trainer(mode, small_cfg(), FakeStore())
            before = {k: v.clone() for k, v in t.head.state_dict().items()}
            t.step()
            self.assertTrue(any(not torch.equal(v, before[k]) for k, v in t.head.state_dict().items()))
            self.assertTrue(all(n.startswith('head.') for n, _ in t.plan.named()))


class Protected(unittest.TestCase):
    def small_cfg(self):
        cfg = kp.protected_design()
        cfg['pair'].update(updates_per_branch=6, cosine_total_updates=6, head_warmup_updates=2, head_peak_lr=1e-2)
        cfg['protected'].update(warmup_updates=2, top_cap_lr=1e-2)
        cfg['task'].update(own_views_per_update=2, pano_views_per_update=2)
        return cfg

    def test_registered_values(self):
        cfg = kp.protected_design()
        self.assertEqual(cfg['pair']['updates_per_branch'], 12000)
        self.assertEqual(cfg['pair']['dev_eval_updates'], [4000, 8000, 12000])
        self.assertEqual((cfg['protected']['top_cap_lr'], cfg['protected']['layer_decay']), (1e-5, 0.65))
        self.assertEqual(kp.design()['pair']['updates_per_branch'], 6000)  # the parent pair is untouched

    def test_depth_groups_cover_encoder_except_structural_zeros(self):
        pe = tiny_encoder()
        with torch.no_grad():
            pe.blocks[5].attn.qkv.bias_mask.zero_()
        cfg = kp.protected_design()
        plan = kp.ParamPlan('UP', kp.build_head(5, DIM), pe, cfg)
        enc = {n for n, _ in plan.named() if not n.startswith('head')}
        every = {f'aggregator.patch_embed.{n}' for n, _ in pe.named_parameters()}
        self.assertEqual(set(plan.excluded), {'aggregator.patch_embed.mask_token', 'aggregator.patch_embed.blocks.5.attn.qkv.bias'})
        self.assertEqual(enc | set(plan.excluded), every)
        g = {x['name']: x for x in plan.groups}
        self.assertAlmostEqual(g['depth24_decay']['peak'], 1e-5)
        self.assertAlmostEqual(g['depth23_decay']['peak'], 6.5e-6)
        self.assertAlmostEqual(g['depth00_decay']['peak'], 1e-5 * 0.65 ** 24)
        self.assertIn('aggregator.patch_embed.norm.weight', [n for n, _ in g['depth24_no_decay']['params']])
        self.assertEqual({n for n, _ in g['depth00_no_decay']['params']},
                         {'aggregator.patch_embed.cls_token', 'aggregator.patch_embed.storage_tokens', 'aggregator.patch_embed.patch_embed.proj.bias'})
        top = g['depth24_decay']
        self.assertAlmostEqual(plan.lr(top, 1), 3e-5 / 200 * plan.envelope(1))  # guard: never above the warming head
        self.assertAlmostEqual(plan.lr(top, 201), 1e-5 * 0.1 * 10 ** 0.1 * plan.envelope(201))
        self.assertAlmostEqual(plan.lr(top, 1001), 1e-5 * 10 ** -0.5 * plan.envelope(1001))
        self.assertAlmostEqual(plan.lr(top, 2001), 1e-5 * plan.envelope(2001))
        self.assertAlmostEqual(plan.lr(g['depth23_decay'], 5000) / plan.lr(top, 5000), 0.65)
        self.assertTrue(all(plan.lr(x, u) <= plan.lr(g['head_decay'], u) + 1e-18 for x in plan.groups for u in (1, 5, 50, 3000, 12000)))
        self.assertEqual(plan.first_active(1), 0)

    def test_embedding_and_every_block_get_gradients(self):
        pe = tiny_encoder()
        t = trainer('UP', self.small_cfg(), FakeStore(need_feats=False), pe)
        t.step()
        for n, p in pe.named_parameters():
            if n == 'mask_token':
                self.assertIsNone(p.grad)
            else:
                self.assertIsNotNone(p.grad, n)
        self.assertGreater(float(pe.patch_embed.proj.weight.grad.abs().sum()), 0)
        self.assertGreater(float(pe.blocks[0].mlp.fc1.weight.grad.abs().sum()), 0)

    def test_resume_is_exact(self):
        cfg = self.small_cfg()
        a = trainer('UP', cfg, FakeStore(need_feats=False), tiny_encoder())
        for _ in range(5):
            a.step()
        b = trainer('UP', cfg, FakeStore(need_feats=False), tiny_encoder())
        for _ in range(2):
            b.step()
        state = copy.deepcopy(b.state_dict())
        c = trainer('UP', cfg, FakeStore(need_feats=False), tiny_encoder())
        c.load_state_dict(state)
        for _ in range(3):
            c.step()
        for (n, p), (_, q) in zip(a.plan.named(), c.plan.named()):
            self.assertTrue(torch.equal(p, q), n)


class Sweep(unittest.TestCase):
    def test_variants_share_budget_and_scale_schedules(self):
        import head_first_kp_sweep_20261024 as sweep
        self.assertTrue({'F_b16', 'F_b32', 'UP_b16', 'UP_b32', 'U_b16', 'U_b32'} <= set(sweep.VARIANTS))
        for name, v in sweep.VARIANTS.items():
            cfg = sweep.variant_cfg(v)
            budget = 2 * sweep.BUDGET if name == 'UX_flat_24k' else sweep.BUDGET  # the convergence check ran twice the budget
            self.assertEqual(2 * v['per_domain'] * cfg['pair']['updates_per_branch'], budget, name)
            self.assertEqual(cfg['pair']['cosine_total_updates'], v['updates'])
            self.assertEqual(cfg['pair']['dev_eval_updates'][-1], v['updates'])
            head_lr = v.get('explore', {}).get('head_peak_lr', 3e-5)  # UX_head1e-4 and the flat 1e-4/3e-4 variants raise the head's rate
            self.assertEqual(cfg['pair']['head_peak_lr'], head_lr, name)
        u32 = sweep.variant_cfg(sweep.VARIANTS['U_b32'])
        self.assertEqual(u32['pair']['phases'], kp.design()['pair']['phases'])  # the registered U, larger batch
        self.assertEqual([p['first_update'] for p in sweep.variant_cfg(sweep.VARIANTS['U_b16'])['pair']['phases']], [1, 4001, 8001])
        self.assertEqual(sweep.variant_cfg(sweep.VARIANTS['UP_b16'])['protected']['warmup_updates'], 2000)
        self.assertEqual(sweep.variant_cfg(sweep.VARIANTS['UP_b32'])['protected']['warmup_updates'], 1000)
        plan = kp.ParamPlan('UP', kp.build_head(5, DIM), tiny_encoder(), sweep.variant_cfg(sweep.VARIANTS['UP_b32']))
        self.assertEqual(plan.ramp, (0.1, 1000))


class Explore(unittest.TestCase):
    def cfg(self, **explore):
        cfg = kp.protected_design()
        cfg['explore'] = explore
        return cfg

    def test_goal_metrics(self):
        rows = [dict(err_norm=0.01, err_ref=float(e), swap=False) for e in range(1, 101)]
        s = kp.stats(rows)
        self.assertEqual(s['p95_ref'], 95.0)  # nearest rank: ceil(0.95 * 100) = 95th value
        self.assertAlmostEqual(s['cov32_ref'], 0.32)

    def test_top8_scope(self):
        pe = tiny_encoder()
        plan = kp.ParamPlan('UX', kp.build_head(5, DIM), pe, self.cfg(blocks=list(range(16, 24)), embed=False))
        enc = {n for n, _ in plan.named() if not n.startswith('head')}
        self.assertTrue(enc and all(n.startswith(tuple(f'aggregator.patch_embed.blocks.{b}.' for b in range(16, 24)))
                                    or n.startswith('aggregator.patch_embed.norm.') for n in enc))
        self.assertEqual(plan.first_active(1), 16)
        self.assertFalse(plan.train_embed)
        self.assertFalse(pe.blocks[3].mlp.fc1.weight.requires_grad)
        self.assertFalse(pe.patch_embed.proj.weight.requires_grad)

    def test_parameter_filters_and_guard(self):
        for filt, must in (('attn', '.attn.'), ('mlp', '.mlp.')):
            plan = kp.ParamPlan('UX', kp.build_head(5, DIM), tiny_encoder(), self.cfg(filter=filt, embed=False))
            blocks = [n for n, _ in plan.named() if '.blocks.' in n]
            self.assertTrue(blocks and all(must in n for n in blocks), filt)
        plan = kp.ParamPlan('UX', kp.build_head(5, DIM), tiny_encoder(),
                            self.cfg(filter='norm_ls', embed=False, top_cap_lr=1e-4, layer_decay=1.0, guard=False))
        blocks = [n for n, _ in plan.named() if '.blocks.' in n]
        self.assertTrue(blocks and all('.norm' in n or n.endswith('gamma') for n in blocks))
        g = {x['name']: x for x in plan.groups}
        self.assertGreater(plan.lr(g['depth01_no_decay'], 6000), plan.lr(g['head_decay'], 6000))  # guard off

    def test_lora_starts_identical_and_trains_only_adapters(self):
        pe = tiny_encoder(); x = torch.rand(2, 3, 512, 512); z = torch.zeros(1, 3, 1, 1); o = torch.ones(1, 3, 1, 1)
        with torch.no_grad():
            before = kp.encode(pe, z, o, x, 24, autocast=False)
            kp.add_lora(pe, rank=4, alpha=4)
            after = kp.encode(pe, z, o, x, 24, autocast=False)
        self.assertTrue(torch.allclose(before, after, atol=1e-6))
        base = pe.blocks[10].attn.qkv.base.weight.detach().clone()
        cfg = self.cfg(filter='lora', embed=False, top_cap_lr=1e-2, layer_decay=1.0, guard=False)
        cfg['pair'].update(updates_per_branch=6, cosine_total_updates=6, head_warmup_updates=2, head_peak_lr=1e-2)
        cfg['protected'].update(warmup_updates=2)
        cfg['task'].update(own_views_per_update=2, pano_views_per_update=2)
        t = trainer('UX', cfg, FakeStore(need_feats=False), pe)
        self.assertTrue(all('lora_' in n or n.startswith('aggregator.patch_embed.norm.')  # final norm trains too
                            for n, _ in t.plan.named() if not n.startswith('head')))
        t.step(); t.step()
        self.assertTrue(torch.equal(pe.blocks[10].attn.qkv.base.weight, base))
        self.assertGreater(float(pe.blocks[10].attn.qkv.lora_b.abs().sum()), 0)

    def test_frozen_layers_trainer_and_wave2_cfgs(self):
        import head_first_kp_sweep_20261024 as sweep
        for name, v in sweep.WAVE2.items():
            cfg = sweep.variant_cfg(v)
            self.assertEqual(2 * v['per_domain'] * v['updates'], sweep.BUDGET, name)
        self.assertEqual(sweep.variant_cfg(sweep.WAVE2['UX_head1e-4'])['pair']['head_peak_lr'], 1e-4)
        self.assertNotIn('head_peak_lr', sweep.variant_cfg(sweep.WAVE2['UX_head1e-4'])['explore'])
        self.assertEqual(sweep.variant_cfg(sweep.WAVE2['HS_L23'])['h0']['cosine_total_updates'], 12000)
        cfg = sweep.variant_cfg(sweep.WAVE2['HS_L20to23'])
        cfg['task'].update(own_views_per_update=2, pano_views_per_update=2)
        pe = tiny_encoder().requires_grad_(False)
        torch.manual_seed(1); head = kp.build_head(5, 2 * DIM).eval()
        t = sweep.FrozenLayersTrainer('H0', 17, cfg, FakeStore(need_feats=False), head, pe, torch.zeros(1, 3, 1, 1),
                                      torch.ones(1, 3, 1, 1), device='cpu', autocast=False, layers=[22, 23])
        self.assertEqual(tuple(t.features([0, 1], 1).shape), (2, 2 * DIM, 32, 32))
        before = {k: v.clone() for k, v in head.state_dict().items()}
        t.step()
        self.assertTrue(any(not torch.equal(v, before[k]) for k, v in head.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in pe.parameters()))


class Gates(unittest.TestCase):
    def test_plateau(self):
        conv = kp.design()['h0']['convergence']
        flat = [dict(update=u, ce=1.0, bottom_paw_pck05=0.9, bottom_paw_swap=0.05) for u in range(4000, 6001, 500)]
        self.assertTrue(kp.plateau(flat, conv))
        self.assertFalse(kp.plateau(flat[:-1], conv))  # last check at 5,500 < minimum
        moving = [dict(r, bottom_paw_pck05=0.9 + 0.001 * i) for i, r in enumerate(flat)]
        self.assertFalse(kp.plateau(moving, conv))  # 0.4 pp span
        dropping = [dict(r, ce=1.0 - 0.003 * i) for i, r in enumerate(flat)]
        self.assertFalse(kp.plateau(dropping, conv))  # 1.2 % CE reduction

    def test_sampler_state_round_trip(self):
        s = kp.Sampler(range(10), 5); s.take(3)
        t = kp.Sampler(range(10), 0); t.restore(copy.deepcopy(s.state()))
        self.assertEqual([s.take(4) for _ in range(6)], [t.take(4) for _ in range(6)])  # crosses refills

    def test_endpoint_weights_are_ema_with_release_frozen(self):
        import head_first_kp_sweep_20261024 as sweep
        pe = tiny_encoder(); cfg = kp.protected_design()
        cfg['explore'] = dict(layer_decay=1.0, top_cap_lr=1e-2)  # the retrain recipe's shape at a test-sized rate
        cfg['pair'].update(updates_per_branch=6, cosine_total_updates=6, head_warmup_updates=2, head_peak_lr=1e-2,
                           dev_eval_updates=[6])
        cfg['protected']['warmup_updates'] = 2
        cfg['task'].update(own_views_per_update=2, pano_views_per_update=2)
        release = {k: v.clone() for k, v in pe.state_dict().items()}
        t = trainer('UX', cfg, FakeStore(need_feats=False), pe)
        for _ in range(3):
            t.step()
        live = {n: p.detach().clone() for n, p in t.plan.named()}
        w = sweep.endpoint_weights(t, 'UX_test', 17, pe, t.head, cfg, None)
        self.assertEqual((w['update'], w['weights'], w['encoder_prefix']), (3, 'ema', 'aggregator.patch_embed'))
        for n, p in t.plan.named():  # the call leaves the live parameters untouched
            self.assertTrue(torch.equal(p.detach(), live[n]), n)
        local = lambda n: n.split('aggregator.patch_embed.')[-1]  # noqa: E731
        trained = [n for n, _ in t.plan.named() if not n.startswith('head')]
        self.assertTrue(trained)
        for n in trained:
            self.assertTrue(torch.equal(w['ema_encoder'][local(n)], t.ema[n].cpu()), n)
        self.assertTrue(any(not torch.equal(w['ema_encoder'][local(n)], release[local(n)]) for n in trained))
        for n in t.plan.excluded:  # mask_token and the masked qkv biases keep their release bytes
            self.assertTrue(torch.equal(w['ema_encoder'][local(n)], release[local(n)]), n)
        self.assertEqual(set(w['ema_encoder']), set(release))  # the whole encoder state, buffers included
        self.assertTrue(any(not torch.equal(w['raw_head'][k], w['ema_head'][k]) for k in w['raw_head']))


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
