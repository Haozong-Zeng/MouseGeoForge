"""CPU fixtures for tools/depth_probe_20261026.py (no release weights, data or GPU)."""
import copy
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
try:  # cv2 before torch: the two OpenMP runtimes crash when cv2 loads second
    if os.environ.get('MGF_NO_CV2'):  # diagnostics: run without cv2 in the process
        raise ImportError
    import cv2
    cv2.setNumThreads(0)
except ImportError:  # the crop test then skips
    cv2 = None
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]/'tools'))
sys.path.insert(0, str(HERE.parent))
import depth_probe_20261026 as dp  # noqa: E402
import head_first_kp_20261024 as kp  # noqa: E402
from test_head_first_kp_20261024 import DIM, tiny_encoder  # noqa: E402

MEAN, STD = torch.zeros(1, 3, 1, 1), torch.ones(1, 3, 1, 1)


def fake_items(n=8):
    items = []
    for i in range(n):
        role = 'bottom' if i % 2 == 0 else 'other'
        items.append(dict(id=i, scene=f's{i}', motion='climb', source_group='g', rig=f'rig{i % 2}', split='train' if i < 6 else 'val',
                          pool='crossed', group='train' if i < 6 else 'unseen_rig', view=0, physical_role='bottom' if role == 'bottom' else 'side',
                          role=role, image='', truth='', depth=None, mask=None, probe=i < 2))
    return items


def fake_store(n=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    crops = (torch.rand((n, 512, 512, 3), generator=g) * 255).to(torch.uint8).numpy()
    depth = (torch.rand((n, 256, 256), generator=g) * 100 + 50).numpy().astype(np.float32)
    flags = np.full((n, 256, 256), 3, np.uint8); flags[:, :16] = 0
    return dp.Store(fake_items(n), crops, depth, flags, [True] * n, 'cpu')


def small_cfg():
    cfg = copy.deepcopy(dp.CFG)
    cfg.update(updates=4, warmup=1, batch_bottom=2, batch_other=2, peak_lr=1e-2)
    return cfg


class Loss(unittest.TestCase):
    def test_silog_is_scale_invariant(self):
        g = torch.Generator().manual_seed(1)
        pred = torch.randn((3, 1, 8, 8), generator=g); z = torch.rand((3, 8, 8), generator=g) * 90 + 10
        mask = torch.rand((3, 8, 8), generator=g) > 0.3
        base, used = dp.silog(pred, z, mask, anchor=0.0)
        self.assertEqual(used, 3)
        self.assertTrue(torch.allclose(base, dp.silog(pred + 3.0, z, mask, anchor=0.0)[0], atol=1e-5))
        self.assertTrue(torch.allclose(base, dp.silog(pred, z * 7.0, mask, anchor=0.0)[0], atol=1e-5))
        self.assertGreater(float(dp.silog(pred + 3.0, z, mask, anchor=0.01)[0]), float(base))  # the anchor sees the offset

    def test_silog_skips_views_without_targets(self):
        pred = torch.zeros((2, 1, 8, 8), requires_grad=True); z = torch.full((2, 8, 8), 5.0)
        mask = torch.zeros((2, 8, 8), dtype=torch.bool); mask[0, :4] = True
        loss, used = dp.silog(pred, z, mask, anchor=0.0)
        self.assertEqual(used, 1); self.assertAlmostEqual(float(loss.detach()), 0.0, places=5)
        loss, used = dp.silog(pred, z, torch.zeros_like(mask), anchor=0.0)
        self.assertEqual(used, 0); self.assertTrue(loss.requires_grad)


class Metrics(unittest.TestCase):
    def test_perfect_shape_scores_zero_after_alignment(self):
        g = torch.Generator().manual_seed(2)
        z = torch.rand((16, 16), generator=g) * 100 + 20; mask = torch.rand((16, 16), generator=g) > 0.2
        r = dp.view_metrics(torch.log(z) + 1.3, z, mask)
        self.assertEqual(r['n'], int(mask.sum()))
        self.assertLess(r['absrel'], 1e-5); self.assertEqual(r['d105'], 1.0); self.assertLess(r['silog'], 1e-5)
        self.assertAlmostEqual(r['scale'], math.exp(-1.3), places=5)
        self.assertIsNone(dp.view_metrics(torch.log(z), z, mask & (torch.arange(16)[:, None] < 1)))

    def test_alignment_uses_median_log_residual(self):
        g = torch.Generator().manual_seed(3)
        z = torch.rand((8, 8), generator=g) * 100 + 20; pred = torch.randn((8, 8), generator=g)
        mask = torch.ones((8, 8), dtype=torch.bool)
        r = dp.view_metrics(pred, z, mask)
        s = torch.exp((torch.log(z) - pred).median())
        self.assertAlmostEqual(r['scale'], float(s), places=5)
        self.assertAlmostEqual(r['absrel'], float(((s * torch.exp(pred) - z).abs() / z).mean()), places=5)


class Crops(unittest.TestCase):
    def test_box_and_affines(self):
        body = np.zeros((100, 80), bool); body[30:70, 10:50] = True
        box = dp.crop_box(body)
        self.assertAlmostEqual(box[2], 52.0); self.assertAlmostEqual(box[0], 3.5); self.assertAlmostEqual(box[1], 23.5)
        a = dp.crop_affine(box, 512)
        np.testing.assert_allclose(a @ np.array([box[0], box[1], 1]), [0, 0], atol=1e-9)
        np.testing.assert_allclose(a @ np.array([box[0] + box[2], box[1] + box[2], 1]), [512, 512], atol=1e-9)
        np.testing.assert_allclose(dp.crop_affine(box, 256), a / 2)
        self.assertIsNone(dp.crop_box(np.zeros((4, 4), bool)))
        self.assertEqual(dp.crop_box(np.ones((4, 4), bool))[2], 32.0)  # minimum side

    @unittest.skipIf(cv2 is None, 'cv2 not installed')
    def test_pfm_round_trip_and_crop_item(self):
        h, w = 64, 96
        z = (np.random.default_rng(0).random((h, w)) * 100 + 50).astype(np.float32); z[:8] = 0
        body = np.zeros((h, w), bool); body[20:44, 30:70] = True
        rgb = (np.random.default_rng(1).random((h, w, 3)) * 255).astype(np.uint8)
        with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as d:
            d = Path(d)
            cv2.imwrite(str(d/'rgb.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            with (d/'z.pfm').open('wb') as f:  # bottom-to-top rows, little endian, as Blender writes them
                f.write(b'Pf\n' + f'{w} {h}\n'.encode() + b'-1.0\n' + np.flipud(z).astype('<f4').tobytes())
            np.testing.assert_array_equal(dp.read_pfm(d/'z.pfm', flip_y=True), z)
            cv2.imwrite(str(d/'mask.png'), body.astype(np.uint8) * 255)
            np.savez(d/'truth.npz', depth=z, body=body, valid=z > 0)
            legacy = dict(id=0, image=str(d/'rgb.png'), truth=None, depth=str(d/'z.pfm'), mask=str(d/'mask.png'))
            crossed = dict(id=1, image=str(d/'rgb.png'), truth=str(d/'truth.npz'))
            for it in (legacy, crossed):
                crop, zt, flags, box, body_px = dp.crop_item(it)
                self.assertEqual(crop.shape, (512, 512, 3)); self.assertEqual(zt.shape, (256, 256)); self.assertEqual(flags.dtype, np.uint8)
                self.assertAlmostEqual(box[2], 1.3 * 40); self.assertGreater(body_px, 0)
                self.assertTrue(set(np.unique(zt)) <= set(np.unique(z)) | {0.0})  # nearest resampling only
                self.assertTrue(set(np.unique(flags)) <= {0, 1, 2, 3})
                ys, xs = np.nonzero((flags & 1) > 0)   # the body lands centred in the target
                self.assertAlmostEqual((xs.min() + xs.max()) / 2, 127.5, delta=1.5); self.assertAlmostEqual((ys.min() + ys.max()) / 2, 127.5, delta=1.5)
                self.assertAlmostEqual((xs.max() - xs.min() + 1) / 256, 1 / 1.3, delta=0.02)
                self.assertTrue(((flags & 2) > 0)[(flags & 1) > 0].all())  # body pixels are valid here (depth > 0)
                np.testing.assert_array_equal(zt[(flags & 3) == 3] > 0, True)


class Augment(unittest.TestCase):
    def setUp(self):
        self.store = fake_store(2)
        self.rgb, self.z, self.flags = self.store.batch([0, 1])

    def test_identity_and_flip(self):
        eye = torch.tensor([[[1.0, 0, 0], [0, 1.0, 0]]]).expand(2, 2, 3)
        rgb, z, flags = dp.apply_warp(self.rgb, self.z, self.flags, eye)
        self.assertTrue(torch.allclose(rgb, self.rgb, atol=1e-3)); self.assertTrue(torch.equal(z, self.z)); self.assertTrue(torch.equal(flags, self.flags))
        flip = torch.tensor([[[-1.0, 0, 0], [0, 1.0, 0]]]).expand(2, 2, 3)
        rgb, z, flags = dp.apply_warp(self.rgb, self.z, self.flags, flip)
        self.assertTrue(torch.allclose(rgb, self.rgb.flip(-1), atol=1e-3)); self.assertTrue(torch.equal(z, self.z.flip(-1)))
        self.assertTrue(torch.equal(flags, self.flags.flip(-1)))

    def test_random_augmentation_is_seeded_and_keeps_targets_exact(self):
        outs = []
        for _ in range(2):
            gen = torch.Generator().manual_seed(5)
            outs.append(dp.augment(self.rgb, self.z, self.flags, gen, dp.CFG['augment']))
        for a, b in zip(*outs):
            self.assertTrue(torch.equal(a, b))
        rgb, z, flags = outs[0]
        self.assertEqual(rgb.shape, self.rgb.shape); self.assertEqual(z.shape, self.z.shape)
        self.assertTrue(set(flags.unique().tolist()) <= {0, 1, 2, 3})
        self.assertTrue(set(z.unique().tolist()) <= set(self.z.unique().tolist()) | {0.0})
        self.assertTrue(float(rgb.min()) >= 0 and float(rgb.max()) <= 1)
        self.assertFalse(torch.equal(rgb, self.rgb))

    def test_photometric_off_is_identity(self):
        B = 2; off = torch.zeros((B, 1, 1, 1), dtype=torch.bool)
        out = dp.apply_photometric(self.rgb, off, torch.full((B, 1, 1, 1), 0.2), torch.full((B, 1, 1, 1), 1.2),
                                   torch.full((B, 1, 1, 1), 0.9), torch.ones((B, 1, 1, 1), dtype=torch.bool), torch.randn(self.rgb.shape))
        self.assertTrue(torch.equal(out, self.rgb))


class Encoder(unittest.TestCase):
    def test_encode_layers_matches_kp_encode_and_has_no_graph(self):
        pe = tiny_encoder(); rgb = torch.rand(1, 3, 512, 512, generator=torch.Generator().manual_seed(7))
        feats = dp.encode_layers(pe, MEAN, STD, rgb, dp.LAYERS, autocast=False)
        self.assertEqual(len(feats), 4); self.assertEqual(tuple(feats[0].shape), (1, DIM, 32, 32))
        ref = kp.encode(pe, MEAN, STD, rgb, 24, autocast=False)
        self.assertTrue(torch.allclose(feats[-1], ref, atol=1e-5)); self.assertFalse(feats[-1].requires_grad)
        self.assertTrue(all(f.is_contiguous() for f in feats))  # a channels-last view crashes the CPU head backward
        self.assertFalse(torch.allclose(feats[0], feats[-1]))

    def test_head_shapes(self):
        head = dp.build_head(feat_dim=DIM, width=16)
        out = head([torch.randn(2, DIM, 32, 32) for _ in range(4)], torch.rand(2, 3, 512, 512))
        self.assertEqual(tuple(out.shape), (2, 1, 256, 256))
        self.assertEqual(tuple(dp.build_head().out[-1].weight.shape), (1, 64, 1, 1))


class Training(unittest.TestCase):
    def test_step_trains_head_only_and_ema_round_trip(self):
        torch.manual_seed(0)
        pe = tiny_encoder(); head = dp.build_head(feat_dim=DIM, width=16); store = fake_store(8)
        t = dp.Trainer(0, store, head, pe, MEAN, STD, small_cfg(), device='cpu', autocast=False)
        self.assertEqual(t.pool_sizes, dict(bottom=3, other=3))
        before_pe = kp.param_hashes(pe, 'pe'); before_head = {n: p.detach().clone() for n, p in head.named_parameters()}
        losses = [t.step() for _ in range(2)]
        self.assertTrue(all(math.isfinite(x) for x in losses)); self.assertEqual(t.u, 2)
        self.assertEqual(kp.param_hashes(pe, 'pe'), before_pe)
        self.assertTrue(all(p.grad is None for p in pe.parameters()))
        self.assertTrue(any(not torch.equal(p, before_head[n]) for n, p in head.named_parameters()))
        live = {n: p.detach().clone() for n, p in head.named_parameters()}
        t.swap_ema(); self.assertTrue(any(not torch.equal(p, live[n]) for n, p in head.named_parameters()))
        t.swap_ema(); self.assertTrue(all(torch.equal(p, live[n]) for n, p in head.named_parameters()))
        self.assertEqual(t.exposure['bottom'], 4); self.assertEqual(t.exposure['other'], 4)
        rows = t.evaluate([0, 6, 7], use_ema=True)
        self.assertEqual({r['group'] for r in rows}, {'train_probe', 'unseen_rig'})
        s = dp.summarize(rows)
        self.assertIn('unseen_rig/bottom', s); self.assertIn('rig/rig0/bottom', s); self.assertEqual(s['unseen_rig/all']['views'], 2)
        self.assertTrue(all(torch.equal(p, live[n]) for n, p in head.named_parameters()))  # evaluation restores the raw head

    def test_lr_schedule(self):
        cfg = small_cfg(); cfg.update(updates=100, warmup=10, peak_lr=1.0, floor=0.1)
        t = dp.Trainer(0, fake_store(8), dp.build_head(feat_dim=DIM, width=16), tiny_encoder(), MEAN, STD, cfg, device='cpu', autocast=False)
        self.assertAlmostEqual(t.lr(1), 0.1 * (0.1 + 0.9 * 0.5 * (1 + math.cos(0))), places=9)
        self.assertAlmostEqual(t.lr(10), 1.0 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * 9 / 100))), places=9)
        self.assertAlmostEqual(t.lr(101), 0.1, places=6)


class Endpoint(unittest.TestCase):
    def test_apply_endpoint_checks_and_loads(self):
        pe, other = tiny_encoder(), tiny_encoder()
        with torch.no_grad():
            for p in other.parameters():
                p.add_(0.01)
        ep = dict(release_sha256=kp.RELEASE_SHA, encoder_prefix='aggregator.patch_embed', weights='ema', ema_encoder=other.state_dict(),
                  seed=17, variant='UX_flat1e-4', update=12000)
        for bad in (dict(release_sha256='0' * 64), dict(weights='raw'), dict(encoder_prefix='aggregator'), dict(seed=29)):
            with self.assertRaises(ValueError):
                dp.apply_endpoint(pe, {**ep, **bad}, 17)
        info = dp.apply_endpoint(pe, ep, 17)
        self.assertEqual(info, dict(variant='UX_flat1e-4', seed=17, update=12000))
        self.assertEqual(kp.param_hashes(pe, 'x'), kp.param_hashes(other, 'x'))


class Items(unittest.TestCase):
    def test_admit(self):
        self.assertIsNone(dp.admit(dict(kind='real_labeled', split='train', scene='x', group='AZL3')))
        self.assertIsNone(dp.admit(dict(kind='synthetic', split='sealed', scene='output/x', group='g', rig='sealed_rig00')))
        self.assertIsNone(dp.admit(dict(kind='synthetic', split='val', scene='output/x', group='g', rig='sealed_rig01', evaluation_role='sealed_confirmation')))
        self.assertEqual(dp.admit(dict(kind='synthetic', split='train', scene='dataset/x', group='crim13/a', evaluation_role='free_motion_replay')),
                         dict(pool='legacy', group='train', rig='legacy/crim13'))
        self.assertEqual(dp.admit(dict(kind='synthetic', split='val', scene='output/x', group='mouse_climb_wall_91', rig='val_rig03',
                                       evaluation_role='unseen_rig_gate')), dict(pool='crossed', group='unseen_rig', rig='val_rig03'))
        self.assertEqual(dp.admit(dict(kind='synthetic', split='val', scene='output/x', group='mouse_wash_face_91', rig='val_rig06',
                                       evaluation_role='free_motion_replay'))['group'], 'replay_val')
        self.assertEqual(dp.admit(dict(kind='synthetic', split='val', scene='dataset/x', group='mouse_wash_face_91', evaluation_role='free_motion_replay'))['group'],
                         'legacy_val')
        self.assertEqual(sum(dp.EXPECTED.values()), 6736)

    def test_eval_ids_and_smoke_cap(self):
        items = fake_items(40)
        for i in items:
            i['group'] = 'train' if i['id'] < 10 else 'unseen_rig'
        ok = [True] * 40; ok[12] = False
        ids = dp.eval_ids(items, ok)
        self.assertEqual(ids, [0, 1] + [i for i in range(10, 40) if i != 12])
        smoke = dp.eval_ids(items, ok, smoke=True)
        self.assertEqual(len(smoke), 2 + 16)


if __name__ == '__main__':
    unittest.main()
