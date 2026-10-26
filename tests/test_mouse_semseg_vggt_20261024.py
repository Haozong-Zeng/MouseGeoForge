"""CPU fixtures for tools/mouse_semseg_vggt_20261024.py (no release weights, data or GPU)."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import mouse_semseg_vggt_20261024 as seg  # noqa: E402

TINY = dict(in_ch=16, d=32, heads=4, queries=4, ffn=64)


def blob(h, w, cy, cx, r):
    yy, xx = np.mgrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2 <= r * r).astype(np.uint8)


class Letterbox(unittest.TestCase):
    def test_geometry_padding_and_round_trip(self):
        g = seg.geometry(300, 400)
        self.assertEqual(g['size'], [384, 512]); self.assertEqual((g['top'], g['left']), (64, 0))
        self.assertEqual(g['roi'], [0, 64, 512, 448])
        lab = blob(300, 400, 150, 200, 60); lab[0, 0] = seg.IGNORE
        lb = seg.letterbox(lab, g, cv2.INTER_NEAREST)
        self.assertTrue((lb[:64] == 0).all() and (lb[448:] == 0).all())  # padding = supervised background
        one = np.stack([lab == 0, lab == 1]).astype(np.float32)
        back = seg.to_native(seg.letterbox(one.transpose(1, 2, 0), g, cv2.INTER_NEAREST).transpose(2, 0, 1), g, 300, 400)
        self.assertGreater(seg.overlap(back.argmax(0) == 1, lab == 1, lab != seg.IGNORE)[0], 0.97)

    def test_labels_from_png_and_npz(self):
        with tempfile.TemporaryDirectory() as d:
            png = Path(d)/'m.png'; cv2.imwrite(str(png), blob(20, 30, 10, 15, 5) * 255)
            npz = Path(d)/'t.npz'; np.savez(npz, body=blob(20, 30, 10, 15, 5).astype(bool), valid=np.ones((20, 30), bool))
            for path in (png, npz):
                lab = seg.read_label(dict(label=str(path), native_hw=[20, 30], id='x'))
                self.assertEqual(set(np.unique(lab)), {0, 1})
                self.assertEqual(int(lab.sum()), int(blob(20, 30, 10, 15, 5).sum()))


class Metrics(unittest.TestCase):
    def test_boundary_f1(self):
        m = blob(64, 64, 32, 32, 12).astype(bool); valid = np.ones_like(m)
        self.assertEqual(seg.boundary_f1(m, m, valid, 1), 1.0)
        self.assertEqual(seg.boundary_f1(np.roll(m, 1, 1), m, valid, 2), 1.0)
        self.assertLess(seg.boundary_f1(np.roll(m, 6, 1), m, valid, 1), 0.5)
        empty = np.zeros_like(m)
        self.assertEqual(seg.boundary_f1(empty, empty, valid, 1), 1.0)
        self.assertEqual(seg.boundary_f1(empty, m, valid, 1), 0.0)

    def test_overlap_ignores_unknown(self):
        p = np.zeros((4, 4), bool); p[:2] = True
        g = np.zeros((4, 4), bool); g[:1] = True
        valid = np.ones((4, 4), bool); valid[1] = False
        self.assertEqual(seg.overlap(p, g, valid), (1.0, 1.0))


class Decoder(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.net = seg.build_decoder(**TINY)
        self.feats = torch.randn(2, 4, 16, 32, 32)

    def test_shapes_and_semantic(self):
        out = self.net(self.feats)
        self.assertEqual(len(out['logits']), 4)
        self.assertEqual(tuple(out['logits'][-1].shape), (2, 4, 3))
        self.assertEqual(tuple(out['masks'][-1].shape), (2, 4, 128, 128))
        prob = seg.semantic(out['logits'][-1], out['masks'][-1])
        self.assertEqual(tuple(prob.shape), (2, 2, 512, 512))
        self.assertGreaterEqual(float(prob.min()), 0.0)  # Mask2Former semantic scores sum over queries (not <= 1)
        frac = prob[:, 1] / prob.sum(1).clamp_min(1e-12)
        self.assertTrue(float(frac.min()) >= 0 and float(frac.max()) <= 1)

    def test_criterion_ignores_unknown_pixels_and_backpropagates(self):
        lab = torch.zeros(2, 128, 128, dtype=torch.long); lab[:, 40:80, 40:80] = 1; lab[:, :10] = seg.IGNORE
        out = self.net(self.feats)
        loss = seg.criterion(out, lab)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(p.grad is not None for p in self.net.parameters() if p.requires_grad))
        with torch.no_grad():
            shifted = dict(logits=[l.clone() for l in out['logits']], masks=[m.clone() for m in out['masks']])
            for m in shifted['masks']:
                m[:, :, :10] += 7.0
            self.assertAlmostEqual(float(seg.criterion(out, lab)), float(seg.criterion(shifted, lab)), places=5)

    def test_perfect_prediction_has_lower_loss(self):
        lab = torch.zeros(1, 128, 128, dtype=torch.long); lab[:, 30:90, 50:100] = 1
        good = dict(logits=[torch.tensor([[[9., -9, -9], [-9, 9, -9], [-9, -9, 9], [-9, -9, 9]]])] * 4,
                    masks=[torch.stack([(lab[0] == 0).float(), (lab[0] == 1).float(), torch.zeros(128, 128),
                                        torch.zeros(128, 128)])[None] * 20 - 10] * 4)
        bad = dict(logits=[torch.zeros(1, 4, 3)] * 4, masks=[torch.zeros(1, 4, 128, 128)] * 4)
        self.assertLess(float(seg.criterion(good, lab)), float(seg.criterion(bad, lab)))

    def test_seg_l_replication(self):
        f = torch.randn(3, 4, 16, 32, 32)
        sel = f[:, seg.ARMS['SEG_L']]
        self.assertTrue(all(torch.equal(sel[:, k], f[:, 3]) for k in range(4)))
        self.assertTrue(torch.equal(f[:, seg.ARMS['SEG_M']], f))


class TrainingState(unittest.TestCase):
    def items(self):
        out = []
        for i in range(32):
            dom = 'real' if i % 2 else 'synthetic'
            out.append(dict(domain=dom, bottom=i % 4 < 2, cohort='real' if dom == 'real' else 'crossed',
                            rig=None if dom == 'real' else f'r{i % 3}', group=f'S{i % 5}', split='train'))
        return out

    def test_stratified_batches_and_state(self):
        items = self.items()
        s = seg.StratSampler(items, range(32), 3)
        for _ in range(10):
            b = s.take()
            self.assertEqual(len(b), 8)
            self.assertEqual(sorted(seg.stratum(items[i]) for i in b), sorted(seg.STRATA * 2))
        t = seg.StratSampler(items, range(32), 0); t.restore(copy.deepcopy(s.state()))
        self.assertEqual([s.take() for _ in range(5)], [t.take() for _ in range(5)])

    def test_arm_resume_is_exact_and_arms_share_initialisation(self):
        cfg = seg.design()
        lab = torch.zeros(2, 128, 128, dtype=torch.long); lab[:, 20:60, 20:60] = 1
        feats = [torch.randn(2, 4, 16, 32, 32, generator=torch.Generator().manual_seed(k)) for k in range(4)]
        a = seg.Arm('SEG_M', 17, cfg, 'cpu', TINY)
        l_arm = seg.Arm('SEG_L', 17, cfg, 'cpu', TINY)
        self.assertTrue(all(torch.equal(v, l_arm.net.state_dict()[k]) for k, v in a.net.state_dict().items()))
        for u in range(1, 5):
            a.step(feats[u - 1], lab, u)
        b = seg.Arm('SEG_M', 17, cfg, 'cpu', TINY)
        for u in range(1, 3):
            b.step(feats[u - 1], lab, u)
        state = copy.deepcopy(b.state_dict())
        c = seg.Arm('SEG_M', 17, cfg, 'cpu', TINY); c.load_state_dict(state)
        for u in range(3, 5):
            c.step(feats[u - 1], lab, u)
        for k, v in a.net.state_dict().items():
            self.assertTrue(torch.equal(v, c.net.state_dict()[k]), k)
        for k in a.ema:
            self.assertTrue(torch.equal(a.ema[k], c.ema[k]), k)

    def test_plateau_and_lr(self):
        cfg = seg.design()['frozen_head_pair']
        flat = [dict(update=u, loss=1.0, iou=0.9, bf1=0.7) for u in range(4000, 6001, 500)]
        self.assertTrue(seg.plateau(flat, cfg['plateau'], cfg['min_updates']))
        self.assertFalse(seg.plateau([dict(r, iou=0.9 + 0.001 * i) for i, r in enumerate(flat)], cfg['plateau'], cfg['min_updates']))
        arm = seg.Arm('SEG_M', 17, seg.design(), 'cpu', TINY)
        self.assertAlmostEqual(arm.lr(1), 1e-4 / 500)
        env = lambda u: 0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * (u - 1) / 18000))
        self.assertAlmostEqual(arm.lr(500), 1e-4 * env(500))
        self.assertAlmostEqual(arm.lr(250), 1e-4 * 0.5 * env(250))
        self.assertAlmostEqual(arm.lr(18000) / 1e-4, 0.05, places=3)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
