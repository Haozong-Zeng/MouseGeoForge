"""CPU fixtures for tools/head_first_kp_synth_20261026.py (no release weights, data or GPU)."""
import copy
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]/'tools'))
sys.path.insert(0, str(HERE.parent))
import head_first_kp_synth_20261026 as ks  # noqa: E402
import head_first_kp_20261024 as kp  # noqa: E402
from test_head_first_kp_20261024 import DIM, tiny_encoder  # noqa: E402

NAMES = ['nose', 'head_center', 'left_ear', 'right_ear', 'neck', 'spine_center', 'lumbar_spine', 'tail_base', 'tail_q1', 'tail_q2',
         'tail_q3', 'tail_q4', 'tail_q5', 'tail_tip', 'left_forepaw_center', 'right_forepaw_center', 'left_hindpaw_center',
         'right_hindpaw_center', 'left_elbow', 'right_elbow', 'left_knee', 'right_knee', 'abdomen_center']


def fake_npz(views=2, instances=None, seed=0):
    g = np.random.default_rng(seed)
    shape = (views, 23, 2) if instances is None else (views, instances, 23, 2)
    uv = g.random(shape) * 400 + 50
    ones = np.ones(shape[:-1], bool)
    return dict(names=np.array(NAMES), uv=uv.astype(np.float32), xy_label_valid=ones.copy(), surface_visible=ones.copy(),
                in_frame=ones.copy(), view_ids=np.arange(views, dtype=np.int32))


RECORD = dict(scene='dataset/BlenderPyMouse/X/frame_000001', group='mouse_climb_wall_91', split='train')
VIEWS = {0: dict(physical_role='bottom', image='v0.png'), 1: dict(physical_role='side', image='v1.png')}


class FakeStore3:
    """own / pano / synth items, train split, random crops and crop-space targets."""

    def __init__(self, n=9):
        g = torch.Generator().manual_seed(1)
        self.items = [dict(dataset=('own', 'pano', 'synth')[i % 3], split='train', bottom=i % 2 == 0, points=[[0, 0]] * 5,
                           known=[True] * 5, box=[0, 0, 100, 100], session=f"S{('own', 'pano', 'synth')[i % 3]}", group=f'G{i}')
                      for i in range(n)]
        self.pts = torch.rand(n, 5, 2, generator=g) * 400 + 50
        self.known = torch.rand(n, 5, generator=g) > 0.2
        self.known_np = self.known.numpy()
        self.scale = np.ones(n); self.shift = np.zeros((n, 2))
        self.crops = torch.randint(0, 255, (n, 512, 512, 3), dtype=torch.uint8, generator=g)
        self.need_feats = False

    def rgb_for(self, ids):
        return self.crops[ids].permute(0, 3, 1, 2).float() / 255

    def feats_for(self, ids):
        raise RuntimeError('UX recomputes features online')


def small_cfg(synth=2, weight=1.0):
    cfg = ks.variant_cfg('synth8')
    cfg['pair'].update(updates_per_branch=4, cosine_total_updates=4, head_warmup_updates=1, dev_eval_updates=[4])
    cfg['protected']['warmup_updates'] = 1
    cfg['task'].update(own_views_per_update=2, pano_views_per_update=2, synth_views_per_update=synth)
    cfg['synth']['weight'] = weight
    return cfg


class Items(unittest.TestCase):
    def test_single_instance_rows(self):
        z = fake_npz(views=2)
        z['surface_visible'][0, NAMES.index('left_forepaw_center')] = False   # occluded in view 0
        z['in_frame'][1, NAMES.index('tail_tip')] = False                     # tail tip outside view 1: not in the box
        rows = ks.synth_rows(z, RECORD, VIEWS)
        self.assertEqual(len(rows), 2)
        r0, r1 = rows
        self.assertEqual((r0['dataset'], r0['split'], r0['session'], r0['camera'], r0['bottom']), ('synth', 'train', 'synth_climb2', '0', True))
        self.assertFalse(r1['bottom']); self.assertEqual(r1['image'], 'v1.png')
        idx = [NAMES.index(n) for n in ks.FIVE]
        np.testing.assert_allclose(r0['points'], z['uv'][0, idx], atol=1e-6)
        self.assertEqual(r0['known'], [True, False, True, True, True])
        p = z['uv'][0]
        np.testing.assert_allclose(r0['box'], [p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()], atol=1e-5)
        p = z['uv'][1][z['in_frame'][1]]
        np.testing.assert_allclose(r1['box'], [p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()], atol=1e-5)
        a = np.array(r0['affine']); box = r0['box']
        np.testing.assert_allclose(a @ np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, 1]), [256, 256], atol=1e-6)
        self.assertEqual(len({r['id'] for r in rows}), 2)

    def test_multi_instance_and_skips(self):
        z = fake_npz(views=1, instances=2)
        rows = ks.synth_rows(z, dict(RECORD, group='crim13/021709_A25', split='val'), {0: VIEWS[0]})
        self.assertEqual([r['instance'] for r in rows], [0, 1]); self.assertEqual(rows[0]['split'], 'dev')
        self.assertEqual(rows[0]['session'], 'synth_crim131'); self.assertTrue(rows[0]['group'].endswith('#0'))
        z['in_frame'][0, 1, :] = False; z['in_frame'][0, 1, :2] = True      # instance 1 has only 2 points in frame
        self.assertEqual([r['instance'] for r in ks.synth_rows(z, RECORD, {0: VIEWS[0]})], [0])
        z = fake_npz(views=1)
        z['surface_visible'][0, [NAMES.index(n) for n in ks.FIVE]] = False  # no visible five-point: nothing to learn
        self.assertEqual(ks.synth_rows(z, RECORD, {0: VIEWS[0]}), [])
        self.assertEqual(ks.synth_rows(fake_npz(views=1), RECORD, {5: VIEWS[0]}), [])  # view not in the record


class Config(unittest.TestCase):
    def test_variant_cfg_extends_the_chosen_recipe(self):
        cfg = ks.variant_cfg('synth8')
        self.assertEqual(cfg['task']['synth_views_per_update'], 8); self.assertEqual(cfg['synth']['weight'], 1.0)
        self.assertEqual((cfg['task']['own_views_per_update'], cfg['task']['pano_views_per_update']), (8, 8))
        self.assertEqual(cfg['pair']['updates_per_branch'], 12000); self.assertEqual(cfg['pair']['head_peak_lr'], 1e-4)
        self.assertEqual(cfg['explore'], dict(layer_decay=1.0, top_cap_lr=1e-4))
        self.assertEqual(ks.variant_cfg('synth4')['synth']['weight'], 0.5)


class Training(unittest.TestCase):
    def trainer(self, cfg, seed=0):
        torch.manual_seed(seed)
        pe = tiny_encoder(); head = kp.build_head(5, DIM).eval(); store = FakeStore3()
        return ks.SynthTrainer('UX', seed, cfg, store, head, pe, torch.zeros(1, 3, 1, 1), torch.ones(1, 3, 1, 1), device='cpu', autocast=False)

    def test_step_uses_three_domains(self):
        t = self.trainer(small_cfg(synth=2))
        self.assertEqual(sorted(t.s_synth.pool), [2, 5, 8])
        losses = [t.step() for _ in range(2)]
        self.assertTrue(all(math.isfinite(x) for x in losses)); self.assertEqual(t.u, 2)
        self.assertEqual(t.exposure['synth/bottom'] + t.exposure['synth/other'], 4)
        self.assertEqual(t.exposure['own/bottom'] + t.exposure['own/other'], 4)
        self.assertIn('loss_synth', t.log[0]); self.assertGreater(t.log[0]['loss_synth'], 0)
        rows = t.evaluate([0, 1, 2], use_ema=True)
        self.assertEqual({r['dataset'] for r in rows}, {'own', 'pano', 'synth'})
        s = kp.summarize(rows)
        self.assertIn('own/all/all', s); self.assertIn('session/Ssynth/bottom/paws', s); self.assertNotIn('synth/all/all', s)

    def test_zero_synthetic_views_reduces_to_the_base_step(self):
        a, b = self.trainer(small_cfg(synth=0)), self.trainer(small_cfg(synth=2, weight=0.0))
        la, lb = a.step(), b.step()
        self.assertEqual(a.log[0]['loss_synth'], 0.0)
        self.assertAlmostEqual(la, lb, places=5)  # same own/pano draws, synthetic term weighted to zero
        self.assertEqual(a.exposure['synth/bottom'] + a.exposure['synth/other'], 0)


if __name__ == '__main__':
    unittest.main()
