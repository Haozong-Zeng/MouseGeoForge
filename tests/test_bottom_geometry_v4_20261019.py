"""Analytical CPU fixtures, including the old teacher's failure boundary."""
import copy
import importlib
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import bottom_geometry_v4_core_20261019 as core


def rz(theta):
    z = theta*0; one = z+1
    return torch.stack((torch.stack((theta.cos(), -theta.sin(), z)),
                        torch.stack((theta.sin(), theta.cos(), z)), torch.stack((z, z, one))))


def decode_fixture(pose, _):
    es = []
    for sample in pose:
        es.append(torch.stack([torch.cat((rz(p[3]), p[:3, None]), -1) for p in sample]))
    return torch.stack(es), None


class GeometryNumerics(unittest.TestCase):
    def test_teacher_gradient_vanishes_with_error(self):
        norms = []
        for angle in [0., 1e-7, 1e-4, .1]:
            theta = torch.tensor(angle, dtype=torch.float64, requires_grad=True)
            value = core.chordal_rotation(rz(theta), torch.eye(3, dtype=torch.float64))
            norms.append(abs(float(torch.autograd.grad(value, theta)[0])))
        self.assertEqual(norms[0], 0.)
        self.assertLess(norms[1], 1.1e-7)
        self.assertLess(norms[1], norms[2])

    def test_complete_teacher_identical_outputs_have_zero_gradient(self):
        pose = torch.zeros(1, 3, 9, dtype=torch.float64, requires_grad=True)
        depth = torch.ones(1, 3, 32, 32, 1, dtype=torch.float64, requires_grad=True)
        target_pose = pose.detach().clone().requires_grad_()
        target_depth = depth.detach().clone().requires_grad_()
        batch = dict(roles=['bottom', 'side', 'side'], E=decode_fixture(pose.detach(), None)[0])
        loss = core.teacher_retention(dict(pose_enc=pose, depth=depth),
                                      dict(pose_enc=target_pose, depth=target_depth), batch, decode_fixture)
        loss.backward()
        self.assertEqual(float(loss.detach()), 0.)
        self.assertEqual(float(pose.grad.norm()), 0.)
        self.assertEqual(float(depth.grad.norm()), 0.)
        self.assertIsNone(target_pose.grad)
        self.assertIsNone(target_depth.grad)

    def test_pack_scale_excludes_background_and_preserves_cross_view_error(self):
        target = torch.ones(2, 2, 2)
        pred = torch.tensor([[[.5, 100.], [.5, 100.]], [[.25, 100.], [.25, 100.]]])
        body = torch.tensor([[[True, False], [True, False]]]*2)
        scale = core.body_scale(pred, target, torch.ones_like(body), body)
        individual = core.body_scale(pred, target, torch.ones_like(body), body, per_view=True)
        self.assertEqual([float(x) for x in individual], [2., 4.])
        self.assertGreater(float(((pred*scale-target)[body]).abs().sum()), 0.)
        self.assertIsNone(core.body_scale(pred, target, torch.zeros_like(body), body))

    def test_projection_joint_scale_invariance_including_tiny_positive_z(self):
        e = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)[:, :3]
        e[1, 0, 3] = .1
        k = torch.eye(3, dtype=torch.float64).repeat(2, 1, 1)
        xy = torch.tensor([[.2, .3], [.5, .1]], dtype=torch.float64)
        z = torch.tensor([2., 3.], dtype=torch.float64)
        expected, _, _ = core.project_depth(e, k, z, 0, 1, xy)
        for scale in [1e-8, .1, 10., 1e8]:
            scaled = e.clone(); scaled[..., :3, 3] *= scale
            actual, _, good = core.project_depth(scaled, k, z*scale, 0, 1, xy)
            self.assertTrue(bool(good.all()))
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

    def test_baseline_scale_is_not_an_oracle_depth_scale(self):
        known = torch.eye(4).repeat(3, 1, 1)[:, :3]; known[2, 0, 3] = 2
        pred = known.clone(); pred[:, :3, 3] /= 4
        self.assertEqual(float(core.baseline_scale(pred, known, ['bottom', 'side', 'side'])), 4.)
        self.assertIsNone(core.baseline_scale(pred[:1], known[:1], ['side']))
        self.assertIsNone(core.baseline_scale(torch.zeros_like(pred), known, ['bottom', 'side', 'side']))

    def test_rotation_swap_preserves_centers_and_canonicalization_preserves_projection(self):
        e = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)[:, :3]
        e[0, :3, :3] = rz(torch.tensor(.3, dtype=torch.float64))
        e[1, :3, :3] = rz(torch.tensor(.6, dtype=torch.float64))
        e[0, :3, 3] = torch.tensor([.1, .2, .3]); e[1, 0, 3] = .5
        known = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)[:, :3]
        known[1, 0, 3] = .2
        k = torch.eye(3, dtype=torch.float64).repeat(2, 1, 1); z = torch.ones(2, 2, 2, dtype=torch.float64)*2
        variants = core.component_oracles(e, k, z, known, k, z, .01)
        canonical = variants['oracle_body_gauge_joint'][0]
        swapped = variants['oracle_replace_R_keep_centers'][0]
        def centers(x):
            return -(x[:, :3, :3].transpose(-1, -2) @ x[:, :3, 3, None])[..., 0]
        torch.testing.assert_close(centers(canonical), centers(swapped), rtol=1e-12, atol=1e-12)
        xy = torch.tensor([[.2, .3]], dtype=torch.float64)
        original, _, _ = core.project_depth(e, k, torch.tensor([2.], dtype=torch.float64), 0, 1, xy)
        changed, _, _ = core.project_depth(canonical, k, torch.tensor([.02], dtype=torch.float64), 0, 1, xy)
        torch.testing.assert_close(original, changed, rtol=1e-12, atol=1e-12)

    def test_gradient_comparison_zero_and_antiparallel(self):
        self.assertEqual(core.gradient_comparison({'p': torch.zeros(3)}, {'p': torch.zeros(3)})['cosine'], 1.)
        self.assertAlmostEqual(core.gradient_comparison({'p': torch.ones(3)}, {'p': -torch.ones(3)})['cosine'], -1.)

    def test_invalid_projection_keeps_failure_denominator(self):
        ev = importlib.import_module('bottom_geometry_v4_eval_20261019')
        self.assertEqual(ev.score([0., None, 40.])['coverage32'], 1/3)
        self.assertIsNone(ev.score([0., None, 40.])['p95'])

    def test_dense_gt_roundtrip_and_scaled_prediction_on_fixed_support(self):
        ev = importlib.import_module('bottom_geometry_v4_eval_20261019')
        e = torch.eye(4).repeat(2, 1, 1)[:, :3]; e[1, 0, 3] = .2
        k = torch.eye(3).repeat(2, 1, 1); k[:, 0, 0] = 200; k[:, 1, 1] = 200
        k[:, 0, 2] = 256; k[:, 1, 2] = 256
        z = torch.ones(2, 512, 512)*2
        valid = torch.ones(1, 2, 512, 512, dtype=torch.bool)
        b = dict(E=e[None], K=k[None], depth=z[None], body=valid, valid=valid, roles=['side', 'side'])
        original = ev.dense_errors(e, k, z, b)
        self.assertEqual(ev.score(original['other'])['coverage32'], 1.)
        for scale in [.001, 1000.]:
            scaled = e.clone(); scaled[..., :3, 3] *= scale
            actual = ev.dense_errors(scaled, k, z*scale, b)
            self.assertEqual(len(actual['other']), len(original['other']))
            self.assertEqual(ev.score(actual['other'])['coverage32'], 1.)

    def test_bootstrap_units_are_rigs_and_sample_mismatch_is_rejected(self):
        compare = importlib.import_module('compare_bottom_geometry_v4_20261019')
        rows = []
        for i in range(6):
            rows.append(dict(index=i, context=[1, 2, 0], rig='rig0' if i < 5 else 'rig1',
                             evaluation_role='unseen_rig_gate', group='motion', body_depth=[],
                             cameras=[dict(star=True, pair_role='bottom', rotation_deg=10.)],
                             errors=dict(raw_joint=dict(bottom=[1., 2.], other=[]))))
        report = dict(version=4, provenance=dict(manifest_sha256='same'), partial=False, rows=rows)
        result = compare.compare(report, copy.deepcopy(report))
        self.assertEqual(result['comparisons'][0]['macro_paired']['rotation_median']['n_units'], 2)
        shorter = copy.deepcopy(report); shorter['rows'].pop()
        with self.assertRaisesRegex(ValueError, 'Sample sets differ'):
            compare.compare(report, shorter)

    def test_real_background_without_masks_is_rejected(self):
        rt = importlib.import_module('bottom_geometry_v4_runtime_20261019')
        with self.assertRaisesRegex(ValueError, 'EVERY view'):
            rt.background(dict(body=torch.zeros(1, 3, 16, 16), images=torch.ones(1, 3, 3, 16, 16)), 'gray')

    def test_same_pack_order_for_train_calibration_and_evaluation(self):
        rt = importlib.import_module('bottom_geometry_v4_runtime_20261019')
        rec = dict(kind='synthetic', views=[dict(id=i, physical_role='bottom' if i == 0 else 'side') for i in range(3)])
        self.assertEqual(list(rt.evaluation_packs(rec, {'rigs': {}})), [[1, 2, 0]])

    def test_complete_objective_batch_path_includes_teacher_and_anchor(self):
        runner = importlib.import_module('run_bottom_geometry_v4_20261019')
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight = torch.nn.Parameter(torch.tensor(.5))
            def forward(self, images):
                y = images*self.weight
                return dict(pose_enc=y.reshape(-1, 1, 1), depth=y.reshape(-1, 1, 1, 1, 1))
        class Feed:
            def next(self, item):
                return dict(images=torch.tensor([item['value']]))
        calls = []
        def terms(pred, batch, *, teacher=None, real_anchor=True):
            calls.append((teacher is not None, real_anchor))
            y = pred['pose_enc']
            result = {'camera': (y-1).square().mean()}
            if real_anchor:
                result['real_camera_anchor'] = (y-2).square().mean()
            if teacher is not None:
                result['teacher'] = (y-teacher['pose_enc'].detach()).square().mean()
            return result
        model = Tiny(); teacher = Tiny(); teacher.weight.data.fill_(.4)
        items = [dict(value=float(i)/8, views=[0, 1, 2]) for i in range(8)]
        with patch.object(runner.core, 'complete_terms', terms), patch.object(runner.batched, 'geometry_forward', lambda m, bs: m(torch.cat([b['images'] for b in bs]))):
            reference = None
            for cap in (1, 2, 4):
                value, losses = runner.backward(model, teacher, Feed(), items,
                                                dict(camera=1., real_camera_anchor=.1, teacher=.05), cap)
                self.assertGreater(losses['teacher'], 0.)
                self.assertGreater(losses['real_camera_anchor'], 0.)
                actual = runner.snapshot(model)
                if reference is not None:
                    self.assertLess(core.gradient_comparison(reference, actual)['relative_l2'], 1e-6)
                reference = actual
        self.assertTrue(all(t and a for t, a in calls))

    def test_calibration_respects_both_module_budgets(self):
        calibrator = importlib.import_module('calibrate_bottom_geometry_v4_20261019')
        weight = calibrator.budget_weight([dict(encoder=10., camera=2.)], dict(encoder=1., camera=1.), .2, .05)
        self.assertAlmostEqual(weight, .02)
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            calibrator.budget_weight([], dict(encoder=1., camera=1.), .2, .05)

    def test_cluster_lock_cannot_silently_bind_another_gpu(self):
        rt = importlib.import_module('bottom_geometry_v4_runtime_20261019')
        with patch.dict('os.environ', {'MOUSEGEOFORGE_GPU_LOCK_ID': 'GPU-test', 'CUDA_VISIBLE_DEVICES': '0'}):
            with self.assertRaises(ValueError):
                rt.gpu_lock_path()

    def test_disabled_training_refuses_before_payload_or_gpu_access(self):
        runner = importlib.import_module('run_bottom_geometry_v4_20261019')
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp)/'config.json'; cfg.write_text('{"execution_authorized":false}')
            with patch.object(runner, 'verified') as verify, self.assertRaisesRegex(ValueError, 'Training disabled'):
                runner.run(cfg, 'P0', 'train', execute=True)
            verify.assert_not_called()

    def test_seeds_change_update_order_without_changing_mixtures_or_exposures(self):
        runner = importlib.import_module('run_bottom_geometry_v4_20261019')
        base = [[dict(index=i*8+j, sampling_role=str(j)) for j in range(8)] for i in range(20)]
        with patch.object(runner.data, 'schedule', return_value=base):
            a = runner.registered_schedule({}, 20, 17)
            b = runner.registered_schedule({}, 20, 29)
            c = runner.registered_schedule({}, 20, 29)
        self.assertEqual(a, base); self.assertNotEqual(a, b); self.assertEqual(b, c)
        self.assertEqual(sorted(x[0]['index'] for x in a), sorted(x[0]['index'] for x in b))
        self.assertTrue(all(len(x) == 8 for x in b))

    def test_optimizer_failure_keeps_last_atomic_checkpoint(self):
        runner = importlib.import_module('run_bottom_geometry_v4_20261019')
        diagnostic = importlib.import_module('diagnose_bottom_geometry_v4_20261019')
        def rng():
            return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(), cuda=[])
        def restore_rng(x):
            random.setstate(x['python']); np.random.set_state(x['numpy']); torch.set_rng_state(x['cpu'])
        class Feed:
            def __init__(self, *args): pass
            def close(self): pass
        model = torch.nn.Linear(2, 1); optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        for pg in optimizer.param_groups: pg['base_lr'] = pg['lr']
        original = {n: p.detach().clone() for n, p in model.named_parameters()}
        def backward(*_):
            for p in model.parameters(): p.grad = torch.ones_like(p)
            return 1., {'camera': 1.}
        def bad_step():
            with torch.no_grad(): next(model.parameters()).add_(1.)
            raise RuntimeError('injected partial optimizer failure')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'P0').mkdir()
            (root/'P0/preflight.json').write_text(json.dumps(dict(passed=True, complete_objective=True, teacher_included=False)))
            (root/'config.json').write_text('{"execution_authorized":true}')
            (root/'protocol.json').write_text('{}'); (root/'manifest.json').write_text('{}')
            (root/'calibration.json').write_text(json.dumps(dict(passed=True, weights={'camera': 1.})))
            future = (runner.dt.datetime.now(runner.dt.timezone.utc)+runner.dt.timedelta(days=1)).isoformat()
            later = (runner.dt.datetime.now(runner.dt.timezone.utc)+runner.dt.timedelta(days=2)).isoformat()
            cfg = dict(execution_authorized=True, updates=300, optimization_deadline_utc=future, hard_deadline_utc=later,
                       calibration_receipt=str(root/'calibration.json'), weights={'camera': 1.}, seed=17,
                       camera_lr=.001, encoder_lr=.001, depth_lr=.001, workers=4, warmup_updates=50, head_batch_cap=4, real_anchor=True, clip_norm=10.)
            c = dict(output_dir=root, manifest=root/'manifest.json', release=root/'release.pt')
            work = [[dict(sampling_role='real')]*8]
            with patch.object(runner, 'verified', return_value=(cfg, c, {'release_sha256':'release'}, work)), \
                 patch.object(runner.rt, 'read_manifest', return_value={}), patch.object(runner.rt, 'model', return_value=(model, optimizer)), \
                 patch.object(runner.rt.geo.old, 'rng_state', rng), patch.object(runner.state, 'restore_rng', restore_rng), \
                 patch.object(runner.data, 'Feed', Feed), patch.object(diagnostic, 'diagnose'), patch.object(runner, 'backward', backward), \
                 patch.object(optimizer, 'step', bad_step), self.assertRaisesRegex(RuntimeError, 'injected'):
                runner.run(root/'config.json', 'P0', 'train', execute=True)
            checkpoint = torch.load(root/'P0/latest.pt', weights_only=False)
            self.assertEqual(checkpoint['step'], 0)
            for name, value in original.items():
                torch.testing.assert_close(checkpoint['model'][name], value, rtol=0, atol=0)

    def test_full_state_roundtrip_and_next_adam_update(self):
        runner = importlib.import_module('run_bottom_geometry_v4_20261019')
        def rng():
            return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(), cuda=[])
        def restore_rng(x):
            random.setstate(x['python']); np.random.set_state(x['numpy']); torch.set_rng_state(x['cpu'])
        model = torch.nn.Linear(2, 1); model.register_buffer('receipt_fixture', torch.ones(1))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
        work = [[dict(sampling_role='real')]*8]*2
        def update():
            optimizer.zero_grad(); loss = model(torch.ones(1, 2)).square().sum(); loss.backward(); optimizer.step()
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner.rt.geo.old, 'rng_state', rng), patch.object(runner.state, 'restore_rng', restore_rng):
            update(); ema = {n: p.detach().clone() for n, p in model.named_parameters()}
            path = Path(tmp)/'state.pt'
            runner.save(path, model, optimizer, ema, 1, work, 'protocol', 'release', {'real': 8})
            update(); expected = copy.deepcopy(model.state_dict())
            _, step, exposure = runner.restore(path, model, optimizer, 'protocol', 'release', work)
            self.assertEqual(step, 1); self.assertEqual(exposure['real'], 8)
            update()
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
