"""Run with PYTHONPATH=.:vggt-omega python -m unittest discover -s tests."""
import itertools
import json
from pathlib import Path
import unittest

import torch
from torch import nn
from vggt_omega.models.heads.dense_head import DenseHead
from vggt_omega.models.heads.instance_pose_head import MultiInstanceMouseHead
from mousegeoforge.models import DepthPoseModel
from mousegeoforge.models.adapters import LocalDepthFeatureBridge


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 32, 16, stride=16)

    def forward(self, images):
        b, v = images.shape[:2]
        patch = self.proj(images.flatten(0, 1)).flatten(2).transpose(1, 2)
        patch = patch.reshape(b, v, -1, 32)
        tokens = torch.cat((torch.zeros(b, v, 2, 32, device=images.device), patch), dim=2)
        cached = [None]*24
        for i in (4, 11, 17, 23):
            cached[i] = tokens
        return cached, 2


def tiny_model():
    torch.manual_seed(42)
    return DepthPoseModel(TinyEncoder(),
        DenseHead(dim_in=32, features=16, out_channels=[16]*4),
        MultiInstanceMouseHead(num_keypoints=5, num_queries=3, dim_in=32,
                               hidden_dim=16, num_heads=4), bottleneck=8)


class TestDepthPose(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_zero_init_and_off_exact_legacy(self):
        model = tiny_model().eval()
        images = torch.rand(1, 2, 3, 64, 64)
        with torch.no_grad():
            cached, start = model.aggregator(images)
            d, c = model.dense_head(cached, images, start)
            pose = model.instance_pose_head(cached, images, start)
            off = model(images, coupling='off')
            on = model(images, coupling='local_depth_features')
        for key, expected in {**pose, 'depth': d, 'depth_conf': c}.items():
            torch.testing.assert_close(off[key], expected, atol=0, rtol=0)
            torch.testing.assert_close(on[key], off[key], atol=0, rtol=0)

    def test_dense_chunk_features(self):
        model = tiny_model().eval()
        images = torch.rand(1, 3, 3, 64, 64)
        cached, start = model.aggregator(images)
        a = model.dense_head(cached, images, start, frames_chunk_size=None, return_features=True)
        b = model.dense_head(cached, images, start, frames_chunk_size=1, return_features=True)
        self.assertEqual(a[2].shape, (1, 3, 16, 16, 16))
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, atol=2e-6, rtol=2e-5)

    def test_shared_adapter_receives_both_tasks(self):
        model = tiny_model().train()
        images = torch.rand(1, 2, 3, 64, 64)
        for task in ('depth', 'instance_keypoint_xy'):
            model.zero_grad(set_to_none=True)
            out = model(images)
            out[task].square().mean().backward()
            grads = [p.grad for p in model.shared_adapter.parameters() if p.grad is not None]
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)
            self.assertTrue(all(p.grad is None for p in model.aggregator.parameters()))
            self.assertTrue(all(p.grad is None for p in model.depth_pose_bridge.parameters()))
        self.assertFalse(model.aggregator.training)

    def test_off_ignores_trained_bridge(self):
        model = tiny_model().eval()
        images = torch.rand(1, 2, 3, 64, 64)
        with torch.no_grad():
            before = model(images)
            model.depth_pose_bridge.output.weight.normal_()
            model.depth_pose_bridge.output.bias.normal_()
            after = model(images)
        for key in before:
            torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)

    def test_time_axis_not_silently_treated_as_views(self):
        with self.assertRaises(ValueError):
            tiny_model()(torch.rand(1, 2, 2, 3, 64, 64))

    def bridge_inputs(self):
        bridge = LocalDepthFeatureBridge(16, 8)
        nn.init.normal_(bridge.output.weight, std=.1)
        return bridge, dict(tokens=torch.randn(2, 3, 5, 16),
            coarse_xy=torch.rand(2, 3, 5, 2)*.6+.2,
            mask_logits=torch.randn(2, 3, 16, 16),
            depth_features=torch.randn(2, 8, 16, 16, requires_grad=True))

    def test_bridge_detach_and_gradient(self):
        for detach in (True, False):
            bridge, inputs = self.bridge_inputs()
            out = bridge(**inputs, detach_depth=detach)
            out.square().mean().backward()
            self.assertGreater(bridge.output.weight.grad.abs().sum(), 0)
            grad = inputs['depth_features'].grad
            if detach:
                self.assertIsNone(grad)
            else:
                self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(grad.abs().sum(), 0)

    def test_query_permutation_equivariance(self):
        bridge, inputs = self.bridge_inputs()
        perm = [2, 0, 1]
        before = bridge(**inputs)
        rearranged = {key: value[:, perm] if key != 'depth_features' else value
                      for key, value in inputs.items()}
        after = bridge(**rearranged)
        torch.testing.assert_close(after, before[:, perm])

    def test_invalid_windows_identity(self):
        for invalid in ('out_of_frame', 'nan_features', 'nan_xy'):
            bridge, inputs = self.bridge_inputs()
            nn.init.constant_(bridge.output.bias, 2.)
            if invalid == 'out_of_frame':
                inputs['coarse_xy'] = torch.full_like(inputs['coarse_xy'], 3.)
            elif invalid == 'nan_xy':
                inputs['coarse_xy'] = torch.full_like(inputs['coarse_xy'], float('nan'))
            else:
                inputs['depth_features'] = torch.full_like(inputs['depth_features'], float('nan'))
            out = bridge(**inputs)
            torch.testing.assert_close(out, inputs['tokens'], atol=0, rtol=0)

    def test_views_do_not_cross_in_bridge(self):
        bridge, inputs = self.bridge_inputs()
        before = bridge(**inputs)
        changed = dict(inputs)
        changed['depth_features'] = inputs['depth_features'].detach().clone()
        changed['depth_features'][1] += 100
        after = bridge(**changed)
        torch.testing.assert_close(before[0], after[0], atol=0, rtol=0)
        self.assertFalse(torch.allclose(before[1], after[1]))

    def test_joint_model_detach_and_both_heads_train(self):
        model = tiny_model().train()
        nn.init.normal_(model.depth_pose_bridge.output.weight, std=.1)
        out = model(torch.rand(1, 2, 3, 64, 64), coupling='local_depth_features')
        (out['depth'].log().square().mean() + out['instance_keypoint_xy'].square().mean()).backward()
        for module in (model.shared_adapter, model.dense_head, model.instance_pose_head,
                       model.depth_pose_bridge):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)

    def test_checkpoint_roundtrip(self):
        a, b = tiny_model().eval(), tiny_model().eval()
        nn.init.normal_(a.depth_pose_bridge.output.weight, std=.1)
        b.load_state_dict(a.state_dict(), strict=True)
        images = torch.rand(1, 2, 3, 64, 64)
        for key, value in a(images, coupling='local_depth_features').items():
            torch.testing.assert_close(value, b(images, coupling='local_depth_features')[key], atol=0, rtol=0)

    def test_factorial_spec_complete(self):
        root = Path(__file__).resolve().parents[1]
        spec = json.loads((root/'configs/experiments/temporal_depth_tail_factorial_v1.json').read_text())
        observed = {tuple(r[k] for k in spec['factors']) for r in spec['variants']}
        self.assertEqual(observed, set(itertools.product((False, True), repeat=3)))


if __name__ == '__main__':
    unittest.main()
