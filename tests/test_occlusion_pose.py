import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from training.keypoint_contract import read_validity, legacy_point, read_occlusion
from training.instance_pose_losses import keypoint_uncertainty_loss, instance_pose_loss, InstancePoseLossWeights
from training.test_instance_pose_geometry_losses import _problem
from vggt_omega.utils.robust_pose_geometry import triangulate_robust, project
from mousegeoforge.pose_association import associate_two_views


class OcclusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_unknown_and_occluded_are_distinct(self):
        a = dict(uv=np.array([[[20., 10.], [np.nan, np.nan], [0., 0.]]]),
                 in_frame=np.ones((1, 3), bool), joint_world_xyz=np.zeros((3, 3)),
                 surface_world_xyz=np.zeros((3, 3)), xy_label_valid=np.array([[True, False, False]]),
                 visibility_valid=np.array([[True, False, False]]))
        with tempfile.TemporaryDirectory() as d:
            f = read_validity(a, Path(d)/'keypoints.npz')
        self.assertEqual(f['xy_label_valid'].tolist(), [[True, False, False]])
        self.assertTrue(f['visibility_valid'][0, 0])

    def test_legacy_does_not_promote_zero_placeholders(self):
        self.assertFalse(legacy_point(dict(x=0, y=0, visible=False), 100, 80)[2])
        xy, inside, label, visible, known = legacy_point(
            dict(uv=[25, 20], visible=False, xy_label_valid=True), 100, 80)
        self.assertTrue(inside and label and known)
        self.assertFalse(visible)

    def test_explicit_flags_override_sidecar(self):
        a = dict(uv=np.zeros((1, 1, 2)), in_frame=np.ones((1, 1), bool),
                 joint_world_xyz=np.zeros((1, 3)), surface_world_xyz=np.zeros((1, 3)),
                 xy_label_valid=np.zeros((1, 1), bool))
        with tempfile.TemporaryDirectory() as d:
            np.savez(Path(d)/'keypoint_validity.npz', schema_version=1,
                     xy_label_valid=np.ones((1, 1), bool))
            self.assertFalse(read_validity(a, Path(d)/'keypoints.npz')['xy_label_valid'].any())

    def test_source_causes_absent_and_stale(self):
        a=dict(in_frame=np.ones((2,1,3),bool))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'keypoints.npz'
            np.savez(path,**a)
            self.assertTrue((read_occlusion(a,path)==-1).all())
            np.savez(path.with_name('keypoint_occlusion.npz'),schema_version=1,annotation_sha256='stale')
            with self.assertRaises(ValueError): read_occlusion(a,path)

    def test_uncertainty_branch_does_not_move_pose_features(self):
        from test_depth_pose import tiny_model
        model=tiny_model()
        out=model(torch.rand(1,2,3,64,64))
        loss=keypoint_uncertainty_loss(out['instance_keypoint_xy'],out['instance_keypoint_sigma'],
                                      torch.zeros_like(out['instance_keypoint_xy']),
                                      torch.ones_like(out['instance_keypoint_sigma'],dtype=torch.bool))
        loss.backward()
        self.assertGreater(model.instance_pose_head.keypoint_log_sigma.weight.grad.abs().sum(),0)
        for name,p in model.named_parameters():
            if not name.startswith('instance_pose_head.keypoint_log_sigma.'):
                self.assertTrue(p.grad is None or not p.grad.any(),name)

    def test_uncertainty_only_gradients(self):
        xy = torch.tensor([[[.1, .2], [.8, .9]]], requires_grad=True)
        sigma = torch.full((1, 2), .03, requires_grad=True)
        loss = keypoint_uncertainty_loss(xy, sigma, torch.zeros_like(xy), torch.tensor([[True, False]]))
        loss.backward()
        self.assertIsNone(xy.grad)
        self.assertNotEqual(float(sigma.grad[0, 0]), 0.)
        self.assertEqual(float(sigma.grad[0, 1]), 0.)
        huge = keypoint_uncertainty_loss(xy, torch.full((1, 2), 100.), torch.zeros_like(xy), torch.ones(1, 2, dtype=torch.bool))
        fitted = keypoint_uncertainty_loss(xy, torch.full((1, 2), .5), torch.zeros_like(xy), torch.ones(1, 2, dtype=torch.bool))
        self.assertGreater(huge, fitted)

    def cameras(self, count=4):
        p = torch.eye(3, 4, dtype=torch.float64).repeat(count, 1, 1)
        p[:, 0, 3] = torch.linspace(-.6, .6, count)
        p[:, 1, 3] = torch.tensor([0., .2, -.2, .4])[:count]
        k = torch.tensor([[600., 0, 320.], [0, 600., 240.], [0, 0, 1.]], dtype=torch.float64)
        p = k@p
        xyz = torch.tensor([.05, .08, 2.], dtype=torch.float64)
        xy, _ = project(p, xyz)
        return p, xyz, xy

    def test_perfect_and_gradients(self):
        p, xyz, xy = self.cameras()
        xy = xy.clone().requires_grad_()
        result = triangulate_robust(xy, p)
        self.assertTrue(result.supported)
        torch.testing.assert_close(result.xyz, xyz, atol=1e-7, rtol=1e-7)
        result.xyz.sum().backward()
        self.assertTrue(torch.isfinite(xy.grad).all())

    def test_outlier_and_permutation(self):
        p, xyz, xy = self.cameras()
        xy[0] += torch.tensor([100., -80.])
        for perm in ([0, 1, 2, 3], [2, 0, 3, 1]):
            r = triangulate_robust(xy[perm], p[perm])
            self.assertTrue(r.supported)
            self.assertEqual(int(r.inliers.sum()), 3)
            torch.testing.assert_close(r.xyz, xyz, atol=1e-5, rtol=1e-5)

    def test_degenerate_missing_and_uncalibrated(self):
        p, xyz, xy = self.cameras()
        r = triangulate_robust(xy, p, torch.tensor([True, False, False, False]))
        self.assertFalse(r.supported)
        self.assertTrue(torch.isnan(r.xyz).all())
        r = triangulate_robust(xy[:1].repeat(2, 1), p[:1].repeat(2, 1, 1))
        self.assertFalse(r.supported)
        with self.assertRaises(ValueError):
            triangulate_robust(xy, p, sigma_px=torch.ones_like(xy))
        r = triangulate_robust(xy[:2], p[:2])
        self.assertIn('no_outlier_redundancy', r.reason)

    def test_scale_invariance(self):
        p, xyz, xy = self.cameras()
        a = triangulate_robust(xy, p, sigma_px=torch.ones_like(xy)*2, uncertainty_calibrated=True)
        pp = p.clone(); pp[:, :2] *= 2
        b = triangulate_robust(xy*2, pp, sigma_px=torch.ones_like(xy)*4, uncertainty_calibrated=True,
                               reprojection_threshold_px=16)
        torch.testing.assert_close(a.xyz, b.xyz)

    def test_per_instance_robust_loss(self):
        pred, target, xy, hw = _problem(perturb=.001)
        loss, logs = instance_pose_loss(pred, target, image_size=hw, geometry_mode='robust')
        self.assertEqual(float(logs['instance_geometry_instances']), 2.)
        loss.backward()
        self.assertTrue(torch.isfinite(xy.grad).all())

    def test_geometry_off_really_skips_solver(self):
        from unittest.mock import patch
        pred,target,xy,hw=_problem()
        with patch('training.instance_pose_losses._matched_multiview_geometry_losses',side_effect=AssertionError('called')):
            loss,logs=instance_pose_loss(pred,target,InstancePoseLossWeights(keypoint_3d=0.,reprojection=0.),
                                         image_size=hw,geometry_mode='robust')
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(logs['instance_3d_points']),0.)

    def test_visibility_labels_without_coordinate_labels(self):
        pred,target,xy,hw=_problem()
        target['keypoint_valid'].fill_(False)
        target['keypoint_visibility_valid'].fill_(True)
        pred['instance_keypoint_visibility_logits'].requires_grad_(True)
        loss,logs=instance_pose_loss(pred,target,InstancePoseLossWeights(keypoint_3d=0.,reprojection=0.))
        self.assertEqual(float(logs['loss_instance_keypoint_l1']),0.)
        self.assertGreater(float(logs['loss_instance_keypoint_visibility']),0.)
        loss.backward()
        self.assertGreater(pred['instance_keypoint_visibility_logits'].grad.abs().sum(),0.)

    def test_prediction_only_association(self):
        pred, target, xy, hw = _problem()
        # Original fixture already permutes queries in view 1.
        out = {k: v[:, :2] for k, v in pred.items()}
        p = target['intrinsics'][0, :2]@target['extrinsics'][0, :2, :3]
        # Shift second animal vertically: horizontal-baseline epipolar geometry
        # alone cannot distinguish animals on the same horizontal line.
        out['instance_keypoint_xy'] = out['instance_keypoint_xy'].detach().clone()
        out['instance_keypoint_xy'][0, 0, 1, :, 1] += .2
        out['instance_keypoint_xy'][0, 1, 0, :, 1] += .2
        pairs = associate_two_views(out, p, hw)
        self.assertEqual({(a, b) for a, b, _ in pairs}, {(0, 1), (1, 0)})


if __name__ == '__main__':
    unittest.main()
