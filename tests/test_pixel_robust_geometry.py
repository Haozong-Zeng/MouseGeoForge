import unittest
import torch
from vggt_omega.utils.robust_pose_geometry import project, triangulate_robust


class PixelKernelTests(unittest.TestCase):
    def problem(self, count=4):
        p = torch.eye(3, 4, dtype=torch.float64).repeat(count, 1, 1)
        p[:, 0, 3] = torch.linspace(-.6, .6, count)
        p[:, 1, 3] = torch.tensor([0., .2, -.2, .4])[:count]
        p = torch.tensor([[600., 0, 320.], [0, 600., 240.], [0, 0, 1.]], dtype=torch.float64) @ p
        xyz = torch.tensor([.05, .08, 2.], dtype=torch.float64)
        return p, xyz, project(p, xyz)[0]

    def test_uniform_legacy_exact_equivalence(self):
        p, _, xy = self.problem(2)
        xy[0] += torch.tensor([4., -3.])
        a = triangulate_robust(xy, p)
        b = triangulate_robust(xy, p, huber_delta_px=3.)
        self.assertEqual(a.supported, b.supported)
        torch.testing.assert_close(a.xyz, b.xyz, atol=0, rtol=0)

    def test_global_sigma_scale_invariance_above_floor(self):
        p, _, xy = self.problem(2)
        xy[0] += torch.tensor([4., -3.])
        sigma = torch.tensor([[2., 3.], [7., 5.]], dtype=torch.float64)
        args = dict(uncertainty_calibrated=True, huber_delta_px=3.)
        a = triangulate_robust(xy, p, sigma_px=sigma, **args)
        b = triangulate_robust(xy, p, sigma_px=sigma*5, **args)
        self.assertTrue(a.supported and b.supported)
        torch.testing.assert_close(a.xyz, b.xyz, atol=1e-8, rtol=1e-8)
        torch.testing.assert_close(a.inliers, b.inliers)

    def test_resolution_scaling(self):
        p, _, xy = self.problem(2)
        xy[0] += torch.tensor([4., -3.])
        sigma = torch.tensor([[2., 3.], [7., 5.]], dtype=torch.float64)
        a = triangulate_robust(xy, p, sigma_px=sigma, uncertainty_calibrated=True, huber_delta_px=3.)
        resized = p.clone(); resized[:, :2] *= 2
        b = triangulate_robust(xy*2, resized, sigma_px=sigma*2, uncertainty_calibrated=True,
                               huber_delta_px=6., reprojection_threshold_px=16.)
        self.assertTrue(a.supported and b.supported)
        torch.testing.assert_close(a.xyz, b.xyz, atol=1e-8, rtol=1e-8)

    def test_outlier_permutation(self):
        p, xyz, xy = self.problem()
        xy[0] += torch.tensor([100., -80.])
        for perm in ([0, 1, 2, 3], [2, 0, 3, 1]):
            result = triangulate_robust(xy[perm], p[perm], huber_delta_px=3.)
            self.assertTrue(result.supported)
            self.assertEqual(int(result.inliers.sum()), 3)
            torch.testing.assert_close(result.xyz, xyz, atol=1e-5, rtol=1e-5)

    def test_coordinate_gradients_not_sigma(self):
        p, _, xy = self.problem()
        xy = xy.clone().requires_grad_()
        sigma = torch.full_like(xy, 2., requires_grad=True)
        r = triangulate_robust(xy, p, sigma_px=sigma, uncertainty_calibrated=True, huber_delta_px=3.)
        self.assertTrue(r.supported)
        r.xyz.sum().backward()
        self.assertTrue(torch.isfinite(xy.grad).all())
        self.assertIsNone(sigma.grad)

    def test_invalid_kernel_and_uncalibrated_guard(self):
        p, _, xy = self.problem()
        for value in (0., -1., float('nan'), float('inf')):
            with self.assertRaises(ValueError): triangulate_robust(xy, p, huber_delta_px=value)
        with self.assertRaises(ValueError):
            triangulate_robust(xy, p, sigma_px=torch.ones_like(xy), huber_delta_px=3.)

    def test_missing_views_remain_unsupported(self):
        p, _, xy = self.problem()
        r = triangulate_robust(xy, p, torch.tensor([True, False, False, False]), huber_delta_px=3.)
        self.assertFalse(r.supported)
        self.assertTrue(torch.isnan(r.xyz).all())


if __name__ == '__main__': unittest.main()
