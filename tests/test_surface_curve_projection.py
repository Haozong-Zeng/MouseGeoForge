import unittest
import torch
from mousegeoforge.surface_curve_projection import project_curve_four_controls
from mousegeoforge.surface_curve_supervision import cubic_bspline


class TestProjection(unittest.TestCase):
    def setUp(self):torch.manual_seed(3)
    def test_cubic_reproduced(self):
        p=cubic_bspline(torch.rand(5,4,2),33);torch.testing.assert_close(project_curve_four_controls(p),p,atol=2e-6,rtol=0)
    def test_endpoint_preservation(self):
        p=torch.randn(3,33,2);torch.testing.assert_close(project_curve_four_controls(p)[:,[0,-1]],p[:,[0,-1]],atol=0,rtol=0)
    def test_idempotent(self):
        p=project_curve_four_controls(torch.rand(3,65,2));torch.testing.assert_close(project_curve_four_controls(p),p,atol=2e-6,rtol=0)
    def test_finite_gradient(self):
        p=torch.randn(2,33,2,requires_grad=True);project_curve_four_controls(p).square().mean().backward();self.assertTrue(torch.isfinite(p.grad).all())


if __name__=='__main__':unittest.main()
