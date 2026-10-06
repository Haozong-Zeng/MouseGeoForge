import unittest
import torch
from mousegeoforge.models.surface_curve_integral import SurfaceCurveIntegralProbe
from mousegeoforge.surface_curve_supervision import cubic_bspline


class TestSurfaceIntegral(unittest.TestCase):
    def setUp(self):torch.manual_seed(8);torch.set_num_threads(2)

    def test_interpolation_reproduces_original_cubic(self):
        m=SurfaceCurveIntegralProbe();controls=torch.rand(4,6,2)
        points=cubic_bspline(controls,6)
        torch.testing.assert_close(torch.einsum('sk,nkd->nsd',m.interpolation,points),cubic_bspline(controls,33),atol=2e-6,rtol=0)

    def test_endpoints_equal_expectation(self):
        m=SurfaceCurveIntegralProbe();curve,points,prob=m(torch.randn(2,1024),torch.randn(2,256,8,9),torch.rand(2,6,2),True)
        self.assertEqual(tuple(curve.shape),(2,33,2));torch.testing.assert_close(curve[:,[0,-1]],points[:,[0,-1]])
        torch.testing.assert_close(prob.sum((-2,-1)),torch.ones(2,6))

    def test_modes_same_init_and_frozen_inputs(self):
        a=SurfaceCurveIntegralProbe('global');b=SurfaceCurveIntegralProbe('spatial');b.load_state_dict(a.state_dict())
        pose=torch.randn(2,1024,requires_grad=True);memory=torch.randn(2,256,8,9,requires_grad=True);controls=torch.rand(2,6,2,requires_grad=True)
        x=a(pose,memory,controls);y=b(pose,memory,controls);torch.testing.assert_close(x,y,atol=0,rtol=0)
        y.square().mean().backward();self.assertIsNone(pose.grad);self.assertIsNone(memory.grad);self.assertIsNone(controls.grad)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in b.parameters()))


if __name__=='__main__':unittest.main()
