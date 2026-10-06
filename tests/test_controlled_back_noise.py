import unittest
import torch
from tools.probe_controlled_back_noise_20261004 import nearest_curve,point_curve_geometry_loss


class CurveMetricTests(unittest.TestCase):
    def test_straight_curve_distance_parameter(self):
        curve=torch.tensor([[[0.,0.],[5.,0.],[10.,0.]]])
        points=torch.tensor([[[3.,4.],[12.,0.]]])
        distance,u=nearest_curve(points,curve)
        self.assertTrue(torch.allclose(distance,torch.tensor([[4.,2.]])))
        self.assertTrue(torch.allclose(u,torch.tensor([[.3,1.]])))

    def test_degenerate_segment_finite(self):
        curve=torch.tensor([[[0.,0.],[0.,0.],[10.,0.]]])
        distance,u=nearest_curve(torch.tensor([[[5.,2.],[0.,0.]]]),curve)
        self.assertTrue(torch.isfinite(u).all())
        self.assertTrue(torch.allclose(distance,torch.tensor([[2.,0.]])))

    def test_curve_distance_not_material_accuracy(self):
        curve=torch.tensor([[[0.,0.],[10.,0.]]])
        distance,u=nearest_curve(torch.tensor([[[9.,0.]]]),curve)
        self.assertEqual(float(distance[0,0]),0.)
        self.assertGreater(float((u-.3).abs()[0,0]),.5)

    def test_geometry_gradient_towards_line(self):
        p=torch.tensor([[[5.,4.],[8.,0.]]],requires_grad=True)
        curve=torch.tensor([[[0.,0.],[10.,0.]]],requires_grad=True)
        loss=point_curve_geometry_loss(p,curve,torch.ones((1,2),dtype=torch.bool));loss.backward()
        self.assertGreater(float(p.grad[0,0,1]),0.)
        self.assertAlmostEqual(float(p.grad[0,0,0]),0.)
        self.assertTrue(torch.isfinite(p.grad).all());self.assertIsNone(curve.grad)

    def test_geometry_ignores_missing(self):
        p=torch.tensor([[[5.,0.],[8.,400.]]],requires_grad=True)
        curve=torch.tensor([[[0.,0.],[10.,0.]]])
        loss=point_curve_geometry_loss(p,curve,torch.tensor([[True,False]]));loss.backward()
        self.assertEqual(float(loss.detach()),0.)
        self.assertTrue(torch.equal(p.grad,torch.zeros_like(p)))


if __name__=='__main__':unittest.main()
