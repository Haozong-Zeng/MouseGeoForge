import unittest
import torch
from mousegeoforge.sparse_surface_observations import sparse_back_observation_loss as loss


class SparseSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.curve=torch.stack([torch.linspace(0,100,33),torch.zeros(33)],-1)
        self.points=torch.tensor([[30.,0.],[70.,0.]])
        self.valid=torch.ones(2,dtype=torch.bool)

    def test_sliding_not_fixed_fractions(self):
        self.assertLess(float(loss(self.curve,torch.tensor([[12.,0.],[87.,0.]]),self.valid)),1e-10)

    def test_normal_error_penalized(self):
        self.assertGreater(float(loss(self.curve,self.points+torch.tensor([0.,10.]),self.valid)),float(loss(self.curve,self.points,self.valid)))

    def test_reversal_and_distinct_point_collapse(self):
        correct=loss(self.curve,self.points,self.valid)
        self.assertGreater(float(loss(self.curve.flip(0),self.points,self.valid)),float(correct)+1e-4)
        self.assertGreater(float(loss(torch.ones_like(self.curve)*torch.tensor([50.,0.]),self.points,self.valid)),float(correct)+1e-4)

    def test_missing_has_no_fake_target(self):
        nan=torch.full((2,2),float('nan'));c=self.curve.clone().requires_grad_()
        result=loss(c,nan,torch.zeros(2,dtype=torch.bool));result.backward()
        self.assertEqual(float(result.detach()),0.);self.assertTrue(torch.equal(c.grad,torch.zeros_like(c)))

    def test_one_point_no_order_or_endpoint_invention(self):
        points=self.points.clone();points[1]=float('nan');valid=torch.tensor([True,False])
        self.assertLess(float(loss(self.curve,points,valid)),1e-10)
        self.assertLess(float(loss(self.curve.flip(0),points,valid)),1e-10)

    def test_unobserved_extension_is_not_identified(self):
        extended=self.curve*2-torch.tensor([50.,0.])
        self.assertAlmostEqual(float(loss(extended,self.points,self.valid)),float(loss(self.curve,self.points,self.valid)),places=9)

    def test_finite_gradient_at_exact_fit_and_offline(self):
        for offset in [0.,7.]:
            c=self.curve.clone().requires_grad_();out=loss(c,self.points+torch.tensor([0.,offset]),self.valid);out.backward()
            self.assertTrue(torch.isfinite(c.grad).all())


if __name__=='__main__':unittest.main()
