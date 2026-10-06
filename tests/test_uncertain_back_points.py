import unittest
import torch
from mousegeoforge.uncertain_back_points import back_axes,uncertain_back_point_loss


class UncertainBackTests(unittest.TestCase):
    def setUp(self):self.xy=torch.tensor([[100.,200.],[160.,200.]]);self.valid=torch.ones(2,dtype=torch.bool)
    def test_default_no_fallback(self):
        t,r=back_axes(self.xy,self.valid);self.assertEqual(float(r.sum()),0.)
    def test_capped_fallback_direction(self):
        t,r=back_axes(self.xy,self.valid,use_pair_fallback=True)
        self.assertTrue(torch.equal(r,torch.full((2,),.5)));self.assertTrue(torch.equal(t,torch.tensor([[1.,0.],[1.,0.]])))
    def test_primary_retained(self):
        t,r=back_axes(self.xy,self.valid,primary_tangent=torch.tensor([[0.,2.],[0.,0.]]),primary_valid=torch.tensor([True,False]),use_pair_fallback=True)
        self.assertTrue(torch.equal(t[0],torch.tensor([0.,1.])));self.assertTrue(torch.equal(r,torch.tensor([1.,.5])))
    def test_missing_and_short_pair_no_direction(self):
        for xy,valid in [(self.xy,torch.tensor([True,False])),(self.xy*.1,self.valid)]:
            self.assertEqual(float(back_axes(xy,valid,use_pair_fallback=True)[1].sum()),0.)
    def test_along_less_than_normal_and_finite_gradient(self):
        t,r=back_axes(self.xy,self.valid,use_pair_fallback=True)
        fn=lambda p:uncertain_back_point_loss(p,self.xy,self.valid,tangent=t,reliability=r)
        along=fn(self.xy+torch.tensor([20.,0.]));p=(self.xy+torch.tensor([0.,20.])).requires_grad_();normal=fn(p);normal.backward()
        self.assertLess(float(along),float(normal.detach())*.5);self.assertTrue(torch.isfinite(p.grad).all())
    def test_all_missing_zero(self):
        xy=torch.full((2,2),float('nan'));p=torch.zeros((2,2),requires_grad=True);valid=torch.zeros(2,dtype=torch.bool);t,r=back_axes(xy,valid)
        value=uncertain_back_point_loss(p,xy,valid,tangent=t,reliability=r);value.backward();self.assertEqual(float(value.detach()),0.)
    def test_axes_do_not_backpropagate_into_labels(self):
        t,r=back_axes(self.xy.clone().requires_grad_(),self.valid,use_pair_fallback=True);self.assertFalse(t.requires_grad);self.assertFalse(r.requires_grad)


if __name__=='__main__':unittest.main()
