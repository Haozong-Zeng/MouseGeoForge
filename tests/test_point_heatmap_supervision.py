import unittest
import torch
from mousegeoforge.point_heatmap_supervision import gaussian_point_targets,point_heatmap_kl


class TestPointHeatmap(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);self.xy=torch.tensor([[[.35,.5],[.7,.7]]]);self.valid=torch.ones(1,2,dtype=torch.bool);self.pixels=torch.tensor([[704.,768.]])
    def test_identical_target_zero(self):
        p,good=gaussian_point_targets(self.xy,self.valid,self.pixels,(48,48))
        torch.testing.assert_close(p.sum((-2,-1)),torch.ones(1,2));self.assertTrue(good.all())
        self.assertLess(abs(float(point_heatmap_kl(p,self.xy,self.valid,self.pixels))),1e-6)
    def test_displaced_more_loss(self):
        p,_=gaussian_point_targets(self.xy+.1,self.valid,self.pixels,(48,48))
        self.assertGreater(float(point_heatmap_kl(p,self.xy,self.valid,self.pixels)),1.)
    def test_missing_not_supervised(self):
        p=torch.full((1,2,8,8),1/64,requires_grad=True);bad=torch.full_like(self.xy,float('nan'))
        loss=point_heatmap_kl(p,bad,self.valid,self.pixels);self.assertEqual(float(loss),0);loss.backward();self.assertEqual(float(p.grad.abs().sum()),0)
    def test_partial_and_finite_gradient(self):
        logits=torch.randn(1,2,8,8,requires_grad=True);p=logits.flatten(2).softmax(-1).reshape_as(logits);valid=torch.tensor([[True,False]])
        loss=point_heatmap_kl(p,self.xy,valid,self.pixels);loss.backward();self.assertTrue(torch.isfinite(logits.grad).all());self.assertEqual(float(logits.grad[:,1].abs().sum()),0)
    def test_target_coordinates_not_trainable(self):
        xy=self.xy.requires_grad_();target,_=gaussian_point_targets(xy,self.valid,self.pixels,(32,48));self.assertFalse(target.requires_grad)

    def test_stable_log_probability_keeps_tiny_mass_gradient(self):
        logits=torch.zeros(1,2,8,8);logits[:,:,3:6,2:6]=-100;logits.requires_grad_()
        logp=logits.flatten(2).log_softmax(-1).reshape_as(logits);p=logp.exp()
        valid=torch.tensor([[True,False]])
        loss=point_heatmap_kl(p,self.xy,valid,self.pixels,log_probability=logp)
        loss.backward();self.assertTrue(torch.isfinite(loss));self.assertGreater(float(logits.grad[0,0,3:6,2:6].abs().sum()),0)
        self.assertEqual(float(logits.grad[0,1].abs().sum()),0)


if __name__=='__main__':unittest.main()
