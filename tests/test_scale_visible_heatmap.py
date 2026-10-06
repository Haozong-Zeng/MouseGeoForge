import unittest
import torch
from mousegeoforge.visible_heatmap_supervision import visible_gaussian_kl
from mousegeoforge.scale_visible_heatmap import body_sigma,variable_visible_kl


class TestScaleHeatmap(unittest.TestCase):
    def test_constant_sigma_exact_loss_and_gradient(self):
        torch.manual_seed(17);scores=torch.randn(1,2,23,9,11,dtype=torch.float64,requires_grad=True)
        xy=torch.rand(1,2,23,2,dtype=torch.float64);valid=torch.rand(1,2,23)>.3
        a=visible_gaussian_kl(scores,xy,valid,image_hw=(90,110));b=variable_visible_kl(scores,xy,valid,torch.full_like(valid,8.,dtype=torch.float64),image_hw=(90,110))
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        ga=torch.autograd.grad(a,scores)[0];gb=torch.autograd.grad(b,scores)[0];torch.testing.assert_close(ga,gb,atol=0,rtol=0)

    def test_width_bounds_fallback_and_tail_exclusion(self):
        xy=torch.zeros(3,23,2);valid=torch.zeros(3,23,dtype=torch.bool);valid[:,:2]=True
        xy[0,1,0]=.1;xy[1,1,0]=.001;xy[2,1,0]=1
        xy[:,7:14]=1000;valid[:,7:14]=True
        sigma,fallback=body_sigma(xy,valid,image_hw=(100,1000));torch.testing.assert_close(sigma,torch.tensor([3.5,2.,8.]));self.assertFalse(fallback.any())
        valid[:,:2]=False;sigma,fallback=body_sigma(xy,valid,image_hw=(100,1000));self.assertTrue(fallback.all());torch.testing.assert_close(sigma,torch.ones(3)*8)

    def test_invalid_points_and_no_target_gradients(self):
        scores=torch.randn(1,1,23,5,7,requires_grad=True);xy=torch.ones(1,1,23,2,requires_grad=True)*.5;valid=torch.zeros(1,1,23,dtype=torch.bool);valid[...,0]=True
        sigma=torch.ones(1,1,23,requires_grad=True)*4
        loss=variable_visible_kl(scores,xy,valid,sigma,image_hw=(50,70));g=torch.autograd.grad(loss,[scores,xy,sigma],allow_unused=True)
        self.assertIsNone(g[1]);self.assertIsNone(g[2]);self.assertEqual(float(g[0][~valid].abs().sum()),0.)


if __name__=='__main__':unittest.main()
