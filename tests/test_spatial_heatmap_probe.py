import unittest
import torch
from mousegeoforge.spatial_heatmap_probe import spatial_expectation


class SpatialHeatmapProbeTests(unittest.TestCase):
    def test_off_identity(self):
        sentinel=object()
        self.assertIs(spatial_expectation(sentinel),sentinel)

    def test_uniform_center_multiple_views_queries(self):
        old=torch.zeros(1,3,6,23,2)
        got=spatial_expectation(old,torch.zeros(1,3,6,23,8,12),enabled=True)
        torch.testing.assert_close(got,torch.full_like(old,.5))

    def test_translation_of_peak(self):
        p=torch.full((1,1,1,16,20),-80.)
        p[0,0,0,5,6]=0
        a=spatial_expectation(torch.zeros(1,1,1,2),p,enabled=True)
        b=spatial_expectation(torch.zeros(1,1,1,2),p.roll(3,-1),enabled=True)
        torch.testing.assert_close(b-a,torch.tensor([[[[3/20,0.]]]]))

    def test_finite_gradients_soft_prior(self):
        logits=torch.randn(2,3,4,8,8,requires_grad=True)
        mask=torch.randn(2,3,8,8,requires_grad=True)
        out=spatial_expectation(torch.zeros(2,3,4,2),logits,enabled=True,mask_logits=mask)
        out.square().sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.isfinite(mask.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()),0)

    def test_reject_wrong_mask(self):
        with self.assertRaises(ValueError):
            spatial_expectation(torch.zeros(2,3,4,2),torch.zeros(2,3,4,8,8),enabled=True,mask_logits=torch.zeros(2,8,8))


if __name__=='__main__':unittest.main()
