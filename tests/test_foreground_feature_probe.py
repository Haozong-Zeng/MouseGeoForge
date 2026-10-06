import unittest
import torch
from mousegeoforge.foreground_feature_probe import ForegroundLinearProbe, balanced_foreground_loss, mask_counts, ratios


class ForegroundProbeTests(unittest.TestCase):
    def test_zero_rng_and_shapes(self):
        rng=torch.get_rng_state().clone();m=ForegroundLinearProbe(4)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        y=m(torch.randn(3,4,2,5));self.assertEqual(tuple(y.shape),(3,2,5))
        self.assertEqual(int(y.count_nonzero()),0)

    def test_fractional_gradients(self):
        m=ForegroundLinearProbe(4);f=torch.randn(2,4,3,5)
        loss=balanced_foreground_loss(m(f),torch.rand(2,3,5));loss.backward()
        self.assertTrue(torch.isfinite(m.weight.grad).all())
        self.assertGreater(float(m.weight.grad.norm()),0)

    def test_empty_classes_finite(self):
        for v in (0.,1.):
            logits=torch.zeros(1,2,2,requires_grad=True)
            loss=balanced_foreground_loss(logits,torch.full_like(logits,v));loss.backward()
            self.assertTrue(torch.isfinite(logits.grad).all())

    def test_fixed_threshold_counts(self):
        c=mask_counts(torch.tensor([.5,.9,.1]),torch.tensor([True,False,True]))
        self.assertEqual(c,dict(tp=1,fp=1,fn=1));self.assertEqual(ratios(c)['iou'],1/3)


if __name__=='__main__':unittest.main()
