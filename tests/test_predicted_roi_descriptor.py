import unittest
import torch
from mousegeoforge.predicted_roi_descriptor import pool_predicted_roi


class ROITests(unittest.TestCase):
    def test_constant(self):
        out=pool_predicted_roi(torch.ones(1,3,20,20),torch.tensor([[[.5,.5,.4,.4]]]))
        torch.testing.assert_close(out['descriptor'],torch.ones(1,1,3));self.assertTrue(out['valid'].all())
    def test_query_order(self):
        f=torch.rand(1,5,20,20);b=torch.tensor([[[.3,.3,.3,.3],[.7,.7,.2,.2]]])
        a=pool_predicted_roi(f,b);c=pool_predicted_roi(f,b.flip(1));torch.testing.assert_close(a['descriptor'],c['descriptor'].flip(1))
    def test_outside_abstains(self):
        out=pool_predicted_roi(torch.ones(1,3,20,20),torch.tensor([[[2.,2.,.4,.4]]]))
        self.assertFalse(out['valid'].any());self.assertEqual(float(out['descriptor'].abs().sum()),0)
    def test_partial_box(self):
        out=pool_predicted_roi(torch.ones(1,3,20,20),torch.tensor([[[0.,.5,.4,.4]]]))
        self.assertTrue(out['valid'].all());self.assertLess(float(out['sampled_fraction'][0,0]),1)
    def test_invalid(self):
        with self.assertRaises(ValueError):pool_predicted_roi(torch.ones(1,3,20,20),torch.tensor([[[.5,.5,0.,.4]]]))


if __name__=='__main__':unittest.main()
