import unittest
import torch
from mousegeoforge.matched_point_loss import matched_image_point_loss

class MatchedPointTests(unittest.TestCase):
    def test_missing_nan_is_not_target(self):
        p=torch.zeros(1,2,2,requires_grad=True);g=torch.tensor([[[1.,1.],[float('nan'),float('nan')]]]);v=torch.tensor([[True,False]])
        loss=matched_image_point_loss(p,g,v,torch.tensor([[768.,768.]]),torch.tensor([0]),1);loss.backward();self.assertTrue(torch.isfinite(loss));self.assertEqual(float(p.grad[0,1].abs().sum()),0)
    def test_equal_image_weights_with_dual_instances(self):
        p=torch.zeros(3,1,2);g=torch.tensor([[[1.,1.]],[[1.,1.]],[[0.,0.]]]);v=torch.ones(3,1,dtype=torch.bool)
        loss=matched_image_point_loss(p,g,v,torch.full((3,2),768.),torch.tensor([0,0,1]),2)
        self.assertAlmostEqual(float(loss),.995/2,places=6)
    def test_empty_labels_zero_gradient(self):
        p=torch.zeros(1,2,2,requires_grad=True);loss=matched_image_point_loss(p,torch.full_like(p,float('nan')),torch.zeros(1,2,dtype=torch.bool),torch.full((1,2),768.),torch.tensor([0]),1);loss.backward();self.assertEqual(float(loss.detach()),0);self.assertEqual(float(p.grad.abs().sum()),0)
    def test_downweight_does_not_inflate_other_gradients(self):
        p=torch.zeros(1,2,2,requires_grad=True);g=torch.ones_like(p);v=torch.ones(1,2,dtype=torch.bool);pixels=torch.full((1,2),768.);ix=torch.tensor([0])
        full=matched_image_point_loss(p,g,v,pixels,ix,1);full.backward();old=p.grad.clone();p.grad=None
        weighted=matched_image_point_loss(p,g,v,pixels,ix,1,torch.tensor([[1.,0.]]));weighted.backward()
        torch.testing.assert_close(p.grad[:,0],old[:,0],atol=0,rtol=0);self.assertEqual(float(p.grad[:,1].abs().sum()),0)
    def test_weight_one_exact_and_invalid_rejected(self):
        p=torch.zeros(1,2,2);g=torch.ones_like(p);v=torch.ones(1,2,dtype=torch.bool);pixels=torch.full((1,2),768.);ix=torch.tensor([0])
        torch.testing.assert_close(matched_image_point_loss(p,g,v,pixels,ix,1),matched_image_point_loss(p,g,v,pixels,ix,1,torch.ones_like(v,dtype=torch.float32)),atol=0,rtol=0)
        with self.assertRaises(ValueError):matched_image_point_loss(p,g,v,pixels,ix,1,torch.tensor([[1.,-1.]]))

if __name__=='__main__':unittest.main()
