import unittest
import torch
from mousegeoforge.gate_losses import refined_coordinate_loss


class TestGateLoss(unittest.TestCase):
    def test_matching_and_invalid_points(self):
        pred=torch.zeros(1,2,3,2,requires_grad=True)
        target=torch.ones(1,1,2,3,2)*.1;valid=torch.zeros(1,1,2,3,dtype=torch.bool);valid[0,0,1,0]=True
        sample=dict(matches=[([0],[1])],target=dict(keypoint_xy=target,keypoint_valid=valid))
        loss=refined_coordinate_loss(pred,sample);self.assertAlmostEqual(float(loss.detach()),.075,places=6)
        loss.backward();self.assertGreater(float(pred.grad[0,0,0].abs().sum()),0.)
        self.assertEqual(float(pred.grad[0,1].abs().sum()),0.);self.assertEqual(float(pred.grad[0,0,1:].abs().sum()),0.)

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):refined_coordinate_loss(torch.zeros(1,1,3,2),dict(matches=[([],[])],target={}))


if __name__=='__main__':unittest.main()
