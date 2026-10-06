import unittest
import torch
from mousegeoforge.visible_heatmap_supervision import visible_gaussian_kl,anchor_scores
from mousegeoforge.spatial_anchor_experiment import SpatialAnchorResidual


class TestVisibleHeatmaps(unittest.TestCase):
    def test_correct_gaussian_and_wrong_location(self):
        y,x=torch.meshgrid((torch.arange(9)+.5)/9,(torch.arange(11)+.5)/11,indexing='ij')
        scores=(-.5*(((x-.5)*110/8)**2+((y-.5)*90/8)**2))[None,None,None].requires_grad_(True)
        xy=torch.tensor([[[[.5,.5]]]]);valid=torch.ones(1,1,1,dtype=torch.bool)
        exact=visible_gaussian_kl(scores,xy,valid,image_hw=(90,110));self.assertLess(abs(float(exact)),1e-6)
        shifted=visible_gaussian_kl(scores,xy+.2,valid,image_hw=(90,110));self.assertGreater(float(shifted),1.)

    def test_invalid_channels_have_zero_gradient(self):
        scores=torch.randn(1,2,3,7,9,requires_grad=True);xy=torch.ones(1,2,3,2)*.4
        valid=torch.zeros(1,2,3,dtype=torch.bool);valid[0,1,2]=True;xy[0,0]=float('nan')
        loss=visible_gaussian_kl(scores,xy,valid,image_hw=(70,90));loss.backward()
        self.assertEqual(float(scores.grad[~valid].abs().sum()),0.);self.assertGreater(float(scores.grad[valid].abs().sum()),0.)

    def test_all_invalid_and_upstream_detached(self):
        b=SpatialAnchorResidual(dim=8,keypoints=3,projection_dim=4)
        tokens=torch.randn(1,2,3,8,requires_grad=True);pixels=torch.randn(1,8,7,9,requires_grad=True);mask=torch.randn(1,2,7,9,requires_grad=True)
        scores=anchor_scores(b,tokens,pixels,mask)
        loss=visible_gaussian_kl(scores,torch.zeros(1,2,3,2),torch.zeros(1,2,3,dtype=torch.bool),image_hw=(70,90));loss.backward()
        self.assertEqual(float(loss),0.);self.assertIsNone(tokens.grad);self.assertIsNone(pixels.grad);self.assertIsNone(mask.grad)


if __name__=='__main__':unittest.main()
