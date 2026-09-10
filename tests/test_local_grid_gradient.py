import unittest
import torch
from vggt_omega.models.heads.instance_pose_head import MultiInstanceMouseHead


class TestLocalGridGradient(unittest.TestCase):
    def test_detached_grid_forward_identity_and_gradient_routing(self):
        torch.manual_seed(17)
        head=MultiInstanceMouseHead(num_keypoints=5,num_queries=2,dim_in=32,hidden_dim=16,num_heads=4)
        pixels=torch.randn(1,16,16,16,requires_grad=True)
        tokens=torch.randn(1,2,5,16,requires_grad=True)
        center=(torch.rand(1,2,5,2)*.6+.2).requires_grad_()
        original_logits,original_xy=head._local_refinement(pixels,tokens,center)
        grad=torch.autograd.grad(original_logits.square().sum(),center,retain_graph=True)[0]
        self.assertGreater(float(grad.abs().sum()),0.)
        head.detach_local_sampling_grid=True
        logits,xy=head._local_refinement(pixels,tokens,center)
        torch.testing.assert_close(logits,original_logits,atol=0,rtol=0)
        torch.testing.assert_close(xy,original_xy,atol=0,rtol=0)
        self.assertIsNone(torch.autograd.grad(logits.square().sum(),center,allow_unused=True,retain_graph=True)[0])
        direct=torch.autograd.grad(xy.sum(),center,retain_graph=True)[0]
        torch.testing.assert_close(direct,torch.ones_like(center),atol=0,rtol=0)
        grads=torch.autograd.grad(logits.square().sum(),[pixels,tokens])
        self.assertTrue(all(torch.isfinite(g).all() and g.abs().sum()>0 for g in grads))
