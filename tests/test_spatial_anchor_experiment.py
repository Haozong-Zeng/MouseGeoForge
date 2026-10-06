import unittest
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vggt-omega'))
from vggt_omega.models.heads.instance_pose_head import MultiInstanceMouseHead
from mousegeoforge.spatial_anchor_experiment import SpatialAnchorResidual,install_spatial_anchor,spatial_anchor_context


class SpatialAnchorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.m=SpatialAnchorResidual(8,3,4)
        self.raw=torch.randn(2,2,3,2)
        self.tokens=torch.randn(2,2,3,8)
        self.pixels=torch.randn(2,8,6,7)
        self.mask=torch.randn(2,2,6,7)

    def test_disabled_identity(self):self.assertIs(self.m(self.raw),self.raw)

    def test_zero_init_exact_finite_gate_gradient(self):
        got=self.m(self.raw,self.tokens,self.pixels,self.mask,enabled=True)
        self.assertTrue(torch.equal(got,self.raw))
        got.square().sum().backward()
        self.assertTrue(torch.isfinite(self.m.gate.grad).all())
        self.assertGreater(float(self.m.gate.grad.abs().sum()),0.)

    def test_projection_grad_after_gate_opens(self):
        with torch.no_grad():self.m.gate.fill_(.1)
        self.m(self.raw,self.tokens,self.pixels,self.mask,enabled=True).square().sum().backward()
        for p in self.m.parameters():self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(float(self.m.query.weight.grad.abs().sum()),0.)

    def test_no_upstream_gradient_through_evidence(self):
        with torch.no_grad():self.m.gate.fill_(.1)
        self.tokens.requires_grad_();self.pixels.requires_grad_();self.mask.requires_grad_()
        self.m(self.raw,self.tokens,self.pixels,self.mask,enabled=True).sum().backward()
        for p in (self.tokens,self.pixels,self.mask):self.assertIsNone(p.grad)

    def test_permuting_query_slots_commutes(self):
        with torch.no_grad():self.m.gate.fill_(.2)
        a=self.m(self.raw,self.tokens,self.pixels,self.mask,enabled=True)
        b=self.m(self.raw.flip(1),self.tokens.flip(1),self.pixels,self.mask.flip(1),enabled=True)
        torch.testing.assert_close(a.flip(1),b)

    def test_bad_geometry_rejected(self):
        with self.assertRaises(ValueError):self.m(self.raw,self.tokens,self.pixels,self.mask[:,:1],enabled=True)


class SpatialAnchorIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17);torch.set_num_threads(2)
        self.head=MultiInstanceMouseHead(num_keypoints=3,num_queries=2,dim_in=64,hidden_dim=32,num_heads=4).eval()
        self.model=SimpleNamespace(instance_pose_head=self.head)
        install_spatial_anchor(self.model);self.head.eval()

    def inputs(self,views):
        cache=[None]*24
        for i in self.head.intermediate_layer_idx:cache[i]=torch.randn(1,views,17,64)
        return cache,torch.randn(1,views,3,64,64)

    def test_one_and_three_views_zero_equal(self):
        for views in (1,3):
            cache,image=self.inputs(views)
            with torch.no_grad():
                off=self.head(cache,image,1)
                with spatial_anchor_context(self.model,enabled=True):on=self.head(cache,image,1)
            for k in off:torch.testing.assert_close(off[k],on[k],rtol=0,atol=0)
            self.assertFalse(self.head.coarse_keypoint_xy._forward_hooks)

    def test_active_branch_only_changes_pose_path(self):
        cache,image=self.inputs(2)
        with torch.no_grad():
            off=self.head(cache,image,1);self.head.spatial_anchor.gate.fill_(.2)
            with spatial_anchor_context(self.model,enabled=True):on=self.head(cache,image,1)
        self.assertFalse(torch.equal(on['instance_keypoint_coarse_xy'],off['instance_keypoint_coarse_xy']))
        for k in ('instance_mask_logits','instance_boxes_cxcywh','instance_objectness_logits','instance_keypoint_visibility_logits'):
            torch.testing.assert_close(on[k],off[k],atol=0,rtol=0)

    def test_context_exception_cleans_hooks(self):
        with self.assertRaisesRegex(RuntimeError,'deliberate'):
            with spatial_anchor_context(self.model,enabled=True):raise RuntimeError('deliberate')
        self.assertFalse(self.head.coarse_keypoint_xy._forward_hooks)
        self.assertFalse(self.head.pyramid_refine[0]._forward_hooks)
        self.assertFalse(self.head.mask_embedding._forward_hooks)


if __name__=='__main__':unittest.main()
