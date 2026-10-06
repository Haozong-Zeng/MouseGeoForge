import unittest
import torch
from mousegeoforge.roi_candidate_quality import RegionQualityProbe
from mousegeoforge.roi_set_objectness import SetObjectnessProbe,set_objectness_targets


class SetObjectnessTests(unittest.TestCase):
    def test_one_per_target_and_ignores_score_gate(self):
        gt=torch.tensor([[[.2,.3,.2,.2],[.8,.7,.2,.2]]]);b=torch.cat((gt,gt[:,:1]),1)
        s=torch.tensor([[.2,.1,.15]])
        labels=set_objectness_targets(b,s,gt,torch.ones(1,2,dtype=torch.bool))
        self.assertEqual(labels.tolist(),[[1.,1.,0.]])

    def test_empty_and_bad_geometry(self):
        gt=torch.tensor([[[.2,.3,.2,.2]]]);b=torch.tensor([[[.8,.8,.1,.1]]]);s=torch.ones(1,1)
        for valid in (False,True):self.assertEqual(float(set_objectness_targets(b,s,gt,torch.tensor([[valid]])).sum()),0.)

    def test_off_exact(self):
        old=RegionQualityProbe(16,8);new=SetObjectnessProbe(16,8,False)
        new.load_state_dict(old.state_dict(),strict=False);x=torch.randn(2,6,37)
        torch.testing.assert_close(old(x),new(x),atol=0,rtol=0)

    def test_permutation_equivariance(self):
        model=SetObjectnessProbe(16,8,True).eval();x=torch.randn(2,6,37);order=torch.tensor([3,1,5,0,4,2])
        torch.testing.assert_close(model(x[:,order]),model(x)[:,order],atol=1e-6,rtol=1e-6)

    def test_finite_gradient_views_and_off_frozen_branch(self):
        for enabled in (False,True):
            for views in (1,2,4):
                model=SetObjectnessProbe(16,8,enabled);x=torch.randn(views,6,37)
                torch.nn.functional.binary_cross_entropy_with_logits(model(x),torch.rand(views,6)).backward()
                for p in model.parameters():
                    if p.requires_grad:self.assertTrue(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0)
                    else:self.assertIsNone(p.grad)

    def test_no_cross_view_attention(self):
        model=SetObjectnessProbe(16,8,True).eval();x=torch.randn(4,6,37)
        torch.testing.assert_close(model(x),torch.cat([model(v[None]) for v in x]),atol=1e-6,rtol=1e-6)


if __name__=='__main__':unittest.main()
