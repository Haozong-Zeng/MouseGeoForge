import unittest
import torch
from mousegeoforge.models.point_localization_quality import PointLocalizationQuality,masked_quality_loss


class PointQualityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(17)
        self.tokens=torch.randn(2,6,5,256);self.scalar=torch.randn(2,6,5,12);self.prior=torch.rand(2,6,5)
        self.model=PointLocalizationQuality()

    def test_off_exact(self):self.assertIs(self.model(self.tokens,self.scalar,self.prior)['probability'],self.prior)

    def test_order_and_input_unchanged(self):
        before=self.tokens.clone();self.model.score[-1].weight.data.normal_()
        a=self.model(self.tokens,self.scalar,self.prior,enabled=True)['probability']
        b=self.model(self.tokens.flip(1),self.scalar.flip(1),self.prior.flip(1),enabled=True)['probability']
        torch.testing.assert_close(a,b.flip(1),atol=0,rtol=0);torch.testing.assert_close(before,self.tokens,atol=0,rtol=0)

    def test_unknown_not_negative(self):
        logits=torch.randn(2,6,5,requires_grad=True);target=torch.rand_like(logits).round();valid=torch.ones_like(logits,dtype=torch.bool);valid[:,2]=False;target[:,2]=float('nan')
        loss=masked_quality_loss(logits,target,valid);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertEqual(float(logits.grad[:,2].abs().max()),0)

    def test_no_known_zero(self):
        logits=torch.randn(2,6,5,requires_grad=True);loss=masked_quality_loss(logits,torch.full_like(logits,float('nan')),torch.zeros_like(logits,dtype=torch.bool))
        loss.backward();self.assertEqual(float(loss.detach()),0);self.assertEqual(float(logits.grad.abs().max()),0)

    def test_scalar_ignores_semantics(self):
        model=PointLocalizationQuality('scalar');model.score[-1].weight.data.normal_()
        a=model(self.tokens,self.scalar,self.prior,enabled=True)['probability'];b=model(self.tokens*100,self.scalar,self.prior,enabled=True)['probability']
        torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_semantic_gradient_and_no_input_gradient(self):
        self.tokens.requires_grad_(True);self.model.score[-1].weight.data.normal_()
        out=self.model(self.tokens,self.scalar,self.prior,enabled=True);target=torch.ones_like(out['logits'])
        loss=masked_quality_loss(out['logits'],target,torch.ones_like(target,dtype=torch.bool));loss.backward()
        self.assertGreater(float(self.model.semantic[1].weight.grad.abs().sum()),0);self.assertIsNone(self.tokens.grad)

    def test_invalid_known_target(self):
        with self.assertRaises(ValueError):masked_quality_loss(torch.zeros(1,6,5),torch.full((1,6,5),float('nan')),torch.ones(1,6,5,dtype=torch.bool))

if __name__=='__main__':unittest.main()

