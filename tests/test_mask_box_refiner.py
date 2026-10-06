import unittest
import numpy as np
import torch
from mousegeoforge.mask_box_refiner import MaskConditionedBoxRefiner,binary_mask_features,features_from_logits


class MaskBoxRefinerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123);self.model=MaskConditionedBoxRefiner(8,16)
        self.box=torch.rand(1,2,3,4)*.7+.1;self.token=torch.rand(1,2,3,8,requires_grad=True)
        self.shape=torch.rand(1,2,3,16,requires_grad=True)
    def test_disabled_and_both_zero_initializations_exact(self):
        self.assertIs(self.model(self.box),self.box)
        for use_mask in (False,True):
            torch.testing.assert_close(self.model(self.box,self.token,self.shape,enabled=True,use_mask=use_mask),self.box,atol=0,rtol=0)
    def test_gradient_and_frozen_features(self):
        opt=torch.optim.AdamW(self.model.parameters(),lr=.01)
        for step in range(3):
            opt.zero_grad();p=self.model(self.box,self.token,self.shape,enabled=True,use_mask=True)
            loss=(p-self.box*.9).square().mean();loss.backward()
            self.assertTrue(all(torch.isfinite(x.grad).all() for x in self.model.parameters()))
            self.assertGreater(float(self.model.output.weight.grad.norm()),0)
            if step:self.assertGreater(float(self.model.input.weight.grad.norm()),0)
            opt.step()
        self.assertIsNone(self.token.grad);self.assertIsNone(self.shape.grad)
    def test_off_ignores_shape_even_after_training(self):
        torch.nn.init.normal_(self.model.output.weight)
        a=self.model(self.box,self.token,self.shape,enabled=True,use_mask=False)
        b=self.model(self.box,self.token,torch.ones_like(self.shape)*100,enabled=True,use_mask=False)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    def test_components_preserved_not_filtered(self):
        mask=np.zeros((20,30),bool);mask[5:12,10:20]=True;mask[15:17,2:7]=True
        f=binary_mask_features(mask)
        self.assertLess(f[9],1);self.assertLess(f[11],1)
        self.assertGreater(f[2],f[6]);self.assertGreater(f[3],f[7])
    def test_noise_features_and_empty_fallback_identity(self):
        logits=torch.full((1,1,1,20,30),-2.);logits[...,5:12,10:20]=2
        clean=features_from_logits(logits,(20,30));noisy=features_from_logits(logits,(20,30),corner_noise=True)
        torch.testing.assert_close(clean[...,4:8],noisy[...,4:8],atol=0,rtol=0)
        self.assertEqual(float(noisy[...,2]),1);self.assertEqual(float(noisy[...,3]),1)
        empty=features_from_logits(torch.full_like(logits,-2.),(20,30));self.assertTrue(torch.isfinite(empty).all());self.assertEqual(float(empty.abs().sum()),0)
    def test_query_permutation_and_validation(self):
        order=[2,0,1];torch.nn.init.normal_(self.model.output.weight,std=.01)
        a=self.model(self.box,self.token,self.shape,enabled=True,use_mask=True)
        b=self.model(self.box[:,:,order],self.token[:,:,order],self.shape[:,:,order],enabled=True,use_mask=True)
        torch.testing.assert_close(a[:,:,order],b,atol=0,rtol=0)
        with self.assertRaises(ValueError):self.model(self.box,self.token,None,enabled=True,use_mask=True)


if __name__=='__main__':unittest.main()
