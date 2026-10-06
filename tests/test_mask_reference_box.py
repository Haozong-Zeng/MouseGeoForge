import unittest
import numpy as np
import torch
from mousegeoforge.mask_reference_box import MaskReferenceBoxRefiner
from mousegeoforge.mask_box_refiner import MaskConditionedBoxRefiner,binary_mask_features


class MaskReferenceBoxTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17);self.model=MaskReferenceBoxRefiner(8,16)
        self.box=torch.tensor([[[[.4,.5,.2,.3]]]])
        self.token=torch.rand(1,1,1,8,requires_grad=True)
        mask=np.zeros((30,40),bool);mask[10:20,20:32]=True;mask[4:6,7:12]=True
        self.shape=torch.tensor(binary_mask_features(mask)).reshape(1,1,1,16)
    def test_identity_and_same_initial_base(self):
        torch.manual_seed(17);base=MaskConditionedBoxRefiner(8,16)
        for k,v in base.state_dict().items():torch.testing.assert_close(v,self.model.state_dict()[k],atol=0,rtol=0)
        self.assertIs(self.model(self.box),self.box)
        for enabled in (False,True):torch.testing.assert_close(self.model(self.box,self.token,self.shape,enabled=True,use_prior=enabled),self.box,atol=0,rtol=0)
    def test_off_equals_trained_base_and_ignores_prior_parameters(self):
        torch.nn.init.normal_(self.model.output.weight,std=.02)
        base=MaskConditionedBoxRefiner(8,16);base.load_state_dict({k:v for k,v in self.model.state_dict().items() if not k.startswith('reference_gate.')})
        torch.nn.init.normal_(self.model.reference_gate.weight)
        torch.testing.assert_close(self.model(self.box,self.token,self.shape,enabled=True),base(self.box,self.token,self.shape,enabled=True,use_mask=True),atol=0,rtol=0)
    def test_gate_receives_gradient_at_zero_without_feature_gradient(self):
        p=self.model(self.box,self.token,self.shape,enabled=True,use_prior=True)
        (p-torch.tensor([.6,.5,.3,.3])).square().sum().backward()
        self.assertGreater(float(self.model.reference_gate.weight.grad.norm()),0)
        self.assertTrue(torch.isfinite(self.model.reference_gate.weight.grad).all());self.assertIsNone(self.token.grad)
    def test_empty_mask_and_two_references(self):
        torch.nn.init.constant_(self.model.reference_gate.bias,.5)
        empty=torch.zeros_like(self.shape)
        a=self.model(self.box,self.token,empty,enabled=True,use_prior=True)
        b=self.model(self.box,self.token,empty,enabled=True,use_prior=False)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        coeff=.5*torch.tanh(torch.tensor(.5))
        expected=self.box+coeff*((self.shape[...,:4]-self.box)+(self.shape[...,4:8]-self.box))
        torch.testing.assert_close(self.model(self.box,self.token,self.shape,enabled=True,use_prior=True),expected)


if __name__=='__main__':unittest.main()
