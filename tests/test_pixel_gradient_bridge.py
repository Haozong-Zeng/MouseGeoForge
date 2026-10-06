import unittest
import torch
from torch import nn
from mousegeoforge.models.low_rank_linear import LowRankLinear
from mousegeoforge.models.decoder_spatial_pose_readout import DecoderSpatialPoseReadout


class PixelGradientTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(17);torch.set_num_threads(2)

    def test_lora_off_and_zero_initial_exact(self):
        base=nn.Linear(16,48);x=torch.randn(2,4,16);expected=base(x).detach();layer=LowRankLinear(base,rank=2,alpha=2)
        torch.testing.assert_close(layer(x),expected,atol=0,rtol=0);layer.enabled=True
        torch.testing.assert_close(layer(x),expected,atol=0,rtol=0)

    def test_zero_initial_then_both_gradients(self):
        layer=LowRankLinear(nn.Linear(16,48),rank=2);layer.enabled=True;x=torch.randn(2,4,16);opt=torch.optim.SGD([layer.a.weight,layer.b.weight],lr=.1)
        layer(x).square().mean().backward();self.assertGreater(float(layer.b.weight.grad.norm()),0);self.assertEqual(float(layer.a.weight.grad.norm()),0)
        self.assertIsNone(layer.base.weight.grad);opt.step();opt.zero_grad();layer(x).square().mean().backward()
        self.assertGreater(float(layer.a.weight.grad.norm()),0)

    def test_disable_after_update_restores_base(self):
        layer=LowRankLinear(nn.Linear(8,24));layer.b.weight.data.normal_();x=torch.randn(2,8)
        torch.testing.assert_close(layer(x),layer.base(x),atol=0,rtol=0)

    def test_custom_linear_forward_preserved(self):
        class Masked(nn.Linear):
            def forward(self,x):return nn.functional.linear(x,self.weight,self.bias*0)
        base=Masked(8,24);layer=LowRankLinear(base);layer.enabled=True;x=torch.randn(2,8)
        torch.testing.assert_close(layer(x),base(x),atol=0,rtol=0)

    def test_decoder_default_detach_and_optin_gradient(self):
        model=DecoderSpatialPoseReadout().eval()
        # Nonzero key-query readout needed: initial zero query otherwise gives no appearance gradient.
        model.spatial_readout.query[-1].weight.data.normal_(std=.02)
        instance=torch.randn(1,6,256,requires_grad=True);memory=torch.randn(1,256,4,5,requires_grad=True);boxes=torch.rand(1,6,4)
        old=model(instance,memory,boxes)['xy'];old.square().mean().backward()
        self.assertIsNone(instance.grad);self.assertIsNone(memory.grad);model.zero_grad()
        with torch.no_grad():
            a=model(instance,memory,boxes)['xy'];b=model(instance,memory,boxes,allow_input_grad=True)['xy']
            torch.testing.assert_close(a,b,atol=0,rtol=0)
        # Input gradients switch PyTorch MHA's fastpath; record normal fp32
        # kernel roundoff, not a claim of bitwise equality across grad modes.
        new=model(instance,memory,boxes,allow_input_grad=True)['xy'];torch.testing.assert_close(old,new,atol=5e-7,rtol=0)
        new.square().mean().backward();self.assertGreater(float(instance.grad.norm()),0);self.assertGreater(float(memory.grad.norm()),0)
        self.assertEqual(new.shape,(1,6,23,2))

if __name__=='__main__':unittest.main()
