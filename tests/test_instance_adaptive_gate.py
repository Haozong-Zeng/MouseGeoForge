import unittest
import torch
from mousegeoforge.instance_adaptive_gate import InstanceAdaptiveGate,predicted_quality


class TestInstanceGate(unittest.TestCase):
    def sample(self):
        torch.manual_seed(17)
        return (torch.randn(3,2,23,2),torch.rand(3,2,23,2)*.8+.1,torch.rand(23)*.2+.35,
                torch.randn(3,2,23,8),torch.rand(3,2,23,5))

    def test_initial_identity_and_gradient(self):
        args=self.sample()
        for dynamic in (False,True):
            m=InstanceAdaptiveGate(token_dim=8,dynamic=dynamic)
            out,g=m(*args);expected=args[0]+args[2][None,None,:,None]*(torch.logit(args[1])-args[0])
            torch.testing.assert_close(out,expected,atol=0,rtol=0)
            out.square().mean().backward();self.assertTrue(torch.isfinite(m.offset.grad).all());self.assertGreater(float(m.offset.grad.abs().sum()),0.)
            if dynamic:self.assertGreater(float(m.network[-1].weight.grad.abs().sum()),0.)

    def test_bounds_and_detached_upstream(self):
        args=list(self.sample());args=[x.requires_grad_(True) for x in args]
        m=InstanceAdaptiveGate(token_dim=8)
        with torch.no_grad():m.offset.copy_(torch.linspace(-100,100,23))
        out,g=m(*args);self.assertTrue(((g>=0)&(g<=1)).all());out.square().mean().backward()
        self.assertTrue(all(x.grad is None for x in args))

    def test_query_permutation_and_single_view(self):
        m=InstanceAdaptiveGate(token_dim=8);args=self.sample()
        with torch.no_grad():m.network[-1].weight.fill_(.02)
        out,_=m(*args);other,_=m(args[0].flip(1),args[1].flip(1),args[2],args[3].flip(1),args[4].flip(1))
        torch.testing.assert_close(other,out.flip(1),atol=0,rtol=0)
        single,_=m(args[0][:1],args[1][:1],args[2],args[3][:1],args[4][:1]);torch.testing.assert_close(single,out[:1])

    def test_quality_no_labels(self):
        quality=predicted_quality(torch.zeros(1,2,23,4,5),torch.zeros(1,2,23),torch.zeros(1,2,4,5),torch.zeros(1,2))
        self.assertEqual(quality.shape,(1,2,23,5));self.assertFalse(quality.requires_grad)
        torch.testing.assert_close(quality[...,0],torch.ones(1,2,23));torch.testing.assert_close(quality[...,1],torch.ones(1,2,23)/20)


if __name__=='__main__':unittest.main()
