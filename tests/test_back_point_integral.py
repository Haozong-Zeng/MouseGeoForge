import unittest
import torch
from mousegeoforge.models.back_point_integral import BackPointIntegralProbe


class TestIntegralProbe(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7);torch.set_num_threads(2)
        self.p=torch.randn(3,1024);self.c=torch.rand(3,2,2);self.m=torch.randn(3,256,8,10)

    def test_normalized_finite_inside(self):
        model=BackPointIntegralProbe();xy,p=model(self.p,self.c,self.m,True)
        self.assertTrue(torch.isfinite(xy).all());self.assertTrue(((xy>=0)&(xy<=1)).all())
        torch.testing.assert_close(p.sum((-2,-1)),torch.ones(3,2))

    def test_init_identical_between_modes(self):
        a=BackPointIntegralProbe('global');b=BackPointIntegralProbe('spatial');b.load_state_dict(a.state_dict())
        torch.testing.assert_close(a(self.p,self.c,self.m),b(self.p,self.c,self.m),rtol=0,atol=0)

    def test_permutation_and_single(self):
        model=BackPointIntegralProbe();model.query[-1].weight.data.normal_(0,.1)
        a=model(self.p,self.c,self.m);order=[2,0,1]
        torch.testing.assert_close(model(self.p[order],self.c[order],self.m[order]),a[order])
        torch.testing.assert_close(model(self.p[:1],self.c[:1],self.m[:1]),a[:1])

    def test_inputs_detached_grad_finite(self):
        model=BackPointIntegralProbe();p=self.p.requires_grad_();c=self.c.requires_grad_();m=self.m.requires_grad_()
        model(p,c,m).square().mean().backward()
        self.assertIsNone(p.grad);self.assertIsNone(c.grad);self.assertIsNone(m.grad)
        self.assertTrue(all(v.grad is None or torch.isfinite(v.grad).all() for v in model.parameters()))
        self.assertGreater(float(model.query[-1].weight.grad.abs().sum()),0)

    def test_global_invariant_to_memory_permutation(self):
        model=BackPointIntegralProbe('global');model.query[-1].weight.data.normal_(0,.1)
        shuffled=self.m.flatten(2)[:,:,torch.randperm(80)].reshape_as(self.m)
        torch.testing.assert_close(model(self.p,self.c,self.m),model(self.p,self.c,shuffled))

    def test_spatial_changes_with_permutation(self):
        model=BackPointIntegralProbe();model.query[-1].weight.data.normal_(0,.1)
        shuffled=self.m.flip(-1)
        self.assertGreater(float((model(self.p,self.c,self.m)-model(self.p,self.c,shuffled)).abs().max()),1e-5)


if __name__=='__main__':unittest.main()
