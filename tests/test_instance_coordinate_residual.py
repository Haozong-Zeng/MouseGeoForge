import unittest
import torch
from mousegeoforge.instance_coordinate_residual import InstanceCoordinateResidual,reference_from_shape


class TestCoordinateResidual(unittest.TestCase):
    def evidence(self,v=3):
        torch.manual_seed(17)
        return torch.rand(v,6,23,2)*.4+.3,torch.rand(v,6,23,2)*.4+.3,torch.randn(v,6,23,256),torch.rand(v,6,23,5),torch.tensor([.5,.5,.4,.3]).expand(v,6,4).clone()

    def test_zero_identity_detach_and_capacity(self):
        xs=[x.requires_grad_() for x in self.evidence()]
        models=[InstanceCoordinateResidual(relative=b) for b in (False,True)]
        self.assertEqual(sum(p.numel() for p in models[0].parameters()),sum(p.numel() for p in models[1].parameters()))
        for model in models:
            y=model(*xs);torch.testing.assert_close(y,xs[0],atol=0,rtol=0);y.sum().backward()
            self.assertGreater(float(model.network[-1].weight.grad.abs().sum()),0)
        self.assertTrue(all(x.grad is None for x in xs))

    def test_affine_relative_equivariance_with_fixed_evidence(self):
        x,a,t,q,r=self.evidence(1);model=InstanceCoordinateResidual(relative=True)
        torch.nn.init.normal_(model.network[-1].weight,std=.001)
        y=model(x,a,t,q,r);scale=.8;shift=torch.tensor([.06,.08])
        rp=torch.cat([r[...,:2]*scale+shift,r[...,2:]*scale],-1)
        yp=model(x*scale+shift,a*scale+shift,t,q,rp)
        torch.testing.assert_close(yp,y*scale+shift,atol=1e-6,rtol=1e-6)

    def test_query_permutation_and_empty_fallback(self):
        x,a,t,q,r=self.evidence();model=InstanceCoordinateResidual(relative=True)
        torch.nn.init.normal_(model.network[-1].weight,std=.001);order=torch.tensor([4,0,3,5,1,2])
        torch.testing.assert_close(model(x[:,order],a[:,order],t[:,order],q[:,order],r[:,order]),model(x,a,t,q,r)[:,order])
        shape=torch.zeros(3,6,16);ref=reference_from_shape(shape)
        torch.testing.assert_close(ref,torch.tensor([.5,.5,1.,1.]).expand(3,6,4))

    def test_empty_reference_matches_global_after_learning(self):
        x,a,t,q,_=self.evidence(1)
        model=InstanceCoordinateResidual(relative=True)
        torch.nn.init.normal_(model.network[-1].weight,std=.001)
        control=InstanceCoordinateResidual(relative=False);control.load_state_dict(model.state_dict())
        reference=reference_from_shape(torch.zeros(1,6,16))
        torch.testing.assert_close(model(x,a,t,q,reference),control(x,a,t,q,reference),atol=0,rtol=0)


if __name__=='__main__':unittest.main()
