import unittest
import torch
from mousegeoforge.instance_coordinate_residual import InstanceCoordinateResidual
from mousegeoforge.origin_coordinate_residual import OriginCoordinateResidual


class OriginTest(unittest.TestCase):
    def evidence(self):
        torch.manual_seed(17)
        return [torch.rand(3,6,23,2)*.4+.3,torch.rand(3,6,23,2)*.4+.3,torch.randn(3,6,23,256),torch.rand(3,6,23,5),torch.tensor([.5,.5,.4,.3]).expand(3,6,4).clone()]

    def test_old_forward_and_parameter_gradients_exact(self):
        xs=self.evidence();old=InstanceCoordinateResidual();new=OriginCoordinateResidual()
        torch.nn.init.normal_(old.network[-1].weight,std=.02);new.load_state_dict(old.state_dict())
        a=old(*xs);b=new(*xs);torch.testing.assert_close(a,b,atol=0,rtol=0)
        a.square().sum().backward();b.square().sum().backward()
        for p,q in zip(old.parameters(),new.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)

    def test_anchor_zero_and_equal_capacity_initialization(self):
        xs=[x.requires_grad_() for x in self.evidence()]
        torch.manual_seed(17);a=OriginCoordinateResidual()
        torch.manual_seed(17);b=OriginCoordinateResidual(anchor_origin=True)
        self.assertEqual(sum(p.numel() for p in a.parameters()),8578)
        self.assertEqual(sum(p.numel() for p in b.parameters()),8578)
        for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q,atol=0,rtol=0)
        torch.testing.assert_close(a(*xs),xs[0],atol=0,rtol=0)
        y=b(*xs);torch.testing.assert_close(y,xs[1],atol=0,rtol=0);y.sum().backward()
        self.assertTrue(all(x.grad is None for x in xs))

    def test_query_permutation_and_fixed_evidence_affine(self):
        xs=self.evidence();model=OriginCoordinateResidual(anchor_origin=True)
        torch.nn.init.normal_(model.network[-1].weight,std=.001)
        order=torch.tensor([4,0,3,5,1,2]);y=model(*xs)
        torch.testing.assert_close(model(*[x[:,order] for x in xs]),y[:,order])
        x,a,t,q,r=xs;scale=.8;shift=torch.tensor([.06,.08]);rp=torch.cat([r[...,:2]*scale+shift,r[...,2:]*scale],-1)
        torch.testing.assert_close(model(x*scale+shift,a*scale+shift,t,q,rp),y*scale+shift,atol=1e-6,rtol=1e-6)


if __name__=='__main__':unittest.main()
