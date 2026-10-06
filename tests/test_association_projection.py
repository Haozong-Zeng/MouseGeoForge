import unittest
import torch
from mousegeoforge.models.association_projection import AssociationProjection,scene_contrastive_loss


class AssociationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(17);self.m=AssociationProjection(12,4);self.x=torch.randn(1,6,12)
        self.ids=torch.tensor([[0,1,0,1,0,-1]]);self.views=torch.tensor([[0,0,1,1,2,2]])
    def test_off_exact(self):self.assertIs(self.m(self.x),self.x)
    def test_permutation(self):
        p=[3,5,1,0,4,2];z=self.m(self.x,enabled=True);zp=self.m(self.x[:,p],enabled=True)
        torch.testing.assert_close(z[:,p],zp);a,_=scene_contrastive_loss(z,self.ids,self.views);b,_=scene_contrastive_loss(zp,self.ids[:,p],self.views[:,p]);torch.testing.assert_close(a,b)
    def test_unknown_exclusion(self):
        z=self.m(self.x,enabled=True).detach().requires_grad_(True);loss,_=scene_contrastive_loss(z,self.ids,self.views);loss.backward();self.assertEqual(float(z.grad[0,-1].abs().sum()),0)
    def test_no_positive_or_negative(self):
        z=self.m(self.x,enabled=True)
        for ids,views in [(torch.zeros_like(self.ids),self.views),(self.ids,torch.zeros_like(self.views)),(torch.full_like(self.ids,-1),self.views)]:
            loss,log=scene_contrastive_loss(z,ids,views);self.assertEqual(float(loss),0);self.assertEqual(int(log['active_anchors']),0)
    def test_finite_gradient(self):
        z=self.m(self.x,enabled=True);loss,_=scene_contrastive_loss(z,self.ids,self.views);loss.backward();self.assertTrue(torch.isfinite(self.m.projection.weight.grad).all());self.assertGreater(float(self.m.projection.weight.grad.norm()),0)
    def test_identity_renaming(self):
        z=self.m(self.x,enabled=True);ids=self.ids.clone();ids[self.ids==0]=7;ids[self.ids==1]=42
        a,_=scene_contrastive_loss(z,self.ids,self.views);b,_=scene_contrastive_loss(z,ids,self.views);torch.testing.assert_close(a,b)
    def test_no_label_forward_and_finite_checks(self):
        with self.assertRaises(TypeError):self.m(self.x,identities=self.ids,enabled=True)
        with self.assertRaises(ValueError):self.m(torch.full_like(self.x,float('nan')),enabled=True)


if __name__=='__main__':unittest.main()
