import unittest
import torch
from mousegeoforge.models.confidence_gated_view_residual import ConfidenceGatedViewResidual

class GatedResidualTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(17);self.model=ConfidenceGatedViewResidual()
        self.args=[torch.rand(1,3,2,4,2),torch.randn(1,3,2,4,256),torch.tensor([[[4.,-4.],[4.,-4.],[4.,-4.]]]),torch.rand(1,3,2,4),torch.tensor([[10,20,30]]),torch.ones(1,3,dtype=torch.bool)]
    def trained(self):
        torch.nn.init.normal_(self.model.delta[-1].weight,std=.1)
        torch.nn.init.constant_(self.model.delta[-1].bias,.05)
    def test_off_exact_identity(self):self.assertIs(self.model(*self.args)['xy'],self.args[0])
    def test_zero_init_exact(self):torch.testing.assert_close(self.model(*self.args,enabled=True)['xy'],self.args[0],atol=0,rtol=0)
    def test_v1_exact_after_training(self):
        self.trained();a=[x[:,:1] for x in self.args];self.assertIs(self.model(*a,enabled=True)['xy'],a[0])
    def test_permutation_equivariance(self):
        self.trained();p=[2,0,1];a=self.model(*self.args,enabled=True);b=self.model(*[x[:,p] for x in self.args],enabled=True)
        torch.testing.assert_close(a['xy'][:,p],b['xy'],atol=2e-7,rtol=1e-6)
    def test_duplicate_camera_mass(self):
        self.trained();original=self.model(*self.args,enabled=True);a=[torch.cat([x,x[:,1:2]],1) for x in self.args];new=self.model(*a,enabled=True)
        torch.testing.assert_close(original['xy'],new['xy'][:,:3],atol=2e-7,rtol=1e-6)
    def test_missing_nan_support_ignored(self):
        self.trained();a=[x.clone() for x in self.args];a[-1][:,2]=False;a[1][:,2]=float('nan');a[2][:,2]=float('nan');a[3][:,2]=float('nan');p=self.model(*a,enabled=True);trim=self.model(*[x[:,:2] for x in self.args],enabled=True)
        torch.testing.assert_close(p['xy'][:,:2],trim['xy'],atol=2e-7,rtol=1e-6);torch.testing.assert_close(p['xy'][:,2],a[0][:,2],atol=0,rtol=0)
    def test_all_invalid_or_zero_quality_exact(self):
        self.trained()
        for field in [3,5]:
            a=[x.clone() for x in self.args];a[field].zero_();r=self.model(*a,enabled=True);torch.testing.assert_close(r['xy'],a[0],atol=0,rtol=0);self.assertFalse(r['active'].any())
    def test_ambiguous_instances_abstain(self):
        self.trained();a=[x.clone() for x in self.args];a[2].fill_(4);r=self.model(*a,enabled=True);torch.testing.assert_close(r['xy'],a[0],atol=0,rtol=0)
    def test_no_physical_support_exact(self):
        self.trained();a=[x.clone() for x in self.args];a[4].fill_(10);torch.testing.assert_close(self.model(*a,enabled=True)['xy'],a[0],atol=0,rtol=0)
    def test_gradient_and_frozen_inputs(self):
        self.args[1].requires_grad_();out=self.model(*self.args,enabled=True);out['xy'].sum().backward();self.assertGreater(float(self.model.delta[-1].weight.grad.abs().sum()),0);self.assertIsNone(self.args[1].grad)
    def test_self_control_same_parameters(self):
        other=ConfidenceGatedViewResidual(mode='self');self.assertEqual(sum(p.numel() for p in self.model.parameters()),sum(p.numel() for p in other.parameters()))
    def test_invalid_shape(self):
        a=self.args.copy();a[3]=a[3][...,0]
        with self.assertRaises(ValueError):self.model(*a,enabled=True)

if __name__=='__main__':unittest.main()
